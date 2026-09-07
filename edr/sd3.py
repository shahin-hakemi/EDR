"""EDR editing for Stable Diffusion 3 (Algorithm A)."""
import torch
import numpy as np
from PIL import Image

from .degradation import degrade, project

_CACHE = {}


def _load(model_id, device):
    if model_id not in _CACHE:
        from diffusers import StableDiffusion3Pipeline
        pipe = StableDiffusion3Pipeline.from_pretrained(
            model_id, torch_dtype=torch.float16, text_encoder_3=None, tokenizer_3=None)
        pipe.enable_model_cpu_offload()
        pipe.vae.enable_tiling()
        _CACHE[model_id] = pipe
    return _CACHE[model_id]


def _encode_image(pipe, image, res, device):
    raw = image.convert("RGB").resize((res, res))
    px = (torch.from_numpy(np.array(raw).astype(np.float32) / 127.5 - 1.0)
          .permute(2, 0, 1).unsqueeze(0).to(device, dtype=torch.float16))
    with torch.no_grad():
        x = pipe.vae.encode(px).latent_dist.sample()
        return (x - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor


def _decode(pipe, lat):
    with torch.no_grad():
        d = pipe.vae.decode(lat / pipe.vae.config.scaling_factor + pipe.vae.config.shift_factor,
                            return_dict=False)[0]
    img = ((d[0] / 2 + 0.5).clamp(0, 1).cpu().permute(1, 2, 0).float().numpy() * 255).astype(np.uint8)
    return Image.fromarray(img)


@torch.no_grad()
def _calc_v(pipe, latent_input, embeds, pooled, src_scale, tar_scale, t):
    v = pipe.transformer(hidden_states=latent_input, timestep=t.expand(latent_input.shape[0]),
                         encoder_hidden_states=embeds, pooled_projections=pooled,
                         return_dict=False)[0]
    s_un, s_tx, t_un, t_tx = v.chunk(4)
    return s_un + src_scale * (s_tx - s_un), t_un + tar_scale * (t_tx - t_un)


@torch.no_grad()
def edit(model_id, image, source, target, t0, gamma, sigma, i_min, i_max,
         steps, seed, src_guidance, tar_guidance, res=1024):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = _load(model_id, device)
    x_src = _encode_image(pipe, image, res, device)
    gen = torch.Generator(device=device).manual_seed(42 if seed is None else seed)

    data = []
    for p, g in [(source, src_guidance), (target, tar_guidance)]:
        pipe._guidance_scale = g
        data.append(pipe.encode_prompt(prompt=p, prompt_2=None, prompt_3=None,
                                       do_classifier_free_guidance=True, device=device))
    embeds = torch.cat([data[0][1], data[0][0], data[1][1], data[1][0]], dim=0)
    pooled = torch.cat([data[0][3], data[0][2], data[1][3], data[1][2]], dim=0)

    pipe.scheduler.set_timesteps(steps, device=device)
    timesteps = pipe.scheduler.timesteps
    start_idx = steps - int(steps * t0)

    zt, z_prev, t_prev, first = x_src.clone(), None, None, True
    for i, tt in enumerate(timesteps):
        if i < start_idx:
            continue
        t = tt.item() / 1000.0
        t_next = timesteps[i + 1].item() / 1000.0 if i + 1 < len(timesteps) else 0.0
        dt = t_next - t

        noise = torch.randn(x_src.shape, generator=gen, device=device, dtype=x_src.dtype)
        z_src = (1 - t) * x_src + t * noise
        z_tar = zt + z_src - x_src
        inp = torch.cat([z_src, z_src, z_tar, z_tar])
        v_s, v_t = _calc_v(pipe, inp, embeds, pooled, src_guidance, tar_guidance, tt)
        v_dir = v_t - v_s

        if first:
            v_edr, first = v_dir, False
        else:
            v_hist = (z_prev - zt) / (t_prev - t + 1e-8)
            v_tilde = degrade(v_hist, sigma, i_min, i_max)
            alpha = t ** gamma
            v_edr = alpha * project(v_dir, v_tilde) + (1 - alpha) * v_dir

        z_prev, t_prev = zt, t
        zt = zt + dt * v_edr

    return _decode(pipe, zt)
