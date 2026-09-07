"""EDR editing for FLUX.1-dev (Algorithm A)."""
import math
import torch
import numpy as np
from PIL import Image

from .degradation import degrade, project

_CACHE = {}


def _load(model_id, device):
    if model_id not in _CACHE:
        from diffusers import FluxPipeline
        pipe = FluxPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
        pipe.enable_sequential_cpu_offload()
        pipe.vae.enable_tiling()
        pipe.vae.enable_slicing()
        _CACHE[model_id] = pipe
    return _CACHE[model_id]


def _image_ids(h, w, device, dtype):
    ids = torch.zeros(h // 2, w // 2, 3)
    ids[..., 1] = ids[..., 1] + torch.arange(h // 2)[:, None]
    ids[..., 2] = ids[..., 2] + torch.arange(w // 2)[None, :]
    ids = ids[None, :].repeat(1, 1, 1, 1).reshape(1, (h // 2) * (w // 2), 3)
    return ids.to(device=device, dtype=dtype)


def _pack(x):
    b, c, h, w = x.shape
    x = x.view(b, c, h // 2, 2, w // 2, 2).permute(0, 2, 4, 1, 3, 5)
    return x.reshape(b, (h // 2) * (w // 2), c * 4)


def _unpack(x, h, w):
    b, seq, c = x.shape
    x = x.view(b, h // 2, w // 2, c // 4, 2, 2).permute(0, 3, 1, 4, 2, 5)
    return x.reshape(b, c // 4, h, w)


def _encode_image(pipe, image, res, device):
    raw = image.convert("RGB").resize((res, res))
    px = (torch.from_numpy(np.array(raw).astype(np.float32) / 127.5 - 1.0)
          .permute(2, 0, 1).unsqueeze(0).to(device, dtype=torch.bfloat16))
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
def edit(model_id, image, source, target, t0, gamma, sigma, i_min, i_max,
         steps, seed, src_guidance, tar_guidance, res=512):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = _load(model_id, device)
    x_src = _encode_image(pipe, image, res, device)
    gen = torch.Generator(device=device).manual_seed(42 if seed is None else seed)
    _, _, h, w = x_src.shape

    mu = None
    if getattr(pipe.scheduler.config, "use_dynamic_shifting", False):
        cfg = pipe.scheduler.config
        m = (cfg.max_shift - cfg.base_shift) / (math.log(cfg.max_image_seq_len) - math.log(cfg.base_image_seq_len))
        b = cfg.base_shift - m * math.log(cfg.base_image_seq_len)
        mu = m * math.log((h // 2) * (w // 2)) + b
    pipe.scheduler.set_timesteps(steps, device=device, **({"mu": mu} if mu is not None else {}))

    timesteps = pipe.scheduler.timesteps
    start_idx = steps - int(steps * t0)
    src_e, src_p, src_ids = pipe.encode_prompt(prompt=source, prompt_2=None)
    tar_e, tar_p, tar_ids = pipe.encode_prompt(prompt=target, prompt_2=None)
    img_ids = _image_ids(h, w, device, x_src.dtype)

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

        def vel(packed, pooled, embeds, tids, scale):
            return pipe.transformer(
                hidden_states=packed, timestep=tt.expand(packed.shape[0]) / 1000.0,
                guidance=torch.tensor([scale], device=device, dtype=x_src.dtype).expand(packed.shape[0]),
                pooled_projections=pooled, encoder_hidden_states=embeds,
                txt_ids=tids, img_ids=img_ids, return_dict=False)[0]

        v_s = _unpack(vel(_pack(z_src), src_p, src_e, src_ids, src_guidance), h, w)
        v_t = _unpack(vel(_pack(z_tar), tar_p, tar_e, tar_ids, tar_guidance), h, w)
        v_dir = v_t - v_s
        torch.cuda.empty_cache()

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
