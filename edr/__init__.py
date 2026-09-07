from PIL import Image
from . import sd3, flux, presets, watermark


def edit(model, model_id, image, source, target, t0, gamma, sigma, i_min, i_max,
         steps, seed=None, src_guidance=None, tar_guidance=None, res=None, variant="paper"):
    backend = {"sd3": sd3, "flux": flux}[model]
    bb = presets.BACKBONES[model]
    kw = dict(model_id=model_id, image=image, source=source, target=target,
              t0=t0, gamma=gamma, sigma=sigma, i_min=i_min, i_max=i_max,
              steps=steps, seed=seed,
              src_guidance=src_guidance if src_guidance is not None else bb["src_guidance"],
              tar_guidance=tar_guidance if tar_guidance is not None else bb["tar_guidance"])
    if res is not None:
        kw["res"] = res
    return backend.edit(**kw)
