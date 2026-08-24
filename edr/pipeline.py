"""EDR editing loop (Algorithm A of the paper).

This file is the integration point for the research implementation:
transplant the working code here, keeping the signature of `edit`.
The skeleton below marks each step of Algorithm A; the model-specific
parts (loading, prompt encoding, velocity prediction for SD3 / FLUX)
come from the research code.
"""
from PIL import Image

from .degradation import degrade


def edit(model: str, model_id: str, image: Image.Image,
         source: str, target: str,
         t0: float, gamma: float, sigma: float,
         i_min: float, i_max: float,
         steps: int, seed: int | None = None) -> Image.Image:
    # 1. load backbone (cached), encode `image` to latents X_src
    # 2. n = floor(t0 * T); t = t0; Z_dir = X_src; dt = t0 / n
    # 3. loop i = n..1:
    #      N ~ N(0, I); Z_src = (1 - t) * X_src + t * N
    #      Z_tar = Z_dir + Z_src - X_src
    #      V_dir = v(Z_tar, t, target) - v(Z_src, t, source)
    #      if first iteration:
    #          V_edr = V_dir                       # first-step convention (Sec. 3.1)
    #      else:
    #          V_tilde = degrade(dZ_dir / dt, sigma, i_min, i_max)
    #          V_proj  = (<V_dir, V_tilde> / ||V_tilde||^2) * V_tilde
    #          alpha   = t ** gamma
    #          V_edr   = alpha * V_proj + (1 - alpha) * V_dir
    #      Z_dir = Z_dir + V_edr; t -= dt
    # 4. decode Z_dir -> PIL image
    raise NotImplementedError("transplant the research implementation here")
