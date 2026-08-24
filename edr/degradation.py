import torch
import torch.nn.functional as F


def gaussian_kernel1d(sigma: float, device, dtype):
    radius = int(3 * sigma)
    x = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    k = torch.exp(-(x ** 2) / (2 * sigma ** 2))
    return k / k.sum()


def degrade(v: torch.Tensor, sigma: float, i_min: float, i_max: float) -> torch.Tensor:
    """D_c of Eq. 9: per-channel Gaussian smoothing over the 2D latent grid,
    followed by the intensity remap I_min + (I_max - I_min) * (.)."""
    k = gaussian_kernel1d(sigma, v.device, v.dtype)
    c = v.shape[1]
    kx = k.view(1, 1, 1, -1).expand(c, 1, 1, -1)
    ky = k.view(1, 1, -1, 1).expand(c, 1, -1, 1)
    pad = k.numel() // 2
    s = F.conv2d(F.pad(v, (pad, pad, 0, 0), mode="reflect"), kx, groups=c)
    s = F.conv2d(F.pad(s, (0, 0, pad, pad), mode="reflect"), ky, groups=c)
    return i_min + (i_max - i_min) * s
