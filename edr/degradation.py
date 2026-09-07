import torch
import torch.nn.functional as F


def degrade(v, sigma=5.0, i_min=0.25, i_max=0.75):
    """Combined degraded representation D_c (Eq. 9): per-channel Gaussian
    smoothing over the 2D latent grid (6-sigma kernel), then the linear
    intensity remap I_min + (I_max - I_min) * (.)."""
    B, C, H, W = v.shape
    vf = v.float()
    ks = 2 * int(3 * sigma) + 1
    coords = torch.arange(ks, dtype=torch.float32) - ks // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = (g / g.sum()).to(v.device)
    kx = g.view(1, 1, 1, -1).repeat(C, 1, 1, 1)
    ky = g.view(1, 1, -1, 1).repeat(C, 1, 1, 1)
    pad = ks // 2
    s = F.conv2d(F.pad(vf, (pad, pad, 0, 0), mode="reflect"), kx, groups=C)
    s = F.conv2d(F.pad(s, (0, 0, pad, pad), mode="reflect"), ky, groups=C)
    return (i_min + (i_max - i_min) * s).to(v.dtype)


def project(a, b, eps=1e-8):
    af = a.view(a.shape[0], -1)
    bf = b.view(b.shape[0], -1)
    p = ((af * bf).sum(1, keepdim=True) / (bf.pow(2).sum(1, keepdim=True) + eps)) * bf
    return p.view_as(a)
