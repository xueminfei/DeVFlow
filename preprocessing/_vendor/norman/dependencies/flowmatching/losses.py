"""Distribution losses adapted from scDFM's unbiased multi-sigma MMD."""

import torch


@torch.no_grad()
def median_sigmas(x, scales=(0.5, 1.0, 2.0, 4.0)):
    distances = torch.cdist(x, x, p=2).square()
    if distances.size(0) < 2:
        return [1.0]
    tri = distances[~torch.eye(distances.size(0), dtype=torch.bool, device=x.device)]
    median = tri.median().clamp_min(1e-12)
    return [float(torch.sqrt(median * scale).item()) for scale in scales]


def mmd2_unbiased_multi_sigma(x, y, sigmas=None):
    """Unbiased Gaussian-kernel MMD for two sample matrices ``[N, D]``."""
    if x.ndim != 2 or y.ndim != 2 or x.size(1) != y.size(1):
        raise ValueError(f"MMD expects [N,D] and [M,D], got {tuple(x.shape)}, {tuple(y.shape)}")
    if x.size(0) < 2 or y.size(0) < 2:
        raise ValueError("MMD requires at least two samples per distribution")
    if sigmas is None:
        sigmas = median_sigmas(torch.cat([x.detach(), y.detach()], dim=0))
    dxx = torch.cdist(x, x, p=2).square()
    dyy = torch.cdist(y, y, p=2).square()
    dxy = torch.cdist(x, y, p=2).square()
    values = []
    for sigma in sigmas:
        beta = 1.0 / (2.0 * float(sigma) ** 2 + 1e-12)
        kxx = torch.exp(-beta * dxx)
        kyy = torch.exp(-beta * dyy)
        kxy = torch.exp(-beta * dxy)
        values.append(
            (kxx.sum() - kxx.diag().sum()) / (x.size(0) * (x.size(0) - 1))
            + (kyy.sum() - kyy.diag().sum()) / (y.size(0) * (y.size(0) - 1))
            - 2.0 * kxy.mean()
        )
    return torch.stack(values).mean()
