"""CellFlow-compatible balanced OT pairing for empirical cell populations."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class SinkhornStats:
    iterations: int
    converged: bool
    marginal_error: float
    cost_mean: float


def squared_euclidean_cost(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if source.ndim != 2 or target.ndim != 2 or source.size(1) != target.size(1):
        raise ValueError("source and target must be [N,D] and [M,D] tensors")
    if min(source.size(0), target.size(0)) < 1:
        raise ValueError("source and target populations cannot be empty")
    source, target = source.float(), target.float()
    cost = source.square().sum(1, keepdim=True) + target.square().sum(1).unsqueeze(0)
    cost.addmm_(source, target.t(), beta=1.0, alpha=-2.0)
    return cost.clamp_min_(0.0)


def balanced_sinkhorn(cost: torch.Tensor, epsilon: float = 1.0,
                      max_iterations: int = 2000, tolerance: float = 1e-3):
    """Match CellFlow/OTT balanced Sinkhorn: mean cost, g-then-f, target L1.

    marginal_error is the SUM of absolute target marginal errors, not a
    maximum per-position error. Check every 10 iterations as OTT defaults.
    """
    if cost.ndim != 2 or min(cost.shape) < 1:
        raise ValueError(f"cost must be a non-empty matrix, got {tuple(cost.shape)}")
    if epsilon <= 0 or max_iterations < 1 or tolerance <= 0:
        raise ValueError("epsilon, max_iterations, and tolerance must be positive")
    cost = cost.float()
    cost_mean = cost.mean()
    if not torch.isfinite(cost_mean) or cost_mean.item() <= 0:
        raise ValueError("OT cost mean must be finite and positive")
    n_source, n_target = cost.shape
    log_kernel = -(cost / cost_mean) / float(epsilon)
    log_u = torch.zeros(n_source, device=cost.device, dtype=cost.dtype)
    log_v = torch.zeros(n_target, device=cost.device, dtype=cost.dtype)
    converged, marginal_error = False, float("inf")
    for iteration in range(1, max_iterations + 1):
        # OTT updates target potential g first, then source potential f.
        log_v = -math.log(n_target) - torch.logsumexp(log_kernel + log_u.unsqueeze(1), dim=0)
        log_u = -math.log(n_source) - torch.logsumexp(log_kernel + log_v.unsqueeze(0), dim=1)
        if iteration == max_iterations or iteration % 10 == 0:
            log_plan = log_u.unsqueeze(1) + log_kernel + log_v.unsqueeze(0)
            # The final source update enforces its marginal; OTT's balanced,
            # sequential solver checks the remaining target marginal with L1.
            col_error = (torch.exp(torch.logsumexp(log_plan, dim=0)) - 1.0 / n_target).abs().sum()
            marginal_error = float(col_error.item())
            if marginal_error < tolerance:
                converged = True
                break
    plan = torch.exp(log_u.unsqueeze(1) + log_kernel + log_v.unsqueeze(0))
    return plan, SinkhornStats(iteration, converged, marginal_error, float(cost_mean.item()))


def sample_joint(plan: torch.Tensor, n_samples: int,
                 generator: torch.Generator | None = None, *, replacement: bool = True):
    """Sample pair positions; without replacement does not ensure unique cells."""
    if plan.ndim != 2 or min(plan.shape) < 1 or n_samples < 1:
        raise ValueError("plan must be non-empty and n_samples must be positive")
    probabilities = plan.float().reshape(-1)
    if not replacement and n_samples > int((probabilities > 0).sum().item()):
        raise ValueError("Not enough positive-probability OT pairs for sampling without replacement")
    flat = torch.multinomial(
        probabilities / probabilities.sum(), n_samples, replacement=replacement, generator=generator
    )
    return torch.div(flat, plan.size(1), rounding_mode="floor"), torch.remainder(flat, plan.size(1))


@torch.no_grad()
def pair_population(source: torch.Tensor, target: torch.Tensor, n_samples: int | None = None,
                    epsilon: float = 1.0, max_iterations: int = 2000,
                    tolerance: float = 1e-3, generator: torch.Generator | None = None, *,
                    replacement: bool = True):
    plan, stats = balanced_sinkhorn(
        squared_euclidean_cost(source, target), epsilon, max_iterations, tolerance
    )
    source_indices, target_indices = sample_joint(
        plan, source.size(0) if n_samples is None else int(n_samples), generator, replacement=replacement
    )
    return source_indices, target_indices, stats
