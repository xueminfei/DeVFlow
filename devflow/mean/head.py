"""A fresh trainable dual kernel regressor for condition mean deltas."""
from __future__ import annotations

import copy
from numbers import Integral
import torch
from torch import nn


class TrainableKernelMeanHead(nn.Module):
    """Learn A[S,N,G] with the original centred dual KRR objective.

    S is the number of strata, N the training anchor count, G the gene count.
    ``targets`` are training treated means minus matched training controls.
    Kernels, intercepts, control means and targets are immutable model buffers.
    """

    def __init__(self, kernel=None, targets=None, bias=None, control_mean=None,
                 alpha=None, metadata=None, *, kernels=None, alphas=None):
        super().__init__()
        kernel = kernels if kernel is None else kernel
        alpha = alphas if alpha is None else alpha
        # A single stratum may arrive as a NumPy [None, ...] view with a zero
        # leading stride. Give checkpoint/DDP broadcasts ordinary dense layouts.
        arrays = {name: torch.as_tensor(value, dtype=torch.float64).detach().clone(
                      memory_format=torch.contiguous_format)
                  for name, value in dict(kernel=kernel, targets=targets, bias=bias,
                                          control_mean=control_mean, alpha=alpha).items()}
        k, y = arrays["kernel"], arrays["targets"]
        if k.ndim != 3 or k.shape[1] != k.shape[2] or y.ndim != 3 or y.shape[:2] != k.shape[:2]:
            raise ValueError("Expected kernel[S,N,N] and training delta targets[S,N,G]")
        s, n, g = y.shape
        if s < 1 or n < 2 or g < 1:
            raise ValueError("At least one stratum, two anchors and one gene are required")
        if arrays["bias"].shape != (s, g) or arrays["control_mean"].shape != (s, g):
            raise ValueError("Mean-head bias and controls must both be [S,G]")
        if arrays["alpha"].shape != (s,) or torch.any(arrays["alpha"] <= 0):
            raise ValueError("Every stratum needs a positive KRR alpha")
        if not all(torch.isfinite(value).all() for value in arrays.values()):
            raise ValueError("Mean-head assets contain nonfinite values")
        if not torch.allclose(k, k.transpose(1, 2), atol=1e-9, rtol=1e-9):
            raise ValueError("Training kernels must be symmetric")
        if not torch.allclose(k.mean(-1), torch.zeros_like(k.mean(-1)), atol=1e-8, rtol=0):
            raise ValueError("Training kernels must be centred on training anchors")
        if not torch.allclose(y.mean(1), arrays["bias"], atol=1e-7, rtol=1e-7):
            raise ValueError("The fixed intercept must be the equally weighted training delta mean")
        for name, value in arrays.items():
            self.register_buffer(name, value)
        self.A = nn.Parameter(torch.zeros_like(y, memory_format=torch.contiguous_format))
        self.metadata = copy.deepcopy(metadata or {})

    def forward(self, query_kernel, stratum, *, output_dtype=torch.float32):
        kernel = torch.as_tensor(query_kernel, device=self.A.device, dtype=self.A.dtype)
        if kernel.ndim == 1:
            kernel = kernel[None]
        if kernel.ndim != 2 or kernel.shape[1] != self.A.shape[1]:
            raise ValueError("Query kernel must be [queries, training anchors]")
        if isinstance(stratum, Integral) or (isinstance(stratum, torch.Tensor) and stratum.ndim == 0):
            s = int(stratum)
            if s < 0 or s >= len(self.A):
                raise IndexError("Unknown mean-head stratum")
            delta = kernel @ self.A[s] + self.bias[s]
        else:
            strata = torch.as_tensor(stratum, dtype=torch.long, device=self.A.device)
            if strata.shape != (len(kernel),):
                raise ValueError("A vector of strata must identify every query row")
            if torch.any((strata < 0) | (strata >= len(self.A))):
                raise IndexError("Unknown mean-head stratum")
            delta = torch.bmm(kernel[:, None], self.A[strata]).squeeze(1) + self.bias[strata]
        return delta.to(output_dtype) if output_dtype is not None else delta

    def predict_mean(self, query_kernel, stratum):
        # Match float64 KRR -> float32 delta, then float32 control + delta.
        # Only the later cell-distribution composition uses float64 arithmetic.
        delta = self(query_kernel, stratum, output_dtype=torch.float32)
        control = self.control_mean[stratum].float()
        return (control + delta).clamp_min(0)

    def loss(self, *, return_parts=False):
        ka = torch.bmm(self.kernel, self.A)
        fitted_delta = ka + self.bias[:, None]
        fitted = (fitted_delta - self.targets).square().mean()
        regularizer = ((self.A * ka).sum(dim=(1, 2)) * self.alpha).sum() / self.targets.numel()
        total = fitted + regularizer
        if return_parts:
            return total, {"head_fit_loss": fitted, "head_regularization_loss": regularizer}
        return total

    @torch.no_grad()
    def closed_form_coefficients(self):
        """Training-only numerical reference; never used to initialise A."""
        eye = torch.eye(self.A.shape[1], device=self.A.device, dtype=self.A.dtype)
        return torch.linalg.solve(self.kernel + self.alpha[:, None, None] * eye,
                                  self.targets - self.bias[:, None])

    def export_assets(self):
        return {**{name: getattr(self, name).detach().cpu().clone()
                   for name in ("kernel", "targets", "bias", "control_mean", "alpha")},
                "metadata": copy.deepcopy(self.metadata)}


KernelMeanHead = TrainableKernelMeanHead
