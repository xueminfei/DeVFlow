"""Compose condition means with cell-level relative expression patterns."""
from __future__ import annotations

import torch


def compose_expression(expression, predicted_mean, *, detach_mean=False,
                       output_dtype=torch.float32, return_parts=False):
    """Scale a *complete condition* together, preserving its prescribed mean.

    The denominator must be calculated over all predicted cells of the
    condition, never independently per inference chunk. A nonpositive raw mean
    uses relative expression 1, hence output ``predicted_mean``. Computation is
    float64 followed by the explicit output conversion used by the reference.
    The denominator retains gradients; only the caller-selected mean path can
    be detached.
    """
    if expression.ndim != 2 or expression.shape[0] == 0:
        raise ValueError("Expression must be a nonempty [cells, genes] matrix")
    mean = torch.as_tensor(predicted_mean, device=expression.device)
    if mean.ndim == 2 and mean.shape[0] == 1:
        mean = mean[0]
    if mean.shape != (expression.shape[1],):
        raise ValueError("One predicted gene-mean vector is required per condition")
    if detach_mean:
        mean = mean.detach()
    values, mean = expression.double(), mean.double()
    raw_mean = values.mean(dim=0)
    positive = raw_mean > 0
    safe_mean = torch.where(positive, raw_mean, torch.ones_like(raw_mean))
    relative = torch.where(positive[None], values / safe_mean[None], torch.ones_like(values))
    prediction = relative * mean[None]
    if output_dtype is not None:
        prediction = prediction.to(output_dtype)
    if return_parts:
        return prediction, {"raw_mean": raw_mean, "relative_expression": relative,
                            "zero_mean_fraction": (~positive).double().mean()}
    return prediction
