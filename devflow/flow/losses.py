"""Condition-local losses used by the checkpoint-compatible SciPlex recipe."""
from dataclasses import asdict, dataclass
import math
from collections.abc import Mapping


@dataclass(frozen=True)
class LossWeights:
    flow_residual: float = 1.0
    flow_mean_shift: float = 32.0
    decoder_residual: float = 1.0
    decoder_mean: float = 8.0
    source_anchor: float = 0.0
    decoder_true_se: float = 0.0

    def __post_init__(self):
        if any(not math.isfinite(x) or x < 0 for x in asdict(self).values()):
            raise ValueError("Loss weights must be finite and nonnegative")

    @classmethod
    def from_config(cls, config=None):
        if config is None:
            return cls()
        if isinstance(config, cls):
            return config
        if not isinstance(config, Mapping):
            raise TypeError("loss_weights must be a mapping or LossWeights")
        return cls(**config)

    def to_dict(self):
        return asdict(self)


def centered_residual_mse(prediction, target):
    """Inputs are [conditions, cells, features]; center cells per condition."""
    pm, tm = prediction.mean(1), target.mean(1)
    return ((prediction - pm[:, None]) - (target - tm[:, None])).square().mean()


def population_mean_mse(prediction_mean, target_population_mean):
    return (prediction_mean - target_population_mean).square().mean()


def centered_source_anchor_mse(source, frozen_source):
    """Anchor within-condition cell differences, allowing mean shifts.

    Both inputs have shape [conditions, cells, features] and refer to the
    same paired control rows. The original frozen SE reference is supervision
    only; it must never receive a gradient.
    """
    if source.ndim != 3 or source.shape != frozen_source.shape:
        raise ValueError("Source anchors require aligned [conditions, cells, features] tensors")
    return centered_residual_mse(source, frozen_source.detach())
