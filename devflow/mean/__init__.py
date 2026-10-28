"""Trainable conditional means and distribution-preserving composition.

The tensor interfaces are dataset-independent. Task adapters provide condition
features, training anchors, strata and gene axes; no dataset sizes live here.
"""

from .composition import compose_expression
from .head import KernelMeanHead, TrainableKernelMeanHead
from .kernels import MixedKernelBank, morgan_fingerprints

__all__ = ["compose_expression", "KernelMeanHead", "TrainableKernelMeanHead",
           "MixedKernelBank", "morgan_fingerprints"]
