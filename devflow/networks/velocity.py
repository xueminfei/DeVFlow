"""Compatibility wrapper for :mod:`devflow.networks.attention_velocity`.

Use ``attention_velocity`` for new imports. This module remains for old scripts
and checkpoint-adjacent code that imported the historical path.
"""
from .attention_velocity import *  # noqa: F401,F403
from .attention_velocity import AttentionCellFlowVelocity, TimeModulatedMAPFlow
