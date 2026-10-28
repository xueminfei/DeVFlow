"""Compact control-cell conditioning built from frozen MAP features."""

import torch
from torch import nn


class ControlConditionFusion(nn.Module):
    """Align the frozen MAP-original SE-600M cell embedding for the flow.

    This branch intentionally conditions velocity on the SE cell embedding
    alone. The original protein/gene summary path remains outside this
    architecture; gene_summary is accepted only for call-site compatibility.
    """

    def __init__(self, cell_dim=2048, hidden_dim=2048, output_dim=1024,
                 dropout=0.1):
        super().__init__()
        self.cell_dim = int(cell_dim)
        self.net = nn.Sequential(
            nn.LayerNorm(self.cell_dim),
            nn.Linear(self.cell_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, cell_embedding, gene_summary=None):
        if cell_embedding is None or cell_embedding.ndim != 2:
            raise ValueError("cell embedding must be a two-dimensional [N,D] tensor")
        if cell_embedding.size(1) != self.cell_dim:
            raise ValueError(f"cell embedding must be [N,{self.cell_dim}]")
        return self.net(cell_embedding)
