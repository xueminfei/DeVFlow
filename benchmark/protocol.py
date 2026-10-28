"""Explicit evaluation contracts; these are not model training recipes."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path


@dataclass(frozen=True)
class EvaluationProtocol:
    protocol: str
    task: str
    seeds: tuple[int, ...]
    condition_count: int
    cells_per_condition: int
    n_genes: int
    distance_backend_hvg: str = "torch_fp32_pca50"
    de50_size: int = 50
    gene_order_sha256: str | None = None
    response_profile: str = "sampled"
    training_binding_sha256: str | None = None

    @classmethod
    def load(cls, path: str | Path) -> "EvaluationProtocol":
        record = json.loads(Path(path).read_text())
        required = ("protocol", "task", "seeds", "condition_count", "cells_per_condition", "n_genes")
        missing = set(required) - set(record)
        if missing:
            raise ValueError(f"Protocol missing fields: {sorted(missing)}")
        fields = {name: record[name] for name in cls.__dataclass_fields__ if name in record}
        fields["seeds"] = tuple(map(int, fields["seeds"]))
        result = cls(**fields)
        if len(result.seeds) < 2 or len(set(result.seeds)) != len(result.seeds):
            raise ValueError("A repeated evaluation requires at least two distinct seeds")
        if min(result.condition_count, result.cells_per_condition, result.n_genes) < 2:
            raise ValueError("Condition, cell, and gene counts must be >=2")
        if result.distance_backend_hvg not in ("torch_fp32_pca50", "scipy_float64", "combo_historical", "norman_historical"):
            raise ValueError("Unknown HVG distance backend")
        if result.response_profile not in ("sampled", "full_treated"):
            raise ValueError("Unknown response_profile definition")
        if not 2 <= result.de50_size <= result.n_genes:
            raise ValueError("DE panel size must lie within the full gene axis")
        return result
