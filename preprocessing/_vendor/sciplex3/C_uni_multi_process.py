#!/usr/bin/env python3
"""C: make a deterministic unseen-drug split for SciPlex3."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from sciplex_utils import (
    dump_json,
    resolve_cell_lines,
    sample_up_to,
    selected_cells,
    stable_cell_line_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create SciPlex3 train/test/control indices.")
    parser.add_argument("--manifest-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--cell-lines", nargs="+", default=None)
    parser.add_argument("--unseen-drug-ratio", type=float, default=0.05)
    parser.add_argument("--internal-test-ratio", type=float, default=0.2)
    parser.add_argument("--min-cells-per-condition", type=int, default=10)
    parser.add_argument("--max-cells-per-condition", type=int, default=5000)
    parser.add_argument("--max-control-cells", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def eligible_perturbations(cells: pd.DataFrame, minimum: int) -> pd.DataFrame:
    perturb = cells[~cells["is_control"]].copy()
    sizes = perturb.groupby(["cell_line", "drug_conc"])["global_index"].transform("size")
    return perturb[sizes >= int(minimum)].copy()


def capped_group_indices(
    frame: pd.DataFrame,
    maximum: int,
    rng: np.random.Generator,
) -> list[int]:
    values: list[int] = []
    for _, group in frame.groupby("drug_conc", sort=True):
        values.extend(sample_up_to(group["global_index"], maximum, rng))
    return sorted(values)


def write_indices(path: Path, values: Iterable[int]) -> None:
    dump_json(sorted(int(value) for value in values), path)


def main() -> None:
    args = parse_args()
    if not 0 < args.unseen_drug_ratio < 1:
        raise ValueError("--unseen-drug-ratio must be between 0 and 1")
    if not 0 <= args.internal_test_ratio < 1:
        raise ValueError("--internal-test-ratio must be in [0, 1)")
    cell_lines = resolve_cell_lines(args.cell_lines)
    cells = selected_cells(args.manifest_dir, cell_lines)
    perturb = eligible_perturbations(cells, args.min_cells_per_condition)
    drugs = sorted(perturb["drug_id"].unique().tolist())
    if len(drugs) < 2:
        raise ValueError("Need at least two eligible drugs to make an unseen-drug split")
    rng_global = np.random.default_rng(args.seed)
    num_unseen = max(1, int(round(len(drugs) * args.unseen_drug_ratio)))
    num_unseen = min(num_unseen, len(drugs) - 1)
    unseen_drugs = sorted(rng_global.choice(drugs, size=num_unseen, replace=False).tolist())

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    summaries: dict[str, object] = {}
    for cell_line in cell_lines:
        line_dir = out_dir / cell_line
        line_dir.mkdir(parents=True, exist_ok=True)
        line_cells = cells[cells["cell_line"] == cell_line]
        line_perturb = perturb[perturb["cell_line"] == cell_line]
        rng = np.random.default_rng(stable_cell_line_seed(args.seed, cell_line))

        external = line_perturb[line_perturb["drug_id"].isin(unseen_drugs)]
        internal = line_perturb[~line_perturb["drug_id"].isin(unseen_drugs)]
        test_indices = capped_group_indices(external, args.max_cells_per_condition, rng)

        train_indices: list[int] = []
        internal_test_indices: list[int] = []
        for _, group in internal.groupby("drug_conc", sort=True):
            sampled = np.asarray(
                sample_up_to(group["global_index"], args.max_cells_per_condition, rng),
                dtype=np.int64,
            )
            rng.shuffle(sampled)
            num_internal_test = (
                max(1, int(round(len(sampled) * args.internal_test_ratio)))
                if args.internal_test_ratio > 0 and len(sampled) > 1
                else 0
            )
            internal_test_indices.extend(int(x) for x in sampled[:num_internal_test])
            train_indices.extend(int(x) for x in sampled[num_internal_test:])

        controls = line_cells[line_cells["is_control"]]["global_index"]
        control_indices = sample_up_to(controls, args.max_control_cells, rng)
        if not train_indices or not test_indices or not control_indices:
            raise ValueError(
                f"Empty split for {cell_line}: train={len(train_indices)}, "
                f"test={len(test_indices)}, control={len(control_indices)}"
            )
        write_indices(line_dir / "train_indices.json", train_indices)
        write_indices(line_dir / "test_indices.json", test_indices)
        write_indices(line_dir / "external_test_indices.json", test_indices)
        write_indices(line_dir / "internal_test_indices.json", internal_test_indices)
        write_indices(line_dir / "control_indices.json", control_indices)
        summaries[cell_line] = {
            "train_cells": len(train_indices),
            "test_cells": len(test_indices),
            "internal_test_cells": len(internal_test_indices),
            "control_cells": len(control_indices),
            "eligible_conditions": int(line_perturb["drug_conc"].nunique()),
        }

    split_info = {
        "split": "unseen_drug",
        "seed": args.seed,
        "unseen_drug_ratio": args.unseen_drug_ratio,
        "unseen_drugs": unseen_drugs,
        "min_cells_per_condition": args.min_cells_per_condition,
        "max_cells_per_condition": args.max_cells_per_condition,
        "cell_lines": summaries,
        "index_type": "global zero-based matrix row / ds_level_index",
    }
    dump_json(split_info, out_dir / "split_info.json")
    print(f"C complete: held out {len(unseen_drugs)}/{len(drugs)} drugs: {unseen_drugs}")
    print(f"Splits written under {out_dir}")


if __name__ == "__main__":
    main()
