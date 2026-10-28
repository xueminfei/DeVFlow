#!/usr/bin/env python3
"""B: validate the SciPlex3 manifest and summarize drugs, doses, and cell counts."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from rdkit import Chem

from sciplex_utils import DiskCSR, dump_json, resolve_cell_lines, selected_cells


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate stage-A SciPlex3 artifacts.")
    parser.add_argument("--manifest-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--cell-lines", nargs="+", default=None)
    parser.add_argument(
        "--allow-invalid-smiles",
        action="store_true",
        help="Report invalid perturbation SMILES without failing",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cell_lines = resolve_cell_lines(args.cell_lines)
    cells = selected_cells(args.manifest_dir, cell_lines)
    matrix = DiskCSR(args.manifest_dir)
    if matrix.shape[0] <= int(cells["global_index"].max()):
        raise ValueError("Cell manifest contains indices outside the count matrix")

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    summaries: dict[str, object] = {}
    invalid_smiles: list[dict[str, str]] = []

    for cell_line in cell_lines:
        frame = cells[cells["cell_line"] == cell_line]
        controls = frame[frame["is_control"]]
        perturb = frame[~frame["is_control"]]
        if controls.empty or perturb.empty:
            raise ValueError(
                f"{cell_line} requires both control and perturbation cells; "
                f"found {len(controls)} controls and {len(perturb)} perturbations"
            )

        conditions = (
            perturb.groupby(["drug_id", "drug_name", "drug_smiles", "dose", "drug_conc"], dropna=False)
            .size()
            .rename("num_cells")
            .reset_index()
            .sort_values(["drug_id", "dose"], kind="stable")
        )
        conditions.to_csv(out_dir / f"B_{cell_line}_conditions.tsv", sep="\t", index=False)
        for row in conditions[["drug_id", "drug_smiles"]].drop_duplicates().itertuples(index=False):
            if not row.drug_smiles or Chem.MolFromSmiles(str(row.drug_smiles)) is None:
                invalid_smiles.append(
                    {"cell_line": cell_line, "drug_id": str(row.drug_id), "smiles": str(row.drug_smiles)}
                )

        summaries[cell_line] = {
            "total_cells": int(len(frame)),
            "control_cells": int(len(controls)),
            "perturbation_cells": int(len(perturb)),
            "unique_drugs": int(perturb["drug_id"].nunique()),
            "unique_drug_dose_conditions": int(perturb["drug_conc"].nunique()),
            "minimum_cells_per_condition": int(conditions["num_cells"].min()),
            "median_cells_per_condition": float(conditions["num_cells"].median()),
            "maximum_cells_per_condition": int(conditions["num_cells"].max()),
        }

    result = {
        "matrix_shape": list(matrix.shape),
        "matrix_nnz": matrix.nnz,
        "cell_lines": summaries,
        "invalid_perturbation_smiles": invalid_smiles,
    }
    dump_json(result, out_dir / "summary.json")
    for cell_line, summary in summaries.items():
        print(f"B {cell_line}: {summary}")
    if invalid_smiles and not args.allow_invalid_smiles:
        raise ValueError(
            f"Found {len(invalid_smiles)} invalid perturbation SMILES; see {out_dir / 'summary.json'}"
        )
    print(f"B complete: {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
