#!/usr/bin/env python3
"""F: assemble SciPlex3 metadata, embeddings, HVGs, and splits for MAP training."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np

from sciplex_utils import (
    dump_json,
    link_or_copy,
    load_json,
    resolve_cell_lines,
    selected_cells,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Assemble final MAP SciPlex3 processed files.")
    parser.add_argument("--manifest-dir", required=True)
    parser.add_argument("--split-dir", required=True)
    parser.add_argument("--embedding-dir", required=True)
    parser.add_argument("--hvg-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--cell-lines", nargs="+", default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _require_complete(progress_path: Path, stage: str) -> None:
    if not progress_path.is_file():
        raise FileNotFoundError(f"Missing {stage} progress file: {progress_path}")
    progress = load_json(progress_path)
    if not isinstance(progress, dict) or not progress.get("complete", False):
        raise RuntimeError(f"{stage} is incomplete according to {progress_path}: {progress}")


def _copy_split_files(split_dir: Path, destination: Path, valid_indices: set[int]) -> None:
    required = ("train_indices.json", "test_indices.json", "control_indices.json")
    optional = (
        "external_test_indices.json",
        "internal_test_indices.json",
        "train_indices_UC.json",
        "test_indices_UC.json",
        "external_test_indices_UC.json",
        "control_indices_UC.json",
    )
    for filename in required + optional:
        source = split_dir / filename
        if not source.is_file():
            if filename in required:
                raise FileNotFoundError(f"Run stage C first; missing {source}")
            continue
        values = load_json(source)
        if not isinstance(values, list):
            raise TypeError(f"Expected a list in {source}")
        unknown = set(int(value) for value in values).difference(valid_indices)
        if unknown:
            raise ValueError(f"{source} contains {len(unknown)} indices outside this cell line")
        dump_json([int(value) for value in values], destination / filename)


def main() -> None:
    args = parse_args()
    cell_lines = resolve_cell_lines(args.cell_lines)
    cells = selected_cells(args.manifest_dir, cell_lines)
    split_root = Path(args.split_dir).expanduser().resolve()
    embedding_root = Path(args.embedding_dir).expanduser().resolve()
    hvg_root = Path(args.hvg_dir).expanduser().resolve()
    out_root = Path(args.out_dir).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    hvg_info = hvg_root / "hvg_info.json"
    if not hvg_info.is_file():
        raise FileNotFoundError(f"Run stage E first; missing {hvg_info}")
    link_or_copy(hvg_info, out_root / "hvg_info.json", overwrite=args.overwrite)

    for cell_line in cell_lines:
        line_cells = cells[cells["cell_line"] == cell_line].reset_index(drop=True)
        total = len(line_cells)
        embedding_source = embedding_root / cell_line / f"{cell_line}_embeddings.npy"
        hvg_source = hvg_root / cell_line / f"{cell_line}_hvg.npy"
        _require_complete(embedding_root / cell_line / "progress.json", "D embedding extraction")
        _require_complete(hvg_root / cell_line / "progress.json", "E HVG extraction")
        embedding = np.load(embedding_source, mmap_mode="r")
        hvg = np.load(hvg_source, mmap_mode="r")
        if embedding.shape != (total, 2048):
            raise ValueError(f"Unexpected embedding shape for {cell_line}: {embedding.shape}")
        if hvg.shape[0] != total:
            raise ValueError(f"Unexpected HVG row count for {cell_line}: {hvg.shape}")
        del embedding, hvg

        line_dir = out_root / cell_line
        line_dir.mkdir(parents=True, exist_ok=True)
        meta_path = line_dir / f"{cell_line}_meta.pkl"
        if meta_path.exists() and not args.overwrite:
            raise FileExistsError(f"Refusing to overwrite {meta_path}")
        meta = {
            "ds_level_index": line_cells["global_index"].astype(int).tolist(),
            "cell_id": line_cells["cell_id"].astype(str).tolist(),
            "cell_line_id": line_cells["cell_line"].astype(str).tolist(),
            "replicate": line_cells["replicate"].astype(str).tolist(),
            "drug_id": line_cells["drug_id"].astype(str).tolist(),
            "drug_name": line_cells["drug_name"].astype(str).tolist(),
            "drug_smiles": line_cells["drug_smiles"].astype(str).tolist(),
            "drug_conc": line_cells["drug_conc"].astype(str).tolist(),
            "dose": line_cells["dose"].astype(float).tolist(),
            "is_control": line_cells["is_control"].astype(bool).tolist(),
        }
        with meta_path.open("wb") as handle:
            pickle.dump(meta, handle, protocol=pickle.HIGHEST_PROTOCOL)

        link_or_copy(
            embedding_source,
            line_dir / f"{cell_line}_embeddings.npy",
            overwrite=args.overwrite,
        )
        link_or_copy(
            hvg_source,
            line_dir / f"{cell_line}_hvg.npy",
            overwrite=args.overwrite,
        )
        _copy_split_files(
            split_root / cell_line,
            line_dir,
            set(line_cells["global_index"].astype(int).tolist()),
        )
        print(f"F {cell_line}: assembled {total:,} aligned cells under {line_dir}")

    dump_json(
        {
            "cell_lines": cell_lines,
            "embedding_dim": 2048,
            "n_hvg": int(np.load(hvg_root / "hvg_indices.npy").shape[0]),
            "index_type": "global zero-based matrix row / ds_level_index",
        },
        out_root / "manifest.json",
    )
    print(f"F complete: use --data_base_dir {out_root}")


if __name__ == "__main__":
    main()
