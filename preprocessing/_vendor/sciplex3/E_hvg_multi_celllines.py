#!/usr/bin/env python3
"""E: select SciPlex3 HVGs and export aligned raw HVG counts for every cell."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy import sparse
from tqdm import tqdm

from sciplex_utils import (
    DiskCSR,
    dump_json,
    load_genes,
    load_json,
    open_npy_for_resume,
    resolve_cell_lines,
    selected_cells,
    stable_cell_line_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select and extract 2,000 SciPlex3 HVGs.")
    parser.add_argument("--manifest-dir", required=True)
    parser.add_argument("--split-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--cell-lines", nargs="+", default=None)
    parser.add_argument("--n-hvg", type=int, default=2000)
    parser.add_argument("--sample-cells-per-line", type=int, default=10_000)
    parser.add_argument("--chunk-cells", type=int, default=1024)
    parser.add_argument("--flavor", choices=["seurat_v3", "variance"], default="seurat_v3")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit-cells", type=int, default=None, help="For smoke tests only")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _load_split_indices(split_dir: Path, cell_line: str) -> np.ndarray:
    values: list[int] = []
    for filename in ("train_indices.json", "control_indices.json"):
        path = split_dir / cell_line / filename
        if not path.is_file():
            raise FileNotFoundError(f"Run stage C first; missing {path}")
        data = load_json(path)
        if not isinstance(data, list):
            raise TypeError(f"Expected a JSON list in {path}")
        values.extend(int(value) for value in data)
    return np.unique(np.asarray(values, dtype=np.int64))


def _balanced_sample(
    split_dir: Path,
    cell_lines: list[str],
    per_line: int,
    seed: int,
) -> np.ndarray:
    samples: list[np.ndarray] = []
    for cell_line in cell_lines:
        indices = _load_split_indices(split_dir, cell_line)
        rng = np.random.default_rng(stable_cell_line_seed(seed, cell_line))
        if per_line > 0 and len(indices) > per_line:
            indices = rng.choice(indices, size=per_line, replace=False)
        samples.append(np.sort(indices))
        print(f"E {cell_line}: using {len(indices):,} train/control cells for HVG selection")
    return np.concatenate(samples)


def _variance_hvgs(matrix: sparse.csr_matrix, n_hvg: int) -> np.ndarray:
    matrix = matrix.astype(np.float64)
    row_sums = np.asarray(matrix.sum(axis=1)).ravel()
    scale = np.divide(10_000.0, row_sums, out=np.zeros_like(row_sums), where=row_sums > 0)
    normalized = sparse.diags(scale).dot(matrix).tocsr()
    normalized.data = np.log1p(normalized.data)
    mean = np.asarray(normalized.mean(axis=0)).ravel()
    mean_square = np.asarray(normalized.power(2).mean(axis=0)).ravel()
    variance = np.maximum(0.0, mean_square - mean * mean)
    order = np.lexsort((np.arange(len(variance)), -variance))
    return np.sort(order[:n_hvg].astype(np.int32))


def _select_hvgs(matrix: sparse.csr_matrix, n_hvg: int, flavor: str) -> np.ndarray:
    if n_hvg > matrix.shape[1]:
        raise ValueError(f"Requested {n_hvg} HVGs from only {matrix.shape[1]} genes")
    if flavor == "variance":
        return _variance_hvgs(matrix, n_hvg)
    try:
        import anndata as ad
        import scanpy as sc
    except ImportError as exc:
        raise RuntimeError(
            "The seurat_v3 flavor requires scanpy, anndata, and scikit-misc. "
            "Install the project requirements or use --flavor variance for a smoke test."
        ) from exc
    adata = ad.AnnData(X=matrix)
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=n_hvg, inplace=True)
    indices = np.flatnonzero(adata.var["highly_variable"].to_numpy())
    if len(indices) != n_hvg:
        raise RuntimeError(f"scanpy selected {len(indices)} HVGs; expected {n_hvg}")
    return indices.astype(np.int32)


def main() -> None:
    args = parse_args()
    cell_lines = resolve_cell_lines(args.cell_lines)
    matrix = DiskCSR(args.manifest_dir)
    cells = selected_cells(args.manifest_dir, cell_lines)
    genes = load_genes(args.manifest_dir)
    split_dir = Path(args.split_dir).expanduser().resolve()
    out_root = Path(args.out_dir).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    hvg_indices_path = out_root / "hvg_indices.npy"
    hvg_info_path = out_root / "hvg_info.json"

    if hvg_indices_path.exists() and not args.overwrite:
        if not hvg_info_path.is_file():
            raise FileNotFoundError(f"Missing metadata for existing HVGs: {hvg_info_path}")
        existing_info = load_json(hvg_info_path)
        if not isinstance(existing_info, dict):
            raise TypeError(f"Invalid HVG metadata: {hvg_info_path}")
        expected = {"n_hvg": args.n_hvg, "flavor": args.flavor, "seed": args.seed}
        actual = {key: existing_info.get(key) for key in expected}
        if actual != expected:
            raise ValueError(
                f"HVG settings changed from {actual} to {expected}; use --overwrite to restart stage E"
            )
        hvg_indices = np.load(hvg_indices_path)
        if len(hvg_indices) != args.n_hvg:
            raise ValueError(
                f"Existing {hvg_indices_path} has {len(hvg_indices)} genes, expected {args.n_hvg}"
            )
        print(f"E: reusing {hvg_indices_path}")
    else:
        sample_indices = _balanced_sample(
            split_dir, cell_lines, args.sample_cells_per_line, args.seed
        )
        print(f"E: materializing a {len(sample_indices):,}-cell sparse matrix for HVG selection")
        sample_matrix = matrix.rows_to_csr(sample_indices)
        hvg_indices = _select_hvgs(sample_matrix, args.n_hvg, args.flavor)
        np.save(hvg_indices_path, hvg_indices)
        del sample_matrix

    hvg_genes = genes.iloc[hvg_indices]
    dump_json(
        {
            "n_hvg": int(len(hvg_indices)),
            "flavor": args.flavor,
            "seed": args.seed,
            "raw_gene_indices": [int(value) for value in hvg_indices],
            "gene_symbols": hvg_genes["gene_symbol"].astype(str).tolist(),
            "ensembl_ids": hvg_genes["ensembl_id"].astype(str).tolist(),
            "values_are_raw_umi_counts": True,
        },
        hvg_info_path,
    )

    for cell_line in cell_lines:
        line_cells = cells[cells["cell_line"] == cell_line].reset_index(drop=True)
        global_indices = line_cells["global_index"].to_numpy(dtype=np.int64)
        total = len(global_indices)
        stop = total if args.limit_cells is None else min(total, args.limit_cells)
        line_dir = out_root / cell_line
        line_dir.mkdir(parents=True, exist_ok=True)
        output_path = line_dir / f"{cell_line}_hvg.npy"
        progress_path = line_dir / "progress.json"
        if args.overwrite:
            for path in (output_path, progress_path):
                if path.exists():
                    path.unlink()
        progress = load_json(progress_path) if progress_path.exists() else {"next_row": 0}
        assert isinstance(progress, dict)
        start = int(progress.get("next_row", 0))
        output = open_npy_for_resume(
            output_path, shape=(total, args.n_hvg), dtype=np.float32, overwrite=False
        )
        batches = range(start, stop, args.chunk_cells)
        for batch_start in tqdm(batches, desc=f"E {cell_line}", unit="chunk"):
            batch_stop = min(batch_start + args.chunk_cells, stop)
            batch = matrix.rows_to_csr(global_indices[batch_start:batch_stop])
            output[batch_start:batch_stop] = batch[:, hvg_indices].toarray().astype(np.float32)
            output.flush()
            dump_json(
                {
                    "next_row": batch_stop,
                    "total_rows": total,
                    "complete": batch_stop == total,
                    "n_hvg": args.n_hvg,
                },
                progress_path,
            )
        print(f"E {cell_line}: HVG counts written to {output_path}")
        if stop < total:
            print(f"E {cell_line}: partial smoke-test output ({stop:,}/{total:,} rows)")


if __name__ == "__main__":
    main()
