#!/usr/bin/env python3
"""A: ingest the official GEO SciPlex3 files into a disk-backed cell-by-gene CSR."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from sciplex_utils import (
    discover_raw_files,
    dump_json,
    ensure_new_file,
    resolve_cell_lines,
)


PDATA_COLUMNS = [
    "cell",
    "sample",
    "Combo",
    "cell_type",
    "replicate",
    "time_point",
    "drug_dose",
    "catalog_number",
    "vehicle",
    "dose",
    "treatment",
    "product_name",
    "target",
    "pathway",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ingest raw GEO SciPlex3 metadata and coordinate count matrix."
    )
    parser.add_argument("--raw-dir", required=True, help="Directory containing the six GEO .txt.gz files")
    parser.add_argument("--out-dir", required=True, help="Output directory for the A-stage manifest")
    parser.add_argument("--cell-lines", nargs="+", default=None, help="Subset of A549 MCF7 K562")
    parser.add_argument(
        "--time-points",
        nargs="+",
        type=float,
        default=[24.0],
        help="Time points to retain (default: 24). Pass 24 72 to include the A549 72 h screen.",
    )
    parser.add_argument("--chunk-rows", type=int, default=2_000_000, help="Coordinate rows per parser chunk")
    parser.add_argument("--max-matrix-rows", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _read_metadata(
    files: dict[str, Path], cell_lines: list[str], time_points: list[float]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    annotations = pd.read_csv(
        files["cells"], sep="\t", header=None, names=["cell_id", "screen"], dtype=str
    )
    pdata = pd.read_csv(
        files["pdata"], sep=r"\s+", usecols=PDATA_COLUMNS, low_memory=False
    )
    pdata = pdata.reset_index(drop=True)
    if len(pdata) != len(annotations):
        raise ValueError(
            f"pData/cell annotation row mismatch: {len(pdata)} versus {len(annotations)}"
        )
    if not np.array_equal(pdata["cell"].astype(str).to_numpy(), annotations["cell_id"].to_numpy()):
        mismatch = np.flatnonzero(
            pdata["cell"].astype(str).to_numpy() != annotations["cell_id"].to_numpy()
        )[:5]
        raise ValueError(f"pData is not aligned to cell annotations; first mismatches: {mismatch.tolist()}")

    hash_meta = pd.read_csv(files["hash_metadata"], sep="\t", low_memory=False)
    duplicate_combo = hash_meta.groupby("Combo", dropna=False)["SMILES"].nunique(dropna=False)
    conflicting = duplicate_combo[duplicate_combo > 1]
    if not conflicting.empty:
        raise ValueError(f"Conflicting SMILES for {len(conflicting)} Combo values")
    combo_meta = hash_meta.drop_duplicates("Combo").set_index("Combo")

    dose = pd.to_numeric(pdata["dose"], errors="coerce").fillna(0.0)
    vehicle = pdata["vehicle"].astype(str).str.upper().isin({"TRUE", "T", "1"})
    treatment = pdata["treatment"].fillna("S0000").astype(str)
    catalog = pdata["catalog_number"].fillna(treatment).astype(str)
    is_control = vehicle | dose.le(0) | treatment.eq("S0000") | catalog.eq("S0000")
    smiles_map = combo_meta["SMILES"].to_dict()
    name_map = combo_meta["name"].to_dict()
    smiles = pdata["Combo"].map(smiles_map)
    compound_name = pdata["Combo"].map(name_map).fillna(pdata["product_name"])

    condition = [
        repr([("DMSO", 0.0, "nM")])
        if control
        else repr([(drug, float(concentration), "nM")])
        for control, drug, concentration in zip(is_control, catalog, dose)
    ]
    numeric_time_point = pd.to_numeric(pdata["time_point"], errors="coerce")
    cells = pd.DataFrame(
        {
            "global_index": np.arange(len(pdata), dtype=np.int64),
            "cell_id": annotations["cell_id"].astype(str),
            "screen": annotations["screen"].astype(str),
            "cell_line": pdata["cell_type"].astype(str),
            "replicate": pdata["replicate"].astype(str),
            "time_point": numeric_time_point,
            "combo": pdata["Combo"].astype(str),
            "drug_id": catalog,
            "drug_name": compound_name.fillna(catalog).astype(str),
            "drug_smiles": smiles.fillna("C").astype(str),
            "dose": dose.astype(np.float32),
            "drug_conc": condition,
            "is_control": is_control.astype(bool),
            "selected": (
                pdata["cell_type"].astype(str).isin(cell_lines)
                & numeric_time_point.isin(time_points)
            ),
        }
    )

    genes = pd.read_csv(files["genes"], sep=r"\s+", dtype=str)
    if list(genes.columns[:2]) != ["id", "gene_short_name"]:
        raise ValueError(f"Unexpected gene annotation columns: {list(genes.columns)}")
    genes = pd.DataFrame(
        {
            "gene_index": np.arange(len(genes), dtype=np.int32),
            "ensembl_id": genes["id"].astype(str),
            "gene_symbol": genes["gene_short_name"].astype(str),
        }
    )
    return cells, genes


def _write_matrix(
    matrix_path: Path,
    matrix_dir: Path,
    num_cells: int,
    num_genes: int,
    chunk_rows: int,
    max_matrix_rows: int | None,
    overwrite: bool,
) -> dict[str, object]:
    matrix_dir.mkdir(parents=True, exist_ok=True)
    final_indices = matrix_dir / "indices.bin"
    final_counts = matrix_dir / "counts.bin"
    final_indptr = matrix_dir / "indptr.npy"
    final_meta = matrix_dir / "matrix.json"
    for path in (final_indices, final_counts, final_indptr, final_meta):
        ensure_new_file(path, overwrite=overwrite)

    indices_part = matrix_dir / "indices.bin.part"
    counts_part = matrix_dir / "counts.bin.part"
    if indices_part.exists():
        indices_part.unlink()
    if counts_part.exists():
        counts_part.unlink()

    cell_nnz = np.zeros(num_cells, dtype=np.int64)
    total_nnz = 0
    previous_cell = 0
    reader = pd.read_csv(
        matrix_path,
        sep="\t",
        header=None,
        names=["gene", "cell", "count"],
        dtype={"gene": np.int32, "cell": np.int32, "count": np.uint32},
        chunksize=chunk_rows,
        nrows=max_matrix_rows,
    )
    with indices_part.open("wb") as index_handle, counts_part.open("wb") as count_handle:
        for chunk in tqdm(reader, desc="A: coordinate matrix", unit="chunk"):
            gene = chunk["gene"].to_numpy(copy=False)
            cell = chunk["cell"].to_numpy(copy=False)
            count = chunk["count"].to_numpy(copy=False)
            if len(cell) == 0:
                continue
            if int(gene.min()) < 1 or int(gene.max()) > num_genes:
                raise ValueError("Matrix contains a gene index outside the annotation range")
            if int(cell.min()) < 1 or int(cell.max()) > num_cells:
                raise ValueError("Matrix contains a cell index outside the annotation range")
            if int(cell[0]) < previous_cell or np.any(cell[1:] < cell[:-1]):
                raise ValueError("Coordinate matrix must be sorted by its 1-based cell column")
            previous_cell = int(cell[-1])
            (gene - 1).astype(np.int32, copy=False).tofile(index_handle)
            count.astype(np.uint32, copy=False).tofile(count_handle)
            cell_nnz += np.bincount(cell - 1, minlength=num_cells)
            total_nnz += len(chunk)

    indptr = np.lib.format.open_memmap(
        final_indptr, mode="w+", dtype=np.int64, shape=(num_cells + 1,)
    )
    indptr[0] = 0
    np.cumsum(cell_nnz, out=indptr[1:])
    indptr.flush()
    del indptr
    os.replace(indices_part, final_indices)
    os.replace(counts_part, final_counts)

    metadata = {
        "format": "csr",
        "shape": [num_cells, num_genes],
        "num_cells": num_cells,
        "num_genes": num_genes,
        "nnz": int(total_nnz),
        "indices_dtype": "int32",
        "counts_dtype": "uint32",
        "source_is_one_based": True,
        "complete": max_matrix_rows is None,
    }
    dump_json(metadata, final_meta)
    return metadata


def main() -> None:
    args = parse_args()
    cell_lines = resolve_cell_lines(args.cell_lines)
    files = discover_raw_files(args.raw_dir)
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    cells_path = out_dir / "cells.pkl"
    genes_path = out_dir / "genes.pkl"
    raw_files_path = out_dir / "raw_files.json"
    for path in (cells_path, genes_path, raw_files_path):
        ensure_new_file(path, overwrite=args.overwrite)

    print("A: reading and aligning SciPlex3 annotations")
    cells, genes = _read_metadata(files, cell_lines, args.time_points)
    cells.to_pickle(cells_path)
    genes.to_pickle(genes_path)
    dump_json({key: str(value) for key, value in files.items()}, raw_files_path)
    matrix_meta = _write_matrix(
        files["matrix"],
        out_dir / "matrix",
        num_cells=len(cells),
        num_genes=len(genes),
        chunk_rows=args.chunk_rows,
        max_matrix_rows=args.max_matrix_rows,
        overwrite=args.overwrite,
    )
    print(
        f"A complete: {len(cells):,} cells, {len(genes):,} genes, "
        f"{matrix_meta['nnz']:,} non-zero counts"
    )
    print(f"Selected cell lines/time points: {cell_lines}/{args.time_points}; output: {out_dir}")


if __name__ == "__main__":
    main()
