"""Shared, disk-backed utilities for the SciPlex3 preprocessing pipeline."""

from __future__ import annotations

import hashlib
import json
import math
import os
import pickle
import shutil
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import numpy as np
import pandas as pd
from scipy import sparse


DEFAULT_CELL_LINES = ("A549", "MCF7", "K562")
DEFAULT_PAD_LENGTH = 2048


RAW_PATTERNS = {
    "matrix": "*screen_UMI.count.matrix.gz",
    "cells": "*screen_cell.annotations.txt.gz",
    "genes": "*screen_gene.annotations.txt.gz",
    "pdata": "*sciPlex3_pData.txt.gz",
    "hash_metadata": "*hashTable_metadata.txt.gz",
}


def discover_raw_files(raw_dir: str | os.PathLike[str]) -> dict[str, Path]:
    raw_dir = Path(raw_dir).expanduser().resolve()
    if not raw_dir.is_dir():
        raise FileNotFoundError(f"SciPlex3 raw directory does not exist: {raw_dir}")

    result: dict[str, Path] = {}
    for key, pattern in RAW_PATTERNS.items():
        matches = sorted(raw_dir.glob(pattern))
        if len(matches) != 1:
            raise FileNotFoundError(
                f"Expected exactly one file matching {pattern!r} under {raw_dir}, "
                f"found {len(matches)}"
            )
        result[key] = matches[0]
    return result


def ensure_new_file(path: Path, overwrite: bool = False) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)


def dump_json(data: object, path: str | os.PathLike[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=True)


def load_json(path: str | os.PathLike[str]) -> object:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_cells(manifest_dir: str | os.PathLike[str]) -> pd.DataFrame:
    path = Path(manifest_dir) / "cells.pkl"
    if not path.is_file():
        raise FileNotFoundError(f"Run A_fetch_cell_line.py first; missing {path}")
    cells = pd.read_pickle(path)
    required = {
        "global_index",
        "cell_id",
        "cell_line",
        "drug_id",
        "drug_smiles",
        "drug_conc",
        "dose",
        "is_control",
        "selected",
    }
    missing = required.difference(cells.columns)
    if missing:
        raise ValueError(f"Cell manifest is missing columns: {sorted(missing)}")
    return cells


def load_genes(manifest_dir: str | os.PathLike[str]) -> pd.DataFrame:
    path = Path(manifest_dir) / "genes.pkl"
    if not path.is_file():
        raise FileNotFoundError(f"Run A_fetch_cell_line.py first; missing {path}")
    genes = pd.read_pickle(path)
    required = {"gene_index", "ensembl_id", "gene_symbol"}
    missing = required.difference(genes.columns)
    if missing:
        raise ValueError(f"Gene manifest is missing columns: {sorted(missing)}")
    return genes


def selected_cells(
    manifest_dir: str | os.PathLike[str], cell_lines: Sequence[str]
) -> pd.DataFrame:
    cells = load_cells(manifest_dir)
    result = cells[cells["selected"] & cells["cell_line"].isin(cell_lines)].copy()
    if result.empty:
        raise ValueError(f"No selected cells found for cell lines: {list(cell_lines)}")
    return result.sort_values("global_index", kind="stable").reset_index(drop=True)


class DiskCSR:
    """Read a cell-by-gene CSR matrix stored as two raw arrays plus .npy indptr."""

    def __init__(self, manifest_dir: str | os.PathLike[str]):
        matrix_dir = Path(manifest_dir) / "matrix"
        meta_path = matrix_dir / "matrix.json"
        if not meta_path.is_file():
            raise FileNotFoundError(f"Run A_fetch_cell_line.py first; missing {meta_path}")
        meta = load_json(meta_path)
        assert isinstance(meta, dict)
        self.shape = (int(meta["num_cells"]), int(meta["num_genes"]))
        self.nnz = int(meta["nnz"])
        self.indptr = np.load(matrix_dir / "indptr.npy", mmap_mode="r")
        self.indices = np.memmap(
            matrix_dir / "indices.bin",
            dtype=np.dtype(str(meta["indices_dtype"])),
            mode="r",
            shape=(self.nnz,),
        )
        self.data = np.memmap(
            matrix_dir / "counts.bin",
            dtype=np.dtype(str(meta["counts_dtype"])),
            mode="r",
            shape=(self.nnz,),
        )
        if self.indptr.shape != (self.shape[0] + 1,):
            raise ValueError(
                f"Invalid indptr shape {self.indptr.shape}; expected {(self.shape[0] + 1,)}"
            )

    def row(self, row_index: int) -> tuple[np.ndarray, np.ndarray]:
        start = int(self.indptr[row_index])
        end = int(self.indptr[row_index + 1])
        return self.indices[start:end], self.data[start:end]

    def rows_to_csr(self, row_indices: Sequence[int]) -> sparse.csr_matrix:
        row_indices = np.asarray(row_indices, dtype=np.int64)
        lengths = self.indptr[row_indices + 1] - self.indptr[row_indices]
        out_indptr = np.empty(len(row_indices) + 1, dtype=np.int64)
        out_indptr[0] = 0
        np.cumsum(lengths, out=out_indptr[1:])
        out_nnz = int(out_indptr[-1])
        out_indices = np.empty(out_nnz, dtype=np.int32)
        out_data = np.empty(out_nnz, dtype=np.float32)
        write = 0
        for row_index, length in zip(row_indices, lengths):
            length = int(length)
            if length:
                start = int(self.indptr[row_index])
                end = start + length
                out_indices[write : write + length] = self.indices[start:end]
                out_data[write : write + length] = self.data[start:end]
                write += length
        return sparse.csr_matrix(
            (out_data, out_indices, out_indptr), shape=(len(row_indices), self.shape[1])
        )


def load_embedding_symbols(
    embedding_path: str | os.PathLike[str],
) -> tuple[list[str], str]:
    import torch

    path = Path(embedding_path).expanduser().resolve()
    embeddings = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(embeddings, dict):
        raise TypeError(
            "The ESM2 embedding file must be an ordered gene-symbol dictionary; "
            f"got {type(embeddings).__name__}"
        )
    symbols = [str(symbol) for symbol in embeddings.keys()]
    digest = hashlib.sha256("\n".join(symbols).encode("utf-8")).hexdigest()
    del embeddings
    return symbols, digest


def build_gene_to_token(
    genes: pd.DataFrame, embedding_symbols: Sequence[str]
) -> np.ndarray:
    symbol_to_token = {symbol: i for i, symbol in enumerate(embedding_symbols)}
    mapping = np.full(len(genes), -1, dtype=np.int32)
    for row in genes.itertuples(index=False):
        symbol = str(row.gene_symbol)
        token = symbol_to_token.get(symbol)
        if token is not None:
            mapping[int(row.gene_index)] = token
    return mapping


class SciPlexTokenizer:
    """Convert one raw SciPlex cell into the token/count format used by SE600M."""

    def __init__(
        self,
        gene_to_token: np.ndarray,
        num_tokens: int,
        pad_length: int = DEFAULT_PAD_LENGTH,
        seed: int = 42,
    ):
        if pad_length < 2:
            raise ValueError("pad_length must be at least 2")
        if num_tokens < pad_length - 1:
            raise ValueError(
                f"Need at least {pad_length - 1} embedding tokens, got {num_tokens}"
            )
        self.gene_to_token = np.asarray(gene_to_token, dtype=np.int32)
        self.num_tokens = int(num_tokens)
        self.pad_length = int(pad_length)
        self.seed = int(seed)
        rng = np.random.default_rng(seed)
        self.fill_order = rng.permutation(num_tokens).astype(np.int32)
        self.tie_priority = rng.permutation(num_tokens).astype(np.int32)
        self._expressed_mask = np.zeros(num_tokens, dtype=bool)

    def encode(
        self,
        gene_indices: np.ndarray,
        raw_counts: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        tokens = self.gene_to_token[np.asarray(gene_indices, dtype=np.int64)]
        counts = np.asarray(raw_counts, dtype=np.float64)
        valid = (tokens >= 0) & (counts > 0)
        tokens = tokens[valid]
        counts = counts[valid]

        gene_ids = np.zeros(self.pad_length, dtype=np.int32)
        expressions = np.zeros(self.pad_length, dtype=np.float32)
        if tokens.size:
            order = np.argsort(tokens, kind="stable")
            sorted_tokens = tokens[order]
            sorted_counts = counts[order]
            first = np.r_[True, sorted_tokens[1:] != sorted_tokens[:-1]]
            starts = np.flatnonzero(first)
            tokens = sorted_tokens[starts]
            counts = np.add.reduceat(sorted_counts, starts)

            transformed = np.log1p(counts)
            total = float(transformed.sum())
            weights = transformed / total if total > 0 else transformed
            ranked = np.lexsort((self.tie_priority[tokens], -transformed))
            keep = ranked[: self.pad_length - 1]
            selected_tokens = tokens[keep]
            selected_weights = weights[keep]
            n_selected = len(selected_tokens)
            gene_ids[1 : n_selected + 1] = selected_tokens
            expressions[1 : n_selected + 1] = (100.0 * selected_weights).astype(np.float32)
        else:
            selected_tokens = np.empty(0, dtype=np.int32)
            n_selected = 0

        remaining = self.pad_length - 1 - n_selected
        if remaining:
            mask = self._expressed_mask
            mask[selected_tokens] = True
            fillers = self.fill_order[~mask[self.fill_order]][:remaining]
            gene_ids[n_selected + 1 :] = fillers
            mask[selected_tokens] = False
        return gene_ids, expressions


def iter_token_batches(
    matrix: DiskCSR,
    global_indices: Sequence[int],
    tokenizer: SciPlexTokenizer,
    batch_size: int,
    start: int = 0,
    stop: int | None = None,
) -> Iterator[tuple[int, np.ndarray, np.ndarray]]:
    global_indices = np.asarray(global_indices, dtype=np.int64)
    stop = len(global_indices) if stop is None else min(int(stop), len(global_indices))
    for batch_start in range(int(start), stop, int(batch_size)):
        batch_stop = min(batch_start + int(batch_size), stop)
        size = batch_stop - batch_start
        gene_ids = np.empty((size, tokenizer.pad_length), dtype=np.int32)
        expressions = np.empty((size, tokenizer.pad_length), dtype=np.float32)
        for local, global_index in enumerate(global_indices[batch_start:batch_stop]):
            indices, counts = matrix.row(int(global_index))
            gene_ids[local], expressions[local] = tokenizer.encode(indices, counts)
        yield batch_start, gene_ids, expressions


def create_tokenizer(
    manifest_dir: str | os.PathLike[str],
    embedding_path: str | os.PathLike[str],
    pad_length: int = DEFAULT_PAD_LENGTH,
    seed: int = 42,
) -> tuple[SciPlexTokenizer, dict[str, object]]:
    genes = load_genes(manifest_dir)
    symbols, symbol_digest = load_embedding_symbols(embedding_path)
    mapping = build_gene_to_token(genes, symbols)
    tokenizer = SciPlexTokenizer(mapping, len(symbols), pad_length=pad_length, seed=seed)
    metadata = {
        "pad_length": int(pad_length),
        "seed": int(seed),
        "num_embedding_tokens": len(symbols),
        "mapped_raw_genes": int((mapping >= 0).sum()),
        "embedding_symbol_sha256": symbol_digest,
        "count_transform": "100 * log1p(raw_count) / sum(log1p(raw_count))",
        "unexpressed_fill": "fixed seeded permutation",
    }
    return tokenizer, metadata


def open_npy_for_resume(
    path: str | os.PathLike[str],
    shape: tuple[int, ...],
    dtype: str | np.dtype,
    overwrite: bool = False,
) -> np.memmap:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        array = np.load(path, mmap_mode="r+")
        if array.shape != shape or array.dtype != np.dtype(dtype):
            raise ValueError(
                f"Existing array {path} has shape/dtype {array.shape}/{array.dtype}; "
                f"expected {shape}/{np.dtype(dtype)}"
            )
        return array
    return np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)


def link_or_copy(source: Path, destination: Path, overwrite: bool = False) -> None:
    source = source.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if not overwrite:
            raise FileExistsError(f"Refusing to overwrite {destination}")
        destination.unlink()
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def stable_cell_line_seed(seed: int, cell_line: str) -> int:
    digest = hashlib.sha256(cell_line.encode("utf-8")).digest()
    return int(seed) + int.from_bytes(digest[:4], "little")


def sample_up_to(values: Iterable[int], maximum: int, rng: np.random.Generator) -> list[int]:
    array = np.asarray(list(values), dtype=np.int64)
    if maximum > 0 and len(array) > maximum:
        array = rng.choice(array, size=maximum, replace=False)
    return sorted(int(value) for value in array)


def resolve_cell_lines(requested: Sequence[str] | None) -> list[str]:
    values = list(requested) if requested else list(DEFAULT_CELL_LINES)
    unknown = sorted(set(values).difference(DEFAULT_CELL_LINES))
    if unknown:
        raise ValueError(
            f"Unsupported SciPlex3 cell lines {unknown}; expected a subset of {DEFAULT_CELL_LINES}"
        )
    return values
