#!/usr/bin/env python3
"""Prepare Norman gene-order/HVG metadata and frozen SE-600M states.

The Norman release used here stores ``X`` in scanpy ``log1p``-normalized
space.  This script therefore passes ``expm1(X)`` to the repository tokenizer:
the tokenizer's internal ``log1p`` returns the stored normalized values before
its 100-percent top-gene weighting.  The inverted values are normalized linear
values, not recovered UMI counts, and are never labelled as raw counts.

The script is deliberately separate from ``build_norman_se_cache.py`` so the
old drug branches and their caches remain untouched.  The three commands are:
``prepare`` (fixed condition/control manifests, train-only Seurat HVGs and
all-cell tokens), ``encode`` (frozen SE shards), and ``merge``.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
EXTRACT_PATH = Path(__file__).resolve().parents[3] / "combosciplex/extract_se.py"
DEFAULT_SPLIT = Path("condition_manifest.json")  # CLI requires an explicit manifest.


def sha_file(path: Path, chunk: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for part in iter(lambda: handle.read(chunk), b""):
            digest.update(part)
    return digest.hexdigest()


def ordered_hash(values) -> str:
    return hashlib.sha256("\n".join(str(value) for value in values).encode("utf-8")).hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def refuse_completed(path: Path, label: str) -> None:
    """Refuse to replace an artifact whose completion marker is present.

    A failed/interrupted run can be resumed by replacing incomplete artifacts,
    but a completed artifact is immutable so that its hashes remain auditable.
    """
    if not path.exists() or path.stat().st_size == 0:
        return
    if path.suffix == ".json":
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            return
        if isinstance(payload, dict) and payload.get("complete") is True:
            raise FileExistsError(f"refusing to overwrite completed {label}: {path}")


def decode(values) -> list[str]:
    array = np.asarray(values)
    return [value.decode() if isinstance(value, (bytes, np.bytes_)) else str(value) for value in array]


def read_column(group) -> np.ndarray:
    if isinstance(group, h5py.Dataset):
        values = group[()]
        if values.dtype.kind in {"S", "O"}:
            return np.asarray(decode(values), dtype=object)
        return values
    categories = decode(group["categories"][()])
    codes = np.asarray(group["codes"][()], dtype=np.int64)
    return np.asarray([categories[int(code)] for code in codes], dtype=object)


def h5_metadata(h5ad: Path) -> dict[str, Any]:
    with h5py.File(h5ad, "r") as handle:
        x = handle["X"]
        if not {"data", "indices", "indptr"}.issubset(x):
            raise ValueError("Norman h5ad/X must be CSR data, indices, indptr")
        shape = tuple(int(value) for value in x.attrs["shape"])
        cell_ids = decode(handle["obs"]["_index"][()])
        conditions = np.asarray(read_column(handle["obs"]["condition"]), dtype=object)
        control = np.asarray(handle["obs"]["control"][()], dtype=np.int8).astype(bool)
        source_genes = decode(handle["var"].get("gene_name", handle["var"]["_index"])[()])
    if shape != (len(cell_ids), len(source_genes)):
        raise ValueError(f"X shape {shape} disagrees with obs/var lengths")
    if len(set(cell_ids)) != len(cell_ids):
        raise ValueError("h5ad obs/_index contains duplicate cell IDs")
    return dict(shape=shape, cell_ids=cell_ids, conditions=conditions,
                control=control, source_genes=source_genes)


def parse_condition_genes(condition: str) -> tuple[str, ...]:
    tokens = [token.strip() for token in str(condition).split("+") if token.strip()]
    tokens = [token for token in tokens if token.lower() not in {"ctrl", "control"}]
    if len(tokens) not in (1, 2):
        raise ValueError(f"cannot parse Norman condition {condition!r}")
    return tuple(tokens)


def load_fixed_condition_manifest(path: Path, metadata: dict[str, Any]) -> dict[str, Any]:
    manifest = json.loads(path.read_text())
    train = [str(value) for value in manifest.get("train_conditions", [])]
    test = [str(value) for value in manifest.get("test_conditions", [])]
    if len(train) != 201 or len(test) != 25 or set(train) & set(test):
        raise ValueError("Norman condition manifest must contain the fixed disjoint 201/25 split")
    available = {str(value) for value, flag in zip(metadata["conditions"], metadata["control"]) if not flag}
    if set(train) | set(test) != available:
        missing = sorted(available - (set(train) | set(test)))
        extra = sorted((set(train) | set(test)) - available)
        raise ValueError(f"condition manifest mismatch; missing={missing[:5]}, extra={extra[:5]}")
    if manifest.get("split_mode") != "double_pair_holdout":
        raise ValueError("the decoder experiment requires double_pair_holdout")
    return manifest


def make_control_split(metadata: dict[str, Any], condition_manifest: dict[str, Any], seed: int,
                       train_fraction: float = 0.75) -> dict[str, Any]:
    if not 0.0 < float(train_fraction) < 1.0:
        raise ValueError("control_train_fraction must be strictly between 0 and 1")
    cell_ids = metadata["cell_ids"]
    conditions = metadata["conditions"]
    control = metadata["control"]
    control_rows = np.flatnonzero(control).astype(np.int64)
    rng = np.random.default_rng(int(seed))
    permutation = rng.permutation(control_rows)
    n_train = int(np.floor(len(permutation) * float(train_fraction)))
    train_control_rows = np.sort(permutation[:n_train])
    test_control_rows = np.sort(permutation[n_train:])
    train_conditions = set(condition_manifest["train_conditions"])
    test_conditions = set(condition_manifest["test_conditions"])
    train_treated_rows = np.flatnonzero((~control) & np.isin(conditions, list(train_conditions))).astype(np.int64)
    test_treated_rows = np.flatnonzero((~control) & np.isin(conditions, list(test_conditions))).astype(np.int64)
    def ids(rows):
        return [cell_ids[int(row)] for row in rows]
    return {
        "format_version": 1,
        "complete": False,
        "seed": int(seed),
        "control_train_fraction": float(train_fraction),
        "split_unit": "condition plus exact control cell IDs",
        "condition_manifest": str(DEFAULT_SPLIT),
        "condition_manifest_sha256": condition_manifest.get("manifest_sha256"),
        "train_conditions": sorted(train_conditions),
        "test_conditions": sorted(test_conditions),
        "train_control_rows": [int(row) for row in train_control_rows],
        "test_control_rows": [int(row) for row in test_control_rows],
        "train_control_cell_ids": ids(train_control_rows),
        "test_control_cell_ids": ids(test_control_rows),
        "train_treated_rows": [int(row) for row in train_treated_rows],
        "test_treated_rows": [int(row) for row in test_treated_rows],
        "train_treated_cell_ids": ids(train_treated_rows),
        "test_treated_cell_ids": ids(test_treated_rows),
        "all_control_rows_sha256": ordered_hash(control_rows),
        "train_control_rows_sha256": ordered_hash(train_control_rows),
        "test_control_rows_sha256": ordered_hash(test_control_rows),
        "train_control_cell_ids_sha256": ordered_hash(ids(train_control_rows)),
        "test_control_cell_ids_sha256": ordered_hash(ids(test_control_rows)),
        "train_treated_cell_ids_sha256": ordered_hash(ids(train_treated_rows)),
        "test_treated_cell_ids_sha256": ordered_hash(ids(test_treated_rows)),
    }


def fit_hvg(h5ad: Path, fit_rows: np.ndarray, source_genes: list[str], n_top: int):
    import anndata as ad
    import scanpy as sc
    adata = ad.read_h5ad(str(h5ad))
    adata = adata[fit_rows].copy()
    # X is already log1p-normalized in this Norman release.  The existing
    # values are passed to Seurat HVG fitting as-is: no normalize_total/log1p.
    sc.pp.highly_variable_genes(adata, n_top_genes=int(n_top), flavor="seurat", inplace=True)
    mask = np.asarray(adata.var["highly_variable"], dtype=bool)
    indices = np.flatnonzero(mask).astype(np.int64)
    if len(indices) != int(n_top):
        raise RuntimeError(f"Seurat selected {len(indices)} HVGs, expected {n_top}")
    if list(map(str, adata.var_names)) != source_genes:
        raise RuntimeError("anndata var order differs from h5ad gene_name order")
    return indices, [source_genes[int(index)] for index in indices], sc.__version__, ad.__version__


def load_extract():
    spec = importlib.util.spec_from_file_location("map_extract_se_norman", EXTRACT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {EXTRACT_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepare(args) -> None:
    if int(args.n_hvg) < 1:
        raise ValueError("n_hvg must be >= 1")
    if int(args.log_every) < 1:
        raise ValueError("log_every must be >= 1")
    h5ad = Path(args.h5ad).expanduser().resolve()
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    split_path = output / "norman_decoder_split_seed42.json"
    hvg_path = output / "hvg_provenance.json"
    tokenization_path = output / "tokenization.json"
    refuse_completed(split_path, "Norman split")
    refuse_completed(hvg_path, "HVG provenance")
    refuse_completed(tokenization_path, "tokenization")
    condition_path = Path(args.condition_manifest or DEFAULT_SPLIT).expanduser().resolve()
    metadata = h5_metadata(h5ad)
    condition_manifest = load_fixed_condition_manifest(condition_path, metadata)
    condition_manifest["manifest_sha256"] = sha_file(condition_path)
    split = make_control_split(metadata, condition_manifest, args.seed, args.control_train_fraction)
    split["condition_manifest"] = str(condition_path)
    split["condition_manifest_sha256"] = condition_manifest["manifest_sha256"]
    split["h5ad_sha256"] = sha_file(h5ad)
    split["cell_order_sha256"] = ordered_hash(metadata["cell_ids"])
    split["n_cells"] = int(metadata["shape"][0])
    split["n_source_genes"] = int(metadata["shape"][1])
    write_json(split_path, split)
    condition_manifest["complete"] = True
    write_json(output / "condition_manifest.json", condition_manifest)

    fit_rows = np.asarray(split["train_control_rows"] + split["train_treated_rows"], dtype=np.int64)
    hvg_indices, genes, scanpy_version, anndata_version = fit_hvg(
        h5ad, fit_rows, metadata["source_genes"], args.n_hvg
    )
    np.save(output / "gene_indices.npy", hvg_indices)
    write_json(output / "genes.json", genes)
    hvg_provenance = {
        "format_version": 1,
        "complete": False,
        "h5ad": str(h5ad),
        "h5ad_sha256": split["h5ad_sha256"],
        "condition_manifest": str(condition_path),
        "condition_manifest_sha256": condition_manifest["manifest_sha256"],
        "control_split": "75/25 exact IDs; seed 42",
        "fit_scope": "all train controls + all treated cells from fixed 201 train conditions",
        "fit_rows": int(len(fit_rows)),
        "fit_rows_sha256": ordered_hash(fit_rows),
        "n_top_genes": int(args.n_hvg),
        "flavor": "seurat",
        "input_assumed": "existing scanpy-style log1p-normalized X",
        "second_normalization_applied": False,
        "index_order": "h5ad source gene order",
        "gene_indices_sha256": sha_file(output / "gene_indices.npy"),
        "gene_order_sha256": ordered_hash(genes),
        "scanpy_version": scanpy_version,
        "anndata_version": anndata_version,
        "numpy_version": np.__version__,
    }
    write_json(hvg_path, hvg_provenance)

    extract = load_extract()
    import torch
    gene_embeddings_path = Path(args.gene_embeddings).expanduser().resolve()
    embeddings = torch.load(str(gene_embeddings_path), map_location="cpu", weights_only=False)
    if not isinstance(embeddings, dict):
        raise TypeError("gene embeddings must be an ESM2 symbol-to-vector dictionary")
    symbols = list(embeddings.keys())
    lookup = {symbol: i for i, symbol in enumerate(symbols)}
    mapping = np.asarray([lookup.get(symbol, -1) for symbol in metadata["source_genes"]], dtype=np.int32)
    del embeddings
    cache = output / "token_cache"
    cache.mkdir(parents=True, exist_ok=True)
    token_ids_path, token_values_path = cache / "token_ids.npy", cache / "token_counts.npy"
    n_cells = int(metadata["shape"][0])
    tokenizer = extract.Tokenizer(mapping, len(symbols), seed=int(args.seed), pad_length=2048)
    ids = np.lib.format.open_memmap(token_ids_path, mode="w+", dtype="int32", shape=(n_cells, 2048))
    values = np.lib.format.open_memmap(token_values_path, mode="w+", dtype="float32", shape=(n_cells, 2048))
    started = time.time()
    with h5py.File(h5ad, "r") as handle:
        data, indices, indptr = handle["X/data"], handle["X/indices"], handle["X/indptr"]
        for row in range(n_cells):
            begin, end = int(indptr[row]), int(indptr[row + 1])
            # X is log1p-normalized; expm1 only recovers its normalized linear
            # representation so extract_se.Tokenizer re-applies log1p once.
            linear_values = np.expm1(np.asarray(data[begin:end], dtype=np.float64))
            ids[row], values[row] = tokenizer.encode(indices[begin:end], linear_values)
            if (row + 1) % int(args.log_every) == 0:
                print(f"tokenized {row + 1}/{n_cells}", flush=True)
    ids.flush(); values.flush()
    if not np.isfinite(values).all():
        raise ValueError("tokenized expression contains NaN/Inf")
    token_meta = {
        "format_version": 2, "complete": True,
        "h5ad": str(h5ad), "h5ad_sha256": split["h5ad_sha256"],
        "cell_order": "h5ad/obs/_index", "cell_order_sha256": split["cell_order_sha256"],
        "n_cells": n_cells, "n_source_genes": int(metadata["shape"][1]),
        "n_mapped_source_genes": int((mapping >= 0).sum()), "n_embedding_tokens": len(symbols),
        "seed": int(args.seed), "pad_length": 2048,
        "transform": "100 * log1p(expm1(X_log1p_normalized)) / sum(log1p(expm1(X_log1p_normalized))); top2047 + CLS",
        "counts_source": "h5ad/X log1p-normalized; expm1 is an inversion to normalized linear values, not raw UMI counts",
        "token_ids_sha256": sha_file(token_ids_path), "token_counts_sha256": sha_file(token_values_path),
        "esm_embeddings": str(gene_embeddings_path),
        "esm_embeddings_sha256": sha_file(gene_embeddings_path),
        "esm_gene_order_sha256": ordered_hash(symbols),
        "seconds": time.time() - started,
    }
    # Mark the split and HVG artifacts complete only after tokenization has
    # finished and its hashes have been written.  This leaves interrupted
    # preparation runs resumable while completed outputs remain immutable.
    split["complete"] = True
    hvg_provenance["complete"] = True
    write_json(split_path, split)
    write_json(hvg_path, hvg_provenance)
    write_json(tokenization_path, token_meta)
    print(json.dumps({
        "n_cells": n_cells,
        "n_source_genes": int(metadata["shape"][1]),
        "train_conditions": len(split["train_conditions"]),
        "test_conditions": len(split["test_conditions"]),
        "train_controls": len(split["train_control_rows"]),
        "test_controls": len(split["test_control_rows"]),
        "train_treated": len(split["train_treated_rows"]),
        "test_treated": len(split["test_treated_rows"]),
        "n_hvg": len(genes),
        "token_ids_sha256": token_meta["token_ids_sha256"],
        "token_counts_sha256": token_meta["token_counts_sha256"],
    }, indent=2), flush=True)


def encode(args) -> None:
    extract = load_extract()
    import torch
    if int(args.num_shards) < 1:
        raise ValueError("num_shards must be >= 1")
    if not 0 <= int(args.shard_index) < int(args.num_shards):
        raise ValueError("shard_index must satisfy 0 <= shard_index < num_shards")
    if int(args.batch_size) < 1:
        raise ValueError("batch_size must be >= 1")
    if int(args.log_every) < 1:
        raise ValueError("log_every must be >= 1")
    input_dir = Path(args.input_dir).expanduser().resolve()
    output = Path(args.output_dir or input_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    token_meta_path = input_dir / "tokenization.json"
    if not token_meta_path.exists():
        raise FileNotFoundError(f"missing tokenization metadata: {token_meta_path}")
    token_meta = json.loads(token_meta_path.read_text())
    if not token_meta.get("complete"):
        raise ValueError("tokenization is not marked complete")
    ids = np.load(input_dir / "token_cache/token_ids.npy", mmap_mode="r")
    values = np.load(input_dir / "token_cache/token_counts.npy", mmap_mode="r")
    if ids.shape != values.shape or ids.ndim != 2 or ids.shape[1] != 2048:
        raise ValueError(f"invalid token cache shapes: ids={ids.shape}, values={values.shape}")
    n_cells = int(ids.shape[0])
    first = n_cells * int(args.shard_index) // int(args.num_shards)
    last = n_cells * (int(args.shard_index) + 1) // int(args.num_shards)
    if first == last:
        raise ValueError("empty shard")
    stem = f"state_shard_{args.shard_index:02d}_of_{args.num_shards:02d}"
    shard_path, meta_path = output / f"{stem}.npy", output / f"{stem}.json"
    refuse_completed(meta_path, f"SE state metadata {stem}")
    device = torch.device(args.device)
    torch.set_num_threads(1)
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))
    model = extract.load_model(str(Path(args.checkpoint).expanduser().resolve()), args.se_config, device)
    states = np.lib.format.open_memmap(shard_path, mode="w+", dtype="float32", shape=(last - first, 2048))
    dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16
    started = time.time()
    progress_every = int(args.log_every)
    next_progress = first + progress_every
    for begin in range(first, last, int(args.batch_size)):
        end = min(begin + int(args.batch_size), last)
        result = extract.infer(model, ids[begin:end], values[begin:end], device, dtype)
        if result.shape != (end - begin, 2048) or not np.isfinite(result).all():
            raise ValueError(f"invalid SE state at rows {begin}:{end}")
        states[begin - first:end - first] = result
        if end >= next_progress or end == last:
            states.flush()
            print(f"encoded {end}/{n_cells} rows ({stem})", flush=True)
            while next_progress <= end:
                next_progress += progress_every
    record = {
        "format_version": 2, "complete": True,
        "first_row": first, "last_row": last, "n_cells": n_cells,
        "shape": [last - first, 2048], "dtype": "float32",
        "cell_order_sha256": token_meta["cell_order_sha256"],
        "token_ids_sha256": token_meta["token_ids_sha256"],
        "token_counts_sha256": token_meta["token_counts_sha256"],
        "checkpoint": str(Path(args.checkpoint).expanduser().resolve()),
        "checkpoint_sha256": sha_file(Path(args.checkpoint).expanduser().resolve()),
        "amp_dtype": args.amp_dtype, "device": str(device), "seed": int(args.seed),
        "seconds": time.time() - started,
    }
    write_json(meta_path, record)


def merge(args) -> None:
    if int(args.num_shards) < 1:
        raise ValueError("num_shards must be >= 1")
    input_dir = Path(args.input_dir).expanduser().resolve()
    output = Path(args.output_dir or input_dir).expanduser().resolve()
    token_meta = json.loads((input_dir / "tokenization.json").read_text())
    if not token_meta.get("complete"):
        raise ValueError("tokenization is not marked complete")
    n_cells = int(token_meta["n_cells"])
    destination = output / "se_embeddings.npy"
    audit_path = output / "state_embeddings.json"
    refuse_completed(audit_path, "merged SE state metadata")
    partial = output / "se_embeddings.partial.npy"
    merged = np.lib.format.open_memmap(partial, mode="w+", dtype="float32", shape=(n_cells, 2048))
    records = []
    for shard_index in range(int(args.num_shards)):
        first = n_cells * shard_index // int(args.num_shards)
        last = n_cells * (shard_index + 1) // int(args.num_shards)
        stem = f"state_shard_{shard_index:02d}_of_{args.num_shards:02d}"
        record = json.loads((input_dir / f"{stem}.json").read_text())
        values = np.load(input_dir / f"{stem}.npy", mmap_mode="r")
        if not record.get("complete") or record["first_row"] != first or record["last_row"] != last:
            raise ValueError(f"incomplete/misaligned {stem}")
        if record["cell_order_sha256"] != token_meta["cell_order_sha256"] or values.shape != (last - first, 2048):
            raise ValueError(f"metadata/shape mismatch in {stem}")
        if not np.isfinite(values).all():
            raise ValueError(f"non-finite values in {stem}")
        merged[first:last] = values
        records.append(record)
    protocol_keys = ("token_ids_sha256", "token_counts_sha256", "checkpoint_sha256", "amp_dtype")
    for key in protocol_keys:
        if any(record[key] != records[0][key] for record in records[1:]):
            raise ValueError(f"shards disagree on {key}")
    if any(record.get("n_cells") != n_cells for record in records):
        raise ValueError("shard n_cells disagree with token metadata")
    merged.flush()
    norms = np.linalg.norm(merged, axis=1)
    if not np.isfinite(norms).all() or float(np.max(np.abs(norms - 1.0))) > 0.02:
        raise ValueError("merged SE states must be finite unit-normalized vectors (max |norm-1| <= 0.02)")
    del merged
    os.replace(partial, destination)
    audit = {
        "format_version": 2, "complete": True, "n_cells": n_cells,
        "shape": [n_cells, 2048], "dtype": "float32", "finite": True,
        "cell_order": token_meta["cell_order"],
        "cell_order_sha256": token_meta["cell_order_sha256"], "state_sha256": sha_file(destination),
        "token_ids_sha256": records[0]["token_ids_sha256"],
        "token_counts_sha256": records[0]["token_counts_sha256"],
        "checkpoint_sha256": records[0]["checkpoint_sha256"],
        "amp_dtype": records[0]["amp_dtype"],
        "norm_min": float(norms.min()), "norm_max": float(norms.max()),
        "tokenization": token_meta, "shards": records,
        "h5ad_sha256": token_meta["h5ad_sha256"],
    }
    write_json(audit_path, audit)
    print(json.dumps({k: v for k, v in audit.items() if k not in {"tokenization", "shards"}}, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--h5ad", required=True)
    prep.add_argument("--condition-manifest", required=True)
    prep.add_argument("--gene-embeddings", required=True)
    prep.add_argument("--output-dir", required=True)
    prep.add_argument("--seed", type=int, default=42)
    prep.add_argument("--control-train-fraction", type=float, default=0.75)
    prep.add_argument("--n-hvg", type=int, default=2000)
    prep.add_argument("--log-every", type=int, default=10000)
    enc = sub.add_parser("encode")
    enc.add_argument("--input-dir", required=True)
    enc.add_argument("--output-dir")
    enc.add_argument("--checkpoint", required=True)
    enc.add_argument("--se-config")
    enc.add_argument("--device", default="cuda:0")
    enc.add_argument("--batch-size", type=int, default=8)
    enc.add_argument("--num-shards", type=int, default=8)
    enc.add_argument("--shard-index", type=int, required=True)
    enc.add_argument("--seed", type=int, default=42)
    enc.add_argument("--log-every", type=int, default=200)
    enc.add_argument("--amp-dtype", choices=["bf16", "fp16"], default="bf16")
    merge_parser = sub.add_parser("merge")
    merge_parser.add_argument("--input-dir", required=True)
    merge_parser.add_argument("--output-dir")
    merge_parser.add_argument("--num-shards", type=int, default=8)
    args = parser.parse_args()
    {"prepare": prepare, "encode": encode, "merge": merge}[args.command](args)


if __name__ == "__main__":
    main()
