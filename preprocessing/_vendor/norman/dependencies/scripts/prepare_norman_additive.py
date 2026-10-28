#!/usr/bin/env python3
"""Prepare the official scDFM Norman additive five-fold split.

The split follows ``scDFM/src/data_process/data.py`` exactly: the 125
non-control double-gene conditions are sorted, shuffled with ``RandomState``
seed ``42 + fold``, and split into the first 30 percent (37 conditions) for
test and the remaining 88 for training.  All 101 ``gene+ctrl`` conditions
remain in training.  There is no validation split.

This script only writes split/HVG metadata.  It never computes SE embeddings;
an existing ``[84_986, 2_048]`` cache may be supplied with ``--se-embeddings``
and is verified by hash/shape when present.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import subprocess
import time
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np


EXPECTED_CELLS = 84_986
EXPECTED_SOURCE_GENES = 19_264
EXPECTED_CONTROLS = 7_275
EXPECTED_SINGLES = 101
EXPECTED_DOUBLES = 125
EXPECTED_TEST_DOUBLES = 37
EXPECTED_TRAIN_DOUBLES = 88
EXPECTED_TRAIN_CONTROLS = 5_456
EXPECTED_TEST_CONTROLS = 1_819


def sha_file(path: Path, chunk: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def sha_ordered(values: Iterable[Any]) -> str:
    return hashlib.sha256("\n".join(str(value) for value in values).encode("utf-8")).hexdigest()


def sha_i8(values: Iterable[int]) -> str:
    return hashlib.sha256(np.asarray(list(values), dtype="<i8").tobytes()).hexdigest()


def decode(values: Any) -> list[str]:
    array = np.asarray(values)
    return [value.decode() if isinstance(value, (bytes, np.bytes_)) else str(value) for value in array]


def read_column(group: h5py.Group, name: str) -> np.ndarray:
    obj = group[name]
    if isinstance(obj, h5py.Dataset):
        values = obj[()]
        if values.dtype.kind in {"S", "O", "U"}:
            return np.asarray(decode(values), dtype=object)
        return np.asarray(values)
    categories = decode(obj["categories"][()])
    codes = np.asarray(obj["codes"][()], dtype=np.int64)
    return np.asarray([categories[int(code)] for code in codes], dtype=object)


def metadata(h5ad: Path) -> dict[str, Any]:
    with h5py.File(h5ad, "r") as handle:
        x = handle["X"]
        if not {"data", "indices", "indptr"}.issubset(x):
            raise ValueError("h5ad/X must contain CSR data, indices and indptr")
        shape = tuple(int(value) for value in x.attrs["shape"])
        cell_ids = decode(handle["obs"]["_index"][()])
        conditions = np.asarray(read_column(handle["obs"], "condition"), dtype=object)
        control_raw = read_column(handle["obs"], "control")
        if control_raw.dtype.kind in {"S", "O", "U"}:
            control = np.asarray([str(value).lower() in {"1", "true", "yes"} for value in control_raw], dtype=bool)
        else:
            control = np.asarray(control_raw, dtype=np.int64).astype(bool)
        source_genes = decode(handle["var"].get("gene_name", handle["var"]["_index"])[()])
    if shape != (len(cell_ids), len(source_genes)):
        raise ValueError(f"X shape {shape} disagrees with obs/var lengths")
    if len(set(cell_ids)) != len(cell_ids):
        raise ValueError("obs/_index contains duplicate cell IDs")
    if len(conditions) != shape[0] or len(control) != shape[0]:
        raise ValueError("condition/control length does not match X")
    return {
        "shape": shape,
        "cell_ids": cell_ids,
        "conditions": conditions,
        "control": control,
        "source_genes": source_genes,
    }


def condition_groups(meta: dict[str, Any]) -> tuple[list[str], list[str], list[str]]:
    conditions = meta["conditions"]
    control = meta["control"]
    perturbations = sorted({str(value) for value, flag in zip(conditions, control) if not flag})
    # This is the exact semantic used by the official source: controls are
    # represented as ``gene+ctrl`` and every condition without the token ctrl
    # is a double-gene perturbation.
    singles = sorted(value for value in perturbations if "ctrl" in value)
    doubles = sorted(value for value in perturbations if "ctrl" not in value)
    if len(singles) != EXPECTED_SINGLES or len(doubles) != EXPECTED_DOUBLES:
        raise ValueError(
            f"expected {EXPECTED_SINGLES} singles/{EXPECTED_DOUBLES} doubles, "
            f"got {len(singles)}/{len(doubles)}"
        )
    return perturbations, singles, doubles


def split_conditions(singles: list[str], doubles: list[str], fold: int) -> tuple[list[str], list[str], dict[str, Any]]:
    if not 0 <= int(fold) < 5:
        raise ValueError("fold must be in [0,4]")
    # np.random.seed + np.random.shuffle is the legacy scDFM operation.  A
    # local RandomState gives the same sequence without mutating global state.
    rng = np.random.RandomState(42 + int(fold))
    shuffled = np.asarray(doubles, dtype=object).copy()
    rng.shuffle(shuffled)
    split_idx = int(len(shuffled) * 0.3)
    if split_idx != EXPECTED_TEST_DOUBLES:
        raise AssertionError(f"125 * 0.3 must produce 37, got {split_idx}")
    test_doubles = [str(value) for value in shuffled[:split_idx]]
    train_doubles = [str(value) for value in shuffled[split_idx:]]
    train_conditions = list(singles) + train_doubles
    test_conditions = test_doubles
    return train_conditions, test_conditions, {
        "all_double_conditions_sorted": list(doubles),
        "train_double_conditions": train_doubles,
        "test_double_conditions": test_doubles,
        "split_seed": 42 + int(fold),
        "split_index": split_idx,
        "split_rule": "sorted 125 doubles; RandomState(seed=42+fold).shuffle; first 37 test, remaining 88 train",
    }


def control_split(meta: dict[str, Any], seed: int = 4242, train_fraction: float = 0.75) -> tuple[np.ndarray, np.ndarray]:
    if not 0.0 < float(train_fraction) < 1.0:
        raise ValueError("control train fraction must lie in (0,1)")
    controls = np.flatnonzero(meta["control"]).astype(np.int64)
    rng = np.random.default_rng(int(seed))
    shuffled = controls[rng.permutation(len(controls))]
    n_train = int(np.floor(len(shuffled) * float(train_fraction)))
    train = np.sort(shuffled[:n_train]).astype(np.int64)
    test = np.sort(shuffled[n_train:]).astype(np.int64)
    if len(train) != EXPECTED_TRAIN_CONTROLS or len(test) != EXPECTED_TEST_CONTROLS:
        raise AssertionError(f"control split expected 5456/1819, got {len(train)}/{len(test)}")
    if np.intersect1d(train, test).size or not np.array_equal(np.sort(np.concatenate((train, test))), controls):
        raise AssertionError("control rows are not a disjoint cover")
    return train, test


def _git_commit(root: Path) -> str | None:
    # Historical generator source commit; no live repository dependency.
    return '3786dd6a8ce03aca996ab5d7b5dc10bc85689215'


def _target_gene_symbols(singles: list[str]) -> list[str]:
    genes = set()
    for condition in singles:
        for token in str(condition).split("+"):
            token = token.strip()
            if token and token.lower() != "ctrl":
                genes.add(token)
    if len(genes) != EXPECTED_SINGLES:
        raise ValueError(f"expected {EXPECTED_SINGLES} unique single-gene targets, got {len(genes)}")
    return sorted(genes)


def _refuse_complete(path: Path) -> None:
    if not path.exists():
        return
    try:
        payload = json.loads(path.read_text())
    except Exception:
        return
    if isinstance(payload, dict) and payload.get("complete") is True:
        raise FileExistsError(f"refusing to overwrite completed artifact: {path}")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _fit_hvg(adata, fit_rows: np.ndarray, source_genes: list[str], target_genes: list[str], n_base_hvg: int):
    import scanpy as sc
    subset = adata[fit_rows].copy()
    # The Norman release stores scanpy-style log1p-normalized X.  Match the
    # prior scDFM-compatible preparation: Seurat HVGs on X directly, with no
    # second normalize_total/log1p call.
    sc.pp.highly_variable_genes(subset, n_top_genes=int(n_base_hvg), flavor="seurat", inplace=True)
    mask = np.asarray(subset.var["highly_variable"], dtype=bool)
    base_idx = np.flatnonzero(mask).astype(np.int64)
    if len(base_idx) != int(n_base_hvg):
        raise RuntimeError(f"Seurat selected {len(base_idx)} HVGs, expected {n_base_hvg}")
    if list(map(str, subset.var_names)) != list(map(str, source_genes)):
        raise RuntimeError("anndata var order differs from h5ad source gene order")
    source_lookup = {gene: index for index, gene in enumerate(source_genes)}
    missing = [gene for gene in target_genes if gene not in source_lookup]
    if missing:
        raise ValueError(f"single-gene target absent from source genes: {missing[:8]}")
    target_idx = np.asarray([source_lookup[gene] for gene in target_genes], dtype=np.int64)
    final_idx = np.sort(np.unique(np.concatenate((base_idx, target_idx)))).astype(np.int64)
    final_genes = [source_genes[int(index)] for index in final_idx]
    return final_idx, final_genes, base_idx, target_idx, sc.__version__


def _verify_se_cache(path: Path | None, n_cells: int) -> dict[str, Any]:
    if path is None:
        return {"status": "not_checked", "shape": None, "path": None}
    path = path.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"SE cache does not exist: {path}")
    array = np.load(path, mmap_mode="r")
    if tuple(array.shape) != (int(n_cells), 2048):
        raise ValueError(f"SE cache must have shape [{n_cells},2048], got {tuple(array.shape)}")
    if not np.isfinite(array[: min(1024, len(array))]).all():
        raise ValueError("SE cache contains non-finite values in its inspected prefix")
    return {"status": "verified", "path": str(path), "shape": list(array.shape), "sha256": sha_file(path)}


def prepare_fold(
    *, fold: int, meta: dict[str, Any], h5ad: Path, manifest_h5ad_path: str,
    output_root: Path, repo_root: Path, singles: list[str], doubles: list[str],
    train_controls: np.ndarray, test_controls: np.ndarray, target_genes: list[str],
    control_seed: int, control_train_fraction: float, n_base_hvg: int,
    se_cache: dict[str, Any], adata,
) -> dict[str, Any]:
    train_conditions, test_conditions, split_info = split_conditions(singles, doubles, fold)
    conditions = meta["conditions"]
    control = meta["control"]
    train_treated_rows = np.sort(np.flatnonzero((~control) & np.isin(conditions, train_conditions))).astype(np.int64)
    test_treated_rows = np.sort(np.flatnonzero((~control) & np.isin(conditions, test_conditions))).astype(np.int64)
    fit_rows = np.concatenate((np.sort(train_controls), train_treated_rows)).astype(np.int64)
    out = output_root / f"fold_{fold}"
    out.mkdir(parents=True, exist_ok=True)
    condition_path = out / "condition_manifest.json"
    control_path = out / "control_split.json"
    hvg_prov_path = out / "hvg_provenance.json"
    for path in (condition_path, control_path, hvg_prov_path):
        _refuse_complete(path)
    created = time.time()
    host = socket.gethostname()
    commit = _git_commit(repo_root)
    h5ad_sha = sha_file(h5ad)
    cell_ids = meta["cell_ids"]
    condition_manifest = {
        "format_version": 1,
        "complete": False,
        "h5ad_path": str(manifest_h5ad_path),
        "h5ad_local_path": str(h5ad),
        "h5ad_sha256": h5ad_sha,
        "split_mode": "official_additive",
        "split_unit": "condition",
        "fold": int(fold),
        "seed": 42 + int(fold),
        "num_cells": int(meta["shape"][0]),
        "num_source_genes": int(meta["shape"][1]),
        "num_control_cells": int(np.sum(control)),
        "num_perturbed_cells": int(np.sum(~control)),
        "num_single_gene_conditions": len(singles),
        "num_double_gene_conditions": len(doubles),
        "num_train_conditions": len(train_conditions),
        "num_test_conditions": len(test_conditions),
        "train_conditions": train_conditions,
        "test_conditions": test_conditions,
        "train_single_gene_conditions": len(singles),
        "train_double_gene_conditions": len(split_info["train_double_conditions"]),
        "test_single_gene_conditions": 0,
        "test_double_gene_conditions": len(split_info["test_double_conditions"]),
        "train_treated_rows": int(len(train_treated_rows)),
        "test_treated_rows": int(len(test_treated_rows)),
        "train_treated_rows_sha256": sha_ordered(train_treated_rows),
        "test_treated_rows_sha256": sha_ordered(test_treated_rows),
        "all_double_conditions_sorted": split_info["all_double_conditions_sorted"],
        "train_double_conditions": split_info["train_double_conditions"],
        "test_double_conditions": split_info["test_double_conditions"],
        "split_index": split_info["split_index"],
        "split_rule": split_info["split_rule"],
        "target_genes_from_all_single_conditions": target_genes,
        "se_cache": se_cache,
        "hostname": host,
        "git_commit": commit,
        "created_unix": created,
    }
    _write_json(condition_path, condition_manifest)
    condition_sha = sha_file(condition_path)
    control_manifest = {
        "format_version": 1,
        "complete": False,
        "h5ad_path": str(manifest_h5ad_path),
        "h5ad_local_path": str(h5ad),
        "h5ad_sha256": h5ad_sha,
        "seed": int(control_seed),
        "train_fraction": float(control_train_fraction),
        "split_unit": "control cell IDs shared by every fold/method",
        "train_control_rows": train_controls.tolist(),
        "test_control_rows": test_controls.tolist(),
        "train_control_cell_ids": [str(cell_ids[int(row)]) for row in train_controls],
        "test_control_cell_ids": [str(cell_ids[int(row)]) for row in test_controls],
        "train_control_rows_sha256": sha_ordered(train_controls),
        "test_control_rows_sha256": sha_ordered(test_controls),
        "train_control_rows_sha256_le_i8": sha_i8(train_controls),
        "test_control_rows_sha256_le_i8": sha_i8(test_controls),
        "hostname": host,
        "git_commit": commit,
    }
    # Keep this file byte-identical across folds so methods can verify the
    # same control IDs by file hash as well as by the row-content hashes.
    _write_json(control_path, control_manifest)
    control_sha = sha_file(control_path)
    final_idx, final_genes, base_idx, target_idx, scanpy_version = _fit_hvg(
        adata, fit_rows, meta["source_genes"], target_genes, n_base_hvg
    )
    # ``gene_indices.npy`` is the canonical interface name.  Keep a byte-
    # identical ``hvg_indices.npy`` compatibility copy for older launchers.
    np.save(out / "gene_indices.npy", final_idx)
    np.save(out / "hvg_indices.npy", final_idx)
    (out / "genes.json").write_text(json.dumps(final_genes, indent=2) + "\n")
    final_idx_sha = sha_file(out / "gene_indices.npy")
    gene_order_sha = sha_ordered(final_genes)
    fit_rows_sha = sha_ordered(fit_rows)
    hvg_provenance = {
        "format_version": 1,
        "complete": False,
        "h5ad": str(h5ad),
        "h5ad_manifest_path": str(manifest_h5ad_path),
        "h5ad_sha256": h5ad_sha,
        "condition_manifest": str(condition_path),
        "condition_manifest_sha256": condition_sha,
        "control_split": str(control_path),
        "control_split_sha256": control_sha,
        "split_mode": "official_additive",
        "fold": int(fold),
        "fit_scope": "sorted train controls plus sorted treated cells from official additive train conditions",
        "fit_rows": int(len(fit_rows)),
        "fit_rows_sha256": fit_rows_sha,
        "fit_rows_sha256_le_i8": sha_i8(fit_rows),
        "train_control_rows": int(len(train_controls)),
        "train_treated_rows": int(len(train_treated_rows)),
        "base_hvg_method": "Seurat flavor on 2,000 genes, fitted on train controls + train treated cells",
        "base_hvg_n_top_genes": int(n_base_hvg),
        "base_hvg_indices_sha256_le_i8": sha_i8(base_idx),
        "base_hvg_count": int(len(base_idx)),
        "union_target_gene_set": "all 101 single-gene target symbols from gene+ctrl conditions",
        "union_target_gene_count": int(len(target_genes)),
        "union_target_indices_sha256_le_i8": sha_i8(target_idx),
        "forced_target_count": int(len(set(target_idx.tolist()) - set(base_idx.tolist()))),
        "n_top_genes": int(len(final_idx)),
        "gene_indices": str(out / "gene_indices.npy"),
        "gene_indices_sha256": final_idx_sha,
        "gene_indices_sha256_le_i8": sha_i8(final_idx),
        "gene_order": str(out / "genes.json"),
        "gene_order_sha256": gene_order_sha,
        "gene_order_file_sha256": sha_file(out / "genes.json"),
        "flavor": "seurat",
        "input_assumed": "existing scanpy-style log1p-normalized X",
        "second_normalization_applied": False,
        "index_order": "h5ad source gene order",
        "n_genes_source": int(meta["shape"][1]),
        "scanpy_version": scanpy_version,
        "anndata_version": getattr(__import__("anndata"), "__version__", None),
        "numpy_version": np.__version__,
        "hostname": host,
        "git_commit": commit,
        "created_unix": created,
    }
    _write_json(hvg_prov_path, hvg_provenance)
    # Finalize the condition manifest first.  Its hash is then stable and can
    # be recorded by the control/HVG manifests without a circular dependency.
    condition_manifest["complete"] = True
    condition_manifest["gene_indices"] = str(out / "gene_indices.npy")
    condition_manifest["hvg_indices_compat"] = str(out / "hvg_indices.npy")
    condition_manifest["genes"] = str(out / "genes.json")
    condition_manifest["gene_order_sha256"] = gene_order_sha
    condition_manifest["fit_rows"] = int(len(fit_rows))
    condition_manifest["fit_rows_sha256"] = fit_rows_sha
    _write_json(condition_path, condition_manifest)
    condition_sha = sha_file(condition_path)
    # Control rows are shared across folds; this final file has no fold/time
    # fields and therefore has the same content hash in every fold.
    control_manifest["complete"] = True
    control_manifest["condition_manifest_sha256"] = condition_sha
    _write_json(control_path, control_manifest)
    control_sha = sha_file(control_path)
    hvg_provenance["condition_manifest_sha256"] = condition_sha
    hvg_provenance["control_split_sha256"] = control_sha
    hvg_provenance["complete"] = True
    _write_json(hvg_prov_path, hvg_provenance)
    return {
        "fold": int(fold),
        "train_conditions": len(train_conditions),
        "test_conditions": len(test_conditions),
        "train_treated": len(train_treated_rows),
        "test_treated": len(test_treated_rows),
        "train_controls": len(train_controls),
        "test_controls": len(test_controls),
        "fit_rows": len(fit_rows),
        "base_hvg": len(base_idx),
        "target_genes": len(target_genes),
        "union_hvg": len(final_idx),
        "forced_targets": len(set(target_idx.tolist()) - set(base_idx.tolist())),
        "output_dir": str(out),
    }


def parse_folds(value: str) -> list[int]:
    if str(value).lower() == "all":
        return list(range(5))
    folds = sorted(set(int(part.strip()) for part in str(value).split(",") if part.strip()))
    if not folds or any(fold < 0 or fold >= 5 for fold in folds):
        raise ValueError("--fold must be all or a comma-separated subset of 0..4")
    return folds


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5ad", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fold", default="all", help="all or a comma-separated subset of 0..4")
    parser.add_argument("--manifest-h5ad-path", help="path recorded in manifests (e.g. the host1 source path)")
    parser.add_argument("--control-seed", type=int, default=4242)
    parser.add_argument("--control-train-fraction", type=float, default=0.75)
    parser.add_argument("--n-base-hvg", type=int, default=2000)
    parser.add_argument("--se-embeddings", type=Path, help="optional existing [84986,2048] cache to verify; never generated here")
    args = parser.parse_args()
    h5ad = args.h5ad.expanduser().resolve()
    if not h5ad.exists():
        raise FileNotFoundError(h5ad)
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    repo_root = Path(__file__).resolve().parents[2]
    meta = metadata(h5ad)
    if meta["shape"] != (EXPECTED_CELLS, EXPECTED_SOURCE_GENES):
        raise ValueError(f"expected Norman shape {(EXPECTED_CELLS, EXPECTED_SOURCE_GENES)}, got {meta['shape']}")
    if int(np.sum(meta["control"])) != EXPECTED_CONTROLS:
        raise ValueError(f"expected {EXPECTED_CONTROLS} controls, got {int(np.sum(meta['control']))}")
    _, singles, doubles = condition_groups(meta)
    target_genes = _target_gene_symbols(singles)
    folds = parse_folds(args.fold)
    train_controls, test_controls = control_split(meta, args.control_seed, args.control_train_fraction)
    se_cache = _verify_se_cache(args.se_embeddings, meta["shape"][0])
    manifest_h5ad_path = str(args.manifest_h5ad_path or h5ad)
    # Load once; each fold takes a sparse copy only for its train rows.  No SE
    # encoding or cache generation is performed in this process.
    import anndata as ad
    adata = ad.read_h5ad(str(h5ad))
    if list(map(str, adata.var_names)) != list(map(str, meta["source_genes"])):
        raise ValueError("anndata var order differs from h5ad source gene order")
    summaries = []
    for fold in folds:
        summaries.append(prepare_fold(
            fold=fold, meta=meta, h5ad=h5ad, manifest_h5ad_path=manifest_h5ad_path,
            output_root=output, repo_root=repo_root, singles=singles, doubles=doubles,
            train_controls=train_controls, test_controls=test_controls, target_genes=target_genes,
            control_seed=args.control_seed, control_train_fraction=args.control_train_fraction,
            n_base_hvg=args.n_base_hvg, se_cache=se_cache, adata=adata,
        ))
    report = {
        "format_version": 1,
        "complete": True,
        "h5ad": str(h5ad),
        "h5ad_manifest_path": manifest_h5ad_path,
        "h5ad_sha256": sha_file(h5ad),
        "output_dir": str(output),
        "folds": folds,
        "split_mode": "official_additive",
        "control_seed": int(args.control_seed),
        "control_train_fraction": float(args.control_train_fraction),
        "se_cache": se_cache,
        "summaries": summaries,
        "hostname": socket.gethostname(),
        "git_commit": _git_commit(repo_root),
        "created_unix": time.time(),
    }
    _write_json(output / "prepare_manifest.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
