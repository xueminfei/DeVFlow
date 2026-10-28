"""Portable, pickle-free prediction archives shared by all task adapters.

IDs describe the exact sampled rows, including intentional repeated samples.
They are never sorted, deduplicated, or matched approximately by the evaluator.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Mapping, Sequence

import numpy as np


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def gene_order_hash(genes: Sequence[str]) -> str:
    """Hash UTF-8 symbols joined by newline, without a trailing newline."""
    return hashlib.sha256("\n".join(map(str, genes)).encode("utf-8")).hexdigest()


def json_sha256(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                                     allow_nan=False).encode("utf-8")).hexdigest()


def write_json(path: str | Path, value: Mapping) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    tmp.replace(path)


@dataclass(frozen=True)
class PredictionArchive:
    seed: int
    condition_names: tuple[str, ...]
    source_global_ids: np.ndarray
    target_global_ids: np.ndarray
    gene_order_sha256: str
    predictions: tuple[np.ndarray, ...]
    metadata: dict

    @property
    def n_conditions(self) -> int:
        return len(self.condition_names)

    @property
    def n_genes(self) -> int:
        return self.predictions[0].shape[1]

    @property
    def cells_per_condition(self) -> int:
        return self.source_global_ids.shape[1]


def _validate(archive: PredictionArchive) -> None:
    names = archive.condition_names
    if not names or len(set(names)) != len(names) or any(not x for x in names):
        raise ValueError("condition_names must be nonempty and unique")
    if not re.fullmatch(r"[a-f0-9]{64}", archive.gene_order_sha256):
        raise ValueError("gene_order_sha256 must be a lowercase SHA256 digest")
    source, target = archive.source_global_ids, archive.target_global_ids
    if source.ndim != 2 or source.shape[0] != len(names) or source.shape[1] < 2:
        raise ValueError("source_global_ids must have shape [conditions, cells>=2]")
    if target.shape != source.shape:
        raise ValueError("target_global_ids must match source_global_ids shape")
    for ids in (source, target):
        if ids.dtype.kind not in "iuUS":
            raise ValueError("row IDs must be integer or Unicode/string arrays; object IDs are forbidden")
    if len(archive.predictions) != len(names):
        raise ValueError("one prediction matrix is required per condition")
    n_genes = None
    for value in archive.predictions:
        if value.ndim != 2 or value.shape[0] != source.shape[1] or value.shape[1] < 2:
            raise ValueError("prediction matrices must have shape [sampled cells, genes>=2]")
        if value.dtype.kind != "f" or not np.isfinite(value).all():
            raise ValueError("predictions must contain finite floating point values")
        n_genes = value.shape[1] if n_genes is None else n_genes
        if value.shape[1] != n_genes:
            raise ValueError("all prediction matrices must share the same gene axis")


def write_prediction_archive(
    path: str | Path, *, seed: int, condition_names: Sequence[str],
    source_global_ids: np.ndarray, target_global_ids: np.ndarray,
    gene_order_sha256: str, predictions: Sequence[np.ndarray], metadata: Mapping | None = None,
) -> None:
    """Write schema v1; preserve prediction precision and sampled row ordering."""
    archive = PredictionArchive(int(seed), tuple(map(str, condition_names)),
                                np.asarray(source_global_ids), np.asarray(target_global_ids),
                                str(gene_order_sha256), tuple(np.asarray(x) for x in predictions),
                                dict(metadata or {}))
    _validate(archive)
    payload = dict(schema_version=np.asarray(1, dtype=np.int64), seed=np.asarray(archive.seed),
                   group_indices=np.arange(archive.n_conditions, dtype=np.int64),
                   condition_names=np.asarray(archive.condition_names, dtype=str),
                   source_global_ids=archive.source_global_ids,
                   target_global_ids=archive.target_global_ids,
                   gene_order_sha256=np.asarray(archive.gene_order_sha256),
                   metadata_json=np.asarray(json.dumps(archive.metadata, sort_keys=True, allow_nan=False)))
    payload.update({f"prediction_{i:04d}": value for i, value in enumerate(archive.predictions)})
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **payload)
    temporary.replace(path)


def read_prediction_archive(path: str | Path) -> PredictionArchive:
    with np.load(path, allow_pickle=False) as values:
        version = int(values["schema_version"]) if "schema_version" in values else 1
        if version != 1:
            raise ValueError(f"Unsupported prediction archive schema: {version}")
        names = tuple(map(str, values["condition_names"].tolist()))
        if not np.array_equal(values["group_indices"], np.arange(len(names))):
            raise ValueError("group_indices must follow the full condition_names order")
        keys = {f"prediction_{i:04d}" for i in range(len(names))}
        present = {k for k in values.files if k.startswith("prediction_")}
        if present != keys:
            raise ValueError("Missing or unexpected prediction matrices")
        metadata = json.loads(str(values["metadata_json"])) if "metadata_json" in values else {}
        if not isinstance(metadata, dict):
            raise ValueError("metadata_json must encode an object")
        result = PredictionArchive(int(values["seed"]), names, np.array(values["source_global_ids"]),
                                   np.array(values["target_global_ids"]), str(values["gene_order_sha256"]),
                                   tuple(np.array(values[f"prediction_{i:04d}"]) for i in range(len(names))),
                                   metadata)
    _validate(result)
    return result


def require_alignment(archive: PredictionArchive, *, seed: int, condition_names: Sequence[str],
                      source_global_ids: np.ndarray, target_global_ids: np.ndarray,
                      gene_order_sha256: str, n_genes: int | None = None) -> None:
    """Reject any sampling, condition, gene, or seed mismatch before scoring."""
    if archive.seed != seed:
        raise ValueError("Prediction seed mismatch")
    if archive.condition_names != tuple(map(str, condition_names)):
        raise ValueError("Prediction condition order mismatch")
    if archive.gene_order_sha256 != gene_order_sha256:
        raise ValueError("Prediction gene-order hash mismatch")
    if not np.array_equal(archive.source_global_ids, source_global_ids):
        raise ValueError("Prediction source/control row IDs mismatch")
    if not np.array_equal(archive.target_global_ids, target_global_ids):
        raise ValueError("Prediction target/treated row IDs mismatch")
    if n_genes is not None and archive.n_genes != n_genes:
        raise ValueError("Prediction gene count mismatch")
