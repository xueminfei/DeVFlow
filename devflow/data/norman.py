"""Norman gene-perturbation indexing and deterministic train/test splits.

The loader intentionally avoids scanpy/anndata.  It reads condition metadata
from the h5ad index and accesses selected sparse expression rows on demand.
The manifest is condition-level: every treated cell from a test condition stays
in test, while the common control pool is available to either split.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Iterable, Optional, Sequence

import h5py
import numpy as np
from torch.utils.data import Dataset


CTRL_NAMES = {"ctrl", "control", "NegCtrl0", "non-targeting", "non-targeting control"}


def _decode(values):
    array = np.asarray(values)
    if array.dtype.kind in {"S", "O"}:
        return [value.decode() if isinstance(value, bytes) else str(value) for value in array]
    return [str(value) for value in array]


def _read_column(group, name):
    obj = group[name]
    if isinstance(obj, h5py.Dataset):
        values = obj[()]
        if values.dtype.kind in {"S", "O"}:
            return _decode(values)
        return values
    categories = _decode(obj["categories"][()])
    codes = np.asarray(obj["codes"][()], dtype=np.int64)
    return np.asarray([categories[int(code)] for code in codes], dtype=object)


def parse_target_genes(condition: str) -> tuple[str, ...]:
    """Parse Norman names such as AHR+ctrl or AHR+FEV."""
    genes = []
    for token in str(condition).split("+"):
        token = token.strip()
        if not token or token in CTRL_NAMES:
            continue
        genes.append(token)
    if len(genes) not in (1, 2):
        raise ValueError(f"expected one or two target genes, got {condition!r}: {genes}")
    return tuple(genes)


def _h5ad_metadata(h5ad_path: str | Path):
    source = Path(h5ad_path).expanduser().resolve()
    with h5py.File(source, "r") as handle:
        conditions = np.asarray(_read_column(handle["obs"], "condition"), dtype=object)
        control = np.asarray(_read_column(handle["obs"], "control"), dtype=np.int64)
        gene_names = _decode(handle["var"]["gene_name"][()])
        cell_types = _decode(_read_column(handle["obs"], "cell_type")) if "cell_type" in handle["obs"] else []
    return source, conditions, control, gene_names, cell_types


def dataset_summary(h5ad_path: str | Path) -> dict:
    """Return the basic Norman data composition without loading expression X."""
    source, conditions, control, gene_names, cell_types = _h5ad_metadata(h5ad_path)
    perturb = [str(value) for value, flag in zip(conditions, control) if not int(flag)]
    unique_conditions = sorted(set(perturb))
    arity = Counter(len(parse_target_genes(value)) for value in unique_conditions)
    return {
        "h5ad_path": str(source),
        "num_cells": int(len(conditions)),
        "num_genes": int(len(gene_names)),
        "num_control_cells": int(np.sum(control.astype(bool))),
        "num_perturbed_cells": int(np.sum(~control.astype(bool))),
        "num_perturbation_conditions": int(len(unique_conditions)),
        "num_single_gene_conditions": int(arity.get(1, 0)),
        "num_double_gene_conditions": int(arity.get(2, 0)),
        "cell_types": sorted(set(cell_types)),
    }


def _select_test_conditions(
    perturb_conditions: Sequence[str],
    test_fraction: float,
    seed: int,
    split_mode: str,
    test_conditions: Optional[Sequence[str]],
):
    available = sorted(str(value) for value in perturb_conditions)
    available_set = set(available)
    if test_conditions is not None:
        chosen = sorted({str(value) for value in test_conditions})
    else:
        if not 0.0 < float(test_fraction) < 1.0:
            raise ValueError("test_fraction must be in (0,1)")
        mode = str(split_mode)
        if mode == "condition_random":
            candidates = available
        elif mode == "double_pair_holdout":
            # Singles are always available to learn the additive components;
            # only unseen double-gene conditions enter the external test set.
            candidates = [value for value in available if len(parse_target_genes(value)) == 2]
        elif mode == "gene_holdout":
            genes = sorted({gene for value in available for gene in parse_target_genes(value)})
            n_genes = max(1, int(round(len(genes) * float(test_fraction))))
            rng = np.random.default_rng(int(seed))
            held_out = set(str(value) for value in rng.choice(np.asarray(genes, dtype=object), n_genes, replace=False))
            return sorted(value for value in available if held_out.intersection(parse_target_genes(value))), sorted(held_out)
        else:
            raise ValueError("split_mode must be condition_random, double_pair_holdout, or gene_holdout")
        if not candidates:
            raise ValueError(f"split_mode={mode!r} has no eligible test conditions")
        count = max(1, int(round(len(candidates) * float(test_fraction))))
        rng = np.random.default_rng(int(seed))
        chosen = sorted(str(value) for value in rng.choice(np.asarray(candidates, dtype=object), count, replace=False))
    unknown = sorted(set(chosen) - available_set)
    if unknown:
        raise ValueError(f"test conditions are absent from h5ad: {unknown[:5]}")
    return chosen, sorted({gene for value in chosen for gene in parse_target_genes(value)})


def build_condition_split(
    h5ad_path: str | Path,
    output_path: str | Path,
    test_fraction: float = 0.2,
    seed: int = 42,
    test_conditions: Optional[Sequence[str]] = None,
    split_mode: str = "double_pair_holdout",
):
    """Write a deterministic condition-level train/test manifest.

    double_pair_holdout is the default Norman benchmark split: all 101
    single-gene conditions are retained for training and a fixed fraction of
    the 125 double-gene conditions is held out as unseen test pairs.  The
    condition_random and gene_holdout modes are provided for ablations.
    There is deliberately no validation split.
    """
    source, conditions, control, gene_names, cell_types = _h5ad_metadata(h5ad_path)
    perturb_conditions = sorted({str(value) for value, flag in zip(conditions, control) if not int(flag)})
    if not perturb_conditions:
        raise ValueError("no perturbed conditions found")
    chosen, test_genes = _select_test_conditions(
        perturb_conditions, test_fraction, seed, split_mode, test_conditions
    )
    test_set = set(chosen)
    train_conditions = [value for value in perturb_conditions if value not in test_set]
    train_arity = Counter(len(parse_target_genes(value)) for value in train_conditions)
    test_arity = Counter(len(parse_target_genes(value)) for value in chosen)
    payload = {
        "format_version": 2,
        "h5ad_path": str(source),
        "split_unit": "condition",
        "split_mode": str(split_mode),
        "seed": int(seed),
        "test_fraction": float(test_fraction),
        "train_conditions": train_conditions,
        "test_conditions": list(chosen),
        "test_target_genes": test_genes,
        "num_cells": int(len(conditions)),
        "num_genes": int(len(gene_names)),
        "num_control_cells": int(np.sum(control.astype(bool))),
        "num_perturbed_cells": int(np.sum(~control.astype(bool))),
        "cell_types": sorted(set(cell_types)),
        "num_perturbation_conditions": len(perturb_conditions),
        "num_train_conditions": len(train_conditions),
        "num_test_conditions": len(chosen),
        "train_single_gene_conditions": int(train_arity.get(1, 0)),
        "train_double_gene_conditions": int(train_arity.get(2, 0)),
        "test_single_gene_conditions": int(test_arity.get(1, 0)),
        "test_double_gene_conditions": int(test_arity.get(2, 0)),
    }
    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


class NormanH5AD(Dataset):
    """Condition-level Norman index with sparse row access."""

    def __init__(
        self,
        h5ad_path: str | Path,
        split_manifest: str | Path,
        split: str = "train",
        gene_indices: Optional[Sequence[int]] = None,
        normalize: bool = False,
    ):
        self.h5ad_path = str(Path(h5ad_path).expanduser().resolve())
        self.manifest_path = str(Path(split_manifest).expanduser().resolve())
        self.split = str(split)
        self.normalize = bool(normalize)
        if self.split not in {"train", "test"}:
            raise ValueError("split must be train or test")
        manifest = json.loads(Path(self.manifest_path).read_text())
        with h5py.File(self.h5ad_path, "r") as handle:
            self.gene_names = _decode(handle["var"]["gene_name"][()])
            self.var_names = _decode(handle["var"]["_index"][()])
            obs = handle["obs"]
            conditions = np.asarray(_read_column(obs, "condition"), dtype=object)
            control = np.asarray(_read_column(obs, "control"), dtype=np.int64)
            self.n_obs = int(handle["X"]["indptr"].shape[0] - 1)
            self.n_vars = int(len(self.gene_names))
        self.control_rows = np.flatnonzero(control.astype(bool)).astype(np.int64)
        self.conditions = conditions
        self.control = control
        allowed = manifest[f"{self.split}_conditions"]
        self.records = []
        for condition in allowed:
            rows = np.flatnonzero((conditions == condition) & (control == 0)).astype(np.int64)
            if len(rows) == 0:
                continue
            self.records.append({
                "condition": str(condition),
                "target_genes": parse_target_genes(str(condition)),
                "treated_rows": rows,
            })
        if not self.records:
            raise ValueError(f"no records found for split={self.split}")
        self.gene_indices = None if gene_indices is None else np.asarray(gene_indices, dtype=np.int64)
        if self.gene_indices is not None:
            if self.gene_indices.ndim != 1 or np.any(self.gene_indices < 0) or np.any(self.gene_indices >= self.n_vars):
                raise ValueError("gene_indices contains invalid columns")
            if len(self.gene_indices) and np.any(self.gene_indices[1:] < self.gene_indices[:-1]):
                raise ValueError("gene_indices must be sorted ascending")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[int(index)]
        return {
            "condition": record["condition"],
            "target_genes": record["target_genes"],
            "treated_rows": record["treated_rows"].copy(),
            "control_rows": self.control_rows.copy(),
        }

    def expression(self, rows: Iterable[int]) -> np.ndarray:
        """Read selected sparse rows as log1p-normalized dense values."""
        rows = np.asarray(list(rows), dtype=np.int64)
        columns = self.gene_indices
        output_dim = self.n_vars if columns is None else len(columns)
        values = np.zeros((len(rows), output_dim), dtype=np.float32)
        with h5py.File(self.h5ad_path, "r") as handle:
            data = handle["X/data"]
            indices = handle["X/indices"]
            indptr = handle["X/indptr"]
            for out_row, row in enumerate(rows):
                start, stop = int(indptr[int(row)]), int(indptr[int(row) + 1])
                cols = np.asarray(indices[start:stop], dtype=np.int64)
                vals = np.asarray(data[start:stop], dtype=np.float32)
                if columns is None:
                    values[out_row, cols] = vals
                elif len(columns):
                    positions = np.searchsorted(columns, cols)
                    valid = (positions < len(columns)) & (columns[np.minimum(positions, len(columns) - 1)] == cols)
                    values[out_row, positions[valid]] = vals[valid]
        if self.normalize:
            totals = values.sum(axis=1, keepdims=True)
            values = np.log1p(10000.0 * values / np.maximum(totals, 1.0))
        return values

    def sample_condition(
        self,
        index: int,
        cells_per_condition: Optional[int] = None,
        seed: int = 42,
        full_treated: bool = True,
    ):
        record = self.records[int(index)]
        rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(index)]))
        treated_rows = record["treated_rows"]
        if not full_treated and cells_per_condition is not None and len(treated_rows) > cells_per_condition:
            treated_rows = np.sort(rng.choice(treated_rows, int(cells_per_condition), replace=False))
        # Full-treated evaluation uses the whole treated population and draws
        # exactly the same number of controls from the shared control pool.
        n_controls = len(treated_rows) if full_treated or cells_per_condition is None else int(cells_per_condition)
        if len(self.control_rows) < n_controls:
            raise ValueError("control pool is smaller than the requested sample")
        control_rows = np.sort(rng.choice(self.control_rows, n_controls, replace=False))
        return {
            "condition": record["condition"],
            "target_genes": record["target_genes"],
            "control_rows": control_rows,
            "treated_rows": treated_rows,
            "control_expression": self.expression(control_rows),
            "treated_expression": self.expression(treated_rows),
        }


def load_norman_metadata(h5ad: Path):
    with h5py.File(h5ad, "r") as handle:
        shape = tuple(int(v) for v in handle["X"].attrs["shape"])
        conditions = np.asarray(_read_column(handle["obs"], "condition"), dtype=object)
        control_raw = _read_column(handle["obs"], "control")
        control = np.asarray(control_raw, dtype=np.int64).astype(bool)
        cell_ids = np.asarray(_decode(handle["obs"]["_index"][()]), dtype=object)
        genes = _decode(handle["var"].get("gene_name", handle["var"]["_index"])[()])
    if shape[0] != len(conditions) or shape[0] != len(cell_ids):
        raise ValueError("h5ad obs/X dimensions disagree")
    return conditions, control, cell_ids, genes, shape

def read_expression_rows(h5ad: Path, rows: Sequence[int], hvg_indices: np.ndarray) -> np.ndarray:
    """Read pre-normalized X rows and select Norman's train-only HVGs."""
    rows = np.asarray(rows, dtype=np.int64)
    hvg_indices = np.asarray(hvg_indices, dtype=np.int64)
    if hvg_indices.ndim != 1 or len(hvg_indices) == 0 or np.any(hvg_indices[1:] <= hvg_indices[:-1]):
        raise ValueError("gene indices must be nonempty, unique, and sorted")
    output = np.zeros((len(rows), len(hvg_indices)), dtype=np.float32)
    with h5py.File(h5ad, "r") as handle:
        data = handle["X/data"]
        indices = handle["X/indices"]
        indptr = handle["X/indptr"]
        for out, row in enumerate(rows.tolist()):
            start = int(indptr[int(row)])
            stop = int(indptr[int(row) + 1])
            cols = np.asarray(indices[start:stop], dtype=np.int64)
            values = np.asarray(data[start:stop], dtype=np.float32)
            pos = np.searchsorted(hvg_indices, cols)
            valid = (pos < len(hvg_indices)) & (hvg_indices[np.minimum(pos, len(hvg_indices) - 1)] == cols)
            output[out, pos[valid]] = values[valid]
    return output
