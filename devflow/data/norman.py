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


def paired_gene_features(condition_names, feature_assets):
    """Build two exchangeable gene slots; absent singleton slots stay zero.

    Gene priors are fixed before training. They are not the condition vectors
    that Flow's trainable interaction encoder produces.
    """
    names = [str(name) for name in feature_assets["gene_names"]]
    lookup = {name: index for index, name in enumerate(names)}
    if len(lookup) != len(names):
        raise ValueError("Fixed gene feature names must be unique")
    first = np.asarray(feature_assets["primary"], dtype=np.float64)
    second = np.asarray(feature_assets["secondary"], dtype=np.float64)
    if (first.ndim != 2 or second.ndim != 2 or len(first) != len(names)
            or len(second) != len(names) or not np.isfinite(first).all()
            or not np.isfinite(second).all()):
        raise ValueError("Fixed gene features must be finite [genes, features] matrices")
    primary = np.zeros((len(condition_names), 2, first.shape[1]), dtype=np.float64)
    secondary = np.zeros((len(condition_names), 2, second.shape[1]), dtype=np.float64)
    present = np.zeros((len(condition_names), 2), dtype=bool)
    for row, condition in enumerate(condition_names):
        for slot, gene in enumerate(sorted(parse_target_genes(condition))):
            if gene not in lookup:
                raise ValueError(f"Missing fixed gene prior: {gene}")
            primary[row, slot] = first[lookup[gene]]
            secondary[row, slot] = second[lookup[gene]]
            present[row, slot] = True
    return primary, secondary, present


def select_gene_mean_kernel(primary, secondary, present, deltas, condition_names,
                            settings=None, progress=None):
    """Select the fixed kernel using only held-out *training* double conditions.

    Each fold retains all singles. Scalers, distance scales, centering and the
    intercept are refitted from that fold's anchors. Analytical KRR fits are
    used for selection, never as the learned head's initial coefficients.
    """
    from scipy.linalg import eigh
    from devflow.mean.kernels import ContinuousPairKernelBank

    options = dict(settings or {})
    folds = int(options.get("cv_folds", 8))
    seed = int(options.get("cv_seed", 42))
    grid = {
        "primary_mix": sorted(set(float(x) for x in options.get(
            "primary_mix_grid", [0.0, 0.25, 0.5, 0.75, 1.0]))),
        "gamma_factor": sorted(set(float(x) for x in options.get(
            "gamma_factor_grid", [0.125, 0.25, 0.5, 1.0, 2.0, 4.0]))),
        "alpha": sorted(set(float(x) for x in options.get(
            "alpha_grid", [1e-5, 3e-5, 1e-4, 3e-4, 0.001, 0.003, 0.01, 0.1, 1.0, 10.0]))),
    }
    if (folds != 8 or any(not values for values in grid.values())
            or any(not np.isfinite(x) or not 0 <= x <= 1 for x in grid["primary_mix"])
            or any(not np.isfinite(x) or x <= 0 for key in ("gamma_factor", "alpha") for x in grid[key])):
        raise ValueError("Expected eight folds and a finite, valid continuous-kernel grid")
    names = list(map(str, condition_names))
    doubles = np.asarray([i for i, name in enumerate(names) if len(parse_target_genes(name)) == 2], dtype=np.int64)
    singles = np.asarray([i for i, name in enumerate(names) if len(parse_target_genes(name)) == 1], dtype=np.int64)
    if len(names) != 189 or len(singles) != 101 or len(doubles) != 88:
        raise ValueError("Norman additive mean selection requires 101 singles and 88 training doubles")
    deltas = np.asarray(deltas, dtype=np.float64)
    if deltas.shape != (189, 2054) or not np.isfinite(deltas).all():
        raise ValueError("Training delta labels must have shape [189,2054] and be finite")
    specs = [{"primary_mix": mix, "gamma_factor": factor}
             for mix in grid["primary_mix"] for factor in grid["gamma_factor"]]
    candidates = [{**spec, "alpha": alpha} for spec in specs for alpha in grid["alpha"]]
    scores = np.empty((len(candidates), folds), dtype=np.float64)
    held_out = [np.sort(part) for part in np.array_split(np.random.default_rng(seed).permutation(doubles), folds)]
    for fold, held in enumerate(held_out):
        keep = np.setdiff1d(np.arange(len(names), dtype=np.int64), held)
        if not set(singles) <= set(keep) or len(keep) != 178 or len(held) != 11:
            raise AssertionError("Every CV fold must retain all singles and hold eleven doubles")
        bias = deltas[keep].mean(axis=0)
        centered_labels = deltas[keep] - bias
        bank = ContinuousPairKernelBank.fit(primary[keep], secondary[keep], present[keep], specs)
        candidate = 0
        for stratum in range(len(specs)):
            kernel = bank.train_kernels[stratum]
            values, vectors = eigh(kernel, check_finite=False)
            if values[0] < -1e-7:
                raise ValueError("The fixed training kernel is not positive semidefinite")
            query = bank.query(primary[held], secondary[held], present[held], stratum)
            projected_query = query @ vectors
            for alpha in grid["alpha"]:
                # Solve against eleven query rows, not 2054 target columns.
                weights = (projected_query / (values + alpha)[None, :]) @ vectors.T
                prediction = weights @ centered_labels + bias
                scores[candidate, fold] = np.square(prediction - deltas[held]).mean()
                if not np.isfinite(scores[candidate, fold]):
                    raise FloatingPointError(f"Nonfinite training CV score at fold {fold + 1}, candidate {candidate}")
                candidate += 1
        if progress is not None:
            progress({"stage": "kernel_cross_validation", "fold": fold + 1,
                      "folds": folds, "train_conditions": len(keep),
                      "held_conditions": [names[i] for i in held],
                      "best_running_mean_delta_mse": float(scores[:, :fold + 1].mean(axis=1).min())})
    means = scores.mean(axis=1)
    chosen = min(range(len(candidates)), key=lambda index: (float(means[index]), index))
    report = {
        "method": "eight_fold_training_double_condition_holdout", "seed": seed,
        "folds": folds, "singles_retained_per_fold": 101,
        "training_anchors_per_fold": 178, "held_doubles_per_fold": 11,
        "fold_conditions": [[names[i] for i in held] for held in held_out],
        "score": "equal-fold mean squared delta error across all 2054 genes",
        "scalers_bandwidths_centering_and_bias": "refitted_on_training_anchors_inside_each_fold",
        "external_test_used": False, "grid": grid,
        "n_candidates": len(candidates),
        "candidates": [{**candidate, "mean_delta_mse": float(means[i]),
                        "fold_delta_mse": scores[i].tolist()} for i, candidate in enumerate(candidates)],
        "selected": candidates[chosen], "selected_mean_delta_mse": float(means[chosen]),
        "tie_break": "first candidate in ascending primary_mix, gamma_factor, alpha order",
        "analytical_fits": "CV selection and training-only reference; never A initialization",
    }
    return dict(candidates[chosen]), report


def prepare_kernel_mean_assets(data, settings=None, progress=None):
    """Build Norman's fixed gene kernel and mean labels from training rows only.

    ``data`` is the validated mapping returned by runtime.norman.load_inputs.
    This function does not access test expression, references or DEG weights.
    The caller creates TrainableKernelMeanHead from the returned assets, which
    starts A at zero just as the chemical mean-head recipes do.
    """
    import hashlib
    import torch
    from devflow.io import gene_order_hash, sha256_file
    from devflow.mean.kernels import ContinuousPairKernelBank
    from devflow.networks.genetic_condition import FrozenMAPGeneProjector

    options = dict(settings or {})
    paths = data["paths"]
    records = data["dataset"].records
    names = [str(record["condition"]) for record in records]
    if len(names) != 189 or len(data["genes"]) != 2054:
        raise ValueError("Expected the original Norman additive 189-condition, 2054-gene training axes")
    train_set = set(data["split"]["train_conditions"])
    if len(train_set) != 189 or set(names) != train_set:
        raise ValueError("Mean labels must cover precisely the declared training conditions")
    if train_set.intersection(data["split"]["test_conditions"]):
        raise ValueError("Training and external conditions overlap")
    ordered_genes = [gene for gene, _ in sorted(data["target_lookup"].items(), key=lambda item: item[1])]
    secondary = data["target_esm"].detach().cpu().float()
    target_path = paths.get("target_se_assets")
    if not target_path:
        raise ValueError("Norman kernel means require data.target_se_assets; prepare the training-control target gene tokens first")
    target_cache = torch.load(str(target_path), map_location="cpu", weights_only=False)
    target_metadata = target_cache.get("metadata", {})
    if target_metadata.get("complete") is not True:
        raise ValueError("Target SE cache is incomplete")
    for key in ("esm_checkpoint_sha256", "se_checkpoint_sha256"):
        expected = data.get("asset_metadata", {}).get(key)
        if not expected or target_metadata.get(key) != expected:
            raise ValueError("Target SE cache uses a different frozen prior: " + key)
    for key in ("h5ad", "split_manifest", "control_split"):
        if target_metadata.get(key + "_sha256") != sha256_file(paths[key]):
            raise ValueError("Target SE cache was built from a different " + key)
    controls = np.asarray(data["train_controls"], dtype=np.int64)
    if (target_metadata.get("train_control_rows") != controls.tolist()
            or target_metadata.get("train_control_rows_sha256") != gene_order_hash(controls)):
        raise ValueError("Target SE cache training controls differ")
    cache_names = list(map(str, target_cache["target_genes"]))
    if len(cache_names) != len(set(cache_names)) or set(cache_names) != set(ordered_genes):
        raise ValueError("Target SE cache must cover precisely the declared target genes")
    cache_lookup = {gene: index for index, gene in enumerate(cache_names)}
    cache_order = torch.as_tensor([cache_lookup[gene] for gene in ordered_genes])
    se_tokens = torch.as_tensor(target_cache["target_se_tokens"], dtype=torch.float32)
    cache_esm = torch.as_tensor(target_cache["target_esm"], dtype=torch.float32)
    if se_tokens.shape != (len(cache_names), 2048) or cache_esm.shape != (len(cache_names), 5120):
        raise ValueError("Target SE cache feature dimensions differ")
    se_tokens, cache_esm = se_tokens[cache_order], cache_esm[cache_order]
    if not torch.isfinite(se_tokens).all() or not torch.equal(cache_esm, secondary):
        raise ValueError("Target SE cache contains invalid tokens or a different ESM2 gene mapping")
    with torch.random.fork_rng(devices=[]):
        projector = FrozenMAPGeneProjector(paths["projector_checkpoint"], trainable_last_layer=False).eval()
    with torch.inference_mode():
        primary = projector.forward_target(
            secondary[None], target_se_token=se_tokens[None])[0].float().numpy().copy()
    features = {
        "gene_names": np.asarray(ordered_genes, dtype=str),
        "primary": primary, "secondary": secondary.numpy().copy(),
        "primary_description": "original_pretrained_MAP_predictor_with_fixed_training_control_SE_gene_token_and_ESM2",
        "secondary_description": "fixed_ESM2_5120dim_target_gene_vectors",
        "independent_biological_modalities": False,
        "target_se_tokens": se_tokens.numpy().copy(),
    }
    primary_pairs, secondary_pairs, presence = paired_gene_features(names, features)
    if progress is not None:
        progress({"stage": "fixed_gene_features", "genes": len(ordered_genes),
                  "primary_dim": int(primary.shape[1]), "secondary_dim": int(secondary.shape[1])})

    def expression_mean(rows):
        rows = np.asarray(rows, dtype=np.int64)
        if len(rows) == 0:
            raise ValueError("A training expression mean requires at least one row")
        total = np.zeros(len(data["genes"]), dtype=np.float64)
        for start in range(0, len(rows), 512):
            values = read_expression_rows(Path(paths["h5ad"]), rows[start:start + 512], data["indices"])
            total += values.sum(axis=0, dtype=np.float64)
        return total / len(rows)

    if not data["control"][controls].all() or np.intersect1d(controls, data["test_controls"]).size:
        raise ValueError("Only the disjoint training control pool may define mean deltas")
    control_mean = expression_mean(controls)
    deltas = []
    for index, record in enumerate(records):
        rows = np.asarray(record["treated_rows"], dtype=np.int64)
        if not np.all(data["conditions"][rows] == record["condition"]) or data["control"][rows].any():
            raise ValueError("Mean target rows do not belong to the declared training condition")
        deltas.append(expression_mean(rows) - control_mean)
        if progress is not None and (index + 1) % 25 == 0:
            progress({"stage": "training_mean_labels", "completed_conditions": index + 1,
                      "conditions": len(records)})
    deltas = np.stack(deltas)
    if progress is not None:
        progress({"stage": "training_mean_labels", "conditions": len(names),
                  "expression_dim": len(data["genes"]), "external_expression_used": False})
    selected, report = select_gene_mean_kernel(
        primary_pairs, secondary_pairs, presence, deltas, names, options, progress)
    spec = {key: selected[key] for key in ("primary_mix", "gamma_factor")}
    bank = ContinuousPairKernelBank.fit(primary_pairs, secondary_pairs, presence, [spec])
    train_rows = np.flatnonzero(np.isin(data["conditions"], data["split"]["train_conditions"]) & ~data["control"])
    binding = {
        "split_manifest_sha256": sha256_file(paths["split_manifest"]),
        "control_split_sha256": sha256_file(paths["control_split"]),
        "gene_order_sha256": gene_order_hash(data["genes"]),
        "train_treated_rows_sha256": gene_order_hash(train_rows),
    }

    def array_hash(value):
        value = np.ascontiguousarray(value)
        digest = hashlib.sha256()
        digest.update(str(value.dtype).encode())
        digest.update(str(value.shape).encode())
        digest.update(value.tobytes())
        return digest.hexdigest()

    identity_keys = ("h5ad", "split_manifest", "control_split", "gene_indices", "genes",
                     "gene_condition_assets", "projector_checkpoint", "hvg_provenance", "target_se_assets")
    identities = {}
    for key in identity_keys:
        if paths.get(key):
            if progress is not None:
                progress({"stage": "input_identity", "input": key})
            identities[key] = {"path": str(Path(paths[key]).resolve()), "sha256": sha256_file(paths[key])}
    metadata = {
        "schema": "norman_fixed_gene_kernel_mean_assets_v2", "task": "norman19",
        "kernel": "permutation_averaged_pair_projector_RBF_ESM2_RBF",
        "pair_permutation_invariant": True, "conditions": names,
        "strata": ["K562"], "training_binding": binding,
        "training_control_rows_sha256": gene_order_hash(controls),
        "training_only_statistics": True, "external_expression_used_for_fitting": False,
        "mean_accumulation": "FP64 full-condition means of existing FP32 pre-normalized expression",
        "selected_hyperparameters": selected, "input_identities": identities,
        "fixed_zero_target_role": False, "fixed_projector": True,
        "target_se_context": "training_control_pool_mean; all targets retained in a single SE forward",
        "target_se_metadata": target_metadata,
        "initial_coefficients": "fresh_zero_A; analytical_CV_coefficients_not_loaded",
        "array_sha256": {
            "target_gene_order": gene_order_hash(ordered_genes),
            "primary_gene_features": array_hash(primary),
            "secondary_gene_features": array_hash(features["secondary"]),
            "target_se_tokens": array_hash(features["target_se_tokens"]),
            "training_deltas": array_hash(deltas), "training_control_mean": array_hash(control_mean),
        },
    }
    features["metadata"] = {
        "input_identities": {key: identities[key] for key in ("gene_condition_assets", "projector_checkpoint", "target_se_assets")},
        "array_sha256": {key: metadata["array_sha256"][key] for key in (
            "target_gene_order", "primary_gene_features", "secondary_gene_features", "target_se_tokens")},
    }
    assets = {
        "kernel": bank.train_kernels.copy(), "targets": deltas[None].copy(),
        "bias": deltas.mean(axis=0)[None], "control_mean": control_mean[None],
        "alpha": np.asarray([selected["alpha"]], dtype=np.float64),
        "kernelbank_assets": bank.to_assets(), "metadata": metadata,
    }
    return assets, report, features
