"""Fixed, training-centred kernels used by conditional mean heads.

The mixed chemical kernel reproduces the historical MAPKG RBF / Morgan
Tanimoto calculation. Its coefficients are learned separately in ``head``.
Features used to fit standardisation and centring must be training anchors.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


def squared_distances(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    distances = np.maximum((a * a).sum(1)[:, None]
                           + (b * b).sum(1)[None, :] - 2.0 * (a @ b.T), 0.0)
    if a is b:
        np.fill_diagonal(distances, 0.0)
    return distances


def tanimoto(a, b):
    """Binary-fingerprint Tanimoto; two empty fingerprints have similarity 1."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    intersection = a @ b.T
    union = a.sum(1)[:, None] + b.sum(1)[None, :] - intersection
    return np.divide(intersection, union, out=np.ones_like(intersection), where=union > 0)


def _swap_pair_slots(values):
    """Exchange the two equal-width slots of an already validated matrix."""
    midpoint = values.shape[1] // 2
    return np.concatenate((values[:, midpoint:], values[:, :midpoint]), axis=1)


def _validate_pair_inputs(features, fingerprints):
    if not features.shape[1] or features.shape[1] % 2:
        raise ValueError("Pair kernels require two equal-width MAPKG slots")
    if not fingerprints.shape[1] or fingerprints.shape[1] % 2:
        raise ValueError("Pair kernels require two equal-width Morgan slots")
    if not np.isin(fingerprints, [0, 1]).all():
        raise ValueError("Morgan Tanimoto expects binary fingerprints")
    if not fingerprints.any(axis=1).all():
        raise ValueError("A pair kernel condition cannot contain two empty drug slots")


def _sum_drug_kernel(x, f, y, g, *, mix, scale):
    """Kernel of summed single-drug features; empty padding contributes zero."""
    xs, ys = x.reshape(len(x), 2, -1), y.reshape(len(y), 2, -1)
    fs, gs = f.reshape(len(f), 2, -1), g.reshape(len(g), 2, -1)
    present_x, present_y = fs.any(2), gs.any(2)
    result = np.zeros((len(x), len(y)), dtype=np.float64)
    for i in range(2):
        for j in range(2):
            values = (1.0 - mix) * tanimoto(fs[:, i], gs[:, j])
            if mix > 0:
                values += mix * np.exp(-scale * squared_distances(xs[:, i], ys[:, j]))
            result += values * (present_x[:, i, None] & present_y[None, :, j])
    return result


def center_training_kernel(kernel):
    kernel = np.asarray(kernel, dtype=np.float64)
    column_mean = kernel.mean(axis=0)
    grand_mean = float(kernel.mean())
    centered = kernel - kernel.mean(axis=1, keepdims=True) - column_mean[None] + grand_mean
    return centered, column_mean, grand_mean


def center_query_kernel(kernel, training_column_mean, training_grand_mean):
    kernel = np.asarray(kernel, dtype=np.float64)
    return (kernel - kernel.mean(axis=1, keepdims=True)
            - np.asarray(training_column_mean)[None] + float(training_grand_mean))


def morgan_fingerprints(smiles: Sequence[str], *, radius=2, n_bits=2048,
                        use_chirality=True):
    """Build explicit binary features; callers record their RDKit version."""
    from rdkit import Chem, DataStructs
    from rdkit.Chem import AllChem
    rows = []
    for value in smiles:
        mol = Chem.MolFromSmiles(str(value))
        if mol is None:
            raise ValueError(f"Cannot parse SMILES: {value!r}")
        fingerprint = AllChem.GetMorganFingerprintAsBitVect(
            mol, int(radius), nBits=int(n_bits), useChirality=bool(use_chirality))
        row = np.zeros(int(n_bits), dtype=np.uint8)
        DataStructs.ConvertToNumpyArray(fingerprint, row)
        if not row.any():
            raise ValueError("Training/prediction molecular fingerprints must be nonempty")
        rows.append(row)
    if not rows:
        raise ValueError("At least one molecule is required")
    return np.stack(rows)


@dataclass
class MixedKernelBank:
    """Serializable fixed anchor features and one kernel specification per stratum."""

    feature_mean: np.ndarray
    feature_std: np.ndarray
    x_train: np.ndarray
    fingerprint_train: np.ndarray
    specs: list[dict]
    median_squared_distance: float
    train_kernels: np.ndarray
    training_column_means: np.ndarray
    training_grand_means: np.ndarray
    pair_permutation_invariant: bool = False
    pair_aggregation: str = "concat"

    @classmethod
    def fit(cls, train_features, train_fingerprints, specs: Sequence[Mapping], *,
            pair_permutation_invariant=False, pair_aggregation="concat"):
        """Fit training-only scales and optionally average both pair alignments.

        A shared slot transform makes simultaneous permutation an isometry.
        The aligned/crossed average is then the full four-term group average,
        so the permutation-invariant RBF and Tanimoto kernels remain PSD.
        ``sum`` instead adds single-drug kernels and ignores empty slots.
        """
        if not isinstance(pair_permutation_invariant, (bool, np.bool_)):
            raise ValueError("pair_permutation_invariant must be a boolean")
        pair_permutation_invariant = bool(pair_permutation_invariant)
        if pair_aggregation not in ("concat", "sum") or (pair_aggregation == "sum" and not pair_permutation_invariant):
            raise ValueError("Single-drug sum kernels require explicit permutation-invariant pairs")
        x = np.asarray(train_features, dtype=np.float64)
        fingerprints = np.asarray(train_fingerprints, dtype=np.uint8)
        if x.ndim != 2 or len(x) < 2 or not np.isfinite(x).all():
            raise ValueError("Training features must be a finite [anchors, features] matrix")
        if fingerprints.ndim != 2 or len(fingerprints) != len(x):
            raise ValueError("Fingerprints must have the same training anchor order")
        if not np.isin(fingerprints, [0, 1]).all():
            raise ValueError("Morgan Tanimoto expects binary fingerprints")
        if pair_permutation_invariant:
            _validate_pair_inputs(x, fingerprints)
            slots = x.reshape(len(x), 2, -1)
            if pair_aggregation == "sum":
                present = fingerprints.reshape(len(x), 2, -1).any(2)
                mean = np.tile(slots[present].mean(0), 2)
                std = np.tile(slots[present].std(0), 2)
            else:
                mean = np.tile(slots.mean(axis=(0, 1)), 2)
                std = np.tile(slots.std(axis=(0, 1)), 2)
        else:
            mean, std = x.mean(0), x.std(0)
        std[std < 1e-6] = 1.0  # Original MAPKG training-only standardisation.
        z = (x - mean) / std
        distances = squared_distances(z, z)
        if pair_aggregation == "sum":
            observed = z.reshape(len(z), 2, -1)[present]
            drug_distances = squared_distances(observed, observed)
            positive = drug_distances[drug_distances > 1e-12]
        elif pair_permutation_invariant:
            crossed_distances = squared_distances(z, _swap_pair_slots(z))
            distinct = np.triu_indices(len(x), k=1)
            scale_distances = np.concatenate((distances[distinct], crossed_distances[distinct]))
            positive = scale_distances[scale_distances > 1e-12]
        else:
            positive = distances[distances > 1e-12]
        if not positive.size:
            raise ValueError("Distinct training features are needed for the RBF distance scale")
        median = float(np.median(positive))
        structural = tanimoto(fingerprints, fingerprints)
        if pair_permutation_invariant:
            structural = 0.5 * (structural + tanimoto(fingerprints, _swap_pair_slots(fingerprints)))
        matrices, columns, grands, clean_specs = [], [], [], []
        for raw in specs:
            spec = dict(raw)
            mix = float(spec["kg_mix"])
            factor = spec.get("gamma_factor")
            if not 0.0 <= mix <= 1.0:
                raise ValueError("kg_mix must lie in [0,1]")
            if mix > 0 and (factor is None or float(factor) <= 0):
                raise ValueError("An RBF component requires a positive gamma_factor")
            if pair_aggregation == "sum":
                kernel = _sum_drug_kernel(z, fingerprints, z, fingerprints,
                    mix=mix, scale=float(factor) / median if mix > 0 else 0.0)
            elif pair_permutation_invariant and mix > 0:
                rbf = 0.5 * (np.exp(-float(factor) / median * distances)
                             + np.exp(-float(factor) / median * crossed_distances))
                kernel = mix * rbf + (1.0 - mix) * structural
            else:
                kernel = structural.copy() if mix == 0 else (
                    mix * np.exp(-float(factor) / median * distances) + (1.0 - mix) * structural)
            centered, col, grand = center_training_kernel(kernel)
            matrices.append(centered)
            columns.append(col)
            grands.append(grand)
            clean_specs.append(spec)
        if not matrices:
            raise ValueError("At least one stratum specification is required")
        return cls(mean, std, z, fingerprints.copy(), clean_specs, median,
                   np.stack(matrices), np.stack(columns), np.asarray(grands),
                   pair_permutation_invariant=pair_permutation_invariant, pair_aggregation=pair_aggregation)

    def query(self, features, fingerprints, stratum: int):
        s = int(stratum)
        if s < 0 or s >= len(self.specs):
            raise IndexError("Unknown mean-head stratum")
        x = np.asarray(features, dtype=np.float64)
        f = np.asarray(fingerprints, dtype=np.uint8)
        if x.ndim == 1:
            x = x[None]
        if f.ndim == 1:
            f = f[None]
        if x.ndim != 2 or x.shape[1] != self.x_train.shape[1] or not np.isfinite(x).all():
            raise ValueError("Query feature axis differs from the fitted training anchors")
        if f.shape != (len(x), self.fingerprint_train.shape[1]):
            raise ValueError("Query fingerprints and feature rows differ")
        if self.pair_permutation_invariant:
            _validate_pair_inputs(x, f)
        z = (x - self.feature_mean) / self.feature_std
        structural = tanimoto(f, self.fingerprint_train)
        if self.pair_permutation_invariant:
            structural = 0.5 * (structural + tanimoto(f, _swap_pair_slots(self.fingerprint_train)))
        spec = self.specs[s]
        mix = float(spec["kg_mix"])
        if self.pair_aggregation == "sum":
            kernel = _sum_drug_kernel(z, f, self.x_train, self.fingerprint_train,
                mix=mix, scale=float(spec["gamma_factor"]) / self.median_squared_distance if mix > 0 else 0.0)
        elif self.pair_permutation_invariant and mix > 0:
            scale = float(spec["gamma_factor"]) / self.median_squared_distance
            rbf = 0.5 * (np.exp(-scale * squared_distances(z, self.x_train))
                         + np.exp(-scale * squared_distances(z, _swap_pair_slots(self.x_train))))
            kernel = mix * rbf + (1.0 - mix) * structural
        else:
            kernel = structural if mix == 0 else (
                mix * np.exp(-float(spec["gamma_factor"]) / self.median_squared_distance
                             * squared_distances(z, self.x_train))
                + (1.0 - mix) * structural)
        return center_query_kernel(kernel, self.training_column_means[s],
                                   self.training_grand_means[s])

    def to_assets(self):
        assets = dict(feature_mean=self.feature_mean.copy(), feature_std=self.feature_std.copy(),
                    x_train=self.x_train.copy(), fingerprint_train=self.fingerprint_train.copy(),
                    specs=[dict(x) for x in self.specs],
                    median_squared_distance=self.median_squared_distance,
                    train_kernels=self.train_kernels.copy(),
                    training_column_means=self.training_column_means.copy(),
                    training_grand_means=self.training_grand_means.copy(),
                    pair_permutation_invariant=self.pair_permutation_invariant)
        if self.pair_aggregation != "concat":
            assets["pair_aggregation"] = self.pair_aggregation
        return assets

    @classmethod
    def from_assets(cls, assets):
        pair_permutation_invariant = assets.get("pair_permutation_invariant", False)
        if not isinstance(pair_permutation_invariant, (bool, np.bool_)):
            raise ValueError("Saved pair_permutation_invariant must be a boolean")
        aggregation = assets.get("pair_aggregation", "concat")
        if aggregation not in ("concat", "sum") or (aggregation == "sum" and not pair_permutation_invariant):
            raise ValueError("Unsupported saved pair kernel aggregation")
        return cls(
            np.asarray(assets["feature_mean"], dtype=np.float64),
            np.asarray(assets["feature_std"], dtype=np.float64),
            np.asarray(assets["x_train"], dtype=np.float64),
            np.asarray(assets["fingerprint_train"], dtype=np.uint8),
            [dict(s) for s in assets["specs"]], float(assets["median_squared_distance"]),
            np.asarray(assets["train_kernels"], dtype=np.float64),
            np.asarray(assets["training_column_means"], dtype=np.float64),
            np.asarray(assets["training_grand_means"], dtype=np.float64),
            pair_permutation_invariant=bool(pair_permutation_invariant), pair_aggregation=aggregation)
