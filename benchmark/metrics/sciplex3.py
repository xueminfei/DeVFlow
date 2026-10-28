"""SciPlex3 formal metric formulas, decoupled from data and model code.

HVG2000 preserves the original FP32 torch MMD and target-fitted PCA energy
distance. DE50 preserves the finalized stable float64 direct distances.
See docs/benchmark_source_provenance.json for the inspected source identities.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import pearsonr, spearmanr

HVG_METRICS = ("pseudobulk_mae", "pseudobulk_mse", "pearson_delta", "pearson_deltapert",
               "weighted_mse", "weighted_r2_deltactrl", "de_spearman", "spearman_delta",
               "endpoint_mmd", "emd", "e_distance", "nir", "pds", "legacy_wmse",
               "legacy_weighted_r2_delta")
DE50_METRICS = ("pseudobulk_mae", "pseudobulk_mse", "pearson_delta", "pearson_deltapert",
                "weighted_mse", "weighted_r2_deltactrl", "endpoint_mmd", "emd", "e_distance",
                "nir", "pds")


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    if a.size < 2 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def centered_pearson(a, b, minimum_norm=0.):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    a, b = a - a.mean(), b - b.mean()
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denominator) if denominator > minimum_norm else float("nan")


def weighted_r2_delta_ctrl(pred_profile, target_profile, control_profile, weights) -> float:
    """Original calibrated weighted R², including its lower bound of -1."""
    weights = np.asarray(weights, dtype=np.float64)
    weight_sum = float(weights.sum())
    if weight_sum <= 0:
        return float("nan")
    true_delta = np.asarray(target_profile, dtype=np.float64) - np.asarray(control_profile, dtype=np.float64)
    pred_delta = np.asarray(pred_profile, dtype=np.float64) - np.asarray(control_profile, dtype=np.float64)
    true_mean = float(np.sum(weights * true_delta) / weight_sum)
    denominator = float(np.sum(weights * np.square(true_delta - true_mean)))
    if denominator <= 0:
        return float("nan")
    numerator = float(np.sum(weights * np.square(true_delta - pred_delta)))
    return float(max(1.0 - numerator / denominator, -1.0))


def truth_only_mmd_torch(pred: np.ndarray, target: np.ndarray) -> float:
    """Exact HVG formal override: FP32 torch.cdist and lower treated median."""
    try:
        import torch
    except ImportError as error:
        raise ImportError("HVG historical MMD requires torch; install the evaluation extra") from error
    x = torch.from_numpy(np.asarray(pred, np.float32))
    y = torch.from_numpy(np.asarray(target, np.float32))
    yy = torch.cdist(y, y).square()
    mask = ~torch.eye(len(y), dtype=torch.bool)
    median = torch.median(yy[mask]).clamp_min(1e-12)
    xx, xy = torch.cdist(x, x).square(), torch.cdist(x, y).square()
    values = []
    for scale in (.5, 1., 2., 4.):
        bandwidth = median * scale
        values.append(((torch.exp(-xx / (2 * bandwidth)).sum() - len(x)) / (len(x) * (len(x) - 1))
                       + (torch.exp(-yy / (2 * bandwidth)).sum() - len(y)) / (len(y) * (len(y) - 1))
                       - 2 * torch.exp(-xy / (2 * bandwidth)).mean()).item())
    return float(np.mean(values))


def truth_only_mmd_scipy(pred: np.ndarray, target: np.ndarray) -> float:
    """Canonical DE50: direct FP64 distances, lower median, no clipping."""
    pred, target = np.asarray(pred, np.float64), np.asarray(target, np.float64)
    n, m = len(pred), len(target)
    if min(n, m) < 2:
        raise ValueError("Unbiased MMD requires at least two cells per population")
    dxx, dyy, dxy = cdist(pred, pred, "sqeuclidean"), cdist(target, target, "sqeuclidean"), cdist(pred, target, "sqeuclidean")
    off_x, off_y = ~np.eye(n, dtype=bool), ~np.eye(m, dtype=bool)
    ordered = np.sort(dyy[off_y])
    median = max(float(ordered[(len(ordered) - 1) // 2]), 1e-12)
    values = []
    for scale in (.5, 1., 2., 4.):
        bandwidth = median * scale
        values.append(float(np.exp(-dxx / (2 * bandwidth))[off_x].mean()
                            + np.exp(-dyy / (2 * bandwidth))[off_y].mean()
                            - 2 * np.exp(-dxy / (2 * bandwidth)).mean()))
    return float(np.mean(values))


def energy_distance(pred: np.ndarray, target: np.ndarray, *, target_pca: bool, pca_float64: bool = False, clip_nonnegative: bool = False) -> float:
    if pca_float64:
        pred, target = np.asarray(pred, np.float64), np.asarray(target, np.float64)
    if target_pca:
        try:
            from sklearn.decomposition import PCA
        except ImportError as error:
            raise ImportError("HVG historical energy requires scikit-learn; install the evaluation extra") from error
        basis = PCA(n_components=max(1, min(50, len(target) - 1, target.shape[1])), svd_solver="full").fit(target)
        pred, target = basis.transform(pred), basis.transform(target)
    value = float(2 * cdist(pred, target).mean() - cdist(pred, pred).mean() - cdist(target, target).mean())
    return max(0., value) if clip_nonnegative else value


def score_group(pred, target, control, *, weights, training_mean, de_mask,
                same_context_targets, position: int, panel: str = "hvg",
                gene_indices=None, distance_backend: str | None = None,
                target_profile=None, control_profile=None,
                include_distribution: bool = True) -> dict[str, float]:
    """Score one condition; profile means are taken in full FP32 space first.

DE50 indices are fixed by the supplied evaluation reference, never selected by
this function. All retrieval candidates use the query's own DE50 gene panel.
"""
    pred, target, control = (np.asarray(x, np.float32) for x in (pred, target, control))
    if pred.shape != target.shape or control.shape != target.shape or pred.ndim != 2:
        raise ValueError("Predicted, treated, and control expression shapes must agree")
    if not all(np.isfinite(x).all() for x in (pred, target, control)):
        raise ValueError("Expression contains non-finite values")
    historical_profiles = distance_backend in ("combo_historical", "norman_historical")
    profile_dtype = np.float64 if historical_profiles else np.float32
    pm, tm, cm = (x.astype(profile_dtype, copy=False).mean(0) for x in (pred, target, control))
    if target_profile is not None:
        tm = np.asarray(target_profile, dtype=profile_dtype)
    if control_profile is not None:
        cm = np.asarray(control_profile, dtype=profile_dtype)
    if tm.shape != pm.shape or cm.shape != pm.shape:
        raise ValueError("Response profiles must match the full gene axis")
    ref, w = np.asarray(training_mean), np.asarray(weights)
    profiles = np.asarray(same_context_targets)
    if historical_profiles:
        ref, w, profiles = (np.asarray(x, np.float64) for x in (ref, w, profiles))
    de_mask = np.asarray(de_mask, bool)
    if panel == "de50":
        if gene_indices is None:
            raise ValueError("DE50 scoring requires frozen per-condition gene_indices")
        indices = np.asarray(gene_indices, np.int64)
        pred, target = pred[:, indices], target[:, indices]
        pm, tm, cm, ref, w = pm[indices], tm[indices], cm[indices], ref[indices], w[indices].copy()
        profiles, de_mask = profiles[:, indices], de_mask[indices]
        if w.sum() > 0:
            w /= w.sum()
    elif panel != "hvg":
        raise ValueError(f"Unknown scoring panel: {panel}")
    dp, dt, error = pm - cm, tm - cm, pm - tm
    # HVG original Pearson uses scipy on FP32; DeltaPert uses calibrated FP64.
    corr = (float(pearsonr(dp, dt)[0]) if np.std(dp) > 0 and np.std(dt) > 0 else float("nan")) if panel == "hvg" and distance_backend != "combo_historical" else pearson(dp, dt)
    if historical_profiles:
        corr = centered_pearson(dp, dt, 1e-12 if distance_backend == "norman_historical" else 0.)
    pert_corr = centered_pearson(pm - ref, tm - ref, 1e-12 if distance_backend == "norman_historical" else 0.) if historical_profiles else pearson(pm - ref, tm - ref)
    result = dict(pseudobulk_mae=float(np.abs(error).mean()), pseudobulk_mse=float(np.square(error).mean()),
                  pearson_delta=corr, pearson_deltapert=pert_corr,
                  weighted_mse=float(np.sum(w * np.square(error)) / w.sum()) if w.sum() > 0 else float("nan"),
                  weighted_r2_deltactrl=weighted_r2_delta_ctrl(pm, tm, cm, w))
    backend = distance_backend or ("torch_fp32_pca50" if panel == "hvg" else "scipy_float64")
    if backend not in ("torch_fp32_pca50", "scipy_float64", "combo_historical", "norman_historical"):
        raise ValueError(f"Unknown distance backend: {backend}")
    if include_distribution:
        if backend in ("combo_historical", "norman_historical"):
            import torch
            from devflow.networks.mlp_velocity import mmd2_unbiased_multi_sigma
            mmd = float(mmd2_unbiased_multi_sigma(torch.from_numpy(pred), torch.from_numpy(target)).item())
            emd = float(np.abs(np.sort(pred.astype(np.float64), axis=0) - np.sort(target.astype(np.float64), axis=0)).mean())
            energy = energy_distance(pred, target, target_pca=backend == "combo_historical", pca_float64=True, clip_nonnegative=backend == "combo_historical")
        else:
            mmd = (truth_only_mmd_torch if backend == "torch_fp32_pca50" else truth_only_mmd_scipy)(pred, target)
            emd = float(np.abs(np.sort(pred, axis=0) - np.sort(target, axis=0)).mean())
            energy = energy_distance(pred, target, target_pca=backend == "torch_fp32_pca50")
        result.update(endpoint_mmd=mmd, emd=emd, e_distance=energy)
    if not 0 <= position < len(profiles):
        raise ValueError("Retrieval position is outside its context")
    other = np.arange(len(profiles)) != position
    l2 = np.linalg.norm(profiles - pm[None, :], axis=1)
    l1 = np.abs(profiles - pm[None, :]).sum(axis=1)
    result.update(nir=float(np.mean(l2[other] > l2[position])) if other.any() else float("nan"),
                  pds=float(np.mean(l1[other] > l1[position])) if other.any() else float("nan"))
    if panel == "hvg":
        result["spearman_delta"] = float(spearmanr(dp, dt)[0]) if np.std(dp) > 0 and np.std(dt) > 0 else float("nan")
        result["de_spearman"] = float(spearmanr(dp[de_mask], dt[de_mask])[0]) if de_mask.sum() >= 2 else float("nan")
        magnitude = np.abs(dt)
        lo, hi = float(magnitude.min()), float(magnitude.max())
        old_w = np.ones_like(magnitude) if hi <= lo else ((magnitude - lo) / (hi - lo)) ** 2
        weight_sum = np.sum(old_w)
        result["legacy_wmse"] = float(np.sum(old_w * error ** 2) / weight_sum) if weight_sum > 0 else float("nan")
        denominator = float(np.sum(old_w * (dt - np.sum(old_w * dt) / weight_sum) ** 2)) if weight_sum > 0 else 0.
        result["legacy_weighted_r2_delta"] = float(max(1 - np.sum(old_w * (dt - dp) ** 2) / denominator, -1)) if denominator > 1e-20 else float("nan")
    return result


def score_condition(pred, target, control, weights, de_mask, training_mean,
                    target_profiles, target_index: int, *, include_distribution: bool = True,
                    panel: str = "hvg", gene_indices=None, distance_backend=None,
                    target_profile=None, control_profile=None) -> dict[str, float]:
    """Public online/formal API; disabling distances omits their three keys.

``target_profiles`` contains the pseudobulk profiles of conditions in the same
retrieval context, in exactly the order indexed by ``target_index``.
"""
    return score_group(pred, target, control, weights=weights, de_mask=de_mask,
                       training_mean=training_mean, same_context_targets=target_profiles,
                       position=target_index, panel=panel, gene_indices=gene_indices,
                       distance_backend=distance_backend, target_profile=target_profile,
                       control_profile=control_profile, include_distribution=include_distribution)
