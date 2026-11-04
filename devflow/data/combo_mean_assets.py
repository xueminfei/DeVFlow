"""Training-only conditional mean assets for the original ComboSciPlex split.

No cell embedding or raw preprocessing is rerun here. This computes 24 mean
labels from the existing count bundle and selects a small kernel grid by
leaving out one *training* combination at a time. Selection can use all 18
folds or the nine where both held-out drugs retain response training data.
All six monotherapies remain in every fold; the final fit uses all 24 anchors.
The final learned coefficients are zero;
closed-form fits are used only to choose fixed kernel hyperparameters.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from devflow.data.combosciplex import (CELL_LINES, NORMALIZATION, TRAIN_CONDITIONS,
    ComboDataset, load_bundle, sequence_sha256, training_binding)
from devflow.mean.kernels import MixedKernelBank, morgan_fingerprints
from devflow.networks.chemical_condition import canonical_drug_names, canonical_smiles, load_drug_bank
from devflow.runtime.artifacts import atomic_json, sha256

SCHEMA = "combosciplex_mlp_mean_assets_v1"
DEFAULT_GRID = {"kg_mix": [0.0, 0.5, 1.0], "gamma_factor": [0.5, 1.0, 2.0],
                "alpha": [0.001, 0.01, 0.1, 1.0, 10.0]}
LEGACY_KERNEL = "canonical_pair_concat_MAPKG_RBF_Morgan_Tanimoto"
PERMUTATION_AVERAGED_KERNEL = "permutation_averaged_pair_concat_MAPKG_RBF_Morgan_Tanimoto"
ADDITIVE_KERNEL = "sum_single_drug_MAPKG_RBF_Morgan_Tanimoto"
ALL_COMBINATION_FOLDS = "leave_one_training_combination_out"
SEEN_DRUG_COMBINATION_FOLDS = "leave_one_seen_drug_training_combination_out"


def configured_selection(config):
    settings = config.get("mean_head", {})
    method = settings.get("selection", ALL_COMBINATION_FOLDS)
    if method not in (ALL_COMBINATION_FOLDS, SEEN_DRUG_COMBINATION_FOLDS):
        raise ValueError("Unsupported Combo kernel selection: " + str(method))
    if settings.get("selection_metric", "delta_mse_2000_genes") != "delta_mse_2000_genes":
        raise ValueError("Combo kernel selection uses training-only 2000-gene delta MSE")
    return method


def validate_selection_mode(config, *, manifest=None, metadata=None, report=None):
    """Old assets imply 18-fold selection; never reuse them as nine-fold assets."""
    method = configured_selection(config)
    for source, values in (("manifest", manifest), ("metadata", metadata)):
        if values is not None and values.get("kernel_selection_method", ALL_COMBINATION_FOLDS) != method:
            raise ValueError("Combo kernel selection differs from " + source
                             + "; prepare matching assets in a new directory")
    if report is not None:
        expected_folds = 9 if method == SEEN_DRUG_COMBINATION_FOLDS else 18
        if report.get("method") != method or report.get("folds") != expected_folds:
            raise ValueError("Combo kernel selection report differs from requested policy")
    return method


def combination_folds(records):
    """Describe drug coverage using remaining training conditions only."""
    folds = [i for i, record in enumerate(records) if len(record["drugs"]) == 2]
    if len(records) != 24 or len(folds) != 18:
        raise ValueError("Kernel selection requires the original 6 singles + 18 training combinations")
    singles = {record["drugs"][0] for record in records if len(record["drugs"]) == 1}
    coverage = []
    for held in folds:
        remaining = {drug for i, record in enumerate(records) if i != held for drug in record["drugs"]}
        drugs = set(records[held]["drugs"])
        unseen = sorted(drugs - remaining)
        coverage.append({"condition": records[held]["condition"], "held_out_drugs": sorted(drugs),
            "drugs_missing_from_remaining_training_responses": unseen,
            "both_drugs_seen_in_remaining_training_responses": not unseen,
            "held_out_drugs_with_remaining_monotherapy": len(drugs & singles)})
    return folds, coverage


def configured_kernel(config):
    name = config.get("mean_head", {}).get("kernel", LEGACY_KERNEL)
    if name not in (LEGACY_KERNEL, PERMUTATION_AVERAGED_KERNEL, ADDITIVE_KERNEL):
        raise ValueError("Unsupported Combo mean kernel: " + str(name))
    return name


def validate_kernel_mode(config, *, manifest=None, metadata=None, kernelbank_assets=None):
    """Keep old assets legacy-only and reject config/asset mode mismatches."""
    name = configured_kernel(config)
    requested = name != LEGACY_KERNEL
    for source, values in (("manifest", manifest), ("metadata", metadata),
                           ("kernelbank_assets", kernelbank_assets)):
        if values is None:
            continue
        recorded = values.get("pair_permutation_invariant", False)
        if not isinstance(recorded, bool) or recorded != requested:
            raise ValueError("Combo mean kernel mode differs from " + source
                             + "; prepare matching assets in a new directory")
        if source == "kernelbank_assets" and values.get("pair_aggregation", "concat") != (
                "sum" if name == ADDITIVE_KERNEL else "concat"):
            raise ValueError("Combo mean kernel aggregation differs from saved assets")
        if source != "kernelbank_assets" and values.get("kernel", LEGACY_KERNEL) != name:
            raise ValueError("Combo mean kernel name differs from " + source)
    return requested


def configured_grid(config):
    settings = config.get("mean_head", {})
    if int(settings.get("morgan_radius", 2)) != 2 or int(settings.get("morgan_bits", 2048)) != 2048:
        raise ValueError("This Combo recipe fixes Morgan radius=2 and 2048 bits per drug")
    return {name: list(settings.get(name + "_grid", default)) for name, default in DEFAULT_GRID.items()}


def paired_features(record, embedding_by_drug, fingerprint_by_drug):
    """The exact lexical two slots used by the native Combo Flow condition."""
    drugs = canonical_drug_names(record["drugs"])
    features = [np.asarray(embedding_by_drug[name], dtype=np.float32) for name in drugs]
    fingerprints = [np.asarray(fingerprint_by_drug[name], dtype=np.uint8) for name in drugs]
    if len(drugs) == 1:
        features.append(np.zeros_like(features[0]))
        fingerprints.append(np.zeros_like(fingerprints[0]))
    return np.concatenate(features), np.concatenate(fingerprints)


def full_expression_mean(dataset, rows):
    rows = np.asarray(rows, dtype=np.int64)
    total = np.zeros(dataset.expression_dim, dtype=np.float64)
    for start in range(0, len(rows), 512):
        values = dataset._expression_tensor(dataset.bundle["counts"], rows[start:start + 512])
        total += values.numpy().astype(np.float64).sum(0)
    return total / len(rows)


def input_identities(config):
    bundle = Path(config["data"]["bundle"])
    paths = {name: bundle / name for name in (
        "obs.csv", "genes.json", "manifest.json", "drug_smiles.json", "counts_hvg.npy", "se_embeddings.npy")}
    paths["drug_bank"] = Path(config["data"]["drug_bank"])
    return {name: {"path": str(path.resolve()), "sha256": sha256(path)}
            for name, path in paths.items()}


def select_kernel(features, fingerprints, deltas, records, grid=None, *,
                  pair_permutation_invariant=False, selection=ALL_COMBINATION_FOLDS,
                  pair_aggregation="concat"):
    """Refit feature scales, kernel centering and intercept inside every fold."""
    grid = {**DEFAULT_GRID, **dict(grid or {})}
    mixes = sorted(set(float(x) for x in grid["kg_mix"]))
    factors = sorted(set(float(x) for x in grid["gamma_factor"]))
    alphas = sorted(set(float(x) for x in grid["alpha"]))
    if (not mixes or not factors or not alphas or any(not 0 <= x <= 1 for x in mixes)
            or any(x <= 0 or not np.isfinite(x) for x in factors + alphas)):
        raise ValueError("Invalid fixed Combo training-only kernel search grid")
    specs = [{"kg_mix": mix, "gamma_factor": factor}
             for mix in mixes for factor in ([None] if mix == 0 else factors)]
    candidates = [{**spec, "alpha": alpha} for spec in specs for alpha in alphas]
    selection = configured_selection({"mean_head": {"selection": selection}})
    folds, coverage = combination_folds(records)
    seen = [i for i, item in enumerate(coverage)
            if item["both_drugs_seen_in_remaining_training_responses"]]
    unseen = [i for i in range(len(folds)) if i not in seen]
    if selection == SEEN_DRUG_COMBINATION_FOLDS and len(seen) != 9:
        raise ValueError("The original Combo split must have exactly nine seen-drug folds")
    scoring = seen if selection == SEEN_DRUG_COMBINATION_FOLDS else list(range(len(folds)))
    losses = np.empty((len(candidates), len(folds)), dtype=np.float64)
    for fold, held in enumerate(folds):
        keep = np.asarray([i for i in range(len(records)) if i != held], dtype=np.int64)
        bias = deltas[keep].mean(0)
        centered_targets = deltas[keep] - bias
        bank = MixedKernelBank.fit(features[keep], fingerprints[keep], specs,
            pair_permutation_invariant=pair_permutation_invariant, pair_aggregation=pair_aggregation)
        candidate = 0
        for s, _spec in enumerate(specs):
            query = bank.query(features[held:held + 1], fingerprints[held:held + 1], s)
            # Solving against the query first avoids a large 23 x 2000 solve
            # for each alpha while giving exactly the same KRR prediction.
            for alpha in alphas:
                weights = np.linalg.solve(bank.train_kernels[s] + alpha * np.eye(len(keep)), query.T).T
                prediction = weights @ centered_targets + bias
                losses[candidate, fold] = np.square(prediction[0] - deltas[held]).mean()
                candidate += 1
    means = losses[:, scoring].mean(1)
    chosen = min(range(len(candidates)), key=lambda i: (float(means[i]), i))
    rows = [{**spec, "mean_delta_mse": float(means[i]),
             "fold_delta_mse": losses[i, scoring].tolist(),
             "all_fold_delta_mse": losses[i].tolist(),
             "all_18_fold_mean_delta_mse": float(losses[i].mean()),
             "seen_drug_fold_mean_delta_mse": float(losses[i, seen].mean()) if seen else None,
             "unseen_drug_fold_mean_delta_mse": float(losses[i, unseen].mean()) if unseen else None}
            for i, spec in enumerate(candidates)]
    report = {"method": selection, "folds": len(scoring), "evaluated_folds": len(folds),
        "monotherapies_retained_in_every_fold": 6,
        "fold_conditions": [records[folds[i]]["condition"] for i in scoring],
        "all_fold_conditions": [records[i]["condition"] for i in folds],
        "selection_fold_indices": scoring,
        "excluded_fold_indices": [i for i in range(len(folds)) if i not in scoring],
        "seen_drug_fold_indices": seen, "unseen_drug_fold_indices": unseen,
        "fold_drug_coverage": coverage,
        "excluded_folds_used_for_selection": False,
        "final_training_conditions": len(records),
        "score": "equal-fold mean squared delta error across all 2000 genes",
        "fold_feature_standardization_and_kernel_centering": "training_anchors_only",
        "pair_permutation_invariant": bool(pair_permutation_invariant),
        "kernel": ADDITIVE_KERNEL if pair_aggregation == "sum" else (
            PERMUTATION_AVERAGED_KERNEL if pair_permutation_invariant else LEGACY_KERNEL),
        "external_test_used": False, "grid": grid, "candidates": rows,
        "selected": candidates[chosen], "selected_mean_delta_mse": float(means[chosen]),
        "tie_break": "first candidate in ascending mix, gamma, alpha order"}
    return candidates[chosen], report


def prepare_assets(config):
    """Prepare immutable training statistics; never fit/load the model's A."""
    import rdkit

    kernel_name = configured_kernel(config)
    selection = configured_selection(config)
    pair_permutation_invariant = validate_kernel_mode(config)
    pair_aggregation = "sum" if kernel_name == ADDITIVE_KERNEL else "concat"
    root = Path(config["data"]["mlp_assets"]).resolve()
    if (root / "manifest.json").exists():
        manifest = json.loads((root / "manifest.json").read_text())
        if manifest.get("schema") != SCHEMA or manifest.get("status") != "complete":
            raise ValueError("Existing Combo mean assets are incomplete or incompatible")
        validate_kernel_mode(config, manifest=manifest)
        validate_selection_mode(config, manifest=manifest,
            report=json.loads((root / "kernel_selection.json").read_text()))
        identities = input_identities(config)
        if {k: v["sha256"] for k, v in identities.items()} != {
                k: v["sha256"] for k, v in manifest["inputs"].items()}:
            raise ValueError("Existing Combo mean assets belong to different input content")
        requested_grid = configured_grid(config)
        if manifest["kernel_grid"] != requested_grid:
            raise ValueError("Kernel grid changed; prepare mean assets in a new directory")
        for name, expected in manifest["files"].items():
            if sha256(root / name) != expected:
                raise ValueError("Existing Combo mean asset changed: " + name)
        saved = torch.load(root / "mean_assets.pt", map_location="cpu", weights_only=False)
        validate_kernel_mode(config, manifest=manifest, metadata=saved["metadata"],
                             kernelbank_assets=saved["kernelbank_assets"])
        validate_selection_mode(config, metadata=saved["metadata"])
        return {"status": "complete", "reused": True, "manifest": str(root / "manifest.json")}
    if root.exists() and any(root.iterdir()):
        raise FileExistsError("Use an empty directory for new Combo mean assets")
    root.mkdir(parents=True, exist_ok=True)
    bundle = load_bundle(config["data"]["bundle"], require_state=True, full_scan=True)
    dataset = ComboDataset(CELL_LINES, "train", config["data"]["bundle"], _bundle=bundle)
    if len(dataset.genes) != 2000 or set(r["condition"] for r in dataset.records) != TRAIN_CONDITIONS:
        raise ValueError("Expected original Combo 2000-gene, 24-condition training split")
    for record in dataset.records:
        if any(bundle["obs"][row]["split"] != "train" for row in record["rows"]):
            raise ValueError("A nontraining cell entered a Combo mean label")
    names = sorted({name for record in dataset.records for name in record["drugs"]})
    smiles_index, bank = load_drug_bank(config["data"]["drug_bank"], "cpu")
    canonical = [canonical_smiles(bundle["drug_smiles"][name]) for name in names]
    features = np.stack([bank[smiles_index[value]].numpy() for value in canonical]).astype(np.float32)
    fingerprints = morgan_fingerprints(canonical, radius=2, n_bits=2048, use_chirality=True)
    by_drug, fp_by_drug = dict(zip(names, features)), dict(zip(names, fingerprints))
    pairs = [paired_features(record, by_drug, fp_by_drug) for record in dataset.records]
    pair_features = np.stack([item[0] for item in pairs])
    pair_fingerprints = np.stack([item[1] for item in pairs])
    control = full_expression_mean(dataset, dataset.control_rows)
    targets = np.stack([full_expression_mean(dataset, record["rows"]) - control for record in dataset.records])
    grid = configured_grid(config)
    selected, cv_report = select_kernel(pair_features, pair_fingerprints, targets, dataset.records, grid,
        pair_permutation_invariant=pair_permutation_invariant, selection=selection,
        pair_aggregation=pair_aggregation)
    spec = {name: selected[name] for name in ("kg_mix", "gamma_factor")}
    kernel = MixedKernelBank.fit(pair_features, pair_fingerprints, [spec],
        pair_permutation_invariant=pair_permutation_invariant, pair_aggregation=pair_aggregation)
    binding = training_binding(bundle)
    metadata = {"task": "combosciplex", "schema": SCHEMA, "training_binding": binding,
        "conditions": [r["condition"] for r in dataset.records], "strata": ["A549_neutral_dose"],
        "training_only_statistics": True, "external_expression_used_for_fitting": False,
        "kernel": kernel_name, "pair_permutation_invariant": pair_permutation_invariant,
        "kernel_selection_method": selection,
        "condition_representation": "canonical_name_sorted_concat_MAPKG_2x1024",
        "structure_representation": "canonical_name_sorted_concat_binary_Morgan_2x2048",
        "single_drug_padding": "second_slot_exact_zero", "rdkit_version": rdkit.__version__,
        "morgan": {"radius": 2, "bits_per_drug": 2048, "use_chirality": True},
        "expression_normalization": NORMALIZATION, "selected_hyperparameters": selected,
        "initial_coefficients": "fresh_zero_A; no_closed_form_initialization"}
    mean_assets = {"kernel": torch.from_numpy(kernel.train_kernels),
        "targets": torch.from_numpy(targets[None]), "bias": torch.from_numpy(targets.mean(0)[None]),
        "control_mean": torch.from_numpy(control[None]), "alpha": torch.tensor([selected["alpha"]], dtype=torch.float64),
        "kernelbank_assets": kernel.to_assets(), "metadata": metadata}
    torch.save(mean_assets, root / "mean_assets.pt")
    np.savez_compressed(root / "drug_features.npz", drug_names=np.asarray(names), canonical_smiles=np.asarray(canonical),
                        embeddings=features, fingerprints=fingerprints)
    np.save(root / "condition_latent_delta.npy",
            np.stack([dataset.population_state_delta(i).numpy() for i in range(len(dataset))]))
    atomic_json(root / "conditions.json", metadata["conditions"])
    atomic_json(root / "kernel_selection.json", cv_report)
    filenames = ["mean_assets.pt", "drug_features.npz", "condition_latent_delta.npy", "conditions.json", "kernel_selection.json"]
    manifest = {"schema": SCHEMA, "status": "complete", "training_binding": binding,
        "training_only_statistics": True, "external_expression_used_for_fitting": False,
        "kernel": kernel_name, "pair_permutation_invariant": pair_permutation_invariant,
        "kernel_selection_method": selection,
        "kernel_grid": grid, "kernel_selection": selected, "inputs": input_identities(config),
        "training_rows_sha256": sequence_sha256([row["cell_id"] for row in bundle["obs"] if row["split"] == "train"]),
        "files": {name: sha256(root / name) for name in filenames}}
    atomic_json(root / "manifest.json", manifest)
    return {"status": "complete", "reused": False, "manifest": str(root / "manifest.json"),
            "selected_kernel": selected, "train_conditions": 24}
