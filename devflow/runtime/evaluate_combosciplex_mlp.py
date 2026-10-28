"""Locked ComboSciPlex evaluation for the trainable mean/MMD model.

The four sampling repeats (4243--4246) share the historical five-condition,
128-cell, full-treated-profile Combo protocol. Online seed 4243 is archived in
exactly the same format as final inference and is reused after identity checks.
Mean diagnostics never run the Flow and omit all distribution measurements.
"""
from __future__ import annotations

from dataclasses import asdict
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist

from benchmark.metrics import HVG_METRICS, score_group
from benchmark.protocol import EvaluationProtocol
from benchmark.references import EvaluationReferences
from devflow.io import (json_sha256, read_prediction_archive, require_alignment,
                        write_prediction_archive)
from .artifacts import atomic_json, sha256
from .checkpoint import source_identity
from .evaluation import (formal_mode, _checkpoint_identity, _cache_ok,
                                  _rank_zero_value, _clean, aggregate_rows)

SEEDS = (4243, 4244, 4245, 4246)
_FORMAL = dict(cells=128, conditions=5, genes=2000, time_points=80,
    rk4_intervals=79, precision="fp32", allow_tf32=False,
    expression="hvg_total_1000_log1p", response_profile="full_treated",
    mmd="combo_historical_pooled_bandwidth_HVG; truth_only_direct_FP64_DE50",
    aggregation="five-condition macro mean then four sampling-seed mean/sample SD",
    sampling_seeds=list(SEEDS))


def evaluation_assets(config):
    return Path(config["data"].get("evaluation_assets",
        str(Path(config["data"]["mlp_assets"]) / "evaluation")))


def prepare_references(config):
    """Copy immutable samples, add the locked DE50 panel, derive four repeats.

    This reads reporting-only assets. It must be called outside the training
    runtime and never supplies evaluation expression/weights to model fitting.
    """
    settings = config["data"]
    original = Path(settings["evaluation_references"]).resolve()
    protocol_path = Path(settings.get("evaluation_protocol",
        str(Path(__file__).resolve().parents[2] / "benchmark/protocols/combosciplex_v3.json")))
    source_protocol = EvaluationProtocol.load(protocol_path)
    if (source_protocol.task != "combosciplex" or source_protocol.condition_count != 5
            or source_protocol.cells_per_condition != 128 or source_protocol.n_genes != 2000
            or source_protocol.distance_backend_hvg != "combo_historical"
            or source_protocol.response_profile != "full_treated"
            or not set(SEEDS) <= set(source_protocol.seeds)):
        raise ValueError("Expected the historical five-condition Combo reference protocol")
    refs = EvaluationReferences(original, source_protocol)
    panel_path = Path(settings["de50_references"]).resolve()
    with np.load(panel_path, allow_pickle=False) as panel:
        if tuple(panel["condition_names"].astype(str)) != refs.condition_names:
            raise ValueError("DE50 panel condition order differs from Combo references")
        indices = np.array(panel["gene_indices"], dtype=np.int64)
        if (indices.shape != (5, 50) or np.any(indices < 0) or np.any(indices >= 2000)
                or any(len(set(row)) != 50 for row in indices)):
            raise ValueError("Invalid locked Combo DE50 indices")
        for name, expected in (("truth_profiles", refs.statistics["full_target_profiles"]),
                               ("continuous_deg_weights", refs.statistics["continuous_deg_weights"])):
            if not np.allclose(panel[name], expected, atol=1e-10, rtol=1e-10):
                raise ValueError("Locked DE50 reference differs from full HVG statistics: " + name)
        reference = refs.statistics["training_condition_mean"]
        if len(reference) != 1 or not np.allclose(panel["perturbation_reference"], reference[0], atol=1e-10, rtol=1e-10):
            raise ValueError("DE50 training perturbation reference differs")
    root = evaluation_assets(config).resolve()
    if root == original or original in root.parents:
        raise ValueError("Derived evaluation assets must not modify the historical reference directory")
    provenance = dict(source_manifest_sha256=sha256(original / "manifest.json"),
        source_protocol_sha256=sha256(protocol_path), de50_reference_sha256=sha256(panel_path),
        source_reference_directory=str(original), source_de50_reference=str(panel_path),
        sampling_seeds=list(SEEDS), de50_panel="historical locked per-condition Top50; never reselected")
    marker = root / "PREPARED.json"
    if marker.exists():
        previous = json.loads(marker.read_text())
        if previous["provenance"] != provenance:
            raise ValueError("Existing evaluation assets have a different source identity")
        protocol = EvaluationProtocol.load(root / "protocol.json")
        EvaluationReferences(root, protocol)
        if sha256(root / "protocol.json") != previous["protocol_sha256"]:
            raise ValueError("Derived evaluation protocol changed")
        return {**previous, "status": "already_prepared", "directory": str(root)}
    if root.exists() and any(root.iterdir()):
        raise FileExistsError("Incomplete derived evaluation directory must be inspected: " + str(root))
    root.mkdir(parents=True, exist_ok=True)
    values = {**refs.statistics, "de50_indices": indices}
    np.savez_compressed(root / "statistics.npz", **values)
    hashes = {"statistics.npz": sha256(root / "statistics.npz")}
    for seed in SEEDS:
        name = f"references_seed_{seed}.npz"
        shutil.copyfile(original / name, root / name)
        hashes[name] = sha256(root / name)
    manifest = dict(refs.manifest)
    manifest.update(files_sha256=hashes, derivation=provenance)
    manifest["evaluation_statistics"] = {**manifest["evaluation_statistics"],
        "de50_indices": "historical fixed per-condition Top50; evaluation/reporting only"}
    atomic_json(root / "manifest.json", manifest)
    protocol = {**asdict(source_protocol), "seeds": list(SEEDS),
        "protocol": "combosciplex_v3_shared_fourrepeat128_mean_mmd_v1",
        "gene_order_sha256": refs.gene_order_sha256,
        "training_binding_sha256": refs.training_binding_sha256}
    atomic_json(root / "protocol.json", protocol)
    EvaluationReferences(root, EvaluationProtocol.load(root / "protocol.json"))
    result = dict(status="complete", provenance=provenance,
                  protocol_sha256=sha256(root / "protocol.json"),
                  reference_manifest_sha256=sha256(root / "manifest.json"))
    atomic_json(marker, result)
    return dict(directory=str(root), **result)


def _references(config, runtime):
    root = evaluation_assets(config)
    protocol = EvaluationProtocol.load(root / "protocol.json")
    if (protocol.seeds != SEEDS or protocol.task != "combosciplex"
            or protocol.condition_count != 5 or protocol.cells_per_condition != 128
            or protocol.n_genes != 2000 or protocol.response_profile != "full_treated"
            or protocol.distance_backend_hvg != "combo_historical"):
        raise ValueError("Combo mean/MMD evaluation requires its locked four-repeat historical protocol")
    prepared = json.loads((root / "PREPARED.json").read_text())
    if (sha256(root / "protocol.json") != prepared["protocol_sha256"]
            or sha256(root / "manifest.json") != prepared["reference_manifest_sha256"]):
        raise ValueError("Prepared evaluation protocol or reference manifest changed")
    refs = EvaluationReferences(root, protocol)
    if (refs.gene_order_sha256 != runtime.data.gene_order_sha256
            or refs.manifest["training_statistics"].get("training_binding") != runtime.data.training_binding()):
        raise ValueError("Evaluation references belong to a different training split or gene axis")
    return refs, protocol


def _sampling(runtime, refs, seed):
    dataset = runtime.test_dataset(seed)
    records, source_ids, target_ids = [], [], []
    obs = runtime.bundle["obs"]
    targets, controls = [], []
    for index, record in enumerate(dataset.records):
        sample = dataset[index]
        source_rows = sample["control_rows"].numpy()
        target_rows = sample["treated_rows"].numpy()
        records.append({**record, "control_rows": source_rows, "target_rows": target_rows})
        source_ids.append([obs[int(row)]["cell_id"] for row in source_rows])
        target_ids.append([obs[int(row)]["cell_id"] for row in target_rows])
        targets.append(sample["treated_expression"].numpy())
        controls.append(sample["control_expression"].numpy())
    source_ids, target_ids = np.asarray(source_ids, dtype=str), np.asarray(target_ids, dtype=str)
    path = refs.directory / f"references_seed_{seed}.npz"
    with np.load(path, allow_pickle=False) as frozen:
        if (int(frozen["seed"]) != seed
                or frozen["condition_names"].astype(str).tolist() != [r["condition"] for r in records]
                or str(frozen["gene_order_sha256"]) != runtime.data.gene_order_sha256
                or not np.array_equal(frozen["source_global_ids"], source_ids)
                or not np.array_equal(frozen["target_global_ids"], target_ids)):
            raise ValueError("Current Combo sampling differs from locked historical cell IDs")
        for index in range(5):
            if (not np.array_equal(targets[index], frozen[f"target_{index:04d}"])
                    or not np.array_equal(controls[index], frozen[f"control_{index:04d}"])):
                raise ValueError("Current normalized expression differs from frozen evaluation samples")
        masks = np.array(frozen["de_mask"] if "de_mask" in frozen else refs.statistics["de_mask"])
    return dict(records=records, source_ids=source_ids, target_ids=target_ids,
        targets=targets, controls=controls, masks=masks,
        row_ids_sha256=json_sha256(dict(source=source_ids.tolist(), target=target_ids.tolist())))


def _score(prediction, index, bundle, refs, include_distribution):
    """Same score_group and arguments as benchmark.reproduce.common.evaluate."""
    stats = refs.statistics
    contexts = stats["context_names"].astype(str)
    group = np.flatnonzero(contexts == contexts[index])
    profiles = stats["full_target_profiles"]
    arguments = dict(weights=stats["continuous_deg_weights"][index],
        training_mean=stats["training_condition_mean"][stats["reference_index"][index]],
        de_mask=bundle["masks"][index], same_context_targets=profiles[group],
        position=int(np.flatnonzero(group == index)[0]), target_profile=profiles[index],
        include_distribution=include_distribution)
    values = score_group(prediction, bundle["targets"][index], bundle["controls"][index],
                         distance_backend="combo_historical", **arguments)
    de50 = score_group(prediction, bundle["targets"][index], bundle["controls"][index],
        panel="de50", gene_indices=stats["de50_indices"][index],
        distance_backend="scipy_float64", **arguments)
    return {**values, **{"de50_" + key: value for key, value in de50.items()}}


def _identity(model, runtime, checkpoint_path, step, refs):
    result = _checkpoint_identity(model, checkpoint_path, step, runtime)
    payload = torch.load(str(checkpoint_path), map_location="cpu", mmap=True, weights_only=False)
    source = source_identity()
    if payload.get("source_identity", {}).get("source_tree_sha256") != source["source_tree_sha256"]:
        raise ValueError("Evaluation source differs from the checkpoint; run its immutable source snapshot")
    result.update(asset_manifest_sha256=runtime.asset_manifest_sha256,
        training_binding=runtime.data.training_binding(),
        gene_order_sha256=runtime.data.gene_order_sha256,
        reference_manifest_sha256=sha256(refs.manifest_path),
        protocol_file_sha256=sha256(refs.directory / "protocol.json"),
        runtime_source_sha256=source["source_tree_sha256"], evaluator_sha256=sha256(__file__),
        protocol=_FORMAL)
    return result


def _validate_config(config):
    for section in ("evaluation", "prediction"):
        value = config.get(section, {})
        if (int(value.get("time_points", 80)) != 80 or int(value.get("cells", 128)) != 128
                or value.get("precision", "fp32") != "fp32" or value.get("allow_tf32", False)):
            raise ValueError("Combo evaluation requires FP32/TF32-off/RK4-80 and all 128 cells")
    if tuple(config.get("prediction", {}).get("seeds", SEEDS)) != SEEDS:
        raise ValueError("New Combo benchmark uses four sampling seeds 4243--4246")
    if list(config.get("evaluation", {}).get("monitor_seeds", [4243])) != [4243]:
        raise ValueError("Online Combo evaluation uses the one formal seed 4243")
    if int(config.get("prediction", {}).get("chunk", 128)) != 128:
        raise ValueError("Mean scaling must use the complete 128-cell condition together")


def _gather_results(action, rank, world):
    try:
        packet = {"result": action(), "error": None}
    except Exception as exc:
        packet = {"result": [], "error": f"rank {rank}: {type(exc).__name__}: {exc}"}
    if world > 1:
        packets = [None] * world
        dist.all_gather_object(packets, packet)
    else:
        packets = [packet]
    errors = [item["error"] for item in packets if item["error"]]
    if errors:
        raise RuntimeError("; ".join(errors))
    return [row for item in packets for row in item["result"]]


def _write_archive(path, seed, predictions, bundle, runtime, identity, branch, method):
    write_prediction_archive(path, seed=seed,
        condition_names=[r["condition"] for r in bundle["records"]],
        source_global_ids=bundle["source_ids"], target_global_ids=bundle["target_ids"],
        gene_order_sha256=runtime.data.gene_order_sha256, predictions=predictions,
        metadata={**identity, "method": method, "branch": branch,
                  "training_binding_sha256": json_sha256(runtime.data.training_binding())})
    archive = read_prediction_archive(path)
    require_alignment(archive, seed=seed,
        condition_names=[r["condition"] for r in bundle["records"]],
        source_global_ids=bundle["source_ids"], target_global_ids=bundle["target_ids"],
        gene_order_sha256=runtime.data.gene_order_sha256, n_genes=2000)


def _selection(output):
    """One declared primary metric, one checkpoint for every reported metric."""
    candidates = []
    for path in sorted((Path(output) / "evaluations").glob("step_*/summary.json")):
        item = json.loads(path.read_text())
        value = item.get("metrics", {}).get("weighted_mse")
        if (item.get("seeds") == [4243] and item.get("mode") == "full_prediction"
                and value is not None and np.isfinite(value)):
            candidates.append(item)
    protocols = {item["protocol_sha256"] for item in candidates}
    if len(protocols) > 1:
        raise ValueError("Checkpoint selection cannot mix evaluation protocols or source identities")
    if not candidates:
        return
    chosen = min(candidates, key=lambda item: (item["metrics"]["weighted_mse"], item["update"]))
    selected = dict(update=chosen["update"], optimizer_step=chosen["update"],
        checkpoint=chosen["checkpoint"], checkpoint_sha256=chosen["checkpoint_sha256"],
        primary_metric="weighted_mse", primary_value=chosen["metrics"]["weighted_mse"],
        selection_seed=4243, selection_scope="external_test_exploratory",
        tie_break="earlier optimizer step for an exactly equal primary value",
        protocol_sha256=chosen["protocol_sha256"],
        single_checkpoint_for_all_reported_metrics=True)
    atomic_json(Path(output) / "SELECTED_CHECKPOINT.json", selected)
    atomic_json(Path(output) / "checkpoint_candidates.json", dict(
        selection_scope="external_test_exploratory", primary_metric="weighted_mse",
        direction="min", selected_checkpoint=selected,
        candidates=[dict(update=item["update"], checkpoint=item["checkpoint"],
            checkpoint_sha256=item["checkpoint_sha256"], metrics=item["metrics"])
            for item in candidates]))


def _formal_benchmark(config, root, rows_by_branch, method):
    """Formal four-seed tables use the repository's standalone scorer unchanged."""
    from benchmark.reproduce.common import evaluate as formal_evaluate
    assets = evaluation_assets(config)
    manifests = {}
    for branch in ("final", "raw"):
        predictions = root / ("predictions" if branch == "final" else "raw_predictions")
        method_name = method if branch == "final" else method + " [raw decoder diagnostic]"
        for panel in ("hvg", "de50"):
            destination = root / "formal" / branch / panel
            marker = destination / "COMPLETED.json"
            if marker.exists():
                saved = json.loads(marker.read_text())
                if (saved["protocol_sha256"] != sha256(assets / "protocol.json")
                        or saved["reference_manifest_sha256"] != sha256(assets / "manifest.json")):
                    raise ValueError("Existing formal score has a different reference protocol")
                for name, digest in saved["prediction_sha256"].items():
                    if sha256(predictions / name) != digest:
                        raise ValueError("Existing formal benchmark prediction archive changed")
                for name, digest in saved["outputs_sha256"].items():
                    if sha256(destination / name) != digest:
                        raise ValueError("Existing formal scoring output changed")
            else:
                formal_evaluate(predictions, assets / "protocol.json", assets, destination,
                                method=method_name, panel=panel, task="combosciplex")
            formal = pd.read_csv(destination / "per_group_metrics.csv").sort_values(["seed", "group_index"])
            online = pd.DataFrame(rows_by_branch[branch]).sort_values(["seed", "group_index"])
            columns = [column for column in formal if column in HVG_METRICS]
            prefix = "de50_" if panel == "de50" else ""
            # This is an explicit numerical check of the monitor/formal bridge.
            for column in columns:
                if not np.allclose(formal[column], online[prefix + column], atol=1e-10, rtol=1e-10, equal_nan=True):
                    raise ValueError(f"Online/formal metric mismatch: {branch}/{panel}/{column}")
            manifests[branch + "/" + panel] = sha256(marker)
    return manifests


def evaluate(model, runtime, config, step, checkpoint_path, *, seeds=(4243,),
             mean_only=False, rank=0, world=1, purpose="monitor"):
    """Synchronous evaluation; all training ranks must enter together."""
    seeds = tuple(map(int, seeds))
    if not seeds or len(set(seeds)) != len(seeds) or not set(seeds) <= set(SEEDS):
        raise ValueError("Use a nonempty subset of locked sampling seeds 4243--4246")
    if purpose not in {"monitor", "benchmark"} or (purpose == "benchmark" and (mean_only or seeds != SEEDS)):
        raise ValueError("A final Combo benchmark requires full prediction for all four seeds")
    _validate_config(config)
    started = time.monotonic()
    refs, protocol = _references(config, runtime)
    identity_base = _rank_zero_value(
        lambda: _identity(model, runtime, checkpoint_path, step, refs), rank, world)
    method = config.get("reporting", {}).get("method", "DeVFlow Combo MLP + trainable mean + decoder MMD")
    mode = "mean_only_diagnostic" if mean_only else "full_prediction"
    identity_base = {**identity_base, "method": method, "mode": mode}
    root = Path(config["output_dir"]) / ("mean_diagnostics" if mean_only else "evaluations") / f"step_{int(step):06d}"
    root.mkdir(parents=True, exist_ok=True)
    rows_by_branch = {name: [] for name in (("head",) if mean_only else ("final", "raw", "head"))}
    seed_outputs = {}
    with formal_mode(model):
        for seed in seeds:
            bundle = _sampling(runtime, refs, seed)
            identity = {**identity_base, "seed": seed, "row_ids_sha256": bundle["row_ids_sha256"]}
            folder = root / f"seed_{seed}"
            folder.mkdir(parents=True, exist_ok=True)
            cached = _rank_zero_value(lambda: _cache_ok(folder, identity), rank, world)
            if cached:
                if rank == 0:
                    for branch in rows_by_branch:
                        rows_by_branch[branch].extend(pd.read_csv(folder / f"{branch}_per_group_metrics.csv").to_dict("records"))
                    seed_outputs[str(seed)] = dict(path=str(folder.resolve()), reused=True)
                continue
            def predict_local():
                results = []
                for index in range(rank, 5, world):
                    record = bundle["records"][index]
                    with torch.autocast(runtime.device.type, enabled=False):
                        mean = model.predict_mean(runtime.mean_query(record)).detach().float().cpu().numpy().reshape(-1)
                        proxy = np.broadcast_to(mean, (128, 2000)).copy()
                        if mean_only:
                            prediction, raw = proxy, None
                        else:
                            parts = model.predict(runtime.source_batch(record, record["control_rows"]),
                                                  time_points=80, return_parts=True)
                            prediction = parts["prediction"].detach().float().cpu().numpy()
                            raw = parts["raw_expression"].detach().float().cpu().numpy()
                    if (prediction.shape != (128, 2000) or not np.isfinite(prediction).all()
                            or mean.shape != (2000,)):
                        raise FloatingPointError("Invalid Combo mean/scaled expression prediction")
                    values = {"head": _score(proxy, index, bundle, refs, False)}
                    parity = None
                    if not mean_only:
                        if raw.shape != prediction.shape or not np.isfinite(raw).all():
                            raise FloatingPointError("Invalid raw decoder prediction")
                        mean_error = float(np.max(np.abs(prediction.mean(0, dtype=np.float64) - mean)))
                        if not np.allclose(prediction.mean(0, dtype=np.float64), mean, atol=2e-6, rtol=2e-6):
                            raise ValueError("Scaled expression mean differs from the mean head")
                        values["final"] = _score(prediction, index, bundle, refs, True)
                        values["raw"] = _score(raw, index, bundle, refs, True)
                        differences = {key: abs(value - values["final"][key])
                            for key, value in values["head"].items()
                            if np.isfinite(value) and np.isfinite(values["final"][key])}
                        parity = dict(seed=seed, group_index=index, max_expression_mean_error=mean_error,
                                      metric_abs_difference=differences)
                    common = dict(seed=seed, group_index=index, condition=record["condition"],
                                  cell_line="A549", context=str(refs.statistics["context_names"][index]))
                    rows = {branch: {**common, **metrics} for branch, metrics in values.items()}
                    results.append(dict(index=index, prediction=None if mean_only else prediction,
                                        raw=raw, rows=rows, parity=parity))
                return results
            local = _gather_results(predict_local, rank, world)
            def archive_seed():
                local.sort(key=lambda item: item["index"])
                if [item["index"] for item in local] != list(range(5)):
                    raise RuntimeError("Evaluation is missing or duplicating Combo conditions")
                hashes = {}
                for branch in rows_by_branch:
                    rows = [item["rows"][branch] for item in local]
                    path = folder / f"{branch}_per_group_metrics.csv"
                    pd.DataFrame(rows).to_csv(path, index=False)
                    hashes[path.name] = sha256(path)
                    rows_by_branch[branch].extend(rows)
                if not mean_only:
                    for branch, field, directory in (("final", "prediction", "predictions"),
                                                     ("raw", "raw", "raw_predictions")):
                        path = root / directory / f"predictions_seed_{seed}.npz"
                        _write_archive(path, seed, [item[field] for item in local], bundle,
                                       runtime, identity, branch, method)
                        hashes[os.path.relpath(path, folder)] = sha256(path)
                    path = folder / "mean_parity.json"
                    atomic_json(path, _clean([item["parity"] for item in local]))
                    hashes[path.name] = sha256(path)
                atomic_json(folder / "COMPLETED.json", dict(identity=identity, files_sha256=hashes))
                seed_outputs[str(seed)] = dict(path=str(folder.resolve()), reused=False)
                return True
            _rank_zero_value(archive_seed, rank, world)
    def finish():
        branches, by_seed_tables = {}, {}
        for branch, rows in rows_by_branch.items():
            branches[branch], by_seed_tables[branch] = aggregate_rows(rows)
        main_branch = "head" if mean_only else "final"
        aggregate = branches[main_branch]
        protocol_hash = json_sha256(dict(protocol=_FORMAL,
            reference_manifest_sha256=identity_base["reference_manifest_sha256"],
            protocol_file_sha256=identity_base["protocol_file_sha256"],
            runtime_source_sha256=identity_base["runtime_source_sha256"]))
        summary = dict(**identity_base, update=int(step), optimizer_step=int(step), split="external_test",
            seeds=list(seeds), n_conditions=5, cells_per_condition=128, precision="fp32", time_points=80,
            protocol_sha256=protocol_hash, selection_scope="external_test_exploratory", purpose=purpose,
            distribution_metrics="omitted; Flow and decoder not evaluated" if mean_only else "native Combo HVG and fixed DE50 distribution metrics",
            elapsed_seconds=time.monotonic() - started, seed_outputs=seed_outputs,
            branches=branches, **aggregate)
        prefix = "benchmark_" if purpose == "benchmark" else ""
        for branch, rows in rows_by_branch.items():
            pd.DataFrame(rows).to_csv(root / f"{prefix}{branch}_per_group_metrics.csv", index=False)
            by_seed_tables[branch].to_csv(root / f"{prefix}{branch}_per_seed_metrics.csv", index=False)
        pd.DataFrame(rows_by_branch[main_branch]).to_csv(root / f"{prefix}per_group_metrics.csv", index=False)
        by_seed_tables[main_branch].to_csv(root / f"{prefix}per_seed_metrics.csv", index=False)
        row = dict(method=method, setting="pooled", n_groups=5, n_sampling_repeats=len(seeds))
        for metric, value in aggregate["metrics"].items():
            row.update({metric + "_mean": value, metric + "_std": aggregate["metrics_std"][metric],
                        metric + "_n_finite": aggregate["n_finite"][metric]})
        pd.DataFrame([row]).to_csv(root / f"{prefix}summary.csv", index=False)
        atomic_json(root / ("benchmark_summary.json" if purpose == "benchmark" else "summary.json"), _clean(summary))
        if purpose == "monitor" and not mean_only:
            _selection(config["output_dir"])
        if purpose == "benchmark":
            manifests = _formal_benchmark(config, root, rows_by_branch, method)
            final = Path(config["output_dir"]) / "benchmark" / f"step_{int(step):06d}"
            final.mkdir(parents=True, exist_ok=True)
            atomic_json(final / "summary.json", _clean(summary))
            pd.DataFrame([row]).to_csv(final / "summary.csv", index=False)
            by_seed_tables["final"].to_csv(final / "per_seed_metrics.csv", index=False)
            pd.DataFrame(rows_by_branch["final"]).to_csv(final / "per_group_metrics.csv", index=False)
            atomic_json(final / "COMPLETED.json", dict(checkpoint_sha256=summary["checkpoint_sha256"],
                sampling_seeds=list(SEEDS), n_sampling_repeats=4, prediction_directories=seed_outputs,
                protocol_sha256=protocol_hash, online_formal_metric_parity="passed (all metrics <=1e-10 absolute/relative)",
                formal_scorer_manifests_sha256=manifests,
                outputs_sha256={name: sha256(final / name) for name in
                    ("summary.json", "summary.csv", "per_seed_metrics.csv", "per_group_metrics.csv")}))
        return summary
    summary = _rank_zero_value(finish, rank, world)
    return summary if rank == 0 else None


def evaluate_checkpoint(config):
    import os
    from devflow.model_mlp import MLPMeanSystem
    from .combosciplex_mlp import make_runtime
    rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    local = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device("cuda", local) if torch.cuda.is_available() else torch.device("cpu")
    if device.type == "cuda":
        torch.cuda.set_device(local)
    if world > 1:
        dist.init_process_group("nccl" if device.type == "cuda" else "gloo")
    settings = config.get("prediction", {})
    checkpoint = settings.get("checkpoint")
    if not checkpoint:
        raise ValueError("Prediction requires an explicit actual Combo checkpoint")
    try:
        payload = torch.load(checkpoint, map_location="cpu", mmap=True, weights_only=False)
        runtime = make_runtime(config, device)
        model = MLPMeanSystem.from_checkpoint(payload).to(device)
        return evaluate(model, runtime, config, int(payload["update"]), checkpoint,
            seeds=settings.get("seeds", SEEDS), mean_only=settings.get("mean_only", False),
            rank=rank, world=world, purpose=settings.get("purpose", "benchmark"))
    finally:
        if world > 1 and dist.is_initialized():
            dist.destroy_process_group()


def predict(config):
    return evaluate_checkpoint(config)
