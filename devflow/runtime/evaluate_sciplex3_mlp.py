"""Locked SciPlex3 evaluation for the MLP and trainable mean model.

Monitoring and final reporting use the same prediction and scoring code. A
monitor may request a subset of the four *sampling* seeds, but never a subset
of conditions/cells or a cheaper integrator. Mean diagnostics intentionally
skip the flow and omit all distribution metrics.
"""
from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist

from benchmark.metrics import HVG_METRICS, score_condition
from devflow.data.sciplex3 import CELL_LINES, SEEDS
from devflow.io import json_sha256, read_prediction_archive, require_alignment
from .artifacts import atomic_json, sha256

_DISTRIBUTION = {"endpoint_mmd", "emd", "e_distance"}
_FORMAL = dict(cells=128, conditions=108, genes=2000, time_points=80,
               rk4_intervals=79, precision="fp32", allow_tf32=False,
               expression="log1p-CP1000-HVG2000", response_profile="sampled_treated",
               mmd="treated_only_lower_median_four_kernel_U_statistic",
               aggregation="condition macro mean then sampling-seed mean/sample SD")


def _clean(value):
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_clean(v) for v in value]
    if isinstance(value, np.generic):
        return _clean(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


@contextmanager
def formal_mode(model):
    """Do not change the training RNG, dropout mode, or TF32 policy."""
    modes = [(module, module.training) for module in model.modules()]
    cpu_rng, numpy_rng, python_rng = torch.get_rng_state(), np.random.get_state(), random.getstate()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    matmul, cudnn = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        model.eval()
        with torch.no_grad():
            yield
    finally:
        for module, training in modes:
            module.training = training
        torch.backends.cuda.matmul.allow_tf32 = matmul
        torch.backends.cudnn.allow_tf32 = cudnn
        torch.set_rng_state(cpu_rng)
        np.random.set_state(numpy_rng)
        random.setstate(python_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)


def _checkpoint_identity(model, checkpoint_path, step, runtime=None):
    path = Path(checkpoint_path)
    payload = torch.load(str(path), map_location="cpu", mmap=True, weights_only=False)
    if int(payload["update"]) != int(step):
        raise ValueError("Requested evaluation step differs from the actual checkpoint")
    if runtime is not None:
        if payload.get("training_binding") != runtime.data.training_binding():
            raise ValueError("Checkpoint training binding differs from the evaluation data")
        if payload.get("asset_manifest_sha256") != getattr(runtime, "asset_manifest_sha256", None):
            raise ValueError("Checkpoint feature/mean assets differ from the evaluation runtime")
    saved, current = payload["model"], model.state_dict()
    if set(saved) != set(current):
        raise ValueError("In-memory model and checkpoint have different state keys")
    for name, value in current.items():
        if value.dtype != saved[name].dtype or value.shape != saved[name].shape or not torch.equal(value.detach().cpu(), saved[name].detach().cpu()):
            raise ValueError("In-memory model differs from the saved checkpoint: " + name)
    return dict(checkpoint=str(path.resolve()), checkpoint_sha256=sha256(path),
                checkpoint_update=int(step), model_matches_checkpoint=True,
                checkpoint_purpose=payload.get("purpose", "training_checkpoint"),
                checkpoint_run_purpose=payload.get("config", {}).get("purpose"))


def _method_name(config, checkpoint):
    """Keep historical replays distinguishable in archives and table exports."""
    default = ("MAP-Flow + mixed-kernel mean head (historical replay)"
               if checkpoint.get("checkpoint_purpose") == "historical_replay_only"
               else "DeVFlow MLP + trainable mixed-kernel mean")
    value = config.get("reporting", {}).get("method", default)
    if not isinstance(value, str) or not value.strip():
        raise ValueError("reporting.method must be a nonempty method name")
    return value.strip()


def _references(config, data):
    """Load reporting inputs only from an explicitly configured evaluation path."""
    path = Path(config["data"]["evaluation_references"])
    key = (str(path.resolve()), str(config["data"].get("de50_references", "")))
    cache = getattr(data, "_mlp_evaluation_reference_cache", {})
    if key in cache:
        return cache[key]
    portable = path.is_dir()
    stats_path = path / "statistics.npz" if portable else path
    if portable:
        manifest = json.loads((path / "manifest.json").read_text())
        if manifest.get("purpose") != "evaluation_only":
            raise ValueError("Reference bundle must declare purpose=evaluation_only")
        if manifest.get("evaluation_statistics", {}).get("use") != "reporting_only":
            raise ValueError("Evaluation-derived statistics must be reporting_only")
        for name, digest in manifest.get("files_sha256", {}).items():
            if name == "statistics.npz" or name.startswith("references_seed_"):
                if sha256(path / name) != digest:
                    raise ValueError("Reference file hash changed: " + name)
        binding = manifest.get("training_statistics", {}).get("training_binding")
        if binding is not None and binding != data.training_binding():
            raise ValueError("Evaluation reference training binding differs")
    else:
        manifest_path = path.parent / "manifest.json"
        if not manifest_path.exists():
            raise ValueError("Legacy evaluation references require their adjacent provenance manifest")
        manifest = json.loads(manifest_path.read_text())
        if "evaluation only" not in manifest.get("scope", "").lower():
            raise ValueError("Legacy reference manifest must identify evaluation-only inputs")
        if manifest.get("training_binding") != data.training_binding():
            raise ValueError("Legacy evaluation reference training binding differs")
        if manifest.get("reference_sha256") != sha256(stats_path):
            raise ValueError("Legacy evaluation reference hash differs")
    with np.load(stats_path, allow_pickle=False) as values:
        stats = {k: np.array(values[k]) for k in values.files}
    names = [r["condition"] for r in data.test_records]
    if stats["condition_names"].tolist() != names:
        raise ValueError("Evaluation reference condition order differs")
    if str(stats["gene_order_sha256"]) != data.gene_order_sha256:
        raise ValueError("Evaluation reference gene axis differs")
    for field in ("continuous_deg_weights", "de_mask"):
        if stats[field].shape != (108, 2000):
            raise ValueError("Unexpected reference shape: " + field)
    if not np.isfinite(stats["continuous_deg_weights"]).all() or np.any(stats["continuous_deg_weights"] < 0):
        raise ValueError("Reference DEG weights must be finite and nonnegative")
    if stats["training_condition_mean"].ndim != 2 or stats["training_condition_mean"].shape[1] != 2000 or not np.isfinite(stats["training_condition_mean"]).all():
        raise ValueError("Invalid training-condition reference profiles")
    digests = dict(statistics_sha256=sha256(stats_path))
    if portable:
        digests["reference_manifest_sha256"] = sha256(path / "manifest.json")
    else:
        digests["reference_manifest_sha256"] = sha256(path.parent / "manifest.json")
    panel = stats.get("de50_indices")
    panel_path = config["data"].get("de50_references")
    if panel_path:
        with np.load(panel_path, allow_pickle=False) as de:
            if de["group_names"].tolist() != names:
                raise ValueError("DE50 condition order differs")
            if not np.array_equal(de["full_weights"], stats["continuous_deg_weights"]):
                raise ValueError("DE50 weights differ from HVG references")
            panel = np.array(de["gene_indices"])
        digests["de50_sha256"] = sha256(panel_path)
    if panel is not None:
        if panel.shape != (108, 50) or panel.dtype.kind not in "iu" or np.any(panel < 0) or np.any(panel >= 2000):
            raise ValueError("Invalid fixed DE50 gene panel")
        if any(len(np.unique(row)) != 50 for row in panel):
            raise ValueError("Duplicate genes in a DE50 panel")
    result = dict(statistics=stats, de50=panel, digests=digests,
                  directory=path if portable else None, samples={})
    cache[key] = result
    data._mlp_evaluation_reference_cache = cache
    return result


def _sample_bundle(data, references, seed):
    if seed in references["samples"]:
        return references["samples"][seed]
    records = data.sampling(seed)
    if len(records) != 108 or any(len(r["control_rows"]) != 128 or len(r["target_rows"]) != 128 for r in records):
        raise ValueError("Formal sampling requires every one of the 108 conditions and 128 cells")
    targets = [data.expression(r["cell_line"], r["target_rows"]) for r in records]
    controls = [data.expression(r["cell_line"], r["control_rows"]) for r in records]
    source_ids = np.stack([data.global_ids(r["cell_line"], r["control_rows"]) for r in records])
    target_ids = np.stack([data.global_ids(r["cell_line"], r["target_rows"]) for r in records])
    masks = references["statistics"]["de_mask"]
    directory = references["directory"]
    if directory:
        with np.load(directory / f"references_seed_{seed}.npz", allow_pickle=False) as frozen:
            if int(frozen["seed"]) != seed or frozen["condition_names"].tolist() != [r["condition"] for r in records]:
                raise ValueError("Portable reference sampling identity differs")
            if not np.array_equal(frozen["source_global_ids"], source_ids) or not np.array_equal(frozen["target_global_ids"], target_ids):
                raise ValueError("Portable references use different sampled cells")
            if str(frozen["gene_order_sha256"]) != data.gene_order_sha256:
                raise ValueError("Portable sampled reference gene axis differs")
            for index, (target, control) in enumerate(zip(targets, controls)):
                if not np.array_equal(target, frozen[f"target_{index:04d}"]) or not np.array_equal(control, frozen[f"control_{index:04d}"]):
                    raise ValueError("Expression values differ from the locked reference bundle")
            if "de_mask" in frozen:
                masks = np.array(frozen["de_mask"])
    bundle = dict(records=records, targets=targets, controls=controls,
                  profiles=np.stack([x.mean(0) for x in targets]), masks=masks,
                  source_ids=source_ids, target_ids=target_ids,
                  row_ids_sha256=json_sha256(dict(source=source_ids.tolist(), target=target_ids.tolist())))
    references["samples"][seed] = bundle
    return bundle


def _score(prediction, index, bundle, references, include_distribution):
    records = bundle["records"]
    record = records[index]
    positions = [i for i, r in enumerate(records) if r["cell_line"] == record["cell_line"]]
    stats = references["statistics"]
    ref_index = int(stats["reference_index"][index]) if "reference_index" in stats else CELL_LINES.index(record["cell_line"])
    args = (prediction, bundle["targets"][index], bundle["controls"][index],
            stats["continuous_deg_weights"][index], bundle["masks"][index],
            stats["training_condition_mean"][ref_index], bundle["profiles"][positions], positions.index(index))
    values = score_condition(*args, include_distribution=include_distribution)
    if references["de50"] is not None:
        panel = score_condition(*args, include_distribution=include_distribution,
                                panel="de50", gene_indices=references["de50"][index])
        values.update({"de50_" + k: v for k, v in panel.items()})
    return values


def _metric_columns(frame):
    return [key for key in frame.columns if key in HVG_METRICS or key.startswith("de50_")]


def aggregate_rows(rows):
    """Equal condition weights, then equal sampling-repeat weights (ddof=1)."""
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError("Cannot aggregate empty evaluation rows")
    metrics = _metric_columns(frame)
    by_seed = frame.groupby("seed", sort=True)[metrics].mean()
    result = dict(metrics={}, metrics_std={}, n_finite={}, n_sampling_repeats=len(by_seed))
    for name in metrics:
        values = by_seed[name].to_numpy(dtype=np.float64)
        values = values[np.isfinite(values)]
        result["metrics"][name] = float(values.mean()) if len(values) else None
        result["metrics_std"][name] = float(values.std(ddof=1)) if len(values) > 1 else None
        result["n_finite"][name] = len(values)
    return _clean(result), by_seed.reset_index()


def _cache_ok(folder, identity):
    marker = Path(folder) / "COMPLETED.json"
    if not marker.exists():
        return False
    saved = json.loads(marker.read_text())
    if saved["identity"] != identity:
        raise ValueError("Existing evaluation belongs to a different model or protocol: " + str(folder))
    for name, digest in saved["files_sha256"].items():
        if sha256(Path(folder) / name) != digest:
            raise ValueError("Completed evaluation output changed: " + name)
    return True


def _scoring_code_identity():
    import benchmark.metrics.sciplex3 as scorer
    package = Path(__file__).resolve().parents[1]
    source = {str(path.relative_to(package)): sha256(path) for path in sorted(package.rglob("*.py"))}
    return dict(evaluator_sha256=sha256(__file__), scorer_sha256=sha256(scorer.__file__),
                runtime_source_sha256=json_sha256(source))


def _rank_zero_value(action, rank, world):
    """Broadcast failures too, so other ranks do not wait after a rank-zero error."""
    packet = None
    if rank == 0:
        try:
            packet = dict(value=action(), error=None)
        except Exception as exc:
            packet = dict(value=None, error=f"{type(exc).__name__}: {exc}")
    if world > 1:
        values = [packet]
        dist.broadcast_object_list(values, src=0)
        packet = values[0]
    if packet["error"] is not None:
        raise ValueError(packet["error"])
    return packet["value"]


def _prediction_for_record(model, runtime, record, *, mean_only):
    """The cheap diagnostic never evaluates velocity or the decoder."""
    device_type = next(model.parameters()).device.type
    with torch.autocast(device_type, enabled=False):
        mean = model.predict_mean(runtime.mean_query(record)).detach().double().cpu().numpy().reshape(-1)
        proxy = np.broadcast_to(mean.astype(np.float32), (len(record["control_rows"]), len(mean))).copy()
        prediction = proxy if mean_only else model.predict(
            runtime.source_batch(record, record["control_rows"]), time_points=80).detach().float().cpu().numpy()
    return prediction, mean, proxy


def _validate_numerical_settings(config):
    for section in ("evaluation", "prediction"):
        settings = config.get(section, {})
        if int(settings.get("time_points", 80)) != 80 or int(settings.get("cells", 128)) != 128:
            raise ValueError("MLP evaluation keeps RK4-80 and 128 sampled cells in every condition")
        if str(settings.get("precision", "fp32")).lower() != "fp32" or settings.get("allow_tf32", False):
            raise ValueError("MLP evaluation requires FP32 with TF32 disabled")
    if int(config.get("prediction", {}).get("chunk", 128)) != 128:
        raise ValueError("This evaluator composes the complete 128-cell condition together; chunk must be 128")


def _per_metric_selection(output):
    """Expose candidates without silently inventing an overall best metric."""
    records = []
    for path in sorted((Path(output) / "evaluations").glob("step_*/summary.json")):
        item = json.loads(path.read_text())
        if item.get("seeds") == [4243] and item.get("mode") == "full_prediction":
            records.append(item)
    minimize = {"pseudobulk_mae", "pseudobulk_mse", "weighted_mse", "endpoint_mmd", "emd", "e_distance", "legacy_wmse"}
    by_protocol = {}
    for protocol in dict.fromkeys(r["protocol_sha256"] for r in records):
        matching = [r for r in records if r["protocol_sha256"] == protocol]
        best = {}
        for metric in HVG_METRICS:
            candidates = [r for r in matching if r["metrics"].get(metric) is not None]
            if candidates:
                chosen = min(candidates, key=lambda r: ((1 if metric in minimize else -1) * r["metrics"][metric], r["update"]))
                best[metric] = dict(update=chosen["update"], value=chosen["metrics"][metric],
                                    checkpoint=chosen["checkpoint"], checkpoint_sha256=chosen["checkpoint_sha256"])
        by_protocol[protocol] = dict(candidate_updates=[r["update"] for r in matching], best_per_metric=best)
    atomic_json(Path(output) / "checkpoint_candidates.json", dict(
        selection_scope="external_test_exploratory", selected_checkpoint=None,
        note="Per-metric candidates only; an overall selection criterion has not been declared.",
        candidate_updates=[r["update"] for r in records], protocol_groups=by_protocol,
        best_per_metric=next(iter(by_protocol.values()))["best_per_metric"] if len(by_protocol) == 1 else {}))


def evaluate(model, runtime, config, step, checkpoint_path, *, seeds=(4243,),
             mean_only=False, rank=0, world=1, purpose="monitor"):
    """All ranks call together; rank zero returns a complete summary.

    Existing complete seeds are reused only when the checkpoint, input assets,
    exact sampled IDs, scoring code and numerical protocol all match. A final
    call with all four seeds can therefore add only the three missing repeats.
    """
    seeds = tuple(int(seed) for seed in seeds)
    if not seeds or len(set(seeds)) != len(seeds) or any(s not in SEEDS for s in seeds):
        raise ValueError("Use a nonempty subset of the four locked sampling seeds")
    if mean_only and seeds != SEEDS:
        raise ValueError("Mean diagnostics use all four locked sampling seeds")
    if purpose not in {"monitor", "benchmark"} or (purpose == "benchmark" and (mean_only or seeds != SEEDS)):
        raise ValueError("A final benchmark requires full predictions for all four seeds")
    _validate_numerical_settings(config)
    started = time.monotonic()
    checkpoint = _rank_zero_value(lambda: _checkpoint_identity(model, checkpoint_path, step, runtime), rank, world)
    method = _method_name(config, checkpoint)
    references = _references(config, runtime.data)
    root = Path(config["output_dir"]) / ("mean_diagnostics" if mean_only else "evaluations") / f"step_{int(step):06d}"
    root.mkdir(parents=True, exist_ok=True)
    identity_base = dict(**checkpoint, method=method, protocol=_FORMAL,
        training_binding=runtime.data.training_binding(), gene_order_sha256=runtime.data.gene_order_sha256,
        sampling_manifest_sha256=sha256(runtime.data.sampling_manifest_path),
        asset_manifest_sha256=getattr(runtime, "asset_manifest_sha256", None),
        mode="mean_only_diagnostic" if mean_only else "full_prediction",
        **references["digests"], **_scoring_code_identity())
    all_rows, seed_outputs, parity_rows = [], {}, []
    with formal_mode(model):
        for seed in seeds:
            bundle = _sample_bundle(runtime.data, references, seed)
            identity = dict(identity_base, seed=seed, row_ids_sha256=bundle["row_ids_sha256"])
            folder = root / f"seed_{seed}"
            folder.mkdir(parents=True, exist_ok=True)
            cached = _rank_zero_value(lambda: _cache_ok(folder, identity), rank, world)
            if cached:
                if rank == 0:
                    all_rows.extend(pd.read_csv(folder / "per_group_metrics.csv").to_dict("records"))
                    seed_outputs[str(seed)] = dict(path=str(folder.resolve()), reused=True)
                continue
            local = []
            for index in range(rank, 108, world):
                record = bundle["records"][index]
                # Deliberately keep all 128 cells together: chunk-local
                # rescaling would change the scientific model.
                prediction, mean, proxy = _prediction_for_record(model, runtime, record, mean_only=mean_only)
                if mean.shape != (2000,):
                    raise ValueError("Mean prediction has a wrong gene axis")
                if prediction.shape != (128, 2000) or not np.isfinite(prediction).all():
                    raise FloatingPointError("Invalid final expression prediction")
                values = _score(prediction, index, bundle, references, not mean_only)
                row = dict(seed=seed, group_index=index, condition=record["condition"],
                           cell_line=record["cell_line"], **values)
                parity = None
                if not mean_only:
                    mean_error = float(np.max(np.abs(prediction.mean(0, dtype=np.float64) - mean)))
                    if not np.allclose(prediction.mean(0, dtype=np.float64), mean, atol=2e-6, rtol=2e-6):
                        raise ValueError("Final prediction mean does not match the mean head")
                    proxy_values = _score(proxy, index, bundle, references, False)
                    differences = {k: abs(v - values[k]) for k, v in proxy_values.items()
                                   if np.isfinite(v) and np.isfinite(values[k])}
                    parity = dict(seed=seed, group_index=index, max_expression_mean_error=mean_error,
                                  metric_abs_difference=differences)
                local.append((index, prediction if not mean_only else None, row, parity))
                atomic_json(folder / f"rank_{rank:02d}_progress.json", dict(
                    update=step, seed=seed, completed=len(local), total=len(range(rank, 108, world)),
                    last_group=index, mode=identity_base["mode"]))
            if world > 1:
                gathered = [None] * world
                dist.all_gather_object(gathered, local)
                local = [item for shard in gathered for item in shard]
            if rank == 0:
                local.sort(key=lambda item: item[0])
                if [x[0] for x in local] != list(range(108)):
                    raise RuntimeError("Evaluation is missing or duplicating conditions")
                rows = [x[2] for x in local]
                pd.DataFrame(rows).to_csv(folder / "per_group_metrics.csv", index=False)
                outputs = {"per_group_metrics.csv": sha256(folder / "per_group_metrics.csv")}
                if not mean_only:
                    archive_path = runtime.data.save_predictions(folder, seed, [x[1] for x in local],
                        method, identity)
                    archive = read_prediction_archive(archive_path)
                    require_alignment(archive, seed=seed, condition_names=[r["condition"] for r in bundle["records"]],
                        source_global_ids=bundle["source_ids"], target_global_ids=bundle["target_ids"],
                        gene_order_sha256=runtime.data.gene_order_sha256, n_genes=2000)
                    outputs[archive_path.name] = sha256(archive_path)
                    parity_rows.extend(x[3] for x in local)
                    atomic_json(folder / "mean_parity.json", _clean([x[3] for x in local]))
                    outputs["mean_parity.json"] = sha256(folder / "mean_parity.json")
                atomic_json(folder / "COMPLETED.json", dict(identity=identity, files_sha256=outputs))
                all_rows.extend(rows)
                seed_outputs[str(seed)] = dict(path=str(folder.resolve()), reused=False)
            if world > 1:
                dist.barrier()
    if rank != 0:
        return None
    aggregate, by_seed = aggregate_rows(all_rows)
    summary = dict(**checkpoint, method=method, update=int(step), optimizer_step=int(step), split="external_test",
        mode=identity_base["mode"], seeds=list(seeds), n_conditions=108, cells_per_condition=128,
        precision="fp32", time_points=80, chunk=128, protocol=_FORMAL,
        protocol_sha256=json_sha256(dict(protocol=_FORMAL, **references["digests"],
                                         sampling_manifest_sha256=identity_base["sampling_manifest_sha256"],
                                         runtime_source_sha256=identity_base["runtime_source_sha256"],
                                         scorer_sha256=identity_base["scorer_sha256"])),
        selection_scope="external_test_exploratory", purpose=purpose,
        distribution_metrics="omitted; mean diagnostic does not run flow" if mean_only else "MMD, EMD and target-PCA energy computed",
        elapsed_seconds=time.monotonic() - started, seed_outputs=seed_outputs, **aggregate)
    prefix = "benchmark_" if purpose == "benchmark" else ""
    pd.DataFrame(all_rows).to_csv(root / (prefix + "per_group_metrics.csv"), index=False)
    by_seed.to_csv(root / (prefix + "per_seed_metrics.csv"), index=False)
    summary_row = dict(method=method, checkpoint_purpose=checkpoint["checkpoint_purpose"], setting="pooled",
                       n_groups=108, n_sampling_repeats=len(seeds))
    for name, value in aggregate["metrics"].items():
        summary_row.update({name + "_mean": value, name + "_std": aggregate["metrics_std"][name],
                            name + "_n_finite": aggregate["n_finite"][name]})
    pd.DataFrame([summary_row]).to_csv(root / (prefix + "summary.csv"), index=False)
    # Keep the fixed-seed monitor series intact when a later final benchmark
    # expands the same step to four repeats.
    summary_name = "summary.json" if purpose == "monitor" else "benchmark_summary.json"
    atomic_json(root / summary_name, _clean(summary))
    if purpose == "benchmark":
        final = Path(config["output_dir"]) / "benchmark" / f"step_{int(step):06d}"
        final.mkdir(parents=True, exist_ok=True)
        atomic_json(final / "summary.json", _clean(summary))
        pd.DataFrame([summary_row]).to_csv(final / "summary.csv", index=False)
        by_seed.to_csv(final / "per_seed_metrics.csv", index=False)
        pd.DataFrame(all_rows).to_csv(final / "per_group_metrics.csv", index=False)
        atomic_json(final / "COMPLETED.json", dict(checkpoint_sha256=checkpoint["checkpoint_sha256"],
            seeds=list(seeds), prediction_directories=seed_outputs,
            summary_sha256=sha256(final / "summary.json"), protocol_sha256=summary["protocol_sha256"],
            outputs_sha256={name: sha256(final / name) for name in
                            ("summary.json", "summary.csv", "per_seed_metrics.csv", "per_group_metrics.csv")}))
    _per_metric_selection(config["output_dir"])
    return summary


def evaluate_checkpoint(config):
    import os
    from devflow.model_mlp import MLPMeanSystem
    from .sciplex3_mlp import make_runtime
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
        raise ValueError("Prediction requires an explicit actual checkpoint path")
    payload = torch.load(checkpoint, map_location="cpu", mmap=True, weights_only=False)
    runtime = make_runtime(config, device)
    if payload.get("training_binding") != runtime.data.training_binding():
        raise ValueError("Checkpoint training binding differs from the active runtime")
    if payload.get("asset_manifest_sha256") != getattr(runtime, "asset_manifest_sha256", None):
        raise ValueError("Checkpoint feature/mean assets differ from the active runtime")
    model = MLPMeanSystem.from_checkpoint(payload).to(device)
    try:
        return evaluate(model, runtime, config, int(payload["update"]), checkpoint,
                        seeds=settings.get("seeds", SEEDS), mean_only=settings.get("mean_only", False),
                        rank=rank, world=world, purpose=settings.get("purpose", "benchmark"))
    finally:
        if world > 1:
            dist.destroy_process_group()


def predict(config):
    """The public prediction entry also scores the archived predictions."""
    return evaluate_checkpoint(config)
