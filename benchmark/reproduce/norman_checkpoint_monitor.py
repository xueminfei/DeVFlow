"""Evaluate saved Norman checkpoints with one formal sampling seed for the monitor."""
from __future__ import annotations

import argparse
import copy
import json
import shutil
import time
from pathlib import Path

import numpy as np

from benchmark.metrics import DE50_METRICS, HVG_METRICS, score_group
from benchmark.protocol import EvaluationProtocol
from benchmark.references import EvaluationReferences
from devflow.io import json_sha256, read_prediction_archive, sha256_file, write_json
from devflow.runtime.norman import predict


def _score_one_seed(archive, refs, protocol, seed: int, panel: str):
    values = refs.statistics
    contexts = values["context_names"].astype(str)
    targets, controls = refs.for_seed(seed, archive)
    if protocol.distance_backend_hvg in ("combo_historical", "norman_historical"):
        profiles = np.stack([matrix.mean(0, dtype=np.float64) for matrix in targets])
    else:
        profiles = np.stack([matrix.mean(0) for matrix in targets])
    rows = []
    mask = refs.seed_statistics[seed].get("de_mask", values["de_mask"])
    for index, (pred, target, control) in enumerate(zip(archive.predictions, targets, controls)):
        group = np.flatnonzero(contexts == contexts[index])
        position = int(np.flatnonzero(group == index)[0])
        rows.append(score_group(
            pred, target, control,
            weights=values["continuous_deg_weights"][index],
            training_mean=values["training_condition_mean"][values["reference_index"][index]],
            de_mask=mask[index], same_context_targets=profiles[group], position=position,
            panel=panel,
            gene_indices=values["de50_indices"][index] if panel == "de50" else None,
            distance_backend=(protocol.distance_backend_hvg if panel == "hvg" else "scipy_float64"),
            target_profile=profiles[index] if protocol.response_profile == "full_treated" else None,
        ))
    metrics = HVG_METRICS if panel == "hvg" else DE50_METRICS
    return {name: float(np.nanmean([row[name] for row in rows])) for name in metrics}


def evaluate_checkpoint(base, run, step, seed=4243, references=None):
    """Score one immutable checkpoint with the locked formal sampling protocol."""
    run, step, seed = Path(run).resolve(), int(step), int(seed)
    refs_dir = Path(references or base.get("references", {}).get(
        "output_dir", run.parent / "evaluation_references")).resolve()
    protocol_path = refs_dir / "protocol.json"
    protocol = EvaluationProtocol.load(protocol_path)
    if seed not in protocol.seeds:
        raise ValueError(f"seed {seed} is not in locked protocol {protocol.seeds}")
    refs = EvaluationReferences(refs_dir, protocol)
    evaluation_root = run / "evaluations"
    evaluation_root.mkdir(parents=True, exist_ok=True)
    checkpoint = run / f"checkpoint_step_{step:08d}.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    destination = evaluation_root / f"step_{step:08d}"
    summary_path = destination / "summary.json"
    if summary_path.is_file():
        existing = json.loads(summary_path.read_text())
        if (existing.get("checkpoint_sha256") != sha256_file(checkpoint)
                or existing.get("protocol_sha256") != sha256_file(protocol_path)
                or existing.get("seeds") != [int(seed)]):
            raise ValueError("Existing checkpoint evaluation has a different identity")
        return existing
    destination.mkdir(parents=True, exist_ok=True)
    temporary = run / f".checkpoint_eval_{step:08d}"
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    config = copy.deepcopy(base)
    prediction_settings = config.setdefault("prediction", {})
    prediction_settings.update({
        "checkpoint": str(checkpoint),
        "output_dir": str(temporary / "predictions"),
        "seeds": [int(seed)],
        "overwrite": True,
        "device": "cuda:0",
        "time_points": 80,
        "cells_per_condition": 128,
        "chunk_size": 128,
    })
    started = time.monotonic()
    try:
        result = predict(config)
        archive_path = Path(result["archives"][0])
        archive = read_prediction_archive(archive_path)
        hvg = _score_one_seed(archive, refs, protocol, seed, "hvg")
        de50 = _score_one_seed(archive, refs, protocol, seed, "de50")
        metrics = {**hvg, **{"de50_" + key: value for key, value in de50.items()}}
        elapsed = time.monotonic() - started
        source_protocol_hash = sha256_file(protocol_path)
        item = {
            "update": int(step),
            "optimizer_step": int(step),
            "split": "external_test",
            "mode": "full_prediction",
            "purpose": "monitor",
            "seeds": [int(seed)],
            "n_sampling_repeats": 1,
            "checkpoint_sha256": sha256_file(checkpoint),
            "protocol_sha256": source_protocol_hash,
            "precision": "FP32; TF32 disabled",
            "time_points": 80,
            "n_conditions": int(protocol.condition_count),
            "cells_per_condition": int(protocol.cells_per_condition),
            "elapsed_seconds": float(elapsed),
            "metrics": metrics,
            "metrics_std": {},
            "branches": {"final": {"metrics": metrics, "metrics_std": {}}},
            "protocol": {
                "name": protocol.protocol,
                "task": protocol.task,
                "seed_subset": [int(seed)],
                "source_protocol_sha256": source_protocol_hash,
                "gene_order_sha256": protocol.gene_order_sha256,
                "training_binding_sha256": protocol.training_binding_sha256,
                "distance_backend_hvg": protocol.distance_backend_hvg,
                "response_profile": protocol.response_profile,
            },
            "alignment": {
                "prediction_archive": str(archive_path),
                "prediction_gene_order_sha256": archive.gene_order_sha256,
                "conditions": int(protocol.condition_count),
                "cells_per_condition": int(protocol.cells_per_condition),
                "same_locked_references": True,
                "same_scorer": "benchmark.metrics.score_group",
            },
        }
        write_json(summary_path, item)
        write_json(destination / "evaluation_manifest.json", {
            "status": "complete", "summary": str(summary_path),
            "checkpoint": str(checkpoint), "checkpoint_sha256": item["checkpoint_sha256"],
            "protocol_sha256": source_protocol_hash,
            "seed": int(seed), "elapsed_seconds": float(elapsed),
            "temporary_prediction_deleted": True,
        })
        print(json.dumps({"step": step, "status": "complete", "elapsed_seconds": elapsed}, sort_keys=True), flush=True)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return item


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--steps", type=int, nargs="+", required=True)
    parser.add_argument("--seed", type=int, default=4243)
    parser.add_argument("--references", type=Path)
    args = parser.parse_args()
    base = json.loads(args.config.read_text())
    completed = [evaluate_checkpoint(base, args.run, step, args.seed, args.references)
                 for step in sorted(set(args.steps))]
    refs_dir = Path(args.references or base.get("references", {}).get(
        "output_dir", args.run.resolve().parent / "evaluation_references")).resolve()
    refs = EvaluationReferences(refs_dir, EvaluationProtocol.load(refs_dir / "protocol.json"))
    write_json(args.run / "RETROACTIVE_EVALUATION_COMPLETE.json", {
        "status": "complete", "purpose": "monitor checkpoint trend",
        "seed": int(args.seed), "steps": [item["optimizer_step"] for item in completed],
        "summary_paths": [str(args.run / "evaluations" / f"step_{item['optimizer_step']:08d}" / "summary.json")
                          for item in completed],
        "protocol_sha256": sha256_file(refs_dir / "protocol.json"),
        "references_manifest_sha256": sha256_file(refs.manifest_path),
    })


if __name__ == "__main__":
    main()
