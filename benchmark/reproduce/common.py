from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from devflow.io import json_sha256, read_prediction_archive, sha256_file, write_json
from benchmark.metrics import DE50_METRICS, HVG_METRICS, score_group
from benchmark.protocol import EvaluationProtocol
from benchmark.references import EvaluationReferences


def evaluate(predictions: str | Path, protocol_path: str | Path, references: str | Path,
             output: str | Path, *, method: str = "DeVFlow", panel: str = "hvg", task: str | None = None):
    protocol = EvaluationProtocol.load(protocol_path)
    if task is not None and protocol.task != task:
        raise ValueError(f"Expected task {task}, got {protocol.task}")
    if panel not in ("hvg", "de50"):
        raise ValueError("panel must be hvg or de50")
    refs = EvaluationReferences(references, protocol)
    if panel == "de50" and "de50_indices" not in refs.statistics:
        raise ValueError("The locked reference bundle has no DE50 panel")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for name in ("COMPLETED.json", "summary.csv", "per_group_metrics.csv"):
        if (output / name).exists():
            raise FileExistsError(f"Refusing to overwrite prior scoring output: {output / name}")
    values = refs.statistics
    contexts = values["context_names"].astype(str)
    rows, checks, input_hashes = [], [], {}
    for seed in protocol.seeds:
        path = Path(predictions) / f"predictions_seed_{seed}.npz"
        archive = read_prediction_archive(path)
        expected_binding = protocol.training_binding_sha256 or refs.training_binding_sha256
        if expected_binding:
            binding = archive.metadata.get("training_binding_sha256")
            if binding is None and "training_binding" in archive.metadata:
                binding = json_sha256(archive.metadata["training_binding"])
            if binding != expected_binding:
                raise ValueError("Prediction training-split binding differs from the protocol")
        targets, controls = refs.for_seed(seed, archive)
        masks = refs.seed_statistics[seed].get("de_mask", values["de_mask"])
        profiles = (values["full_target_profiles"] if protocol.response_profile == "full_treated"
                    else np.stack([matrix.mean(0, dtype=np.float64) if protocol.distance_backend_hvg in ("combo_historical", "norman_historical") else matrix.mean(0) for matrix in targets]))
        for index, (pred, target, control) in enumerate(zip(archive.predictions, targets, controls)):
            group = np.flatnonzero(contexts == contexts[index])
            position = int(np.flatnonzero(group == index)[0])
            score = score_group(pred, target, control,
                                weights=values["continuous_deg_weights"][index],
                                training_mean=values["training_condition_mean"][values["reference_index"][index]],
                                de_mask=masks[index], same_context_targets=profiles[group],
                                position=position, panel=panel,
                                gene_indices=values["de50_indices"][index] if panel == "de50" else None,
                                distance_backend=protocol.distance_backend_hvg if panel == "hvg" else "scipy_float64",
                                target_profile=profiles[index] if protocol.response_profile == "full_treated" else None)
            rows.append(dict(method=method, seed=seed, group_index=index, condition=archive.condition_names[index],
                             context=contexts[index],
                             scope=str(values["reporting_scopes"][index]) if "reporting_scopes" in values else "",
                             n_prediction_cells=len(pred), n_target_cells=int(values["full_target_counts"][index]) if "full_target_counts" in values else len(target),
                             n_control_cells=len(control), **score))
        digest = sha256_file(path)
        input_hashes[path.name] = digest
        checks.append(dict(seed=seed, conditions=len(targets), source_ids_exact=True, target_ids_exact=True,
                           gene_order_exact=True, prediction_sha256=digest))
    frame = pd.DataFrame(rows)
    metrics = HVG_METRICS if panel == "hvg" else DE50_METRICS
    repeated, summary = [], []
    settings = [("pooled", frame)] + [(context, frame[frame.context == context]) for context in dict.fromkeys(contexts)]
    if "reporting_scopes" in values:
        settings += [(scope, frame[frame.scope == scope]) for scope in dict.fromkeys(frame.scope) if scope]
    for setting, subset in settings:
        by_seed = subset.groupby("seed", sort=False)[list(metrics)].mean()
        for seed, row in by_seed.iterrows():
            repeated.append(dict(method=method, setting=setting, seed=int(seed),
                                 n_groups=int(subset.group_index.nunique()), **row.to_dict()))
        record = dict(method=method, setting=setting, n_groups=int(subset.group_index.nunique()),
                      n_sampling_repeats=len(by_seed))
        for metric in metrics:
            record.update({metric + "_mean": float(by_seed[metric].mean()),
                           metric + "_std": float(by_seed[metric].std(ddof=1)),
                           metric + "_n_finite": int(np.isfinite(by_seed[metric]).sum())})
        summary.append(record)
    frame.to_csv(output / "per_group_metrics.csv", index=False)
    pd.DataFrame(repeated).to_csv(output / "per_seed_metrics.csv", index=False)
    pd.DataFrame(summary).to_csv(output / "summary.csv", index=False)
    manifest = dict(status="complete", method=method, panel=panel, protocol=asdict(protocol),
                    protocol_sha256=sha256_file(protocol_path), reference_manifest_sha256=sha256_file(refs.manifest_path),
                    prediction_sha256=input_hashes, alignment=checks, n_rows=len(frame),
                    aggregation="condition macro-mean within seed; mean and sample SD(ddof=1) across seeds",
                    uncertainty="sampling repeats for one fixed model, not independent retraining",
                    training_or_inference_run=False,
                    outputs_sha256={name: sha256_file(output / name) for name in
                                    ("per_group_metrics.csv", "per_seed_metrics.csv", "summary.csv")})
    write_json(output / "benchmark_manifest.json", manifest)
    write_json(output / "COMPLETED.json", manifest)
    return frame, pd.DataFrame(summary)


def main(task: str):
    parser = argparse.ArgumentParser(description=f"Score {task} predictions without training or inference")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--method", default="DeVFlow")
    parser.add_argument("--panel", choices=("hvg", "de50"), default="hvg")
    args = parser.parse_args()
    _, summary = evaluate(args.predictions, args.protocol, args.references, args.output,
                          method=args.method, panel=args.panel, task=task)
    print(summary[summary.setting == "pooled"].to_string(index=False))
