"""Norman benchmark entry and training-only kernel mean-head experiment."""
from __future__ import annotations

from .common import main


def _score_mean_archive(archive, refs, protocol):
    """Use the locked sampled-128 scorer, excluding cell-distribution metrics."""
    import numpy as np
    from benchmark.metrics import score_group

    values = refs.statistics
    targets, controls = refs.for_seed(archive.seed, archive)
    profiles = np.stack([x.mean(0, dtype=np.float64) for x in targets])
    contexts = values["context_names"].astype(str)
    masks = refs.seed_statistics[archive.seed].get("de_mask", values["de_mask"])
    rows = []
    for panel in ("hvg", "de50"):
        for i, (pred, target, control) in enumerate(zip(archive.predictions, targets, controls)):
            group = np.flatnonzero(contexts == contexts[i])
            metrics = score_group(
                pred, target, control, weights=values["continuous_deg_weights"][i],
                training_mean=values["training_condition_mean"][values["reference_index"][i]],
                de_mask=masks[i], same_context_targets=profiles[group],
                position=int(np.flatnonzero(group == i)[0]), panel=panel,
                gene_indices=values["de50_indices"][i] if panel == "de50" else None,
                distance_backend=protocol.distance_backend_hvg if panel == "hvg" else "scipy_float64",
                include_distribution=False,
            )
            rows.append(dict(seed=archive.seed, panel=panel, condition=archive.condition_names[i], **metrics))
    return rows


def _aggregate_mean_rows(rows, method):
    import numpy as np

    per_seed, summary = [], []
    for panel in ("hvg", "de50"):
        selected = [row for row in rows if row["panel"] == panel]
        metrics = sorted(set(selected[0]) - {"seed", "panel", "condition"})
        for seed in sorted({row["seed"] for row in selected}):
            items = [row for row in selected if row["seed"] == seed]
            per_seed.append(dict(method=method, panel=panel, seed=seed,
                **{name: float(np.nanmean([row[name] for row in items])) for name in metrics}))
        for name in metrics:
            values = np.asarray([row[name] for row in per_seed if row["panel"] == panel])
            summary.append(dict(method=method, panel=panel, metric=name,
                mean=float(np.nanmean(values)), std=float(np.nanstd(values, ddof=1))))
    return per_seed, summary


def _write_rows(path, rows):
    import csv

    if not rows:
        return
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def mean_head_main(argv=None):
    """Fit A from zero; external references are loaded only after fitting ends."""
    import argparse
    import copy
    import json
    import shutil
    import time
    from pathlib import Path

    import numpy as np
    import torch

    from benchmark.protocol import EvaluationProtocol
    from benchmark.references import EvaluationReferences
    from devflow.data.norman import prepare_kernel_mean_assets, paired_gene_features
    from devflow.io import (PredictionArchive, json_sha256, read_prediction_archive,
                            sha256_file, write_json)
    from devflow.mean.head import TrainableKernelMeanHead
    from devflow.mean.kernels import ContinuousPairKernelBank
    from devflow.runtime.norman import load_inputs, training_binding
    from devflow.runtime.monitor import build_monitor

    parser = argparse.ArgumentParser(description=mean_head_main.__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--references", type=Path,
                        default=Path("outputs/norman19/additive/evaluation_references"))
    parser.add_argument("--min-steps", type=int, default=2000)
    parser.add_argument("--max-steps", type=int, default=50000)
    parser.add_argument("--check-every", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--fit-tolerance", type=float, default=1e-6)
    parser.add_argument("--objective-tolerance", type=float, default=1e-9)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--target-se-assets", type=Path,
                        help="Frozen training-control target gene tokens; overrides data.target_se_assets")
    args = parser.parse_args(argv)
    if not 0 < args.min_steps <= args.max_steps or args.check_every < 1:
        raise ValueError("Invalid training budget")
    if min(args.learning_rate, args.fit_tolerance, args.objective_tolerance, args.threads) <= 0:
        raise ValueError("Learning rate, tolerances and thread count must be positive")
    root = args.output.expanduser().resolve()
    if root.exists() and any(path.name != "launcher.log" for path in root.iterdir()):
        raise FileExistsError("Use a new experiment directory: " + str(root))
    root.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    torch.manual_seed(42)
    np.random.seed(42)
    base = json.loads(args.config.read_text())
    if args.target_se_assets:
        base["data"]["target_se_assets"] = str(args.target_se_assets.resolve())
    config = copy.deepcopy(base)
    config.update(run_mode="mean_head_only", model_family="mlp_mean", world_size=1,
        train_conditions=189, conditions_per_update=189, start_update=0,
        final_update=args.max_steps, output_dir=str(root),
        purpose="CPU kernel mean head only; no Flow or decoder training; external evaluation after training-only stopping")
    config["model"] = {"family": "mlp_mean", "mean_head_kind": "kernel"}
    config["training"] = dict(max_updates=args.max_steps, min_updates=args.min_steps,
        learning_rates={"head": args.learning_rate}, gradient_clip={"head": 1.0},
        gradient_accumulation=1, precision="float64", world_size=1,
        check_every=args.check_every, fit_tolerance=args.fit_tolerance,
        objective_tolerance=args.objective_tolerance, seed=42)
    config["selection"] = dict(protocol="training_convergence", external_used_for_selection=False,
        primary_metric="training_prediction_max_abs_error_to_closed_form",
        stopping="min_updates then both training-fit/objective tolerances; hard cap max_updates")
    config["evaluation"] = dict(monitor_seeds=[4243, 4244, 4245, 4246],
        mean_every=0, full_every=0, references=str(args.references.resolve()),
        distribution_metrics_computed=False, external_used_during_fitting=False)
    config["mean_head"] = dict(cv_folds=8, cv_seed=42,
        primary_mix_grid=[0., .25, .5, .75, 1.], gamma_factor_grid=[.125, .25, .5, 1., 2., 4.],
        alpha_grid=[1e-5, 3e-5, 1e-4, 3e-4, .001, .003, .01, .1, 1., 10.])
    write_json(root / "run_config.json", config)
    write_json(root / "invocation.json", dict(config=str(args.config.resolve()),
        source_config_sha256=sha256_file(args.config), arguments=vars(args) | {
            "config": str(args.config.resolve()), "output": str(root),
            "references": str(args.references.resolve()),
            "target_se_assets": str(args.target_se_assets.resolve()) if args.target_se_assets else None}))
    write_json(root / "status.json", dict(status="preparing_training_assets", optimizer_step=0))
    project = Path(__file__).resolve().parents[2]
    sources = {}
    for package in ("devflow", "benchmark"):
        for source in sorted((project / package).rglob("*.py")):
            relative = source.relative_to(project)
            destination = root / "source_snapshot" / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            sources[str(relative)] = sha256_file(source)
    write_json(root / "source_identity.json", dict(files=sources, source_tree_sha256=json_sha256(sources)))
    (root / "run_mean_head.py").write_text(
        "from pathlib import Path\nimport sys\n"
        "HERE = Path(__file__).resolve().parent\n"
        "sys.path.insert(0, str(HERE / 'source_snapshot'))\n"
        "from benchmark.reproduce.norman19 import mean_head_main\n"
        "if __name__ == '__main__':\n    mean_head_main()\n")

    started = time.monotonic()
    print(json.dumps(dict(phase="load_training_data", device="cpu")), flush=True)
    data = load_inputs(base, "cpu")
    assets, cv_report, features = prepare_kernel_mean_assets(data, config["mean_head"],
        progress=lambda row: print(json.dumps(row), flush=True))
    write_json(root / "cv_report.json", cv_report)
    torch.save(assets, root / "mean_assets.pt")
    np.savez_compressed(root / "gene_features.npz", **{name: features[name]
        for name in ("gene_names", "primary", "secondary", "target_se_tokens")})
    head = TrainableKernelMeanHead(**{name: assets[name] for name in
        ("kernel", "targets", "bias", "control_mean", "alpha", "metadata")})
    if torch.count_nonzero(head.A).item():
        raise AssertionError("Mean coefficients must start from zero")
    reference = head.closed_form_coefficients()
    reference_fit = torch.bmm(head.kernel, reference) + head.bias[:, None]
    reference_objective = ((reference_fit - head.targets).square().mean()
        + ((reference * torch.bmm(head.kernel, reference)).sum((1, 2)) * head.alpha).sum()
        / head.targets.numel()).item()
    optimizer = torch.optim.Adam([head.A], lr=args.learning_rate)
    training_rows = []
    train_start = time.monotonic()
    converged = False
    for step in range(1, args.max_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        loss = head.loss()
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite kernel training loss")
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_([head.A], 1.0, error_if_nonfinite=True)
        optimizer.step()
        if step == 1 or step % args.check_every == 0 or step == args.max_steps:
            with torch.no_grad():
                loss, parts = head.loss(return_parts=True)
                max_error = float((torch.bmm(head.kernel, head.A) + head.bias[:, None] - reference_fit).abs().max())
                gap = abs(float(loss) - reference_objective)
            row = dict(update=step, optimizer_step=step, micro_step=step, condition_passes=step,
                head_fit_loss=float(parts["head_fit_loss"]),
                head_regularization_loss=float(parts["head_regularization_loss"]),
                head_loss=float(loss), weighted_head_loss=float(loss), total_loss=float(loss),
                head_gradient_norm=float(gradient_norm), learning_rate_head=args.learning_rate,
                training_prediction_max_abs_error=max_error, training_objective_gap=gap,
                elapsed_seconds=time.monotonic() - started)
            training_rows.append(row)
            _write_rows(root / "train_metrics.csv", training_rows)
            write_json(root / "status.json", dict(status="training", **row))
            print(json.dumps(dict(phase="head_training", **row)), flush=True)
            if step == 1 or step % 2000 == 0:
                build_monitor(root)
            converged = step >= args.min_steps and max_error <= args.fit_tolerance and gap <= args.objective_tolerance
            if converged:
                break
    training_seconds = time.monotonic() - train_start
    torch.save(dict(schema="devflow_kernel_mean_head_only_v1", step=step,
        model=head.state_dict(), mean_assets=assets, optimizer=optimizer.state_dict(),
        training_binding=training_binding(data), convergence=training_rows[-1]), root / "head_checkpoint.pt")
    write_json(root / "training_complete.json", dict(optimizer_step=step, converged=converged,
        reference_used_for_initialization=False, training_seconds=training_seconds,
        reference_objective=reference_objective, last=training_rows[-1]))

    # The frozen external references are deliberately opened after fitting ends.
    protocol_path = args.references / "protocol.json"
    protocol = EvaluationProtocol.load(protocol_path)
    refs = EvaluationReferences(args.references, protocol)
    binding = training_binding(data)
    if (protocol.response_profile != "sampled" or protocol.condition_count != 37
            or protocol.cells_per_condition != 128 or protocol.n_genes != 2054
            or protocol.training_binding_sha256 != json_sha256(binding)):
        raise ValueError("Head evaluation must use the original locked Norman protocol")
    bank = ContinuousPairKernelBank.from_assets(assets["kernelbank_assets"])
    query = bank.query(*paired_gene_features(refs.condition_names, features), stratum=0)
    with torch.no_grad():
        means = head.predict_mean(query, 0).numpy()
        reference_means = (head.control_mean[0].float()
            + (torch.as_tensor(query) @ reference[0] + head.bias[0]).float()).clamp_min(0).numpy()
    np.savez_compressed(root / "predicted_means.npz", condition_names=np.asarray(refs.condition_names),
        genes=np.asarray(data["genes"]), predicted_mean=means, closed_form_reference_mean=reference_means)
    all_per_condition, all_per_seed, all_summary = [], [], []
    method = "Norman control-SE kernel mean head (Adam from zero)"
    head_rows = []
    for seed in protocol.seeds:
        with np.load(args.references / f"references_seed_{seed}.npz", allow_pickle=False) as sampled:
            archive = PredictionArchive(int(seed), refs.condition_names,
                np.array(sampled["source_global_ids"]), np.array(sampled["target_global_ids"]),
                protocol.gene_order_sha256,
                tuple(np.broadcast_to(mean, (protocol.cells_per_condition, len(mean))) for mean in means),
                dict(mode="mean_only_diagnostic", training_binding_sha256=json_sha256(binding)))
        head_rows.extend(_score_mean_archive(archive, refs, protocol))
    per_seed, summary = _aggregate_mean_rows(head_rows, method)
    all_per_condition.extend(dict(method=method, **row) for row in head_rows)
    all_per_seed.extend(per_seed)
    all_summary.extend(summary)
    # Numerical reference is scored only after actual zero-initialized training.
    reference_method = "Control-SE kernel closed-form reference (not Adam-trained weights)"
    reference_rows = []
    for seed in protocol.seeds:
        with np.load(args.references / f"references_seed_{seed}.npz", allow_pickle=False) as sampled:
            archive = PredictionArchive(int(seed), refs.condition_names,
                np.array(sampled["source_global_ids"]), np.array(sampled["target_global_ids"]),
                protocol.gene_order_sha256,
                tuple(np.broadcast_to(mean, (protocol.cells_per_condition, len(mean))) for mean in reference_means),
                dict(mode="mean_only_diagnostic", numerical_reference_only=True,
                     training_binding_sha256=json_sha256(binding)))
        reference_rows.extend(_score_mean_archive(archive, refs, protocol))
    per_seed, reference_summary = _aggregate_mean_rows(reference_rows, reference_method)
    all_per_condition.extend(dict(method=reference_method, **row) for row in reference_rows)
    all_per_seed.extend(per_seed)
    all_summary.extend(reference_summary)
    prior = project / "outputs/norman19/additive/kernel_mean_head_only_20261029"
    if prior.exists():
        previous_complete = json.loads((prior / "HEAD_ONLY_COMPLETE.json").read_text())
        if (previous_complete["training_binding_sha256"] != json_sha256(binding)
                or previous_complete["evaluation_protocol_sha256"] != sha256_file(protocol_path)
                or previous_complete["evaluation_manifest_sha256"] != sha256_file(refs.manifest_path)):
            raise ValueError("The zero-input comparison has a different training/evaluation identity")
        with np.load(prior / "predicted_means.npz", allow_pickle=False) as saved:
            if (tuple(saved["condition_names"]) != refs.condition_names
                    or list(saved["genes"]) != list(data["genes"])):
                raise ValueError("The zero-input comparison condition/gene axes differ")
            old_means = np.array(saved["predicted_mean"])
        prior_rows = []
        prior_method = "Previous zero-input kernel mean head (Adam 50k)"
        for seed in protocol.seeds:
            with np.load(args.references / f"references_seed_{seed}.npz", allow_pickle=False) as sampled:
                archive = PredictionArchive(int(seed), refs.condition_names,
                    np.array(sampled["source_global_ids"]), np.array(sampled["target_global_ids"]),
                    protocol.gene_order_sha256,
                    tuple(np.broadcast_to(mean, (protocol.cells_per_condition, len(mean))) for mean in old_means),
                    dict(mode="mean_only_diagnostic", training_binding_sha256=json_sha256(binding)))
            prior_rows.extend(_score_mean_archive(archive, refs, protocol))
        per_seed, prior_summary = _aggregate_mean_rows(prior_rows, prior_method)
        all_per_condition.extend(dict(method=prior_method, **row) for row in prior_rows)
        all_per_seed.extend(per_seed)
        all_summary.extend(prior_summary)
    baselines = {"Existing MLP head 25k (archived prediction means)":
                    project / "outputs/norman19/additive/checkpoint_25000/predictions",
                 "Existing MLP head 50k (archived prediction means)":
                    project / "outputs/norman19/additive/predictions"}
    baseline_identity = {}
    for baseline, directory in baselines.items():
        rows, hashes = [], {}
        for seed in protocol.seeds:
            path = directory / f"predictions_seed_{seed}.npz"
            archive = read_prediction_archive(path)
            rows.extend(_score_mean_archive(archive, refs, protocol))
            hashes[str(path)] = sha256_file(path)
        per_seed, summary = _aggregate_mean_rows(rows, baseline)
        all_per_condition.extend(dict(method=baseline, **row) for row in rows)
        all_per_seed.extend(per_seed)
        all_summary.extend(summary)
        baseline_identity[baseline] = hashes
    if prior.exists():
        baseline_identity[prior_method] = {str(prior / "predicted_means.npz"):
                                         sha256_file(prior / "predicted_means.npz")}
    _write_rows(root / "comparison_per_condition.csv", all_per_condition)
    _write_rows(root / "comparison_per_seed.csv", all_per_seed)
    _write_rows(root / "comparison_summary.csv", all_summary)
    metrics = {("de50_" if row["panel"] == "de50" else "") + row["metric"]: row["mean"]
               for row in all_summary if row["method"] == method}
    stds = {("de50_" if row["panel"] == "de50" else "") + row["metric"]: row["std"]
            for row in all_summary if row["method"] == method}
    summary = dict(update=step, optimizer_step=step, split="external_test",
        mode="mean_only_diagnostic", purpose="mean_head_comparison", seeds=list(protocol.seeds),
        n_sampling_repeats=len(protocol.seeds), n_conditions=protocol.condition_count,
        cells_per_condition=protocol.cells_per_condition, precision="FP64 kernels; FP32 expression means",
        metrics=metrics, metrics_std=stds, branches={"head": dict(metrics=metrics, metrics_std=stds)},
        protocol=dict(name=protocol.protocol, response_profile=protocol.response_profile,
            source_protocol_sha256=sha256_file(protocol_path)), distribution_metrics_computed=False)
    write_json(root / "mean_diagnostics" / f"step_{step:08d}" / "summary.json", summary)
    write_json(root / "summary.json", summary)
    with torch.no_grad():
        restored = TrainableKernelMeanHead(**{name: assets[name] for name in
            ("kernel", "targets", "bias", "control_mean", "alpha", "metadata")})
        restored.load_state_dict(torch.load(root / "head_checkpoint.pt", map_location="cpu", weights_only=False)["model"])
        restore_error = float((restored.predict_mean(query, 0) - torch.from_numpy(means)).abs().max())
    complete = dict(status="complete", run_mode="mean_head_only", optimizer_step=step,
        training_converged=converged, training_seconds=training_seconds,
        total_seconds=time.monotonic() - started, selected=cv_report["selected"],
        evaluation_protocol_sha256=sha256_file(protocol_path),
        evaluation_manifest_sha256=sha256_file(refs.manifest_path), baseline_identity=baseline_identity,
        training_binding_sha256=json_sha256(binding), head_checkpoint_sha256=sha256_file(root / "head_checkpoint.pt"),
        checkpoint_restore_max_abs_error=restore_error,
        test_prediction_max_abs_error_to_closed_form=float(np.max(np.abs(means - reference_means))),
        source_tree_sha256=json_sha256(sources), no_flow_or_decoder_training=True,
        no_gpu_used=True, no_external_selection=True)
    write_json(root / "HEAD_ONLY_COMPLETE.json", complete)
    write_json(root / "status.json", dict(status="complete", optimizer_step=step))
    build_monitor(root)
    print(json.dumps(complete), flush=True)


if __name__ == "__main__":
    import sys

    if sys.argv[1:2] == ["mean-head"]:
        mean_head_main(sys.argv[2:])
    else:
        main("norman19")
