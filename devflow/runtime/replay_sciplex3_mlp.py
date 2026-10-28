"""Explicit, inference-only replay of the historical mixed-kernel table row.

This entry converts the *locked* 52k MLP and fitted mixed-kernel coefficients
into one inference checkpoint. The normal training entry must never resume
this converted checkpoint: the new recipe initialises Flow and A fresh.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch

from devflow.config import load_config
from devflow.model_mlp import MLPMeanSystem
from devflow.networks.state_encoder import SEControlLoRATail
from .artifacts import atomic_json, sha256


FLOW_SHA256 = "300d08750d2224e3a84216cec1eb0c10ce43c370b1da058d7916af17b7aec3e8"
HEAD_SHA256 = "629781e221243774ed354fa4cc95d2f07bf69de8e670fea116a0a905562b9769"
HISTORICAL_STEP = 52000


def _array(value):
    return value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)


def validate_historical_head(old, assets, gene_order_sha256):
    """Verify anchors/strata and all fixed prediction statistics before loading A."""
    metadata = assets["metadata"]
    if str(old["gene_order_sha256"]) != gene_order_sha256:
        raise ValueError("Historical mean-head gene axis differs")
    if old["drug_ids"].tolist() != list(metadata["anchor_drug_ids"]):
        raise ValueError("Historical mean-head anchor order differs")
    strata = [f"{line}|{float(dose):g}" for line, dose in metadata["strata"]]
    if old["strata"].tolist() != strata:
        raise ValueError("Historical mean-head stratum order differs")
    bank = assets["kernelbank_assets"]
    for field in ("feature_mean", "feature_std", "x_train", "fingerprint_train"):
        np.testing.assert_allclose(old[field], _array(bank[field]), rtol=1e-12, atol=1e-12,
                                   err_msg="Historical kernel asset differs: " + field)
    np.testing.assert_allclose(old["kernel_train_column_mean"], _array(bank["training_column_means"]),
                               rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(old["kernel_train_grand_mean"], _array(bank["training_grand_means"]),
                               rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(old["y_mean"], _array(assets["bias"]))
    lines = list(dict.fromkeys(str(line) for line, _ in metadata["strata"]))
    controls = np.stack([old["control_mean"][lines.index(str(line))]
                         for line, _ in metadata["strata"]])
    np.testing.assert_array_equal(controls, _array(assets["control_mean"]))
    mixes = np.asarray([float(s["kg_mix"]) for s in bank["specs"]])
    gammas = np.asarray([0. if s["kg_mix"] == 0 else
                         float(s["gamma_factor"]) / float(bank["median_squared_distance"])
                         for s in bank["specs"]])
    np.testing.assert_array_equal(old["kg_mix"], mixes)
    np.testing.assert_allclose(old["gamma"], gammas, rtol=1e-12, atol=1e-12)
    # The locked historical candidate list iterated these four alphas inside
    # each kernel family. This interpretation belongs only to the replay adapter.
    alphas = np.asarray([.001, .01, .1, 1.])[old["spec_ids"].astype(int) % 4]
    np.testing.assert_array_equal(alphas, _array(assets.get("alpha", assets.get("alphas"))))
    coefficient = np.asarray(old["coef"], dtype=np.float64).transpose(1, 0, 2)
    if coefficient.shape != _array(assets["targets"]).shape or not np.isfinite(coefficient).all():
        raise ValueError("Historical coefficient axes differ from the checked anchors/genes")
    return torch.from_numpy(coefficient.copy())


def load_historical_components(flow_checkpoint, head_checkpoint, runtime):
    """Only these hashed source artifacts can instantiate this reference row."""
    flow_checkpoint, head_checkpoint = Path(flow_checkpoint), Path(head_checkpoint)
    if sha256(flow_checkpoint) != FLOW_SHA256 or sha256(head_checkpoint) != HEAD_SHA256:
        raise ValueError("Historical replay requires the locked 52k/mixed-kernel artifact hashes")
    payload = torch.load(str(flow_checkpoint), map_location="cpu", mmap=True, weights_only=False)
    if payload.get("model_family") != "map_original_se_decoder_joint_otfm_control_lora_v1":
        raise ValueError("Historical checkpoint is not the basic MLP control-LoRA family")
    if int(payload.get("optimizer_step", -1)) != HISTORICAL_STEP:
        raise ValueError("Historical replay source step differs")
    args = payload["args"]
    if (not args.get("no_source_condition") or not args.get("direct_condition")
            or int(args.get("condition_dim", 0)) != 1028
            or float(args.get("condition_output_dropout", -1)) != 0.):
        raise ValueError("Historical checkpoint uses different condition architecture")
    old = payload["model"]
    tail_state = {key[len("flow.control_lora."):]: value for key, value in old.items()
                  if key.startswith("flow.control_lora.")}
    zero_b = [key for key in tail_state if key.endswith("lora_B")]
    if len(zero_b) != 2 or any(torch.count_nonzero(tail_state[key]).item() for key in zero_b):
        raise ValueError("Historical source LoRA changed: a frozen-source replay would be invalid")
    manifest = runtime.asset_manifest
    tail_path = Path(manifest["inputs"]["tail_checkpoint"])
    if sha256(tail_path) != manifest["source_policy"]["tail_checkpoint_sha256"]:
        raise ValueError("The source-cache SE tail identity changed")
    with torch.random.fork_rng(devices=[]):
        tail = SEControlLoRATail.load_tail(tail_path, map_location="cpu")
    current_tail = tail.state_dict()
    if set(current_tail) != set(tail_state):
        raise ValueError("Historical source encoder topology differs from prepared source assets")
    if any(not torch.equal(current_tail[key], value) for key, value in tail_state.items()):
        raise ValueError("Historical source encoder differs from prepared source assets")
    del tail, current_tail
    with np.load(head_checkpoint, allow_pickle=False) as archive:
        coefficient = validate_historical_head(archive, runtime.mean_assets,
                                               runtime.data.gene_order_sha256)
    with torch.random.fork_rng(devices=[]):
        model = MLPMeanSystem({}, runtime.mean_assets,
                             expected_gene_order_sha256=runtime.data.gene_order_sha256)
    flow_state = {key[len("flow."):]: value for key, value in old.items()
                  if key.startswith("flow.") and not key.startswith("flow.control_lora.")}
    decoder_state = {key[len("decoder."):]: value for key, value in old.items()
                     if key.startswith("decoder.")}
    allowed = {"flow." + key for key in flow_state} | {"decoder." + key for key in decoder_state}
    allowed |= {"flow.control_lora." + key for key in tail_state}
    if allowed != set(old):
        raise ValueError("Historical source contains unaccounted model parameters")
    model.flow.load_state_dict(flow_state, strict=True)
    model.decoder.load_state_dict(decoder_state, strict=True)
    with torch.no_grad():
        model.mean_head.A.copy_(coefficient)
    model.requires_grad_(False)
    model.eval()
    audit = dict(purpose="historical_replay_only", resume_allowed=False,
                 source_optimizer_step=HISTORICAL_STEP,
                 source_flow_checkpoint=str(flow_checkpoint.resolve()),
                 source_flow_sha256=FLOW_SHA256,
                 source_mean_checkpoint=str(head_checkpoint.resolve()),
                 source_mean_sha256=HEAD_SHA256, source_lora_B_zero=True,
                 source_encoder_matches_prepared_cache=True,
                 strictly_loaded_flow_tensors=len(flow_state),
                 strictly_loaded_decoder_tensors=len(decoder_state),
                 mean_anchor_gene_stratum_order_verified=True,
                 training_binding=runtime.data.training_binding(),
                 asset_manifest_sha256=runtime.asset_manifest_sha256)
    return model, audit


def verify_reference(model, runtime, config, reference_root, *, verify_cells=True):
    """Check all condition means and optionally one real 128-cell prediction."""
    from .evaluate_sciplex3_mlp import formal_mode, _references, _sample_bundle, _score
    root = Path(reference_root)
    with np.load(root / "condition_mean_predictions.npz", allow_pickle=False) as original:
        if original["condition_names"].tolist() != [r["condition"] for r in runtime.data.test_records]:
            raise ValueError("Historical reference condition axis differs")
        if str(original["gene_order_sha256"]) != runtime.data.gene_order_sha256:
            raise ValueError("Historical reference gene axis differs")
        with torch.no_grad():
            means = np.stack([model.predict_mean(runtime.mean_query(record)).cpu().numpy()
                              for record in runtime.data.test_records])
        old_means = np.array(original["mean"])
        np.testing.assert_allclose(means, old_means, rtol=1e-6, atol=1e-7)
        result = dict(status="passed", mean_conditions=len(means),
                      max_mean_absolute_error=float(np.abs(means - old_means).max()),
                      cell_prediction_checked=False)
    if verify_cells:
        import pandas as pd
        records = runtime.data.sampling(4243)
        record = records[0]
        with np.load(root / "reference_seed4243_condition0.npz", allow_pickle=False) as original:
            if int(original["seed"]) != 4243 or str(original["gene_order_sha256"]) != runtime.data.gene_order_sha256:
                raise ValueError("Historical sampled prediction identity differs")
            np.testing.assert_array_equal(original["source_global_ids"][0],
                runtime.data.global_ids(record["cell_line"], record["control_rows"]))
            np.testing.assert_array_equal(original["target_global_ids"][0],
                runtime.data.global_ids(record["cell_line"], record["target_rows"]))
            expected = np.array(original["prediction_0000"])
        with formal_mode(model):
            prediction = model.predict(runtime.source_batch(record, record["control_rows"]),
                                       time_points=80).cpu().numpy()
        np.testing.assert_allclose(prediction, expected, rtol=2e-5, atol=2e-6)
        references = _references(config, runtime.data)
        bundle = _sample_bundle(runtime.data, references, 4243)
        scores = _score(prediction, 0, bundle, references, include_distribution=True)
        frame = pd.read_csv(root / "per_group_metrics.csv")
        prior = frame[(frame.seed == 4243) & (frame.group_index == 0)].iloc[0]
        errors = {key: abs(float(value) - float(prior[key])) for key, value in scores.items()
                  if key in prior and np.isfinite(value) and np.isfinite(prior[key])}
        # FP32 GPU inference/cache batching can differ at the final few ulps.
        if any(value > 2e-5 for value in errors.values()):
            raise AssertionError("Historical condition metric discrepancy: " + str(errors))
        result.update(cell_prediction_checked=True, seed=4243, condition=record["condition"],
                      cells=128, max_cell_absolute_error=float(np.abs(prediction - expected).max()),
                      metric_absolute_errors=errors, metrics=scores)
    return result


def replay(config, flow_checkpoint, head_checkpoint, *, reference_root=None,
           device="cpu", evaluation="none", verify_cells=True):
    from .sciplex3_mlp import make_runtime
    from .evaluate_sciplex3_mlp import evaluate
    root = Path(config["output_dir"])
    if root.exists() and any(root.iterdir()):
        raise FileExistsError("Historical replay requires a new experiment output directory")
    root.mkdir(parents=True, exist_ok=True)
    runtime = make_runtime(config, device)
    model, audit = load_historical_components(flow_checkpoint, head_checkpoint, runtime)
    model.to(device)
    checkpoint = root / "checkpoints" / "historical_mixed_kernel_052000.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(schema=model.checkpoint_schema, purpose="historical_replay_only",
                   resume_allowed=False, model=model.state_dict(), model_spec=model.export_spec(),
                   update=HISTORICAL_STEP, historical_provenance=audit,
                   training_binding=runtime.data.training_binding(),
                   asset_manifest_sha256=runtime.asset_manifest_sha256)
    temporary = checkpoint.with_suffix(".pt.tmp")
    torch.save(payload, temporary)
    temporary.replace(checkpoint)
    atomic_json(root / "run_config.json", {**config, "purpose": "historical_replay_only"})
    atomic_json(root / "historical_conversion.json", {**audit, "checkpoint": str(checkpoint.resolve()),
                                                     "checkpoint_sha256": sha256(checkpoint)})
    if reference_root is not None:
        result = verify_reference(model, runtime, config, reference_root, verify_cells=verify_cells)
        atomic_json(root / "historical_reference_verification.json", result)
    if evaluation != "none":
        mean_only = evaluation == "mean"
        seeds = (4243,) if evaluation == "seed4243" else (4243, 4244, 4245, 4246)
        evaluate(model, runtime, config, HISTORICAL_STEP, checkpoint, seeds=seeds,
                 mean_only=mean_only, purpose="benchmark" if evaluation == "full" else "monitor")
    atomic_json(root / "REPLAY_COMPLETE.json", dict(status="complete", checkpoint=str(checkpoint),
        evaluation=evaluation, full_benchmark_completed=evaluation == "full",
        reference_checked=reference_root is not None))
    return checkpoint


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--flow-checkpoint", required=True)
    parser.add_argument("--head-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--reference-root")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--evaluation", choices=("none", "mean", "seed4243", "full"), default="none")
    parser.add_argument("--skip-cell-check", action="store_true")
    args = parser.parse_args(argv)
    config = copy.deepcopy(load_config(args.config))
    config["output_dir"] = args.output
    torch.set_num_threads(4)
    replay(config, args.flow_checkpoint, args.head_checkpoint, reference_root=args.reference_root,
           device=args.device, evaluation=args.evaluation, verify_cells=not args.skip_cell_check)


if __name__ == "__main__":
    main()
