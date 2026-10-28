"""Native, checkpoint-compatible ComboSciPlex direct-concatenation model."""
from __future__ import annotations
import torch
import torch.nn.functional as F
from torch import nn
from devflow.networks.expression_decoder import LatentToGeneDecoder
from devflow.networks.mlp_velocity import (ControlOTFlowMatchingSystem, median_sigmas, mmd2_unbiased_multi_sigma)
from devflow.flow.matching import pair_population
from devflow.networks.chemical_condition import DRUG_DIM, CONCAT_DRUG_DIM, MODEL_FAMILY, CONDITION_POOL
from devflow.model_attention import register_task_model


class EmbeddingControlOTFlowMatchingSystem(nn.Module):
    """MAP-conditioned OTFM whose transported state is a frozen-SE latent."""

    model_family = "map_original_se_decoder_otfm_v1"

    def __init__(self, state_dim=256, expression_dim=2000, path_sigma=0.5,
                 cell_line_dim=3, source_condition=True,
                 decoder_hidden_dims=(512, 1024), decoder_dropout=0.1,
                 condition_dim=64, direct_condition=False,
                 condition_output_dropout=None,
                 decoder_output_activation="linear") :
        super().__init__()
        self.state_dim = int(state_dim)
        self.expression_dim = int(expression_dim)
        self.path_sigma = float(path_sigma)
        self.source_condition = bool(source_condition)
        self.flow = ControlOTFlowMatchingSystem(
            state_dim=self.state_dim, path_sigma=self.path_sigma,
            cell_line_dim=cell_line_dim, condition_dim=int(condition_dim),
            source_condition=self.source_condition,
            direct_condition=direct_condition,
            condition_output_dropout=condition_output_dropout,
        )
        self.decoder = LatentToGeneDecoder(
            latent_dim=self.state_dim,
            gene_dim=self.expression_dim,
            hidden_dims=list(decoder_hidden_dims),
            dropout=float(decoder_dropout),
            output_activation=str(decoder_output_activation),
        )

    @staticmethod
    @torch.no_grad()
    def pair(source_state, source_h, source_g, target_state, target_expression,
             epsilon=1.0, max_iterations=2000, tolerance=1e-3, generator=None):
        if source_state.ndim != 2 or target_state.ndim != 2:
            raise ValueError("state populations must be two-dimensional")
        if source_state.size(1) != target_state.size(1):
            raise ValueError("source and target state dimensions must match")
        if target_expression.size(0) != target_state.size(0):
            raise ValueError("target expression and target state rows must align")
        source_index, target_index, stats = pair_population(
            source_state, target_state, source_state.size(0), epsilon,
            max_iterations, tolerance, generator
        )
        return (
            source_state.index_select(0, source_index),
            None if source_h is None else source_h.index_select(0, source_index),
            None if source_g is None else source_g.index_select(0, source_index),
            target_state.index_select(0, target_index),
            target_expression.index_select(0, target_index),
            stats,
        )

    def flow_loss(self, source_state, source_h, source_g, target_state,
                  drug_embedding, dose_log10, cell_line_onehot, generator=None,
                  mean_shift_target=None, mean_shift_weight=1.0, calculate_endpoint_mean=True):
        return self.flow.flow_matching_loss(
            source_state, source_h, source_g, target_state,
            drug_embedding, dose_log10, cell_line_onehot, generator=generator,
            mean_shift_target=mean_shift_target,
            mean_shift_weight=mean_shift_weight,
            return_components=True,
        )

    def oracle_loss(self, target_state, target_expression):
        return F.mse_loss(
            self.decoder(target_state), target_expression.float(), reduction="mean"
        )

    def forward(self, source_state, source_h, source_g, target_state,
                target_expression, drug_embedding, dose_log10, cell_line_onehot,
                ot_epsilon=1.0, ot_max_iterations=2000, ot_tolerance=1e-3,
                endpoint_time_points=20, pair_weight=0.1, mmd_weight=0.1,
                oracle_weight=0.1, mmd_cells=128, generator=None,
                adaptive=False, adaptive_rtol=1e-5, adaptive_atol=1e-5,
                calculate_mmd=True, mean_shift_target=None,
                mean_shift_weight=1.0, calculate_endpoint_mean=True):
        paired = self.pair(
            source_state, source_h, source_g, target_state, target_expression,
            epsilon=ot_epsilon, max_iterations=ot_max_iterations,
            tolerance=ot_tolerance, generator=generator,
        )
        source_p, source_h_p, source_g_p, target_p, target_expr_p, stats = paired
        flow_loss, residual_flow_loss, mean_shift_loss = self.flow_loss(
            source_p, source_h_p, source_g_p, target_p, drug_embedding,
            dose_log10, cell_line_onehot, generator=generator,
            mean_shift_target=mean_shift_target,
            mean_shift_weight=mean_shift_weight,
        )
        sample_kwargs = {"time_points": endpoint_time_points}
        if adaptive:
            sample_kwargs.update({
                "adaptive": True, "rtol": adaptive_rtol, "atol": adaptive_atol,
            })
        if calculate_endpoint_mean:
            endpoint_state = self.flow.sample(
                source_p, source_h_p, source_g_p, drug_embedding,
                dose_log10, cell_line_onehot, **sample_kwargs,
            )
            endpoint_expression = self.decoder(endpoint_state)
            predicted_bulk = endpoint_expression.mean(dim=0)
            target_bulk = target_expr_p.float().mean(dim=0)
            bulk_loss = F.mse_loss(predicted_bulk, target_bulk, reduction="mean")
            n_mmd = min(int(mmd_cells), endpoint_expression.size(0))
            if n_mmd < 2:
                raise ValueError("mmd_cells must leave at least two cells")
            mmd_loss = endpoint_expression.new_zeros((), dtype=torch.float32)
            if calculate_mmd:
                mmd_loss = mmd2_unbiased_multi_sigma(
                    endpoint_expression[:n_mmd].float(),
                    target_expr_p[:n_mmd].float(),
                    median_sigmas(target_expr_p[:n_mmd].float().detach()),
                )
        else:
            endpoint_state = None
            endpoint_expression = target_expr_p.new_zeros(target_expr_p.shape)
            bulk_loss = flow_loss.new_zeros(())
            mmd_loss = flow_loss.new_zeros(())
            n_mmd = 0
        oracle_loss = (
            self.oracle_loss(target_state, target_expression)
            if calculate_endpoint_mean else flow_loss.new_zeros(())
        )
        total = (
            flow_loss + float(pair_weight) * bulk_loss
            + float(mmd_weight) * mmd_loss + float(oracle_weight) * oracle_loss
        )
        return total, {
            "flow_loss": flow_loss,
            "residual_flow_loss": residual_flow_loss,
            "mean_shift_loss": mean_shift_loss,
            "bulk_loss": bulk_loss,
            "set_mmd_loss": mmd_loss,
            "oracle_loss": oracle_loss,
            "endpoint_state": endpoint_state,
            "endpoint_expression": endpoint_expression,
            "ot_stats": stats,
            "mmd_cells": n_mmd,
        }

    @torch.no_grad()
    def sample(self, source_state, source_h, source_g, drug_embedding,
               dose_log10, cell_line_onehot, time_points=20,
               adaptive=False, rtol=1e-5, atol=1e-5):
        endpoint_state = self.flow.sample(
            source_state, source_h, source_g, drug_embedding,
            dose_log10, cell_line_onehot, time_points=time_points,
            adaptive=adaptive, rtol=rtol, atol=atol,
        )
        return endpoint_state, self.decoder(endpoint_state)


class JointDecoderEmbeddingControlOTFlowMatchingSystem(
    EmbeddingControlOTFlowMatchingSystem
):
    """MAP-original SE Flow model with a trainable, decoder-only endpoint loss.

    The flow objective is the inherited OT velocity residual loss plus its
    population mean-shift term. The endpoint expression objective receives a
    detached endpoint state, so it updates the MAP decoder without updating the
    flow or its condition encoder. Its decomposition mirrors flow matching:
    centered single-cell residual MSE plus a weighted full-population mean MSE.
    """

    model_family = "map_original_se_decoder_joint_otfm_v1"

    def __init__(self, *args, decoder_mean_loss_weight=32.0, **kwargs):
        kwargs.setdefault("decoder_output_activation", "relu")
        super().__init__(*args, **kwargs)
        self.decoder_mean_loss_weight = float(decoder_mean_loss_weight)

    def forward(
        self,
        source_state,
        source_h,
        source_g,
        target_state,
        target_expression,
        drug_embedding,
        dose_log10,
        cell_line_onehot,
        ot_epsilon=1.0,
        ot_max_iterations=2000,
        ot_tolerance=1e-3,
        endpoint_time_points=80,
        generator=None,
        mean_shift_target=None,
        mean_shift_weight=32.0,
        endpoint_mean_target=None,
        decoder_mean_loss_weight=None,
        adaptive=False,
        adaptive_rtol=1e-5,
        adaptive_atol=1e-5,
    ):
        paired = self.pair(
            source_state,
            source_h,
            source_g,
            target_state,
            target_expression,
            epsilon=ot_epsilon,
            max_iterations=ot_max_iterations,
            tolerance=ot_tolerance,
            generator=generator,
        )
        source_p, source_h_p, source_g_p, target_p, target_expr_p, stats = paired
        flow_loss, residual_flow_loss, mean_shift_loss = self.flow_loss(
            source_p,
            source_h_p,
            source_g_p,
            target_p,
            drug_embedding,
            dose_log10,
            cell_line_onehot,
            generator=generator,
            mean_shift_target=mean_shift_target,
            mean_shift_weight=mean_shift_weight,
        )

        # Endpoint rollout is deliberately detached from the decoder objective.
        # No ODE graph is retained, and the endpoint loss can only update decoder.
        with torch.no_grad():
            endpoint_state = self.flow.sample(
                source_p,
                source_h_p,
                source_g_p,
                drug_embedding,
                dose_log10,
                cell_line_onehot,
                time_points=endpoint_time_points,
                adaptive=adaptive,
                rtol=adaptive_rtol,
                atol=adaptive_atol,
            )
        endpoint_expression = self.decoder(endpoint_state.detach())
        target_expr_p = target_expr_p.detach().float()
        prediction = endpoint_expression.float()
        prediction_centered = prediction - prediction.mean(dim=0, keepdim=True)
        target_centered = target_expr_p - target_expr_p.mean(dim=0, keepdim=True)
        decoder_residual_loss = F.mse_loss(prediction_centered, target_centered)
        if endpoint_mean_target is None:
            mean_target = target_expr_p.mean(dim=0)
        else:
            if endpoint_mean_target.shape != (self.expression_dim,):
                raise ValueError(
                    f"endpoint_mean_target must be [{self.expression_dim}], "
                    f"got {tuple(endpoint_mean_target.shape)}"
                )
            mean_target = endpoint_mean_target.detach().float()
        decoder_mean_loss = F.mse_loss(prediction.mean(dim=0), mean_target)
        decoder_cell_mse = F.mse_loss(prediction, target_expr_p)
        mean_weight = (
            self.decoder_mean_loss_weight
            if decoder_mean_loss_weight is None
            else float(decoder_mean_loss_weight)
        )
        decoder_loss = decoder_residual_loss + mean_weight * decoder_mean_loss
        zero = flow_loss.detach().new_zeros(())
        return flow_loss + decoder_loss, {
            "flow_loss": flow_loss,
            "residual_flow_loss": residual_flow_loss,
            "mean_shift_loss": mean_shift_loss,
            "bulk_loss": decoder_loss,
            "decoder_loss": decoder_loss,
            "decoder_residual_loss": decoder_residual_loss,
            "decoder_mean_loss": decoder_mean_loss,
            "weighted_decoder_mean_loss": mean_weight * decoder_mean_loss,
            "decoder_cell_mse": decoder_cell_mse,
            "set_mmd_loss": zero,
            "oracle_loss": zero,
            "endpoint_state": endpoint_state,
            "endpoint_expression": endpoint_expression,
            "ot_stats": stats,
            "mmd_cells": 0,
        }


def widen_direct_condition(model):
    """Widen exactly one existing Linear, without advancing CPU/CUDA RNG."""
    encoder, velocity = model.flow.perturbation_encoder, model.flow.velocity
    if not encoder.direct_concat or encoder.drug_dim != DRUG_DIM:
        raise ValueError("Adapter requires the unchanged 1024-D direct-condition model")
    expected_old_condition = DRUG_DIM + 1 + model.flow.cell_line_dim
    if velocity.condition_dim != expected_old_condition:
        raise ValueError("Unexpected baseline condition width")
    old = velocity.decoder.net[0]
    if not isinstance(old, nn.Linear):
        raise TypeError("Expected the existing first velocity decoder layer to be Linear")
    prefix = old.in_features - expected_old_condition
    # Forward concatenation is [encoded_time, state, optional_source, drug, dose, line].
    if prefix != (velocity.time_encoder.net[-1].out_features + model.state_dim
                  + (velocity.source_dim if velocity.use_source_condition else 0)):
        raise AssertionError("Unexpected velocity concatenation order/width")
    devices = [old.weight.device.index] if old.weight.is_cuda else []
    with torch.random.fork_rng(devices=devices):
        widened = nn.Linear(old.in_features + DRUG_DIM, old.out_features,
                            bias=old.bias is not None,
                            device=old.weight.device, dtype=old.weight.dtype)
    with torch.no_grad():
        widened.weight[:, :prefix].copy_(old.weight[:, :prefix])
        drug_columns = old.weight[:, prefix:prefix + DRUG_DIM]
        widened.weight[:, prefix:prefix + DRUG_DIM].copy_(drug_columns)
        widened.weight[:, prefix + DRUG_DIM:prefix + CONCAT_DRUG_DIM].copy_(drug_columns)
        widened.weight[:, prefix + CONCAT_DRUG_DIM:].copy_(old.weight[:, prefix + DRUG_DIM:])
        if old.bias is not None:
            widened.bias.copy_(old.bias)
    widened.weight.requires_grad_(old.weight.requires_grad)
    if old.bias is not None:
        widened.bias.requires_grad_(old.bias.requires_grad)
    velocity.decoder.net[0] = widened
    velocity.decoder.input_dim = widened.in_features
    velocity.decoder_input_dim = widened.in_features
    velocity.condition_dim = expected_old_condition + DRUG_DIM
    # The direct branch never invokes encoder.drug: retain its baseline parameters
    # exactly, including its original unused input width, to preserve initialization.
    encoder.drug_dim = CONCAT_DRUG_DIM
    model.direct_concat_metadata = dict(
        condition_dim=velocity.condition_dim, drug_condition_dim=CONCAT_DRUG_DIM,
        condition_pool=CONDITION_POOL, drug_slot_order="lexical_canonical_drug_name",
        single_drug_padding="second_slot_exact_zero", random_order_augmentation=False,
        initial_widening="duplicate_baseline_drug_columns_without_rng_advance",
        initial_function_equivalence="dropout_disabled_float_roundoff_tolerance",
        velocity_input_dim=widened.in_features,
        added_trainable_parameters=DRUG_DIM * old.out_features,
    )
    return model


class DirectConcatJointDecoder(JointDecoderEmbeddingControlOTFlowMatchingSystem):
    model_family = MODEL_FAMILY

    def __init__(self, *args, **kwargs):
        if args:
            raise TypeError("Use keyword model arguments for the direct-concat interface")
        if not kwargs.get("direct_condition", False):
            raise ValueError("Direct drug concatenation requires direct_condition=True")
        cell_line_dim = int(kwargs.get("cell_line_dim", 3))
        if cell_line_dim != 3:
            raise ValueError("Retain the existing three-element cell-line vocabulary")
        old_condition = DRUG_DIM + 1 + cell_line_dim
        requested = int(kwargs.get("condition_dim", old_condition + DRUG_DIM))
        if requested not in (old_condition, old_condition + DRUG_DIM):
            raise ValueError("Unexpected direct-condition dimension")
        kwargs["condition_dim"] = old_condition
        super().__init__(**kwargs)
        widen_direct_condition(self)


class ComboSciPlexModel(DirectConcatJointDecoder):
    """The paper model; defaults preserve the final direct-concat experiment."""

    def __init__(self, **kwargs):
        defaults = dict(state_dim=2048, expression_dim=2000, path_sigma=0.01,
                        cell_line_dim=3, source_condition=False, condition_dim=2052,
                        direct_condition=True, condition_output_dropout=0.1,
                        decoder_mean_loss_weight=1.0, decoder_output_activation="relu")
        defaults.update(kwargs)
        super().__init__(**defaults)

    @classmethod
    def from_checkpoint(cls, checkpoint_path, *, map_location="cpu", **overrides):
        payload = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
        if payload.get("model_family") != MODEL_FAMILY:
            raise ValueError("Expected the direct-concat ComboSciPlex checkpoint, not a summed-drug model")
        args = payload["args"]
        kwargs = dict(state_dim=payload["model"]["flow.velocity.output.weight"].shape[0],
                      expression_dim=payload["model"]["decoder.decoder.8.0.weight"].shape[0],
                      path_sigma=args["path_sigma"],
                      cell_line_dim=len(args["cell_lines"]),
                      source_condition=not args["no_source_condition"],
                      condition_dim=args["condition_dim"], direct_condition=args["direct_condition"],
                      condition_output_dropout=args["condition_output_dropout"],
                      decoder_mean_loss_weight=args["decoder_mean_loss_weight"])
        kwargs.update(overrides)
        model = cls(**kwargs)
        model.load_state_dict(payload["model"], strict=True)
        model.checkpoint_gene_order_sha256 = payload.get("gene_order_sha256")
        model.checkpoint_cell_order_sha256 = args.get("combo_cell_order_sha256")
        model.checkpoint_training_binding = payload.get("training_binding")
        return model


ComboSciPlexFlowMatchingSystem = ComboSciPlexModel
register_task_model("combosciplex", ComboSciPlexModel)



from devflow.data.combosciplex import (PROTOCOL, NORMALIZATION, sequence_sha256, _population_stats, _corr, _de_mask, _clean_json, _write_json)
from devflow.networks.chemical_condition import sample_conditions
from pathlib import Path
import numpy as np
import csv
import math
METRICS = ("pseudobulk_mae", "pseudobulk_mse", "pearson_delta", "de_spearman", "endpoint_mmd")

def evaluate_combo(model, dataset, smiles_to_index, embedding_bank,
                   device, rank, world_size, global_step, output_dir, time_points):
    """Five test conditions, fixed shared real controls, no parameter updates."""
    from devflow.networks.mlp_velocity import mmd2_unbiased_multi_sigma
    if dataset.split != "external_test" or len(dataset) != 5 or dataset.set_size != 128 or time_points != 80:
        raise ValueError("Combo evaluator requires its fixed 5-group/128-source-cell/RK4-80 protocol")
    unwrapped = getattr(model, "module", model)
    model.eval()
    expected_strata = {"both_monotherapy": 2, "one_monotherapy": 3}
    observed_strata = {name: sum(r["stratum"] == name for r in dataset.records) for name in expected_strata}
    if observed_strata != expected_strata:
        raise ValueError(f"Test strata must contain 2 both-monotherapy and 3 one-monotherapy conditions: {observed_strata}")
    control_stats = _population_stats(dataset, dataset.eval_control_rows)
    local = []
    with torch.inference_mode(), torch.autocast(device_type=device.type, enabled=False):
        for index in range(rank, len(dataset), world_size):
            sample = dataset[index]
            drug, dose, line = sample_conditions(sample, dataset, smiles_to_index, embedding_bank, device)
            _, prediction = unwrapped.sample(sample["control_state"].to(device).float(), None, None,
                                             drug.float(), dose.float(), line.float(), time_points=int(time_points), adaptive=False)
            prediction = prediction.float().cpu()
            target_sample = sample["treated_expression"].float()
            if not torch.isfinite(prediction).all() or not torch.isfinite(target_sample).all():
                raise FloatingPointError(f"Nonfinite prediction or target for {sample['drug_id']}")
            record = dataset.records[index]
            stats = _population_stats(dataset, record["rows"])
            target_mean, _, n_target = stats
            pred_mean = prediction.numpy().astype(np.float64).mean(0)
            pred_delta, true_delta = pred_mean - control_stats[0], target_mean - control_stats[0]
            mask = _de_mask(stats, control_stats)
            mmd_value = float(mmd2_unbiased_multi_sigma(prediction, target_sample))
            if not np.isfinite(target_mean).all() or not math.isfinite(mmd_value):
                raise FloatingPointError(f"Nonfinite expression statistics or MMD for {record['condition']}")
            mae = float(np.abs(pred_mean-target_mean).mean())
            mse = float(np.square(pred_mean-target_mean).mean())
            if not math.isfinite(mae) or not math.isfinite(mse):
                raise FloatingPointError(f"Nonfinite pseudobulk errors for {record['condition']}")
            row = dict(condition=record["condition"], stratum=record["stratum"], n_prediction_cells=128,
                       n_target_cells=n_target, n_de_genes=int(mask.sum()),
                       pseudobulk_mae=mae,
                       pseudobulk_mse=mse,
                       pearson_delta=_corr(pred_delta, true_delta),
                       de_spearman=_corr(pred_delta[mask], true_delta[mask], spearman=True),
                       endpoint_mmd=mmd_value)
            local.append(dict(index=index, row=row, prediction=prediction.numpy(),
                              target_sample_rows=sample["treated_rows"].tolist()))
    if world_size > 1:
        gathered = [None] * world_size if rank == 0 else None
        torch.distributed.gather_object(local, gathered, dst=0)
        records = [r for shard in gathered for r in shard] if rank == 0 else []
    else:
        records = local
    if rank != 0:
        return
    records.sort(key=lambda x: x["index"])
    if [r["index"] for r in records] != list(range(5)):
        raise ValueError("Distributed evaluator did not gather all five unique conditions")
    def macro(rows):
        return {metric: float(np.mean([r[metric] for r in rows])) for metric in METRICS}
    rows = [r["row"] for r in records]
    strata = {name: macro([r for r in rows if r["stratum"] == name]) for name in ("both_monotherapy", "one_monotherapy")}
    summary = dict(protocol=PROTOCOL, checkpoint_step=int(global_step), n_conditions=5,
                   all_conditions=macro(rows), strata=strata,
                   balanced_strata={key: float(np.mean([strata[name][key] for name in strata])) for key in METRICS},
                   model_selection="none; periodic test monitoring only", primary_metric="pearson_delta",
                   expression_normalization=NORMALIZATION, gene_order_sha256=sequence_sha256(dataset.genes),
                   mmd_definition="unbiased_MMD_squared; gene_space; pooled_median_bandwidth; scales=.5,1,2,4; fixed128draw",
                   pseudobulk_targets="all observed target cells", de_definition="Welch_ttest_full_target_vs_shared128control_BH_q<.05")
    dest = Path(output_dir) / "expression_evaluation" / "combo_test" / f"update_{global_step:06d}"
    dest.mkdir(parents=True, exist_ok=True)
    _write_json(dest / "summary.json", _clean_json(summary))
    with (dest / "per_condition.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    obs = dataset.bundle["obs"]
    sampling = dict(seed=dataset.seed, shared_control_cell_ids=[obs[i]["cell_id"] for i in dataset.eval_control_rows],
                    conditions={r["row"]["condition"]: dict(mmd_target_cell_ids=[obs[i]["cell_id"] for i in r["target_sample_rows"]],
                                                            pseudobulk_target_cell_ids=[obs[i]["cell_id"] for i in dataset.records[r["index"]]["rows"]]) for r in records})
    _write_json(dest / "sampling.json", sampling)
    np.savez_compressed(dest / "predictions.npz", condition_names=np.asarray([r["condition"] for r in rows]),
                        gene_order_sha256=np.asarray(sequence_sha256(dataset.genes)),
                        **{f"prediction_{i:04d}": r["prediction"] for i, r in enumerate(records)})
    print("Combo test monitoring:", json.dumps(_clean_json(summary), sort_keys=True), flush=True)

