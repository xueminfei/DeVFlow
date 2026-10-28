"""Basic MLP Flow, trainable conditional mean, and distribution decoder.

The three losses have deliberately separate parameter routes:
Flow residual + population mean -> Flow; dual KRR -> mean A;
MMD of the composed expression -> decoder only. One checkpoint contains all
learned parameters and fixed mean-head assets. Inference performs composition
inside this model, without a separately fitted post-processing program.
"""
from __future__ import annotations

import copy
from pathlib import Path

import torch
from torch import nn

from .mean import KernelMeanHead, compose_expression
from .networks.expression_decoder import LatentToGeneDecoder
from .networks.mlp_velocity import (ControlOTFlowMatchingSystem, median_sigmas,
                                    mmd2_unbiased_multi_sigma, rk4_rollout)


DEFAULT_CONFIG = {
    "state_dim": 2048, "condition_dim": 1028, "source_dim": 1024,
    "source_hidden_dim": 2048, "cell_line_dim": 3,
    "condition_branch_dims": [256, 256], "condition_after_pool_dims": [256, 256],
    "state_dims": [2048, 2048, 2048], "time_dims": [1024, 1024, 1024],
    "decoder_dims": [4096, 4096, 4096], "time_frequencies": 1024,
    "path_sigma": 0.01, "condition_output_dropout": 0.0,
    "expression_hidden_dims": [512, 1024], "expression_dropout": 0.1,
    "expression_output_activation": "relu",
    "initialization_policy": "direct",
    "decoder_checkpoint_family": "SciPlex3_true_SE_MSE_MMD_decoder",
}
DEFAULT_LOSS_WEIGHTS = {"flow_residual": 1.0, "flow_mean": 32.0,
                        "decoder_mmd": 1.0, "head": 1.0}


class MLPMeanSystem(nn.Module):
    model_family = "mlp_mean_mmd_v1"
    checkpoint_schema = "devflow_mlp_mean_training_v1"

    def __init__(self, model_config=None, mean_assets=None, decoder_checkpoint=None,
                 expected_gene_order_sha256=None, *, loss_weights=None):
        super().__init__()
        settings = dict(model_config or {})
        # These compatibility flags are fixed by this model family.
        if settings.pop("source_condition", False) or not settings.pop("direct_condition", True):
            raise ValueError("This MLP family requires direct condition and no source-condition branch")
        unknown = set(settings) - set(DEFAULT_CONFIG)
        if unknown:
            raise ValueError(f"Unknown basic MLP model settings: {sorted(unknown)}")
        self.model_config = {**copy.deepcopy(DEFAULT_CONFIG), **settings}
        cfg = self.model_config
        self._validate_model_config(cfg)
        self.loss_weights = {**DEFAULT_LOSS_WEIGHTS, **dict(loss_weights or {})}
        if set(self.loss_weights) != set(DEFAULT_LOSS_WEIGHTS):
            raise ValueError("Loss weights must name flow_residual, flow_mean, decoder_mmd and head")
        self.loss_weights = {k: float(v) for k, v in self.loss_weights.items()}
        if any(not torch.isfinite(torch.tensor(v)) or v < 0 for v in self.loss_weights.values()):
            raise ValueError("Loss weights must be finite and nonnegative")
        if mean_assets is None:
            raise ValueError("Training-only fixed mean assets are required")
        target = mean_assets["targets"]
        gene_dim = int(target.shape[-1])
        self.flow, self.decoder = self._make_networks(cfg, gene_dim)
        # Direct conditioning bypasses these parameterised branches. They stay
        # in state_dict for topology/RNG compatibility but must not enter DDP.
        self.flow.perturbation_encoder.requires_grad_(False)
        self.gene_order_sha256 = expected_gene_order_sha256
        if decoder_checkpoint is not None:
            self._load_decoder(decoder_checkpoint, expected_gene_order_sha256)
        self.decoder.requires_grad_(True)
        self.mean_head = KernelMeanHead(
            kernel=mean_assets.get("kernel", mean_assets.get("kernels")),
            targets=mean_assets["targets"], bias=mean_assets["bias"],
            control_mean=mean_assets["control_mean"],
            alpha=mean_assets.get("alpha", mean_assets.get("alphas")),
            metadata=mean_assets.get("metadata"))
        self._extra_mean_assets = copy.deepcopy({key: value for key, value in mean_assets.items()
            if key not in {"kernel", "kernels", "targets", "bias", "control_mean", "alpha", "alphas", "metadata"}})

    def _validate_model_config(self, cfg):
        if not 0 <= float(cfg["condition_output_dropout"]) < 1:
            raise ValueError("Condition dropout must lie in [0,1)")
        if cfg["initialization_policy"] not in {"direct", "duplicate_drug_slots"}:
            raise ValueError("Unknown MLP initialization policy")
        if cfg["initialization_policy"] == "duplicate_drug_slots":
            if int(cfg["condition_dim"]) != 2052 or int(cfg["cell_line_dim"]) != 3:
                raise ValueError("Two-slot initialization requires condition_dim=2052 and three cell-line slots")

    def _make_networks(self, cfg, gene_dim):
        """Construct one backbone using an explicit initialization policy."""
        duplicate = cfg["initialization_policy"] == "duplicate_drug_slots"
        if duplicate:
            from .runtime.combosciplex import widen_direct_condition
        # Instantiate the original condition submodules before the velocity
        # network, retaining its parameter topology and initialisation RNG.
        flow_settings = {key: cfg[key] for key in (
                "state_dim", "path_sigma", "source_dim", "source_hidden_dim",
                "condition_dim", "cell_line_dim", "condition_branch_dims",
                "condition_after_pool_dims", "state_dims", "time_dims",
                "decoder_dims", "time_frequencies", "condition_output_dropout")}
        if duplicate:
            flow_settings["condition_dim"] = 1028
        flow = ControlOTFlowMatchingSystem(**flow_settings,
            objective="ot_velocity", source_condition=False, direct_condition=True)
        decoder = LatentToGeneDecoder(
            int(cfg["state_dim"]), gene_dim, list(cfg["expression_hidden_dims"]),
            dropout=float(cfg["expression_dropout"]),
            output_activation=cfg["expression_output_activation"])
        if duplicate:
            # The decoder is constructed before widening, exactly as in the
            # original two-drug initialization; widening does not advance RNG.
            from types import SimpleNamespace
            adapter = SimpleNamespace(flow=flow, state_dim=int(cfg["state_dim"]))
            widen_direct_condition(adapter)
            self.direct_concat_metadata = adapter.direct_concat_metadata
        return flow, decoder

    def _load_decoder(self, checkpoint_path, expected_gene_order_sha256):
        payload = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
        if payload.get("model_family") != self.model_config["decoder_checkpoint_family"]:
            raise ValueError("Pretrained decoder family differs from its configured format")
        if not expected_gene_order_sha256:
            raise ValueError("Fresh decoder initialisation requires an explicit gene-order SHA256")
        if payload.get("gene_order_sha256") != expected_gene_order_sha256:
            raise ValueError("Pretrained decoder gene order differs from the active dataset")
        state = payload.get("decoder_state_dict")
        if state is None:
            source = payload.get("model", payload.get("model_state_dict", payload.get("state_dict")))
            if source is None:
                raise ValueError("Pretrained decoder has no model state")
            state = {key[len("decoder."):]: value for key, value in source.items()
                     if key.startswith("decoder.")}
            if not state:
                state = source
        self.decoder.load_state_dict(state, strict=True)

    def _condition(self, batch, count):
        condition = batch["condition"]
        if condition.ndim == 1:
            condition = condition[None]
        if condition.shape[1] != int(self.model_config["condition_dim"]):
            raise ValueError("Condition feature dimension differs from the model")
        if condition.shape[0] == 1:
            condition = condition.expand(count, -1)
        if condition.shape[0] != count:
            # One condition can have different Flow and decoder batch sizes.
            if not torch.equal(condition, condition[:1].expand_as(condition)):
                raise ValueError("A basic MLP loss batch must contain one condition")
            condition = condition[:1].expand(count, -1)
        # One condition mask per loss/rollout; RK4 reuses it at every substep.
        # Materialize only when dropout is active, matching the native concat.
        if self.flow.perturbation_encoder.dropout.p:
            condition = condition.contiguous()
        return self.flow.perturbation_encoder.dropout(condition)

    def predict_mean(self, batch):
        mean = self.mean_head.predict_mean(batch["head_kernel"], batch["head_stratum"])
        return mean[0] if mean.ndim == 2 and len(mean) == 1 else mean

    def _flow_losses(self, batch):
        source, target = batch["source"].float(), batch["target"].float()
        if source.shape != target.shape or source.ndim != 2:
            raise ValueError("OT paired source and target must have matching [cells, latent] shapes")
        time = batch["time"].to(source)
        state = ((1 - time[:, None]) * source + time[:, None] * target
                 + float(self.model_config["path_sigma"]) * batch["noise"].to(source))
        velocity = self.flow.velocity(state, None, self._condition(batch, len(source)), time).float()
        truth = target - source
        prediction_mean, target_mean = velocity.mean(0), truth.mean(0)
        residual = ((velocity - prediction_mean) - (truth - target_mean)).square().mean()
        population_mean = batch["mean_delta"].to(source).detach()
        if population_mean.shape != (source.shape[1],):
            raise ValueError("Population latent mean delta must follow the complete latent axis")
        mean = (prediction_mean - population_mean).square().mean()
        return residual, mean

    def _rollout(self, source, batch, time_points):
        return rk4_rollout(self.flow.velocity, source, None,
                           self._condition(batch, len(source)), time_points=int(time_points))

    def forward(self, batch, time_points=80, *, return_loss_tensors=False):
        flow_residual, flow_mean = self._flow_losses(batch)
        with torch.no_grad():
            endpoint = self._rollout(batch["decoder_source"].float(), batch, time_points)
        # Decoder dropout follows the original MLP training mode. Formal
        # prediction uses eval mode. No graph through the endpoint is retained.
        expression = self.decoder(endpoint.detach()).float()
        mean = self.predict_mean(batch)
        prediction, composition = compose_expression(expression, mean, detach_mean=True,
                                                     return_parts=True)
        truth = batch["decoder_target"].detach().float()
        # Force the train MMD out of BF16 autocast; its bandwidth uses real
        # training treated cells only. Evaluation uses its locked scorer.
        with torch.autocast(device_type=expression.device.type, enabled=False):
            sigmas = median_sigmas(truth, scales=(0.5, 1.0, 2.0, 4.0))
            decoder_mmd = mmd2_unbiased_multi_sigma(prediction.float(), truth, sigmas=sigmas)
        head_loss, head_parts = self.mean_head.loss(return_parts=True)
        weights = self.loss_weights
        weighted_residual = weights["flow_residual"] * flow_residual
        weighted_flow_mean = weights["flow_mean"] * flow_mean
        weighted_mmd = weights["decoder_mmd"] * decoder_mmd
        weighted_head = weights["head"] * head_loss
        flow_loss = weighted_residual + weighted_flow_mean
        total = flow_loss + weighted_mmd + weighted_head.float()
        values = dict(residual_flow_loss=flow_residual, mean_shift_loss=flow_mean,
                      weighted_residual_flow_loss=weighted_residual,
                      weighted_mean_shift_loss=weighted_flow_mean, flow_loss=flow_loss,
                      decoder_mmd_loss=decoder_mmd, weighted_decoder_mmd_loss=weighted_mmd,
                      decoder_loss=weighted_mmd, head_loss=head_loss,
                      weighted_head_loss=weighted_head, **head_parts, total_loss=total,
                      zero_decoder_mean_fraction=composition["zero_mean_fraction"])
        parts = {key: value.detach() for key, value in values.items()}
        if return_loss_tensors:
            return total, parts, {"flow_loss": flow_loss, "head_loss": weighted_head,
                                  "decoder_loss": weighted_mmd}
        return total, parts

    @torch.no_grad()
    def predict(self, batch, time_points=80, *, return_parts=False):
        was_training = self.training
        self.eval()
        try:
            endpoint = self._rollout(batch["source"].float(), batch, time_points)
            expression = self.decoder(endpoint).float()
            mean = self.predict_mean(batch)
            prediction = compose_expression(expression, mean)
            if return_parts:
                return dict(prediction=prediction, raw_expression=expression,
                            mean=mean, endpoint=endpoint)
            return prediction
        finally:
            self.train(was_training)

    def parameter_groups(self):
        return {"flow": [p for p in self.flow.parameters() if p.requires_grad],
                "decoder": [p for p in self.decoder.parameters() if p.requires_grad],
                "head": [p for p in self.mean_head.parameters() if p.requires_grad]}

    def export_spec(self):
        return dict(model_family=self.model_family, model_config=copy.deepcopy(self.model_config),
                    mean_assets={**copy.deepcopy(self._extra_mean_assets), **self.mean_head.export_assets()},
                    loss_weights=dict(self.loss_weights), gene_order_sha256=self.gene_order_sha256)

    @classmethod
    def from_checkpoint(cls, checkpoint, *, map_location="cpu"):
        payload = (torch.load(str(checkpoint), map_location=map_location, weights_only=False)
                   if isinstance(checkpoint, (str, Path)) else checkpoint)
        if payload.get("schema") != cls.checkpoint_schema:
            raise ValueError("Unsupported basic MLP + mean checkpoint schema")
        spec = payload.get("model_spec", payload)
        family = spec.get("model_family", cls.model_family)
        if family not in {cls.model_family, "sciplex3_mlp_mean_mmd_v1", "combosciplex_mlp_mean_mmd_v1"}:
            raise ValueError("Checkpoint model family differs")
        settings = dict(spec["model_config"])
        if family == "combosciplex_mlp_mean_mmd_v1":
            settings.setdefault("initialization_policy", "duplicate_drug_slots")
            settings.setdefault("decoder_checkpoint_family", "MAP_true_SE_decoder_reconstruction")
        with torch.random.fork_rng(devices=[]):
            model = cls(settings, spec["mean_assets"],
                        expected_gene_order_sha256=spec.get("gene_order_sha256"),
                        loss_weights=spec.get("loss_weights"))
        model.load_state_dict(payload["model"], strict=True)
        return model


SciPlex3MLPMeanModel = MLPMeanSystem
