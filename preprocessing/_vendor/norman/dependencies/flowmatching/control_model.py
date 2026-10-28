"""MAP-conditioned CellFlow OTFM in a 100-dimensional PCA state space."""

from __future__ import annotations

import math

import torch
from torch import nn

from .model import adaptive_rollout, rk4_rollout
from .ot_matching import pair_population
from .upstream import ControlConditionFusion


class MLPBlock(nn.Module):
    """Dense SiLU block matching CellFlow's MLPBlock ordering."""

    def __init__(self, input_dim, dims, dropout=0.0, activate_last=True):
        super().__init__()
        self.input_dim = int(input_dim)
        layers = []
        left = int(input_dim)
        for index, right in enumerate(dims):
            layers.append(nn.Linear(left, int(right)))
            if index < len(dims) - 1 or activate_last:
                layers.append(nn.SiLU())
                if dropout:
                    layers.append(nn.Dropout(float(dropout)))
            left = int(right)
        self.net = nn.Sequential(*layers)

    def forward(self, values):
        return self.net(values)


def sinusoidal_time_encoding(time, dimensions=1024, max_period=10000):
    if time.ndim != 1:
        raise ValueError("time must be one-dimensional")
    half = dimensions // 2
    frequencies = torch.exp(
        -math.log(max_period)
        * torch.arange(half, device=time.device, dtype=torch.float32)
        / half
    )
    angles = (time.float() * float(max_period))[:, None] * frequencies[None]
    encoded = torch.cat((torch.cos(angles), torch.sin(angles)), dim=-1)
    if dimensions % 2:
        encoded = torch.cat((encoded, torch.zeros_like(encoded[:, :1])), dim=-1)
    return encoded.to(time.dtype)


class CellFlowPerturbationEncoder(nn.Module):
    """Deterministic CellFlow condition encoder for one-drug perturbations."""

    def __init__(self, drug_dim=1024, cell_line_dim=3, output_dim=64,
                 branch_dims=(256, 256), after_pool_dims=(256, 256),
                 output_dropout=0.9, direct_concat=False):
        super().__init__()
        self.direct_concat = bool(direct_concat)
        self.drug_dim = int(drug_dim)
        self.drug = MLPBlock(drug_dim, branch_dims)
        self.dose = MLPBlock(1, branch_dims)
        self.cell_line = MLPBlock(cell_line_dim, branch_dims)
        joined_dim = 3 * int(branch_dims[-1])
        self.after_pool = MLPBlock(joined_dim, after_pool_dims)
        if not self.direct_concat:
            self.output = nn.Linear(int(after_pool_dims[-1]), int(output_dim))
        self.dropout = nn.Dropout(float(output_dropout))

    def forward(self, drug_embedding, dose_log10, cell_line_onehot):
        if dose_log10.ndim == 1:
            dose_log10 = dose_log10[:, None]
        if drug_embedding.ndim != 2 or drug_embedding.size(1) != self.drug_dim:
            raise ValueError(f"drug_embedding must be [N,{self.drug_dim}]")
        if dose_log10.shape != (drug_embedding.size(0), 1):
            raise ValueError("dose_log10 must be [N] or [N,1]")
        if cell_line_onehot.shape != (drug_embedding.size(0), self.cell_line.input_dim):
            raise ValueError(
                f"cell_line_onehot must be [N,{self.cell_line.input_dim}]"
            )
        if self.direct_concat:
            raw_condition = torch.cat((
                drug_embedding, dose_log10.to(drug_embedding),
                cell_line_onehot.to(drug_embedding),
            ), dim=-1)
            return self.dropout(raw_condition)
        joined = torch.cat((
            self.drug(drug_embedding),
            self.dose(dose_log10.to(drug_embedding)),
            self.cell_line(cell_line_onehot.to(drug_embedding)),
        ), dim=-1)
        return self.dropout(self.output(self.after_pool(joined)))


class MAPCellFlowVelocity(nn.Module):
    """CellFlow concatenation velocity field plus an explicit MAP source branch."""

    def __init__(self, state_dim=100, source_dim=1024, condition_dim=64,
                 state_dims=(2048, 2048, 2048), time_dims=(1024, 1024, 1024),
                 decoder_dims=(4096, 4096, 4096), time_frequencies=1024,
                 use_source_condition=True):
        super().__init__()
        self.state_dim = int(state_dim)
        self.source_dim = int(source_dim)
        self.condition_dim = int(condition_dim)
        self.time_frequencies = int(time_frequencies)
        self.use_source_condition = bool(use_source_condition)
        self.time_encoder = MLPBlock(
            time_frequencies, time_dims, activate_last=False
        )
        decoder_input = (
            int(state_dim) + int(time_dims[-1])
            + (int(source_dim) if self.use_source_condition else 0)
            + int(condition_dim)
        )
        self.decoder_input_dim = decoder_input
        self.decoder = MLPBlock(
            decoder_input, decoder_dims, activate_last=False
        )
        self.output = nn.Linear(int(decoder_dims[-1]), state_dim)

    def forward(self, state, source_condition, perturbation_condition, time):
        if state.ndim != 2 or state.size(1) != self.state_dim:
            raise ValueError(f"state must be [N,{self.state_dim}]")
        if self.use_source_condition and (
            source_condition is None
            or source_condition.shape != (state.size(0), self.source_dim)
        ):
            raise ValueError(f"source_condition must be [N,{self.source_dim}]")
        if perturbation_condition.shape != (state.size(0), self.condition_dim):
            raise ValueError(f"perturbation_condition must be [N,{self.condition_dim}]")
        encoded = torch.cat((
            self.time_encoder(sinusoidal_time_encoding(time, self.time_frequencies)),
            state,
            *((source_condition,) if self.use_source_condition else ()),
            perturbation_condition,
        ), dim=-1)
        return self.output(self.decoder(encoded))


OBJECTIVES = ("ot_velocity", "delta_cfm")


class ControlOTFlowMatchingSystem(nn.Module):
    """OTFM transport from real control cells to perturbed populations."""

    model_family = "map_conditioned_cellflow_otfm_v1"

    def __init__(self, state_dim=100, path_sigma=1.5, source_dim=1024,
                 source_hidden_dim=2048, condition_dim=64,
                 cell_line_dim=3,
                 condition_branch_dims=(256, 256),
                 condition_after_pool_dims=(256, 256),
                 state_dims=(2048, 2048, 2048),
                 time_dims=(1024, 1024, 1024),
                 decoder_dims=(4096, 4096, 4096), time_frequencies=1024,
                 objective="ot_velocity", source_condition=True,
                 direct_condition=False, condition_output_dropout=None):
        super().__init__()
        if objective not in OBJECTIVES:
            raise ValueError(f"objective must be one of {OBJECTIVES}, got {objective!r}")
        self.state_dim = int(state_dim)
        self.path_sigma = float(path_sigma)
        self.cell_line_dim = int(cell_line_dim)
        self.objective = objective
        self.source_condition = bool(source_condition)
        if condition_output_dropout is None:
            condition_output_dropout = 0.0 if direct_condition else 0.9
        if not 0.0 <= float(condition_output_dropout) < 1.0:
            raise ValueError("condition_output_dropout must be in [0,1)")
        if self.source_condition:
            self.source_fusion = ControlConditionFusion(
                hidden_dim=source_hidden_dim, output_dim=source_dim, dropout=0.1
            )
        self.perturbation_encoder = CellFlowPerturbationEncoder(
            cell_line_dim=self.cell_line_dim,
            output_dim=condition_dim, branch_dims=condition_branch_dims,
            after_pool_dims=condition_after_pool_dims,
            output_dropout=float(condition_output_dropout),
            direct_concat=direct_condition,
        )
        self.velocity = MAPCellFlowVelocity(
            state_dim=state_dim, source_dim=source_dim,
            condition_dim=condition_dim, state_dims=state_dims,
            time_dims=time_dims, decoder_dims=decoder_dims,
            time_frequencies=time_frequencies,
            use_source_condition=self.source_condition,
        )

    @staticmethod
    def _validate(source, source_h, source_g, target, use_source=True):
        if source.ndim != 2 or target.ndim != 2 or source.shape != target.shape:
            raise ValueError("paired source and target must have the same [N,D] shape")
        if not use_source:
            return
        if source_h is None or source_h.shape != (source.size(0), 2048):
            raise ValueError("source_h must be [N,2048] SE-600M embeddings")

    @torch.no_grad()
    def pair(self, source, source_h, source_g, target, epsilon=1.0,
             max_iterations=2000, tolerance=1e-3, generator=None):
        if source.ndim != 2 or target.ndim != 2:
            raise ValueError("unpaired populations must be [N,D] and [M,D]")
        if source_h is None or source_h.size(0) != source.size(0):
            raise ValueError("source expression and SE cell embedding rows must be aligned")
        source_index, target_index, stats = pair_population(
            source, target, source.size(0), epsilon, max_iterations, tolerance, generator
        )
        return (
            source.index_select(0, source_index),
            None if source_h is None else source_h.index_select(0, source_index),
            None if source_g is None else source_g.index_select(0, source_index),
            target.index_select(0, target_index),
            stats,
        )

    def flow_matching_loss(self, source, source_h, source_g, target,
                           drug_embedding, dose_log10, cell_line_onehot,
                           generator=None, mean_shift_target=None,
                           mean_shift_weight=1.0, return_components=False):
        self._validate(source, source_h, source_g, target, self.source_condition)
        count = source.size(0)
        time = torch.rand(count, device=source.device, dtype=source.dtype, generator=generator)
        source_condition = (
            self.source_fusion(source_h, source_g) if self.source_condition else None
        )
        perturbation_condition = self.perturbation_encoder(
            drug_embedding, dose_log10, cell_line_onehot
        )
        if self.objective == "delta_cfm":
            # Regress the velocity of the OT-paired delta (target - source)
            # along a straight Gaussian-noise-to-delta path, rather than the
            # velocity of a control-to-target bridge. Conditions still come
            # from the paired source cell, so pairing semantics are unchanged.
            delta = target - source
            noise = torch.randn(
                delta.shape, device=delta.device, dtype=delta.dtype, generator=generator
            )
            state = (1.0 - time[:, None]) * noise + time[:, None] * delta
            target_velocity = delta - noise
        else:
            noise = torch.randn(
                source.shape, device=source.device, dtype=source.dtype, generator=generator
            )
            state = (
                (1.0 - time[:, None]) * source
                + time[:, None] * target
                + self.path_sigma * noise
            )
            target_velocity = target - source
        prediction = self.velocity(
            state, source_condition, perturbation_condition, time
        )
        prediction = prediction.float()
        target_velocity = target_velocity.float()
        prediction_mean = prediction.mean(dim=0)
        target_batch_mean = target_velocity.mean(dim=0)
        if mean_shift_target is None:
            stable_mean_target = target_batch_mean
        else:
            if self.objective != "ot_velocity":
                raise ValueError("external mean-shift targets require ot_velocity objective")
            if mean_shift_target.shape != (self.state_dim,):
                raise ValueError(
                    f"mean_shift_target must be [{self.state_dim}], "
                    f"got {tuple(mean_shift_target.shape)}"
                )
            stable_mean_target = mean_shift_target.float()
        residual_loss = (
            (prediction - prediction_mean)
            - (target_velocity - target_batch_mean)
        ).square().mean()
        mean_shift_loss = (prediction_mean - stable_mean_target).square().mean()
        loss = residual_loss + float(mean_shift_weight) * mean_shift_loss
        if return_components:
            return loss, residual_loss, mean_shift_loss
        return loss

    def forward(self, source, source_h, source_g, target, drug_embedding,
                dose_log10, cell_line_onehot, generator=None, already_paired=False):
        paired = (
            (source, source_h, source_g, target, None)
            if already_paired
            else self.pair(source, source_h, source_g, target, generator=generator)
        )
        loss = self.flow_matching_loss(
            *paired[:4], drug_embedding, dose_log10, cell_line_onehot, generator=generator
        )
        return loss, paired[-1]

    def sample(self, source, source_h, source_g, drug_embedding,
               dose_log10, cell_line_onehot, time_points=20,
               adaptive=False, rtol=1e-5, atol=1e-5, generator=None):
        self._validate(source, source_h, source_g, source, self.source_condition)
        source_condition = (
            self.source_fusion(source_h, source_g) if self.source_condition else None
        )
        perturbation_condition = self.perturbation_encoder(
            drug_embedding, dose_log10, cell_line_onehot
        )
        start = (
            torch.randn(
                source.shape, device=source.device, dtype=source.dtype, generator=generator
            )
            if self.objective == "delta_cfm"
            else source
        )
        if adaptive:
            rolled = adaptive_rollout(
                self.velocity, start, source_condition, perturbation_condition,
                rtol=rtol, atol=atol,
            )
        else:
            rolled = rk4_rollout(
                self.velocity, start, source_condition, perturbation_condition,
                steps=time_points - 1,
            )
        return source + rolled if self.objective == "delta_cfm" else rolled
