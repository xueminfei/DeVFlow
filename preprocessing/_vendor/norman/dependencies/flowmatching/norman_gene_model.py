"""Norman gene-perturbation condition branch for MAP-Flow.

This module is intentionally additive.  It does not modify the existing drug
condition encoder or the shared MAP-Flow velocity implementation.  The gene
branch produces a 1024-dimensional condition and reuses
``MAPCellFlowVelocity`` with ``condition_dim=1024``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from model.components import ResidualProjector
from model.gene_decoders import LatentToGeneDecoder
from .control_model import ControlConditionFusion, MAPCellFlowVelocity
from .model import adaptive_rollout, rk4_rollout
from .losses import median_sigmas, mmd2_unbiased_multi_sigma
from .ot_matching import pair_population


MAP_GENE_INPUT_DIM = 2048 + 5120
MAP_GENE_OUTPUT_DIM = 1024
SE_GENE_DIM = 2048
ESM2_GENE_DIM = 5120


def _checkpoint_state(checkpoint):
    if not isinstance(checkpoint, dict):
        raise ValueError("checkpoint must contain a state dictionary")
    for key in ("model_state_dict", "state_dict", "model"):
        value = checkpoint.get(key)
        if isinstance(value, dict):
            return value
    return checkpoint


class FrozenMAPGeneProjector(nn.Module):
    """Frozen gene projector with explicit MAP-KG and MAP-predictor modes.

    projector_kind="mapkg" loads the independent MAP-KG projector
    (ESM2 5120 -> 1024; hidden size 1536).  projector_kind="map_predictor"
    is a documented fallback for the MAP predictor projector (SE gene token
    2048 + ESM2 5120 -> 1024; hidden size 2048).  The host currently contains
    the latter checkpoint but no MAP-KG gene-projector checkpoint, so callers
    must choose the mode explicitly rather than silently conflating them.
    """

    def __init__(self, checkpoint_path: str | Path, projector_kind: str = "map_predictor"):
        super().__init__()
        kind = str(projector_kind)
        if kind not in {"mapkg", "map_predictor"}:
            raise ValueError("projector_kind must be mapkg or map_predictor")
        self.projector_kind = kind
        self.input_dim = ESM2_GENE_DIM if kind == "mapkg" else MAP_GENE_INPUT_DIM
        self.requires_se_token = kind == "map_predictor"
        hidden_dim = 1536 if kind == "mapkg" else 2048
        self.projector = ResidualProjector(
            input_dim=self.input_dim,
            output_dim=MAP_GENE_OUTPUT_DIM,
            hidden_dim=hidden_dim,
        )
        checkpoint = torch.load(
            str(checkpoint_path), map_location="cpu", weights_only=False
        )
        state = _checkpoint_state(checkpoint)
        target = self.projector.state_dict()
        loaded = {}
        if kind == "mapkg":
            prefixes = ("gene_projector.", "module.gene_projector.", "model.gene_projector.")
            missing_label = "gene_projector"
        else:
            prefixes = (
                "pert_model.gene_tokens_projector.",
                "module.pert_model.gene_tokens_projector.",
                "gene_tokens_projector.",
            )
            missing_label = "gene_tokens_projector"
        for key in target:
            source_key = next(
                (prefix + key for prefix in prefixes if prefix + key in state), None
            )
            if source_key is None:
                raise KeyError(f"checkpoint is missing {missing_label}.{key}")
            loaded[key] = state[source_key]
        self.projector.load_state_dict(loaded, strict=True)
        self.projector.requires_grad_(False)
        self.projector.eval()

    def train(self, mode: bool = True):
        # Parent model.train() recursively reaches this module.  The
        # projector is frozen and must remain in eval mode so its dropout does
        # not make target/background tokens stochastic.
        super().train(mode)
        self.projector.eval()
        return self

    @torch.no_grad()
    def forward_background(
        self,
        se_tokens: Optional[torch.Tensor],
        esm_tokens: torch.Tensor,
    ) -> torch.Tensor:
        if esm_tokens.ndim != 3 or esm_tokens.size(-1) != ESM2_GENE_DIM:
            raise ValueError(f"esm_tokens must be [B,G,{ESM2_GENE_DIM}]")
        if self.requires_se_token:
            if se_tokens is None or se_tokens.ndim != 3 or se_tokens.size(-1) != SE_GENE_DIM:
                raise ValueError(f"se_tokens must be [B,G,{SE_GENE_DIM}] for map_predictor")
            if esm_tokens.shape[:2] != se_tokens.shape[:2]:
                raise ValueError("SE and ESM background token shapes must agree")
            values = torch.cat((se_tokens, esm_tokens), dim=-1)
        else:
            values = esm_tokens
        return self.projector(values)

    def forward_target(
        self,
        esm_tokens: torch.Tensor,
        target_se_token: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if esm_tokens.ndim != 3 or esm_tokens.size(-1) != ESM2_GENE_DIM:
            raise ValueError(f"esm_tokens must be [B,K,{ESM2_GENE_DIM}]")
        if not self.requires_se_token:
            return self.projector(esm_tokens)
        if target_se_token is None:
            target_se_token = esm_tokens.new_zeros(
                esm_tokens.size(0), esm_tokens.size(1), SE_GENE_DIM
            )
        if target_se_token.shape != (*esm_tokens.shape[:2], SE_GENE_DIM):
            raise ValueError(f"target_se_token must be [B,K,{SE_GENE_DIM}]")
        return self.projector(torch.cat((target_se_token, esm_tokens), dim=-1))


class CrossAttentionBlock(nn.Module):
    """Pre-norm target-query/background-memory cross-attention block."""

    def __init__(self, dim: int = 1024, heads: int = 8, dropout: float = 0.1,
                 ff_multiplier: int = 2):
        super().__init__()
        self.query_norm = nn.LayerNorm(dim)
        self.memory_norm = nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attention_dropout = nn.Dropout(dropout)
        self.ff_norm = nn.LayerNorm(dim)
        hidden = int(dim * ff_multiplier)
        self.ff = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        query: torch.Tensor,
        memory: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        q = self.query_norm(query)
        m = self.memory_norm(memory)
        attended, weights = self.attention(
            q,
            m,
            m,
            key_padding_mask=key_padding_mask,
            need_weights=return_attention,
            average_attn_weights=False,
        )
        query = query + self.attention_dropout(attended)
        query = query + self.ff(self.ff_norm(query))
        return query, weights if return_attention else None


class SymmetricDoubleGeneComposer(nn.Module):
    """Permutation-invariant composer for two gene contexts."""

    def __init__(self, dim: int = 1024, hidden_dim: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.dim = int(dim)
        self.net = nn.Sequential(
            nn.LayerNorm(3 * self.dim),
            nn.Linear(3 * self.dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.dim),
        )

    def forward(self, contexts: torch.Tensor) -> torch.Tensor:
        if contexts.ndim != 3 or contexts.size(1) != 2 or contexts.size(2) != self.dim:
            raise ValueError(f"contexts must be [B,2,{self.dim}]")
        first, second = contexts[:, 0], contexts[:, 1]
        symmetric = torch.cat((first + second, (first - second).abs(), first * second), dim=-1)
        return first + second + self.net(symmetric)


class GenePerturbationEncoder(nn.Module):
    """1024-dimensional Norman gene condition encoder.

    Inputs can be supplied as frozen MAP-projected tokens (``*_projected``),
    which is useful for cached training, or as ESM2/SE tokens together with a
    ``FrozenMAPGeneProjector``.  The latter keeps the external MAP projector
    frozen and trains only the Norman interaction layers.
    """

    def __init__(
        self,
        map_projector: Optional[FrozenMAPGeneProjector] = None,
        dim: int = 1024,
        heads: int = 8,
        attention_layers: int = 2,
        dropout: float = 0.1,
        target_role_trainable: bool = True,
    ):
        super().__init__()
        if int(dim) != MAP_GENE_OUTPUT_DIM:
            raise ValueError("the MAP-compatible gene branch currently requires dim=1024")
        if attention_layers < 1:
            raise ValueError("attention_layers must be positive")
        self.dim = int(dim)
        self.map_projector = map_projector
        self.target_role = nn.Parameter(
            torch.zeros(1, 1, SE_GENE_DIM), requires_grad=bool(target_role_trainable)
        )
        self.cross_attention = nn.ModuleList(
            [CrossAttentionBlock(dim=self.dim, heads=heads, dropout=dropout)
             for _ in range(int(attention_layers))]
        )
        self.pair_composer = SymmetricDoubleGeneComposer(
            dim=self.dim, hidden_dim=2 * self.dim, dropout=dropout
        )
        self.condition_projector = nn.Sequential(
            nn.LayerNorm(self.dim),
            nn.Linear(self.dim, 2 * self.dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * self.dim, self.dim),
        )

    def _project_target(
        self,
        target_gene_esm: Optional[torch.Tensor],
        target_projected: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if target_projected is not None:
            if target_projected.ndim == 2:
                target_projected = target_projected[:, None, :]
            if target_projected.ndim != 3 or target_projected.size(-1) != self.dim:
                raise ValueError(f"target_projected must be [B,K,{self.dim}]")
            return target_projected
        if self.map_projector is None or target_gene_esm is None:
            raise ValueError("provide target_projected or map_projector + target_gene_esm")
        role = self.target_role.to(device=target_gene_esm.device, dtype=target_gene_esm.dtype)
        role = role.expand(target_gene_esm.size(0), target_gene_esm.size(1), -1)
        # The projector is frozen; the optional target role is the only
        # trainable input-side parameter in the MAP-predictor fallback path.
        return self.map_projector.forward_target(target_gene_esm, role)

    def _project_background(
        self,
        background_se_tokens: Optional[torch.Tensor],
        background_gene_esm: Optional[torch.Tensor],
        background_projected: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if background_projected is not None:
            if background_projected.ndim != 3 or background_projected.size(-1) != self.dim:
                raise ValueError(f"background_projected must be [B,G,{self.dim}]")
            return background_projected
        if self.map_projector is None or background_se_tokens is None or background_gene_esm is None:
            raise ValueError(
                "provide background_projected or map_projector + background_se_tokens + background_gene_esm"
            )
        with torch.no_grad():
            return self.map_projector.forward_background(
                background_se_tokens, background_gene_esm
            )

    def forward(
        self,
        target_gene_esm: Optional[torch.Tensor] = None,
        background_se_tokens: Optional[torch.Tensor] = None,
        background_gene_esm: Optional[torch.Tensor] = None,
        target_projected: Optional[torch.Tensor] = None,
        background_projected: Optional[torch.Tensor] = None,
        background_padding_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ):
        query = self._project_target(target_gene_esm, target_projected)
        memory = self._project_background(
            background_se_tokens, background_gene_esm, background_projected
        )
        if query.size(0) != memory.size(0):
            raise ValueError("target and background batches must match")
        contexts = query
        attention_maps = []
        for block in self.cross_attention:
            contexts, weights = block(
                contexts,
                memory,
                key_padding_mask=background_padding_mask,
                return_attention=return_attention,
            )
            if return_attention:
                attention_maps.append(weights)
        if contexts.size(1) == 1:
            condition = contexts[:, 0]
        elif contexts.size(1) == 2:
            condition = self.pair_composer(contexts)
        else:
            raise ValueError("Norman gene condition supports one or two target genes")
        condition = condition + self.condition_projector(condition)
        if return_attention:
            return condition, contexts, attention_maps
        return condition


class NormanGeneFlowMatchingSystem(nn.Module):
    """Norman gene branch that reuses the unchanged MAP velocity implementation."""

    model_family = "map_norman_gene_cross_attention_otfm_v1"

    def __init__(
        self,
        state_dim: int = 2048,
        expression_dim: int = 2000,
        map_projector: Optional[FrozenMAPGeneProjector] = None,
        path_sigma: float = 0.5,
        source_dim: int = 1024,
        source_hidden_dim: int = 2048,
        gene_condition_dim: int = 1024,
        attention_heads: int = 8,
        attention_layers: int = 2,
        attention_dropout: float = 0.1,
        target_role_trainable: bool = True,
        velocity_state_dims=(2048, 2048, 2048),
        velocity_time_dims=(1024, 1024, 1024),
        velocity_decoder_dims=(4096, 4096, 4096),
        time_frequencies: int = 1024,
        decoder_hidden_dims=(512, 1024),
        decoder_dropout: float = 0.1,
        decoder_output_activation: str = "relu",
    ):
        super().__init__()
        if int(gene_condition_dim) != 1024:
            raise ValueError("gene_condition_dim must be 1024 for the MAP-compatible branch")
        self.state_dim = int(state_dim)
        self.expression_dim = int(expression_dim)
        self.path_sigma = float(path_sigma)
        self.source_dim = int(source_dim)
        self.source_fusion = ControlConditionFusion(
            hidden_dim=source_hidden_dim, output_dim=source_dim, dropout=0.1
        )
        self.gene_condition = GenePerturbationEncoder(
            map_projector=map_projector,
            dim=gene_condition_dim,
            heads=attention_heads,
            attention_layers=attention_layers,
            dropout=attention_dropout,
            target_role_trainable=target_role_trainable,
        )
        self.velocity = MAPCellFlowVelocity(
            state_dim=self.state_dim,
            source_dim=self.source_dim,
            condition_dim=gene_condition_dim,
            state_dims=tuple(velocity_state_dims),
            time_dims=tuple(velocity_time_dims),
            decoder_dims=tuple(velocity_decoder_dims),
            time_frequencies=int(time_frequencies),
            use_source_condition=True,
        )
        self.decoder = LatentToGeneDecoder(
            latent_dim=self.state_dim,
            gene_dim=self.expression_dim,
            hidden_dims=list(decoder_hidden_dims),
            dropout=float(decoder_dropout),
            output_activation=str(decoder_output_activation),
        )

    @staticmethod
    def _take_condition(value: Optional[torch.Tensor], index: torch.Tensor, count: int):
        if value is None:
            return None
        if value.size(0) == 1:
            return value.expand(count, *value.shape[1:])
        if value.size(0) != count:
            raise ValueError(
                f"condition batch must be 1 or {count}, got {value.size(0)}"
            )
        return value.index_select(0, index)

    def _pair(
        self,
        source_state: torch.Tensor,
        source_h: torch.Tensor,
        target_state: torch.Tensor,
        condition_values,
        epsilon: float,
        max_iterations: int,
        tolerance: float,
        generator=None,
    ):
        if source_state.ndim != 2 or target_state.ndim != 2:
            raise ValueError("source_state and target_state must be [N,D]")
        if source_state.size(1) != self.state_dim or target_state.size(1) != self.state_dim:
            raise ValueError(f"states must have dimension {self.state_dim}")
        if source_h is None or source_h.shape != (source_state.size(0), 2048):
            raise ValueError("source_h must be [N,2048]")
        source_index, target_index, stats = pair_population(
            source_state,
            target_state,
            source_state.size(0),
            epsilon,
            max_iterations,
            tolerance,
            generator,
        )
        paired_values = []
        for value in condition_values:
            paired_values.append(
                self._take_condition(value, source_index, source_state.size(0))
            )
        return (
            source_state.index_select(0, source_index),
            source_h.index_select(0, source_index),
            target_state.index_select(0, target_index),
            paired_values,
            stats,
        )

    def _gene_condition(self, values):
        return self.gene_condition(**values)

    @staticmethod
    def _shared_gene_kwargs(values: dict) -> bool:
        tensors = [value for value in values.values() if value is not None]
        return bool(tensors) and all(value.size(0) == 1 for value in tensors)

    def _gene_condition_for_cells(self, values: dict, count: int) -> torch.Tensor:
        # A control-pool mean background and the perturbation targets are shared
        # by every cell of a condition. Encode that context once; expanding the
        # resulting 1024-vector keeps gradients while avoiding B copies of the
        # 2047-token attention memory. Per-cell caches retain their old path.
        if self._shared_gene_kwargs(values):
            return self._gene_condition(values).expand(count, -1)
        return self._gene_condition(self._expand_gene_kwargs(values, count))

    @staticmethod
    def _expand_gene_kwargs(values: dict, count: int) -> dict:
        expanded = {}
        for key, value in values.items():
            if value is None:
                expanded[key] = None
            elif value.size(0) == 1:
                expanded[key] = value.expand(count, *value.shape[1:])
            elif value.size(0) == count:
                expanded[key] = value
            else:
                raise ValueError(f"{key} batch must be 1 or {count}, got {value.size(0)}")
        return expanded

    def flow_matching_loss(
        self,
        source_state: torch.Tensor,
        source_h: torch.Tensor,
        target_state: torch.Tensor,
        gene_condition_kwargs: dict,
        epsilon: float = 1.0,
        max_iterations: int = 2000,
        tolerance: float = 1e-3,
        mean_shift_target: Optional[torch.Tensor] = None,
        mean_shift_weight: float = 1.0,
        generator=None,
        return_components: bool = False,
    ):
        condition_values = [
            value for value in (
                gene_condition_kwargs.get("target_gene_esm"),
                gene_condition_kwargs.get("background_se_tokens"),
                gene_condition_kwargs.get("background_gene_esm"),
                gene_condition_kwargs.get("target_projected"),
                gene_condition_kwargs.get("background_projected"),
                gene_condition_kwargs.get("background_padding_mask"),
            ) if value is not None
        ]
        shared_condition = self._shared_gene_kwargs(gene_condition_kwargs)
        # Keep per-cell caches source-aligned. Shared group tokens need no
        # cell-wise pairing and stay at batch one through the condition encoder.
        paired = self._pair(
            source_state, source_h, target_state,
            [] if shared_condition else condition_values,
            epsilon, max_iterations, tolerance, generator,
        )
        source_p, source_h_p, target_p, paired_values, stats = paired
        iterator = iter(paired_values)
        paired_kwargs = {}
        for key in (
            "target_gene_esm", "background_se_tokens", "background_gene_esm",
            "target_projected", "background_projected", "background_padding_mask",
        ):
            original = gene_condition_kwargs.get(key)
            paired_kwargs[key] = (
                original if shared_condition else next(iterator)
            ) if original is not None else None
        count = source_p.size(0)
        gene_condition = self._gene_condition_for_cells(paired_kwargs, count)
        source_condition = self.source_fusion(source_h_p, None)
        time = torch.rand(count, device=source_p.device, dtype=source_p.dtype, generator=generator)
        noise = torch.randn(source_p.shape, device=source_p.device, dtype=source_p.dtype, generator=generator)
        state = (1.0 - time[:, None]) * source_p + time[:, None] * target_p + self.path_sigma * noise
        target_velocity = target_p - source_p
        prediction = self.velocity(state, source_condition, gene_condition, time).float()
        target_velocity = target_velocity.float()
        prediction_mean = prediction.mean(dim=0)
        target_mean = target_velocity.mean(dim=0)
        if mean_shift_target is None:
            stable_mean = target_mean
        else:
            if mean_shift_target.shape != (self.state_dim,):
                raise ValueError(f"mean_shift_target must be [{self.state_dim}]")
            stable_mean = mean_shift_target.float()
        residual_loss = ((prediction - prediction_mean) - (target_velocity - target_mean)).square().mean()
        mean_shift_loss = (prediction_mean - stable_mean).square().mean()
        loss = residual_loss + float(mean_shift_weight) * mean_shift_loss
        if return_components:
            return loss, {
                "flow_loss": loss,
                "residual_flow_loss": residual_loss,
                "mean_shift_loss": mean_shift_loss,
                "ot_stats": stats,
            }
        return loss

    def forward_joint(
        self,
        source_state: torch.Tensor,
        source_h: torch.Tensor,
        target_state: torch.Tensor,
        target_expression: torch.Tensor,
        gene_condition_kwargs: dict,
        epsilon: float = 1.0,
        max_iterations: int = 2000,
        tolerance: float = 1e-3,
        mean_shift_target: Optional[torch.Tensor] = None,
        mean_shift_weight: float = 1.0,
        endpoint_mean_target: Optional[torch.Tensor] = None,
        decoder_mean_loss_weight: float = 8.0,
        decoder_mmd_loss_weight: float = 0.0,
        mmd_cells: Optional[int] = None,
        endpoint_time_points: int = 80,
        generator=None,
        adaptive: bool = False,
        adaptive_rtol: float = 1e-5,
        adaptive_atol: float = 1e-5,
    ):
        """Joint Stage-2 loss with one shared OT pairing.

        The target state and target expression are indexed by the same OT
        target index. The differentiable endpoint rollout therefore sends the
        decoder objective to both the thawed decoder and the velocity branch.
        """
        if target_expression.ndim != 2 or target_expression.size(0) != target_state.size(0):
            raise ValueError('target_expression rows must align with target_state')
        condition_values = [
            gene_condition_kwargs.get('target_gene_esm'),
            gene_condition_kwargs.get('background_se_tokens'),
            gene_condition_kwargs.get('background_gene_esm'),
            gene_condition_kwargs.get('target_projected'),
            gene_condition_kwargs.get('background_projected'),
            gene_condition_kwargs.get('background_padding_mask'),
        ]
        source_index, target_index, stats = pair_population(
            source_state, target_state, source_state.size(0),
            epsilon, max_iterations, tolerance, generator,
        )
        source_p = source_state.index_select(0, source_index)
        source_h_p = source_h.index_select(0, source_index)
        target_p = target_state.index_select(0, target_index)
        target_expr_p = target_expression.index_select(0, target_index)
        paired_kwargs = {}
        shared_condition = self._shared_gene_kwargs(gene_condition_kwargs)
        for key, value in zip((
            'target_gene_esm', 'background_se_tokens', 'background_gene_esm',
            'target_projected', 'background_projected', 'background_padding_mask',
        ), condition_values):
            paired_kwargs[key] = (
                None if value is None else value if shared_condition else self._take_condition(
                    value, source_index, source_state.size(0)
                )
            )
        count = source_p.size(0)
        gene_condition = self._gene_condition_for_cells(paired_kwargs, count)
        source_condition = self.source_fusion(source_h_p, None)
        time = torch.rand(count, device=source_p.device, dtype=source_p.dtype, generator=generator)
        noise = torch.randn(source_p.shape, device=source_p.device, dtype=source_p.dtype, generator=generator)
        state = (1.0 - time[:, None]) * source_p + time[:, None] * target_p + self.path_sigma * noise
        target_velocity = (target_p - source_p).float()
        prediction = self.velocity(state, source_condition, gene_condition, time).float()
        prediction_mean = prediction.mean(dim=0)
        target_mean = target_velocity.mean(dim=0)
        if mean_shift_target is None:
            stable_mean = target_mean
        else:
            if mean_shift_target.shape != (self.state_dim,):
                raise ValueError(f'mean_shift_target must be [{self.state_dim}]')
            stable_mean = mean_shift_target.detach().float()
        residual_loss = ((prediction - prediction_mean) - (target_velocity - target_mean)).square().mean()
        mean_shift_loss = (prediction_mean - stable_mean).square().mean()
        flow_loss = residual_loss + float(mean_shift_weight) * mean_shift_loss
        endpoint_state = self.rollout(
            source_p, source_h_p, paired_kwargs, time_points=endpoint_time_points,
            adaptive=adaptive, rtol=adaptive_rtol, atol=adaptive_atol,
        )
        decoder_loss, decoder_parts = self.decoder_loss(
            endpoint_state, target_expr_p, endpoint_mean_target=endpoint_mean_target,
            mean_loss_weight=decoder_mean_loss_weight,
            mmd_loss_weight=decoder_mmd_loss_weight, mmd_cells=mmd_cells,
        )
        total_loss = flow_loss + decoder_loss
        components = {
            'total_loss': total_loss,
            'flow_loss': flow_loss,
            'residual_flow_loss': residual_loss,
            'mean_shift_loss': mean_shift_loss,
            'endpoint_state': endpoint_state,
            'ot_stats': stats,
            'target_expression_paired': target_expr_p,
        }
        components.update(decoder_parts)
        return total_loss, components

    def decoder_loss(
        self,
        endpoint_state: torch.Tensor,
        target_expression: torch.Tensor,
        endpoint_mean_target: Optional[torch.Tensor] = None,
        mean_loss_weight: float = 8.0,
        mmd_loss_weight: float = 0.0,
        mmd_cells: Optional[int] = None,
    ):
        """Compute the decoder objective used by the thawed SciPlex3 joint run.

        SciPlex3 joint decoder training uses centered per-cell expression MSE
        plus 8 times full-condition population-mean MSE. The
        endpoint_mean_target is the mean over all treated cells in a condition,
        rather than the mean of an OT subset. MMD is optional and zero by default.
        """
        if endpoint_state.ndim != 2 or endpoint_state.size(1) != self.state_dim:
            raise ValueError(f'endpoint_state must be [N,{self.state_dim}]')
        if target_expression.ndim != 2 or target_expression.shape != (
            endpoint_state.size(0), self.expression_dim
        ):
            raise ValueError(
                f'target_expression must be [N,{self.expression_dim}] aligned to endpoint_state'
            )
        prediction = self.decoder(endpoint_state).float()
        target = target_expression.float()
        prediction_centered = prediction - prediction.mean(dim=0, keepdim=True)
        target_centered = target - target.mean(dim=0, keepdim=True)
        decoder_residual_loss = F.mse_loss(prediction_centered, target_centered)
        if endpoint_mean_target is None:
            mean_target = target.mean(dim=0)
        else:
            if endpoint_mean_target.shape != (self.expression_dim,):
                raise ValueError(
                    f'endpoint_mean_target must be [{self.expression_dim}], '
                    f'got {tuple(endpoint_mean_target.shape)}'
                )
            mean_target = endpoint_mean_target.detach().float()
        decoder_mean_loss = F.mse_loss(prediction.mean(dim=0), mean_target)
        mean_weight = float(mean_loss_weight)
        if mean_weight < 0:
            raise ValueError('mean_loss_weight must be nonnegative')
        mmd_loss = prediction.new_zeros((), dtype=torch.float32)
        n_mmd = 0
        if float(mmd_loss_weight) != 0.0:
            n_mmd = prediction.size(0) if mmd_cells is None else min(
                int(mmd_cells), prediction.size(0)
            )
            if n_mmd < 2:
                raise ValueError('mmd_cells must leave at least two cells')
            target_subset = target[:n_mmd]
            mmd_loss = mmd2_unbiased_multi_sigma(
                prediction[:n_mmd], target_subset,
                median_sigmas(target_subset.detach()),
            )
        weighted_mean = mean_weight * decoder_mean_loss
        weighted_mmd = float(mmd_loss_weight) * mmd_loss
        decoder_loss = decoder_residual_loss + weighted_mean + weighted_mmd
        return decoder_loss, {
            'decoder_loss': decoder_loss,
            'decoder_residual_loss': decoder_residual_loss,
            'decoder_mean_loss': decoder_mean_loss,
            'weighted_decoder_mean_loss': weighted_mean,
            'decoder_mmd_loss': mmd_loss,
            'weighted_decoder_mmd_loss': weighted_mmd,
            'decoder_cell_mse': F.mse_loss(prediction, target),
            'endpoint_expression': prediction,
            'mmd_cells': n_mmd,
        }

    def rollout(
        self,
        source_state: torch.Tensor,
        source_h: torch.Tensor,
        gene_condition_kwargs: dict,
        time_points: int = 80,
        adaptive: bool = False,
        rtol: float = 1e-5,
        atol: float = 1e-5,
    ):
        """Differentiable endpoint rollout for joint flow/decoder training."""
        if source_state.ndim != 2 or source_state.size(1) != self.state_dim:
            raise ValueError(f'source_state must be [N,{self.state_dim}]')
        if source_h is None or source_h.shape != (source_state.size(0), 2048):
            raise ValueError('source_h must be [N,2048]')
        source_condition = self.source_fusion(source_h, None)
        gene_condition = self._gene_condition_for_cells(
            gene_condition_kwargs, source_state.size(0)
        )
        if adaptive:
            return adaptive_rollout(
                self.velocity, source_state, source_condition, gene_condition,
                rtol=rtol, atol=atol,
            )
        return rk4_rollout(
            self.velocity, source_state, source_condition, gene_condition,
            time_points=time_points,
        )

    @torch.no_grad()
    def sample(
        self,
        source_state: torch.Tensor,
        source_h: torch.Tensor,
        gene_condition_kwargs: dict,
        time_points: int = 80,
        adaptive: bool = False,
        rtol: float = 1e-5,
        atol: float = 1e-5,
    ):
        """Inference-only wrapper around rollout."""
        return self.rollout(
            source_state, source_h, gene_condition_kwargs,
            time_points=time_points, adaptive=adaptive,
            rtol=rtol, atol=atol,
        )
