"""Scratch MAP-style joint attention with time modulation and an endpoint head.

Only frozen SE/ESM/drug *features* are inputs. No MAP predictor weights are
loaded. Each call recomputes Fourier time features, the time MLP, and every
block's modulation. prepare_context caches only the trainable gene projection
and drug/dose/line embedding within a parameter version; keep its graph for
Flow-loss training and detach it only for decoder-only ODE rollouts.

The full bidirectional sequence is [cell(z_t), drug, control genes]. Padding
is the only attention mask. There are no position embeddings, RoPE, causal
masks, extra learned CLS, or residual gates. Four-dimensional modulation
means four hidden-width vectors (attention shift/scale, MLP shift/scale), not
four scalar parameters. Its zero initialization leaves ordinary residual
attention and MLP branches active.

The CLS head predicts a clean endpoint. velocity = endpoint - ORIGINAL z0;
this is source-relative endpoint parameterization, not (endpoint-z_t)/(1-t).
It retains the existing constant-noise path's target z1-z0. ODE integration
must bind the initial source once, for example with bind_source().
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint


@dataclass(frozen=True)
class PreparedAttentionContext:
    """Per-cell context; gene_valid=True means a real, usable token.

    Do not reuse projected values across parameter updates. A zero expression
    value or gene ID zero does not identify padding. No time-dependent value
    is stored here.
    """

    projected_genes: torch.Tensor
    drug_token: torch.Tensor
    gene_valid: torch.Tensor
    has_padding: bool = True

    def detach(self) -> "PreparedAttentionContext":
        return PreparedAttentionContext(
            self.projected_genes.detach(), self.drug_token.detach(), self.gene_valid, self.has_padding
        )

    def index_select(self, index: torch.Tensor) -> "PreparedAttentionContext":
        """Apply the SAME OT source indices, including repeated indices."""
        if index.ndim != 1 or index.dtype != torch.long:
            raise ValueError("source index must be a one-dimensional int64 tensor")
        index = index.to(device=self.projected_genes.device)
        return PreparedAttentionContext(
            self.projected_genes.index_select(0, index),
            self.drug_token.index_select(0, index),
            self.gene_valid.index_select(0, index),
            self.has_padding,  # Conservative if the selected subset removes all padding.
        )


class RMSNorm(nn.Module):
    def __init__(self, hidden_dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_dim))
        self.eps = float(eps)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value_float = value.float()
        normalized = value_float * torch.rsqrt(
            value_float.square().mean(dim=-1, keepdim=True) + self.eps
        )
        return normalized.to(value.dtype) * self.weight.to(value.dtype)


class GaussianFourierTimeEmbedding(nn.Module):
    """Fixed Gaussian frequencies, sin/cos concatenation, then a learned MLP.

    A frequency count of 256 produces 512 Fourier features. The fixed
    frequencies are checkpointed as a buffer, never optimized.
    """

    def __init__(self, hidden_dim: int, frequencies: int = 256,
                 frequency_scale: float = 1.0):
        super().__init__()
        if frequencies < 1 or frequency_scale <= 0:
            raise ValueError("frequencies and frequency_scale must be positive")
        self.register_buffer(
            "frequencies", torch.randn(frequencies) * float(frequency_scale)
        )
        self.mlp = nn.Sequential(
            nn.Linear(2 * frequencies, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        if time.ndim != 1:
            raise ValueError("time must be [B]")
        angles = time.float()[:, None] * self.frequencies.float()[None, :] * (2 * torch.pi)
        features = torch.cat((angles.sin(), angles.cos()), dim=-1)
        return self.mlp(features.to(dtype=self.mlp[0].weight.dtype))


class CellProjector(nn.Module):
    """The original MAP four-linear-layer cell projection, initialized anew."""

    def __init__(self, cell_dim: int, hidden_dim: int, inner_dim: int,
                 dropout: float = 0.0):
        super().__init__()
        dims = [cell_dim, inner_dim, inner_dim, inner_dim, hidden_dim]
        layers = []
        for index, (left, right) in enumerate(zip(dims[:-1], dims[1:])):
            layers.append(nn.Linear(left, right))
            if index < len(dims) - 2:
                layers.extend((nn.GELU(), nn.Dropout(dropout)))
        self.net = nn.Sequential(*layers)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.net(value)


class GeneResidualProjector(nn.Module):
    """MAP-shaped 7168->2048->1024 projector with linear skip and sigmoid gate."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 dropout: float = 0.0):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim),
            nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, output_dim),
        )
        self.residual_proj = nn.Linear(input_dim, output_dim)
        self.gate = nn.Sequential(nn.Linear(input_dim, output_dim), nn.Sigmoid())

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        transformed = self.mlp(value)
        residual = self.residual_proj(value)
        gate = self.gate(value)
        return gate * transformed + (1 - gate) * residual


class BidirectionalAttention(nn.Module):
    """Full-width Q/K/V, no RoPE, True=allowed SDPA key mask."""

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.num_heads = int(num_heads)
        self.head_dim = hidden_dim // num_heads
        self.dropout = float(dropout)
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, value: torch.Tensor, valid_tokens: torch.Tensor | None) -> torch.Tensor:
        batch, length, hidden = value.shape
        def split(projection):
            return projection(value).reshape(
                batch, length, self.num_heads, self.head_dim
            ).transpose(1, 2)
        q, k, v = split(self.q_proj), split(self.k_proj), split(self.v_proj)
        attended = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None if valid_tokens is None else valid_tokens[:, None, None, :],
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )
        return self.o_proj(attended.transpose(1, 2).reshape(batch, length, hidden))


class SwiGLU(nn.Module):
    def __init__(self, hidden_dim: int, intermediate_dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.up_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.down_proj = nn.Linear(intermediate_dim, hidden_dim, bias=False)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(value)) * self.up_proj(value))


class TimeModulatedMAPBlock(nn.Module):
    """RMSNorm + shift/scale adapted from V-RAE, with ungated MAP residuals."""

    def __init__(self, hidden_dim: int, num_heads: int, intermediate_dim: int,
                 dropout: float = 0.0):
        super().__init__()
        self.attention_norm = RMSNorm(hidden_dim)
        self.mlp_norm = RMSNorm(hidden_dim)
        self.attention = BidirectionalAttention(hidden_dim, num_heads, dropout)
        self.mlp = SwiGLU(hidden_dim, intermediate_dim)
        self.time_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_dim, 4 * hidden_dim))
        nn.init.zeros_(self.time_modulation[-1].weight)
        nn.init.zeros_(self.time_modulation[-1].bias)

    def forward(self, hidden: torch.Tensor, time_embedding: torch.Tensor,
                valid_tokens: torch.Tensor | None) -> torch.Tensor:
        shift_a, scale_a, shift_m, scale_m = self.time_modulation(
            time_embedding
        ).unsqueeze(1).chunk(4, dim=-1)
        attended = self.attention(
            self.attention_norm(hidden) * (1 + scale_a) + shift_a, valid_tokens
        )
        hidden = hidden + attended
        return hidden + self.mlp(self.mlp_norm(hidden) * (1 + scale_m) + shift_m)


class AttentionCellFlowVelocity(nn.Module):
    """Joint cell/drug/gene attention with a source-relative endpoint velocity.

    Production defaults: 2048-D state, 2047 genes, 1024-D tokens, four layers,
    eight heads of 128 dimensions, and a 2688-wide SwiGLU. Gene count is dynamic
    to permit padding and small isolated tests. Predictor dropout is zero by
    default so repeated calls evaluate a deterministic ODE field even in train
    mode. If enabling dropout, use eval mode for ODE prediction. Gene projection
    is chunked before ESM lookup/FP32 conversion; optional non-reentrant
    activation checkpointing covers both the gene projector and all blocks.
    """

    model_family = "scratch_map_attention_time_shift_scale_source_endpoint_v1"

    def __init__(self, cell_dim=2048, raw_gene_dim=2048, esm_dim=5120,
                 drug_dim=1024, cell_line_dim=3, hidden_dim=1024,
                 num_layers=4, num_heads=8, intermediate_dim=2688,
                 cell_projector_dim=None, gene_projector_dim=2048,
                 time_frequencies=256, dropout=0.0, activation_checkpointing=False,
                 gene_projection_chunk_size=256, checkpoint_gene_projection=True):
        super().__init__()
        dimensions = (cell_dim, raw_gene_dim, esm_dim, drug_dim, cell_line_dim,
                      hidden_dim, num_layers, num_heads, intermediate_dim,
                      gene_projector_dim, time_frequencies)
        if any(int(value) < 1 for value in dimensions):
            raise ValueError("all dimensions and layer/head counts must be positive")
        if int(hidden_dim) % int(num_heads):
            raise ValueError("hidden_dim must be divisible by num_heads")
        if not 0 <= float(dropout) < 1:
            raise ValueError("dropout must be in [0,1)")
        self.cell_dim = int(cell_dim)
        self.state_dim = self.cell_dim
        self.raw_gene_dim = int(raw_gene_dim)
        self.esm_dim = int(esm_dim)
        self.drug_dim = int(drug_dim)
        self.cell_line_dim = int(cell_line_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.activation_checkpointing = bool(activation_checkpointing)
        self.checkpoint_gene_projection = bool(checkpoint_gene_projection)
        self.gene_projection_chunk_size = int(gene_projection_chunk_size)
        if self.gene_projection_chunk_size < 1:
            raise ValueError("gene_projection_chunk_size must be positive")
        cell_projector_dim = int(cell_projector_dim or max(cell_dim, hidden_dim))
        if cell_projector_dim < 1:
            raise ValueError("cell_projector_dim must be positive")
        self.cell_projector = CellProjector(
            self.cell_dim, self.hidden_dim, cell_projector_dim, dropout
        )
        self.gene_projector = GeneResidualProjector(
            self.raw_gene_dim + self.esm_dim, int(gene_projector_dim),
            self.hidden_dim, dropout,
        )
        self.drug_projector = (
            nn.Identity() if self.drug_dim == self.hidden_dim
            else nn.Linear(self.drug_dim, self.hidden_dim)
        )
        self.dose_encoder = nn.Sequential(
            nn.Linear(1, self.hidden_dim), nn.SiLU(), nn.Linear(self.hidden_dim, self.hidden_dim)
        )
        self.line_encoder = nn.Sequential(
            nn.Linear(self.cell_line_dim, self.hidden_dim), nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.time_embedder = GaussianFourierTimeEmbedding(self.hidden_dim, int(time_frequencies))
        self.blocks = nn.ModuleList([
            TimeModulatedMAPBlock(self.hidden_dim, int(num_heads), int(intermediate_dim), dropout)
            for _ in range(self.num_layers)
        ])
        self.final_norm = RMSNorm(self.hidden_dim)
        self.project_out = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, self.cell_dim),
        )

    def _batch_feature(self, value, count, width, name, device, dtype):
        if not torch.is_tensor(value) or not value.is_floating_point():
            raise ValueError(f"{name} must be a floating-point tensor")
        if value.device != device:
            raise ValueError(f"{name} must share the gene feature device")
        if value.ndim == 0 and width == 1:
            value = value.reshape(1, 1)
        elif value.ndim == 1:
            value = value[:, None] if width == 1 else value[None, :]
        if value.ndim != 2 or value.shape[1] != width or value.shape[0] not in (1, count):
            raise ValueError(f"{name} must be [B,{width}] or [1,{width}]")
        return value.detach().to(dtype=dtype).expand(count, width)

    def prepare_context(self, raw_gene_tokens, esm_tokens, drug_embedding,
                        dose_log10, cell_line_onehot, gene_valid, *,
                        gene_ids=None, esm_table=None) -> PreparedAttentionContext:
        """Project source features once, preserving predictor-parameter gradients.

        dose_log10 is ALREADY log10(1+dose_nM), not raw nM. ESM can be provided
        densely as [B,L,E] / [L,E], or with esm_tokens=None and external
        esm_table[G,E] plus gene_ids[B,L]. The table is never registered on this
        module. Invalid padding IDs are ignored; ID zero remains a valid gene.
        """
        if (raw_gene_tokens.ndim != 3 or raw_gene_tokens.shape[-1] != self.raw_gene_dim
                or not raw_gene_tokens.is_floating_point() or raw_gene_tokens.shape[1] < 1):
            raise ValueError(f"raw_gene_tokens must be floating-point [B,L,{self.raw_gene_dim}]")
        batch, length, _ = raw_gene_tokens.shape
        device = raw_gene_tokens.device
        parameter = self.gene_projector.mlp[0].weight
        if parameter.device != device:
            raise ValueError("model and raw_gene_tokens must share a device")
        if gene_valid.dtype != torch.bool or gene_valid.shape != (batch, length):
            raise ValueError("gene_valid must be boolean [B,L], True for valid genes")
        if gene_valid.device != device:
            raise ValueError("gene_valid and raw_gene_tokens must share a device")
        if esm_tokens is None:
            if esm_table is None or gene_ids is None:
                raise ValueError("provide esm_tokens or both esm_table and gene_ids")
            if esm_table.ndim != 2 or esm_table.shape[1] != self.esm_dim or not esm_table.is_floating_point():
                raise ValueError(f"esm_table must be floating-point [G,{self.esm_dim}]")
            if gene_ids.shape != (batch, length) or gene_ids.dtype not in (torch.int32, torch.int64):
                raise ValueError("gene_ids must be integer [B,L]")
            if esm_table.device != device or gene_ids.device != device:
                raise ValueError("ESM table, gene IDs and raw genes must share a device")
            safe_ids = gene_ids.to(dtype=torch.long).masked_fill(~gene_valid, 0)
            # Invalid IDs at real tokens raise an indexing error rather than
            # silently changing the cached gene identity.
            if ((safe_ids < 0) | (safe_ids >= esm_table.shape[0])).any():
                raise ValueError("valid gene IDs are outside the ESM table")
        else:
            if esm_table is not None or gene_ids is not None:
                raise ValueError("pass dense esm_tokens OR esm_table/gene_ids, not both")
            if esm_tokens.device != device or not esm_tokens.is_floating_point():
                raise ValueError("esm_tokens must be floating point on the raw gene device")
            if esm_tokens.shape == (length, self.esm_dim):
                esm_tokens = esm_tokens[None].expand(batch, -1, -1)
            if esm_tokens.shape != (batch, length, self.esm_dim):
                raise ValueError(f"esm_tokens must be [B,L,{self.esm_dim}] or [L,{self.esm_dim}]")
        # Gather/cast/concatenate only a bounded number of gene tokens. A dense
        # [B,L,7168] FP32 tensor is never constructed. Non-reentrant checkpoint
        # saves the low-precision chunk inputs, then recomputes casts/projection.
        projected_chunks = []
        use_checkpoint = (self.activation_checkpointing and self.checkpoint_gene_projection
                          and self.training and torch.is_grad_enabled())
        for start in range(0, batch * length, self.gene_projection_chunk_size):
            flat_index = torch.arange(start, min(start + self.gene_projection_chunk_size,
                                                batch * length), device=device)
            batch_index = torch.div(flat_index, length, rounding_mode="floor")
            token_index = flat_index.remainder(length)
            raw_chunk = raw_gene_tokens.detach()[batch_index, token_index]
            if esm_tokens is None:
                esm_chunk = F.embedding(safe_ids[batch_index, token_index], esm_table.detach())
            else:
                esm_chunk = esm_tokens.detach()[batch_index, token_index]
            valid_chunk = gene_valid[batch_index, token_index]
            if use_checkpoint:
                projected = checkpoint(self._project_gene_chunk, raw_chunk, esm_chunk,
                                       valid_chunk, use_reentrant=False)
            else:
                projected = self._project_gene_chunk(raw_chunk, esm_chunk, valid_chunk)
            projected_chunks.append(projected)
        genes = torch.cat(projected_chunks, dim=0).reshape(batch, length, self.hidden_dim)
        drug = self._batch_feature(drug_embedding, batch, self.drug_dim,
                                   "drug_embedding", device, parameter.dtype)
        dose = self._batch_feature(dose_log10, batch, 1, "dose_log10", device, parameter.dtype)
        line = self._batch_feature(cell_line_onehot, batch, self.cell_line_dim,
                                   "cell_line_onehot", device, parameter.dtype)
        drug_token = self.drug_projector(drug) + self.dose_encoder(dose) + self.line_encoder(line)
        # Evaluate padding once per context, never once per layer/ODE call.
        # None enables the torch 2.1 Flash SDPA path for the actual all-valid
        # 2047-gene sequences. Gene ID zero and zero expression stay valid.
        has_padding = not bool(gene_valid.all().item())
        return PreparedAttentionContext(genes, drug_token, gene_valid, has_padding)

    def _project_gene_chunk(self, raw, esm, valid):
        dtype = self.gene_projector.mlp[0].weight.dtype
        # Padding may contain NaNs/large values; remove it before any matmul.
        raw = raw.to(dtype=dtype).masked_fill(~valid[:, None], 0)
        esm = esm.to(dtype=dtype).masked_fill(~valid[:, None], 0)
        value = torch.cat((raw, esm), dim=-1)
        # CUDA autocast reuses one casted projector weight across many gene
        # chunks in a forward. The BF16 gradient accumulation per cast therefore
        # depends on how chunks are partitioned between grouped and
        # condition-wise calls. Disable only this cache for the gene projector
        # so both executions use the same BF16 arithmetic; CPU and non-autocast
        # paths retain the original behavior.
        if value.device.type == "cuda" and torch.is_autocast_enabled():
            with torch.autocast(
                device_type="cuda",
                dtype=torch.get_autocast_gpu_dtype(),
                cache_enabled=False,
            ):
                return self.gene_projector(value)
        return self.gene_projector(value)

    def _validate_context(self, state, context):
        if state.ndim != 2 or state.shape[1] != self.cell_dim or not state.is_floating_point():
            raise ValueError(f"state must be floating-point [B,{self.cell_dim}]")
        if not isinstance(context, PreparedAttentionContext):
            raise TypeError("context must come from prepare_context")
        batch = state.shape[0]
        if (context.projected_genes.ndim != 3 or context.projected_genes.shape[0] != batch
                or context.projected_genes.shape[-1] != self.hidden_dim):
            raise ValueError("projected gene context has an incompatible shape")
        if context.drug_token.shape != (batch, self.hidden_dim):
            raise ValueError("drug context has an incompatible shape")
        if context.gene_valid.shape != context.projected_genes.shape[:2] or context.gene_valid.dtype != torch.bool:
            raise ValueError("context gene_valid must be boolean [B,L]")
        if any(value.device != state.device for value in (
                context.projected_genes, context.drug_token, context.gene_valid)):
            raise ValueError("state and context must share a device")

    def predict_endpoint(self, state, time, context, return_cls=False):
        self._validate_context(state, context)
        batch = state.shape[0]
        if not torch.is_tensor(time):
            time = torch.as_tensor(time, device=state.device, dtype=torch.float32)
        if time.ndim == 0:
            time = time.expand(batch)
        if time.ndim != 1 or time.shape[0] != batch or time.device != state.device:
            raise ValueError("time must be scalar or [B] on the state device")
        cell = self.cell_projector(state.to(dtype=self.cell_projector.net[0].weight.dtype))
        hidden = torch.cat((cell[:, None, :], context.drug_token[:, None, :].to(cell.dtype),
                            context.projected_genes.to(cell.dtype)), dim=1)
        valid = None
        if context.has_padding:
            prefix_valid = torch.ones(batch, 2, device=state.device, dtype=torch.bool)
            valid = torch.cat((prefix_valid, context.gene_valid), dim=1)
        # Deliberately evaluated in EVERY call, including RK4 calls at the
        # same t. No API accepts a precomputed time embedding/modulation.
        time_embedding = self.time_embedder(time)
        for block in self.blocks:
            if self.activation_checkpointing and self.training and torch.is_grad_enabled():
                hidden = checkpoint(block, hidden, time_embedding, valid, use_reentrant=False)
            else:
                hidden = block(hidden, time_embedding, valid)
        cls = self.final_norm(hidden)[:, 0, :]
        endpoint = self.project_out(cell + cls)
        return (endpoint, cls) if return_cls else endpoint

    def forward(self, state, time, source, context, return_cls=False):
        if (source.shape != state.shape or source.device != state.device
                or not source.is_floating_point()):
            raise ValueError("original source must match the state shape/device")
        result = self.predict_endpoint(state, time, context, return_cls=return_cls)
        if return_cls:
            endpoint, cls = result
            return endpoint - source, cls
        return result - source

    def bind_source(self, source, context) -> Callable:
        """Return field(state, time), fixing a copy of ORIGINAL z0 for an ODE.

        This does not detach the prepared parameter graph. For decoder-only
        prediction call under no_grad and pass context.detach(). Never replace
        this bound source with an evolving solver state.
        """
        self._validate_context(source, context)
        fixed_source = source.detach().clone()

        def field(state, time):
            return self(state, time, fixed_source, context)

        return field


# Compatibility alias for callers using the pre-rename class name.
TimeModulatedMAPFlow = AttentionCellFlowVelocity


__all__ = ["PreparedAttentionContext", "AttentionCellFlowVelocity", "TimeModulatedMAPFlow",
           "GaussianFourierTimeEmbedding", "TimeModulatedMAPBlock"]
