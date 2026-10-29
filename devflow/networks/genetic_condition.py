"""Native gene-context attention and symmetric combination conditioning.

Parameter names and forward operations match the evaluated Norman additive
model. Public embeddings/projector checkpoints are supplied by configuration.
"""
from __future__ import annotations
from pathlib import Path
from typing import Optional, Tuple
import torch
from torch import nn

MAP_GENE_INPUT_DIM = 2048 + 5120
MAP_GENE_OUTPUT_DIM = 1024
SE_GENE_DIM = 2048
ESM2_GENE_DIM = 5120

class ResidualProjector(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = (input_dim + output_dim) // 2
        
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim),
        )
        
        self.residual_proj = nn.Linear(input_dim, output_dim)
        
        self.gate = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        transformed = self.mlp(x)
        residual = self.residual_proj(x)
        gate_value = self.gate(x)
        
        return gate_value * transformed + (1 - gate_value) * residual

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

    def __init__(
        self,
        checkpoint_path: str | Path,
        projector_kind: str = "map_predictor",
        *,
        trainable_last_layer: bool = False,
    ):
        super().__init__()
        kind = str(projector_kind)
        if kind not in {"mapkg", "map_predictor"}:
            raise ValueError("projector_kind must be mapkg or map_predictor")
        self.projector_kind = kind
        self.trainable_last_layer = bool(trainable_last_layer)
        self.input_dim = ESM2_GENE_DIM if kind == "mapkg" else MAP_GENE_INPUT_DIM
        self.requires_se_token = kind == "map_predictor"
        hidden_dim = 1536 if kind == "mapkg" else 2048
        self.projector = ResidualProjector(
            input_dim=self.input_dim,
            output_dim=MAP_GENE_OUTPUT_DIM,
            hidden_dim=hidden_dim,
        )
        checkpoint = (checkpoint_path if isinstance(checkpoint_path, dict) else torch.load(
            str(checkpoint_path), map_location="cpu", weights_only=False
        ))
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
                "gene_condition.map_projector.projector.",
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
        # The MAP projector is a transferred feature map. Norman's main
        # recipe optionally adapts only its final 1024-dimensional projection
        # layer; the residual path, gate, hidden layer and normalization stay
        # frozen. Keeping this policy here makes the trainable parameter
        # topology explicit and checkpoint-compatible.
        self.projector.requires_grad_(False)
        if self.trainable_last_layer:
            self.projector.mlp[-1].requires_grad_(True)
        self.projector.eval()

    def train(self, mode: bool = True):
        # Parent model.train() recursively reaches this module.  The
        # projector is frozen and must remain in eval mode so its dropout does
        # not make target/background tokens stochastic.
        super().train(mode)
        self.projector.eval()
        return self

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
        self.scgpt_target_projector = None
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

    def enable_scgpt_target(self, input_dim=512):
        """Optional target-only adapter; call after the complete system is initialized."""
        if self.scgpt_target_projector is not None:
            raise RuntimeError("scGPT target projector already initialized")
        self.scgpt_target_projector = nn.Linear(input_dim, self.dim).to(self.target_role.device)
        self.target_role.requires_grad_(False)

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
        # When the projector output layer is trainable, raw tokens must be
        # used even if a cached projection is present; otherwise the new
        # parameters would receive no gradient. Cached projections remain the
        # fast frozen compatibility path.
        use_raw_projector = (
            self.map_projector is not None
            and getattr(self.map_projector, "trainable_last_layer", False)
            and background_se_tokens is not None
            and background_gene_esm is not None
        )
        if background_projected is not None and not use_raw_projector:
            if background_projected.ndim != 3 or background_projected.size(-1) != self.dim:
                raise ValueError(f"background_projected must be [B,G,{self.dim}]")
            return background_projected
        if self.map_projector is None or background_se_tokens is None or background_gene_esm is None:
            raise ValueError(
                "provide background_projected or map_projector + background_se_tokens + background_gene_esm"
            )
        return self.map_projector.forward_background(background_se_tokens, background_gene_esm)

    def forward(
        self,
        target_gene_esm: Optional[torch.Tensor] = None,
        background_se_tokens: Optional[torch.Tensor] = None,
        background_gene_esm: Optional[torch.Tensor] = None,
        target_projected: Optional[torch.Tensor] = None,
        background_projected: Optional[torch.Tensor] = None,
        background_padding_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
        target_gene_scgpt: Optional[torch.Tensor] = None,
    ):
        if target_gene_scgpt is not None:
            if self.scgpt_target_projector is None or target_gene_scgpt.ndim != 3 or target_gene_scgpt.size(-1) != self.scgpt_target_projector.in_features:
                raise ValueError("scGPT targets require an initialized projector and [B,K,512] features")
            if target_gene_esm is not None or target_projected is not None:
                raise ValueError("Use exactly one target embedding mode")
            query = self.scgpt_target_projector(target_gene_scgpt)
        else:
            if self.scgpt_target_projector is not None:
                raise ValueError("scGPT mode requires target_gene_scgpt")
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
