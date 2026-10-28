"""Control-only LoRA in the *real* final SE transformer feed-forward block.

The cached input is the final layer's post-attention, pre-FFN CLS token. All
earlier tokens/layers are frozen, so this cache is exact for adapters placed in
linear1 and linear2 of that final block. This is not an adapter after the SE
embedding. Treated target embeddings must continue to use the unmodified SE.

The SE model is deliberately not retained by SEControlLoRATail. Its state dict
contains just the final FFN/norm, SE projection decoder, and small LoRA matrices.
"""

from __future__ import annotations

import copy
from collections import OrderedDict
import hashlib
import math
import pickle
import types
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import Tensor, nn
import torch.nn.functional as F


FORMAT_VERSION = 1


def _unwrap(module: nn.Module) -> nn.Module:
    while hasattr(module, "_orig_mod"):
        module = module._orig_mod
    return module


def _last_layer(se: nn.Module, *, allow_stochastic_attention: bool = False) -> nn.Module:
    encoder = _unwrap(se.transformer_encoder)
    layer = _unwrap(encoder.layers[-1])
    if layer.__class__.__name__ != "FlashTransformerEncoderLayer":
        raise TypeError(
            "This exact cached-prefix implementation requires MAP's "
            "FlashTransformerEncoderLayer; standard Transformer fast paths "
            "must be audited separately."
        )
    if not all(hasattr(layer, k) for k in ("linear1", "linear2", "norm2", "dropout_layer")):
        raise TypeError("Unexpected final SE transformer block")
    # MAP flash_transformer passes self.dropout to SDPA even in eval mode.
    # Thus nonzero attention dropout makes a one-time prefix cache stochastic.
    for index, candidate in enumerate(encoder.layers):
        candidate = _unwrap(candidate)
        if float(getattr(candidate, "dropout", 0.0)) != 0.0 and not allow_stochastic_attention:
            raise ValueError(
                f"SE layer {index} has nonzero attention dropout in eval; "
                "pass allow_stochastic_attention=True to cache a fixed native "
                "attention realization and record its RNG seed/state."
            )
    return layer


class LoRALinear(nn.Module):
    """Frozen original projection plus alpha/r * B(A(x)); B starts at zero."""

    def __init__(self, original: nn.Linear, rank: int = 8, alpha: float = 8.0):
        super().__init__()
        if rank <= 0:
            raise ValueError("rank must be positive")
        self.base = copy.deepcopy(original).requires_grad_(False)
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scale = self.alpha / self.rank
        self.lora_A = nn.Parameter(original.weight.new_empty(rank, original.in_features))
        self.lora_B = nn.Parameter(original.weight.new_zeros(original.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.base.eval()

    def forward(self, x: Tensor, enabled: bool = True) -> Tensor:
        output = self.base(x)
        if enabled:
            output = output + self.scale * F.linear(F.linear(x, self.lora_A), self.lora_B)
        return output


class _FrozenSESkipBlock(nn.Module):
    """Portable copy of native SE SkipBlock, without importing its SE model."""

    def __init__(self, original: nn.Module):
        super().__init__()
        self.intermediate_dense = copy.deepcopy(original.intermediate_dense)
        self.dense = copy.deepcopy(original.dense)
        self.activation = copy.deepcopy(getattr(original, "activation", nn.ReLU()))
        self.layer_norm = copy.deepcopy(original.layer_norm)

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        x = self.intermediate_dense(x)
        x = self.activation(x)
        x = self.dense(x)
        return self.layer_norm(x + residual)


def _portable_decoder(module: nn.Module) -> nn.Module:
    module = _unwrap(module)
    if isinstance(module, nn.Sequential):
        return nn.Sequential(OrderedDict(
            (name, _portable_decoder(child)) for name, child in module.named_children()
        ))
    if module.__class__.__name__ == "SkipBlock":
        return _FrozenSESkipBlock(module)
    if isinstance(module, (nn.Linear, nn.LayerNorm, nn.ReLU, nn.Dropout)):
        return copy.deepcopy(module)
    raise TypeError(f"Unexpected native SE projection decoder module: {type(module)}")


class SEControlLoRATail(nn.Module):
    """Exact final SE CLS FFN + projection, with only four trainable matrices.

    `forward(prefix)` returns normalized SE embeddings. `base_forward(prefix)`
    uses the original FFN for retention/auditing. Set the module to train mode
    normally: its override keeps every frozen submodule in evaluation mode.
    No original SE dropout and no LoRA dropout are active.
    """

    def __init__(self, linear1: nn.Linear, linear2: nn.Linear,
                 norm2: nn.Module, decoder: nn.Module,
                 rank: int = 8, alpha: float = 8.0):
        super().__init__()
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.lora_dropout = 0.0
        self.native_attention_dropout = ()
        self.linear1 = LoRALinear(linear1, rank, alpha)
        self.linear2 = LoRALinear(linear2, rank, alpha)
        self.norm2 = copy.deepcopy(_unwrap(norm2)).requires_grad_(False)
        self.decoder = _portable_decoder(decoder).requires_grad_(False)
        self.train(True)

    @classmethod
    def from_se_model(cls, se: nn.Module, rank: int = 8,
                      alpha: float = 8.0, *,
                      allow_stochastic_attention: bool = False) -> "SEControlLoRATail":
        layer = _last_layer(se, allow_stochastic_attention=allow_stochastic_attention)
        se.requires_grad_(False)
        se.eval()
        tail = cls(layer.linear1, layer.linear2, layer.norm2,
                   se.decoder, rank=rank, alpha=alpha)
        tail.native_attention_dropout = tuple(
            float(getattr(_unwrap(candidate), "dropout", 0.0))
            for candidate in _unwrap(se.transformer_encoder).layers
        )
        return tail

    def train(self, mode: bool = True) -> "SEControlLoRATail":
        super().train(mode)
        for module in (self.linear1.base, self.linear2.base, self.norm2, self.decoder):
            module.eval()
        return self

    def _forward(self, prefix: Tensor, enabled: bool) -> Tensor:
        if prefix.ndim != 2 or prefix.shape[-1] != self.linear1.base.in_features:
            raise ValueError("prefix must have shape [cells, SE d_model]")
        # Keep the native prefix dtype inside autocast so the residual add has
        # the same precision/order as SE. Outside autocast disk fp16 caches
        # must be promoted to the frozen linear weights' dtype (normally fp32).
        autocast_enabled = torch.is_autocast_enabled()
        if prefix.device.type == "cpu":
            autocast_enabled = autocast_enabled or torch.is_autocast_cpu_enabled()
        x = prefix if autocast_enabled else prefix.to(dtype=self.linear1.base.weight.dtype)
        ff = self.linear2(F.gelu(self.linear1(x, enabled)), enabled)
        hidden = self.norm2(x + ff)
        return F.normalize(self.decoder(hidden), dim=-1)

    def forward(self, prefix: Tensor) -> Tensor:
        return self._forward(prefix, enabled=True)

    def base_forward(self, prefix: Tensor) -> Tensor:
        return self._forward(prefix, enabled=False)

    def lora_parameters(self):
        return (p for p in self.parameters() if p.requires_grad)

    def adapter_state_dict(self) -> dict[str, Tensor]:
        return {name: tensor.detach().cpu().clone()
                for name, tensor in self.state_dict().items() if ".lora_" in name}

    def frozen_base_sha256(self) -> str:
        digest = hashlib.sha256()
        for name, tensor in sorted(self.state_dict().items()):
            if ".lora_" in name:
                continue
            tensor = tensor.detach().cpu().contiguous()
            digest.update(name.encode())
            digest.update(str(tensor.dtype).encode())
            digest.update(str(tuple(tensor.shape)).encode())
            digest.update(tensor.view(torch.uint8).numpy().tobytes())
        return digest.hexdigest()

    def manifest(self) -> dict[str, Any]:
        return {
            "format_version": FORMAT_VERSION,
            "placement": "last_transformer.linear1+linear2",
            "input": "last_post_attention_pre_ffn_CLS",
            "source_only": True, "target_frozen": True,
            "rank": self.rank, "alpha": self.alpha, "lora_dropout": 0.0,
            "compute_precision": "follow_external_autocast_else_weight_dtype",
            "native_attention_dropout": list(self.native_attention_dropout),
            "prefix_sampling": "fixed_native_attention_realization",
            "input_dim": self.linear1.base.in_features,
            "ffn_dim": self.linear1.base.out_features,
            "trainable_parameters": sum(p.numel() for p in self.parameters() if p.requires_grad),
            "total_tail_parameters": sum(p.numel() for p in self.parameters()),
            "frozen_base_sha256": self.frozen_base_sha256(),
        }

    def save_adapter(self, path: str | Path,
                     metadata: Mapping[str, Any] | None = None) -> None:
        """Only save LoRA parameters; recreate frozen tail from the same SE."""
        torch.save({"manifest": self.manifest(),
                    "adapter_state_dict": self.adapter_state_dict(),
                    "metadata": dict(metadata or {})}, path)

    def load_adapter(self, path: str | Path, *, verify_base: bool = True) -> dict:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        manifest = payload["manifest"]
        for key, expected in (("format_version", FORMAT_VERSION),
                              ("rank", self.rank), ("alpha", self.alpha)):
            if manifest[key] != expected:
                raise ValueError(f"Adapter {key} differs: {manifest[key]} vs {expected}")
        if verify_base and manifest["frozen_base_sha256"] != self.frozen_base_sha256():
            raise ValueError("Frozen SE tail differs from adapter checkpoint")
        expected = set(self.adapter_state_dict())
        if set(payload["adapter_state_dict"]) != expected:
            raise ValueError("Adapter checkpoint has missing or unexpected LoRA tensors")
        self.load_state_dict(payload["adapter_state_dict"], strict=False)
        return payload.get("metadata", {})

    def save_tail(self, path: str | Path,
                  metadata: Mapping[str, Any] | None = None) -> None:
        """Standalone small tail, without the full SE/embedding tables."""
        torch.save({"manifest": self.manifest(),
                    "tail_template": copy.deepcopy(self).cpu(),
                    "metadata": dict(metadata or {})}, path)

    @classmethod
    def load_tail(cls, path: str | Path,
                  map_location: str | torch.device = "cpu") -> "SEControlLoRATail":
        payload = torch.load(path, map_location=map_location, weights_only=False,
                             pickle_module=_tail_pickle_module())
        tail = payload["tail_template"]
        if not isinstance(tail, cls):
            raise TypeError("Not an SEControlLoRATail checkpoint")
        if payload["manifest"]["frozen_base_sha256"] != tail.frozen_base_sha256():
            raise ValueError("Tail checkpoint frozen weights do not match manifest")
        return tail


def from_se_model(se: nn.Module, rank: int = 8,
                  alpha: float = 8.0, *,
                  allow_stochastic_attention: bool = False) -> SEControlLoRATail:
    return SEControlLoRATail.from_se_model(
        se, rank=rank, alpha=alpha,
        allow_stochastic_attention=allow_stochastic_attention)


class _LegacyTailUnpickler(pickle.Unpickler):
    """Resolve historical tail classes without importing its experiment directory."""

    def find_class(self, module, name):
        if module == "se_control_lora" and name in {
            "SEControlLoRATail", "LoRALinear", "_FrozenSESkipBlock"
        }:
            return globals()[name]
        return super().find_class(module, name)


def _tail_pickle_module():
    module = types.ModuleType("devflow_tail_pickle")
    module.Unpickler = _LegacyTailUnpickler
    module.Pickler = pickle.Pickler
    module.load = pickle.load
    module.loads = pickle.loads
    module.dump = pickle.dump
    module.dumps = pickle.dumps
    return module
