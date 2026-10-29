"""Native Norman training, decoder transfer, and prediction recipe runtime."""
from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple, Union
from collections import Counter
from contextlib import nullcontext
import csv
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from devflow.networks.expression_decoder import LatentToGeneDecoder
from devflow.mean.head import TrainableKernelMeanHead
from devflow.mean.kernels import ContinuousPairKernelBank
from devflow.mean.composition import compose_expression
from devflow.networks.genetic_condition import FrozenMAPGeneProjector, GenePerturbationEncoder
from devflow.networks.mlp_velocity import (ControlConditionFusion, MLPCellFlowVelocity,
    pair_population, rk4_rollout, adaptive_rollout, median_sigmas, mmd2_unbiased_multi_sigma)
from devflow.data.norman import NormanH5AD, parse_target_genes, load_norman_metadata, read_expression_rows
from devflow.io import write_prediction_archive, gene_order_hash, write_json, sha256_file, json_sha256

PathLike = Union[str, Path]
GeneSource = Union[PathLike, Sequence[str]]


_KNOWN_STATE_PREFIXES = (
    "module.pert_model.gene_decoder.",
    "pert_model.gene_decoder.",
    "module.gene_decoder.",
    "gene_decoder.",
    "module.decoder.",
)


def _checkpoint_state(payload: Any) -> Mapping[str, torch.Tensor]:
    """Return a state dictionary from common MAP checkpoint containers."""
    if not isinstance(payload, Mapping):
        raise ValueError("checkpoint must be a mapping containing a state dictionary")
    for key in ("decoder_state_dict", "model_state_dict", "state_dict", "model", "decoder"):
        value = payload.get(key)
        if isinstance(value, Mapping):
            return value
    return payload


def load_decoder_checkpoint(
    checkpoint_path: PathLike,
    map_location: Union[str, torch.device] = "cpu",
) -> Dict[str, torch.Tensor]:
    """Load and canonicalize decoder parameters from an Original MAP checkpoint.

    Original MAP stores keys such as ``pert_model.gene_decoder.decoder.8.weight``
    and uses a flat final Linear module.  The current ``LatentToGeneDecoder``
    wraps the final Linear in an activation container, so its corresponding key
    is ``decoder.8.0.weight``.  Both forms are returned under canonical keys.
    """
    payload = torch.load(str(checkpoint_path), map_location=map_location, weights_only=False)
    state = _checkpoint_state(payload)
    result: Dict[str, torch.Tensor] = {}
    for raw_key, value in state.items():
        if not torch.is_tensor(value):
            continue
        key = str(raw_key)
        # A full Norman model can expose decoder.decoder.*; remove only the
        # outer module prefix so decoder.8.* remains a decoder key.
        if key.startswith("module.decoder.decoder."):
            key = key[len("module.decoder."):]
        elif key.startswith("decoder.decoder."):
            key = key[len("decoder."):]
        else:
            for prefix in _KNOWN_STATE_PREFIXES:
                if key.startswith(prefix):
                    key = key[len(prefix):]
                    break
        if key == "decoder.8.weight":
            key = "decoder.8.0.weight"
        elif key == "decoder.8.bias":
            key = "decoder.8.0.bias"
        if not torch.isfinite(value).all():
            raise ValueError(f"non-finite tensor in decoder checkpoint: {raw_key!r}")
        if key in result:
            raise ValueError(f"duplicate canonical decoder parameter {key!r} in checkpoint")
        result[key] = value.detach().cpu()
    if not result:
        raise ValueError(f"no tensor decoder parameters found in {checkpoint_path}")
    return result


def _read_gene_source(source: GeneSource) -> list[str]:
    """Read gene symbols from a list, JSON list/object, CSV, or text file."""
    if isinstance(source, (str, Path)):
        path = Path(source).expanduser()
        suffix = path.suffix.lower()
        if suffix == ".json":
            value = json.loads(path.read_text())
            if isinstance(value, Mapping):
                for key in ("gene_symbols", "genes", "gene_names", "symbols"):
                    if key in value:
                        value = value[key]
                        break
            if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
                raise ValueError(f"{path} must contain a JSON gene list or gene_symbols field")
            values = list(value)
        elif suffix == ".csv":
            with path.open(newline="") as handle:
                rows = list(csv.reader(handle))
            if not rows:
                values = []
            else:
                header = [str(v).strip().lower() for v in rows[0]]
                column = next((header.index(k) for k in ("gene_symbol", "gene_symbols", "gene", "gene_name", "symbol") if k in header), None)
                if column is None:
                    column = 0
                start = 1 if header[column] in {"gene", "gene_name", "gene_symbol", "gene_symbols", "symbol"} else 0
                values = [row[column] for row in rows[start:] if len(row) > column]
        else:
            values = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    else:
        values = list(source)
    genes = [str(value).strip() for value in values]
    if any(not gene for gene in genes):
        raise ValueError("gene lists may not contain empty symbols")
    return genes


def read_gene_symbols(source: GeneSource) -> list[str]:
    """Public wrapper for loading an ordered gene-symbol list."""
    genes = _read_gene_source(source)
    if not genes:
        raise ValueError("gene list is empty")
    return genes


def _decoder_from_object(decoder_or_model: nn.Module) -> nn.Module:
    """Accept either LatentToGeneDecoder or a Norman system containing .decoder."""
    if isinstance(decoder_or_model, LatentToGeneDecoder):
        return decoder_or_model
    decoder = getattr(decoder_or_model, "decoder", None)
    if isinstance(decoder, nn.Module):
        return decoder
    raise TypeError("expected LatentToGeneDecoder or a model exposing .decoder")


def _output_parameter_keys(
    target_state: Mapping[str, torch.Tensor],
    target_gene_count: int,
) -> Tuple[str, str]:
    """Find final output Linear weight/bias keys by output width."""
    weights = [
        key for key, value in target_state.items()
        if key.endswith(".weight") and value.ndim == 2 and int(value.shape[0]) == target_gene_count
    ]
    biases = [
        key for key, value in target_state.items()
        if key.endswith(".bias") and value.ndim == 1 and int(value.shape[0]) == target_gene_count
    ]
    # The shape criterion is unambiguous for the MAP decoder.  The final-key
    # fallback keeps this helper usable for an unusual decoder whose hidden
    # width happens to equal the output width.
    if len(weights) != 1:
        ordered = [key for key in target_state if key.endswith(".weight")]
        if not ordered:
            raise ValueError("decoder state has no Linear weight parameter")
        weights = [ordered[-1]]
    if len(biases) != 1:
        ordered = [key for key in target_state if key.endswith(".bias")]
        if not ordered:
            raise ValueError("decoder state has no Linear bias parameter")
        biases = [ordered[-1]]
    return weights[0], biases[0]


def _seeded_linear_parameter(
    shape: Sequence[int], dtype: torch.dtype, seed: int, *, fan_in: Optional[int] = None
) -> torch.Tensor:
    """Match nn.Linear's default uniform range with a local deterministic RNG."""
    if len(shape) == 0:
        raise ValueError("parameter shape may not be empty")
    fan_in = int(fan_in) if fan_in is not None else (int(shape[1]) if len(shape) >= 2 else int(shape[0]))
    if fan_in <= 0:
        raise ValueError(f"invalid fan-in for shape {tuple(shape)}")
    bound = 1.0 / math.sqrt(float(fan_in))
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    values = torch.rand(tuple(int(x) for x in shape), generator=generator, dtype=torch.float32)
    return ((2.0 * values - 1.0) * bound).to(dtype=dtype)


def initialize_decoder_from_map_checkpoint(
    decoder_or_model: nn.Module,
    checkpoint_path: PathLike,
    source_genes: GeneSource,
    target_genes: GeneSource,
    *,
    seed: int = 42,
    strict_trunk: bool = True,
) -> Dict[str, Any]:
    """Transfer an Original MAP decoder using unique gene-symbol alignment.

    All hidden/trunk parameters must match exactly when ``strict_trunk`` is
    true.  The final output rows are initialized with a local seeded Linear
    initialization, then rows whose target symbol has exactly one occurrence in
    ``source_genes`` receive the corresponding checkpoint weight and bias.
    Rows with no source match or an ambiguous duplicate source symbol remain
    freshly initialized.  The returned audit contains one mapping record per
    target row and can be serialized directly as JSON.
    """
    decoder = _decoder_from_object(decoder_or_model)
    source = read_gene_symbols(source_genes)
    target = read_gene_symbols(target_genes)
    if len(set(target)) != len(target):
        raise ValueError("target gene list must contain unique symbols")
    checkpoint_state = load_decoder_checkpoint(checkpoint_path)
    target_state = decoder.state_dict()
    weight_key, bias_key = _output_parameter_keys(target_state, len(target))
    if weight_key not in checkpoint_state or bias_key not in checkpoint_state:
        raise ValueError(
            f"checkpoint lacks final output parameters {weight_key!r}/{bias_key!r}; "
            "check decoder architecture and key prefixes"
        )
    source_weight = checkpoint_state[weight_key]
    source_bias = checkpoint_state[bias_key]
    target_weight = target_state[weight_key]
    target_bias = target_state[bias_key]
    if source_weight.ndim != 2 or source_bias.ndim != 1:
        raise ValueError("checkpoint final decoder parameters have invalid ranks")
    if tuple(source_weight.shape) != (len(source), int(target_weight.shape[1])):
        raise ValueError(
            f"source final weight shape {tuple(source_weight.shape)} does not match "
            f"({len(source)}, {int(target_weight.shape[1])})"
        )
    if tuple(source_bias.shape) != (len(source),):
        raise ValueError(f"source final bias shape {tuple(source_bias.shape)} does not match ({len(source)},)")
    if tuple(target_bias.shape) != (len(target),):
        raise ValueError(f"target decoder output width {tuple(target_bias.shape)} does not match target genes")

    trunk_keys = [key for key in target_state if key not in {weight_key, bias_key}]
    missing = [key for key in trunk_keys if key not in checkpoint_state]
    shape_mismatch = [
        key for key in trunk_keys
        if key in checkpoint_state and tuple(checkpoint_state[key].shape) != tuple(target_state[key].shape)
    ]
    if strict_trunk and (missing or shape_mismatch):
        raise ValueError(
            "MAP decoder trunk mismatch: "
            f"missing={missing[:8]}, shape_mismatch={shape_mismatch[:8]}"
        )

    # Keep the target decoder's device and dtype while replacing the full state.
    merged: Dict[str, torch.Tensor] = {}
    for key, value in target_state.items():
        if key == weight_key or key == bias_key:
            continue
        if key in checkpoint_state and tuple(checkpoint_state[key].shape) == tuple(value.shape):
            merged[key] = checkpoint_state[key].to(device=value.device, dtype=value.dtype)
        elif strict_trunk:
            raise ValueError(f"cannot transfer decoder trunk parameter {key}")
        else:
            merged[key] = value.detach().clone()

    fresh_weight = _seeded_linear_parameter(target_weight.shape, target_weight.dtype, int(seed))
    fresh_bias = _seeded_linear_parameter(
        target_bias.shape, target_bias.dtype, int(seed) + 1, fan_in=int(target_weight.shape[1])
    )
    merged[weight_key] = fresh_weight.to(device=target_weight.device)
    merged[bias_key] = fresh_bias.to(device=target_bias.device)

    counts = Counter(source)
    lookup = {gene: index for index, gene in enumerate(source) if counts[gene] == 1}
    mappings = []
    copied = 0
    ambiguous = 0
    unmatched = 0
    with torch.no_grad():
        for target_index, gene in enumerate(target):
            source_index = lookup.get(gene)
            if source_index is None:
                if counts.get(gene, 0) > 1:
                    status = "ambiguous_source_duplicate"
                    ambiguous += 1
                else:
                    status = "unmatched_fresh_initialization"
                    unmatched += 1
            else:
                status = "copied_unique_symbol"
                copied += 1
                merged[weight_key][target_index].copy_(source_weight[source_index].to(
                    device=target_weight.device, dtype=target_weight.dtype
                ))
                merged[bias_key][target_index].copy_(source_bias[source_index].to(
                    device=target_bias.device, dtype=target_bias.dtype
                ))
            mappings.append({
                "target_index": int(target_index),
                "target_gene": gene,
                "source_index": None if source_index is None else int(source_index),
                "status": status,
            })
        decoder.load_state_dict(merged, strict=True)
        decoder.requires_grad_(True)

    return {
        "checkpoint": str(Path(checkpoint_path).expanduser().resolve()),
        "source_gene_count": len(source),
        "target_gene_count": len(target),
        "copied_output_rows": copied,
        "unmatched_output_rows": unmatched,
        "ambiguous_output_rows": ambiguous,
        "trunk_reused": True,
        "trunk_parameter_count": len(trunk_keys),
        "source_output_weight_key": weight_key,
        "target_output_weight_key": weight_key,
        "source_output_bias_key": bias_key,
        "target_output_bias_key": bias_key,
        "mapping_policy": "unique_gene_symbol_only; unmatched and duplicate source symbols use seeded fresh Linear initialization",
        "seed": int(seed),
        "mapping": mappings,
    }



def save_finetuned_decoder_checkpoint(
    path: PathLike,
    decoder_or_model: nn.Module,
    target_genes: GeneSource,
    *,
    mapping_audit: Optional[Mapping[str, Any]] = None,
    step: Optional[int] = None,
    extra_metadata: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Save a decoder-only checkpoint with an explicit target gene order."""
    decoder = _decoder_from_object(decoder_or_model)
    genes = read_gene_symbols(target_genes)
    if len(set(genes)) != len(genes):
        raise ValueError("target gene list must contain unique symbols")
    state = {key: value.detach().cpu() for key, value in decoder.state_dict().items()}
    payload: Dict[str, Any] = {
        "format_version": 1,
        "model_family": "norman_gene_aligned_decoder_v1",
        "gene_symbols": genes,
        "decoder_state_dict": state,
    }
    if mapping_audit is not None:
        payload["mapping_audit"] = dict(mapping_audit)
    if step is not None:
        payload["step"] = int(step)
    if extra_metadata is not None:
        payload["metadata"] = dict(extra_metadata)
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, str(destination))
    return {
        "path": str(destination),
        "format_version": 1,
        "gene_count": len(genes),
        "gene_order_sha256": _gene_order_hash(genes),
        "step": None if step is None else int(step),
    }


def _gene_order_hash(genes: Sequence[str]) -> str:
    import hashlib
    payload = "\n".join(str(gene) for gene in genes).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_finetuned_decoder_checkpoint(
    decoder_or_model: nn.Module,
    checkpoint_path: PathLike,
    target_genes: GeneSource,
    *,
    map_location: Union[str, torch.device] = "cpu",
    strict_gene_order: bool = True,
) -> Dict[str, Any]:
    """Load a Norman decoder checkpoint and verify its exact gene order."""
    decoder = _decoder_from_object(decoder_or_model)
    expected = read_gene_symbols(target_genes)
    payload = torch.load(str(checkpoint_path), map_location=map_location, weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError("fine-tuned decoder checkpoint must be a mapping")
    observed = payload.get("gene_symbols", payload.get("target_genes"))
    if observed is None:
        if strict_gene_order:
            raise ValueError("fine-tuned decoder checkpoint does not record gene_symbols")
    else:
        observed = [str(value).strip() for value in observed]
        if strict_gene_order and observed != expected:
            first_difference = next(
                (index for index, (left, right) in enumerate(zip(observed, expected)) if left != right),
                min(len(observed), len(expected)),
            )
            raise ValueError(
                "fine-tuned decoder gene order mismatch at index "
                f"{first_difference}: checkpoint={observed[first_difference:first_difference+1]}, "
                f"expected={expected[first_difference:first_difference+1]}"
            )
    raw_state = payload.get("decoder_state_dict", payload.get("state_dict", payload.get("decoder")))
    if not isinstance(raw_state, Mapping):
        raise ValueError("fine-tuned decoder checkpoint lacks decoder_state_dict")
    target_state = decoder.state_dict()
    normalized: Dict[str, torch.Tensor] = {}
    for raw_key, value in raw_state.items():
        if not torch.is_tensor(value):
            continue
        key = str(raw_key)
        if key.startswith("module.decoder."):
            key = key[len("module."):]
        elif key.startswith("decoder.decoder."):
            key = key[len("decoder."):]
        elif key == "decoder.8.weight":
            key = "decoder.8.0.weight"
        elif key == "decoder.8.bias":
            key = "decoder.8.0.bias"
        if key in target_state:
            normalized[key] = value
    missing = sorted(set(target_state) - set(normalized))
    unexpected = sorted(set(normalized) - set(target_state))
    if missing or unexpected:
        raise ValueError(f"fine-tuned decoder state mismatch: missing={missing[:8]}, unexpected={unexpected[:8]}")
    decoder.load_state_dict(normalized, strict=True)
    return {
        "checkpoint": str(Path(checkpoint_path).expanduser().resolve()),
        "gene_count": len(expected),
        "gene_order_verified": observed is not None and observed == expected,
        "step": payload.get("step"),
        "mapping_audit": payload.get("mapping_audit"),
    }

def decoder_finetune_loss(
    decoder: nn.Module,
    latent: torch.Tensor,
    target_expression: torch.Tensor,
    *,
    mean_loss_weight: float = 1.0,
    mmd_weight: float = 0.1,
    mmd_cells: Optional[int] = 128,
    mmd_indices: Optional[torch.Tensor] = None,
    sigmas: Optional[Iterable[float]] = None,
    return_components: bool = False,
):
    """Compute centered residual MSE + weighted mean MSE + optional MMD.

    ``latent`` and ``target_expression`` must describe cells from one Norman
    perturbation condition.  Residuals are centered over the cell dimension,
    and the mean term compares per-gene batch means.  These two terms sum to
    paired-cell MSE when ``mean_loss_weight=1`` (the backward-compatible
    default).  MMD compares the
    predicted and observed cell distributions from the same condition; when
    ``mmd_cells`` is set, the first rows (or explicit ``mmd_indices``) are used
    to bound the quadratic kernel cost.  Set ``mmd_weight`` to zero to disable
    MMD, or pass a strictly positive value for the requested MSE+MMD objective.
    """
    coefficient = float(mmd_weight)
    mean_coefficient = float(mean_loss_weight)
    if not math.isfinite(coefficient) or coefficient < 0.0:
        raise ValueError("mmd_weight must be finite and nonnegative")
    if not math.isfinite(mean_coefficient) or mean_coefficient < 0.0:
        raise ValueError("mean_loss_weight must be finite and nonnegative")
    if latent.ndim != 2:
        raise ValueError(f"latent must be [N,D], got {tuple(latent.shape)}")
    if target_expression.ndim != 2 or target_expression.size(0) != latent.size(0):
        raise ValueError(
            "target_expression must be [N,G] with the same N as latent; "
            f"got latent={tuple(latent.shape)}, target={tuple(target_expression.shape)}"
        )
    if latent.size(0) == 0 or target_expression.size(1) == 0:
        raise ValueError("decoder loss requires nonempty cell and gene dimensions")
    prediction = decoder(latent).float()
    target = target_expression.float()
    if prediction.shape != target.shape:
        raise ValueError(f"decoder output {tuple(prediction.shape)} != target {tuple(target.shape)}")
    cell_mse = F.mse_loss(prediction, target)
    prediction_mean = prediction.mean(dim=0, keepdim=True)
    target_mean = target.mean(dim=0, keepdim=True)
    residual_mse = F.mse_loss(prediction - prediction_mean, target - target_mean)
    mean_mse = F.mse_loss(prediction_mean, target_mean)
    mmd = prediction.new_zeros((), dtype=torch.float32)
    n_mmd = 0
    if coefficient > 0.0:
        if mmd_indices is not None:
            indices = torch.as_tensor(mmd_indices, device=prediction.device, dtype=torch.long)
            if indices.ndim != 1 or indices.numel() < 2:
                raise ValueError("mmd_indices must contain at least two cell indices")
            if int(indices.min()) < 0 or int(indices.max()) >= prediction.size(0):
                raise ValueError("mmd_indices contains an invalid cell index")
        else:
            requested = prediction.size(0) if mmd_cells is None else int(mmd_cells)
            n_mmd = min(requested, prediction.size(0))
            if n_mmd < 2:
                raise ValueError("MMD requires at least two cells in each condition")
            indices = torch.arange(n_mmd, device=prediction.device, dtype=torch.long)
        pred_subset = prediction.index_select(0, indices)
        target_subset = target.index_select(0, indices)
        kernel_sigmas = list(sigmas) if sigmas is not None else median_sigmas(
            torch.cat((pred_subset.detach(), target_subset.detach()), dim=0)
        )
        mmd = mmd2_unbiased_multi_sigma(pred_subset, target_subset, kernel_sigmas)
        n_mmd = int(indices.numel())
    # Preserve the exact historical arithmetic for callers using the default.
    reconstruction = cell_mse if mean_coefficient == 1.0 else residual_mse + mean_coefficient * mean_mse
    total = reconstruction + coefficient * mmd
    if not return_components:
        return total
    return total, {
        "decoder_loss": total,
        "decoder_cell_mse": cell_mse,
        "decoder_residual_loss": residual_mse,
        "weighted_decoder_residual_loss": residual_mse,
        "decoder_mean_loss": mean_mse,
        "weighted_decoder_mean_loss": mean_coefficient * mean_mse,
        "decoder_mmd_loss": mmd,
        "weighted_decoder_mmd_loss": coefficient * mmd,
        "mmd_cells": int(n_mmd),
        "mmd_weight": coefficient,
        "mean_loss_weight": mean_coefficient,
        "endpoint_expression": prediction,
    }




class NormanGeneFlowMatchingSystem(nn.Module):
    """Norman gene branch that reuses the unchanged MAP velocity implementation."""

    model_family = "map_norman_gene_cross_attention_otfm_v1"

    def __init__(
        self,
        state_dim: int = 2048,
        expression_dim: int = 2000,
        map_projector: Optional[FrozenMAPGeneProjector] = None,
        projector_trainable_last_layer: bool = False,
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
        use_mean_head: bool = False,
        mean_head_kind: str = "kernel",
        mean_assets=None,
        mean_features=None,
    ):
        super().__init__()
        if int(gene_condition_dim) != 1024:
            raise ValueError("gene_condition_dim must be 1024 for the MAP-compatible branch")
        self.state_dim = int(state_dim)
        self.expression_dim = int(expression_dim)
        self.path_sigma = float(path_sigma)
        self.projector_trainable_last_layer = bool(projector_trainable_last_layer)
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
        self.velocity = MLPCellFlowVelocity(
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
        self.use_mean_head = bool(use_mean_head)
        if mean_head_kind != "kernel":
            raise ValueError("Norman uses the shared kernel mean head")
        self.mean_head = None
        self.mean_kernelbank = None
        self.mean_features = None
        if self.use_mean_head:
            if mean_assets is None or mean_features is None:
                raise ValueError("Kernel mean head requires fixed training assets and gene features")
            self.mean_head = TrainableKernelMeanHead(**{
                key: mean_assets[key] for key in
                ("kernel", "targets", "bias", "control_mean", "alpha", "metadata")
            })
            self.mean_kernelbank = ContinuousPairKernelBank.from_assets(mean_assets["kernelbank_assets"])
            self.mean_features = mean_features
            if self.mean_head.A.shape[-1] != self.expression_dim:
                raise ValueError("Kernel mean head and decoder gene dimensions differ")

    def predict_condition_mean(self, condition_name):
        """A fixed gene prior supplies kernel queries independently of Flow."""
        if self.mean_head is None:
            raise ValueError("This checkpoint has no kernel mean head")
        from devflow.data.norman import paired_gene_features
        primary, secondary, presence = paired_gene_features([condition_name], self.mean_features)
        query = self.mean_kernelbank.query(primary, secondary, presence, 0)
        return self.mean_head.predict_mean(query, 0)[0]

    def export_mean_assets(self):
        if self.mean_head is None:
            return None
        return {**self.mean_head.export_assets(),
                "kernelbank_assets": self.mean_kernelbank.to_assets()}

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
                gene_condition_kwargs.get("target_gene_scgpt"),
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
            "target_gene_esm", "target_gene_scgpt", "background_se_tokens", "background_gene_esm",
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
            gene_condition_kwargs.get('target_gene_scgpt'),
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
            'target_gene_esm', 'target_gene_scgpt', 'background_se_tokens', 'background_gene_esm',
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

def centered_expression_wasserstein(prediction, truth):
    """Mean gene-wise empirical W2² of centered, unpaired cell populations.

    Each gene is sorted independently: this matches residual distributions,
    without claiming a biological correspondence between sampled cells.
    Existing expression MMD remains responsible for joint gene structure.
    """
    if prediction.ndim != 2 or prediction.shape != truth.shape:
        raise ValueError('Expression residual requires equal [cells, genes] shapes')
    if prediction.size(0) < 2:
        raise ValueError('Expression residual needs at least two cells')
    predicted_residual = prediction - prediction.mean(0, keepdim=True)
    true_residual = truth - truth.mean(0, keepdim=True)
    return (predicted_residual.sort(dim=0).values
            - true_residual.sort(dim=0).values).square().mean()

class FlowTrainingForward(nn.Module):
    """Route every differentiable training call through DDP.forward()."""
    def __init__(self, system, expression_mmd_weight=0., expression_mmd_time_points=5,
                 train_decoder=False, expression_mean_weight=0.,
                 expression_mean_decoder_only=False, expression_residual_weight=0.,
                 external_expression_loss=False, mean_head_weight=1.0,
                 decoder_mmd_weight=0.0):
        super().__init__()
        self.system = system
        self.expression_mmd_weight = float(expression_mmd_weight)
        self.expression_mmd_time_points = int(expression_mmd_time_points)
        self.train_decoder = bool(train_decoder)
        self.expression_mean_weight = float(expression_mean_weight)
        self.expression_mean_decoder_only = bool(expression_mean_decoder_only)
        self.expression_residual_weight = float(expression_residual_weight)
        self.external_expression_loss = bool(external_expression_loss)
        self.mean_head_weight = float(mean_head_weight)
        self.decoder_mmd_weight = float(decoder_mmd_weight)
        weights = (self.expression_mmd_weight, self.expression_mean_weight,
                   self.expression_residual_weight, self.mean_head_weight if self.external_expression_loss else 0.0,
                   self.decoder_mmd_weight if self.external_expression_loss else 0.0)
        if any(not np.isfinite(weight) or weight < 0 for weight in weights):
            raise ValueError('Expression loss weights must be finite and nonnegative')
        self.expression_enabled = any(weight > 0 for weight in weights)
        if self.expression_mmd_time_points < 2:
            raise ValueError('Differentiable expression RK4 needs >=2 time points')
        if self.train_decoder and not self.expression_enabled:
            raise ValueError('Trainable decoder requires a positive expression loss weight')
        if any(parameter.requires_grad != self.train_decoder
               for parameter in self.system.decoder.parameters()):
            raise ValueError('Decoder requires_grad must match train_decoder')

    def forward(self, source, target, gene_kwargs, mean_shift, generator,
                epsilon, iterations, tolerance, mean_weight, source_h=None,
                target_expression=None, apply_expression_mmd=True,
                mean_condition=None):
        loss, parts = self.system.flow_matching_loss(
            source, source if source_h is None else source_h, target, gene_kwargs,
            epsilon=epsilon, max_iterations=iterations, tolerance=tolerance,
            mean_shift_target=mean_shift, mean_shift_weight=mean_weight,
            generator=generator, return_components=True,
        )
        # The shared kernel head has fixed gene priors and a full-anchor loss.
        # MMD sees a detached Flow endpoint and detached predicted mean, so
        # only the decoder is updated by the expression distribution loss.
        if self.external_expression_loss:
            if self.system.mean_head is None:
                raise ValueError('external expression loss requires a mean head')
            if target_expression is None or target_expression.ndim != 2:
                raise ValueError('Expression losses require original target expression rows')
            if target_expression.size(0) != target.size(0):
                raise ValueError('Target expression rows must match the original treated batch')
            if mean_condition is None:
                raise ValueError('The kernel mean head needs the current condition name')
            head_loss, head_parts = self.system.mean_head.loss(return_parts=True)
            if not apply_expression_mmd:
                parts = dict(parts)
                parts.update(head_parts)
                parts.update(head_loss=head_loss,
                             weighted_head_loss=self.mean_head_weight * head_loss)
                total = loss + self.mean_head_weight * head_loss
                parts["total_loss"] = total
                return total, parts
            final_mean = self.system.predict_condition_mean(mean_condition).detach()
            self.system.decoder.eval()
            with torch.no_grad():
                endpoint = self.system.rollout(
                    source, source if source_h is None else source_h, gene_kwargs,
                    time_points=self.expression_mmd_time_points,
                )
            raw_prediction = self.system.decoder(endpoint.detach()).float()
            composed = compose_expression(raw_prediction, final_mean, detach_mean=True,
                                          output_dtype=torch.float32)
            truth = target_expression.float()
            with torch.autocast(device_type=truth.device.type, enabled=False):
                mmd = mmd2_unbiased_multi_sigma(
                    composed.float(), truth, sigmas=median_sigmas(truth.detach())
                ) if self.decoder_mmd_weight > 0 else truth.new_zeros((), dtype=torch.float32)
            parts = dict(parts)
            parts.update(head_parts)
            parts['head_loss'] = head_loss
            parts['weighted_head_loss'] = self.mean_head_weight * head_loss
            parts['decoder_mmd_loss'] = mmd
            parts['weighted_decoder_mmd_loss'] = self.decoder_mmd_weight * mmd
            parts['expression_mean_loss'] = head_loss
            parts['weighted_expression_mean_loss'] = self.mean_head_weight * head_loss
            parts['expression_mmd_loss'] = mmd
            parts['weighted_expression_mmd_loss'] = self.decoder_mmd_weight * mmd
            parts['endpoint_expression'] = composed
            parts['endpoint_expression_raw'] = raw_prediction
            total = loss + self.mean_head_weight * head_loss + self.decoder_mmd_weight * mmd
            parts['total_loss'] = total
            return total, parts
        # Preserve the exact old objective and random stream when disabled.
        if not self.expression_enabled or not apply_expression_mmd:
            return loss, parts
        if target_expression is None or target_expression.ndim != 2:
            raise ValueError('Expression losses require original target expression rows')
        if target_expression.size(0) != target.size(0):
            raise ValueError('Target expression rows must match the original treated batch')
        # Keep the baseline decoder forward behavior (dropout disabled) in both
        # modes. eval() does not disable gradients to its parameters or input.
        self.system.decoder.eval()
        # Use the ORIGINAL controls and treated population, not OT-resampled
        # pairs. rollout() is differentiable; sample() would cut the gradient.
        endpoint = self.system.rollout(
            source, source if source_h is None else source_h, gene_kwargs,
            time_points=self.expression_mmd_time_points,
        )
        prediction = self.system.decoder(endpoint)
        # Only the mean branch may detach the endpoint; MMD and residual keep
        # their original graph through decoder, RK4, velocity and conditions.
        mean_prediction = (self.system.decoder(endpoint.detach())
                           if self.expression_mean_decoder_only and self.expression_mean_weight > 0
                           else prediction)
        # AMP is useful for the network, but kernel distances and bandwidths
        # must be computed in FP32. Bandwidth sees only the real train batch.
        with torch.autocast(device_type=prediction.device.type, enabled=False):
            prediction = prediction.float()
            truth = target_expression.float()
            zero = prediction.new_zeros(())
            expression_mmd = zero
            expression_mean = zero
            expression_residual = zero
            if self.expression_mmd_weight > 0:
                sigmas = median_sigmas(truth.detach())
                expression_mmd = mmd2_unbiased_multi_sigma(prediction, truth, sigmas=sigmas)
            if self.expression_mean_weight > 0:
                expression_mean = (mean_prediction.float().mean(0) - truth.mean(0)).square().mean()
            if self.expression_residual_weight > 0:
                expression_residual = centered_expression_wasserstein(prediction, truth)
        parts = dict(parts)
        parts['expression_mmd_loss'] = expression_mmd
        parts['weighted_expression_mmd_loss'] = self.expression_mmd_weight * expression_mmd
        parts['expression_mean_loss'] = expression_mean
        parts['weighted_expression_mean_loss'] = self.expression_mean_weight * expression_mean
        parts['expression_residual_loss'] = expression_residual
        parts['weighted_expression_residual_loss'] = self.expression_residual_weight * expression_residual
        with torch.no_grad():
            parts['endpoint_latent_variance_ratio'] = (
                endpoint.float().var(0, unbiased=False).mean()
                / target.float().var(0, unbiased=False).mean().clamp_min(1e-12))
            parts['endpoint_expression_variance_ratio'] = (
                prediction.var(0, unbiased=False).mean()
                / truth.var(0, unbiased=False).mean().clamp_min(1e-12))
        # Keep the original MMD-only addition and RNG stream exactly when the
        # new weights are zero. Disabled branches perform no extra forwards.
        total = loss
        if self.expression_mmd_weight > 0:
            total = total + parts['weighted_expression_mmd_loss']
        if self.expression_mean_weight > 0:
            total = total + parts['weighted_expression_mean_loss']
        if self.expression_residual_weight > 0:
            total = total + parts['weighted_expression_residual_loss']
        parts['total_loss'] = total
        return total, parts


def load_inputs(config, device="cpu"):
    """Load declared caches and enforce split, cell-order and train-only assets."""
    paths = config["data"]
    required = ("h5ad", "split_manifest", "control_split", "gene_indices", "genes",
                "state_embeddings", "state_metadata", "gene_condition_assets")
    for name in required:
        if not paths.get(name) or not Path(paths[name]).is_file():
            raise FileNotFoundError(f"data.{name} must name an existing artifact")
    conditions, control, cell_ids, all_genes, shape = load_norman_metadata(Path(paths["h5ad"]))
    split = json.loads(Path(paths["split_manifest"]).read_text())
    train, test = list(split["train_conditions"]), list(split["test_conditions"])
    observed = set(str(x) for x in conditions[~control])
    if len(set(train)) != len(train) or len(set(test)) != len(test) or set(train) & set(test):
        raise ValueError("Conditions must be unique and disjoint")
    if set(train) | set(test) != observed:
        raise ValueError("Every observed perturbation condition must be assigned exactly once")
    setting = config.get("setting", "additive")
    if setting == "additive":
        singles = {parse_target_genes(x)[0] for x in train if len(parse_target_genes(x)) == 1}
        if any(len(parse_target_genes(x)) != 2 or not set(parse_target_genes(x)) <= singles for x in test):
            raise ValueError("Additive test pairs require both singles in training")
    elif setting == "gene_holdout":
        held = set(config.get("heldout_genes", split.get("heldout_genes", [])))
        if not held:
            raise ValueError("gene_holdout requires an explicit heldout_genes list")
        if any(held.intersection(parse_target_genes(x)) for x in train):
            raise ValueError("Held-out genes occur in fitted single/pair conditions")
        if any(not held.intersection(parse_target_genes(x)) for x in test):
            raise ValueError("Test set must be exactly the held-out-gene closure")
    else:
        raise ValueError("setting must be additive or gene_holdout")
    expected = config.get("expected_counts", {})
    for key, actual in (("train_conditions", len(train)), ("test_conditions", len(test))):
        if key in expected and expected[key] != actual:
            raise ValueError(f"Unexpected {key}: {actual}")
    controls = json.loads(Path(paths["control_split"]).read_text())
    train_controls = np.asarray(controls["train_control_rows"], dtype=np.int64)
    test_controls = np.asarray(controls["test_control_rows"], dtype=np.int64)
    if len(np.unique(train_controls)) != len(train_controls) or len(np.unique(test_controls)) != len(test_controls):
        raise ValueError("Control split contains duplicates")
    if np.intersect1d(train_controls, test_controls).size or set(train_controls) | set(test_controls) != set(np.flatnonzero(control)):
        raise ValueError("Control partitions must be disjoint and cover precisely all control cells")
    indices = np.load(paths["gene_indices"], allow_pickle=False)
    genes = read_gene_symbols(paths["genes"])
    if indices.ndim != 1 or not len(indices) or np.any(indices[1:] <= indices[:-1]):
        raise ValueError("Expression columns must be nonempty, sorted and unique")
    if genes != [all_genes[int(i)] for i in indices]:
        raise ValueError("Expression columns disagree with the declared gene order")
    if "genes" in expected and len(genes) != expected["genes"]:
        raise ValueError("Unexpected expression gene count")
    if paths.get("hvg_provenance"):
        provenance = json.loads(Path(paths["hvg_provenance"]).read_text())
        if provenance.get("complete") is not True:
            raise ValueError("Incomplete train-only HVG provenance")
        train_treated = np.flatnonzero(np.isin(conditions, train) & ~control)
        fit_rows = np.concatenate((train_controls, train_treated))
        if provenance.get("fit_rows") != len(fit_rows):
            raise ValueError("HVG fit row count differs from train controls and train treated")
        # Historical manifests used either controls-first or globally sorted rows.
        # Verify the full ordered hash under its actual convention; never accept
        # a count alone or include any held-out expression rows.
        candidates = [fit_rows, np.sort(fit_rows)]
        def agrees(rows):
            text = "\n".join(str(int(x)) for x in rows)
            text_ok = provenance.get("fit_rows_sha256") in (None, hashlib.sha256(text.encode()).hexdigest(),
                hashlib.sha256((text + "\n").encode()).hexdigest())
            binary_ok = provenance.get("fit_rows_sha256_le_i8") in (None,
                hashlib.sha256(np.asarray(rows, dtype="<i8").tobytes()).hexdigest())
            return text_ok and binary_ok
        if not any(provenance.get(k) for k in ("fit_rows_sha256", "fit_rows_sha256_le_i8")) or not any(agrees(r) for r in candidates):
            raise ValueError("HVG provenance fit row identity/order differs from the training split")
        for key, path_key in (("condition_manifest_sha256", "split_manifest"),
                              ("control_split_sha256", "control_split"),
                              ("gene_indices_sha256", "gene_indices")):
            if provenance.get(key) not in (None, sha256_file(paths[path_key])):
                raise ValueError("HVG provenance artifact identity mismatch: " + key)
        if provenance.get("gene_order_sha256") not in (None, gene_order_hash(genes)):
            raise ValueError("HVG provenance gene order mismatch")
    elif setting == "gene_holdout":
        raise ValueError("gene_holdout requires train-only hvg_provenance")
    state_meta = json.loads(Path(paths["state_metadata"]).read_text())
    if state_meta.get("finite") is not True or state_meta.get("complete") is False:
        raise ValueError("Incomplete/nonfinite SE state cache")
    if state_meta.get("cell_order_sha256") != gene_order_hash(cell_ids):
        raise ValueError("SE cache cell row order differs from the h5ad")
    states = np.load(paths["state_embeddings"], mmap_mode="r", allow_pickle=False)
    if states.shape != (shape[0], 2048):
        raise ValueError("Expected a frozen 2048-dimensional state for every cell")
    bank = torch.load(paths["gene_condition_assets"], map_location="cpu", weights_only=False)
    target_genes = list(bank["target_genes"])
    esm = bank["target_esm"].float()
    background = bank["background_projected"].float()
    if background.ndim == 2:
        background = background.unsqueeze(0)
    background_se_tokens = bank.get("background_se_tokens")
    background_gene_esm = bank.get("background_gene_esm")
    if background_se_tokens is not None:
        background_se_tokens = torch.as_tensor(background_se_tokens, dtype=torch.float32)
    if background_gene_esm is not None:
        background_gene_esm = torch.as_tensor(background_gene_esm, dtype=torch.float32)
    targets = {g for name in train + test for g in parse_target_genes(name)}
    if len(set(target_genes)) != len(target_genes) or not targets <= set(target_genes):
        raise ValueError("External gene identities must uniquely cover all targets")
    if esm.shape != (len(target_genes), 5120) or background.ndim != 3 or background.shape[0] != 1 or background.shape[-1] != 1024:
        raise ValueError("Invalid ESM/background asset dimensions")
    if not torch.isfinite(esm).all() or not torch.isfinite(background).all():
        raise ValueError("Nonfinite condition assets")
    if background_se_tokens is not None or background_gene_esm is not None:
        if background_se_tokens is None or background_gene_esm is None:
            raise ValueError("Raw Norman background assets must provide both SE and ESM tokens")
        if background_se_tokens.shape != (1, 2047, 2048) or background_gene_esm.shape != (1, 2047, 5120):
            raise ValueError("Raw Norman background assets must be [1,2047,2048] and [1,2047,5120]")
        if not torch.isfinite(background_se_tokens).all() or not torch.isfinite(background_gene_esm).all():
            raise ValueError("Nonfinite raw Norman condition assets")
    metadata = bank.get("metadata", {})
    if metadata.get("complete") is not True or metadata.get("projector_kind") != "map_predictor":
        raise ValueError("Expected complete MAP-predictor gene projection assets")
    if metadata.get("train_control_rows") != train_controls.tolist():
        raise ValueError("Background was fitted on different controls")
    if metadata.get("train_control_rows_sha256") != gene_order_hash(train_controls):
        raise ValueError("Background train-control hash mismatch")
    if metadata.get("train_control_ids") != [str(cell_ids[int(i)]) for i in train_controls]:
        raise ValueError("Background train-control cell IDs mismatch")
    datasets = {}
    for name, rows in (("train", train_controls), ("test", test_controls)):
        dataset = NormanH5AD(paths["h5ad"], paths["split_manifest"], name, indices, normalize=False)
        dataset.control_rows = rows
        datasets[name] = dataset
    return dict(config=config, paths=paths, genes=genes, indices=indices, conditions=conditions,
                control=control, cell_ids=cell_ids, states=states, state_metadata=state_meta,
                split=split, train_controls=train_controls, test_controls=test_controls,
                dataset=datasets["train"], test_dataset=datasets["test"],
                target_lookup={g: i for i, g in enumerate(target_genes)},
                target_esm=esm.to(device), background=background.to(device),
                background_se_tokens=None if background_se_tokens is None else background_se_tokens.to(device),
                background_gene_esm=None if background_gene_esm is None else background_gene_esm.to(device),
                asset_metadata=metadata)


def training_binding(data):
    """Bind prediction/scoring to the same fitted split, controls and gene axis."""
    train_rows = np.flatnonzero(np.isin(data["conditions"], data["split"]["train_conditions"]) & ~data["control"])
    return dict(split_manifest_sha256=sha256_file(data["paths"]["split_manifest"]),
                control_split_sha256=sha256_file(data["paths"]["control_split"]),
                gene_order_sha256=gene_order_hash(data["genes"]),
                train_treated_rows_sha256=gene_order_hash(train_rows))


def kwargs_for(data, target_genes):
    indices = [data["target_lookup"][g] for g in target_genes]
    return {"target_gene_esm": data["target_esm"][indices].unsqueeze(0),
            "background_projected": data["background"],
            "background_se_tokens": data.get("background_se_tokens"),
            "background_gene_esm": data.get("background_gene_esm")}


def _from_checkpoint(cls, checkpoint_path, *, model_config=None, projector_checkpoint=None,
                     expected_genes=None, map_location="cpu"):
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if payload.get("model_family") != cls.model_family:
        raise ValueError("Checkpoint is not the native Norman gene-flow architecture")
    genes = payload.get("gene_symbols")
    if not genes or (expected_genes is not None and list(genes) != list(expected_genes)):
        raise ValueError("Checkpoint gene order differs from the dataset")
    state = payload["model_state_dict"]
    if any(key.startswith("mean_head.net.") for key in state):
        raise ValueError("This checkpoint uses the retired neural mean head. Use its archived experiment source or Git history to reproduce it; it cannot be loaded as a kernel head.")
    options = dict(payload.get("model_config", model_config or {}))
    if model_config is not None:
        options.update(model_config)
    options.pop("mean_head_hidden_dim", None)  # Old Flow-only model metadata.
    options.setdefault("expression_dim", len(genes))
    options.setdefault("path_sigma", payload.get("args", {}).get("path_sigma", 0.5))
    has_mean_head = any(key.startswith("mean_head.") for key in state)
    options["use_mean_head"] = has_mean_head
    if has_mean_head:
        if payload.get("mean_assets") is None or payload.get("mean_features") is None:
            raise ValueError("Kernel checkpoint lacks its fixed mean-query assets")
        options["mean_assets"] = payload["mean_assets"]
        options["mean_features"] = payload["mean_features"]
    if any(k.startswith("gene_condition.map_projector.") for k in state):
        options["map_projector"] = FrozenMAPGeneProjector(
            projector_checkpoint or {"model_state_dict": state},
            trainable_last_layer=bool(options.get("projector_trainable_last_layer", False)),
        )
    model = cls(**options)
    model.load_state_dict(state, strict=True)
    return model.to(map_location)


NormanGeneFlowMatchingSystem.from_checkpoint = classmethod(_from_checkpoint)
from devflow.model_attention import register_task_model
register_task_model("norman19", NormanGeneFlowMatchingSystem)
register_task_model("norman", NormanGeneFlowMatchingSystem)


def validate_config(config):
    data = load_inputs(config)
    return {"status": "passed", "task": "norman19", "setting": config.get("setting", "additive"),
            "train_conditions": len(data["dataset"]), "test_conditions": len(data["test_dataset"]),
            "genes": len(data["genes"]), "train_controls": len(data["train_controls"]),
            "test_controls": len(data["test_controls"]), "gene_order_sha256": gene_order_hash(data["genes"])}


def _capture_rng(rng, device):
    return {"sampler": rng.bit_generator.state, "python": random.getstate(),
            "numpy_global": np.random.get_state(), "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state(device) if device.type == "cuda" else None}


def _restore_rng(state, rng, device):
    rng.bit_generator.state = state["sampler"]
    random.setstate(state["python"])
    np.random.set_state(state["numpy_global"])
    torch.set_rng_state(state["torch_cpu"])
    if device.type == "cuda":
        torch.cuda.set_rng_state(state["torch_cuda"], device)


def _atomic_save(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def _append_csv(path, row):
    exists = path.exists()
    with path.open("a", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def train(config):
    """Native condition-sampled OTFM; invoke with torchrun for multi-GPU DDP."""
    settings = config["training"]
    rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(settings.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda":
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    if world > 1 and not dist.is_initialized():
        dist.init_process_group("nccl" if device.type == "cuda" else "gloo")
    seed = int(settings.get("seed", 42))
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    data = load_inputs(config, device)
    paths, genes = data["paths"], data["genes"]
    out = Path(config["output_dir"])
    if rank == 0:
        existing = [item for item in out.iterdir()
                    if item.name not in {"run_config.json", "launcher.log"}] if out.exists() else []
        if existing and not settings.get("resume_checkpoint"):
            raise FileExistsError("Use a new output directory for a fresh run")
        out.mkdir(parents=True, exist_ok=True)
    if world > 1:
        dist.barrier()
    options = dict(config.get("model", {})); options["expression_dim"] = len(genes)
    options.setdefault("path_sigma", settings.get("path_sigma", 0.5))
    projector_trainable = bool(settings.get(
        "projector_trainable_last_layer",
        options.get("projector_trainable_last_layer", False),
    ))
    options["projector_trainable_last_layer"] = projector_trainable
    resume = None
    if settings.get("resume_checkpoint"):
        resume = torch.load(settings["resume_checkpoint"], map_location="cpu", weights_only=False)
        if resume.get("split_manifest_sha256") != sha256_file(paths["split_manifest"]):
            raise ValueError("Resume checkpoint was fitted on a different condition split")
        system = NormanGeneFlowMatchingSystem.from_checkpoint(settings["resume_checkpoint"], model_config=options,
                    expected_genes=genes, map_location=device)
    else:
        if options.get("use_mean_head", False):
            mean_path = out / "kernel_mean_assets.pt"
            if rank == 0:
                from devflow.data.norman import prepare_kernel_mean_assets
                mean_assets, cv_report, mean_features = prepare_kernel_mean_assets(
                    data, config.get("mean_head", {}))
                _atomic_save(mean_path, {"mean_assets": mean_assets, "mean_features": mean_features})
                write_json(out / "kernel_cv_report.json", cv_report)
            if world > 1:
                dist.barrier()
            prepared = torch.load(mean_path, map_location="cpu", weights_only=False)
            options.update(prepared)
        options["map_projector"] = FrozenMAPGeneProjector(
            paths["projector_checkpoint"],
            trainable_last_layer=projector_trainable,
        )
        system = NormanGeneFlowMatchingSystem(**options).to(device)
        load_finetuned_decoder_checkpoint(system.decoder, paths["decoder_checkpoint"], genes)
        manifest = json.loads(Path(paths["decoder_training_manifest"]).read_text())
        if manifest.get("status") != "complete" or manifest.get("split_manifest_sha256") != sha256_file(paths["split_manifest"]):
            raise ValueError("Decoder must have completed fitting on precisely this split")
        decoder_genes = manifest.get("target_gene_symbols", manifest.get("gene_symbols", manifest.get("genes")))
        if list(decoder_genes or []) != genes:
            raise ValueError("Decoder training gene schema mismatch")
    system.decoder.requires_grad_(bool(settings.get("train_decoder", True)))
    system.decoder.eval()
    wrapper = FlowTrainingForward(system,
        # Norman's new route computes these inside the shared DDP forward.
        # The legacy expression weights remain available for old checkpoints.
        expression_mmd_weight=0.0 if settings.get("external_expression_loss", True) else settings.get("expression_mmd_weight", 5e-6),
        expression_mmd_time_points=settings.get("expression_mmd_time_points", 5),
        train_decoder=settings.get("train_decoder", True),
        expression_mean_weight=0.0 if settings.get("external_expression_loss", True) else settings.get("expression_mean_weight", 4.966990102795232e-5),
        expression_mean_decoder_only=settings.get("expression_mean_decoder_only", True),
        expression_residual_weight=settings.get("expression_residual_weight", 0.0),
        external_expression_loss=settings.get("external_expression_loss", True),
        mean_head_weight=settings.get("mean_head_weight", 1.0),
        decoder_mmd_weight=settings.get("decoder_mmd_weight", settings.get("expression_mmd_weight", 5e-6)))
    if world > 1:
        wrapper = DDP(wrapper, device_ids=[local_rank] if device.type == "cuda" else None,
                      broadcast_buffers=False, find_unused_parameters=True)
    named_parameters = [(name, parameter) for name, parameter in system.named_parameters()
                        if parameter.requires_grad]
    projector_prefix = "gene_condition.map_projector.projector.mlp.4."
    projector_parameters = [p for name, p in named_parameters if name.startswith(projector_prefix)]
    head_parameters = [p for name, p in named_parameters if name.startswith("mean_head.")]
    decoder_parameters = [p for name, p in named_parameters if name.startswith("decoder.")]
    flow_parameters = [p for name, p in named_parameters
                       if not name.startswith(("mean_head.", "decoder."))]
    base_parameters = [p for name, p in named_parameters
                       if not name.startswith(("mean_head.", "decoder.", projector_prefix))]
    base_lr = float(settings.get("learning_rate", 1e-4))
    projector_lr = float(settings.get("projector_last_layer_learning_rate", 1e-5))
    groups = []
    for group_parameters, lr in (
        (base_parameters, base_lr), (decoder_parameters, base_lr),
        (head_parameters, float(settings.get("mean_head_learning_rate", 1e-3))),
        (projector_parameters, projector_lr),
    ):
        if group_parameters:
            groups.append({"params": group_parameters, "lr": lr})
    clip_options = settings.get("gradient_clip", 1.0)
    clipping_groups = {"flow": flow_parameters, "head": head_parameters, "decoder": decoder_parameters}
    all_parameters = [parameter for _, parameter in named_parameters]
    optimizer = torch.optim.Adam(groups)
    rng = np.random.default_rng(np.random.SeedSequence([seed, rank]))
    start = 0
    if resume is not None:
        if settings.get("resume_optimizer", True):
            names = [n for n, p in system.named_parameters() if p.requires_grad]
            if resume.get("trainable_parameter_names") not in (None, names):
                raise ValueError("Optimizer parameter topology differs")
            optimizer.load_state_dict(resume["optimizer_state_dict"])
            if settings.get("override_learning_rate", False):
                for group, configured_group in zip(optimizer.param_groups, groups):
                    group["lr"] = configured_group["lr"]
        start = int(resume["step"])
        if settings.get("resume_rng", True):
            if resume.get("world_size") != world or len(resume.get("rank_rng_states", [])) != world:
                raise ValueError("Exact sampler resume requires unchanged world size and saved rank RNG states")
            _restore_rng(resume["rank_rng_states"][rank], rng, device)
    else:
        torch.manual_seed(seed + rank)
    steps = int(settings.get("steps", 50000))
    if steps <= start:
        raise ValueError("Target steps must exceed the starting checkpoint")
    control_mean = np.asarray(data["states"][data["train_controls"]], dtype=np.float64).mean(0).astype(np.float32)
    shifts = [np.asarray(data["states"][r["treated_rows"]], dtype=np.float64).mean(0).astype(np.float32) - control_mean
              for r in data["dataset"].records]
    batch = int(settings.get("batch_size", 128))
    if batch < 2:
        raise ValueError("batch_size must be at least two")
    model_options = {k: v for k, v in options.items()
                     if k not in {"map_projector", "mean_assets", "mean_features"}}
    if rank == 0:
        run_metadata = {"status": "training", "task": "norman19", "config": config,
                   "model_config": model_options, "gene_symbols": genes, "world_size": world,
                   "split_manifest_sha256": sha256_file(paths["split_manifest"]),
                   "start_step": start, "selection": "fixed update budget; no test selection",
                   "model_family": "mlp_mean",
                   "trainable_modules": [name for name, parameter in system.named_parameters() if parameter.requires_grad],
                   "gradient_routes": {"flow": "flow residual + latent mean shift",
                                      "mean_head": "full training-anchor KRR loss; only coefficients A",
                                      "decoder": "MMD on composed expression with detached Flow endpoint and head mean"},
                   "projector_last_layer_learning_rate": float(settings.get("projector_last_layer_learning_rate", 1e-5)),
                   "mean_head_weight": float(settings.get("mean_head_weight", 1.0)),
                   "decoder_mmd_weight": float(settings.get("decoder_mmd_weight", settings.get("expression_mmd_weight", 5e-6)))}
        write_json(out / "run_manifest.json", run_metadata)
        write_json(out / "run_config.json", config)
    system.train(); system.decoder.eval()
    started = time.monotonic()
    evaluation = config.get("evaluation", {})
    evaluation_every = int(evaluation.get("full_every", 0))
    evaluation_seeds = evaluation.get("monitor_seeds", [4243])
    if evaluation_every > 0 and list(evaluation_seeds) != [4243]:
        raise ValueError("Norman online evaluation uses the fixed formal seed 4243")
    probes = {}
    initial_head_max = None
    if rank == 0:
        for route, prefix in (("flow", "velocity."), ("decoder", "decoder."), ("head", "mean_head.")):
            candidates = [(name, parameter) for name, parameter in named_parameters
                          if name.startswith(prefix) and (parameter.ndim == 1 or route == "head")]
            probes[route] = [(name, parameter, parameter.detach().reshape(-1)[:128].clone())
                             for name, parameter in candidates[:4]]
        if system.mean_head is not None:
            initial_head_max = float(system.mean_head.A.detach().abs().max())
    for step in range(start + 1, steps + 1):
        index = int(rng.integers(len(data["dataset"])))
        record = data["dataset"].records[index]
        pool = record["treated_rows"]
        source_rows = rng.choice(data["train_controls"], batch, replace=len(data["train_controls"]) < batch)
        target_rows = rng.choice(pool, batch, replace=len(pool) < batch)
        source = torch.as_tensor(np.array(data["states"][source_rows], copy=True), device=device)
        target = torch.as_tensor(np.array(data["states"][target_rows], copy=True), device=device)
        expression = torch.as_tensor(read_expression_rows(paths["h5ad"], target_rows, data["indices"]), device=device)
        generator = torch.Generator(device=device).manual_seed(seed * 1000003 + (step - 1) * world + rank)
        optimizer.zero_grad(set_to_none=True)
        context = torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" and settings.get("amp_dtype", "bf16") == "bf16" else nullcontext()
        with context:
            condition_kwargs = kwargs_for(data, record["target_genes"])
            loss, parts = wrapper(source, target, condition_kwargs,
                torch.as_tensor(shifts[index], device=device), generator,
                float(settings.get("ot_epsilon", 1.0)), int(settings.get("ot_max_iterations", 2000)),
                float(settings.get("ot_tolerance", 1e-3)), float(settings.get("mean_shift_weight", 0.005)),
                target_expression=expression,
                apply_expression_mmd=step % int(settings.get("expression_mmd_every", 1)) == 0,
                mean_condition=record["condition"])
        finite = torch.tensor(int(torch.isfinite(loss).item()), device=device)
        if world > 1:
            dist.all_reduce(finite, op=dist.ReduceOp.MIN)
        if not int(finite.item()):
            raise FloatingPointError(f"Nonfinite loss at step {step}")
        # Backward deliberately occurs outside autocast.
        loss.backward()
        if isinstance(clip_options, Mapping):
            group_norms = {
                name: torch.nn.utils.clip_grad_norm_(group_parameters, float(clip_options.get(name, 1.0))).to(device)
                for name, group_parameters in clipping_groups.items() if group_parameters
            }
            norm = torch.stack(list(group_norms.values())).norm()
        else:
            # Scalar settings preserve the original single global norm clip.
            norm = torch.nn.utils.clip_grad_norm_(all_parameters, float(clip_options)).to(device)
            group_norms = {"global": norm}
        finite = torch.tensor(int(torch.isfinite(norm).item()), device=device)
        if world > 1:
            dist.all_reduce(finite, op=dist.ReduceOp.MIN)
        if not int(finite.item()):
            raise FloatingPointError(f"Nonfinite gradient at step {step}")
        optimizer.step()
        if step == start + 1 and rank == 0:
            updates = {route: {name: float((parameter.detach().reshape(-1)[:128] - before).abs().max())
                              for name, parameter, before in items}
                       for route, items in probes.items()}
            write_json(out / "FIRST_UPDATE.json", {
                "optimizer_step": step, "start_step": start, "world_size": world,
                "flow_initialized_from_scratch": resume is None,
                "decoder_initial_checkpoint": str(paths["decoder_checkpoint"]),
                "decoder_initial_checkpoint_sha256": sha256_file(paths["decoder_checkpoint"]),
                "head_initial_max_abs_coefficient": initial_head_max,
                "head_initialized_from_zero": initial_head_max == 0.0 if resume is None else None,
                "sampled_parameter_max_abs_updates": updates,
                "updated": {route: any(value > 0 for value in changes.values())
                            for route, changes in updates.items()},
                "gradient_clipping": "per_group" if isinstance(clip_options, Mapping) else "global",
                "gradient_clip": clip_options,
            })
            probes.clear()
        if step == start + 1 or step % int(settings.get("log_every", 10)) == 0 or step == steps:
            row = {"step": step, "total_loss": float(loss.detach()), "gradient_norm_before_clip": float(norm),
                   "learning_rate": optimizer.param_groups[0]["lr"],
                   "projector_last_layer_learning_rate": (
                       optimizer.param_groups[-1]["lr"] if projector_parameters else 0.0
                   ),
                   "elapsed_seconds": time.monotonic() - started}
            row.update({f"gradient_norm_{name}_before_clip": float(value)
                        for name, value in group_norms.items()})
            metric_names = ("flow_loss", "residual_flow_loss", "mean_shift_loss",
                            "head_loss", "weighted_head_loss", "head_fit_loss", "head_regularization_loss",
                            "decoder_mmd_loss", "weighted_decoder_mmd_loss",
                            "expression_mmd_loss", "weighted_expression_mmd_loss",
                            "expression_mean_loss", "weighted_expression_mean_loss",
                            "expression_residual_loss", "weighted_expression_residual_loss",
                            "endpoint_latent_variance_ratio", "endpoint_expression_variance_ratio")
            for name in metric_names:
                value = parts.get(name)
                row[name] = float(value.detach()) if isinstance(value, torch.Tensor) else 0.0
            row["weighted_mean_shift_loss"] = float(settings.get("mean_shift_weight", 0.005)) * row["mean_shift_loss"]
            if world > 1:
                keys = list(row)
                values = torch.tensor([row[k] for k in keys], device=device, dtype=torch.float64)
                dist.all_reduce(values); values /= world
                row = dict(zip(keys, values.tolist()))
            if rank == 0:
                _append_csv(out / "train_metrics.csv", row)
                write_json(out / "status.json", {"status": "training", "step": step})
        pause = torch.tensor(int(rank == 0 and (out / "REQUEST_PAUSE").exists()), device=device)
        if world > 1:
            dist.broadcast(pause, 0)
        evaluate_now = evaluation_every > 0 and step % evaluation_every == 0 and not pause.item()
        if step % int(settings.get("save_every", 5000)) == 0 or step == steps or pause.item() or evaluate_now:
            local_state = _capture_rng(rng, device)
            states = [None] * world if rank == 0 else None
            if world > 1:
                dist.gather_object(local_state, states, dst=0)
            else:
                states = [local_state]
            if rank == 0:
                _atomic_save(out / f"checkpoint_step_{step:08d}.pt", {
                    "model_family": system.model_family, "model_state_dict": system.state_dict(),
                    "mean_assets": system.export_mean_assets(), "mean_features": system.mean_features,
                    "optimizer_state_dict": optimizer.state_dict(), "model_config": model_options,
                    "step": step, "gene_symbols": genes, "world_size": world,
                    "resume_schema": 2, "rank_rng_states": states,
                    "trainable_parameter_names": [n for n, p in system.named_parameters() if p.requires_grad],
                    "split_manifest_sha256": sha256_file(paths["split_manifest"]),
                    "config": config, "decoder_frozen": not settings.get("train_decoder", True)})
            if world > 1:
                dist.barrier()
        if evaluate_now:
            evaluation_ok = torch.ones((), device=device, dtype=torch.int32)
            if rank == 0:
                write_json(out / "status.json", {"status": "evaluating", "step": step})
                env = os.environ.copy()
                for key in list(env):
                    if key in {"RANK", "WORLD_SIZE", "LOCAL_RANK", "LOCAL_WORLD_SIZE", "GROUP_RANK", "ROLE_RANK", "ROLE_WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"} or key.startswith("TORCHELASTIC_"):
                        env.pop(key, None)
                command = [sys.executable, "-m", "benchmark.reproduce.norman_checkpoint_monitor",
                           "--config", str(out / "run_config.json"), "--run", str(out),
                           "--steps", str(step), "--seed", "4243",
                           "--references", str(Path(config["references"]["output_dir"]).resolve())]
                try:
                    with (out / "online_evaluation.log").open("a") as stream:
                        subprocess.run(command, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)
                except Exception as exc:
                    evaluation_ok.zero_()
                    write_json(out / "ONLINE_EVALUATION_FAILED.json", {
                        "optimizer_step": step, "error": str(exc), "command": command})
            if world > 1:
                dist.broadcast(evaluation_ok, 0)
            if not int(evaluation_ok.item()):
                raise RuntimeError(f"Formal online evaluation failed at optimizer step {step}; see online_evaluation.log")
            if rank == 0:
                write_json(out / "status.json", {"status": "training", "step": step})
        if pause.item():
            if rank == 0:
                write_json(out / "status.json", {"status": "paused", "step": step})
            return {"status": "paused", "step": step}
    if rank == 0:
        manifest_path = out / "run_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest.update(status="complete", final_step=steps)
        write_json(manifest_path, manifest)
        write_json(out / "status.json", {"status": "complete", "step": steps})
        write_json(out / "TRAINING_COMPLETE.json", {"status": "complete", "step": steps,
                   "checkpoint": str(out / f"checkpoint_step_{steps:08d}.pt")})
    return {"status": "complete", "step": steps}


def train_decoder(config):
    """Gene-align the decoder then fit only declared training populations."""
    settings = config["decoder_training"]
    device = torch.device(settings.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    data = load_inputs(config, device)
    paths, genes = data["paths"], data["genes"]
    seed = int(settings.get("seed", 42)); torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    decoder = LatentToGeneDecoder(2048, len(genes), [512, 1024], output_activation="relu").to(device)
    if settings.get("resume_checkpoint"):
        if not settings.get("resume_training_manifest"):
            raise ValueError("Decoder warm start requires its training provenance manifest")
        prior = json.loads(Path(settings["resume_training_manifest"]).read_text())
        if prior.get("status") != "complete" or prior.get("split_manifest_sha256") != sha256_file(paths["split_manifest"]):
            raise ValueError("Warm-start decoder was fitted on a different/incomplete split")
        if list(prior.get("target_gene_symbols", prior.get("gene_symbols", []))) != genes:
            raise ValueError("Warm-start decoder training gene order mismatch")
        load_finetuned_decoder_checkpoint(decoder, settings["resume_checkpoint"], genes)
    else:
        initialize_decoder_from_map_checkpoint(decoder, paths["source_decoder_checkpoint"],
                source_genes=paths["source_decoder_genes"], target_genes=genes, seed=seed)
    out = Path(settings["output_dir"])
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Decoder fitting requires a new output directory")
    out.mkdir(parents=True)
    optimizer = torch.optim.AdamW(decoder.parameters(), lr=float(settings.get("learning_rate", 1e-4)))
    group_rows = {r["condition"]: r["treated_rows"] for r in data["dataset"].records}
    group_rows["__control__"] = data["train_controls"]
    groups = [group_rows[name] for name in sorted(group_rows)]
    batch = int(settings.get("batch_size", 128)); steps = int(settings.get("steps", 50000))
    decoder.train()
    for step in range(1, steps + 1):
        pool = groups[int(rng.integers(len(groups)))]
        rows = rng.choice(pool, batch, replace=len(pool) < batch)
        latent = torch.as_tensor(np.array(data["states"][rows], copy=True), device=device)
        expression = torch.as_tensor(read_expression_rows(paths["h5ad"], rows, data["indices"]), device=device)
        optimizer.zero_grad(set_to_none=True)
        loss, parts = decoder_finetune_loss(decoder, latent, expression,
            mean_loss_weight=float(settings.get("mean_loss_weight", 1.0)),
            mmd_weight=float(settings.get("mmd_weight", 1.0)), return_components=True)
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite decoder objective")
        loss.backward(); optimizer.step()
        if step == 1 or step % int(settings.get("log_every", 100)) == 0 or step == steps:
            _append_csv(out / "train_metrics.csv", {"step": step, "total_loss": float(loss.detach())})
        if step % int(settings.get("save_every", 5000)) == 0 or step == steps:
            save_finetuned_decoder_checkpoint(out / f"checkpoint_step_{step:08d}.pt", decoder, genes, step=step,
                extra_metadata={"split_manifest_sha256": sha256_file(paths["split_manifest"])})
    write_json(out / "run_manifest.json", {"status": "complete", "step": steps,
               "split_manifest_sha256": sha256_file(paths["split_manifest"]), "target_gene_symbols": genes,
               "config": config, "train_conditions": data["split"]["train_conditions"],
               "train_control_rows": data["train_controls"].tolist()})
    write_json(out / "status.json", {"status": "complete", "step": steps})
    return {"status": "complete", "step": steps}


def build_sampling_plan(data, seeds, cells_per_condition=128):
    """Exact original repeated-128 RNG order, with replacement only if needed."""
    if int(cells_per_condition) < 2:
        raise ValueError("At least two evaluation cells are required")
    output = []
    for seed in seeds:
        rng = np.random.default_rng(int(seed))
        for record in sorted(data["test_dataset"].records, key=lambda r: r["condition"]):
            target = np.sort(rng.choice(record["treated_rows"], cells_per_condition,
                                      replace=len(record["treated_rows"]) < cells_per_condition))
            source = np.sort(rng.choice(data["test_controls"], cells_per_condition,
                                      replace=len(data["test_controls"]) < cells_per_condition))
            output.append({"seed": int(seed), "condition": record["condition"],
                           "target_genes": list(record["target_genes"]),
                           "treated_rows": target.tolist(), "control_rows": source.tolist()})
    return output



def _validated_sampling_plan(data, settings):
    seeds = [int(x) for x in settings.get("seeds", [4243, 4244, 4245, 4246])]
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Prediction seeds must be nonempty and unique")
    count = int(settings.get("cells_per_condition", 128))
    plan = build_sampling_plan(data, seeds, count)
    if settings.get("sampling_manifest"):
        declared = json.loads(Path(settings["sampling_manifest"]).read_text())
        plan = declared.get("records", declared) if isinstance(declared, dict) else declared
    by_name = {r["condition"]: r for r in data["test_dataset"].records}
    if {int(r["seed"]) for r in plan} != set(seeds):
        raise ValueError("Sampling manifest seeds differ from the declared protocol")
    for seed in seeds:
        rows = [r for r in plan if int(r["seed"]) == seed]
        if len(rows) != len(by_name) or {r["condition"] for r in rows} != set(by_name):
            raise ValueError("Sampling manifest must cover every test condition once per seed")
        for record in rows:
            source = np.asarray(record["control_rows"], dtype=np.int64)
            target = np.asarray(record["treated_rows"], dtype=np.int64)
            if source.shape != (count,) or target.shape != (count,):
                raise ValueError("Sampling rows differ from cells_per_condition")
            if not np.isin(source, data["test_controls"]).all():
                raise ValueError("Sampling controls leave the held-out control pool")
            if not np.isin(target, by_name[record["condition"]]["treated_rows"]).all():
                raise ValueError("Sampling targets leave their declared test condition")
            if tuple(record.get("target_genes", by_name[record["condition"]]["target_genes"])) != tuple(by_name[record["condition"]]["target_genes"]):
                raise ValueError("Sampling target identities differ from their condition")
    return seeds, sorted(plan, key=lambda r: (seeds.index(int(r["seed"])), r["condition"]))


def _reporting_bh_mask(pvalues):
    pvalues = np.asarray(pvalues, dtype=np.float64)
    order = np.flatnonzero(np.isfinite(pvalues))
    order = order[np.argsort(pvalues[order])]
    hit = np.flatnonzero(pvalues[order] <= .05 * np.arange(1, len(order) + 1) / max(len(order), 1))
    mask = np.zeros(len(pvalues), dtype=bool)
    if len(hit):
        mask[order[:hit[-1] + 1]] = True
    return mask


def export_references(config):
    """Write evaluation-only references; never called by either training action."""
    from scipy.stats import ttest_ind
    from benchmark.export_references import write_reference_bundle
    from devflow.io import json_sha256

    data = load_inputs(config, "cpu")
    settings = config.get("references", {})
    output = Path(settings["output_dir"])
    if (output / "manifest.json").exists():
        raise FileExistsError("Refusing to overwrite an existing reference bundle")
    seeds, plan = _validated_sampling_plan(data, config["prediction"])
    groups = {}
    for name in sorted(data["split"]["train_conditions"] + data["split"]["test_conditions"]):
        rows = np.flatnonzero((data["conditions"] == name) & ~data["control"])
        values = read_expression_rows(data["paths"]["h5ad"], rows, data["indices"]).astype(np.float64)
        groups[name] = (values.mean(0), values.var(0, ddof=1), len(rows))
    train_names = sorted(data["split"]["train_conditions"])
    train_mean = np.mean(np.stack([groups[name][0] for name in train_names]), axis=0)
    names = sorted(data["split"]["test_conditions"])
    held_controls = read_expression_rows(data["paths"]["h5ad"], data["test_controls"], data["indices"])
    records = {r["condition"]: r for r in data["test_dataset"].records}
    weights, masks, panels = [], [], []
    for name in names:
        mu, var, count = groups[name]
        rest = [value for other, value in groups.items() if other != name]
        rest_count = sum(int(n) for _, _, n in rest)
        rest_mean = sum(np.asarray(m, dtype=np.float64) * int(n) for m, _, n in rest) / rest_count
        rest_squares = sum(np.asarray(v, dtype=np.float64) * (int(n)-1) + np.square(m) * int(n)
                           for m, v, n in rest)
        rest_var = np.maximum((rest_squares - np.square(rest_mean) * rest_count) / max(rest_count - 1, 1), 0)
        denominator = np.sqrt((np.maximum(var, 0) + rest_var) / float(count))
        score = np.divide(mu-rest_mean, denominator, out=np.zeros_like(mu), where=denominator > 0)
        strength = np.nan_to_num(np.abs(score), nan=0., posinf=0., neginf=0.)
        lo, hi = float(strength.min()), float(strength.max())
        weight = np.square((strength-lo)/(hi-lo)) if hi > lo else np.zeros_like(strength)
        weights.append(weight / weight.sum() if weight.sum() > 0 else weight)
        values = read_expression_rows(data["paths"]["h5ad"], records[name]["treated_rows"], data["indices"])
        # Original fixed masks used FP32 X; later fixed DE50 used FP64 Welch.
        mask_test = ttest_ind(values, held_controls, axis=0, equal_var=False, nan_policy="omit")
        masks.append(_reporting_bh_mask(mask_test.pvalue))
        panel_test = ttest_ind(values.astype(np.float64), held_controls.astype(np.float64),
                              axis=0, equal_var=False, nan_policy="omit")
        strength = np.nan_to_num(np.abs(panel_test.statistic), nan=-np.inf, posinf=np.inf, neginf=-np.inf)
        panels.append(np.lexsort((np.arange(len(data["genes"])), -strength))[:min(50, len(data["genes"]))])
    held = set(config.get("heldout_genes", []))
    scopes = ["single" if len(parse_target_genes(name)) == 1 else
              ("two_unseen" if len(held.intersection(parse_target_genes(name))) == 2 else "one_unseen")
              for name in names] if config.get("setting") == "gene_holdout" else ["additive"] * len(names)
    statistics = dict(condition_names=np.asarray(names), context_names=np.asarray(["K562"] * len(names)),
        gene_order_sha256=np.asarray(gene_order_hash(data["genes"])),
        continuous_deg_weights=np.stack(weights), de_mask=np.stack(masks), de50_indices=np.stack(panels),
        training_condition_mean=train_mean[None], reference_index=np.zeros(len(names), dtype=np.int64),
        reporting_scopes=np.asarray(scopes))
    samples = {}
    for seed in seeds:
        sampled = [r for r in plan if int(r["seed"]) == seed]
        payload = dict(seed=np.asarray(seed), condition_names=np.asarray(names),
            source_global_ids=np.asarray([r["control_rows"] for r in sampled], dtype=np.int64),
            target_global_ids=np.asarray([r["treated_rows"] for r in sampled], dtype=np.int64),
            gene_order_sha256=np.asarray(gene_order_hash(data["genes"])))
        for i, record in enumerate(sampled):
            payload[f"target_{i:04d}"] = read_expression_rows(data["paths"]["h5ad"], record["treated_rows"], data["indices"])
            payload[f"control_{i:04d}"] = read_expression_rows(data["paths"]["h5ad"], record["control_rows"], data["indices"])
        samples[seed] = payload
    binding = training_binding(data)
    manifest = write_reference_bundle(output, statistics=statistics, samples=samples,
        training_provenance=dict(fitted_on="training_only", fit_row_ids_sha256=binding["train_treated_rows_sha256"],
            training_binding=binding, averaging=f"Equal average of {len(train_names)} training-treated condition means; FP64"),
        evaluation_provenance=dict(use="reporting_only", frozen=True,
            continuous_deg_weights="Current condition vs pooled other treated conditions; overestimated t variance; squared min-max absolute t normalized to sum 1",
            de_mask="All treated in condition vs complete held-out controls; Welch/BH q<0.05; fixed across seeds",
            de50_indices="Full-condition vs held-out-control contrast; FP64 absolute Welch t top50; stable retained-index ties",
            sampling="Shared sorted NumPy default_rng rows; replacement only for undersized pools"),
        source_hashes={key: sha256_file(data["paths"][key]) for key in
                       ("h5ad", "split_manifest", "control_split", "gene_indices", "genes", "hvg_provenance")
                       if data["paths"].get(key)})
    template = settings.get("protocol_template")
    if not template:
        name = "norman19_holdout12.json" if config.get("setting") == "gene_holdout" else "norman19_additive.json"
        template = Path(__file__).resolve().parents[2] / "benchmark" / "protocols" / name
    protocol = json.loads(Path(template).read_text())
    protocol.update(seeds=seeds, condition_count=len(names), n_genes=len(data["genes"]),
                    cells_per_condition=int(config["prediction"].get("cells_per_condition", 128)),
                    gene_order_sha256=gene_order_hash(data["genes"]),
                    training_binding_sha256=json_sha256(binding))
    write_json(output / "protocol.json", protocol)
    write_json(output / "sampling_manifest.json", {"records": plan})
    return {"status": "complete", "reference_directory": str(output), "files": len(manifest["files_sha256"])}


def predict(config):
    settings = config["prediction"]
    device = torch.device(settings.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    data = load_inputs(config, device)
    binding = training_binding(data)
    model = NormanGeneFlowMatchingSystem.from_checkpoint(settings["checkpoint"], model_config=config.get("model"),
                    expected_genes=data["genes"], map_location=device).eval().requires_grad_(False)
    payload = torch.load(settings["checkpoint"], map_location="cpu", weights_only=False)
    if payload["split_manifest_sha256"] != sha256_file(data["paths"]["split_manifest"]):
        raise ValueError("Evaluation split differs from the checkpoint training split")
    seeds, plan = _validated_sampling_plan(data, settings)
    records_by_name = {r["condition"]: r for r in data["test_dataset"].records}
    # Checkpointed kernel assets supply the condition mean. Compose only after
    # all raw decoder chunks for a condition have been concatenated.
    out = Path(settings.get("output_dir", str(Path(config["output_dir"]) / "predictions")))
    out.mkdir(parents=True, exist_ok=True)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    paths = []
    with torch.inference_mode():
        for seed in seeds:
            records = [r for r in plan if int(r["seed"]) == int(seed)]
            if len(records) != len(records_by_name) or {r["condition"] for r in records} != set(records_by_name):
                raise ValueError("Sampling manifest must cover every test condition once per seed")
            predictions, sources, targets, names = [], [], [], []
            for r in records:
                source = np.asarray(r["control_rows"], dtype=np.int64)
                target = np.asarray(r["treated_rows"], dtype=np.int64)
                if len(source) != len(target) or not np.isin(source, data["test_controls"]).all():
                    raise ValueError("Sampling rows leave the held-out control pool")
                if not np.isin(target, records_by_name[r["condition"]]["treated_rows"]).all():
                    raise ValueError("Sampling target rows leave the declared condition")
                kwargs = kwargs_for(data, records_by_name[r["condition"]]["target_genes"])
                raw_chunks = []
                for start in range(0, len(source), int(settings.get("chunk_size", 128))):
                    states = torch.as_tensor(np.array(data["states"][source[start:start + int(settings.get("chunk_size", 128))]], copy=True), device=device)
                    endpoint = model.sample(states, states, kwargs, time_points=int(settings.get("time_points", 80)))
                    raw_chunks.append(model.decoder(endpoint).float())
                raw_prediction = torch.cat(raw_chunks, dim=0)
                if model.mean_head is not None:
                    final_mean = model.predict_condition_mean(r["condition"])
                    prediction = compose_expression(
                        raw_prediction, final_mean, detach_mean=True,
                        output_dtype=torch.float32,
                    )
                else:
                    prediction = raw_prediction
                predictions.append(prediction.cpu().numpy())
                sources.append(source); targets.append(target); names.append(r["condition"])
            path = out / f"predictions_seed_{int(seed)}.npz"
            if path.exists() and not settings.get("overwrite", False):
                raise FileExistsError(path)
            write_prediction_archive(path, seed=seed, condition_names=names,
                source_global_ids=np.stack(sources), target_global_ids=np.stack(targets),
                gene_order_sha256=gene_order_hash(data["genes"]), predictions=predictions,
                metadata={"task": "norman19", "checkpoint_sha256": sha256_file(settings["checkpoint"]),
                          "training_binding": binding, "training_binding_sha256": json_sha256(binding),
                          "time_points": int(settings.get("time_points", 80)), "precision": "FP32; TF32 disabled"})
            paths.append(str(path))
    write_json(out / "sampling_manifest.json", {"records": plan})
    return {"status": "complete", "archives": paths}
