#!/usr/bin/env python3
"""Build a frozen, train-control-mean gene background for Norman conditions.

X is already log-normalized. expm1(mean(X_control)) is only an interface
conversion into the original tokenizer, whose log1p restores that mean profile.
Gene-level SE outputs (not CLS cell states) and aligned ESM2 embeddings are
projected by the original MAP predictor's frozen gene_tokens_projector.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import shutil
import h5py
import numpy as np
import torch
import torch.nn.functional as F
from scipy import sparse

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "dependencies"))
from flowmatching.norman_gene_model import FrozenMAPGeneProjector
from prepare_norman_decoder import h5_metadata, parse_condition_genes

def sha(path):
    d = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(8 * 1024**2), b""):
            d.update(b)
    return d.hexdigest()

def ordered(values):
    return hashlib.sha256("\n".join(map(str, values)).encode()).hexdigest()

def _repository_commit():
    """Return the containing repository HEAD, or None outside a Git checkout."""
    repository = Path(__file__).resolve().parents[5]
    try:
        return subprocess.check_output(
            ["git", "-C", str(repository), "rev-parse", "HEAD"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _encode_target_complete(tokenizer, indices, raw_counts, target_ids):
    """Keep every target within the original gene budget and expression scale.

    This changes selection only: duplicate-symbol aggregation, log1p and the
    denominator over *all* mapped positive genes match Tokenizer.encode().
    Zero-expression targets retain zero weight, rather than pseudo-counts.
    """
    target_ids = np.asarray(target_ids, dtype=np.int32)
    if len(np.unique(target_ids)) != len(target_ids):
        raise ValueError("Target token IDs must be unique")
    budget = tokenizer.pad_length - 1
    if len(target_ids) > budget:
        raise ValueError("Targets exceed the SE gene-token budget")
    tokens = tokenizer.gene_to_token[np.asarray(indices, dtype=np.int64)]
    counts = np.asarray(raw_counts, dtype=np.float64)
    valid = (tokens >= 0) & (counts > 0)
    tokens, counts = tokens[valid], counts[valid]
    if not len(tokens):
        raise ValueError("Training control profile has no mapped positive genes")
    order = np.argsort(tokens, kind="stable")
    tokens, counts = tokens[order], counts[order]
    starts = np.flatnonzero(np.r_[True, tokens[1:] != tokens[:-1]])
    tokens, counts = tokens[starts], np.add.reduceat(counts, starts)
    transformed = np.log1p(counts)
    denominator = float(transformed.sum())
    if not np.isfinite(denominator) or denominator <= 0:
        raise ValueError("Invalid mapped expression denominator")
    all_weights = np.zeros(len(tokenizer.fill_order), dtype=np.float64)
    all_transformed = np.zeros_like(all_weights)
    all_weights[tokens] = 100.0 * (transformed / denominator)
    all_transformed[tokens] = transformed
    ranked = tokens[np.lexsort((tokenizer.tie_priority[tokens], -transformed))]
    others = ranked[~np.isin(ranked, target_ids)][:budget - len(target_ids)]
    selected = np.concatenate((target_ids, others))
    if len(selected) < budget:
        fillers = tokenizer.fill_order[~np.isin(tokenizer.fill_order, selected)]
        selected = np.concatenate((selected, fillers[:budget - len(selected)]))
    selected = selected[np.lexsort((tokenizer.tie_priority[selected], -all_transformed[selected]))]
    if len(selected) != budget or len(np.unique(selected)) != budget:
        raise AssertionError("SE gene positions must contain unique tokens")
    ids = np.zeros(tokenizer.pad_length, dtype=np.int32)
    weights = np.zeros(tokenizer.pad_length, dtype=np.float32)
    ids[1:], weights[1:] = selected, all_weights[selected].astype(np.float32)
    positions = np.asarray([int(np.flatnonzero(selected == token)[0]) for token in target_ids])
    return ids, weights, positions, denominator


def _extract_target_se(a, meta, rows, targets, target_esm, symbols, lookup,
                       tokenizer, ext, profile):
    """Produce one immutable train-control-context token for every target."""
    started = time.monotonic()
    target_ids = np.asarray([lookup[g] for g in targets], dtype=np.int32)
    if not set(targets) <= set(meta["source_genes"]):
        raise ValueError("Every target must occur in the source expression axis")
    raw = np.expm1(profile)
    original_ids, original_weights = tokenizer.encode(np.arange(len(profile)), raw)
    ids, weights, positions, denominator = _encode_target_complete(
        tokenizer, np.arange(len(profile)), raw, target_ids)
    # Every retained original token has exactly its previous expression weight.
    original_lookup = {int(token): i + 1 for i, token in enumerate(original_ids[1:])}
    shared = [(i + 1, original_lookup[int(token)]) for i, token in enumerate(ids[1:])
              if int(token) in original_lookup]
    if not all(weights[new] == original_weights[old] for new, old in shared):
        raise AssertionError("Target inclusion changed shared gene expression weights")
    if not np.array_equal(ids[positions + 1], target_ids):
        raise AssertionError("Target positions do not match token IDs")
    device = torch.device(a.device)
    model = ext.load_model(a.se_checkpoint, None, device)
    if model.training or any(p.requires_grad for p in model.parameters()):
        raise AssertionError("SE must be frozen and in eval mode")
    embedding_rows = model.pe_embedding.weight[
        torch.as_tensor(target_ids, device=device, dtype=torch.long)].detach().float().cpu()
    if not torch.equal(embedding_rows, target_esm):
        raise ValueError("Target ESM2 dictionary order differs from the SE checkpoint")
    ids_t = torch.from_numpy(ids[None].copy()).long().to(device)
    counts_t = torch.from_numpy(weights[None].copy()).float().to(device)
    with torch.inference_mode(), torch.autocast(
            device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
        src = F.normalize(model.pe_embedding(ids_t), dim=2)
        src = torch.cat((model.cls_token.expand(1, 1, -1), src[:, 1:, :]), dim=1)
        if model.dataset_token is not None:
            src = torch.cat((src, model.dataset_token.expand(1, 1, -1)), dim=1)
        gene_output, _, _ = model.forward(src=src, counts=counts_t)
        target_se = gene_output[0, torch.as_tensor(positions + 1, device=device)].float().cpu().clone()
    if target_se.shape != (len(targets), 2048) or not torch.isfinite(target_se).all():
        raise ValueError("Invalid target SE gene tokens")
    if (target_se.norm(dim=1) <= 0).any():
        raise ValueError("A target SE token has zero norm")
    del model, src, gene_output
    if device.type == "cuda":
        torch.cuda.empty_cache()
    repository = Path(__file__).resolve().parents[5]
    source_files = [Path(__file__).resolve(), repository / "preprocessing/norman.py",
                    Path(ext.__file__).resolve()]
    se_root = repository / "preprocessing/_vendor/state_encoder"
    source_files += [se_root / value for value in (
        "model/se.py", "model/flash_transformer.py", "data/utils.py", "configs/se600m.yaml")]
    source_identity = {}
    for source in source_files:
        relative = source.relative_to(repository)
        destination = a.output_dir / "source_snapshot" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        source_identity[str(relative)] = sha(source)
    record = {
        "schema": "norman_target_se_control_context_v1", "format_version": 1,
        "complete": True, "created_unix": time.time(),
        "git_commit": _repository_commit(), "source_files_sha256": source_identity,
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()},
        "h5ad_sha256": sha(a.h5ad), "split_manifest_sha256": sha(a.split_manifest),
        "control_split_sha256": sha(a.control_split),
        "esm_checkpoint_sha256": sha(a.esm_checkpoint),
        "se_checkpoint_sha256": sha(a.se_checkpoint),
        "projector_checkpoint_sha256": sha(a.projector_checkpoint),
        "cell_order_sha256": ordered(meta["cell_ids"]),
        "train_control_rows": rows.tolist(),
        "train_control_ids": [meta["cell_ids"][i] for i in rows],
        "train_control_rows_sha256": ordered(rows), "n_train_controls": len(rows),
        "target_genes": targets, "target_gene_order_sha256": ordered(targets),
        "context_genes": [symbols[int(i)] for i in ids[1:]],
        "context_gene_order_sha256": ordered([symbols[int(i)] for i in ids[1:]]),
        "selection": "all 101 targets plus highest-expression 1946 non-targets; original expression/tie ordering",
        "normalization": "mean stored lognormalized training-control X; expm1 interface conversion; original tokenizer log1p and denominator over all mapped positive genes",
        "mapped_log_expression_denominator": denominator,
        "background_kind": "train_control_pool_mean",
        "se_tokens": "same-forward gene_output at target token positions; excluding CLS and dataset token",
        "se_frozen": True, "se_eval": True, "n_forward_passes": 1,
        "amp_dtype": "float16" if device.type == "cuda" else "float32",
        "stored_dtype": "float32", "test_expression_used": False,
        "n_gene_positions": len(ids) - 1, "n_targets": len(targets),
        "original_background_target_count": sum(int(x) in original_lookup for x in target_ids),
        "shared_token_expression_weights_bitwise_equal": True,
        "zero_expression_targets": [targets[i] for i in range(len(targets)) if weights[positions[i] + 1] == 0],
        "shapes": {"target_se_tokens": list(target_se.shape), "target_esm": list(target_esm.shape)},
        "seconds": time.monotonic() - started,
    }
    output = a.output_dir / "target_se_assets.pt"
    torch.save({"target_genes": targets, "target_se_tokens": target_se,
                "target_esm": target_esm.cpu(), "target_token_ids": torch.from_numpy(target_ids.copy()),
                "target_counts": torch.from_numpy(weights[positions + 1].copy()),
                "target_positions": torch.from_numpy(positions + 1),
                "token_ids": torch.from_numpy(ids), "token_counts": torch.from_numpy(weights),
                "metadata": record}, output)
    record["asset_sha256"] = sha(output)
    (a.output_dir / "target_se_assets.json").write_text(json.dumps(record, indent=2) + "\n")
    (a.output_dir / "run_config.json").write_text(json.dumps(record["args"], indent=2) + "\n")
    (a.output_dir / "invocation.json").write_text(json.dumps({
        "executable": sys.executable, "argv": sys.argv, "cwd": os.getcwd(),
        "source_files_sha256": source_identity}, indent=2) + "\n")
    print(json.dumps({key: value for key, value in record.items() if key not in (
        "train_control_rows", "train_control_ids", "context_genes", "target_genes")}, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("h5ad", "split-manifest", "control-split", "esm-checkpoint",
                 "se-checkpoint", "projector-checkpoint", "output-dir"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--target-se-only", action="store_true",
                   help="Cache all target SE gene tokens in one fixed train-control context; preserve all targets within the 2047 gene budget")
    a = p.parse_args()
    if a.output_dir.exists() and any(a.output_dir.iterdir()):
        raise FileExistsError(a.output_dir)
    a.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.manual_seed(42)
    meta = h5_metadata(a.h5ad)
    split = json.loads(a.split_manifest.read_text())
    if split["split_mode"] != "official_additive":
        raise ValueError("expected official_additive")
    controls = json.loads(a.control_split.read_text())
    rows = np.asarray(controls["train_control_rows"], dtype=np.int64)
    test_rows = np.asarray(controls["test_control_rows"], dtype=np.int64)
    if len(np.unique(rows)) != len(rows) or not len(rows) or not meta["control"][rows].all():
        raise ValueError("invalid training controls")
    if np.intersect1d(rows, test_rows).size or set(rows.tolist()) | set(test_rows.tolist()) != set(np.flatnonzero(meta["control"]).tolist()):
        raise ValueError("control split is not a disjoint partition")
    targets = sorted(set(g for c in split["train_conditions"] for g in parse_condition_genes(c)))
    if len(targets) != 101:
        raise ValueError("expected all 101 perturbation targets in train")
    protein = torch.load(a.esm_checkpoint, map_location="cpu", weights_only=False)
    symbols = list(protein)
    lookup = {g: i for i, g in enumerate(symbols)}
    missing = [g for g in targets if g not in lookup]
    if missing:
        raise ValueError(f"Missing target ESM embeddings: {missing}")
    target_esm = torch.stack([protein[g].float().reshape(-1) for g in targets])
    if target_esm.shape != (101, 5120) or not torch.isfinite(target_esm).all():
        raise ValueError("invalid target ESM")
    with h5py.File(a.h5ad, "r") as f:
        x = f["X"]
        if x.attrs.get("encoding-type") != "csr_matrix":
            raise ValueError("expected CSR X")
        indptr = x["indptr"][...]
        profile = np.zeros(meta["shape"][1], dtype=np.float64)
        for row in rows:
            first, last = indptr[row:row+2]
            np.add.at(profile, x["indices"][first:last], x["data"][first:last])
        profile /= len(rows)
    extract_path = Path(__file__).resolve().parents[3] / "combosciplex/extract_se.py"
    spec = importlib.util.spec_from_file_location("norman_original_se", extract_path)
    ext = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ext)
    mapping = np.array([lookup.get(g, -1) for g in meta["source_genes"]], dtype=np.int32)
    tokenizer = ext.Tokenizer(mapping, len(symbols), seed=42)
    if a.target_se_only:
        _extract_target_se(a, meta, rows, targets, target_esm, symbols, lookup,
                           tokenizer, ext, profile)
        return
    ids, weights = tokenizer.encode(np.arange(len(profile)), np.expm1(profile))
    background_genes = [symbols[int(i)] for i in ids[1:]]
    esm = torch.stack([protein[g].float().reshape(-1) for g in background_genes])[None]
    del protein
    device = torch.device(a.device)
    model = ext.load_model(a.se_checkpoint, None, device)
    ids_t = torch.from_numpy(ids[None].copy()).long().to(device)
    counts = torch.from_numpy(weights[None].copy()).float().to(device)
    with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
        src = F.normalize(model.pe_embedding(ids_t), dim=2)
        src = torch.cat((model.cls_token.expand(1, 1, -1), src[:, 1:, :]), dim=1)
        if model.dataset_token is not None:
            src = torch.cat((src, model.dataset_token.expand(1, 1, -1)), dim=1)
        gene_output, cell_state, _ = model.forward(src=src, counts=counts)
        se_tokens = gene_output[:, 1:2048].float().clone()
    if se_tokens.shape != (1, 2047, 2048) or not torch.isfinite(se_tokens).all():
        raise ValueError("invalid gene-level SE token shape")
    del model, src, gene_output
    if device.type == "cuda":
        torch.cuda.empty_cache()
    projector = FrozenMAPGeneProjector(a.projector_checkpoint, projector_kind="map_predictor").to(device)
    with torch.inference_mode():
        projected = projector.forward_background(se_tokens, esm.to(device)).float().cpu()
    if projected.shape != (1, 2047, 1024) or not torch.isfinite(projected).all():
        raise ValueError("invalid projected background")
    record = {
        "format_version": 1, "complete": True, "created_unix": time.time(),
        "host_alias": os.environ.get("MAPFLOW_HOST_ALIAS"), "hostname": socket.gethostname(),
        "code_root": str(ROOT), "git_commit": _repository_commit(), "source_git_commit": '3786dd6a8ce03aca996ab5d7b5dc10bc85689215',
        "args": {k: str(v) for k, v in vars(a).items()},
        "h5ad_sha256": sha(a.h5ad), "cell_order_sha256": ordered(meta["cell_ids"]),
        "control_split_sha256": sha(a.control_split),
        "train_control_rows": rows.tolist(), "train_control_ids": [meta["cell_ids"][i] for i in rows],
        "train_control_rows_sha256": ordered(rows), "n_train_controls": len(rows),
        "target_genes": targets, "background_genes": background_genes,
        "background_gene_order_sha256": ordered(background_genes),
        "projector_kind": "map_predictor", "projector_checkpoint_sha256": sha(a.projector_checkpoint),
        "esm_checkpoint_sha256": sha(a.esm_checkpoint), "se_checkpoint_sha256": sha(a.se_checkpoint),
        "shapes": {"target_esm": list(target_esm.shape), "background_projected": list(projected.shape), "se_gene_tokens": list(se_tokens.shape)},
        "normalization": "mean of stored lognormalized X over train controls; expm1 interface conversion then original tokenizer log1p, weights sum-to-100 over mapped genes",
        "background_kind": "train_control_pool_mean", "se_tokens": "gene_output[:,1:2048], excluding CLS and dataset token",
        "background_scope": "shared only when train_control_rows and source data match; independent of test conditions",
    }
    output = a.output_dir / "gene_condition_assets.pt"
    torch.save({"target_genes": targets, "target_esm": target_esm,
                "background_projected": projected, "metadata": record}, output)
    np.save(a.output_dir / "background_token_ids.npy", ids)
    np.save(a.output_dir / "background_token_counts.npy", weights)
    record["asset_sha256"] = sha(output)
    (a.output_dir / "gene_condition_assets.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({k: v for k, v in record.items() if k not in ("train_control_rows", "train_control_ids", "target_genes", "background_genes")}, indent=2))

if __name__ == "__main__":
    main()

