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


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("h5ad", "split-manifest", "control-split", "esm-checkpoint",
                 "se-checkpoint", "projector-checkpoint", "output-dir"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
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

