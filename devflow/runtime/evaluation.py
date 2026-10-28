"""Shared evaluation mechanics; dataset protocols stay in their evaluators.

These helpers preserve model/RNG state, verify saved predictions and weights,
and aggregate condition metrics with equal sampling-repeat weights.
"""
from contextlib import contextmanager
import json
from pathlib import Path
import random

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist

from benchmark.metrics import HVG_METRICS
from .artifacts import sha256


def _clean(value):
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_clean(v) for v in value]
    if isinstance(value, np.generic):
        return _clean(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


@contextmanager
def formal_mode(model):
    """Do not change the training RNG, dropout mode, or TF32 policy."""
    modes = [(module, module.training) for module in model.modules()]
    cpu_rng, numpy_rng, python_rng = torch.get_rng_state(), np.random.get_state(), random.getstate()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    matmul, cudnn = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        model.eval()
        with torch.no_grad():
            yield
    finally:
        for module, training in modes:
            module.training = training
        torch.backends.cuda.matmul.allow_tf32 = matmul
        torch.backends.cudnn.allow_tf32 = cudnn
        torch.set_rng_state(cpu_rng)
        np.random.set_state(numpy_rng)
        random.setstate(python_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)


def _checkpoint_identity(model, checkpoint_path, step, runtime=None):
    path = Path(checkpoint_path)
    payload = torch.load(str(path), map_location="cpu", mmap=True, weights_only=False)
    if int(payload["update"]) != int(step):
        raise ValueError("Requested evaluation step differs from the actual checkpoint")
    if runtime is not None:
        if payload.get("training_binding") != runtime.data.training_binding():
            raise ValueError("Checkpoint training binding differs from the evaluation data")
        if payload.get("asset_manifest_sha256") != getattr(runtime, "asset_manifest_sha256", None):
            raise ValueError("Checkpoint feature/mean assets differ from the evaluation runtime")
    saved, current = payload["model"], model.state_dict()
    if set(saved) != set(current):
        raise ValueError("In-memory model and checkpoint have different state keys")
    for name, value in current.items():
        if value.dtype != saved[name].dtype or value.shape != saved[name].shape or not torch.equal(value.detach().cpu(), saved[name].detach().cpu()):
            raise ValueError("In-memory model differs from the saved checkpoint: " + name)
    return dict(checkpoint=str(path.resolve()), checkpoint_sha256=sha256(path),
                checkpoint_update=int(step), model_matches_checkpoint=True,
                checkpoint_purpose=payload.get("purpose", "training_checkpoint"),
                checkpoint_run_purpose=payload.get("config", {}).get("purpose"))


def _metric_columns(frame):
    return [key for key in frame.columns if key in HVG_METRICS or key.startswith("de50_")]


def aggregate_rows(rows):
    """Equal condition weights, then equal sampling-repeat weights (ddof=1)."""
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError("Cannot aggregate empty evaluation rows")
    metrics = _metric_columns(frame)
    by_seed = frame.groupby("seed", sort=True)[metrics].mean()
    result = dict(metrics={}, metrics_std={}, n_finite={}, n_sampling_repeats=len(by_seed))
    for name in metrics:
        values = by_seed[name].to_numpy(dtype=np.float64)
        values = values[np.isfinite(values)]
        result["metrics"][name] = float(values.mean()) if len(values) else None
        result["metrics_std"][name] = float(values.std(ddof=1)) if len(values) > 1 else None
        result["n_finite"][name] = len(values)
    return _clean(result), by_seed.reset_index()


def _cache_ok(folder, identity):
    marker = Path(folder) / "COMPLETED.json"
    if not marker.exists():
        return False
    saved = json.loads(marker.read_text())
    if saved["identity"] != identity:
        raise ValueError("Existing evaluation belongs to a different model or protocol: " + str(folder))
    for name, digest in saved["files_sha256"].items():
        if sha256(Path(folder) / name) != digest:
            raise ValueError("Completed evaluation output changed: " + name)
    return True


def _rank_zero_value(action, rank, world):
    """Broadcast failures too, so other ranks do not wait after a rank-zero error."""
    packet = None
    if rank == 0:
        try:
            packet = dict(value=action(), error=None)
        except Exception as exc:
            packet = dict(value=None, error=f"{type(exc).__name__}: {exc}")
    if world > 1:
        values = [packet]
        dist.broadcast_object_list(values, src=0)
        packet = values[0]
    if packet["error"] is not None:
        raise ValueError(packet["error"])
    return packet["value"]

