"""Complete, atomic checkpoints for the MLP and trainable mean model.

One checkpoint contains all three modules, the fixed mean features/statistics,
the three optimizers, and every rank's sampling and random-number states.
"""
from pathlib import Path
import hashlib
import json
import random
import subprocess

import numpy as np
import torch
import torch.distributed as dist

SCHEMA = "devflow_mlp_mean_training_v1"


def source_identity(root=None):
    root = Path(root or Path(__file__).resolve().parents[2]).resolve()
    files = {}
    for folder in ("devflow", "benchmark", "recipes"):
        for path in sorted((root / folder).rglob("*.py")):
            files[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    for name in ("pyproject.toml",):
        if (root / name).exists():
            files[name] = hashlib.sha256((root / name).read_bytes()).hexdigest()
    result = subprocess.run(["git", "rev-parse", "--verify", "HEAD"], cwd=root,
                            capture_output=True, text=True)
    return {"git_commit": result.stdout.strip() if result.returncode == 0 else None,
            "source_tree_sha256": hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest(),
            "files": files}


def snapshot_source(output, identity, root=None):
    """Keep the actual code even before the user's first Git commit exists."""
    root = Path(root or Path(__file__).resolve().parents[2]).resolve()
    target = Path(output) / "source_snapshot"
    target.mkdir(parents=True, exist_ok=True)
    for name, expected in identity["files"].items():
        content = (root / name).read_bytes()
        if hashlib.sha256(content).hexdigest() != expected:
            raise RuntimeError("Source changed while preparing the experiment snapshot: " + name)
        path = target / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and path.read_bytes() != content:
            raise FileExistsError("Source snapshot is immutable: " + str(path))
        path.write_bytes(content)
    (target / "manifest.json").write_text(json.dumps(identity, indent=2) + "\n")


def capture_rng(candidate_rng, pair_generator, decoder_rng):
    return {"candidate_numpy": candidate_rng.bit_generator.state,
            "decoder_numpy": decoder_rng.bit_generator.state,
            "pair_generator": pair_generator.get_state().cpu(),
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
            "numpy_global": np.random.get_state(), "python": random.getstate()}


def restore_rng(state, candidate_rng, pair_generator, decoder_rng):
    candidate_rng.bit_generator.state = state["candidate_numpy"]
    decoder_rng.bit_generator.state = state["decoder_numpy"]
    pair_generator.set_state(state["pair_generator"].cpu())
    torch.set_rng_state(state["torch_cpu"].cpu())
    if state.get("torch_cuda") is not None:
        torch.cuda.set_rng_state(state["torch_cuda"].cpu())
    np.random.set_state(state["numpy_global"])
    random.setstate(state["python"])


def make_optimizers(model, rates):
    modules = {"flow": model.flow, "decoder": model.decoder, "head": model.mean_head}
    optimizers, names = {}, {}
    for group, module in modules.items():
        parameters = [(name, p) for name, p in module.named_parameters() if p.requires_grad]
        if not parameters:
            raise ValueError("No trainable parameters in " + group)
        names[group] = [name for name, _ in parameters]
        # In particular A already has its explicit RKHS regularizer: no decay.
        optimizers[group] = torch.optim.Adam([p for _, p in parameters],
                                           lr=float(rates[group]), weight_decay=0.0)
    return optimizers, names


def validate_resume(payload, runtime, config, world):
    if payload.get("resume_allowed") is False or payload.get("purpose") == "historical_replay_only":
        raise ValueError("Historical replay snapshots cannot initialize or resume new training")
    if payload.get("schema") != SCHEMA:
        raise ValueError("A fresh MLP experiment cannot resume an attention/legacy checkpoint")
    saved_source = payload.get("source_identity", {}).get("source_tree_sha256")
    actual_source = source_identity()["source_tree_sha256"]
    if saved_source != actual_source and not config["training"].get("allow_source_change", False):
        raise ValueError("Exact-resume source identity differs; an explicit source-change continuation is required")
    if int(payload["world_size"]) != world:
        raise ValueError("Exact resume requires the checkpoint world size")
    if payload["training_binding"] != runtime.data.training_binding():
        raise ValueError("Checkpoint training rows or gene axis differ")
    if payload["asset_manifest_sha256"] != runtime.asset_manifest_sha256:
        raise ValueError("Checkpoint fixed features/mean statistics differ")
    previous = payload["config"]
    for key in ("pairs_per_condition", "ot_candidates", "gradient_accumulation",
                "decoder_cells", "time_points", "seed", "precision", "allow_tf32",
                "learning_rates", "schedule", "gradient_clip"):
        if previous["training"].get(key) != config["training"].get(key):
            raise ValueError("Exact-resume training setting changed: " + key)
    # Initialisation files are not read on resume. Physical paths may move;
    # the actual pretrained identity is retained in initialization_provenance.
    previous_model = {k: v for k, v in previous["model"].items() if k != "decoder_checkpoint"}
    current_model = {k: v for k, v in config["model"].items() if k != "decoder_checkpoint"}
    if previous_model != current_model:
        raise ValueError("Exact-resume model/loss/initialization configuration changed")


def save_checkpoint(path, model, optimizers, optimizer_names, step, config, runtime,
                    candidate_rng, pair_generator, decoder_rng, rank, world, source):
    local = capture_rng(candidate_rng, pair_generator, decoder_rng)
    states = [None] * world
    if world > 1:
        dist.all_gather_object(states, local)
    else:
        states[0] = local
    path = Path(path)
    if rank == 0:
        if path.exists():
            raise FileExistsError("Immutable checkpoint already exists: " + str(path))
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"schema": SCHEMA, "model": model.state_dict(),
                   "model_spec": model.export_spec(), "update": int(step),
                   "optimizer_step": int(step), "world_size": int(world),
                   "optimizers": {name: opt.state_dict() for name, opt in optimizers.items()},
                   "optimizer_parameter_names": optimizer_names,
                   "rng_states": states, "config": config,
                   "training_binding": runtime.data.training_binding(),
                   "asset_manifest_sha256": runtime.asset_manifest_sha256,
                   "initialization_provenance": getattr(runtime, "initialization_provenance", {}),
                   "source_identity": source}
        temporary = path.with_name(path.name + ".tmp")
        torch.save(payload, temporary)
        temporary.replace(path)
    if world > 1:
        dist.barrier()
    return path


def restore_optimizers(payload, optimizers, names):
    if payload["optimizer_parameter_names"] != names:
        raise ValueError("Optimizer parameter identities/order differ")
    for group, optimizer in optimizers.items():
        optimizer.load_state_dict(payload["optimizers"][group])


def checkpoint_path(output, step):
    return Path(output) / "checkpoints" / f"step_{int(step):06d}.pt"
