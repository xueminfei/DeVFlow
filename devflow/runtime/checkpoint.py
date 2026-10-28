"""Checkpoint persistence for MLP/mean training and historical attention runs.

MLP checkpoints contain Flow, decoder, mean head, fixed statistics, three
optimizers and every rank's random-number states. Attention checkpoints retain
their original schema and explicitly named attention APIs for compatibility.
"""
from pathlib import Path
import hashlib
import json
import random
import subprocess

import numpy as np
import torch
import torch.distributed as dist

from .artifacts import sha256

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


def capture_attention_rng(rng, generator):
    return dict(numpy=rng.bit_generator.state, torch_cpu=torch.get_rng_state(),
                torch_cuda=torch.cuda.get_rng_state(), generator=generator.get_state(),
                python=random.getstate())



def restore_attention_rng(state, rng, generator):
    rng.bit_generator.state = state["numpy"]
    torch.set_rng_state(state["torch_cpu"])
    torch.cuda.set_rng_state(state["torch_cuda"])
    generator.set_state(state["generator"])
    random.setstate(state["python"])



def grouped_adam(model, checkpoint, learning_rates, freeze_lora=False):
    """Split the original single Adam group using its exact parameter order.

    Parameter names are persisted for future resumes. Frozen LoRA is removed
    from both optimization and clipping while its weights remain in the model.
    """
    named = dict(model.named_parameters())
    original_names = [name for name, p in named.items() if p.requires_grad]
    old = checkpoint["optimizer"]
    names = checkpoint.get("optimizer_parameter_names")
    if names is None:
        if len(old["param_groups"]) != 1:
            raise ValueError("A grouped checkpoint must record optimizer_parameter_names")
        names = [original_names]
    if len(names) != len(old["param_groups"]):
        raise ValueError("Optimizer group/name counts differ")
    state_by_name = {}
    for group, group_names in zip(old["param_groups"], names):
        if len(group["params"]) != len(group_names):
            raise ValueError("Optimizer parameter/name counts differ")
        for identity, name in zip(group["params"], group_names):
            if name not in named or name in state_by_name:
                raise ValueError(f"Invalid optimizer parameter identity: {name}")
            state_by_name[name] = old["state"].get(identity, {})
    for name,state in checkpoint.get("optimizer_frozen_states",{}).items():
        if name not in state_by_name:state_by_name[name]=state
    model.control_lora.requires_grad_(not freeze_lora)
    # The SE base must remain frozen even when LoRA is active.
    for name, parameter in model.control_lora.named_parameters():
        parameter.requires_grad_(not freeze_lora and (name.endswith("lora_A") or name.endswith("lora_B")))
    groups = {"flow": [], "lora": [], "decoder": []}
    group_names = {key: [] for key in groups}
    for name, parameter in named.items():
        if not parameter.requires_grad:
            continue
        key = "lora" if name.startswith("flow.control_lora.") else "decoder" if name.startswith("decoder.") else "flow"
        if name not in state_by_name:
            raise ValueError(f"No saved Adam identity for trainable parameter {name}")
        groups[key].append(parameter)
        group_names[key].append(name)
    first = old["param_groups"][0]
    options = {key: first[key] for key in ("betas", "eps", "weight_decay", "amsgrad") if key in first}
    for group in old["param_groups"][1:]:
        if any(group.get(key) != value for key, value in options.items()):
            raise ValueError("Cannot merge Adam groups with different optimizer settings")
    optimizer = torch.optim.Adam([
        {"params": parameters, "name": key, "lr": float(learning_rates[key])}
        for key, parameters in groups.items() if parameters
    ], **options)
    restored = []
    for key, parameters in groups.items():
        for name, parameter in zip(group_names[key], parameters):
            state = state_by_name[name]
            copied = {}
            for field, value in state.items():
                if isinstance(value, torch.Tensor):
                    if field != "step" and value.shape != parameter.shape:
                        raise ValueError(f"Adam moment shape mismatch for {name}/{field}")
                    copied[field] = value.clone() if field == "step" else value.to(parameter.device).clone()
                else:
                    copied[field] = value
            optimizer.state[parameter] = copied
            restored.append({"parameter": name, "group": key,
                             "step": float(copied["step"]) if "step" in copied else None,
                             "moments_restored": all(x in copied for x in ("exp_avg", "exp_avg_sq"))})
    ordered_names = [group_names[group["name"]] for group in optimizer.param_groups]
    return optimizer, ordered_names, restored



def save_attention_checkpoint(path, model, optimizer, optimizer_names, step, config,
                    runtime, rng, generator, rank, world, parent):
    import torch.distributed as dist
    local = capture_attention_rng(rng, generator)
    states = [None] * world
    if world > 1:
        dist.all_gather_object(states, local)
    else:
        states[0] = local
    if rank == 0:
        old_names=parent.get("optimizer_parameter_names")
        if old_names is None:
            old_names=[[n for n,p in model.named_parameters() if p.requires_grad or
                        (n.startswith("flow.control_lora.") and (n.endswith("lora_A") or n.endswith("lora_B")))]]
        frozen_states=dict(parent.get("optimizer_frozen_states",{}))
        active_names={n for group in optimizer_names for n in group}
        for group,names in zip(parent["optimizer"]["param_groups"],old_names):
            if len(group["params"])!=len(names):raise ValueError("Cannot identify saved frozen optimizer states")
            for identity,name in zip(group["params"],names):
                if name not in active_names:frozen_states[name]=parent["optimizer"]["state"].get(identity,{})
        frozen_states={n:v for n,v in frozen_states.items() if n not in active_names}
        payload = dict(schema="scratch_attention_cell_lora_training_v1", model=model.state_dict(),
                       optimizer=optimizer.state_dict(), optimizer_parameter_names=optimizer_names,
                       optimizer_frozen_states=frozen_states,
                       update=step, world_size=world, rng_states=states,
                       model_config=model.model_config, loss_weights=model.loss_weights.to_dict(),
                       config=parent["config"], tuning_config=config,
                       cache_manifest_sha256=runtime.cache_manifest_sha256,
                       prefix_manifest_sha256=runtime.prefix_manifest_sha256,
                       initialization_hashes=runtime.initialization_hashes,
                       training_binding=runtime.data.training_binding())
        path = Path(path)
        temporary = path.with_name(path.name + ".tmp")
        torch.save(payload, temporary)
        temporary.replace(path)
    if world > 1:
        dist.barrier()



def validate_checkpoint_runtime(checkpoint, runtime, tail_path):
    """Validate every immutable input identity before training or prediction."""
    if checkpoint.get("schema") != "scratch_attention_cell_lora_training_v1":
        raise ValueError("Unsupported SciPlex3 checkpoint schema")
    checks={"cache_manifest_sha256":runtime.cache_manifest_sha256,
            "prefix_manifest_sha256":runtime.prefix_manifest_sha256,
            "training_binding":runtime.data.training_binding()}
    for key,actual in checks.items():
        if checkpoint.get(key)!=actual:
            raise ValueError("Checkpoint runtime identity differs: "+key)
    tail_hash=sha256(tail_path)
    if checkpoint["initialization_hashes"]["tail"]!=tail_hash or runtime.prefix_manifest.get("tail_checkpoint_sha256")!=tail_hash:
        raise ValueError("Checkpoint SE tail and prefix initialization differ")
    return {"status":"passed","training_binding":checks["training_binding"],
            "tail_sha256":tail_hash,"cache_manifest_sha256":checks["cache_manifest_sha256"],
            "prefix_manifest_sha256":checks["prefix_manifest_sha256"]}

