"""Restore model-independent Adam and RNG state without resetting moments."""
from pathlib import Path
import random
import numpy as np
import torch
from .artifacts import atomic_json, sha256


def capture_rng(rng, generator):
    return dict(numpy=rng.bit_generator.state, torch_cpu=torch.get_rng_state(),
                torch_cuda=torch.cuda.get_rng_state(), generator=generator.get_state(),
                python=random.getstate())


def restore_rng(state, rng, generator):
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


def save_checkpoint(path, model, optimizer, optimizer_names, step, config,
                    runtime, rng, generator, rank, world, parent):
    import torch.distributed as dist
    local = capture_rng(rng, generator)
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
