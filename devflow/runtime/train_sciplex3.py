"""Checkpoint-compatible distributed continuation with explicit optimizer groups."""
from pathlib import Path
import json
import os
import random
import time
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from devflow.data.sciplex3 import SciPlexData
from devflow.model_attention import SciPlex3Model
from .sciplex3 import LoraRuntimeData
from .artifacts import atomic_json, append_csv, sha256
from .checkpoint import capture_rng, restore_rng, grouped_adam, save_checkpoint, validate_checkpoint_runtime
from .diagnostics import tracked_parameters, gradient_snapshot, route_probe


def make_runtime(config, device):
    data = config["data"]
    dataset = SciPlexData(data["base_dir"], data["sampling_manifest"],
                         data.get("source_split_root"), data.get("training_statistics"))
    return LoraRuntimeData(data["feature_cache"], data["prefix_cache"], dataset, device)


def train(config):
    settings = config["training"]
    seed = int(settings.get("seed", 42))
    rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    local = int(os.environ.get("LOCAL_RANK", 0))
    if world != 8:
        raise ValueError("This reproduction keeps 8 ranks and 32 conditions / 256 pairs per update")
    torch.cuda.set_device(local)
    device = torch.device("cuda", local)
    dist.init_process_group("nccl")
    torch.set_num_threads(2)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed + rank)
    random.seed(seed + rank)
    torch.backends.cuda.matmul.allow_tf32 = bool(settings.get("allow_tf32", True))
    if settings.get("pairs_per_condition", 8) != 8 or settings.get("conditions_per_rank", 4) != 4:
        raise ValueError("Initial continuation trials retain four conditions × eight pairs per rank")
    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    if (output / "train_metrics.csv").exists():
        raise FileExistsError("Use a new output directory for every continuation")
    checkpoint_path = Path(settings["checkpoint"]) if settings.get("checkpoint") else None
    runtime = make_runtime(config, device)
    if checkpoint_path is None:
        decoder_path = config["model"].get("decoder_checkpoint")
        if not decoder_path:
            raise ValueError("Fresh training needs an independently pretrained decoder_checkpoint")
        model_config = config["model"].get("config", dict(activation_checkpointing=True, gene_projection_chunk_size=256))
        model = SciPlex3Model(model_config, config["model"]["tail_checkpoint"], decoder_path,
            runtime.data.gene_order_sha256, loss_weights=config["model"].get("loss_weights")).to(device)
        runtime.initialization_hashes = dict(tail=sha256(config["model"]["tail_checkpoint"]), decoder=sha256(decoder_path))
        initial_optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=float(settings["learning_rates"]["flow"]))
        parent = dict(update=0, world_size=world, optimizer=initial_optimizer.state_dict(),
            rng_states=[capture_rng(np.random.default_rng(seed+rank*100003), torch.Generator(device=device).manual_seed(seed+rank*100003))]*world,
            cache_manifest_sha256=runtime.cache_manifest_sha256,
            prefix_manifest_sha256=runtime.prefix_manifest_sha256,
            initialization_hashes=runtime.initialization_hashes,training_binding=runtime.data.training_binding(),
            config=dict(prefix_cache=config["data"]["prefix_cache"],tail_checkpoint=config["model"]["tail_checkpoint"],decoder_checkpoint=decoder_path))
        del initial_optimizer
    else:
        parent = torch.load(str(checkpoint_path), map_location="cpu", mmap=True, weights_only=False)
        if parent["world_size"] != world:
            raise ValueError("World size changed from the checkpoint")
        runtime.initialization_hashes = parent["initialization_hashes"]
        validate_checkpoint_runtime(parent, runtime, config["model"]["tail_checkpoint"])
        model = SciPlex3Model.from_checkpoint(checkpoint_path,
            tail_checkpoint=config["model"]["tail_checkpoint"],
            loss_weights=config["model"].get("loss_weights")).to(device)
    model.bind_control_prefixes(runtime.prefixes)
    rates = settings["learning_rates"]
    freeze = bool(settings.get("freeze_lora", False))
    optimizer, optimizer_names, restored = grouped_adam(model, parent, rates, freeze)
    rng = np.random.default_rng(seed + rank * 100003)
    generator = torch.Generator(device=device).manual_seed(seed + rank * 100003)
    restore_rng(parent["rng_states"][rank], rng, generator)
    start = int(parent["update"])
    final = start + int(settings["additional_updates"])
    if final <= start:
        raise ValueError("additional_updates must be positive")
    if rank == 0:
        atomic_json(output / "run_config.json", {**config, "start_update": start,
            "final_update": final, "world_size": world,
            "train_conditions": len(runtime.data.train_records),
            "conditions_per_update": 32, "effective_pairs_per_update": 256,
            "loss_weights": model.loss_weights.to_dict(), "expression_gradient_to_flow": False,
            "parent_checkpoint_sha256": sha256(checkpoint_path) if checkpoint_path else None,
            "initialization_mode": "continuation" if checkpoint_path else "scratch_flow_with_pretrained_decoder",
            "selection_protocol": "predeclared fixed pilots; external online monitoring is exploratory"})
        atomic_json(output / ("ADAM_RESTORE_VERIFIED.json" if checkpoint_path else "OPTIMIZER_INITIALIZED.json"), {"status": "passed",
            "parameters": restored, "optimizer_parameter_names": optimizer_names,
            "freeze_lora": freeze, "learning_rates": rates})
    records = runtime.data.train_records

    def condition_batch(step):
        selected = []
        for condition in range(4):
            ordinal = ((step - 1) * 4 + condition) * world + rank
            epoch, offset = divmod(ordinal, len(records))
            permutation = np.random.default_rng(seed + epoch).permutation(len(records))
            selected.append(records[int(permutation[offset])])
        return runtime.training_group(selected, rng, generator, model,
                    pairs=8, candidates=int(settings.get("ot_candidates", 256)))

    # Probe before installing DDP hooks. Restore RNG so it does not consume the
    # first actual update's dropout, time, noise, OT or cell sampling draws.
    before_probe = capture_rng(rng, generator)
    model.train()
    probe_batch, _ = condition_batch(start + 1)
    route = route_probe(model, probe_batch, runtime.esm, int(settings.get("time_points", 10)))
    restore_rng(before_probe, rng, generator)
    del probe_batch
    if rank == 0:
        atomic_json(output / "GRADIENT_ROUTE_VERIFIED.json", route)
    wrapped = DistributedDataParallel(model, device_ids=[local])
    tracked = tracked_parameters(model)
    elapsed_start = time.monotonic()
    initial_lora = {n: p.detach().clone() for n, p in model.control_lora.named_parameters()
                    if n.endswith("lora_A") or n.endswith("lora_B")}
    try:
        for step in range(start + 1, final + 1):
            model.train()
            schedule=settings.get("schedule",{})
            factor=1.0
            if schedule.get("type")=="cosine":
                import math
                progress=(step-start-float(schedule.get("start",500)))/max(1.,float(schedule.get("end",13000))-float(schedule.get("start",500)))
                fraction=min(1.,max(0.,progress))
                minimum=float(schedule.get("min_factor",0.1))
                factor=minimum+(1.-minimum)*(1.+math.cos(math.pi*fraction))/2.
            for group in optimizer.param_groups:group["lr"]=float(rates[group["name"]])*factor
            optimizer.zero_grad(set_to_none=True)
            batch, statistics = condition_batch(step)
            first = step == start + 1
            before = {name: p.detach().clone() for name, p in tracked.items()} if first else {}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss, parts = wrapped(batch, runtime.esm, int(settings.get("time_points", 10)))
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite loss at {step}, rank {rank}")
            loss.backward()  # Deliberately outside autocast.
            before_clip = gradient_snapshot(tracked) if first else None
            flow_norm = torch.nn.utils.clip_grad_norm_(model.flow.parameters(),
                float(settings.get("flow_clip", 1.0)), error_if_nonfinite=True)
            decoder_norm = torch.nn.utils.clip_grad_norm_(model.decoder.parameters(),
                float(settings.get("decoder_clip", 1.0)), error_if_nonfinite=True)
            after_clip = gradient_snapshot(tracked) if first else None
            optimizer.step()
            if first:
                changes = {name: {"changed_elements": int(torch.count_nonzero(p.detach() - before[name])),
                    "max_abs_change": float((p.detach() - before[name]).abs().max()),
                    "before_clip": before_clip[name], "after_clip": after_clip[name]}
                    for name, p in tracked.items()}
                lora_changes = [v["changed_elements"] for n, v in changes.items() if "control_lora" in n]
                if freeze and any(lora_changes):
                    raise RuntimeError("Frozen LoRA changed")
                if not freeze and any(v["changed_elements"] == 0 for n,v in changes.items() if "control_lora" in n and (start > 0 or n.endswith("lora_B"))):
                    raise RuntimeError("An enabled LoRA matrix did not update")
                all_passed = torch.ones(1, device=device)
                dist.all_reduce(all_passed, op=dist.ReduceOp.MIN)
                if rank == 0:
                    atomic_json(output / "FIRST_UPDATE_VERIFIED.json", {"status": "passed",
                        "update": step, "all_ranks_passed": bool(all_passed.item()),
                        "freeze_lora": freeze, "parameters": changes,
                        "optimizer_groups": [{"name": g["name"], "lr": g["lr"],
                            "steps": sorted(set(float(optimizer.state[p]["step"]) for p in g["params"]
                                                if "step" in optimizer.state[p]))}
                            for g in optimizer.param_groups]})
            metrics = {key: float(value) for key, value in parts.items()}
            metrics.update(ot_marginal_error=sum(x.marginal_error for x in statistics) / len(statistics),
                           ot_converged_fraction=sum(float(x.converged) for x in statistics) / len(statistics),
                           gradient_norm_before_clip=float(flow_norm), decoder_gradient_norm_before_clip=float(decoder_norm))
            values = torch.tensor(list(metrics.values()), device=device, dtype=torch.float64)
            dist.all_reduce(values)
            values /= world
            metrics = dict(zip(metrics, values.cpu().tolist()))
            if rank == 0:
                elapsed = time.monotonic() - elapsed_start
                row = dict(update=step, optimizer_step=step, continuation_update=step-start,
                    micro_step=step*4, condition_passes=step*32/len(records), elapsed_seconds=elapsed,
                    seconds_per_update=elapsed/(step-start), learning_rate_flow=float(rates["flow"])*factor,
                    learning_rate_lora=0.0 if freeze else float(rates["lora"])*factor,
                    learning_rate_decoder=float(rates["decoder"])*factor, time_points=int(settings.get("time_points", 10)),
                    **metrics)
                append_csv(output / "train_metrics.csv", row)
                atomic_json(output / "status.json", {"status": "training", **row})
                if (step-start) % int(settings.get("log_every", 10)) == 0 or first:
                    print(json.dumps(row), flush=True)
            if (step-start) % int(settings.get("checkpoint_every", 250)) == 0 or step == final:
                save_checkpoint(output / f"update_{step:06d}.pt", model, optimizer,
                    optimizer_names, step, config, runtime, rng, generator, rank, world, parent)
            every = int(settings.get("eval_every", 500))
            if every and ((step-start) % every == 0 or step == final):
                from .evaluate_sciplex3 import evaluate_online
                evaluate_online(model, runtime, config, step, rank, world)
                dist.barrier()
        if freeze:
            for name, initial in initial_lora.items():
                if not torch.equal(dict(model.control_lora.named_parameters())[name], initial):
                    raise RuntimeError(f"Frozen LoRA changed after training: {name}")
        if rank == 0:
            atomic_json(output / "TRAINING_COMPLETE.json", {"status": "complete", "updates": final,
                "additional_updates": final-start, "checkpoint": str(output / f"update_{final:06d}.pt")})
            atomic_json(output / "status.json", {"status": "complete", "update": final})
    except BaseException as exc:
        if rank == 0:
            atomic_json(output / "FAILED.json", {"status": "failed", "error": repr(exc)})
        raise
    finally:
        dist.destroy_process_group()
