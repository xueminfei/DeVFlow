"""Basic MLP Flow, trainable kernel mean, and decoder distribution training.

Flow uses OT-paired latent losses. The mean head uses its training-only KRR
objective. The decoder uses independently sampled cells and final-expression
MMD. Endpoint and mean detaches keep these three gradient routes separate.
"""
from contextlib import nullcontext
from pathlib import Path
import json
import math
import os
import random
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from .artifacts import append_csv, atomic_json, sha256
from .checkpoint import (capture_rng, restore_rng, source_identity, snapshot_source,
    make_optimizers, validate_resume, restore_optimizers, save_checkpoint, checkpoint_path)


def task_components(config):
    """Share the training loop while keeping each dataset's native protocol."""
    if config["task"] == "sciplex3":
        from devflow.model_mlp import MLPMeanSystem
        from .sciplex3_mlp import make_runtime
        from .evaluate_sciplex3_mlp import evaluate
        return make_runtime, MLPMeanSystem, evaluate
    if config["task"] == "combosciplex":
        from devflow.model_mlp import MLPMeanSystem
        from .combosciplex_mlp import make_runtime
        from .evaluate_combosciplex_mlp import evaluate
        return make_runtime, MLPMeanSystem, evaluate
    raise ValueError("No mean/MMD training adapter for " + str(config["task"]))


def condition_index(step, micro, accumulation, count, seed):
    """One shared condition per microstep; a deterministic permutation per pass."""
    ordinal = (int(step) - 1) * accumulation + micro
    epoch, offset = divmod(ordinal, count)
    return int(np.random.default_rng(seed + epoch).permutation(count)[offset])


def learning_rate_factor(step, schedule):
    if schedule.get("type", "constant") == "constant":
        return 1.0
    if schedule["type"] != "cosine":
        raise ValueError("Unknown learning-rate schedule")
    start, end = float(schedule.get("start", 500)), float(schedule.get("end", 13000))
    fraction = min(1.0, max(0.0, (step - start) / max(1.0, end - start)))
    floor = float(schedule.get("min_factor", 0.1))
    return floor + (1.0 - floor) * (1.0 + math.cos(math.pi * fraction)) / 2.0


def _loss_weights(config):
    return {"flow_residual": 1.0, "flow_mean": 32.0, "decoder_mmd": 1.0,
            "head": 1.0, **config["model"].get("loss_weights", {})}


def refresh_monitor(output):
    """Plotting failures must be visible without terminating parameter updates."""
    try:
        from .monitor import build_monitor
        return build_monitor(output)
    except Exception as exc:
        atomic_json(Path(output) / "MONITOR_REFRESH_ERROR.json",
                    {"status": "monitor_error", "error": repr(exc), "time_unix": time.time()})
        print("Monitor refresh failed: " + repr(exc), flush=True)
        return None


def train(config):
    make_runtime, model_class, evaluate = task_components(config)
    settings = config["training"]
    rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    expected_world = int(settings.get("world_size", 8))
    if world != expected_world:
        raise ValueError(f"Recipe requires {expected_world} ranks; received {world}")
    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")
    if device.type == "cuda":
        torch.cuda.set_device(local_rank)
    if world > 1:
        dist.init_process_group("nccl" if device.type == "cuda" else "gloo")
    seed = int(settings.get("seed", 42))
    torch.set_num_threads(int(settings.get("cpu_threads", 2)))
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed + rank)
    random.seed(seed + rank)
    torch.backends.cuda.matmul.allow_tf32 = bool(settings.get("allow_tf32", True))
    accumulation = int(settings.get("gradient_accumulation", 4))
    if accumulation < 1:
        raise ValueError("gradient_accumulation must be positive")
    output = Path(config["output_dir"]).resolve()
    occupied = torch.tensor([int((output / "run_config.json").exists() or
        (output / "train_metrics.csv").exists()) if rank == 0 else 0], device=device)
    if world > 1:
        dist.broadcast(occupied, src=0)
    if occupied.item():
        raise FileExistsError("Use a distinct run directory; resume may explicitly reference an older checkpoint")
    output.mkdir(parents=True, exist_ok=True)
    runtime = make_runtime(config, device)
    resume = settings.get("checkpoint")
    parent = torch.load(resume, map_location="cpu", weights_only=False) if resume else None
    if parent is not None:
        validate_resume(parent, runtime, config, world)
        model = model_class.from_checkpoint(parent).to(device)
        start = int(parent["update"])
    else:
        model = model_class(config["model"].get("config", {}), runtime.mean_assets,
            decoder_checkpoint=config["model"]["decoder_checkpoint"],
            expected_gene_order_sha256=runtime.data.gene_order_sha256,
            loss_weights=_loss_weights(config)).to(device)
        start = 0
    runtime.initialization_provenance = (parent.get("initialization_provenance", {}) if parent is not None else {
        "flow": "fresh_random_initialization", "mean_coefficients": "fresh_zero_initialization",
        "decoder_checkpoint_sha256": sha256(config["model"]["decoder_checkpoint"]),
        "decoder_source_path": str(config["model"]["decoder_checkpoint"]),
        "gene_order_sha256": runtime.data.gene_order_sha256})
    final = int(settings["max_updates"])
    if final <= start:
        raise ValueError("max_updates must exceed the starting optimizer step")
    rates = settings["learning_rates"]
    optimizers, optimizer_names = make_optimizers(model, rates)
    # Candidates are shared across ranks. OT pairs/time/noise and the independent
    # decoder samples have their own streams, all included in the checkpoint.
    candidate_rng = np.random.default_rng(seed)
    pair_generator = torch.Generator(device=device).manual_seed(seed + rank * 100003)
    decoder_rng = np.random.default_rng(seed + 770003 + rank * 100003)
    # Combo's historical dropout stream starts immediately after native model
    # construction, identically seeded on each rank. Keep that initialization
    # policy; its OT and decoder cell draws already use separate streams.
    if config["task"] != "combosciplex":
        torch.manual_seed(seed + rank * 100003)
        if device.type == "cuda":
            torch.cuda.manual_seed(seed + rank * 100003)
    if parent is not None:
        restore_optimizers(parent, optimizers, optimizer_names)
        restore_rng(parent["rng_states"][rank], candidate_rng, pair_generator, decoder_rng)
    source = source_identity()
    metadata = {**config, "model_family": "mlp_mean", "start_update": start,
        "final_update": final, "world_size": world, "train_conditions": len(runtime.data.train_records),
        "conditions_per_update": accumulation,
        "pair_computations_per_update": world * accumulation * int(settings.get("pairs_per_condition", 256)),
        "decoder_cells_per_rank_per_microstep": int(settings.get("decoder_cells", 256)),
        "loss_weights": _loss_weights(config), "expression_gradient_to_flow": False,
        "mmd_gradient_to_head": False, "source_identity": source,
        "asset_manifest_sha256": runtime.asset_manifest_sha256,
        "training_binding": runtime.data.training_binding(),
        "initialization_mode": (("source_change_continuation" if settings.get("allow_source_change", False)
                                 else "exact_resume") if resume else "fresh_flow_and_zero_head_pretrained_decoder"),
        "parent_checkpoint_sha256": sha256(resume) if resume else None,
        "initialization_provenance": runtime.initialization_provenance,
        "selection_protocol": "external-selected exploratory; all candidate metrics retained"}
    if rank == 0:
        snapshot_source(output, source)
        atomic_json(output / "run_config.json", metadata)
        atomic_json(output / "run_manifest.json", metadata)
        atomic_json(output / "status.json", {"status": "initializing", "optimizer_step": start})
    def save(step):
        path = checkpoint_path(output, step)
        create = torch.tensor([int(not path.exists()) if rank == 0 else 0], device=device)
        if world > 1:
            dist.broadcast(create, src=0)
        if create.item():
            save_checkpoint(path, model, optimizers, optimizer_names, step, config, runtime,
                candidate_rng, pair_generator, decoder_rng, rank, world, source)
        elif world > 1:
            dist.barrier()
        return path
    save(start)
    wrapped = (DistributedDataParallel(model, device_ids=[local_rank] if device.type == "cuda" else None,
                broadcast_buffers=False,
                # Keep Combo's native fixed bucket policy. With False, DDP
                # rebuilds buckets after warmup; a resumed process then reduces
                # gradients differently from uninterrupted training.
                find_unused_parameters=config["task"] == "combosciplex") if world > 1 else model)
    modules = {"flow": model.flow, "decoder": model.decoder, "head": model.mean_head}
    tracked = {key: [(n, p) for n, p in module.named_parameters() if p.requires_grad][-1]
               for key, module in modules.items()}
    training_seconds = 0.0
    evaluation_seconds = 0.0
    checkpoint_seconds = 0.0
    started = time.monotonic()
    every_mean = int(config.get("evaluation", {}).get("mean_every", 500))
    every_full = int(config.get("evaluation", {}).get("full_every", 2000))
    every_save = int(settings.get("checkpoint_every", 500))
    try:
        for step in range(start + 1, final + 1):
            tick = time.monotonic()
            model.train()
            factor = learning_rate_factor(step, settings.get("schedule", {}))
            for name, optimizer in optimizers.items():
                # The kernel regression has its own stable learning rate; do not
                # silently decay it with the historical Flow schedule.
                group_factor = 1.0 if name == "head" else factor
                for group in optimizer.param_groups:
                    group["lr"] = float(rates[name]) * group_factor
                optimizer.zero_grad(set_to_none=True)
            first = step == start + 1
            before = {key: p.detach().clone() for key, (_, p) in tracked.items()} if first else {}
            averages, ot_errors, ot_convergence = {}, [], []
            for micro in range(accumulation):
                if hasattr(runtime, "set_training_context"):
                    runtime.set_training_context(step, micro, accumulation, rank, world)
                if hasattr(runtime, "training_condition"):
                    index = runtime.training_condition(step, micro, accumulation, seed, rank, world)
                else:
                    index = condition_index(step, micro, accumulation, len(runtime.data.train_records), seed)
                record = runtime.data.train_records[index]
                batch, ot = runtime.training_batch(record, candidate_rng, pair_generator, decoder_rng,
                    pairs=int(settings.get("pairs_per_condition", 256)),
                    candidates=int(settings.get("ot_candidates", 256)),
                    decoder_cells=int(settings.get("decoder_cells", 256)))
                sync = wrapped.no_sync() if world > 1 and micro < accumulation - 1 else nullcontext()
                with sync:
                    with torch.autocast(device.type, dtype=torch.bfloat16,
                            enabled=device.type == "cuda" and settings.get("precision", "bf16") == "bf16"):
                        loss, parts = wrapped(batch, time_points=int(settings.get("time_points", 80)))
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f"Nonfinite training loss at update {step}, rank {rank}")
                    (loss / accumulation).backward()
                for key, value in parts.items():
                    averages[key] = averages.get(key, 0.0) + float(value.detach() if torch.is_tensor(value) else value) / accumulation
                ot_errors.append(float(ot.marginal_error))
                ot_convergence.append(float(ot.converged))
            for name, module in modules.items():
                maximum = float(settings.get("gradient_clip", {}).get(name, 1.0))
                norm = torch.nn.utils.clip_grad_norm_(module.parameters(), maximum, error_if_nonfinite=True)
                averages[name + "_gradient_norm"] = float(norm)
            for optimizer in optimizers.values():
                optimizer.step()
            if first:
                changes = {key: {"parameter": name,
                    "changed_elements": int(torch.count_nonzero(p.detach() - before[key])),
                    "max_abs_change": float((p.detach() - before[key]).abs().max())}
                    for key, (name, p) in tracked.items()}
                if any(item["changed_elements"] == 0 for item in changes.values()):
                    raise RuntimeError("A trainable module did not update: " + str(changes))
                if rank == 0:
                    atomic_json(output / "FIRST_UPDATE_VERIFIED.json", {"status": "passed", "update": step,
                        "parameters": changes, "optimizer_parameter_names": optimizer_names})
            averages["ot_marginal_error"] = sum(ot_errors) / accumulation
            averages["ot_converged_fraction"] = sum(ot_convergence) / accumulation
            values = torch.tensor(list(averages.values()), device=device, dtype=torch.float64)
            if world > 1:
                dist.all_reduce(values)
                values /= world
            averages = dict(zip(averages, values.cpu().tolist()))
            duration = time.monotonic() - tick
            training_seconds += duration
            if rank == 0:
                row = {"update": step, "optimizer_step": step, "micro_step": step * accumulation,
                    "condition_passes": step * accumulation / len(runtime.data.train_records),
                    "elapsed_seconds": time.monotonic() - started, "training_seconds": training_seconds,
                    "seconds_per_update": training_seconds / (step - start),
                    "evaluation_seconds": evaluation_seconds, "checkpoint_seconds": checkpoint_seconds,
                    "learning_rate_flow": optimizers["flow"].param_groups[0]["lr"],
                    "learning_rate_decoder": optimizers["decoder"].param_groups[0]["lr"],
                    "learning_rate_head": optimizers["head"].param_groups[0]["lr"], **averages}
                append_csv(output / "train_metrics.csv", row)
                atomic_json(output / "status.json", {"status": "training", **row})
                if first or step % int(settings.get("log_every", 10)) == 0:
                    print(json.dumps(row), flush=True)
            mean_due = every_mean > 0 and (step % every_mean == 0 or step == final)
            full_due = every_full > 0 and (step % every_full == 0 or step == final)
            if step == final or (every_save > 0 and step % every_save == 0) or mean_due or full_due:
                tick = time.monotonic()
                saved = save(step)
                checkpoint_seconds += time.monotonic() - tick
            if mean_due or full_due:
                state = capture_rng(candidate_rng, pair_generator, decoder_rng)
                tick = time.monotonic()
                if rank == 0:
                    atomic_json(output / "status.json", {"status": "evaluating", "optimizer_step": step})
                    refresh_monitor(output)
                if mean_due:
                    diagnostic_seeds = tuple(config.get("evaluation", {}).get("mean_seeds", [4243,4244,4245,4246]))
                    evaluate(model, runtime, config, step, saved, seeds=diagnostic_seeds,
                             mean_only=True, rank=rank, world=world, purpose="monitor")
                if full_due:
                    seeds = tuple(config.get("evaluation", {}).get("monitor_seeds", [4243]))
                    evaluate(model, runtime, config, step, saved, seeds=seeds,
                             mean_only=False, rank=rank, world=world, purpose="monitor")
                    if (step == final and config.get("evaluation", {}).get("final_four_seeds", False)
                            and not config.get("selection", {}).get("benchmark_selected", False)):
                        evaluate(model, runtime, config, step, saved, seeds=(4243,4244,4245,4246),
                                 mean_only=False, rank=rank, world=world, purpose="benchmark")
                if world > 1:
                    dist.barrier()
                restore_rng(state, candidate_rng, pair_generator, decoder_rng)
                evaluation_seconds += time.monotonic() - tick
                torch.backends.cuda.matmul.allow_tf32 = bool(settings.get("allow_tf32", True))
            if rank == 0 and (first or step % int(settings.get("monitor_every", 50)) == 0 or step == final):
                refresh_monitor(output)
        selected_path = output / "SELECTED_CHECKPOINT.json"
        selected = None
        if config.get("selection", {}).get("benchmark_selected", False):
            if world > 1:
                dist.barrier()
            if not selected_path.is_file():
                raise RuntimeError("No complete online evaluation is available for checkpoint selection")
            selected = json.loads(selected_path.read_text())
            selected_checkpoint = selected["checkpoint"]
            selected_step = int(selected["update"])
            if sha256(selected_checkpoint) != selected["checkpoint_sha256"]:
                raise ValueError("Selected checkpoint content changed after its online evaluation")
            # The training model and its optimizer remain untouched. Every
            # final metric comes from this single selected saved model.
            final_model = model_class.from_checkpoint(selected_checkpoint).to(device).eval()
            tick = time.monotonic()
            evaluate(final_model, runtime, config, selected_step, selected_checkpoint,
                     seeds=tuple(config["prediction"]["seeds"]), mean_only=False,
                     rank=rank, world=world, purpose="benchmark")
            evaluation_seconds += time.monotonic() - tick
            del final_model
        if rank == 0:
            atomic_json(output / "TRAINING_COMPLETE.json", {"status": "complete", "optimizer_step": final,
                "checkpoint": str(checkpoint_path(output, final)),
                "training_seconds": training_seconds, "evaluation_seconds": evaluation_seconds,
                "checkpoint_seconds": checkpoint_seconds,
                "seconds_per_update": training_seconds / (final - start),
                "selected_checkpoint": selected,
                "selection_note": ("External-selected exploratory; all final metrics use the selected checkpoint"
                    if selected else "The last checkpoint is not automatically the best; inspect the candidate index.")})
            atomic_json(output / "status.json", {"status": "complete", "optimizer_step": final,
                "condition_passes": final * accumulation / len(runtime.data.train_records)})
            refresh_monitor(output)
    except BaseException as exc:
        if rank == 0:
            atomic_json(output / "FAILED.json", {"status": "failed", "error": repr(exc)})
        raise
    finally:
        if world > 1 and dist.is_initialized():
            dist.destroy_process_group()
    return {"status": "complete", "output_dir": str(output), "optimizer_step": final}
