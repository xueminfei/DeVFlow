"""Native ComboSciPlex paper recipe; model/data code has no legacy imports."""
from __future__ import annotations
import argparse
import csv
import json
import math
import os
from pathlib import Path
import random
from contextlib import nullcontext
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from devflow.config import load_config
from devflow.data.combosciplex import (ComboDataset, CELL_LINES, load_bundle,
    population_expression_mean, export_evaluation_references, training_binding)
from devflow.networks.chemical_condition import load_drug_bank, sample_conditions
from devflow.runtime.combosciplex import ComboSciPlexModel, evaluate_combo
from devflow.io import (write_json, write_prediction_archive, sha256_file, json_sha256)


def validate_config(config):
    if config.get("task") != "combosciplex":
        raise ValueError("Expected task=combosciplex")
    bundle = load_bundle(config["data"]["bundle"], require_state=True, full_scan=True)
    names, bank = load_drug_bank(config["data"]["drug_bank"])
    for key, expected in dict(state_dim=2048, expression_dim=2000, condition_dim=2052,
                              source_condition=False, direct_condition=True).items():
        if config.get("model", {}).get(key, expected) != expected:
            raise ValueError(f"The paper Combo model requires {key}={expected}")
    return dict(status="validated", **bundle["audit"], drug_bank_size=len(names),
                training_binding=training_binding(bundle))


def _sample_ids(dataset, sample, field):
    return [dataset.bundle["obs"][int(index)]["cell_id"] for index in sample[field]]


def predict(config):
    validate_config(config)
    options = config["prediction"]
    device = torch.device(options.get("device", config.get("device", "cuda:0")))
    model = ComboSciPlexModel.from_checkpoint(options["checkpoint"]).to(device).eval()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    smiles, bank = load_drug_bank(config["data"]["drug_bank"], device)
    output = Path(options["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    seeds = tuple(options.get("seeds", (4242, 4243, 4244, 4245, 4246)))
    cells, points = int(options.get("cells_per_condition", 128)), int(options.get("time_points", 80))
    chunk = int(options.get("chunk_size", cells))
    if cells != 128 or points != 80 or chunk < 1:
        raise ValueError("Formal Combo inference requires128 cells and RK4-80")
    checkpoint_hash = sha256_file(options["checkpoint"])
    paths = []
    bundle = load_bundle(config["data"]["bundle"])
    if model.checkpoint_gene_order_sha256 != bundle["manifest"]["gene_order_sha256"]:
        raise ValueError("Checkpoint decoder gene order differs from the input bundle")
    if model.checkpoint_cell_order_sha256 and model.checkpoint_cell_order_sha256 != bundle["manifest"]["cell_order_sha256"]:
        raise ValueError("Checkpoint cell-order identity differs from the input bundle")
    if model.checkpoint_training_binding and model.checkpoint_training_binding != training_binding(bundle):
        raise ValueError("Checkpoint training split differs from the input bundle")
    for seed in seeds:
        destination = output/f"predictions_seed_{seed}.npz"
        if destination.exists():
            raise FileExistsError(destination)
        dataset = ComboDataset(CELL_LINES, "external_test", config["data"]["bundle"],
                               set_size=cells, seed=seed, _bundle=bundle)
        predictions, sources, targets = [], [], []
        with torch.inference_mode(), torch.autocast(device_type=device.type, enabled=False):
            for index in range(len(dataset)):
                sample = dataset[index]
                drug, dose, line = sample_conditions(sample, dataset, smiles, bank, device)
                pieces = []
                for first in range(0, cells, chunk):
                    sl = slice(first, first+chunk)
                    _, expression = model.sample(sample["control_state"][sl].to(device).float(),
                        None, None, drug[sl].float(), dose[sl].float(), line[sl].float(),
                        time_points=points, adaptive=False)
                    pieces.append(expression.float().cpu().numpy())
                predictions.append(np.concatenate(pieces))
                sources.append(_sample_ids(dataset, sample, "control_rows"))
                targets.append(_sample_ids(dataset, sample, "treated_rows"))
        write_prediction_archive(destination, seed=seed,
            condition_names=[record["condition"] for record in dataset.records],
            source_global_ids=np.asarray(sources), target_global_ids=np.asarray(targets),
            gene_order_sha256=bundle["manifest"]["gene_order_sha256"], predictions=predictions,
            metadata=dict(task="combosciplex", model_family=model.model_family,
                checkpoint_sha256=checkpoint_hash, training_binding=training_binding(bundle),
                training_binding_sha256=json_sha256(training_binding(bundle)),
                precision="FP32", time_points=points, chunk_size=chunk,
                condition_pool="canonical_sorted_direct_concat_frozen_MAPKG_2x1024",
                response_profile="full_treated", distribution_target="sampled128"))
        paths.append(str(destination))
    return dict(status="complete", predictions=paths, checkpoint_sha256=checkpoint_hash)


def export_references(config):
    options = config.get("evaluation", {})
    output = options.get("references_dir", str(Path(config["prediction"]["output_dir"]).parent/"references"))
    return export_evaluation_references(config["data"]["bundle"], output,
        tuple(config["prediction"].get("seeds", (4242, 4243, 4244, 4245, 4246))))


def evaluate(config):
    from benchmark.reproduce.common import evaluate as score_archives
    options = config.get("evaluation", {})
    references = Path(options.get("references_dir", str(Path(config["prediction"]["output_dir"]).parent/"references")))
    if not (references/"manifest.json").exists():
        export_references(config)
    output = options.get("output_dir", str(Path(config["prediction"]["output_dir"]).parent/"benchmark"))
    _, summary = score_archives(config["prediction"]["output_dir"], references/"protocol.json",
        references, output, method=options.get("method", "DeVFlow"), task="combosciplex")
    return dict(status="complete", summary=str(Path(output)/"summary.csv"),
                pooled=summary[summary.setting == "pooled"].to_dict("records"))


def _save_checkpoint(path, model, optimizer, step, micro_step, config, world):
    bundle = load_bundle(config["data"]["bundle"])
    payload = dict(format_version=1, model_family=model.model_family,
        model=model.state_dict(), optimizer=optimizer.state_dict(),
        optimizer_step=step, micro_step=micro_step, world_size=world,
        args={**config["training"], **config.get("model", {}),
            "no_source_condition": True, "cell_lines": CELL_LINES, "direct_condition": True,
            "condition_dim": 2052, "condition_output_dropout": 0.1,
            "decoder_mean_loss_weight": config["training"].get("decoder_mean_loss_weight", 1.0)},
        devflow_config=config, condition_dim=2052,
        gene_order_sha256=bundle["manifest"]["gene_order_sha256"],
        split_protocol=bundle["manifest"]["split_protocol"],
        training_binding=training_binding(bundle),
        condition_pool="canonical_sorted_direct_concat_frozen_MAPKG_2x1024")
    tmp = Path(str(path)+".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def train(config):
    """Fit the native paper model; this function is never invoked by import."""
    validate_config(config)
    options = config["training"]
    world, rank, local = (int(os.getenv(key, default)) for key, default in
                          (("WORLD_SIZE", "1"), ("RANK", "0"), ("LOCAL_RANK", "0")))
    if world != int(options.get("world_size", 4)):
        raise ValueError("world size differs from the explicit training profile")
    device = torch.device("cuda", local) if torch.cuda.is_available() else torch.device("cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    if world > 1:
        dist.init_process_group("nccl" if device.type == "cuda" else "gloo")
    seed = int(options.get("seed", 42))
    random.seed(seed); np.random.seed(seed+rank); torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(seed)
    output = Path(config["output_dir"])
    if (output/"train_metrics.csv").exists() or (output/"TRAINING_COMPLETE.json").exists():
        raise FileExistsError("Choose a new experiment output directory")
    output.mkdir(parents=True, exist_ok=True)
    bundle = load_bundle(config["data"]["bundle"])
    dataset = ComboDataset(CELL_LINES, "train", config["data"]["bundle"],
        set_size=int(options.get("cells_per_condition", 256)), seed=seed, _bundle=bundle)
    smiles, bank = load_drug_bank(config["data"]["drug_bank"], device)
    model_options = dict(config.get("model", {}))
    model_options["decoder_mean_loss_weight"] = float(options.get("decoder_mean_loss_weight", 1.0))
    model_options["path_sigma"] = float(options.get("path_sigma", 0.01))
    model = ComboSciPlexModel(**model_options)
    resume = options.get("resume_checkpoint")
    saved = torch.load(resume, map_location="cpu", weights_only=False) if resume else None
    if saved is not None:
        if saved["model_family"] != model.model_family or saved["world_size"] != world:
            raise ValueError("Resume checkpoint family/world size mismatch")
        if saved.get("gene_order_sha256") != bundle["manifest"]["gene_order_sha256"]:
            raise ValueError("Resume checkpoint gene order differs from the training bundle")
        if saved.get("training_binding") and saved["training_binding"] != training_binding(bundle):
            raise ValueError("Resume checkpoint training split differs from the input bundle")
        model.load_state_dict(saved["model"], strict=True)
    else:
        decoder_path = config["data"]["decoder_checkpoint"]
        decoder = torch.load(decoder_path, map_location="cpu", weights_only=False)
        if decoder.get("gene_order_sha256") != bundle["manifest"]["gene_order_sha256"]:
            raise ValueError("Pretrained decoder gene-order mismatch")
        state = decoder.get("decoder_state_dict")
        if state is None:
            source = decoder.get("model", decoder.get("model_state_dict", decoder.get("state_dict")))
            if source is None:
                raise ValueError("Decoder artifact has no state dictionary")
            state = {name[len("decoder."):]: value for name, value in source.items() if name.startswith("decoder.")}
            if not state:
                state = source
        model.decoder.load_state_dict(state, strict=True)
    model.to(device)
    flow_parameters = [p for p in model.flow.parameters() if p.requires_grad]
    decoder_parameters = [p for p in model.decoder.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam([dict(params=flow_parameters, component="flow"),
                                  dict(params=decoder_parameters, component="decoder")],
                                 lr=float(options.get("learning_rate", 5e-5)))
    if saved is not None:
        optimizer.load_state_dict(saved["optimizer"])
    start_step, micro = ((int(saved["optimizer_step"]), int(saved["micro_step"])) if saved else (0, 0))
    del saved
    wrapped = DDP(model, device_ids=[local] if device.type == "cuda" else None,
                  broadcast_buffers=False, find_unused_parameters=True) if world > 1 else model
    if rank == 0:
        write_json(output/"run_config.json", config)
        write_json(output/"training_binding.json", training_binding(bundle))
    accumulation = int(options.get("gradient_accumulation", 4))
    steps = int(options.get("steps", 13000))
    metrics_path = output/"train_metrics.csv"
    for step in range(start_step+1, steps+1):
        wrapped.train(); optimizer.zero_grad(set_to_none=True); totals = {}
        before_flow = model.flow.velocity.decoder.net[0].weight.detach().clone() if step == start_step+1 else None
        before_decoder = model.decoder.decoder[0].weight.detach().clone() if step == start_step+1 else None
        for slot in range(accumulation):
            epoch = micro // len(dataset)
            order = np.random.default_rng(np.random.SeedSequence([seed, epoch])).permutation(len(dataset))
            index = int(order[micro % len(dataset)])
            dataset.set_epoch(epoch); sample = dataset[index]
            drug, dose, line = sample_conditions(sample, dataset, smiles, bank, device)
            generator = torch.Generator(device=device).manual_seed(seed*1_000_003+micro*world+rank)
            context = wrapped.no_sync() if world > 1 and slot+1 < accumulation else nullcontext()
            amp = torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" and options.get("amp_dtype", "bf16") == "bf16" else nullcontext()
            with context:
                with amp:
                    loss, parts = wrapped(sample["control_state"].to(device), None, None,
                        sample["treated_state"].to(device), sample["treated_expression"].to(device),
                        drug, dose, line, ot_epsilon=float(options.get("ot_epsilon", 0.01)),
                        ot_max_iterations=int(options.get("ot_max_iterations", 2000)),
                        ot_tolerance=float(options.get("ot_tolerance", 1e-3)),
                        endpoint_time_points=int(options.get("endpoint_time_points", 80)),
                        generator=generator, mean_shift_target=dataset.population_state_delta(index).to(device),
                        mean_shift_weight=float(options.get("mean_shift_loss_weight", 32.0)),
                        endpoint_mean_target=population_expression_mean(dataset, index).to(device),
                        decoder_mean_loss_weight=float(options.get("decoder_mean_loss_weight", 1.0)))
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite Combo loss")
                (loss/accumulation).backward()
            for name in ("flow_loss", "residual_flow_loss", "mean_shift_loss", "decoder_loss", "decoder_residual_loss", "decoder_mean_loss"):
                totals[name] = totals.get(name, 0.0)+float(parts[name].detach())/accumulation
            micro += 1
        flow_norm = torch.nn.utils.clip_grad_norm_(flow_parameters, float(options.get("gradient_clip", 1.0)), error_if_nonfinite=True)
        decoder_norm = torch.nn.utils.clip_grad_norm_(decoder_parameters, float(options.get("gradient_clip", 1.0)), error_if_nonfinite=True)
        decay_start, decay_end = int(options.get("lr_decay_start_update", 500)), int(options.get("lr_decay_end_update", 13000))
        fraction = min(1.0, max(0.0, (step-decay_start)/(decay_end-decay_start)))
        minimum, base = float(options.get("min_learning_rate", 5e-6)), float(options.get("learning_rate", 5e-5))
        learning_rate = minimum+(base-minimum)*0.5*(1+math.cos(math.pi*fraction))
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        optimizer.step()
        if step == start_step+1 and rank == 0:
            flow_changes = int(torch.count_nonzero(model.flow.velocity.decoder.net[0].weight.detach()-before_flow))
            decoder_changes = int(torch.count_nonzero(model.decoder.decoder[0].weight.detach()-before_decoder))
            if not flow_changes or not decoder_changes:
                raise RuntimeError("Flow or decoder failed to update")
            write_json(output/"FIRST_UPDATE_VERIFIED.json", dict(step=step,
                flow_changed_elements=flow_changes, decoder_changed_elements=decoder_changes,
                flow_gradient_before_clip=float(flow_norm), decoder_gradient_before_clip=float(decoder_norm)))
        names = list(totals)
        tensor = torch.tensor([totals[name] for name in names], device=device, dtype=torch.float64)
        if world > 1:
            dist.all_reduce(tensor); tensor /= world
        row = dict(update=step, micro_step=micro, learning_rate=learning_rate,
                   gradient_norm_before_clip=float(flow_norm), decoder_gradient_norm_before_clip=float(decoder_norm),
                   **dict(zip(names, tensor.cpu().tolist())))
        if rank == 0:
            if step == start_step+1 or step % int(options.get("log_every", 25)) == 0:
                exists = metrics_path.exists()
                with metrics_path.open("a", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(row))
                    if not exists: writer.writeheader()
                    writer.writerow(row)
            write_json(output/"status.json", dict(status="training", **row))
        if step % int(options.get("save_every", 13000)) == 0 or step == steps:
            if rank == 0:
                _save_checkpoint(output/f"update_{step:06d}.pt", model, optimizer, step, micro, config, world)
        every = int(options.get("eval_every", 1000))
        if every and (step % every == 0 or step == steps):
            evaluation = ComboDataset(CELL_LINES, "external_test", config["data"]["bundle"],
                                      set_size=128, seed=4242, _bundle=bundle)
            evaluate_combo(wrapped, evaluation, smiles, bank, device, rank, world, step, output, 80)
        if world > 1: dist.barrier()
    if rank == 0:
        write_json(output/"TRAINING_COMPLETE.json", dict(status="complete", updates=steps,
            checkpoint=str(output/f"update_{steps:06d}.pt"), model_family=model.model_family))
    if world > 1: dist.destroy_process_group()
    return dict(status="complete", output_dir=str(output), updates=steps)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("action", choices=("validate", "train", "predict", "evaluate", "export-references"))
    args = parser.parse_args(argv)
    config = load_config(args.config)
    function = {"validate": validate_config, "train": train, "predict": predict,
                "evaluate": evaluate, "export-references": export_references}[args.action]
    result = function(config)
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    main()
