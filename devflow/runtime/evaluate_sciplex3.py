"""Online diagnostics and formal prediction through the same native model."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from devflow.data.sciplex3 import CELL_LINES
from devflow.runtime.artifacts import atomic_json, sha256
from benchmark.metrics import score_condition
from .checkpoint import validate_checkpoint_runtime


def sampled_records(data, seed=4242, cells=128):
    if seed in (4243,4244,4245,4246):
        if cells != 128:
            raise ValueError("Formal samples lock 128 cells")
        return data.sampling(seed)
    records=[]
    for index,record in enumerate(data.test_records):
        rng=np.random.default_rng(np.random.SeedSequence([seed,0,index]))
        control=rng.choice(record["control_pool"],cells,replace=len(record["control_pool"])<cells)
        treated=rng.choice(record["rows"],cells,replace=len(record["rows"])<cells)
        records.append(dict(record,control_rows=control,target_rows=treated))
    return records


@torch.no_grad()
def predict_record(model,runtime,record,time_points=80,chunk=128,precision="fp32"):
    latent,expression=[],[]
    for start in range(0,len(record["control_rows"]),chunk):
        batch=runtime.source_batch(record,record["control_rows"][start:start+chunk])
        with torch.autocast("cuda",dtype=torch.bfloat16,enabled=precision=="bf16"):
            z,x=model.predict(batch,runtime.esm,time_points)
        latent.append(z.float().cpu().numpy())
        expression.append(x.float().cpu().numpy())
    z,x=np.concatenate(latent),np.concatenate(expression)
    if not np.isfinite(z).all() or not np.isfinite(x).all():
        raise FloatingPointError("Nonfinite evaluation prediction")
    return z,x


@torch.no_grad()
def evaluate_online(model,runtime,config,step,rank=0,world=1):
    model.eval()
    settings=config.get("evaluation",{})
    time_points=int(settings.get("online_time_points",10))
    chunk=int(settings.get("online_chunk",8))
    precision=settings.get("online_precision","bf16")
    folder=Path(config["output_dir"])/"evaluations"/f"update_{step:06d}"
    folder.mkdir(parents=True,exist_ok=True)
    records=sampled_records(runtime.data)
    refs=np.load(config["data"]["evaluation_references"],allow_pickle=False)
    if refs["condition_names"].tolist()!=[r["condition"] for r in records]:
        raise ValueError("Evaluation reference condition order changed")
    if str(refs["gene_order_sha256"])!=runtime.data.gene_order_sha256:
        raise ValueError("Evaluation reference gene axis changed")
    de_path=config["data"].get("de50_references")
    de=np.load(de_path,allow_pickle=False) if de_path else None
    if de is not None:
        indices=de["gene_indices"]
        if de["group_names"].tolist()!=refs["condition_names"].tolist():
            raise ValueError("DE50 condition order changed")
        if indices.shape!=(108,50) or indices.dtype.kind not in "iu" or np.any(indices<0) or np.any(indices>=2000):
            raise ValueError("DE50 gene indices are invalid")
        if any(len(np.unique(row))!=50 for row in indices):
            raise ValueError("DE50 contains duplicate genes")
        if not np.array_equal(de["full_weights"],refs["continuous_deg_weights"]):
            raise ValueError("DE50 weights differ from HVG references")
        expected=config["data"].get("de50_sha256")
        if expected and sha256(de_path)!=expected:
            raise ValueError("DE50 reference identity changed")
    targets=[runtime.data.expression(r["cell_line"],r["target_rows"]) for r in records]
    profiles=np.stack([target.mean(0) for target in targets])
    rows=[]
    for index in range(rank,len(records),world):
        record=records[index]
        latent,prediction=predict_record(model,runtime,record,time_points,chunk,precision)
        control=runtime.data.expression(record["cell_line"],record["control_rows"])
        positions=[i for i,r in enumerate(records) if r["cell_line"]==record["cell_line"]]
        args=(prediction,targets[index],control,refs["continuous_deg_weights"][index],refs["de_mask"][index],
              refs["training_condition_mean"][CELL_LINES.index(record["cell_line"])],
              profiles[positions],positions.index(index))
        values=score_condition(*args,include_distribution=False)
        row=dict(group_index=index,condition=record["condition"],cell_line=record["cell_line"],**values)
        if de is not None:
            panel=score_condition(*args,include_distribution=False,panel="de50",gene_indices=de["gene_indices"][index])
            row.update({"de50_"+key:value for key,value in panel.items()})
        rows.append(row)
        np.savez_compressed(folder/f"group_{index:04d}.npz",prediction=prediction,latent=latent,
            source_rows=record["control_rows"],target_rows=record["target_rows"])
        atomic_json(folder/f"rank_{rank:02d}_progress.json",dict(completed=len(rows),total=len(range(rank,len(records),world)),last_group=index))
    if world>1:
        gathered=[None]*world
        dist.all_gather_object(gathered,rows)
        rows=[row for shard in gathered for row in shard]
    if rank==0:
        rows.sort(key=lambda row:row["group_index"])
        if [row["group_index"] for row in rows]!=list(range(108)):
            raise RuntimeError("Online evaluation missing conditions")
        pd.DataFrame(rows).to_csv(folder/"per_group.csv",index=False)
        metrics={}
        counts={}
        for key in rows[0]:
            if key in ("group_index","condition","cell_line"):
                continue
            values=np.asarray([row[key] for row in rows],dtype=np.float64)
            finite=np.isfinite(values)
            metrics[key]=float(values[finite].mean()) if finite.any() else None
            counts[key]=int(finite.sum())
        atomic_json(folder/"summary.json",dict(update=step,split="external_test",n_conditions=108,
            protocol="online128_seed4242",seed=4242,time_points=time_points,chunk=chunk,precision=precision,
            metrics=metrics,n_finite=counts,distribution_metrics="not computed online",
            de50_panel="fixed original condition-specific panel" if de is not None else "unavailable",
            note="Exploratory monitoring; sampling variability is distinct from model retraining."))
    return rows


def evaluate_checkpoint(config):
    import os
    from devflow.model_attention import SciPlex3Model
    from .train_sciplex3 import make_runtime
    rank,world=int(os.environ.get("RANK",0)),int(os.environ.get("WORLD_SIZE",1))
    local=int(os.environ.get("LOCAL_RANK",0))
    torch.cuda.set_device(local)
    if world>1:dist.init_process_group("nccl")
    torch.set_num_threads(2)
    runtime=make_runtime(config,torch.device("cuda",local))
    checkpoint=config.get("prediction",{}).get("checkpoint") or config.get("training",{}).get("checkpoint")
    if not checkpoint:raise ValueError("Evaluation needs a full model checkpoint")
    payload=torch.load(checkpoint,map_location="cpu",mmap=True,weights_only=False)
    validate_checkpoint_runtime(payload,runtime,config["model"]["tail_checkpoint"])
    model=SciPlex3Model.from_checkpoint(checkpoint,tail_checkpoint=config["model"]["tail_checkpoint"]).to(runtime.device)
    model.bind_control_prefixes(runtime.prefixes)
    step=int(payload["update"])
    try:evaluate_online(model,runtime,config,step,rank,world)
    finally:
        if world>1:dist.destroy_process_group()


def predict(config):
    import os
    from devflow.model_attention import SciPlex3Model
    from .train_sciplex3 import make_runtime
    rank,world=int(os.environ.get("RANK",0)),int(os.environ.get("WORLD_SIZE",1))
    local=int(os.environ.get("LOCAL_RANK",0))
    torch.cuda.set_device(local)
    if world>1:dist.init_process_group("nccl")
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.set_float32_matmul_precision("highest")
    runtime=make_runtime(config,torch.device("cuda",local))
    settings=config["prediction"]
    payload=torch.load(settings["checkpoint"],map_location="cpu",mmap=True,weights_only=False)
    validate_checkpoint_runtime(payload,runtime,config["model"]["tail_checkpoint"])
    model=SciPlex3Model.from_checkpoint(settings["checkpoint"],tail_checkpoint=config["model"]["tail_checkpoint"]).to(runtime.device).eval()
    model.bind_control_prefixes(runtime.prefixes)
    output=Path(config["output_dir"])/"predictions"
    output.mkdir(parents=True,exist_ok=True)
    try:
        for seed in settings.get("seeds",[4243,4244,4245,4246]):
            sampled=runtime.data.sampling(seed)
            local_predictions=[]
            for index in range(rank,len(sampled),world):
                _,x=predict_record(model,runtime,sampled[index],int(settings.get("time_points",80)),int(settings.get("chunk",128)),"fp32")
                local_predictions.append((index,x))
            if world>1:
                gathered=[None]*world
                dist.all_gather_object(gathered,local_predictions)
                local_predictions=[item for shard in gathered for item in shard]
            if rank==0:
                local_predictions.sort(key=lambda item:item[0])
                if [item[0] for item in local_predictions]!=list(range(len(sampled))):raise RuntimeError("Missing predictions")
                runtime.data.save_predictions(output,seed,[item[1] for item in local_predictions],"DeVFlow attention control-SE-LoRA",
                    dict(checkpoint_sha256=sha256(settings["checkpoint"]),precision="FP32; TF32 disabled",time_points=int(settings.get("time_points",80))))
    finally:
        if world>1:dist.destroy_process_group()
