"""Read the complete SciPlex3 flow before launching a distributed recipe."""
import argparse
from pathlib import Path
import json
from devflow.config import load_config


def validate_config(config):
    if config.get("model", {}).get("family") == "mlp_mean":
        from recipes.sciplex3_mlp import validate_config as validate_mlp
        return validate_mlp(config)
    for key in ("data","model","output_dir"):
        if key not in config:raise ValueError(f"Missing configuration key: {key}")
    for key in ("base_dir","source_split_root","sampling_manifest","feature_cache","prefix_cache"):
        if not Path(config["data"][key]).exists():raise FileNotFoundError(config["data"][key])
    if not Path(config["model"]["tail_checkpoint"]).is_file():raise FileNotFoundError("SE tail asset")
    if config.get("training",{}).get("checkpoint") and not Path(config["training"]["checkpoint"]).is_file():raise FileNotFoundError("Parent checkpoint")
    return {"status":"passed","task":"sciplex3","architecture":"time-modulated attention + control SE LoRA + strong decoder","mean_head":False}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",required=True)
    parser.add_argument("command",choices=("validate","train","evaluate","predict","monitor"))
    args=parser.parse_args()
    config=load_config(args.config)
    if config.get("model", {}).get("family") == "mlp_mean":
        from devflow.cli import run
        result=run("sciplex3",args.command,config)
        if result is not None:print(json.dumps(result,indent=2,default=str))
        return
    if args.command=="validate":print(json.dumps(validate_config(config),indent=2));return
    if args.command=="monitor":
        from devflow.runtime.monitor import build_monitor
        build_monitor(config["output_dir"])
        return
    validate_config(config)
    if args.command=="train":
        from devflow.runtime.train_sciplex3 import train
        train(config)
    elif args.command=="evaluate":
        from devflow.runtime.evaluate_sciplex3 import evaluate_checkpoint
        evaluate_checkpoint(config)
    else:
        from devflow.runtime.evaluate_sciplex3 import predict
        predict(config)

if __name__=="__main__":main()
