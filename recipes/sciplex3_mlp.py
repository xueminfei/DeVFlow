"""Validate the basic MLP, trainable mean, and decoder-MMD recipe."""
from pathlib import Path
import json


def validate_config(config):
    if config.get("task") != "sciplex3" or config.get("model", {}).get("family") != "mlp_mean":
        raise ValueError("Expected the sciplex3 / mlp_mean recipe")
    for name in ("base_dir", "source_split_root", "sampling_manifest", "mlp_assets"):
        value = config["data"].get(name)
        if not value or not Path(value).exists():
            raise FileNotFoundError(f"Missing data.{name}: {value}")
    manifest_path = Path(config["data"]["mlp_assets"]) / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != "sciplex3_mlp_mean_assets_v1" or manifest.get("status") != "complete":
        raise ValueError("MLP asset preparation is incomplete or incompatible")
    resume = config.get("training", {}).get("checkpoint")
    inference = config.get("prediction", {}).get("checkpoint")
    decoder = config["model"].get("decoder_checkpoint")
    if not resume and not inference and (not decoder or not Path(decoder).is_file()):
        raise FileNotFoundError("Fresh training requires model.decoder_checkpoint")
    if resume and not Path(resume).is_file():
        raise FileNotFoundError(resume)
    return {"status": "passed", "task": "sciplex3", "model_family": "mlp_mean",
            "mean_head": True, "gradient_routes": {"flow": "flow_residual + 32*flow_mean",
            "head": "head_fit + RKHS_regularizer", "decoder": "final_expression_MMD"},
            "selection": "external-selected exploratory", "asset_manifest": str(manifest_path)}
