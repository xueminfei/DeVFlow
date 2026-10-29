"""ComboSciPlex's native data and architecture with trainable mean/MMD losses."""
import json
from pathlib import Path


def validate_config(config):
    if config.get("task") != "combosciplex" or config.get("model", {}).get("family") != "mlp_mean":
        raise ValueError("Expected combosciplex / mlp_mean")
    data, model = config["data"], config["model"]
    for name in ("bundle", "drug_bank", "mlp_assets", "evaluation_assets"):
        if not data.get(name) or not Path(data[name]).exists():
            raise FileNotFoundError("Missing data." + name + "; run prepare first")
    manifest = json.loads((Path(data["mlp_assets"]) / "manifest.json").read_text())
    if manifest.get("status") != "complete" or manifest.get("external_expression_used_for_fitting"):
        raise ValueError("Combo training assets must be complete and fitted only on training cells")
    for key, expected in {"state_dim": 2048, "condition_dim": 2052, "source_condition": False,
                          "direct_condition": True, "condition_output_dropout": 0.1,
                          "initialization_policy": "duplicate_drug_slots",
                          "decoder_checkpoint_family": "MAP_true_SE_decoder_reconstruction"}.items():
        if model.get("config", {}).get(key, expected) != expected:
            raise ValueError("Native Combo setting changed: " + key)
    resume, inference = config["training"].get("checkpoint"), config.get("prediction", {}).get("checkpoint")
    needed = resume or inference or model.get("decoder_checkpoint")
    if not needed or not Path(needed).is_file():
        raise FileNotFoundError("Missing selected checkpoint or native strong decoder")
    if tuple(config["prediction"]["seeds"]) != (4243, 4244, 4245, 4246):
        raise ValueError("The new Combo recipe uses four sampling seeds 4243–4246")
    if config["evaluation"].get("monitor_seeds") != [4243]:
        raise ValueError("Online full evaluation must use seed4243")
    from devflow.runtime.evaluate_combosciplex_mlp import selection_policy
    policy = selection_policy(config)
    return {"status": "passed", "task": "combosciplex", "mean_head": True,
            "training_conditions": 24, "test_conditions": 5,
            "sampling_seeds": [4243, 4244, 4245, 4246],
            "selection_policy": policy["policy"],
            "gradient_routes": {"flow": "latent residual + 32*population mean",
                "head": "training-only KRR", "decoder": "scaled expression MMD only"}}


def prepare(config):
    from devflow.data.combo_mean_assets import prepare_assets
    from devflow.runtime.evaluate_combosciplex_mlp import prepare_references
    mean = prepare_assets(config)
    evaluation = prepare_references(config)
    return {"mean": mean, "evaluation": evaluation}
