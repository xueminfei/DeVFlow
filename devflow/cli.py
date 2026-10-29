"""Unified entry point dispatching to the original task-specific recipes."""
from __future__ import annotations

import argparse
import importlib
import json

from .config import load_config


_ACTIONS = {
    "sciplex3": {
        "prepare": ("devflow.data.mean_assets", "prepare_assets"),
        "validate": ("recipes.sciplex3", "validate_config"),
        "train": ("devflow.runtime.train_sciplex3", "train"),
        "predict": ("devflow.runtime.evaluate_sciplex3", "predict"),
        "evaluate": ("devflow.runtime.evaluate_sciplex3", "evaluate_checkpoint"),
        "monitor": ("devflow.runtime.monitor", "build_monitor"),
    },
    "combosciplex": {
        "prepare": ("devflow.data.combo_mean_assets", "prepare_assets"),
        "validate": ("recipes.combosciplex", "validate_config"),
        "train": ("recipes.combosciplex", "train"),
        "predict": ("recipes.combosciplex", "predict"),
        "evaluate": ("recipes.combosciplex", "evaluate"),
        "references": ("recipes.combosciplex", "export_references"),
        "export-references": ("recipes.combosciplex", "export_references"),
        "monitor": ("devflow.runtime.monitor", "build_monitor"),
    },
    "norman": {
        "validate": ("devflow.runtime.norman", "validate_config"),
        "train-decoder": ("devflow.runtime.norman", "train_decoder"),
        "train": ("devflow.runtime.norman", "train"),
        "predict": ("devflow.runtime.norman", "predict"),
        "references": ("devflow.runtime.norman", "export_references"),
        "export-references": ("devflow.runtime.norman", "export_references"),
        "monitor": ("devflow.runtime.monitor", "build_monitor"),
    },
}


def run(task, action, config):
    """Execute a native action without changing its data or training protocol.

    Prediction consumes the configured checkpoint and prepared assets. Imports
    are deferred until dispatch; importing this module never starts a run.
    """
    if task not in _ACTIONS or action not in _ACTIONS[task]:
        raise ValueError(f"Unsupported task/action: {task}/{action}")
    expected = {"norman", "norman19"} if task == "norman" else {task}
    if config.get("task", task) not in expected:
        raise ValueError(f"Configuration task {config.get('task')!r} does not match {task!r}")
    mlp = task == "sciplex3" and config.get("model", {}).get("family") == "mlp_mean"
    if task == "combosciplex" and config.get("model", {}).get("family") == "mlp_mean":
        actions = {
            "prepare": ("recipes.combosciplex_mlp", "prepare"),
            "validate": ("recipes.combosciplex_mlp", "validate_config"),
            "train": ("devflow.runtime.train_mlp_mean", "train"),
            "predict": ("devflow.runtime.evaluate_combosciplex_mlp", "predict"),
            "evaluate": ("devflow.runtime.evaluate_combosciplex_mlp", "evaluate_checkpoint"),
            "references": ("devflow.runtime.evaluate_combosciplex_mlp", "prepare_references"),
            "export-references": ("devflow.runtime.evaluate_combosciplex_mlp", "prepare_references"),
            "monitor": ("devflow.runtime.monitor", "build_monitor"),
        }
        if action in {"train", "predict", "evaluate"}:
            importlib.import_module("recipes.combosciplex_mlp").validate_config(config)
        module_name, function_name = actions[action]
        function = getattr(importlib.import_module(module_name), function_name)
        return function(config["output_dir"]) if action == "monitor" else function(config)
    if mlp:
        actions = {
            "prepare": ("devflow.data.mean_assets", "prepare_assets"),
            "validate": ("recipes.sciplex3_mlp", "validate_config"),
            "train": ("devflow.runtime.train_mlp_mean", "train"),
            "predict": ("devflow.runtime.evaluate_sciplex3_mlp", "predict"),
            "evaluate": ("devflow.runtime.evaluate_sciplex3_mlp", "evaluate_checkpoint"),
            "monitor": ("devflow.runtime.monitor", "build_monitor"),
        }
        if action in {"train", "predict", "evaluate"}:
            importlib.import_module("recipes.sciplex3_mlp").validate_config(config)
        module_name, function_name = actions[action]
        function = getattr(importlib.import_module(module_name), function_name)
        return function(config["output_dir"]) if action == "monitor" else function(config)
    if task == "sciplex3" and action == "prepare":
        raise ValueError("SciPlex3 prepare currently requires model.family=mlp_mean")
    if task == "sciplex3" and action in {"train", "predict", "evaluate"}:
        # Preserve the native SciPlex recipe's preflight before any work.
        importlib.import_module("recipes.sciplex3").validate_config(config)
    module_name, function_name = _ACTIONS[task][action]
    function = getattr(importlib.import_module(module_name), function_name)
    if action == "monitor":
        return function(config["output_dir"])
    return function(config)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="task", required=True)
    for task, actions in _ACTIONS.items():
        task_parser = subparsers.add_parser(task, help=f"Run the native {task} recipe")
        task_parser.add_argument("--config", required=True, help="Portable JSON recipe")
        task_parser.add_argument("action", choices=tuple(actions))
    args = parser.parse_args(argv)
    result = run(args.task, args.action, load_config(args.config))
    if result is not None:
        print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    main()
