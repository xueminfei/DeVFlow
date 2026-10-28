"""Honest capability registry; no synthetic baseline results or train stubs."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json


@dataclass(frozen=True)
class Baseline:
    name: str
    tasks: tuple[str, ...]
    upstream: str
    dependency: str
    training_status: str = "external_native_implementation_required"
    prediction_adapter: str = "benchmark.baselines.export"


REGISTRY = {
    "cellflow": Baseline("CellFlow", ("sciplex3", "combosciplex", "norman19"), "https://github.com/theislab/CellFlow", "cellflow"),
    "crisp": Baseline("CRISP", ("sciplex3", "combosciplex"), "Use the exact upstream revision recorded by the experiment", "CRISP native environment"),
    "chemcpa": Baseline("chemCPA", ("sciplex3", "combosciplex"), "https://github.com/theislab/chemCPA", "chemCPA"),
    "gears": Baseline("GEARS", ("norman19",), "https://github.com/snap-stanford/GEARS", "cell-gears"),
    "scgpt": Baseline("scGPT", ("norman19",), "https://github.com/bowang-lab/scGPT", "scgpt"),
    "scdfm": Baseline("scDFM", ("sciplex3", "combosciplex", "norman19"), "Use the paper's released implementation and checkpoint revision", "scDFM native environment"),
}


def baseline_config(name: str, task: str) -> dict:
    if name not in REGISTRY:
        raise ValueError(f"Unknown baseline: {name}")
    result = REGISTRY[name]
    if task not in result.tasks:
        raise ValueError(f"{result.name} is not registered for {task}")
    return dict(**asdict(result), task=task, trained_in_this_repository=False,
                required_native_config=["upstream_revision", "training_split_hash", "gene_axis", "hyperparameters", "checkpoint_sha256"],
                scoring="Use identical locked evaluation references and archive row IDs")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", choices=tuple(REGISTRY))
    parser.add_argument("--task", choices=("sciplex3", "combosciplex", "norman19"))
    args = parser.parse_args()
    if bool(args.baseline) != bool(args.task):
        parser.error("--baseline and --task must be supplied together")
    result = baseline_config(args.baseline, args.task) if args.baseline else {key: asdict(value) for key, value in REGISTRY.items()}
    print(json.dumps(result, indent=2))
