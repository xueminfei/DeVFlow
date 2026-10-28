"""Portable entry points for the evaluated Norman additive split and frozen features."""
from __future__ import annotations
import json
import os
from pathlib import Path
import subprocess
import sys

PREPROCESSING = Path(__file__).resolve().parent
VENDOR = PREPROCESSING / "_vendor"


def _value(args, flag, default=None):
    result = default
    for i, value in enumerate(args):
        if value == flag and i + 1 < len(args):
            result = args[i + 1]
        elif value.startswith(flag + "="):
            result = value.split("=", 1)[1]
    return result


def _refuse(paths):
    for path in paths:
        if path.exists():
            raise SystemExit("Refusing to replace existing preprocessing output: " + str(path))


def _run(script, args, roots):
    # Each historical namespace runs in its own process, with repository-local sources.
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(str(root) for root in roots)
    return subprocess.call([sys.executable, str(script), *args], env=env)


_HELP = {
    "prepare": "--h5ad H5AD --output-dir NEW_DIR [--fold all --manifest-h5ad-path PATH --control-seed 4242 --control-train-fraction 0.75 --n-base-hvg 2000 --se-embeddings NPY]",
    "prepare-se": "--h5ad H5AD --condition-manifest JSON --gene-embeddings PT --output-dir NEW_DIR [--seed 42 --control-train-fraction 0.75 --n-hvg 2000 --log-every 10000]",
    "encode": "--input-dir DIR --checkpoint CKPT --shard-index N [--output-dir NEW_DIR --se-config YAML --device cuda:0 --batch-size 8 --num-shards 8 --seed 42 --log-every 200 --amp-dtype bf16]",
    "merge": "--input-dir DIR [--output-dir NEW_DIR --num-shards 8]",
    "gene-assets": "--h5ad H5AD --split-manifest JSON --control-split JSON --esm-checkpoint PT --se-checkpoint CKPT --projector-checkpoint PT --output-dir NEW_DIR [--device cuda:0]",
}


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in ("-h", "--help"):
        print("Usage: python -m preprocessing norman STEP [original flags]\nSteps: " + ", ".join(_HELP) + "\nprepare writes official additive split/HVG metadata and reuses an existing SE cache.\nprepare-se reproduces the original all-cell cache producer, whose explicit condition manifest is the historical 201/25 double-pair split.\nextract-se is an alias followed by prepare, encode or merge.")
        return 0
    step = args.pop(0)
    if step == "prepare-additive":
        step = "prepare"
    if step == "extract-se":
        if not args or args[0] in ("-h", "--help"):
            print("Usage: python -m preprocessing norman extract-se {prepare,encode,merge} [original flags]")
            return 0
        step = args.pop(0)
        if step == "prepare":
            step = "prepare-se"
        elif step not in ("encode", "merge"):
            raise SystemExit("Unknown Norman SE extraction step: " + step)
    if step not in _HELP:
        raise SystemExit("Unknown Norman preprocessing step: " + step)
    if any(arg in ("-h", "--help") for arg in args):
        print("Usage: python -m preprocessing norman " + step + " " + _HELP[step])
        return 0
    task = VENDOR / "norman" / "dependencies"
    scripts = task / "scripts"
    roots = [task, scripts, VENDOR / "state_encoder"]
    output = _value(args, "--output-dir")
    if step in ("prepare", "prepare-se", "gene-assets") and output:
        path = Path(output)
        if path.exists() and any(path.iterdir()):
            raise SystemExit("Choose an empty --output-dir; completed artifacts are immutable: " + output)
    if step == "prepare":
        return _run(scripts / "prepare_norman_additive.py", args, roots)
    if step == "gene-assets":
        return _run(scripts / "prepare_norman_gene_assets.py", args, roots)
    if step in ("encode", "merge"):
        output = output or _value(args, "--input-dir")
        if output:
            path = Path(output)
            if step == "merge":
                _refuse([path / "se_embeddings.npy", path / "state_embeddings.json"])
            else:
                index, shards = _value(args, "--shard-index"), _value(args, "--num-shards", "8")
                if index is not None:
                    stem = f"state_shard_{int(index):02d}_of_{int(shards):02d}"
                    _refuse([path / (stem + ".npy"), path / (stem + ".json")])
    original_step = "prepare" if step == "prepare-se" else step
    return _run(scripts / "prepare_norman_decoder.py", [original_step, *args], roots)


if __name__ == "__main__":
    raise SystemExit(main())
