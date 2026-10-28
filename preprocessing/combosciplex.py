"""Portable entry points for the historical ComboSciPlex preprocessing pipeline."""
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
    "prepare": "--source H5AD --out-dir NEW_DIR --test-controls-file JSON --drug-smiles-file JSON --old-decoder-hvg-info JSON [--n-hvg 2000]",
    "prepare-tokens": "--bundle-dir DIR --source-h5ad H5AD --gene-embeddings PT [--seed 42]",
    "encode": "--bundle-dir DIR --checkpoint CKPT --shard-index N [--se-config YAML --device cuda:0 --batch-size 4 --num-shards 8 --limit-cells N --amp-dtype fp16]",
    "merge": "--bundle-dir DIR [--num-shards 8]",
    "drug-embeddings": "--bundle-dir DIR --checkpoint PT --vocabulary TXT --output PT [--device cuda:0 --batch-size 64]",
}


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in ("-h", "--help"):
        print("Usage: python -m preprocessing combosciplex STEP [original flags]\nSteps: " + ", ".join(_HELP) + "\nextract-se is an alias followed by prepare-tokens, encode or merge.")
        return 0
    step = args.pop(0)
    if step == "extract-se":
        if not args or args[0] in ("-h", "--help"):
            print("Usage: python -m preprocessing combosciplex extract-se {prepare-tokens,encode,merge} [original flags]")
            return 0
        step = args.pop(0)
        if step not in ("prepare-tokens", "encode", "merge"):
            raise SystemExit("Unknown SE extraction step: " + step)
    if step not in _HELP:
        raise SystemExit("Unknown ComboSciPlex preprocessing step: " + step)
    if any(arg in ("-h", "--help") for arg in args):
        print("Usage: python -m preprocessing combosciplex " + step + " " + _HELP[step])
        return 0
    task = VENDOR / "combosciplex"
    if step == "prepare":
        output = _value(args, "--out-dir")
        if output and Path(output).exists() and any(Path(output).iterdir()):
            raise SystemExit("Choose an empty --out-dir; prepared bundles are immutable: " + output)
        return _run(task / "prepare.py", args, [task / "dependencies", VENDOR / "state_encoder"])
    if step == "drug-embeddings":
        output = _value(args, "--output")
        if output:
            _refuse([Path(output)])
        return _run(task / "prepare_drugs.py", args, [task / "dependencies", VENDOR / "state_encoder"])
    bundle = _value(args, "--bundle-dir")
    if bundle:
        bundle = Path(bundle)
        if step == "prepare-tokens":
            _refuse([bundle / "se_cache" / name for name in ("token_ids.npy", "token_counts.npy", "tokenization.json")])
        if step == "merge":
            _refuse([bundle / "se_embeddings.npy", bundle / "se_cache_audit.json"])
        if step == "encode":
            index, shards = _value(args, "--shard-index"), _value(args, "--num-shards", "8")
            if index is not None:
                path = bundle / "se_cache" / (f"shard_{int(index):02d}_of_{int(shards):02d}.npy")
                meta = path.with_suffix(".json")
                # Original encoder can resume only an incomplete, provenance-matched shard.
                if path.exists() and (not meta.exists() or json.loads(meta.read_text()).get("complete")):
                    _refuse([path])
    return _run(task / "extract_se.py", [step, *args], [VENDOR / "state_encoder"])


if __name__ == "__main__":
    raise SystemExit(main())
