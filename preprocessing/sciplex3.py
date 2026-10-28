"""Portable launchers for the audited MAP-derived SciPlex3 pipeline.

Each stage executes in a fresh process with this repository's vendored sources.
Existing datasets are explicit inputs; this module never runs a stage on import.
"""
from __future__ import annotations

import argparse
import ast
import os
from pathlib import Path
import subprocess
import sys

_ROOT = Path(__file__).resolve().parent
_STAGES = {
    "raw": ("sciplex3/A_fetch_cell_line.py", "GEO files to disk-backed cell-by-gene CSR"),
    "check": ("sciplex3/B_watch_cell_line.py", "inspect metadata and molecular strings"),
    "split": ("sciplex3/C_uni_multi_process.py", "create the original unseen-drug split"),
    "embeddings": ("sciplex3/D_extract_se_emb.py", "extract the original frozen SE embedding cache"),
    "hvg": ("sciplex3/E_hvg_multi_celllines.py", "select HVGs from original train and controls"),
    "hvg-compat": ("hvg_compat/hvg_compat.py", "explicit optional deterministic LOESS-jitter variant"),
    "assemble": ("sciplex3/F_gemini_opt_preprocess.py", "assemble processed metadata, arrays and splits"),
    "tokens": ("sciplex3/G_extract_se_input_for_lora.py", "extract SE token arrays"),
    "control-prefix": ("control_prefix/prepare_prefix.py", "extract the native stochastic control prefix"),
}


def _stage_help(script: Path, stage: str) -> int:
    """Render the original argparse function without importing its data modules."""
    tree = ast.parse(script.read_text(), filename=str(script))
    parser_function = next(node for node in tree.body
                           if isinstance(node, ast.FunctionDef) and node.name == "parse_args")
    isolated = ast.Module(body=[parser_function], type_ignores=[])
    namespace = {"argparse": argparse}
    exec(compile(isolated, str(script), "exec"), namespace)
    previous = sys.argv
    try:
        sys.argv = [f"python -m preprocessing sciplex3 {stage}", "--help"]
        namespace["parse_args"]()
    except SystemExit as exc:
        return int(exc.code or 0)
    finally:
        sys.argv = previous
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    descriptions = "\n".join(f"  {name:16} {description}" for name, (_, description) in _STAGES.items())
    parser = argparse.ArgumentParser(
        prog="python -m preprocessing sciplex3",
        description="Audited SciPlex3 preprocessing stages; all paths are explicit.",
        epilog=descriptions + "\n  train-union      combine original train and internal_test indices",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("stage", choices=[*_STAGES, "train-union"])
    parser.add_argument("stage_args", nargs=argparse.REMAINDER,
                        help="arguments forwarded unchanged; use STAGE --help")
    if not argv or argv[0] in ("-h", "--help"):
        parser.print_help()
        return 0
    args = parser.parse_args(argv)
    forwarded = args.stage_args
    if forwarded[:1] == ["--"]:
        forwarded = forwarded[1:]
    if args.stage == "train-union":
        script = _ROOT / "train_union.py"
        vendor_root = _ROOT
    else:
        script = _ROOT / "_vendor" / _STAGES[args.stage][0]
        vendor_root = _ROOT / "_vendor" / "sciplex3"
        if args.stage == "control-prefix":
            vendor_root = _ROOT / "_vendor" / "state_encoder"
        # The compatibility wrapper adds no CLI flags; original E's help avoids
        # loading Scanpy/LOESS just to describe the available arguments.
        if args.stage == "hvg-compat" and any(x in ("-h", "--help") for x in forwarded):
            script = _ROOT / "_vendor" / _STAGES["hvg"][0]
    if not script.is_file():
        raise FileNotFoundError(f"Missing bundled preprocessing entry: {script}")
    if args.stage not in ("train-union", "control-prefix") and any(
            x in ("-h", "--help") for x in forwarded):
        return _stage_help(script, args.stage)
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONPATH"] = str(vendor_root)
    return subprocess.call([sys.executable, "-B", str(script), *forwarded], env=env)


if __name__ == "__main__":
    raise SystemExit(main())
