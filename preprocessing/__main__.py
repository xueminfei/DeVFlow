"""Original data preparation and frozen-SE caching entry points."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from pathlib import Path
import sys

TASKS = ("sciplex3", "combosciplex", "norman")


def verify_sources(argv=None):
    parser = argparse.ArgumentParser(description="Verify migrated source bytes without processing data.")
    parser.parse_args(argv)
    root = Path(__file__).resolve().parent
    manifest = json.loads((root / "sources.json").read_text())
    compiled = 0
    for name, expected in manifest["files_sha256"].items():
        path = root / name
        if not path.resolve().is_relative_to(root.resolve()):
            raise ValueError("Source manifest path escapes the package: " + name)
        content = path.read_bytes()
        if hashlib.sha256(content).hexdigest() != expected:
            raise ValueError("Migrated source changed: " + name)
        if path.suffix == ".py":
            compile(content, str(path), "exec")
            compiled += 1
    print(json.dumps({"status": "passed", "verified_files": len(manifest["files_sha256"]),
                      "compiled_python_files": compiled, "raw_processing_executed": False}, indent=2))
    return 0


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__,
        epilog="Each dataset exposes its original processing steps. Prepared-data training does not run these commands.")
    parser.add_argument("task", nargs="?", choices=TASKS + ("verify",),
                        help="Dataset or source-integrity check")
    if not arguments or arguments[0] in ("-h", "--help"):
        parser.print_help()
        return 0
    task = parser.parse_args(arguments[:1]).task
    if task == "verify":
        return verify_sources(arguments[1:])
    module = importlib.import_module("preprocessing." + task)
    result = module.main(arguments[1:])
    return 0 if result is None else result


if __name__ == "__main__":
    raise SystemExit(main())
