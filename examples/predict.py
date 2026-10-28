"""Produce shared-format predictions using a prepared task JSON recipe.

Each native recipe validates its own input axes and sampling IDs, loads the
configured checkpoint, and writes pickle-free prediction archives. Use the
corresponding config template rather than inventing a common tensor layout.
"""
from __future__ import annotations

import argparse
from devflow.cli import run
from devflow.config import load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=("sciplex3", "combosciplex", "norman"))
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    return run(args.task, "predict", load_config(args.config))


if __name__ == "__main__":
    main()
