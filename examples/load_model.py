"""Load a SciPlex3 MLP/attention checkpoint or another native task model."""
from __future__ import annotations

import argparse
from devflow import DeVFlow


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=("sciplex3", "combosciplex", "norman"))
    parser.add_argument("checkpoint")
    parser.add_argument("--family", choices=("mlp_mean", "attention"),
                        help="SciPlex3 model family (default: mlp_mean)")
    parser.add_argument("--tail-checkpoint", help="Required SE tail for SciPlex3 attention only")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    if args.task == "sciplex3":
        family = args.family or "mlp_mean"
        if family == "mlp_mean":
            if args.tail_checkpoint:
                parser.error("--tail-checkpoint applies only to --family attention")
            from devflow.model_mlp import MLPMeanSystem
            model = MLPMeanSystem.from_checkpoint(args.checkpoint, map_location="cpu")
        else:
            if not args.tail_checkpoint:
                parser.error("SciPlex3 attention requires --tail-checkpoint")
            model = DeVFlow.from_checkpoint(args.task, args.checkpoint,
                                           tail_checkpoint=args.tail_checkpoint)
    else:
        if args.family is not None or args.tail_checkpoint is not None:
            parser.error("--family and --tail-checkpoint apply only to SciPlex3")
        model = DeVFlow.from_checkpoint(args.task, args.checkpoint)
    model = model.to(args.device).eval().requires_grad_(False)
    print(f"{type(model).__name__}: {sum(p.numel() for p in model.parameters()):,} parameters")
    return model


if __name__ == "__main__":
    main()
