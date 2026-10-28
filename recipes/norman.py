"""Native Norman recipe: configuration validation, decoder/flow fitting and prediction."""
import argparse
import json
from devflow.config import load_config
from devflow.runtime.norman import validate_config, train, train_decoder, predict, export_references


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("action", choices=("validate", "train-decoder", "train", "predict", "references"))
    args = parser.parse_args(argv)
    config = load_config(args.config)
    actions = {"validate": validate_config, "train-decoder": train_decoder, "train": train, "predict": predict, "references": export_references}
    result = actions[args.action](config)
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    main()
