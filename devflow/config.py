"""Portable JSON configuration with explicit environment expansion."""
from pathlib import Path
import json
import os
import re


def load_config(path):
    path = Path(path).expanduser().resolve()
    config = json.loads(path.read_text())

    def expand(value):
        if isinstance(value, str):
            value = os.path.expandvars(os.path.expanduser(value))
            if re.search(r"\$\{[^}]+\}", value):
                raise ValueError(f"Unset environment variable in {value!r}")
            return value
        if isinstance(value, dict):
            return {key: expand(item) for key, item in value.items()}
        if isinstance(value, list):
            return [expand(item) for item in value]
        return value

    return expand(config)
