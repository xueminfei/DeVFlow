"""Adapt a native NPZ archive with explicit existing sampling identities.

This tool never invents IDs from references. If the producer did not save row
IDs, re-export predictions in its native inference code with those IDs attached.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from devflow.io import sha256_file, write_prediction_archive


def export_native(input_path, output_path, *, prediction_prefix="prediction_", source_ids_key="source_global_ids",
                  target_ids_key="target_global_ids", condition_names_key="condition_names", seed_key="seed",
                  gene_hash_key="gene_order_sha256", metadata=None):
    with np.load(input_path, allow_pickle=False) as values:
        required = [source_ids_key, target_ids_key, condition_names_key, seed_key, gene_hash_key]
        missing = set(required) - set(values.files)
        if missing:
            raise ValueError(f"Native export is missing sampling identities: {sorted(missing)}; re-export from native inference")
        names = values[condition_names_key].astype(str).tolist()
        predictions = [np.array(values[f"{prediction_prefix}{index:04d}"]) for index in range(len(names))]
        write_prediction_archive(output_path, seed=int(values[seed_key]), condition_names=names,
                                 source_global_ids=np.array(values[source_ids_key]), target_global_ids=np.array(values[target_ids_key]),
                                 gene_order_sha256=str(values[gene_hash_key]), predictions=predictions,
                                 metadata=dict(native_archive_sha256=sha256_file(input_path),
                                               adapter="explicit_existing_ids", **(metadata or {})))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prediction-prefix", default="prediction_")
    parser.add_argument("--source-ids-key", default="source_global_ids")
    parser.add_argument("--target-ids-key", default="target_global_ids")
    parser.add_argument("--condition-names-key", default="condition_names")
    parser.add_argument("--seed-key", default="seed")
    parser.add_argument("--gene-hash-key", default="gene_order_sha256")
    parser.add_argument("--metadata", type=Path, help="JSON native provenance, including the actual training_binding")
    args = parser.parse_args()
    export_native(args.input, args.output, prediction_prefix=args.prediction_prefix,
                  source_ids_key=args.source_ids_key, target_ids_key=args.target_ids_key,
                  condition_names_key=args.condition_names_key, seed_key=args.seed_key, gene_hash_key=args.gene_hash_key,
                  metadata=json.loads(args.metadata.read_text()) if args.metadata else None)


if __name__ == "__main__":
    main()
