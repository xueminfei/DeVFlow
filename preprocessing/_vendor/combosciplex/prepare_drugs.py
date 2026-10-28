#!/usr/bin/env python3
"""Prepare the original frozen MAP-KG molecule bank without importing a trainer.

The canonical SMILES ordering, encoder call and tensor reorder match
MAP/train_flowmatching.py:build_drug_bank. All paths are explicit.
"""
import argparse
import csv
import json
import os
from datetime import datetime, timezone
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--vocabulary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(args.output)
    import torch
    from flowmatching.drug import canonical_smiles, encode_mapkg_smiles
    drug_smiles = json.loads((args.bundle_dir / "drug_smiles.json").read_text())
    names = set()
    with (args.bundle_dir / "obs.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["split"] not in ("train", "test"):
                continue
            names.update(value for value in (row.get("drug1"), row.get("drug2")) if value and value in drug_smiles)
    if not names:
        raise ValueError("No constituent drugs in train/test bundle rows")
    canonical = sorted(set(canonical_smiles(drug_smiles[name]) for name in names))
    encoded_smiles, embeddings = encode_mapkg_smiles(canonical, str(args.checkpoint), str(args.vocabulary), torch.device(args.device), batch_size=args.batch_size)
    order = {value: index for index, value in enumerate(encoded_smiles)}
    embeddings = torch.stack([embeddings[order[value]] for value in canonical])
    if embeddings.shape != (len(canonical), 1024) or not torch.isfinite(embeddings).all():
        raise ValueError("Invalid original MAP-KG molecule embeddings")
    payload = {"format_version": 1, "created_at": datetime.now(timezone.utc).isoformat(), "canonical_smiles": canonical, "embeddings": embeddings,
               "mapkg_checkpoint": str(args.checkpoint.resolve()), "bart_vocabulary": str(args.vocabulary.resolve()), "dose_encoding": "paper_scale_log10_1_plus_nM"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".partial")
    if temporary.exists():
        raise FileExistsError(temporary)
    torch.save(payload, temporary)
    # Exclusive hard link ensures that a concurrent producer cannot be overwritten.
    os.link(temporary, args.output)
    temporary.unlink()


if __name__ == "__main__":
    main()
