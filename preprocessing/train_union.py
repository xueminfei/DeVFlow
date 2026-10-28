"""Build SciPlex3 train + internal_test from an existing processed dataset.

This is a new DeVFlow builder reconstructed from the 2026-10-21 union manifest.
It preserves all original arrays and split files, replaces only train_indices,
and never treats the retained internal_test indices as a validation set.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile


CELL_LINES = ("A549", "MCF7", "K562")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def index_bytes(values) -> bytes:
    """Match the existing union's sorted JSON list, spacing and final newline."""
    return (json.dumps(sorted(int(v) for v in values)) + "\n").encode("utf-8")


def load_indices(path: Path) -> set[int]:
    values = json.loads(path.read_text())
    if not isinstance(values, list):
        raise TypeError(f"Expected an index list: {path}")
    ids = {int(v) for v in values}
    if len(ids) != len(values):
        raise ValueError(f"Duplicate IDs: {path}")
    return ids


def preserve_file(source: Path, destination: Path, mode: str) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    if mode == "copy":
        shutil.copyfile(source, destination)
    else:
        destination.symlink_to(os.path.relpath(source, destination.parent))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", type=Path, required=True,
                        help="Original stage-F processed directory, before the train union")
    parser.add_argument("--out-dir", type=Path, required=True,
                        help="New, nonexistent processed directory; existing outputs are never overwritten")
    parser.add_argument("--link-mode", choices=("symlink", "copy"), default="symlink")
    parser.add_argument("--reference-manifest", type=Path,
                        help="Optional historical union manifest: require identical cells, splits and gene order")
    args = parser.parse_args(argv)
    # Data libraries load only after argument parsing; --help reads no dataset.
    import pickle
    import pandas as pd

    source = args.processed_dir.expanduser().resolve()
    output = args.out_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    if source == output or source in output.parents:
        raise ValueError("The union output must be outside the immutable source dataset")
    reference = json.loads(args.reference_manifest.read_text()) if args.reference_manifest else None
    genes = json.loads((source / "hvg_info.json").read_text())["gene_symbols"]
    if len(genes) != 2000:
        raise ValueError(f"Expected the fixed 2000-gene axis, got {len(genes)}")
    gene_hash = hashlib.sha256("\n".join(map(str, genes)).encode()).hexdigest()
    # Validate all partitions before creating an output directory.
    audits = {}
    unions = {}
    for line in CELL_LINES:
        base = source / line
        train = load_indices(base / "train_indices.json")
        internal = load_indices(base / "internal_test_indices.json")
        external = load_indices(base / "external_test_indices.json")
        controls = load_indices(base / "control_indices.json")
        partitions = (train, internal, external, controls)
        if any(a & b for i, a in enumerate(partitions) for b in partitions[i + 1:]):
            raise ValueError(f"{line}: original train/internal/external/control partitions overlap")
        with (base / f"{line}_meta.pkl").open("rb") as stream:
            meta = pd.DataFrame(pickle.load(stream))
        ids = meta["ds_level_index"].astype(int)
        if not ids.is_unique:
            raise ValueError(f"{line}: duplicate metadata cell IDs")
        available = set(ids)
        if not set.union(*partitions) <= available:
            raise ValueError(f"{line}: split refers to cells missing from metadata")
        union = train | internal
        train_rows = meta.loc[ids.isin(union)]
        test_rows = meta.loc[ids.isin(external)]
        control_rows = meta.loc[ids.isin(controls)]
        if train_rows["is_control"].astype(bool).any() or test_rows["is_control"].astype(bool).any():
            raise ValueError(f"{line}: treated split contains controls")
        if not len(control_rows) or not control_rows["is_control"].astype(bool).all():
            raise ValueError(f"{line}: invalid control pool")
        train_drugs = sorted(set(train_rows["drug_id"].astype(str)))
        external_drugs = sorted(set(test_rows["drug_id"].astype(str)))
        if set(train_drugs) & set(external_drugs):
            raise ValueError(f"{line}: external drugs overlap the training pool")
        unions[line] = index_bytes(union)
        audits[line] = dict(
            source_train_cells=len(train), source_internal_cells=len(internal),
            train_union_cells=len(union), control_cells=len(controls), external_cells=len(external),
            source_train_sha256=sha256(base / "train_indices.json"),
            source_internal_sha256=sha256(base / "internal_test_indices.json"),
            train_union_sha256=hashlib.sha256(unions[line]).hexdigest(),
            control_sha256=sha256(base / "control_indices.json"),
            external_sha256=sha256(base / "external_test_indices.json"),
            train_drugs=len(train_drugs), external_drugs=len(external_drugs),
            train_drug_ids=train_drugs, external_drug_ids=external_drugs,
            train_external_drug_overlap=[], train_external_cell_overlap=0,
            original_partitions_disjoint=True, union_contains_entire_original_internal=True,
            source_references={p.name: str(p) for p in sorted(base.iterdir())
                               if p.is_file() and p.name != "train_indices.json"},
        )
    manifest = dict(
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        producer="DeVFlow preprocessing.train_union (new portable reconstruction)",
        source_base_dir=str(source), data_base_dir=str(output),
        training_scope="original train union original internal_test; fixed original external_test drugs; internal_test filename retained as historical provenance only",
        gene_order_sha256=gene_hash, lines=audits,
        total_training_treated=sum(v["train_union_cells"] for v in audits.values()),
        total_controls=sum(v["control_cells"] for v in audits.values()),
        total_original_internal=sum(v["source_internal_cells"] for v in audits.values()),
        total_external=sum(v["external_cells"] for v in audits.values()),
        source_files_unchanged=True,
    )
    if reference is not None:
        for key in ("gene_order_sha256", "total_training_treated", "total_controls",
                    "total_original_internal", "total_external"):
            if manifest[key] != reference[key]:
                raise ValueError(f"Historical union mismatch: {key}")
        for line, audit in audits.items():
            for key, value in audit.items():
                if key != "source_references" and value != reference["lines"][line][key]:
                    raise ValueError(f"Historical union mismatch: {line}.{key}")
        manifest["reference_manifest_sha256"] = sha256(args.reference_manifest)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.building-", dir=output.parent))
    try:
        for name in ("hvg_info.json", "manifest.json"):
            preserve_file(source / name, staging / name, args.link_mode)
        for line in CELL_LINES:
            (staging / line).mkdir()
            for name in audits[line]["source_references"]:
                preserve_file(source / line / name, staging / line / name, args.link_mode)
            (staging / line / "train_indices.json").write_bytes(unions[line])
        (staging / "union_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        staging.rename(output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(json.dumps(dict(output=str(output), training_treated=manifest["total_training_treated"],
                          controls=manifest["total_controls"], external=manifest["total_external"])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
