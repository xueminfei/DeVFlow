"""Export frozen scoring inputs without changing their statistics or gene panel.

This utility reads treated test expression only for evaluation reference output.
It does not create any training artifact, select a checkpoint, or run a model.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from devflow.io import json_sha256, sha256_file, write_json


def _write_npz(path, payload):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as stream:
        np.savez_compressed(stream, **payload)
    tmp.replace(path)


def write_reference_bundle(directory, *, statistics, samples, training_provenance,
                           evaluation_provenance, source_hashes=None):
    """Generic task exporter; matrices must already follow the locked gene axis.

Each samples[seed] is a dict containing seed, condition_names,
source_global_ids, target_global_ids, gene_order_sha256, target_0000... and
control_0000... . Reference hashes bind the exact row order and expression.
"""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "manifest.json").exists():
        raise FileExistsError("Refusing to overwrite an existing reference bundle")
    _write_npz(directory / "statistics.npz", statistics)
    names = ["statistics.npz"]
    for seed, payload in samples.items():
        name = f"references_seed_{int(seed)}.npz"
        _write_npz(directory / name, payload)
        names.append(name)
    manifest = dict(schema_version=1, purpose="evaluation_only",
                    training_statistics=dict(training_provenance),
                    evaluation_statistics=dict(evaluation_provenance),
                    source_sha256=dict(source_hashes or {}),
                    files_sha256={name: sha256_file(directory / name) for name in names})
    write_json(directory / "manifest.json", manifest)
    return manifest


def export_sciplex3(dataset, legacy_references, output, *, legacy_manifest=None,
                    de50_reference=None, protocol_template=None):
    """Preserve union-training means, DEG weights, DE masks, fixed DE50 order.

``dataset`` is devflow.data.sciplex3.SciPlexData. ``legacy_manifest`` is the
small provenance JSON beside the original evaluation-only references.npz.
The original panel is supplied explicitly, so no sampled-test DEG fitting runs.
"""
    from devflow.data.sciplex3 import CELL_LINES, SEEDS

    legacy_references = Path(legacy_references)
    legacy_manifest = Path(legacy_manifest or legacy_references.with_name("manifest.json"))
    provenance = json.loads(legacy_manifest.read_text())
    if provenance.get("reference_sha256") != sha256_file(legacy_references):
        raise ValueError("Original evaluation reference hash mismatch")
    binding = dataset.training_binding()
    if provenance.get("training_binding") != binding:
        raise ValueError("Original evaluation references use a different training split")
    names = [record["condition"] for record in dataset.test_records]
    with np.load(legacy_references, allow_pickle=False) as values:
        if values["condition_names"].tolist() != names or str(values["gene_order_sha256"]) != dataset.gene_order_sha256:
            raise ValueError("Original evaluation condition/gene order mismatch")
        statistics = {key: np.array(values[key]) for key in
                      ("condition_names", "gene_order_sha256", "continuous_deg_weights", "de_mask", "training_condition_mean")}
    contexts = [record["cell_line"] for record in dataset.test_records]
    statistics.update(context_names=np.asarray(contexts),
                      reference_index=np.asarray([CELL_LINES.index(line) for line in contexts], dtype=np.int64))
    hashes = {"legacy_references": sha256_file(legacy_references), "legacy_manifest": sha256_file(legacy_manifest),
              "sampling_manifest": sha256_file(dataset.sampling_manifest_path)}
    if de50_reference is not None:
        de50_reference = Path(de50_reference)
        with np.load(de50_reference, allow_pickle=False) as panel:
            if panel["group_names"].tolist() != names:
                raise ValueError("Fixed DE50 condition order mismatch")
            np.testing.assert_array_equal(panel["full_weights"], statistics["continuous_deg_weights"])
            indices = np.array(panel["gene_indices"])
            if indices.shape != (108, 50) or indices.dtype.kind not in "iu" or (indices < 0).any() or (indices >= 2000).any():
                raise ValueError("Invalid fixed DE50 gene indices")
            if any(len(set(row)) != 50 for row in indices):
                raise ValueError("Duplicate fixed DE50 gene indices")
            statistics["de50_indices"] = indices
        hashes["fixed_de50_reference"] = sha256_file(de50_reference)
    samples = {}
    for seed in SEEDS:
        sampled = dataset.sampling(seed)
        values = dict(seed=np.asarray(seed), condition_names=np.asarray(names),
                      source_global_ids=np.stack([r["control_global_ids"] for r in sampled]),
                      target_global_ids=np.stack([r["treated_global_ids"] for r in sampled]),
                      gene_order_sha256=np.asarray(dataset.gene_order_sha256))
        for index, record in enumerate(sampled):
            values[f"target_{index:04d}"] = dataset.expression(record["cell_line"], record["target_rows"])
            values[f"control_{index:04d}"] = dataset.expression(record["cell_line"], record["control_rows"])
        samples[seed] = values
    manifest = write_reference_bundle(output, statistics=statistics, samples=samples,
        training_provenance=dict(fitted_on="training_only", use="DeltaPert reference only",
                                 fit_row_ids_sha256=json_sha256(binding["train_rows_sha256"]),
                                 training_binding=binding,
                                 averaging="equal weight across all 716 training conditions per cell line"),
        evaluation_provenance=dict(use="reporting_only", test_derived=["continuous_deg_weights", "de_mask", "de50_indices"],
                                   frozen=True, panel_reselected=False), source_hashes=hashes)
    template = Path(protocol_template) if protocol_template else Path(__file__).parent / "protocols/sciplex3_exact108.json"
    protocol = json.loads(template.read_text())
    protocol.update(gene_order_sha256=dataset.gene_order_sha256, training_binding_sha256=json_sha256(binding))
    write_json(Path(output) / "protocol.json", protocol)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", required=True)
    parser.add_argument("--sampling-manifest", required=True)
    parser.add_argument("--source-split-root", required=True)
    parser.add_argument("--legacy-references", required=True)
    parser.add_argument("--legacy-manifest")
    parser.add_argument("--de50-reference", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    from devflow.data.sciplex3 import SciPlexData
    dataset = SciPlexData(args.base_dir, args.sampling_manifest, source_split_root=args.source_split_root)
    export_sciplex3(dataset, args.legacy_references, args.output, legacy_manifest=args.legacy_manifest,
                    de50_reference=args.de50_reference)


if __name__ == "__main__":
    main()
