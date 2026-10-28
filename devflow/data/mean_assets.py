"""Prepare training-only SciPlex3 mean labels and immutable MLP state assets.

This command consumes explicitly supplied processed data, MAPKG features and a
frozen historical SE prefix/tail. It never imports the historical repository,
loads fitted mean coefficients, or reads external treated expression to fit a
statistic. Prepared assets are relocatable; original paths remain provenance.
"""
from __future__ import annotations

import argparse
import contextlib
import json
from pathlib import Path

import numpy as np
import torch

from devflow.data.sciplex3 import CELL_LINES, SciPlexData
from devflow.runtime.artifacts import atomic_json, ordered_sha256, sha256

SCHEMA = "sciplex3_mlp_mean_assets_v1"
DOSES = (10.0, 100.0, 1000.0, 10000.0)
# These train-drug CV choices belong to the historical mixed-kernel reference.
# They are task configuration, not hard-coded dimensions in the generic head.
HISTORICAL_SPECS = (
    (.5, 8., .01), (.5, 8., .001), (.5, 8., .001), (.5, 4., .001),
    (1., 8., .001), (1., 8., .001), (.5, 4., .001), (.5, 4., .001),
    (.5, 8., .1), (.5, 8., .01), (.5, 8., .001), (.5, 4., .01),
)


def canonical_smiles(smiles):
    from rdkit import Chem
    molecule = Chem.MolFromSmiles(str(smiles))
    if molecule is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    return Chem.MolToSmiles(molecule, canonical=True)


def morgan_fingerprints(smiles):
    from rdkit import Chem, DataStructs, rdBase
    from rdkit.Chem import AllChem
    rows = []
    for value in smiles:
        mol = Chem.MolFromSmiles(str(value))
        if mol is None:
            raise ValueError(f"Invalid SMILES: {value}")
        # Preserve the audited historical API without a warning per molecule/rank.
        with rdBase.BlockLogs():
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048, useChirality=True)
        row = np.zeros(2048, np.uint8)
        DataStructs.ConvertToNumpyArray(fp, row)
        if not row.any():
            raise ValueError("Empty Morgan fingerprint")
        rows.append(row)
    return np.stack(rows)


def read_drug_bank(path):
    bank = torch.load(str(path), map_location="cpu", weights_only=False)
    names = [canonical_smiles(s) for s in bank["canonical_smiles"]]
    values = np.asarray(bank["embeddings"], dtype=np.float32)
    if values.shape != (len(names), 1024) or not np.isfinite(values).all():
        raise ValueError("MAPKG table must have one finite 1024-vector per SMILES")
    if len(set(names)) != len(names):
        raise ValueError("MAPKG canonical SMILES are duplicated")
    return names, values


def training_mean_arrays(dataset, bank_names, bank_values):
    """Labels preserve the historical FP32 expression-mean reduction order."""
    lookup = dict(zip(bank_names, bank_values))
    drugs = sorted({str(r["drug_id"]) for r in dataset.train_records})
    strata = [(line, dose) for line in CELL_LINES for dose in DOSES]
    di, si = {v: i for i, v in enumerate(drugs)}, {v: i for i, v in enumerate(strata)}
    features = np.empty((len(drugs), bank_values.shape[1]), np.float32)
    targets = np.empty((len(strata), len(drugs), len(dataset.genes)), np.float32)
    seen = np.zeros(targets.shape[:2], bool)
    allowed = {line: set(dataset.training_cells(line, True).tolist()) for line in CELL_LINES}

    def training_expression(line, rows):
        if not set(np.asarray(rows, dtype=np.int64).tolist()) <= allowed[line]:
            raise ValueError("Nontraining expression requested while fitting mean assets")
        return dataset.expression(line, rows)

    control = {line: training_expression(line, dataset.controls(line)).mean(0)
               for line in CELL_LINES}
    smiles_by_drug = {}
    for record in dataset.train_records:
        drug = str(record["drug_id"])
        smiles = canonical_smiles(record["drug_smiles"])
        if drug in smiles_by_drug and smiles_by_drug[drug] != smiles:
            raise ValueError("One drug ID maps to multiple structures")
        smiles_by_drug[drug] = smiles
        d, s = di[drug], si[(record["cell_line"], float(record["dose_nm"]))]
        if seen[s, d]:
            raise ValueError("Duplicate training drug/stratum")
        features[d] = lookup[smiles]
        targets[s, d] = training_expression(record["cell_line"], record["rows"]).mean(0) - control[record["cell_line"]]
        seen[s, d] = True
    if not seen.all() or not np.isfinite(targets).all():
        raise ValueError("SciPlex3 training condition grid is incomplete")
    return dict(features=features, targets=targets,
                bias=targets.astype(np.float64).mean(1),
                control_mean=np.stack([control[line] for line, _ in strata]),
                drug_ids=drugs, strata=strata,
                canonical_smiles=[smiles_by_drug[d] for d in drugs])


def _encode_tail(tail, prefixes, device, chunk, bf16=False):
    outputs = []
    for start in range(0, len(prefixes), chunk):
        tensor = torch.as_tensor(np.array(prefixes[start:start + chunk], copy=True),
                                 dtype=torch.float32, device=device)
        autocast = torch.autocast(device.type, dtype=torch.bfloat16) if bf16 else contextlib.nullcontext()
        with torch.no_grad(), autocast:
            outputs.append(tail(tensor).float().cpu().numpy())
    return np.concatenate(outputs)


def prepare_assets(config, device="cuda:0"):
    from rdkit import rdBase
    from devflow.mean.kernels import MixedKernelBank
    from devflow.networks.state_encoder import SEControlLoRATail

    data = config["data"]
    output = Path(data["mlp_assets"])
    if (output / "manifest.json").exists():
        raise FileExistsError(f"Prepared assets already exist: {output}")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Use a fresh asset directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    dataset = SciPlexData(data["base_dir"], data["sampling_manifest"], data.get("source_split_root"))
    prefix_root = Path(data["prefix_cache"])
    prefix_manifest = json.loads((prefix_root / "manifest.json").read_text())
    if (prefix_manifest.get("status") != "complete"
            or prefix_manifest.get("native_attention_dropout") != .1
            or prefix_manifest.get("cache_seed") != 20261021
            or not prefix_manifest.get("frozen_prefix_realization")):
        raise ValueError("This recipe requires the audited fixed stochastic historical SE prefix")
    tail_path = Path(data["tail_checkpoint"])
    if sha256(tail_path) != prefix_manifest["tail_checkpoint_sha256"]:
        raise ValueError("SE tail does not belong to the supplied prefix")
    device = torch.device(device)
    if device.type != "cuda":
        raise ValueError("Historical BF16/FP32 source preparation requires a CUDA device")
    torch.cuda.set_device(device)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    tail = SEControlLoRATail.load_tail(tail_path).to(device).eval().requires_grad_(False)
    for name, tensor in tail.state_dict().items():
        if name.endswith("lora_B") and torch.count_nonzero(tensor):
            raise ValueError("Frozen historical tail must have zero LoRA B matrices")
    names, values = read_drug_bank(data["mapkg_embeddings"])
    means = training_mean_arrays(dataset, names, values)
    fingerprints = morgan_fingerprints(means["canonical_smiles"])
    specs = config.get("mean", {}).get("kernel_specs") or [
        dict(kg_mix=w, gamma_factor=g, alpha=a) for w, g, a in HISTORICAL_SPECS]
    if len(specs) != len(means["strata"]):
        raise ValueError("One kernel specification is required for each stratum")
    if rdBase.rdkitVersion != "2025.09.2" and not data.get("allow_rdkit_version_change", False):
        raise ValueError("Reference fingerprints require RDKit 2025.09.2; explicitly record any version change")
    bank = MixedKernelBank.fit(means["features"], fingerprints, specs)
    mean_assets = dict(kernel=bank.train_kernels, targets=means["targets"],
        bias=means["bias"], control_mean=means["control_mean"],
        alpha=np.asarray([s["alpha"] for s in specs], np.float64),
        kernelbank_assets=bank.to_assets(), metadata=dict(
            schema=SCHEMA, training_binding=dataset.training_binding(),
            gene_order_sha256=dataset.gene_order_sha256,
            gene_symbols=list(dataset.genes), anchor_drug_ids=means["drug_ids"],
            anchor_smiles=means["canonical_smiles"], strata=[list(s) for s in means["strata"]],
            kernel_specs=specs, labels="FP32 population treated mean minus FP32 train-control mean",
            fingerprint=dict(radius=2, nBits=2048, useChirality=True, rdkit_version=rdBase.rdkitVersion),
            external_expression_used=False, fitted_coefficients_included=False))
    torch.save(mean_assets, output / "mean_assets.pt")
    np.savez(output / "drug_features.npz", canonical_smiles=np.asarray(names), embeddings=values)
    line_reports = {}
    target_means = np.empty((len(dataset.train_records), 2048), np.float32)
    source_means = []
    for line in CELL_LINES:
        folder = output / line
        folder.mkdir()
        old = prefix_root / line
        info = prefix_manifest["lines"][line]
        for filename, key in (("rows.npy", "rows_sha256"), ("prefix.npy", "prefix_sha256")):
            if sha256(old / filename) != info[key]:
                raise ValueError(f"Historical prefix identity mismatch: {line}/{filename}")
        rows = np.load(old / "rows.npy", allow_pickle=False)
        prefixes = np.load(old / "prefix.npy", mmap_mode="r", allow_pickle=False)
        np.testing.assert_array_equal(rows, dataset.controls(line))
        if prefixes.shape != (len(rows), 2048) or prefixes.dtype != np.float32:
            raise ValueError("Unexpected historical control prefix shape/dtype")
        # The original training used BF16 tail calls at candidate batch 256;
        # formal prediction uses FP32 at batch 128. Keep both numerical paths.
        train_source = _encode_tail(tail, prefixes, device, 256, True)
        eval_source = _encode_tail(tail, prefixes, device, 128, False)
        mean_source = _encode_tail(tail, prefixes, device, 1024, True)
        source_means.append(mean_source.mean(0, dtype=np.float64).astype(np.float32))
        np.save(folder / "control_rows.npy", rows)
        np.save(folder / "control_global_ids.npy", dataset.global_ids(line, rows))
        np.save(folder / "source_train.npy", train_source)
        np.save(folder / "source_eval.npy", eval_source)
        state_path = dataset.base / line / f"{line}_embeddings.npy"
        states = np.load(state_path, mmap_mode="r", allow_pickle=False)
        if states.shape != (len(dataset.metadata[line]), 2048) or states.dtype != np.float32:
            raise ValueError("Original processed SE target state axis differs")
        for rec in dataset.train_records:
            if rec["cell_line"] == line:
                target_means[rec["index"]] = np.asarray(states[rec["rows"]]).mean(0, dtype=np.float64)
        files = {p.name: sha256(p) for p in folder.glob("*.npy")}
        line_reports[line] = dict(n_cells=len(states), n_controls=len(rows), files=files,
            control_row_order_sha256=ordered_sha256(rows),
            control_global_order_sha256=ordered_sha256(dataset.global_ids(line, rows)),
            metadata_sha256=sha256(dataset.base / line / f"{line}_meta.pkl"),
            target_latent_sha256=sha256(state_path), target_latent_original_path=str(state_path),
            target_latent_producer="original processed MAP SE; upstream full producer version not established",
            source_prefix_sha256=info["prefix_sha256"],
            source_fp32_bf16_max_difference=float(np.max(np.abs(train_source-eval_source))))
        print(json.dumps({"prepared_line": line, "n_controls": len(rows)}), flush=True)
    np.save(output / "condition_latent_mean.npy", target_means)
    np.save(output / "control_latent_mean.npy", np.stack(source_means))
    atomic_json(output / "conditions.json", [r["condition"] for r in dataset.train_records])
    manifest = dict(schema=SCHEMA, status="complete", training_binding=dataset.training_binding(),
        gene_order_sha256=dataset.gene_order_sha256, source_policy=dict(
            prefix_sampling="fixed_stochastic_realization", native_attention_dropout=.1,
            cache_seed=20261021, lora_B="zero_frozen", training_tail_precision="BF16",
            prediction_tail_precision="FP32", training_tail_chunk=256,
            prediction_tail_chunk=128, population_mean_tail_chunk=1024,
            allow_tf32=False, parent_prefix_manifest_sha256=sha256(prefix_root / "manifest.json"),
            tail_checkpoint_sha256=sha256(tail_path)),
        inputs=dict(base_dir=str(dataset.base), prefix_cache=str(prefix_root),
            tail_checkpoint=str(tail_path), mapkg_embeddings=str(data["mapkg_embeddings"]),
            mapkg_sha256=sha256(data["mapkg_embeddings"]),
            sampling_manifest_sha256=sha256(data["sampling_manifest"])),
        training_only_statistics=True, external_expression_used_for_fitting=False,
        lines=line_reports, files={name: sha256(output / name) for name in (
            "mean_assets.pt", "drug_features.npz", "condition_latent_mean.npy",
            "control_latent_mean.npy", "conditions.json")},
        versions=dict(torch=torch.__version__, numpy=np.__version__, rdkit=rdBase.rdkitVersion))
    atomic_json(output / "manifest.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    from devflow.config import load_config
    config = load_config(args.config)
    prepare_assets(config, args.device)


if __name__ == "__main__":
    main()
