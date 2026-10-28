#!/usr/bin/env python3
"""Reproduce the CPU-only ComboSciPlex v3 condition-holdout data bundle.

This script prepares the independent ComboSciPlex task bundle.
Run using a CPU environment containing scanpy, anndata, pandas, numpy, scipy.
The SE state cache is deliberately not produced by this preparation step.
"""
import argparse
import hashlib
import json
import platform
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy
from scipy import sparse


EXPECTED_SIZE = 775136305
EXPECTED_MD5 = "a345084044a77e13f5d8a9247e412945"
EXPECTED_SHA256 = "56943cf1c66c0e7d6f67dcc2d9d4cb6b6f13ac01417ad8be1e58be2d1215234d"
SINGLES = {"Alvespimycin", "Dacinostat", "Dasatinib", "Givinostat", "Panobinostat", "SRT2104"}
TEST = {
    "Alvespimycin+Panobinostat", "Dasatinib+Givinostat",
    "Crizotinib+Panobinostat", "Givinostat+SRT1720", "Panobinostat+Sorafenib",
}
EXCLUDED = {"Cediranib+PCI-34051", "Cediranib+SRT3025"}


def ordered_hash(values):
    return hashlib.sha256("\n".join(map(str, values)).encode("utf-8")).hexdigest()


def hashes(path):
    result = {"md5": hashlib.md5(), "sha256": hashlib.sha256()}
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            for value in result.values():
                value.update(chunk)
    return {name: value.hexdigest() for name, value in result.items()}


def write_json(path, payload):
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--test-controls-file", required=True)
    parser.add_argument("--drug-smiles-file", required=True)
    parser.add_argument("--old-decoder-hvg-info", required=True)
    parser.add_argument("--n-hvg", type=int, default=2000)
    args = parser.parse_args()
    source = Path(args.source).resolve()
    out = Path(args.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / "counts_hvg.npy").exists():
        raise FileExistsError("Refusing to overwrite a prepared expression bundle")
    (out / "prepare_provenance.py").write_text(Path(__file__).read_text())
    source_hashes = hashes(source)
    if (source.stat().st_size != EXPECTED_SIZE or source_hashes["md5"] != EXPECTED_MD5
            or source_hashes["sha256"] != EXPECTED_SHA256):
        raise ValueError("Source checksum or byte-size differs from the audited full-gene source")
    print("Source checksum verified; loading full-gene source", flush=True)
    a = ad.read_h5ad(source)
    if a.shape != (63378, 27518):
        raise ValueError("Unexpected source shape: {}".format(a.shape))
    names = a.var_names.astype(str).tolist()
    if len(set(names)) != len(names):
        duplicated = pd.Index(names)[pd.Index(names).duplicated()].tolist()
        raise ValueError("Source gene symbols must be unique; duplicates={}".format(duplicated))
    cell_ids = a.obs_names.astype(str).tolist()
    if len(set(cell_ids)) != len(cell_ids):
        raise ValueError("Source cell IDs are not unique")
    counts = a.layers["counts"]
    counts = sparse.csr_matrix(counts, dtype=np.float32)
    if not np.isfinite(counts.data).all() or np.any(counts.data < 0):
        raise ValueError("Raw counts contain nonfinite or negative values")
    if not np.allclose(counts.data, np.rint(counts.data), atol=1e-6):
        raise ValueError("Raw counts layer is not integer-valued")
    rows = []
    for x, y in zip(a.obs["Drug1"].astype(str), a.obs["Drug2"].astype(str)):
        active = sorted(d for d in (x, y) if d.lower() != "control")
        if not active:
            condition, drug1, drug2, kind, group = "control", "control", "control", "control", "control"
        elif len(active) == 1:
            condition, drug1, drug2, kind, group = active[0], active[0], "control", "single", "single"
        elif len(active) == 2:
            drug1, drug2 = active
            condition, kind = "+".join(active), "combination"
            n_single = sum(d in SINGLES for d in active)
            group = {2: "both_monotherapy", 1: "one_monotherapy", 0: "no_monotherapy"}[n_single]
        else:
            raise ValueError("Only singleton and pair conditions are supported")
        split = "test" if condition in TEST else "excluded" if condition in EXCLUDED else "train"
        rows.append((condition, drug1, drug2, kind, group, split))
    obs = pd.DataFrame(rows, columns=["condition", "drug1", "drug2", "kind", "monotherapy_group", "split"])
    obs.insert(0, "cell_id", cell_ids)
    obs["source_row"] = np.arange(a.n_obs)
    obs["source_condition"] = a.obs["condition"].astype(str).to_numpy()
    obs["is_control"] = obs["kind"].eq("control")
    obs["cell_line"] = "A549"
    obs["replicate"] = "merged"
    for source_col in ("Well", "well", "RT_well", "sample"):
        if source_col in a.obs:
            obs[source_col] = a.obs[source_col].astype(str).to_numpy()
    test_control_ids = json.loads(Path(args.test_controls_file).read_text())
    if len(test_control_ids) != 145 or len(set(test_control_ids)) != 145:
        raise ValueError("Expected exactly 145 unique held-out legacy control IDs")
    controls = set(obs.loc[obs.is_control, "cell_id"])
    if not set(test_control_ids).issubset(controls):
        raise ValueError("Held-out control IDs are not all source control cells")
    obs.loc[obs.cell_id.isin(test_control_ids), "split"] = "test"
    observed_singles = set(obs.loc[obs.kind.eq("single"), "condition"])
    if observed_singles != SINGLES:
        raise ValueError("Monotherapy vocabulary differs from audited six drugs")
    combo_condition_sets = {split: set(obs.loc[obs.kind.eq("combination") & obs.split.eq(split), "condition"])
                            for split in ("train", "test", "excluded")}
    if {k: len(v) for k, v in combo_condition_sets.items()} != {"train": 18, "test": 5, "excluded": 2}:
        raise ValueError("Unexpected combination condition allocation")
    if combo_condition_sets["test"] != TEST or combo_condition_sets["excluded"] != EXCLUDED:
        raise ValueError("Fixed holdout/exclusion lists differ")
    train_vocab = set(obs.loc[obs.split.eq("train"), ["drug1", "drug2"]].to_numpy().ravel()) - {"control"}
    test_vocab = set(obs.loc[obs.split.eq("test"), ["drug1", "drug2"]].to_numpy().ravel()) - {"control"}
    if not test_vocab.issubset(train_vocab):
        raise ValueError("Test contains constituent drugs missing from every training condition")
    library = np.asarray(counts.sum(axis=1)).ravel()
    obs["library_size_all_genes"] = library
    train_mask = obs.split.eq("train").to_numpy()
    print("Fitting {} HVGs on {} train cells only".format(args.n_hvg, int(train_mask.sum())), flush=True)
    fit = ad.AnnData(X=counts[train_mask].copy(), var=pd.DataFrame(index=names))
    sc.pp.normalize_total(fit, target_sum=10000)
    sc.pp.log1p(fit)
    sc.pp.highly_variable_genes(fit, n_top_genes=args.n_hvg, flavor="seurat", inplace=True)
    indices = np.flatnonzero(fit.var["highly_variable"].to_numpy())
    if len(indices) != args.n_hvg:
        raise ValueError("HVG tie behavior returned {} genes; expected {}".format(len(indices), args.n_hvg))
    genes = [names[int(i)] for i in indices]
    gene_meta = pd.DataFrame({"gene_index": np.arange(len(genes)), "gene_symbol": genes,
                              "source_gene_row": indices, "ensembl_id": a.var["id"].astype(str).iloc[indices].to_numpy()})
    gene_meta.to_csv(out / "genes.csv", index=False)
    write_json(out / "genes.json", genes)
    write_json(out / "hvg_fit_statistics.json", {
        "flavor": "seurat", "n_top_genes": args.n_hvg,
        "normalization": "all_gene_total_10000_log1p", "fit_n_cells": int(train_mask.sum()),
        "fit_cell_order_sha256": ordered_hash(obs.loc[train_mask, "cell_id"]),
        "source_gene_indices": indices.tolist(),
    })
    del fit
    print("Saving raw counts with {} source rows and {} HVGs".format(a.n_obs, len(genes)), flush=True)
    destination = out / "counts_hvg.npy"
    expression = np.lib.format.open_memmap(destination, mode="w+", dtype=np.float32,
                                         shape=(a.n_obs, len(genes)))
    for start in range(0, a.n_obs, 512):
        stop = min(start + 512, a.n_obs)
        expression[start:stop] = counts[start:stop, indices].toarray()
    expression.flush()
    del expression
    obs.to_csv(out / "obs.csv", index=False)
    smiles = json.loads(Path(args.drug_smiles_file).read_text())
    vocabulary = sorted(set(obs[["drug1", "drug2"]].to_numpy().ravel()) - {"control"})
    if not set(vocabulary).issubset(smiles):
        raise ValueError("SMILES missing for {}".format(sorted(set(vocabulary).difference(smiles))))
    write_json(out / "drug_smiles.json", {drug: smiles[drug] for drug in vocabulary})
    condition_rows = []
    for condition, frame in obs.groupby("condition", sort=True):
        condition_rows.append({"condition": condition, "drug1": frame.drug1.iloc[0],
            "drug2": frame.drug2.iloc[0], "kind": frame.kind.iloc[0],
            "monotherapy_group": frame.monotherapy_group.iloc[0], "n_cells": len(frame),
            "train_cells": int(frame.split.eq("train").sum()),
            "test_cells": int(frame.split.eq("test").sum()),
            "excluded_cells": int(frame.split.eq("excluded").sum())})
    pd.DataFrame(condition_rows).to_csv(out / "conditions.csv", index=False)
    old = json.loads(Path(args.old_decoder_hvg_info).read_text())
    old_genes = old["gene_symbols"]
    overlap = sorted(set(old_genes) & set(genes))
    split_counts = {str(k): int(v) for k, v in obs.split.value_counts().items()}
    cell_groups = obs.groupby(["split", "kind", "monotherapy_group"]).size()
    audit = {
        "source_shape": list(a.shape), "unique_gene_symbols": True,
        "unique_cell_ids": True, "raw_count_values_are_finite_nonnegative_integers": True,
        "split_counts": split_counts,
        "cells_by_split_kind_monotherapy_group": [dict(split=k[0], kind=k[1], monotherapy_group=k[2], n_cells=int(n)) for k, n in cell_groups.items()],
        "combination_condition_counts": {k: len(v) for k, v in combo_condition_sets.items()},
        "combination_conditions": {k: sorted(v) for k, v in combo_condition_sets.items()},
        "train_test_combination_overlap": [], "all_single_cells_in_train": bool(obs.loc[obs.kind.eq("single"), "split"].eq("train").all()),
        "train_controls": int((obs.is_control & obs.split.eq("train")).sum()),
        "test_controls": int((obs.is_control & obs.split.eq("test")).sum()),
        "control_cell_overlap": [], "control_split_policy": "reuse_gpu0_v1_145_test_cell_ids_all_remaining_controls_train",
        "train_constituent_drugs": sorted(train_vocab), "test_constituent_drugs": sorted(test_vocab),
        "test_drugs_absent_from_training": sorted(test_vocab - train_vocab),
        "monotherapy_drugs": sorted(SINGLES), "all_drugs": vocabulary,
        "hvg_gene_count": len(genes), "old_decoder_gene_count": len(old_genes),
        "old_decoder_gene_overlap_count": len(overlap), "old_decoder_gene_overlap": overlap,
        "zero_all_gene_library_cells": int((library == 0).sum()),
        "state_embeddings_present": False,
    }
    write_json(out / "split_audit.json", audit)
    manifest = {
        "format_version": 1, "split_protocol": "combosciplex_combo_holdout_no_validation_v3",
        "created_at": datetime.now(timezone.utc).isoformat(), "seed": 42,
        "source": str(source), "source_url": "https://exampledata.scverse.org/pertpy/combosciplex.h5ad",
        "source_doi": "10.6084/m9.figshare.25062230.v1", "source_size_bytes": source.stat().st_size,
        "source_md5": source_hashes["md5"], "source_sha256": source_hashes["sha256"],
        "n_cells": a.n_obs, "n_source_genes": a.n_vars, "n_genes": len(genes), "state_dim": 2048,
        "cell_order_sha256": ordered_hash(cell_ids), "gene_order_sha256": ordered_hash(genes),
        "hvg_fit_scope": "train_only", "hvg_fit_cell_order_sha256": ordered_hash(obs.loc[train_mask, "cell_id"]),
        "hvg_fit_normalization": "all_gene_total_10000_log1p", "hvg_flavor": "seurat", "hvg_gene_order": "source_gene_order",
        "expression_input": "raw_counts", "expression_normalization": "hvg_total_1000_log1p",
        "expression_file": "counts_hvg.npy", "expression_dtype": "float32", "expression_shape": [a.n_obs, len(genes)],
        "genes_file": "genes.json", "obs_file": "obs.csv", "drug_smiles_file": "drug_smiles.json",
        "state_encoder": "MAP_original_SE600M_embedding_2048", "state_embeddings_file": "se_embeddings.npy",
        "state_embeddings_present": False,
        "missing_requirements": ["Frozen original SE600M 2048D embeddings for every source cell, aligned to obs.csv cell order"],
        "split_counts": split_counts, "monotherapy_drugs": sorted(SINGLES),
        "test_conditions": sorted(TEST), "excluded_conditions": sorted(EXCLUDED),
        "train_constituent_drugs": sorted(train_vocab), "test_constituent_drugs": sorted(test_vocab),
        "validation_conditions": [], "n_train_combination_conditions": 18,
        "control_policy": "1306_train_145_test_reused_v1_test_ids", "replicate_policy": "pooled_controls_ignore_condition_specific_well",
        "old_decoder_hvg_info": str(Path(args.old_decoder_hvg_info).resolve()),
        "old_decoder_gene_overlap_count": len(overlap),
        "provenance_script": "prepare_provenance.py",
        "software": {"python": platform.python_version(), "anndata": ad.__version__, "scanpy": sc.__version__,
                     "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__},
    }
    write_json(out / "manifest.json", manifest)
    print(json.dumps({"out_dir": str(out), "split_counts": split_counts,
        "n_hvg": len(genes), "old_decoder_overlap": len(overlap),
        "cell_order_sha256": manifest["cell_order_sha256"], "gene_order_sha256": manifest["gene_order_sha256"]}), flush=True)


if __name__ == "__main__":
    main()
