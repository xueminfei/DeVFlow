"""Read evaluation-only references and exact shared treated/control samples.

No model training code should import this module or receive this directory.
Training-only statistics for loss construction belong in a separate directory.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from devflow.io import PredictionArchive, json_sha256, require_alignment, sha256_file
from .protocol import EvaluationProtocol


class EvaluationReferences:
    def __init__(self, directory: str | Path, protocol: EvaluationProtocol):
        self.directory = Path(directory)
        self.protocol = protocol
        self.seed_statistics = {}
        self.manifest_path = self.directory / "manifest.json"
        self.manifest = json.loads(self.manifest_path.read_text())
        if self.manifest.get("purpose") != "evaluation_only":
            raise ValueError("References must explicitly declare purpose=evaluation_only")
        train = self.manifest.get("training_statistics", {})
        evaluation = self.manifest.get("evaluation_statistics", {})
        if train.get("fitted_on") != "training_only" or not train.get("fit_row_ids_sha256"):
            raise ValueError("DeltaPert training reference requires training-only fit-row provenance")
        self.training_binding_sha256 = train.get("training_binding_sha256")
        if "training_binding" in train:
            content_hash = json_sha256(train["training_binding"])
            if self.training_binding_sha256 and content_hash != self.training_binding_sha256:
                raise ValueError("Reference training binding hash mismatch")
            self.training_binding_sha256 = content_hash
        if protocol.training_binding_sha256 and self.training_binding_sha256 != protocol.training_binding_sha256:
            raise ValueError("Reference training binding differs from protocol")
        if evaluation.get("use") != "reporting_only":
            raise ValueError("Test-derived DEG weights/panels must be labeled reporting_only")
        hashes = self.manifest.get("files_sha256", {})
        expected = {"statistics.npz"} | {f"references_seed_{seed}.npz" for seed in protocol.seeds}
        if not expected <= set(hashes):
            raise ValueError("Reference manifest must hash statistics and every sampled reference archive")
        for name in expected:
            if sha256_file(self.directory / name) != hashes[name]:
                raise ValueError(f"Reference hash mismatch: {name}")
        with np.load(self.directory / "statistics.npz", allow_pickle=False) as data:
            self.statistics = {key: np.array(data[key]) for key in data.files}
        values = self.statistics
        self.condition_names = tuple(map(str, values["condition_names"].tolist()))
        if len(self.condition_names) != protocol.condition_count or len(set(self.condition_names)) != len(self.condition_names):
            raise ValueError("Reference condition count or uniqueness mismatch")
        self.gene_order_sha256 = str(values["gene_order_sha256"])
        if protocol.gene_order_sha256 and self.gene_order_sha256 != protocol.gene_order_sha256:
            raise ValueError("Reference gene hash differs from protocol")
        for name in ("continuous_deg_weights", "de_mask"):
            if values[name].shape != (protocol.condition_count, protocol.n_genes):
                raise ValueError(f"Wrong {name} shape")
        weights = values["continuous_deg_weights"]
        if not np.isfinite(weights).all() or (weights < 0).any():
            raise ValueError("DEG weights must be finite and nonnegative")
        means = values["training_condition_mean"]
        indices = values["reference_index"]
        if means.ndim != 2 or means.shape[1] != protocol.n_genes or not np.isfinite(means).all():
            raise ValueError("Wrong training reference shape or nonfinite values")
        if indices.shape != (protocol.condition_count,) or indices.dtype.kind not in "iu" or (indices < 0).any() or (indices >= len(means)).any():
            raise ValueError("Invalid training reference indices")
        if values["context_names"].shape != (protocol.condition_count,):
            raise ValueError("Each condition requires a retrieval context")
        if "reporting_scopes" in values and values["reporting_scopes"].shape != (protocol.condition_count,):
            raise ValueError("Reporting scopes must assign one category per condition")
        if "de50_indices" in values:
            panel = values["de50_indices"]
            if panel.shape != (protocol.condition_count, protocol.de50_size) or panel.dtype.kind not in "iu":
                raise ValueError("Wrong fixed DE panel shape or dtype")
            if (panel < 0).any() or (panel >= protocol.n_genes).any() or any(len(set(row)) != len(row) for row in panel):
                raise ValueError("DE panel indices must be unique and within the gene axis")
        if protocol.response_profile == "full_treated":
            profiles = values.get("full_target_profiles")
            if profiles is None or profiles.shape != (protocol.condition_count, protocol.n_genes) or not np.isfinite(profiles).all():
                raise ValueError("Full-treated protocol requires frozen full_target_profiles")

    def for_seed(self, seed: int, archive: PredictionArchive):
        if seed not in self.protocol.seeds:
            raise ValueError("Seed is outside the locked protocol")
        path = self.directory / f"references_seed_{seed}.npz"
        with np.load(path, allow_pickle=False) as data:
            if int(data["seed"]) != seed or tuple(map(str, data["condition_names"].tolist())) != self.condition_names:
                raise ValueError("Reference seed or condition order mismatch")
            if str(data["gene_order_sha256"]) != self.gene_order_sha256:
                raise ValueError("Reference gene order mismatch")
            source, target_ids = np.array(data["source_global_ids"]), np.array(data["target_global_ids"])
            shape = (self.protocol.condition_count, self.protocol.cells_per_condition)
            if source.shape != shape or target_ids.shape != shape:
                raise ValueError("Reference row ID shape differs from protocol")
            require_alignment(archive, seed=seed, condition_names=self.condition_names,
                              source_global_ids=source, target_global_ids=target_ids,
                              gene_order_sha256=self.gene_order_sha256, n_genes=self.protocol.n_genes)
            self.seed_statistics[seed] = {}
            if "de_mask" in data:
                mask = np.array(data["de_mask"])
                if mask.shape != (self.protocol.condition_count, self.protocol.n_genes):
                    raise ValueError("Seed-specific DE mask shape mismatch")
                self.seed_statistics[seed]["de_mask"] = mask
            targets = tuple(np.array(data[f"target_{i:04d}"], dtype=np.float32) for i in range(len(self.condition_names)))
            controls = tuple(np.array(data[f"control_{i:04d}"], dtype=np.float32) for i in range(len(self.condition_names)))
        for matrix in (*targets, *controls):
            if matrix.shape != (self.protocol.cells_per_condition, self.protocol.n_genes) or not np.isfinite(matrix).all():
                raise ValueError("Sampled reference expression shape or finite-value mismatch")
        return targets, controls
