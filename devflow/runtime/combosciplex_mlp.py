"""Original Combo split/OT sampling with shared mean and decoder-MMD batches."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from devflow.data.combo_mean_assets import (SCHEMA, input_identities, paired_features,
    validate_kernel_mode, validate_selection_mode)
from devflow.data.combosciplex import (CELL_LINES, ComboDataset, load_bundle, training_binding)
from devflow.flow.matching import pair_population
from devflow.mean.kernels import MixedKernelBank
from devflow.runtime.artifacts import sha256


class ComboMLPData(ComboDataset):
    """Add the shared trainer's metadata without replacing the native dataset."""

    def __init__(self, config, bundle):
        super().__init__(CELL_LINES, "train", config["data"]["bundle"],
            set_size=256, seed=int(config["training"].get("seed", 42)), _bundle=bundle)
        self.train_records = self.records
        self.gene_order_sha256 = bundle["manifest"]["gene_order_sha256"]
        for index, record in enumerate(self.train_records):
            record["index"] = index
            record["control_pool"] = self.control_rows

    def training_binding(self):
        return training_binding(self.bundle)


class ComboMLPRuntimeData:
    def __init__(self, config, device="cuda"):
        self.config = config
        self.device = torch.device(device)
        self.bundle = load_bundle(config["data"]["bundle"], require_state=True, full_scan=True)
        self.data = self.train_dataset = ComboMLPData(config, self.bundle)
        self.root = Path(config["data"]["mlp_assets"])
        manifest_path = self.root / "manifest.json"
        self.asset_manifest = manifest = json.loads(manifest_path.read_text())
        if manifest.get("schema") != SCHEMA or manifest.get("status") != "complete":
            raise ValueError("Combo mean assets are incomplete or incompatible")
        validate_kernel_mode(config, manifest=manifest)
        validate_selection_mode(config, manifest=manifest,
            report=json.loads((self.root / "kernel_selection.json").read_text()))
        if (manifest["training_binding"] != self.data.training_binding()
                or not manifest.get("training_only_statistics")
                or manifest.get("external_expression_used_for_fitting")):
            raise ValueError("Combo mean statistics must match the training-only split/gene axis")
        self.asset_manifest_sha256 = self.cache_manifest_sha256 = sha256(manifest_path)
        if config["data"].get("verify_asset_hashes", True):
            actual = input_identities(config)
            if {k: v["sha256"] for k, v in actual.items()} != {
                    k: v["sha256"] for k, v in manifest["inputs"].items()}:
                raise ValueError("Combo raw/cache/drug-bank content differs from prepared assets")
            for name, expected in manifest["files"].items():
                if sha256(self.root / name) != expected:
                    raise ValueError("Prepared Combo mean asset changed: " + name)
        self.mean_assets = torch.load(self.root / "mean_assets.pt", map_location="cpu", weights_only=False)
        metadata = self.mean_assets["metadata"]
        validate_kernel_mode(config, manifest=manifest, metadata=metadata,
                             kernelbank_assets=self.mean_assets["kernelbank_assets"])
        validate_selection_mode(config, metadata=metadata)
        if metadata["training_binding"] != self.data.training_binding():
            raise ValueError("Combo mean-head training identity differs")
        names = [record["condition"] for record in self.data.train_records]
        if (metadata["conditions"] != names or json.loads((self.root / "conditions.json").read_text()) != names
                or tuple(self.mean_assets["targets"].shape) != (1, 24, 2000)):
            raise ValueError("Combo mean anchors or gene axis differ")
        self.kernel_bank = MixedKernelBank.from_assets(self.mean_assets["kernelbank_assets"])
        with np.load(self.root / "drug_features.npz", allow_pickle=False) as archive:
            drug_names = archive["drug_names"].tolist()
            self._drug_features = dict(zip(drug_names, np.array(archive["embeddings"], copy=True)))
            self._fingerprints = dict(zip(drug_names, np.array(archive["fingerprints"], copy=True)))
        self._drug_tensors = {name: torch.as_tensor(value, device=self.device)
                              for name, value in self._drug_features.items()}
        self.condition_latent_delta = torch.from_numpy(np.load(self.root / "condition_latent_delta.npy")).to(self.device)
        if self.condition_latent_delta.shape != (24, 2048):
            raise ValueError("Combo latent mean-delta shape differs")
        self._queries, self._test_datasets = {}, {}
        self._context = None

    def test_dataset(self, seed):
        seed = int(seed)
        if seed not in self._test_datasets:
            self._test_datasets[seed] = ComboDataset(CELL_LINES, "external_test",
                self.config["data"]["bundle"], set_size=128, seed=seed, _bundle=self.bundle)
        return self._test_datasets[seed]

    def mean_query(self, record):
        key = record["condition"]
        if key not in self._queries:
            feature, fingerprint = paired_features(record, self._drug_features, self._fingerprints)
            query = self.kernel_bank.query(feature[None], fingerprint[None], 0)
            self._queries[key] = torch.as_tensor(query, dtype=torch.float64, device=self.device)
        return {"head_kernel": self._queries[key], "head_stratum": 0}

    def condition(self, record):
        # Use the same canonical name ordering and exact-zero second slot as
        # sample_conditions; no new dose/cell-line encoding is introduced.
        drugs = tuple(sorted(record["drugs"]))
        slots = [self._drug_tensors[name] for name in drugs]
        if len(slots) == 1:
            slots.append(torch.zeros_like(slots[0]))
        suffix = torch.tensor([0.0, 1.0, 0.0, 0.0], device=self.device)
        return torch.cat((*slots, suffix)).unsqueeze(0)

    def source_states(self, line, rows, *, training=False):
        if line != "A549":
            raise ValueError("The original Combo split contains A549 only")
        values = np.array(self.bundle["states"][np.asarray(rows, dtype=np.int64)], dtype=np.float32, copy=True)
        return torch.from_numpy(values).to(self.device)

    def target_states(self, line, rows):
        return self.source_states(line, rows)

    def source_batch(self, record, rows):
        return {"source": self.source_states("A549", rows), "condition": self.condition(record),
                **self.mean_query(record)}

    def training_condition(self, step, micro, accumulation, seed, rank=0, world=1):
        ordinal = (int(step) - 1) * int(accumulation) + int(micro)
        epoch, offset = divmod(ordinal, len(self.data))
        order = np.random.default_rng(np.random.SeedSequence([int(seed), epoch])).permutation(len(self.data))
        return int(order[offset])

    def set_training_context(self, step, micro, accumulation, rank=0, world=1):
        self._context = {"ordinal": (int(step) - 1) * int(accumulation) + int(micro),
                         "rank": int(rank), "world": int(world)}

    def training_batch(self, record, candidate_rng, pair_generator, decoder_rng,
                       pairs=256, candidates=256, decoder_cells=256):
        if self._context is None:
            raise RuntimeError("Combo training requires an explicit optimizer/microstep context")
        ordinal, rank, world = (self._context[name] for name in ("ordinal", "rank", "world"))
        seed = int(self.config["training"].get("seed", 42))
        epoch = ordinal // len(self.data)
        index = int(record["index"])
        # Native ComboDataset recomputes a per-epoch/per-condition generator.
        # Preserve those candidates, rather than advancing SciPlex's stream.
        rng = np.random.default_rng(np.random.SeedSequence([seed, epoch, index]))
        source_rows = rng.choice(self.data.control_rows, candidates, replace=True)
        target_rows = rng.choice(record["rows"], candidates, replace=True)
        source, target = self.source_states("A549", source_rows), self.target_states("A549", target_rows)
        pair_generator.manual_seed(seed * 1_000_003 + ordinal * world + rank)
        settings = self.config["training"]
        with torch.autocast(self.device.type, enabled=False):
            source_index, target_index, stats = pair_population(source, target,
                n_samples=pairs, epsilon=float(settings.get("ot_epsilon", 0.01)),
                max_iterations=int(settings.get("ot_max_iterations", 2000)),
                tolerance=float(settings.get("ot_tolerance", 1e-3)), generator=pair_generator)
        # New MMD training alone uses independent population draws. These do
        # not advance the native Flow candidate/OT/time/path-noise streams.
        decoder_source_rows = decoder_rng.choice(self.data.control_rows, decoder_cells, replace=True)
        decoder_target_rows = decoder_rng.choice(record["rows"], decoder_cells, replace=True)
        decoder_target = self.data._expression_tensor(self.bundle["counts"], decoder_target_rows).to(self.device)
        batch = {"source": source[source_index], "target": target[target_index],
            "condition": self.condition(record), "mean_delta": self.condition_latent_delta[index],
            "decoder_source": self.source_states("A549", decoder_source_rows),
            "decoder_target": decoder_target,
            "time": torch.rand(pairs, device=self.device, generator=pair_generator),
            "noise": torch.randn(pairs, 2048, device=self.device, generator=pair_generator),
            "source_rows": source_rows[source_index.cpu().numpy()],
            "target_rows": target_rows[target_index.cpu().numpy()],
            "decoder_source_rows": decoder_source_rows, "decoder_target_rows": decoder_target_rows,
            **self.mean_query(record)}
        return batch, stats


def make_runtime(config, device="cuda"):
    return ComboMLPRuntimeData(config, device)
