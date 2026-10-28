"""Basic-MLP SciPlex3 data binding, independent decoder draws and mean queries."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from devflow.data.mean_assets import SCHEMA, canonical_smiles, morgan_fingerprints
from devflow.data.sciplex3 import CELL_LINES, SciPlexData
from devflow.flow.matching import pair_population
from devflow.mean.kernels import MixedKernelBank
from devflow.runtime.artifacts import ordered_sha256, sha256


class MLPRuntimeData:
    """All trainable quantities live in the model; this runtime is immutable.

    Flow candidates use a shared generator across ranks. Discrete OT pairing,
    artificial time and path noise use a rank-specific Torch generator. The
    decoder has a separate rank-specific NumPy generator and receives fresh
    unpaired cell populations, so its sampling never advances the Flow stream.
    """

    def __init__(self, config, device="cuda"):
        self.config = config
        settings = config["data"]
        self.device = torch.device(device)
        self.data = SciPlexData(settings["base_dir"], settings["sampling_manifest"],
                               settings.get("source_split_root"))
        self.root = Path(settings["mlp_assets"])
        self.asset_manifest = json.loads((self.root / "manifest.json").read_text())
        manifest = self.asset_manifest
        if manifest.get("schema") != SCHEMA or manifest.get("status") != "complete":
            raise ValueError("MLP/mean preparation is incomplete or has a different schema")
        if manifest["training_binding"] != self.data.training_binding():
            raise ValueError("MLP assets belong to a different training split/gene axis")
        if not manifest.get("training_only_statistics") or manifest.get("external_expression_used_for_fitting"):
            raise ValueError("Mean fitting statistics must come only from the training pool")
        policy = manifest["source_policy"]
        if (policy.get("native_attention_dropout") != .1 or policy.get("cache_seed") != 20261021
                or policy.get("lora_B") != "zero_frozen"):
            raise ValueError("This MLP recipe requires the historical frozen stochastic source")
        self.asset_manifest_sha256 = sha256(self.root / "manifest.json")
        self.cache_manifest_sha256 = self.asset_manifest_sha256
        verify = bool(settings.get("verify_asset_hashes", True))
        if verify:
            for filename, expected in manifest["files"].items():
                if sha256(self.root / filename) != expected:
                    raise ValueError("Prepared MLP asset changed: " + filename)
        self.mean_assets = torch.load(self.root / "mean_assets.pt", map_location="cpu", weights_only=False)
        if self.mean_assets["metadata"]["training_binding"] != self.data.training_binding():
            raise ValueError("Mean training assets differ from the active dataset")
        self.kernel_bank = MixedKernelBank.from_assets(self.mean_assets["kernelbank_assets"])
        self._strata = {(str(line), float(dose)): i for i, (line, dose) in
                        enumerate(self.mean_assets["metadata"]["strata"])}
        with np.load(self.root / "drug_features.npz", allow_pickle=False) as archive:
            names = archive["canonical_smiles"].tolist()
            values = np.array(archive["embeddings"], dtype=np.float32)
        self._drug_features = dict(zip(names, values))
        self._drug_tensors = {key: torch.from_numpy(value).to(self.device) for key, value in self._drug_features.items()}
        self._fingerprints = dict(zip(names, morgan_fingerprints(names)))
        self._queries = {}
        self._sources = {}
        self._source_lookups = {}
        self._targets = {}
        self._line_onehot = torch.eye(len(CELL_LINES), device=self.device)
        self.condition_latent = torch.from_numpy(np.load(self.root / "condition_latent_mean.npy")).to(self.device)
        self.control_latent = torch.from_numpy(np.load(self.root / "control_latent_mean.npy")).to(self.device)
        if json.loads((self.root / "conditions.json").read_text()) != [r["condition"] for r in self.data.train_records]:
            raise ValueError("Prepared latent population statistics have different condition order")
        for line in CELL_LINES:
            folder = self.root / line
            info = manifest["lines"][line]
            if verify:
                for name, expected in info["files"].items():
                    if sha256(folder / name) != expected:
                        raise ValueError(f"Prepared source cache changed: {line}/{name}")
                if sha256(self.data.base / line / f"{line}_meta.pkl") != info["metadata_sha256"]:
                    raise ValueError("Processed metadata row identity changed")
            rows = np.load(folder / "control_rows.npy", allow_pickle=False)
            global_ids = np.load(folder / "control_global_ids.npy", allow_pickle=False)
            np.testing.assert_array_equal(rows, self.data.controls(line))
            np.testing.assert_array_equal(global_ids, self.data.global_ids(line, rows))
            if ordered_sha256(rows) != info["control_row_order_sha256"]:
                raise ValueError("Source row-order hash differs")
            lookup = np.full(len(self.data.metadata[line]), -1, np.int64)
            lookup[rows] = np.arange(len(rows))
            self._source_lookups[line] = lookup
            self._sources[line] = {}
            for mode in ("train", "eval"):
                values = np.load(folder / f"source_{mode}.npy", allow_pickle=False)
                if values.shape != (len(rows), 2048) or not np.isfinite(values).all():
                    raise ValueError("Invalid prepared source matrix")
                self._sources[line][mode] = torch.from_numpy(values).to(self.device)
            state_path = self.data.base / line / f"{line}_embeddings.npy"
            if verify and sha256(state_path) != info["target_latent_sha256"]:
                raise ValueError("Frozen target SE content differs from prepared training statistics")
            states = np.load(state_path, mmap_mode="r", allow_pickle=False)
            if states.shape != (len(lookup), 2048) or states.dtype != np.float32:
                raise ValueError("Frozen target state axis differs")
            self._targets[line] = states

    def source_states(self, line, rows, *, training=False):
        rows = np.asarray(rows, dtype=np.int64)
        lookup = self._source_lookups[line]
        if np.any(rows < 0) or np.any(rows >= len(lookup)):
            raise IndexError("Source row outside processed dataset")
        positions = lookup[rows]
        if np.any(positions < 0):
            raise ValueError("Only matched control cells may initialize the Flow")
        return self._sources[line]["train" if training else "eval"].index_select(
            0, torch.as_tensor(positions, device=self.device))

    def target_states(self, line, rows):
        values = np.array(self._targets[line][np.asarray(rows, np.int64)], dtype=np.float32, copy=True)
        return torch.from_numpy(values).to(self.device)

    def mean_query(self, record):
        """No target/control expression or SE computation is needed here."""
        smiles = canonical_smiles(record["drug_smiles"])
        stratum = self._strata[(record["cell_line"], float(record["dose_nm"]))]
        key = (smiles, stratum)
        if key not in self._queries:
            query = self.kernel_bank.query(self._drug_features[smiles][None],
                                           self._fingerprints[smiles][None], stratum)
            self._queries[key] = torch.as_tensor(query, dtype=torch.float64, device=self.device)
        return {"head_kernel": self._queries[key], "head_stratum": stratum}

    def condition(self, record):
        smiles = canonical_smiles(record["drug_smiles"])
        return torch.cat((self._drug_tensors[smiles],
            torch.tensor([np.log10(1.0 + float(record["dose_nm"]))], device=self.device, dtype=torch.float32),
            self._line_onehot[CELL_LINES.index(record["cell_line"])])).unsqueeze(0)

    def source_batch(self, record, rows):
        return dict(source=self.source_states(record["cell_line"], rows),
                    condition=self.condition(record), **self.mean_query(record))

    def training_batch(self, record, candidate_rng, pair_generator, decoder_rng,
                       pairs=256, candidates=256, decoder_cells=256):
        line = record["cell_line"]
        # Historical candidate draws are WITH replacement even for large pools.
        source_rows = candidate_rng.choice(record["control_pool"], candidates, replace=True)
        target_rows = candidate_rng.choice(record["rows"], candidates, replace=True)
        source = self.source_states(line, source_rows, training=True)
        target = self.target_states(line, target_rows)
        with torch.autocast(self.device.type, enabled=False):
            source_index, target_index, stats = pair_population(source, target,
                n_samples=pairs, epsilon=.01, max_iterations=2000, tolerance=1e-3,
                generator=pair_generator)
        # A separate draw avoids feeding OT-pair resampling to a set loss.
        decoder_source_rows = decoder_rng.choice(record["control_pool"], decoder_cells, replace=True)
        decoder_target_rows = decoder_rng.choice(record["rows"], decoder_cells, replace=True)
        batch = dict(source=source[source_index], target=target[target_index],
            condition=self.condition(record),
            mean_delta=self.condition_latent[record["index"]] - self.control_latent[CELL_LINES.index(line)],
            decoder_source=self.source_states(line, decoder_source_rows, training=True),
            decoder_target=torch.from_numpy(self.data.expression(line, decoder_target_rows)).to(self.device),
            time=torch.rand(pairs, device=self.device, generator=pair_generator),
            noise=torch.randn(pairs, 2048, device=self.device, generator=pair_generator),
            source_rows=source_rows[source_index.cpu().numpy()],
            target_rows=target_rows[target_index.cpu().numpy()],
            decoder_source_rows=decoder_source_rows, decoder_target_rows=decoder_target_rows,
            **self.mean_query(record))
        return batch, stats


def make_runtime(config, device="cuda"):
    return MLPRuntimeData(config, device)
