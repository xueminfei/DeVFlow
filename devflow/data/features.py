"""Read only deterministic frozen SE features; no trainable projection/time cache."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
import torch

SCHEMA = "sciplex3_raw_frozen_features_v1"

def file_sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def ordered_sha256(values):
    return hashlib.sha256("\n".join(map(str, values)).encode()).hexdigest()

def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n")
    tmp.replace(path)

class FrozenFeatureCache:
    """Rows refer to the original processed dataset, not compact control indices.

    Raw gene tensors are stored as BF16 bits in uint16 NPY files (NumPy has no
    native BF16 dtype). Gene ID zero is valid; only gene_valid masks padding.
    Every latent and raw gene row uses the identical frozen SE/no-dropout policy.
    """
    def __init__(self, root, require_complete=True):
        self.root = Path(root)
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        if self.manifest.get("schema") != SCHEMA:
            raise ValueError("Unsupported raw frozen feature cache schema")
        if require_complete and self.manifest.get("status") != "complete":
            raise ValueError("Frozen feature extraction/final verification is incomplete")
        if self.manifest.get("trainable_projection_cached") or self.manifest.get("time_cached"):
            raise ValueError("This cache must not contain trainable or time projections")
        self._arrays = {}
        self._lookups = {}
        self._esm_cpu = None
        self._drugs = None
        for line, info in self.manifest["lines"].items():
            rows = self._array(line, "control_rows")
            if ordered_sha256(rows) != info["control_row_order_sha256"]:
                raise ValueError("Control row order differs from manifest")
            if len(np.unique(rows)) != len(rows) or np.any(rows < 0) or np.any(rows >= info["n_cells"]):
                raise ValueError("Invalid control row map")
            expected = (info["n_controls"], self.manifest["token_length"])
            ids = self._array(line, "gene_ids")
            valid = self._array(line, "gene_valid")
            if ids.shape != expected or valid.shape != expected:
                raise ValueError("Gene IDs and validity axes differ")
            raw = self._array(line, "raw_gene_tokens")
            if raw.shape != expected + (self.manifest["raw_gene_dim"],) or raw.dtype != np.uint16:
                raise ValueError("Raw gene cache format differs")
            if self._array(line, "cell_embeddings").shape != (info["n_cells"], 2048):
                raise ValueError("Cell cache shape differs")
            lookup = np.full(info["n_cells"], -1, np.int64)
            lookup[rows] = np.arange(len(rows))
            self._lookups[line] = lookup

    def _array(self, line, name):
        key = (line, name)
        if key not in self._arrays:
            self._arrays[key] = np.load(self.root / line / (name + ".npy"), mmap_mode="r")
        return self._arrays[key]

    def get(self, line, original_rows, device="cpu", dtype=torch.bfloat16):
        rows = np.asarray(original_rows, dtype=np.int64).reshape(-1)
        if np.any(rows < 0) or np.any(rows >= len(self._lookups[line])):
            raise IndexError("Control source row outside original dataset")
        positions = self._lookups[line][rows]
        if np.any(positions < 0):
            raise ValueError("Gene conditioning is available only for control rows")
        bits = np.array(self._array(line, "raw_gene_tokens")[positions], copy=True)
        raw = torch.from_numpy(bits.view(np.int16)).view(torch.bfloat16).to(device=device, dtype=dtype)
        ids = torch.from_numpy(np.array(self._array(line, "gene_ids")[positions], dtype=np.int64)).to(device)
        valid = torch.from_numpy(np.array(self._array(line, "gene_valid")[positions], dtype=bool)).to(device)
        return dict(raw_gene_tokens=raw, gene_ids=ids, gene_valid=valid)

    def cell_embeddings(self, line, original_rows=None, device=None):
        values = self._array(line, "cell_embeddings")
        if original_rows is not None:
            values = np.array(values[np.asarray(original_rows, np.int64)], dtype=np.float32)
        if device is None:
            return values
        return torch.from_numpy(np.array(values, dtype=np.float32)).to(device)

    def esm_table(self, device="cpu", dtype=torch.float32):
        if self._esm_cpu is None:
            self._esm_cpu = torch.from_numpy(np.load(self.root / "esm_embeddings.npy"))
        return self._esm_cpu.to(device=device, dtype=dtype)

    def drug_table(self):
        if self._drugs is None:
            self._drugs = torch.load(self.root / "drug_embeddings.pt", map_location="cpu", weights_only=False)
        return self._drugs

    def control_rows(self, line):
        return np.asarray(self._array(line, "control_rows"))
