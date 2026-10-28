"""SciPlex3 immutable feature runtime and condition-local OT batches."""
from pathlib import Path
import json
import numpy as np
import torch
from devflow.data.sciplex3 import SciPlexData, CELL_LINES as LINES
from devflow.data.features import FrozenFeatureCache, ordered_sha256
from devflow.flow.matching import pair_population
from devflow.runtime.artifacts import sha256

class RuntimeData:
    def __init__(self, cache_root, dataset, device="cuda"):
        self.cache = FrozenFeatureCache(cache_root)
        self.data = dataset
        if self.cache.manifest["data_binding"] != self.data.training_binding():
            raise ValueError("Frozen cache training binding differs from the active dataset")
        self.device = torch.device(device)
        self.esm = self.cache.esm_table(self.device, torch.bfloat16)
        bank = self.cache.drug_table()
        from rdkit import Chem
        self.drug = {}
        for name, emb in zip(bank["canonical_smiles"], bank["embeddings"]):
            self.drug[str(name)] = torch.as_tensor(emb, dtype=torch.float32, device=self.device)
        self.canonicalize = lambda s: Chem.MolToSmiles(Chem.MolFromSmiles(s), canonical=True)
        stats = Path(cache_root)/"statistics"
        condition_records = json.loads((stats/"conditions.json").read_text())
        if isinstance(condition_records, dict):
            condition_records = condition_records["conditions"]
        expected = [r["condition"] for r in self.data.train_records]
        actual = [r["condition"] for r in condition_records]
        if actual != expected:
            raise ValueError("Cache training-condition order differs")
        self.condition_expression = torch.from_numpy(
            np.load(stats/"condition_expression_mean.npy")).to(self.device)
        self.condition_latent = torch.from_numpy(
            np.load(stats/"condition_latent_mean.npy")).to(self.device)
        self.control_latent = torch.from_numpy(
            np.load(stats/"control_latent_mean.npy")).to(self.device)
        if json.loads((stats/"control_line_order.json").read_text()) != list(LINES):
            raise ValueError("Cache control line order differs")
        self.cache_manifest_sha256 = sha256(Path(cache_root)/"manifest.json")

    def source_batch(self, rec, source_rows):
        line = rec["cell_line"]
        features = self.cache.get(line, source_rows, self.device, torch.bfloat16)
        features["source"] = self.cache.cell_embeddings(line, source_rows, self.device)
        features["drug"] = self.drug[self.canonicalize(rec["drug_smiles"])][None]
        features["dose"] = torch.tensor([[np.log10(1+rec["dose_nm"])]],
                                        dtype=torch.float32, device=self.device)
        features["line"] = torch.eye(3, device=self.device)[LINES.index(line)][None]
        return features

    def training_batch(self, rec, rng, generator, pairs=8, candidates=256):
        line = rec["cell_line"]
        source_rows = rng.choice(rec["control_pool"], candidates,
                                 replace=len(rec["control_pool"]) < candidates)
        target_rows = rng.choice(rec["rows"], candidates, replace=len(rec["rows"]) < candidates)
        source = self.cache.cell_embeddings(line, source_rows, self.device)
        target = self.cache.cell_embeddings(line, target_rows, self.device)
        # Exactly one pairing. These same indices select latent, genes and expression.
        si, ti, stats = pair_population(source, target, n_samples=pairs, epsilon=0.01,
                                       max_iterations=2000, tolerance=1e-3,
                                       generator=generator)
        paired_source = source_rows[si.cpu().numpy()]
        paired_target = target_rows[ti.cpu().numpy()]
        batch = self.source_batch(rec, paired_source)
        batch["target"] = target[ti]
        batch["expression"] = torch.from_numpy(
            self.data.expression(line, paired_target)).to(self.device)
        batch["mean_delta"] = self.condition_latent[rec["index"]] - self.control_latent[LINES.index(line)]
        batch["mean_expression"] = self.condition_expression[rec["index"]]
        batch["time"] = torch.rand(pairs, device=self.device, generator=generator)
        batch["noise"] = torch.randn(pairs, 2048, device=self.device, generator=generator)
        batch["source_rows"] = paired_source
        batch["target_rows"] = paired_target
        return batch, stats


class LoraRuntimeData(RuntimeData):
    def __init__(self,cache_root,prefix_root,dataset,device="cuda"):
        super().__init__(cache_root,dataset,device)
        self.prefix_root = Path(prefix_root)
        self.prefix_manifest = json.loads((self.prefix_root/"manifest.json").read_text())
        m = self.prefix_manifest
        if m.get("schema") != "sciplex3_deterministic_control_prefix_v1" or m.get("status") != "complete":
            raise ValueError("Deterministic control-prefix preparation is incomplete")
        if m["frozen_cache_manifest_sha256"] != self.cache_manifest_sha256:
            raise ValueError("Control-prefix cache belongs to a different frozen gene/target cache")
        if m["training_binding"] != self.data.training_binding():
            raise ValueError("Control-prefix training rows differ")
        if m["se_policy"]["functional_attention_dropout"] != 0.0:
            raise ValueError("Historical stochastic control prefixes cannot be mixed into this experiment")
        self.prefix_manifest_sha256 = sha256(self.prefix_root/"manifest.json")
        self.prefixes = {}
        for line in LINES:
            folder = self.prefix_root/line
            rows = np.load(folder/"control_rows.npy",allow_pickle=False)
            global_ids = np.load(folder/"control_global_ids.npy",allow_pickle=False)
            np.testing.assert_array_equal(rows,self.cache.control_rows(line))
            np.testing.assert_array_equal(global_ids,self.data.global_ids(line,rows))
            info = m["lines"][line]
            if ordered_sha256(rows) != info["control_row_order_sha256"]:
                raise ValueError("Control-prefix row-order hash differs")
            if ordered_sha256(global_ids) != info["control_global_order_sha256"]:
                raise ValueError("Control-prefix global cell-ID order differs")
            if sha256(folder/"prefix.npy") != info["prefix_sha256"]:
                raise ValueError("Control-prefix tensor hash differs")
            values = np.load(folder/"prefix.npy",allow_pickle=False)
            if values.dtype != np.float32 or values.shape != (len(rows),2048) or not np.isfinite(values).all():
                raise ValueError("Invalid immutable control-prefix array")
            self.prefixes[line] = torch.from_numpy(values).to(self.device)

    def prefix_rows(self,line,rows):
        positions = self.cache._lookups[line][np.asarray(rows,dtype=np.int64)]
        if np.any(positions<0):
            raise ValueError("SE LoRA is restricted to control cells")
        return self.prefixes[line].index_select(0,torch.as_tensor(positions,device=self.device))

    def source_batch(self,rec,source_rows):
        batch = super().source_batch(rec,source_rows)
        batch["source_prefix"] = self.prefix_rows(rec["cell_line"],source_rows)
        return batch

    def training_batch(self,rec,rng,generator,pairs=8,candidates=256,source_model=None):
        if source_model is None:
            raise ValueError("Online OT requires the current cell SE LoRA model")
        line = rec["cell_line"]
        source_rows = rng.choice(rec["control_pool"],candidates,replace=len(rec["control_pool"])<candidates)
        target_rows = rng.choice(rec["rows"],candidates,replace=len(rec["rows"])<candidates)
        prefix = self.prefix_rows(line,source_rows)
        target = self.cache.cell_embeddings(line,target_rows,self.device)
        # These temporary source values select discrete OT indices only. The
        # selected prefixes are encoded again INSIDE DDP forward with gradients.
        with torch.no_grad(),torch.autocast(self.device.type,dtype=torch.bfloat16,
                enabled=self.device.type=="cuda",cache_enabled=False):
            source_for_ot = source_model.encode_source(prefix)
            si,ti,stats = pair_population(source_for_ot.float(),target,n_samples=pairs,
                epsilon=.01,max_iterations=2000,tolerance=1e-3,generator=generator)
        paired_source = source_rows[si.cpu().numpy()]
        paired_target = target_rows[ti.cpu().numpy()]
        batch = self.source_batch(rec,paired_source)
        batch.update(target=target[ti],expression=torch.from_numpy(self.data.expression(line,paired_target)).to(self.device),
            target_population_mean=self.condition_latent[rec["index"]],mean_expression=self.condition_expression[rec["index"]],
            time=torch.rand(pairs,device=self.device,generator=generator),
            noise=torch.randn(pairs,2048,device=self.device,generator=generator),
            source_rows=paired_source,target_rows=paired_target)
        return batch,stats

    def training_group(self,records,rng,generator,source_model,pairs=8,candidates=256):
        batches,stats=[],[]
        for rec in records:
            b,s=self.training_batch(rec,rng,generator,pairs,candidates,source_model)
            batches.append(b);stats.append(s)
        return concatenate_conditions(batches,[r["cell_line"] for r in records],pairs),stats


def concatenate_conditions(batches,lines,pairs):
    if not batches or len(batches)!=len(lines):
        raise ValueError("Every grouped condition needs its own cell-line and batch")
    result={}
    stacked={"target_population_mean","mean_expression"}
    numpy_keys={"source_rows","target_rows"}
    conditions={"drug","dose","line"}
    for key in batches[0]:
        if key in numpy_keys:
            result[key]=np.concatenate([b[key] for b in batches])
        elif key in stacked:
            result[key]=torch.stack([b[key] for b in batches])
        else:
            values=[b[key].expand(pairs,-1) if key in conditions and b[key].shape[0]==1 else b[key] for b in batches]
            result[key]=torch.cat(values,dim=0)
    result["pairs_per_condition"]=pairs
    result["condition_lines"]=list(lines)
    return result
