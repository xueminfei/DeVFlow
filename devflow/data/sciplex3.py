"""SciPlex3 union data with content-bound rows and shared external sampling.

The original manifest identity is retained for checkpoint compatibility, while
physical roots are explicit inputs so the dataset can be relocated.
"""
from pathlib import Path
import ast
import json
import pickle
import numpy as np
import pandas as pd
from devflow.runtime.artifacts import sha256, ordered_sha256
CELL_LINES=("A549","MCF7","K562")
SEEDS=(4243,4244,4245,4246)
TRAINING_SPLITS=("train","internal_test")
ordered_hash=ordered_sha256

def normalize(x):
 x=np.asarray(x,dtype=np.float32)
 return np.log1p(1000.0*x/np.maximum(x.sum(axis=1,keepdims=True),1.0)).astype(np.float32)
def dose_nm(value):
 if isinstance(value,(int,float,np.number)):return float(value)
 _,amount,unit=ast.literal_eval(str(value))[0]
 return float(amount)*{"nM":1.,"uM":1000.,"mM":1000000.}[str(unit)]
def condition_key(rec):return "{}|{}|{:g}".format(rec["cell_line"],rec["drug_id"],rec["dose_nm"])
class SciPlexData:
 def __init__(self, base, sampling_manifest, source_split_root=None, training_statistics=None):
  self.base=Path(base);self.metadata={};self.raw={};self.lines={};self._records={k:[] for k in ("train","external_test")};self._allowed={};self._lookup={}
  self.union_manifest=json.loads((self.base/"union_manifest.json").read_text())
  self.union_manifest_sha256=sha256(self.base/"union_manifest.json")
  self.source_split_root = Path(source_split_root or self.union_manifest["source_base_dir"])
  self.sampling_manifest_path = Path(sampling_manifest)
  self.training_statistics = Path(training_statistics) if training_statistics else None
  assert self.union_manifest["total_training_treated"]==632869
  assert self.union_manifest["total_controls"]==12708
  self.genes=json.loads((self.base/"hvg_info.json").read_text())["gene_symbols"]
  self.gene_order_sha256=ordered_hash(self.genes)
  assert len(self.genes)==2000
  for line in CELL_LINES:
   path=self.base/line
   with (path/f"{line}_meta.pkl").open("rb") as f:meta=pd.DataFrame(pickle.load(f))
   raw=np.load(path/f"{line}_hvg.npy",mmap_mode="r")
   assert raw.shape==(len(meta),2000)
   self.metadata[line]=meta;self.raw[line]=raw
   lookup={int(v):i for i,v in enumerate(meta.ds_level_index)}
   assert len(lookup)==len(meta)
   self._lookup[line]=lookup
   controls=set(json.loads((path/"control_indices.json").read_text()))
   control_rows=np.array([i for i,v in enumerate(meta.ds_level_index) if int(v) in controls],np.int64)
   assert len(control_rows) and meta.iloc[control_rows].is_control.astype(bool).all()
   self.lines[line]=dict(meta=meta,raw=raw,control_rows=control_rows,root=path)
   self._allowed[line]={}
   for split in self._records:
    allowed=set(json.loads((path/f"{split}_indices.json").read_text()))
    rows=np.array([i for i,v in enumerate(meta.ds_level_index) if int(v) in allowed],np.int64)
    self._allowed[line][split]=rows
    assert not meta.iloc[rows].is_control.astype(bool).any()
    groups={}
    for i in rows:
     key=(str(meta.drug_id.iloc[i]),round(dose_nm(meta.drug_conc.iloc[i]),8));groups.setdefault(key,[]).append(int(i))
    for (drug,dose),members in sorted(groups.items()):
     index=len(self._records[split]);smiles=str(meta.drug_smiles.iloc[members[0]])
     rec=dict(index=index,group_index=index,cell_line=line,drug_id=drug,drug_smiles=smiles,smiles=smiles,
              dose_nm=dose,rows=np.asarray(members,np.int64),treated_rows=np.asarray(members,np.int64),
              control_pool=control_rows,split=split)
     rec["condition"]=condition_key(rec);self._records[split].append(rec)
   # The preserved internal_test filename records the ORIGINAL held-in source.
   # It is part of the new train union, never a validation/evaluation dataset.
   internal_ids=set(json.loads((path/"internal_test_indices.json").read_text()))
   internal_rows=np.array([i for i,v in enumerate(meta.ds_level_index) if int(v) in internal_ids],np.int64)
   self._allowed[line]["internal_test"]=internal_rows
   tr=set(self._allowed[line]["train"]);it=set(internal_rows);te=set(self._allowed[line]["external_test"])
   source=self.source_split_root/line
   source_train=set(json.loads((source/"train_indices.json").read_text()))
   source_internal=set(json.loads((source/"internal_test_indices.json").read_text()))
   current_train=set(meta.ds_level_index.iloc[list(tr)].astype(int))
   audit=self.union_manifest["lines"][line]
   assert not source_train&source_internal
   assert internal_ids==source_internal and current_train==source_train|source_internal
   assert it<=tr and not tr&te
   assert len(tr)==audit["train_union_cells"] and len(it)==audit["source_internal_cells"]
   assert sha256(path/"train_indices.json")==audit["train_union_sha256"]
   assert sha256(source/"train_indices.json")==audit["source_train_sha256"]
   assert sha256(path/"internal_test_indices.json")==audit["source_internal_sha256"]
   assert sha256(path/"control_indices.json")==audit["control_sha256"]
   assert sha256(path/"external_test_indices.json")==audit["external_sha256"]
   td=set(meta.iloc[list(tr)].drug_id.astype(str));ed=set(meta.iloc[list(te)].drug_id.astype(str))
   assert len(td)==179 and len(ed)==9 and not td&ed
  self.train_records=self._records["train"];self.test_records=self._records["external_test"]
  assert len(self.train_records)==2148 and len(self.test_records)==108
  self._sampling_manifest=json.loads((self.sampling_manifest_path).read_text())
  assert self._sampling_manifest["new_replicate_seeds"]==list(SEEDS)
  self._samples={}
 def records(self,split="train"):
  if split=="internal_test":raise ValueError("Original internal_test cells are now training cells, not an evaluation split")
  return self._records[split]
 def training_binding(self):
  return dict(data_base_dir=self.union_manifest["data_base_dir"],training_splits=list(TRAINING_SPLITS),
   union_manifest_sha256=self.union_manifest_sha256,gene_order_sha256=self.gene_order_sha256,
   train_rows_sha256={line:ordered_hash(self.global_ids(line,self.training_cells(line,False))) for line in CELL_LINES},
   control_rows_sha256={line:ordered_hash(self.global_ids(line,self.controls(line))) for line in CELL_LINES})
 def expression(self,cell_line,rows):return normalize(self.raw[cell_line][np.asarray(rows,np.int64)])
 def controls(self,cell_line):return self.lines[cell_line]["control_rows"]
 def training_cells(self,cell_line,include_controls=True):
  rows=self._allowed[cell_line]["train"]
  return np.concatenate([rows,self.controls(cell_line)]) if include_controls else rows
 def global_ids(self,cell_line,rows):return self.metadata[cell_line].ds_level_index.iloc[np.asarray(rows,np.int64)].to_numpy(dtype=np.int64)
 def sampling(self,seed):
  seed=int(seed)
  if seed in self._samples:return self._samples[seed]
  locked=next(r for r in self._sampling_manifest["replicates"] if r["seed"]==seed)
  assert len(locked["groups"])==108
  records=[]
  for rec,lock in zip(self.test_records,locked["groups"]):
   assert (rec["group_index"],rec["cell_line"],rec["drug_id"],rec["dose_nm"])==(lock["group_index"],lock["cell_line"],lock["drug_id"],lock["dose_nm"])
   lookup=self._lookup[rec["cell_line"]]
   control=np.array([lookup[int(x)] for x in lock["control_global_ids"]],np.int64)
   target=np.array([lookup[int(x)] for x in lock["treated_global_ids"]],np.int64)
   assert len(control)==len(target)==128
   assert set(control)<=set(rec["control_pool"]) and set(target)<=set(rec["rows"])
   records.append(dict(rec,control_rows=control,target_rows=target,sampled_treated_rows=target,
      control_global_ids=np.asarray(lock["control_global_ids"],np.int64),treated_global_ids=np.asarray(lock["treated_global_ids"],np.int64)))
  self._samples[seed]=records;return records
 def training_de(self):
  if self.training_statistics is None:raise ValueError("training_statistics was not configured")
  manifest=json.loads((self.training_statistics/"training_de_manifest.json").read_text())
  assert manifest["training_binding"]==self.training_binding(), "Training DE cache binding differs from train union"
  return (np.load(self.training_statistics/"training_de_scores.npy",mmap_mode="r"),np.load(self.training_statistics/"training_de_bh_mask.npy",mmap_mode="r"))
 def save_predictions(self, output_dir, seed, predictions, method, provenance=None):
  from devflow.io import write_prediction_archive
  records=self.sampling(seed)
  path=Path(output_dir)/f"predictions_seed_{seed}.npz"
  write_prediction_archive(path,seed=seed,
    condition_names=[r["condition"] for r in records],
    source_global_ids=np.stack([r["control_global_ids"] for r in records]),
    target_global_ids=np.stack([r["treated_global_ids"] for r in records]),
    gene_order_sha256=self.gene_order_sha256,predictions=list(predictions),
    metadata={"method":method,"training_binding":self.training_binding(),
      "sampling_manifest_sha256":sha256(self.sampling_manifest_path),**(provenance or {})})
  return path
