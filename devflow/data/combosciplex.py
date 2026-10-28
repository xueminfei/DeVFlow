"""Original raw-count ComboSciPlex paper split and population sampler.

All expression targets use exactly one HVG-total1000/log1p transform.
Controls and treated cells retain the fixed train/test/excluded assignments.
"""
from __future__ import annotations
import csv
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import torch


def _deps():
    pass


PROTOCOL = "combosciplex_combo_holdout_no_validation_v3"
NORMALIZATION = "hvg_total_1000_log1p"
STATE_ENCODER = "MAP_original_SE600M_embedding_2048"
CELL_LINES = ["A549", "MCF7", "K562"]
SINGLE_DRUGS = frozenset(("Alvespimycin", "Dacinostat", "Dasatinib", "Givinostat", "Panobinostat", "SRT2104"))
TEST_CONDITIONS = frozenset(("Alvespimycin+Panobinostat", "Dasatinib+Givinostat", "Crizotinib+Panobinostat", "Givinostat+SRT1720", "Panobinostat+Sorafenib"))
EXCLUDED_CONDITIONS = frozenset(("Cediranib+PCI-34051", "Cediranib+SRT3025"))
COMBINATIONS = frozenset("+".join(sorted(pair)) for pair in (
    [("Givinostat", x) for x in ("Carmofur", "Cediranib", "Crizotinib", "Curcumin", "Dasatinib", "SRT1720", "SRT2104", "Sorafenib", "Tanespimycin")]
    + [("Panobinostat", x) for x in ("Alvespimycin", "Crizotinib", "Curcumin", "Dasatinib", "PCI-34051", "SRT1720", "SRT2104", "SRT3025", "Sorafenib")]
    + [("Dacinostat", x) for x in ("Danusertib", "Dasatinib", "PCI-34051")]
    + [("Alvespimycin", x) for x in ("Pirarubicin", "SRT2104")]
    + [("Cediranib", x) for x in ("PCI-34051", "SRT3025")]
))
TRAIN_CONDITIONS = SINGLE_DRUGS | (COMBINATIONS - TEST_CONDITIONS - EXCLUDED_CONDITIONS)

_BUNDLE_CACHE = {}

def sequence_sha256(values):
    return hashlib.sha256("\n".join(str(x) for x in values).encode("utf-8")).hexdigest()


def _json(path):
    return json.loads(Path(path).read_text())


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def _clean_json(value):
    if isinstance(value, dict):
        return {str(k): _clean_json(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_clean_json(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def read_genes(path, allow_duplicates=False):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing ordered decoder gene list: {path}")
    if path.suffix.lower() == ".json":
        value = _json(path)
        if isinstance(value, dict):
            value = next((value[k] for k in ("gene_symbols", "genes", "gene_names") if k in value), None)
    else:
        value = path.read_text().splitlines()
    if not isinstance(value, list) or not value or any(not isinstance(x, str) or not x for x in value):
        raise ValueError(f"{path} must contain a gene list (JSON gene_symbols/genes/gene_names supported)")
    if not allow_duplicates and len(value) != len(set(value)):
        raise ValueError(f"Duplicate gene names in {path}; target gene names must be unique")
    return value


def _row_drugs(row):
    values = [str(row.get(k, "")).strip() for k in ("drug1", "drug2")]
    values = [x for x in values if x and x.lower() not in ("control", "dmso", "none", "nan")]
    if not values and row["condition"].lower() not in ("control", "dmso"):
        values = [x for x in row["condition"].split("+") if x.lower() not in ("control", "dmso")]
    if len(values) > 2 or len(set(values)) != len(values):
        raise ValueError(f"Invalid constituent drugs for {row['condition']!r}")
    return tuple(sorted(values))


def load_bundle(base_dir, require_state=True, full_scan=True):
    """Validate explicit raw-count/gene/cell-order provenance and the fixed split.

    --audit allows a missing SE file, reports state_ready=False, and never makes
    random substitute states. Actual training requires the complete real cache.
    """
    _deps()
    base = Path(base_dir).expanduser().resolve()
    required = ("obs.csv", "counts_hvg.npy", "genes.json", "drug_smiles.json", "manifest.json")
    missing = [str(base / name) for name in required if not (base / name).is_file()]
    if missing:
        raise FileNotFoundError("Missing ComboSciPlex bundle files:\n" + "\n".join(missing))
    key = (str(base), require_state, full_scan, tuple((name, (base / name).stat().st_mtime_ns) for name in required),
           (base / "se_embeddings.npy").stat().st_mtime_ns if (base / "se_embeddings.npy").exists() else None)
    if key in _BUNDLE_CACHE:
        return _BUNDLE_CACHE[key]
    with (base / "obs.csv").open(newline="") as handle:
        reader = csv.DictReader(handle)
        if not {"cell_id", "condition", "split"}.issubset(reader.fieldnames or []):
            raise ValueError("obs.csv requires cell_id,condition,split columns")
        obs = list(reader)
    ids = [r["cell_id"] for r in obs]
    if len(ids) != 63378 or len(set(ids)) != len(ids) or any(not x for x in ids):
        raise ValueError("Expected exactly 63378 uniquely named raw cells in obs.csv")
    for row in obs:
        row["drugs"] = _row_drugs(row)
        expected = "+".join(row["drugs"]) if row["drugs"] else "control"
        if row["condition"] != expected:
            raise ValueError(f"Noncanonical condition {row['condition']!r}; expected {expected!r}")
        if row["split"] not in ("train", "test", "excluded"):
            raise ValueError("This task has only train/test/excluded; no validation split")
        if row["drugs"]:
            expected_split = ("test" if expected in TEST_CONDITIONS else "excluded" if expected in EXCLUDED_CONDITIONS else "train" if expected in TRAIN_CONDITIONS else None)
            if row["split"] != expected_split:
                raise ValueError(f"Condition leakage or unknown condition: {expected}, split={row['split']}")
        elif row["split"] == "excluded":
            raise ValueError("All controls must belong to train or test")
    split_counts = {name: sum(r["split"] == name for r in obs) for name in ("train", "test", "excluded")}
    if split_counts != {"train": 48725, "test": 9476, "excluded": 5177}:
        raise ValueError(f"Unexpected cell split sizes: {split_counts}")
    control_counts = {name: sum(not r["drugs"] and r["split"] == name for r in obs) for name in ("train", "test")}
    if control_counts != {"train": 1306, "test": 145}:
        raise ValueError(f"Unexpected control split sizes: {control_counts}")
    for split, expected in (("train", TRAIN_CONDITIONS), ("test", TEST_CONDITIONS), ("excluded", EXCLUDED_CONDITIONS)):
        observed = {r["condition"] for r in obs if r["drugs"] and r["split"] == split}
        if observed != expected:
            raise ValueError(f"{split} condition set mismatch: missing={sorted(expected-observed)}, extra={sorted(observed-expected)}")
    training_drugs = {d for r in obs if r["split"] == "train" for d in r["drugs"]}
    if any(set(r["drugs"]) - training_drugs for r in obs if r["split"] == "test"):
        raise ValueError("A test drug is absent from all training conditions")
    genes = read_genes(base / "genes.json")
    if len(genes) not in (2000, 5000):
        raise ValueError("Combo interface supports train-only 2000 or 5000 HVGs")
    manifest = _json(base / "manifest.json")
    expected_metadata = {
        "format_version": 1, "split_protocol": PROTOCOL,
        "cell_order_sha256": sequence_sha256(ids), "gene_order_sha256": sequence_sha256(genes),
        "hvg_fit_scope": "train_only",
        "hvg_fit_cell_order_sha256": sequence_sha256([r["cell_id"] for r in obs if r["split"] == "train"]),
        "expression_input": "raw_counts", "expression_normalization": NORMALIZATION,
        "state_encoder": STATE_ENCODER,
    }
    for name, expected in expected_metadata.items():
        if manifest.get(name) != expected:
            raise ValueError(f"manifest[{name!r}] mismatch: expected {expected!r}, got {manifest.get(name)!r}")
    counts = np.load(base / "counts_hvg.npy", mmap_mode="r", allow_pickle=False)
    if counts.shape != (len(obs), len(genes)) or counts.dtype.kind not in "iuf":
        raise ValueError(f"Raw count array shape/type mismatch: {counts.shape}, {counts.dtype}")
    # A complete streaming scan prevents an already log-normalized matrix being
    # passed through the original normalization a second time.
    if full_scan:
        for start in range(0, len(obs), 1024):
            block = np.asarray(counts[start:start+1024])
            if not np.isfinite(block).all() or (block < 0).any() or not np.allclose(block, np.rint(block), atol=1e-5, rtol=0):
                raise ValueError("counts_hvg.npy must contain finite nonnegative RAW integer counts, not normalized expression")
    smiles = _json(base / "drug_smiles.json")
    if any(not isinstance(smiles.get(d), str) or not smiles[d] for d in training_drugs):
        raise ValueError("drug_smiles.json lacks a constituent drug's real molecular SMILES")
    state_path = base / "se_embeddings.npy"
    states = None
    if state_path.exists():
        if manifest.get("se_cell_order_sha256") != expected_metadata["cell_order_sha256"]:
            raise ValueError("The SE cache requires its own matching se_cell_order_sha256 in manifest.json")
        states = np.load(state_path, mmap_mode="r", allow_pickle=False)
        if states.shape != (len(obs), 2048) or states.dtype.kind != "f":
            raise ValueError(f"Expected float SE state [63378,2048], got {states.shape}/{states.dtype}")
        if full_scan:
            included = np.asarray([i for i, r in enumerate(obs) if r["split"] != "excluded"], dtype=np.int64)
            for start in range(0, len(included), 1024):
                if not np.isfinite(states[included[start:start+1024]]).all():
                    raise ValueError("SE states contain nonfinite values for included cells")
    elif require_state:
        raise FileNotFoundError(f"Missing real frozen SE-600M cache: {state_path}. Generate aligned 2048-D states first; no random fallback exists.")
    audit = {
        "protocol": PROTOCOL, "raw_cells": len(obs), "included_cells": 58201,
        "split_cells": split_counts, "control_cells": control_counts,
        "train_single_conditions": 6, "train_combination_conditions": 18,
        "test_conditions": sorted(TEST_CONDITIONS), "excluded_conditions": sorted(EXCLUDED_CONDITIONS),
        "test_drugs_seen_in_training": True, "validation": None,
        "gene_count": len(genes), **expected_metadata,
        "state_ready": states is not None, "state_dim": 2048, "condition_dim": 2052,
        "condition_pool": "canonical_sorted_direct_concat_frozen_MAPKG_2x1024", "dose": "neutral_zero",
        "cell_line_vocabulary": CELL_LINES, "test_used_for_gradient": False,
    }
    bundle = dict(base=base, obs=obs, counts=counts, states=states, genes=genes, drug_smiles=smiles, manifest=manifest, audit=audit)
    _BUNDLE_CACHE[key] = bundle
    return bundle


class ComboDataset:
    """Population sampler implementing the unchanged trainer's Dataset contract."""

    def __init__(self, cell_lines, split, base_dir, set_size=256, seed=42,
                 upstream_cache_dir=None, normalize_hvg=True, cellflow_sampling=True,
                 return_evaluator_expression=False, embedding_pca_transform=None,
                 state_encoder=None, full_population=False, _bundle=None, **unused):
        _deps()
        if list(cell_lines) != CELL_LINES or split not in ("train", "external_test"):
            raise ValueError("Combo task requires original three-slot cell-line vocabulary and train/external_test")
        if upstream_cache_dir or embedding_pca_transform is not None or state_encoder is not None or full_population or not normalize_hvg:
            raise ValueError("Combo task uses raw SE2048 states, no source branch/PCA/VAE, and one raw-count normalization")
        bundle = _bundle if _bundle is not None else load_bundle(base_dir, require_state=True)
        if bundle["states"] is None:
            raise FileNotFoundError("Real aligned SE state cache is not ready")
        self.bundle = bundle
        self.cell_lines, self.split, self.seed, self.epoch = list(cell_lines), split, int(seed), 0
        self.set_size, self.expression_dim = int(set_size), len(bundle["genes"])
        self.embedding_dim, self.full_population, self.cellflow_sampling = 2048, False, bool(cellflow_sampling)
        self.genes = bundle["genes"]
        self.expressions, self.embeddings = {"A549": bundle["counts"]}, {"A549": bundle["states"]}
        active_split = "train" if split == "train" else "test"
        self.control_rows = np.asarray([i for i, r in enumerate(bundle["obs"]) if r["split"] == active_split and not r["drugs"]], dtype=np.int64)
        self.controls = {"A549": {"merged": self.control_rows.tolist()}}
        if len(self.control_rows) < 2:
            raise ValueError("At least two actual controls are required")
        groups = {}
        for i, row in enumerate(bundle["obs"]):
            if row["split"] == active_split and row["drugs"]:
                groups.setdefault(row["condition"], []).append(i)
        self.records = []
        for condition, rows in sorted(groups.items()):
            drugs = bundle["obs"][rows[0]]["drugs"]
            self.records.append(dict(cell_line="A549", drug_id=condition, drug_index=len(self.records),
                                     condition=condition, dose_nm=0.0, replicate="merged", rows=rows,
                                     drugs=drugs, smiles=tuple(bundle["drug_smiles"][d] for d in drugs),
                                     stratum="single" if len(drugs) == 1 else "both_monotherapy" if all(d in SINGLE_DRUGS for d in drugs) else "one_monotherapy"))
        self._population_state_mean_cache = {}
        self._eval_rows = {}
        if split == "external_test":
            if self.set_size > len(self.control_rows):
                raise ValueError("Evaluation requires source controls without replacement")
            rng = np.random.default_rng(self.seed)
            self.eval_control_rows = rng.choice(self.control_rows, self.set_size, replace=False)
            for i, record in enumerate(self.records):
                if len(record["rows"]) < self.set_size:
                    raise ValueError("Evaluation requires enough real target cells without replacement")
                self._eval_rows[i] = np.random.default_rng(np.random.SeedSequence([self.seed, 1, i])).choice(record["rows"], self.set_size, replace=False)

    def __len__(self):
        return len(self.records)

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def _expression_tensor(self, expression, rows):
        values = torch.from_numpy(np.asarray(expression[rows]).copy()).float()
        return torch.log1p(1000.0 * values / values.sum(dim=1, keepdim=True).clamp_min(1.0))

    def _state_mean(self, rows, key):
        if key not in self._population_state_mean_cache:
            total = np.zeros(2048, dtype=np.float64)
            for start in range(0, len(rows), 512):
                total += np.asarray(self.embeddings["A549"][rows[start:start+512]], dtype=np.float64).sum(0)
            self._population_state_mean_cache[key] = torch.from_numpy((total / len(rows)).astype(np.float32))
        return self._population_state_mean_cache[key]

    def population_state_delta(self, index):
        if self.split != "train":
            raise RuntimeError("Test states cannot be used to construct training mean-shift targets")
        return self._state_mean(self.records[index]["rows"], ("target", index)) - self._state_mean(self.control_rows, "control")

    def __getitem__(self, index):
        record = self.records[index]
        if self.split == "external_test":
            control, target = self.eval_control_rows, self._eval_rows[index]
        else:
            rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch, int(index)]))
            control = rng.choice(self.control_rows, self.set_size, replace=self.cellflow_sampling or len(self.control_rows) < self.set_size)
            target = rng.choice(record["rows"], self.set_size, replace=self.cellflow_sampling or len(record["rows"]) < self.set_size)
        ctrl_expr = self._expression_tensor(self.expressions["A549"], control)
        target_expr = self._expression_tensor(self.expressions["A549"], target)
        return dict(control_state=torch.from_numpy(np.asarray(self.embeddings["A549"][control]).copy()).float(),
                    treated_state=torch.from_numpy(np.asarray(self.embeddings["A549"][target]).copy()).float(),
                    control_expression=ctrl_expr, treated_expression=target_expr,
                    control_evaluator_expression=ctrl_expr, treated_evaluator_expression=target_expr,
                    drug_id=record["drug_id"], drug_index=record["drug_index"], drug_names=record["drugs"],
                    drug_smiles=record["smiles"], dose_nm=torch.tensor(0.), cell_line="A549", replicate="merged",
                    control_rows=torch.as_tensor(control), treated_rows=torch.as_tensor(target), state_dim=2048)



def population_expression_mean(dataset, index):
    """Full merged-condition expression mean used by the decoder mean loss."""
    if dataset.split != "train":
        raise RuntimeError("Decoder loss population means require the training split")
    if not hasattr(dataset, "_joint_decoder_expression_means"):
        dataset._joint_decoder_expression_means = {}
    index = int(index)
    if index not in dataset._joint_decoder_expression_means:
        record = dataset.records[index]
        expression = dataset.expressions[record["cell_line"]]
        rows = np.asarray(record["rows"], dtype=np.int64)
        values = dataset._expression_tensor(expression, rows)
        dataset._joint_decoder_expression_means[index] = values.double().mean(0).float()
    return dataset._joint_decoder_expression_means[index]


def _population_stats(dataset, rows):
    total = np.zeros(dataset.expression_dim, dtype=np.float64)
    squared = np.zeros_like(total)
    for start in range(0, len(rows), 512):
        values = dataset._expression_tensor(dataset.expressions["A549"], rows[start:start+512]).numpy().astype(np.float64)
        total += values.sum(0)
        squared += (values * values).sum(0)
    mean = total / len(rows)
    var = np.maximum((squared - len(rows) * mean * mean) / max(len(rows)-1, 1), 0)
    return mean, var, len(rows)


def _corr(x, y, spearman=False):
    from scipy.stats import pearsonr, spearmanr
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float((spearmanr if spearman else pearsonr)(x, y)[0])


def _de_mask(target_stats, control_stats):
    from scipy.stats import ttest_ind_from_stats
    mean, var, n = target_stats
    ctrl, ctrl_var, ctrl_n = control_stats
    p = np.asarray(ttest_ind_from_stats(mean, np.sqrt(var), n, ctrl, np.sqrt(ctrl_var), ctrl_n, equal_var=False).pvalue)
    p = np.nan_to_num(p, nan=1., posinf=1., neginf=1.)
    order = np.argsort(p)
    adjusted = np.minimum.accumulate((p[order] * len(p) / np.arange(1, len(p)+1))[::-1])[::-1]
    result = np.empty(len(p), dtype=bool)
    result[order] = np.minimum(adjusted, 1.) < .05
    return result



def continuous_deg_weights(current_stats, rest_stats):
    current_mean, current_var, n_current = current_stats
    rest_mean, rest_var, n_rest = rest_stats
    current_mean, rest_mean = np.asarray(current_mean, dtype=np.float64), np.asarray(rest_mean, dtype=np.float64)
    if n_current < 2 or n_rest < 2:
        return np.zeros_like(current_mean)
    denominator = np.sqrt((np.maximum(current_var, 0.0) + np.maximum(rest_var, 0.0)) / n_current)
    score = np.divide(current_mean - rest_mean, denominator, out=np.zeros_like(current_mean), where=denominator > 0)
    absolute = np.nan_to_num(np.abs(score), nan=0.0, posinf=0.0, neginf=0.0)
    lo, hi = float(absolute.min()), float(absolute.max())
    if hi <= lo:
        return np.zeros_like(absolute)
    weights = np.square((absolute - lo) / (hi - lo))
    return weights / weights.sum() if weights.sum() > 0 else np.zeros_like(weights)


def _combine_stats(values):
    count = sum(stat[2] for stat in values)
    total = sum(np.asarray(mean) * n for mean, _, n in values)
    squared = sum(np.asarray(var) * (n - 1) + np.square(mean) * n for mean, var, n in values)
    mean = total / count
    variance = np.maximum((squared - np.square(mean) * count) / max(count - 1, 1), 0.0)
    return mean, variance, count


def training_binding(bundle):
    return dict(protocol=PROTOCOL,
                cell_order_sha256=bundle["manifest"]["cell_order_sha256"],
                gene_order_sha256=bundle["manifest"]["gene_order_sha256"],
                hvg_fit_cell_order_sha256=bundle["manifest"]["hvg_fit_cell_order_sha256"],
                train_conditions=sorted(TRAIN_CONDITIONS),
                control_rows_sha256=sequence_sha256([r["cell_id"] for r in bundle["obs"]
                                                     if r["split"] == "train" and not r["drugs"]]))


def export_evaluation_references(base_dir, output_dir, seeds=(4242, 4243, 4244, 4245, 4246)):
    """Export the original full-treated scoring references, never loss inputs."""
    from benchmark.export_references import write_reference_bundle
    from devflow.io import write_json, json_sha256
    bundle = load_bundle(base_dir, full_scan=True)
    datasets = {seed: ComboDataset(CELL_LINES, "external_test", base_dir,
                                  set_size=128, seed=seed, _bundle=bundle) for seed in seeds}
    first = datasets[seeds[0]]
    groups = {}
    for index, row in enumerate(bundle["obs"]):
        if row["split"] in ("train", "test") and row["drugs"]:
            groups.setdefault(row["condition"], []).append(index)
    stats = {name: _population_stats(first, np.asarray(rows, dtype=np.int64))
             for name, rows in groups.items()}
    if set(groups) != set(TRAIN_CONDITIONS | TEST_CONDITIONS):
        raise ValueError("Included treated groups differ from the paper protocol")
    names = [record["condition"] for record in first.records]
    weights = np.stack([continuous_deg_weights(stats[name], _combine_stats(
        [value for other, value in stats.items() if other != name])) for name in names])
    perturbation_mean = np.stack([stats[name][0] for name in sorted(TRAIN_CONDITIONS)]).mean(0)
    samples, masks = {}, {}
    for seed, dataset in datasets.items():
        control_stats = _population_stats(dataset, dataset.eval_control_rows)
        masks[seed] = np.stack([_de_mask(stats[name], control_stats) for name in names])
        sampled = [dataset[index] for index in range(len(dataset))]
        payload = dict(seed=np.asarray(seed), condition_names=np.asarray(names),
                       source_global_ids=np.asarray([[bundle["obs"][int(i)]["cell_id"] for i in sample["control_rows"]]
                                                     for sample in sampled]),
                       target_global_ids=np.asarray([[bundle["obs"][int(i)]["cell_id"] for i in sample["treated_rows"]]
                                                     for sample in sampled]),
                       gene_order_sha256=np.asarray(bundle["manifest"]["gene_order_sha256"]),
                       de_mask=masks[seed])
        for index, sample in enumerate(sampled):
            payload[f"target_{index:04d}"] = sample["treated_expression"].numpy()
            payload[f"control_{index:04d}"] = sample["control_expression"].numpy()
        samples[seed] = payload
    binding = training_binding(bundle)
    static = dict(condition_names=np.asarray(names),
                  gene_order_sha256=np.asarray(bundle["manifest"]["gene_order_sha256"]),
                  context_names=np.asarray(["A549"]*len(names)),
                  reporting_scopes=np.asarray(["A" if r["stratum"] == "both_monotherapy" else "B" for r in first.records]),
                  continuous_deg_weights=weights, de_mask=masks[seeds[0]],
                  full_target_profiles=np.stack([stats[name][0] for name in names]),
                  training_condition_mean=perturbation_mean[None],
                  reference_index=np.zeros(len(names), dtype=np.int64))
    manifest = write_reference_bundle(output_dir, statistics=static, samples=samples,
        training_provenance=dict(fitted_on="training_only", training_binding=binding,
            fit_row_ids_sha256=sequence_sha256([r["cell_id"] for r in bundle["obs"]
                                                if r["split"] == "train" and r["drugs"]]),
            averaging="equal weight across 24 train treated conditions"),
        evaluation_provenance=dict(use="reporting_only", test_derived=["full_target_profiles", "continuous_deg_weights", "de_mask"],
            continuous_weights="full current condition versus remaining 28 included treated conditions",
            de_mask="per-seed full target versus sampled128 control Welch/BH q<.05"))
    template = Path(__file__).resolve().parents[2]/"benchmark/protocols/combosciplex_v3.json"
    protocol = json.loads(template.read_text())
    protocol.update(seeds=list(seeds), gene_order_sha256=bundle["manifest"]["gene_order_sha256"],
                    training_binding_sha256=json_sha256(binding))
    write_json(Path(output_dir)/"protocol.json", protocol)
    return manifest
