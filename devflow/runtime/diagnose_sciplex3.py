"""CPU-only fixed-latent decoder replay for SciPlex3 error diagnosis.

Oracle true-treated SE latents contain target information. Their scores are
diagnostics, never prospective prediction benchmark scores.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import time
import numpy as np
import pandas as pd
import torch

from benchmark.metrics import score_condition
from devflow.data.features import FrozenFeatureCache
from devflow.data.sciplex3 import CELL_LINES, SciPlexData
from devflow.networks.expression_decoder import LatentToGeneDecoder
from devflow.runtime.artifacts import atomic_json, ordered_sha256, sha256
from devflow.runtime.evaluate_sciplex3 import sampled_records


def file_identity(path):
    path = Path(path)
    return dict(path=str(path.resolve()), bytes=path.stat().st_size, sha256=sha256(path))


def load_decoder(state):
    decoder = LatentToGeneDecoder(2048, 2000, [512, 1024], dropout=0.1, output_activation="relu")
    decoder.load_state_dict(state, strict=True)
    return decoder.float().cpu().eval().requires_grad_(False)


def decoder_state(payload):
    if "decoder_state_dict" in payload:
        return payload["decoder_state_dict"]
    return {key.removeprefix("decoder."): value for key, value in payload["model"].items()
            if key.startswith("decoder.")}


def macro_summary(rows):
    frame = pd.DataFrame(rows)
    metadata = ["latent_source", "decoder", "oracle", "panel"]
    metrics = [column for column in frame if column not in metadata + ["group_index", "condition", "cell_line"]]
    summaries = []
    for keys, group in frame.groupby(metadata, sort=False):
        for scope in ("pooled", *CELL_LINES):
            selected = group if scope == "pooled" else group[group.cell_line == scope]
            row = dict(zip(metadata, keys), scope=scope, n_conditions=len(selected))
            for metric in metrics:
                values = selected[metric].to_numpy(dtype=np.float64)
                finite = np.isfinite(values)
                row[metric] = float(values[finite].mean()) if finite.any() else np.nan
                row[metric + "_n_finite"] = int(finite.sum())
            summaries.append(row)
    return frame, pd.DataFrame(summaries)


def analysis(summary, output, provenance):
    def take(latent, decoder, panel="hvg"):
        row = summary[(summary.latent_source == latent) & (summary.decoder == decoder) &
                      (summary.panel == panel) & (summary.scope == "pooled")]
        assert len(row) == 1
        return row.iloc[0]
    predicted, oracle = "flow19k_predicted", "oracle_true_treated_frozen_SE"
    current, pretrained = "joint19k", "original_true_SE_pretrained"
    lines = [
        "# SciPlex3 19k decoder 诊断",
        "",
        "本诊断固定 108 个外测条件、seed4242 和每条件相同的 128 个 treated/control row IDs，",
        "只在 CPU FP32 下重放 decoder；不重新运行 Flow，不训练、不调参、不使用 GPU。",
        "Flow latent 来自现有 BF16、RK4-10 在线基线；FP32 解码重放与原在线 BF16 表达结果可能有微小数值差异。",
        "",
        "**oracle 行使用真实 treated 细胞的 frozen-SE latent，包含目标信息，只用于定位误差，不能作为预测方法的 benchmark 分数。**",
        "",
    ]
    for panel in ("hvg", "de50"):
        lines += [f"## {panel.upper()}：条件等权宏平均（单 seed，无重复采样 SD）", "",
                  "| Latent | Decoder | Pearson ΔCtrl ↑ | Pearson ΔPert ↑ | wMSE ↓ | MAE ↓ |",
                  "|---|---|---:|---:|---:|---:|"]
        for latent in (predicted, oracle):
            for dec in (current, pretrained):
                row = take(latent, dec, panel)
                lines.append(f"| {latent} | {dec} | {row.pearson_delta:.6f} | {row.pearson_deltapert:.6f} | {row.weighted_mse:.6f} | {row.pseudobulk_mae:.6f} |")
        lines.append("")
    p19, ppre, o19, opre = (take(l, d) for l, d in
                             ((predicted, current), (predicted, pretrained), (oracle, current), (oracle, pretrained)))
    lines += [
        "## 可支持的判断",
        "",
        f"固定预测 latent，19k decoder → 原预训练 decoder：ΔPert {p19.pearson_deltapert:.6f} → {ppre.pearson_deltapert:.6f}，"
        f"wMSE {p19.weighted_mse:.6f} → {ppre.weighted_mse:.6f}。这反映两个 decoder 对同一 Flow 端点的适配差异。",
        "",
        f"固定 oracle latent，19k decoder → 原预训练 decoder：ΔPert {o19.pearson_deltapert:.6f} → {opre.pearson_deltapert:.6f}，"
        f"wMSE {o19.weighted_mse:.6f} → {opre.weighted_mse:.6f}。这直接比较两者对真实 frozen-SE 表示的解码能力；"
        "若19k较差，说明其在这些外测真实SE输入上的重建能力退化，但单次外测诊断不能独立证明训练因果机制。",
        "",
        f"固定19k decoder，预测 latent → oracle latent：ΔPert {p19.pearson_deltapert:.6f} → {o19.pearson_deltapert:.6f}，"
        f"wMSE {p19.weighted_mse:.6f} → {o19.weighted_mse:.6f}。",
        f"固定原预训练 decoder，对应变化为 ΔPert {ppre.pearson_deltapert:.6f} → {opre.pearson_deltapert:.6f}，"
        f"wMSE {ppre.weighted_mse:.6f} → {opre.weighted_mse:.6f}。较大差距支持继续改善 Flow 端点与真实 treated 表示的分布匹配。",
        "",
        "这四组比较用于缩小模型误差来源，不能把 Flow 和 decoder 的误差作可加的因果分解。"
        "预测细胞与 treated 细胞没有逐细胞配对，latent_diagnostics.csv 只比较条件均值和群体尺度，不计算伪配对误差。",
        "",
        "summary.csv 包含 pooled 与三种细胞系、HVG2000 与 fixed-DE50 面板的全部均值指标及有限值计数；"
        "per_condition.csv 保存逐条件分数。MMD/EMD/energy distance 未在本次均值诊断中计算。",
        "",
        f"CUDA initialized: {provenance['cuda_initialized']}；decoder dtype: float32；无 optimizer、无梯度、无训练。",
    ]
    latent_summary = pd.read_csv(output / "latent_diagnostics.csv").groupby("cell_line").mean(numeric_only=True)
    lines += ["", "## 端点群体尺度", "",
              "| Cell line | 预测 latent 条件内平均方差 | 真实 SE 条件内平均方差 | 方差比（预测/真实） |",
              "|---|---:|---:|---:|"]
    for line in CELL_LINES:
        row = latent_summary.loc[line]
        ratio = row.predicted_mean_cell_variance / row.oracle_mean_cell_variance
        lines.append(f"| {line} | {row.predicted_mean_cell_variance:.8g} | {row.oracle_mean_cell_variance:.8g} | {ratio:.6%} |")
    lines += ["",
        "现有预测端点存在明显的条件内群体收缩；条件均值的高余弦相似度不足以说明端点分布正确。",
        "",
        "在真实SE输入上，19k decoder 的 wMSE 为原预训练 decoder 的 "
        f"{o19.weighted_mse / opre.weighted_mse:.2f} 倍；而直接将原 decoder 接到现有预测端点，wMSE 反而变为当前的 "
        f"{ppre.weighted_mse / p19.weighted_mse:.2f} 倍。当前证据支持同时关注端点分布与解码保真。",
        "",
        "后续模型实验可优先考虑：用训练 treated 的真实 frozen-SE latent 增加 decoder 重建锚定，"
        "保留原始SE解码能力；在训练条件内约束预测端点的方差/分布，或调整均值与残差损失的平衡。"
        "这些方向需要独立验证，权重应由训练内验证确定，不能在这些外测诊断结果上拟合。"]
    (output / "analysis_zh.md").write_text("\n".join(lines) + "\n")


@torch.inference_mode()
def run(args):
    started = time.time()
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.set_float32_matmul_precision("highest")
    if torch.cuda.is_initialized():
        raise RuntimeError("CUDA was unexpectedly initialized")
    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text())
    tuning_root = Path(config["output_dir"]).parent
    baseline = Path(args.baseline_dir or tuning_root / "baseline19k/evaluations/update_019000")
    output = Path(args.output or tuning_root / "decoder_diagnostics")
    output.mkdir(parents=True, exist_ok=True)
    settings = config["data"]
    data = SciPlexData(settings["base_dir"], settings["sampling_manifest"],
                      source_split_root=settings.get("source_split_root"))
    cache = FrozenFeatureCache(settings["feature_cache"])
    if cache.manifest["data_binding"] != data.training_binding():
        raise ValueError("Frozen cache is not bound to this training union")
    records = sampled_records(data, seed=4242, cells=128)
    refs = np.load(settings["evaluation_references"], allow_pickle=False)
    de = np.load(settings["de50_references"], allow_pickle=False)
    names = [r["condition"] for r in records]
    if refs["condition_names"].tolist() != names or de["group_names"].tolist() != names:
        raise ValueError("Reference condition order differs")
    if str(refs["gene_order_sha256"]) != data.gene_order_sha256:
        raise ValueError("Reference gene axis differs")
    indices = de["gene_indices"]
    if indices.shape != (108, 50) or indices.dtype.kind not in "iu" or np.any(indices < 0) or np.any(indices >= 2000):
        raise ValueError("DE50 indices are invalid")
    if any(len(np.unique(row)) != 50 for row in indices):
        raise ValueError("DE50 duplicates")
    if not np.array_equal(de["full_weights"], refs["continuous_deg_weights"]):
        raise ValueError("DE50 and HVG weights differ")

    checkpoint_path = Path(config["training"]["checkpoint"])
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu", mmap=True, weights_only=False)
    if checkpoint["update"] != 19000 or checkpoint["training_binding"] != data.training_binding():
        raise ValueError("Checkpoint update or training data binding differs")
    pretrained_path = Path(checkpoint["config"]["decoder_checkpoint"])
    pretrained_hash = sha256(pretrained_path)
    if pretrained_hash != checkpoint["initialization_hashes"]["decoder"]:
        raise ValueError("Original decoder initialization hash differs")
    pretrained = torch.load(str(pretrained_path), map_location="cpu", mmap=True, weights_only=False)
    if pretrained["gene_order_sha256"] != data.gene_order_sha256:
        raise ValueError("Pretrained decoder gene axis differs")
    decoders = dict(joint19k=load_decoder(decoder_state(checkpoint)),
                    original_true_SE_pretrained=load_decoder(decoder_state(pretrained)))
    targets = [data.expression(record["cell_line"], record["target_rows"]) for record in records]
    profiles = np.stack([target.mean(0) for target in targets])
    group_files = sorted(baseline.glob("group_*.npz"))
    if [p.name for p in group_files] != [f"group_{i:04d}.npz" for i in range(108)]:
        raise ValueError("Expected exactly 108 completed baseline files")
    rows, latent_rows, sample_provenance, baseline_delta = [], [], [], []
    for index, (record, path) in enumerate(zip(records, group_files)):
        saved = np.load(path, allow_pickle=False)
        if not np.array_equal(saved["source_rows"], record["control_rows"]) or not np.array_equal(saved["target_rows"], record["target_rows"]):
            raise ValueError(f"Baseline sampling differs at group {index}")
        pred_z = np.asarray(saved["latent"], np.float32)
        oracle_z = cache.cell_embeddings(record["cell_line"], record["target_rows"], device="cpu").numpy()
        if pred_z.shape != (128, 2048) or oracle_z.shape != pred_z.shape:
            raise ValueError("Latent dimensions differ")
        if not np.isfinite(pred_z).all() or not np.isfinite(oracle_z).all():
            raise ValueError("Nonfinite latent")
        control = data.expression(record["cell_line"], record["control_rows"])
        positions = [i for i, r in enumerate(records) if r["cell_line"] == record["cell_line"]]
        center_error = pred_z.mean(0).astype(np.float64) - oracle_z.mean(0)
        pmean, omean = pred_z.mean(0).astype(np.float64), oracle_z.mean(0).astype(np.float64)
        latent_rows.append(dict(group_index=index, condition=record["condition"], cell_line=record["cell_line"],
            latent_mean_mse=float(np.square(center_error).mean()),
            latent_mean_cosine=float(np.dot(pmean, omean) / (np.linalg.norm(pmean) * np.linalg.norm(omean))),
            predicted_mean_cell_variance=float(pred_z.var(0).mean()),
            oracle_mean_cell_variance=float(oracle_z.var(0).mean())))
        for latent_source, latent, oracle in (("flow19k_predicted", pred_z, False),
                    ("oracle_true_treated_frozen_SE", oracle_z, True)):
            for decoder_name, decoder in decoders.items():
                prediction = decoder(torch.from_numpy(latent)).numpy()
                if prediction.dtype != np.float32 or not np.isfinite(prediction).all():
                    raise ValueError("Decoder prediction is not finite FP32")
                if latent_source == "flow19k_predicted" and decoder_name == "joint19k":
                    baseline_delta.append(float(np.abs(prediction - saved["prediction"]).max()))
                score_args = (prediction, targets[index], control, refs["continuous_deg_weights"][index],
                              refs["de_mask"][index], refs["training_condition_mean"][CELL_LINES.index(record["cell_line"])],
                              profiles[positions], positions.index(index))
                for panel in ("hvg", "de50"):
                    values = score_condition(*score_args, include_distribution=False, panel=panel,
                                             gene_indices=indices[index] if panel == "de50" else None)
                    rows.append(dict(latent_source=latent_source, decoder=decoder_name, oracle=oracle,
                                     panel=panel, group_index=index, condition=record["condition"],
                                     cell_line=record["cell_line"], **values))
        sample_provenance.append(dict(group_index=index, condition=record["condition"],
            baseline_npz=file_identity(path),
            control_global_ids_sha256=ordered_sha256(data.global_ids(record["cell_line"], record["control_rows"])),
            target_global_ids_sha256=ordered_sha256(data.global_ids(record["cell_line"], record["target_rows"]))))
        if (index + 1) % 12 == 0:
            print(json.dumps(dict(phase="decoder_replay", completed=index + 1, total=108)), flush=True)

    per_condition, summary = macro_summary(rows)
    per_condition.to_csv(output / "per_condition.csv", index=False)
    summary.to_csv(output / "summary.csv", index=False)
    pd.DataFrame(latent_rows).to_csv(output / "latent_diagnostics.csv", index=False)
    print(json.dumps(dict(phase="hashing_provenance")), flush=True)
    provenance = dict(schema="sciplex3_cpu_decoder_diagnostic_v1",
        created_utc=datetime.now(timezone.utc).isoformat(), host=platform.node(),
        config=file_identity(config_path), checkpoint=file_identity(checkpoint_path),
        original_decoder=file_identity(pretrained_path),
        original_decoder_initialization_hash_verified=True,
        cache_manifest=file_identity(Path(settings["feature_cache"]) / "manifest.json"),
        cache_cell_embeddings={line: file_identity(Path(settings["feature_cache"]) / line / "cell_embeddings.npy") for line in CELL_LINES},
        evaluation_references=file_identity(settings["evaluation_references"]),
        de50_references=file_identity(settings["de50_references"]),
        sampling_manifest=file_identity(settings["sampling_manifest"]),
        gene_order_sha256=data.gene_order_sha256, training_binding=data.training_binding(),
        source_code={name: file_identity(name) for name in
                     [__file__, "devflow/networks/expression_decoder.py", "benchmark/metrics/sciplex3.py",
                      "devflow/data/sciplex3.py", "devflow/data/features.py"]},
        baseline_summary=file_identity(baseline / "summary.json"),
        baseline_protocol=json.loads((baseline / "summary.json").read_text()),
        diagnostic_protocol=dict(seed=4242, cells=128, conditions=108, decoder_device="cpu",
            decoder_dtype="float32", fixed_latent_source="existing BF16 RK4-10 chunk8 baseline",
            reran_flow=False, trained=False, optimizer_created=False, gradients_enabled=False,
            oracle_contains_target_information=True, oracle_is_benchmark=False,
            aggregation="equal condition macro average", sampling_repeats=1,
            distribution_metrics_computed=False, torch_threads=args.threads),
        maximum_abs_change_vs_original_bf16_expression=float(max(baseline_delta)),
        samples=sample_provenance, cuda_initialized=torch.cuda.is_initialized(),
        torch_version=torch.__version__, numpy_version=np.__version__, seconds=time.time() - started)
    if provenance["cuda_initialized"]:
        raise RuntimeError("CUDA was initialized during CPU-only diagnosis")
    atomic_json(output / "provenance.json", provenance)
    analysis(summary, output, provenance)
    print(summary[summary.scope == "pooled"].to_string(index=False), flush=True)
    print(json.dumps(dict(output=str(output), seconds=time.time() - started, cuda_initialized=False)), flush=True)



def population_stats(values):
    values = np.asarray(values, np.float64)
    variance = float(values.var(axis=0).mean())
    return dict(mean_feature_variance=variance, rms_within_population_std=float(np.sqrt(variance)),
                rms_coordinate=float(np.sqrt(np.square(values).mean())),
                mean_norm=float(np.linalg.norm(values, axis=1).mean()))


@torch.inference_mode()
def followup(args):
    from devflow.networks.state_encoder import SEControlLoRATail
    from devflow.flow.matching import squared_euclidean_cost, balanced_sinkhorn
    started = time.time()
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    config = json.loads(Path(args.config).read_text())
    output = Path(args.output or Path(config["output_dir"]).parent / "decoder_diagnostics")
    output.mkdir(parents=True, exist_ok=True)
    settings = config["data"]
    data = SciPlexData(settings["base_dir"], settings["sampling_manifest"],
                      source_split_root=settings.get("source_split_root"))
    cache = FrozenFeatureCache(settings["feature_cache"])
    binding = data.training_binding()
    if cache.manifest["data_binding"] != binding:
        raise ValueError("Cache data binding differs")
    checkpoint_path = Path(config["training"]["checkpoint"])
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu", mmap=True, weights_only=False)
    if checkpoint["update"] != 19000 or checkpoint["training_binding"] != binding:
        raise ValueError("Expected original 19k checkpoint and matching data")
    checkpoint_hash = sha256(checkpoint_path)
    prefix_root = Path(settings["prefix_cache"])
    prefix_manifest = json.loads((prefix_root / "manifest.json").read_text())
    if prefix_manifest["training_binding"] != binding or prefix_manifest["frozen_cache_manifest_sha256"] != sha256(Path(settings["feature_cache"]) / "manifest.json"):
        raise ValueError("Prefix/cache data binding differs")
    tail_path = Path(config["model"]["tail_checkpoint"])
    if sha256(tail_path) != checkpoint["initialization_hashes"]["tail"]:
        raise ValueError("Tail initialization hash differs")
    tail = SEControlLoRATail.load_tail(tail_path).float().cpu().eval()
    original_frozen_hash = tail.frozen_base_sha256()
    tail.load_state_dict({key.removeprefix("flow.control_lora."): value for key, value in checkpoint["model"].items()
                          if key.startswith("flow.control_lora.")}, strict=True)
    tail.requires_grad_(False)
    if tail.frozen_base_sha256() != original_frozen_hash:
        raise ValueError("Frozen tail weights changed")
    controls, prefix_provenance, source_rows = {}, {}, []
    for line in CELL_LINES:
        folder = prefix_root / line
        ids = np.load(folder / "control_rows.npy")
        globals_ = np.load(folder / "control_global_ids.npy")
        if not np.array_equal(ids, cache.control_rows(line)) or not np.array_equal(globals_, data.global_ids(line, ids)):
            raise ValueError("Control prefix row mapping differs")
        if sha256(folder / "prefix.npy") != prefix_manifest["lines"][line]["prefix_sha256"]:
            raise ValueError("Prefix hash differs")
        prefix = np.load(folder / "prefix.npy", mmap_mode="r")
        original, trained = [], []
        for start in range(0, len(ids), 128):
            tensor = torch.from_numpy(np.array(prefix[start:start+128], np.float32))
            original.append(tail.base_forward(tensor).numpy())
            trained.append(tail(tensor).numpy())
        original, trained = np.concatenate(original), np.concatenate(trained)
        frozen = cache.cell_embeddings(line, ids, device="cpu").numpy()
        lookup = {int(row): i for i, row in enumerate(ids)}
        controls[line] = dict(ids=ids, lookup=lookup, frozen=frozen, base=original, trained=trained)
        ref_var = population_stats(frozen)["mean_feature_variance"]
        for mode, values in (("frozen_cached_SE", frozen), ("original_base_tail_CPU_FP32", original),
                             ("trained19k_LoRA_tail_CPU_FP32", trained)):
            stats = population_stats(values)
            source_rows.append(dict(cell_line=line, scope="full_control_pool", representation=mode,
                                    n_cells=len(ids), **stats,
                                    variance_ratio_to_frozen=stats["mean_feature_variance"]/ref_var,
                                    max_abs_vs_frozen=float(np.abs(values-frozen).max()),
                                    mse_vs_frozen=float(np.square(values.astype(np.float64)-frozen).mean()),
                                    centered_mse_vs_frozen=float(np.square(
                                        (values.astype(np.float64)-values.astype(np.float64).mean(0))-
                                        (frozen.astype(np.float64)-frozen.astype(np.float64).mean(0))).mean())))
        prefix_provenance[line] = dict(prefix=file_identity(folder / "prefix.npy"),
                                       row_ids_sha256=ordered_sha256(ids))
        print(json.dumps(dict(phase="tail_control_encoding", cell_line=line, n_controls=len(ids))), flush=True)
    pd.DataFrame(source_rows).to_csv(output / "source_variance.csv", index=False)

    formal_dir = Path(args.formal_dir or checkpoint_path.parent.parent /
                      "benchmark_checkpoint_019000_rollout80_20261026/group_predictions/seed_4243")
    records = data.sampling(4243)
    formal_rows, formal_provenance = [], []
    for index, record in enumerate(records):
        path = formal_dir / f"group_{index:04d}.npz"
        saved = np.load(path, allow_pickle=False)
        if int(saved["seed"]) != 4243 or int(saved["group_index"]) != index or str(saved["condition"]) != record["condition"]:
            raise ValueError("Formal record identity differs")
        for key, expected in (("source_rows", record["control_rows"]), ("target_rows", record["target_rows"]),
                              ("source_global_ids", record["control_global_ids"]), ("target_global_ids", record["treated_global_ids"])):
            if not np.array_equal(saved[key], expected):
                raise ValueError(f"Formal {key} differs")
        if str(saved["checkpoint_sha256"]) != checkpoint_hash or str(saved["gene_order_sha256"]) != data.gene_order_sha256:
            raise ValueError("Formal checkpoint/gene hash differs")
        formal_config = json.loads(str(saved["config_json"]))
        if not (formal_config["time_points"] == 80 and formal_config["precision"] == "fp32"
                and formal_config["autocast_enabled"] is False and formal_config["allow_tf32"] is False):
            raise ValueError("Expected formal FP32 RK4-80 with TF32 disabled")
        line = record["cell_line"]
        positions = [controls[line]["lookup"][int(row)] for row in record["control_rows"]]
        frozen, trained = (controls[line][key][positions] for key in ("frozen", "trained"))
        target = cache.cell_embeddings(line, record["target_rows"], device="cpu").numpy()
        endpoint = saved["latent"]
        if endpoint.shape != (128,2048) or endpoint.dtype != np.float32 or not np.isfinite(endpoint).all():
            raise ValueError("Invalid formal endpoint")
        row = dict(group_index=index, condition=record["condition"], cell_line=line, seed=4243)
        stats = {name:population_stats(values) for name,values in
                 (("frozen_source",frozen),("lora_source",trained),("true_treated",target),("formal80_endpoint",endpoint))}
        for name, values in stats.items():
            row.update({name + "_" + key: value for key,value in values.items()})
        row.update(endpoint_to_true_variance=stats["formal80_endpoint"]["mean_feature_variance"]/stats["true_treated"]["mean_feature_variance"],
                   lora_to_frozen_source_variance=stats["lora_source"]["mean_feature_variance"]/stats["frozen_source"]["mean_feature_variance"],
                   endpoint_to_lora_source_variance=stats["formal80_endpoint"]["mean_feature_variance"]/stats["lora_source"]["mean_feature_variance"])
        formal_rows.append(row)
        formal_provenance.append(dict(group_index=index, archive=file_identity(path)))
    formal_frame = pd.DataFrame(formal_rows)
    formal_frame.to_csv(output / "variance_formal80.csv", index=False)

    # Predeclared training-only diagnostic subset: four conditions per cell line,
    # evenly spaced in original sorted condition order. This does not fit weights.
    ot_rows = []
    chosen = []
    for line in CELL_LINES:
        pool = [record for record in data.train_records if record["cell_line"] == line]
        chosen.extend(pool[i] for i in np.linspace(0,len(pool)-1,4,dtype=int))
    for i, record in enumerate(chosen):
        line = record["cell_line"]
        rng = np.random.default_rng(np.random.SeedSequence([731991,i]))
        sr = rng.choice(record["control_pool"],256,replace=len(record["control_pool"])<256)
        tr = rng.choice(record["rows"],256,replace=len(record["rows"])<256)
        positions = [controls[line]["lookup"][int(row)] for row in sr]
        source = torch.from_numpy(controls[line]["trained"][positions])
        target = cache.cell_embeddings(line,tr,device="cpu")
        cost = squared_euclidean_cost(source,target)
        plan, stats = balanced_sinkhorn(cost,epsilon=.01,max_iterations=2000,tolerance=1e-3)
        conditional = plan / plan.sum(1,keepdim=True)
        entropy = -(conditional * conditional.clamp_min(1e-30).log()).sum(1).mean()
        cost_np = cost.numpy()
        source_stats, target_stats = population_stats(source.numpy()), population_stats(target.numpy())
        row = dict(condition=record["condition"],cell_line=line,training_only=True,n_candidates=256,
                   source_rows_sha256=ordered_sha256(sr),target_rows_sha256=ordered_sha256(tr),
                   cost_mean=float(cost.mean()), cost_std=float(cost.std(unbiased=False)),
                   cost_min=float(cost.min()),cost_p05=float(np.quantile(cost_np,.05)),
                   cost_p50=float(np.quantile(cost_np,.5)),cost_p95=float(np.quantile(cost_np,.95)),cost_max=float(cost.max()),
                   normalized_cost_std=float(cost.std(unbiased=False)/cost.mean()),
                   epsilon_normalized=.01,effective_epsilon_raw=.01*float(cost.mean()),
                   mean_row_centered_cost_std=float(cost.std(1,unbiased=False).mean()),
                   mean_col_centered_cost_std=float(cost.std(0,unbiased=False).mean()),
                   conditional_entropy=float(entropy),entropy_fraction_of_uniform=float(entropy/np.log(256)),
                   effective_targets_per_source=float(entropy.exp()),
                   mean_conditional_max_probability=float(conditional.max(1).values.mean()),
                   mean_total_variation_from_uniform=float((conditional-1/256).abs().sum(1).mean()/2),
                   transport_cost=float((plan*cost).sum()/plan.sum()),
                   ot_cost_ratio_to_random=float(((plan*cost).sum()/plan.sum())/cost.mean()),
                   sinkhorn_iterations=stats.iterations,sinkhorn_converged=stats.converged,marginal_error=stats.marginal_error,
                   source_rms_within_std=source_stats["rms_within_population_std"],
                   target_rms_within_std=target_stats["rms_within_population_std"],
                   noise_sigma=.01,noise_rms_L2=.01*np.sqrt(2048),
                   noise_to_source_within_std=.01/source_stats["rms_within_population_std"],
                   noise_to_target_within_std=.01/target_stats["rms_within_population_std"])
        ot_rows.append(row)
        print(json.dumps(dict(phase="training_OT_measurement",completed=i+1,total=len(chosen),
                             entropy_fraction=row["entropy_fraction_of_uniform"])),flush=True)
    ot_frame = pd.DataFrame(ot_rows)
    ot_frame.to_csv(output / "ot_training_scale.csv",index=False)
    lines = ["# SciPlex3 19k：正式端点、LoRA 输入与 OT 尺度诊断", "",
             "全程 CPU FP32，只读取既有19k模型和正式FP32 RK4-80 latent，不重新运行Flow、不训练。"
             "真实treated SE仅用于离线诊断，未作为方法预测输入。", "",
             "## 正式 seed4243、108条件端点", "",
             "| Cell line | Endpoint / true treated 方差比 | LoRA source / frozen source 方差比 | Endpoint / LoRA source 方差比 |",
             "|---|---:|---:|---:|"]
    for line in CELL_LINES:
        group=formal_frame[formal_frame.cell_line==line]
        lines.append(f"| {line} | {group.endpoint_to_true_variance.mean():.6%} | {group.lora_to_frozen_source_variance.mean():.6%} | {group.endpoint_to_lora_source_variance.mean():.6%} |")
    lines += ["", "表格为每条件方差比的等权宏平均，方差均指各维在128细胞内的总体方差后对2048维取均值。"
              "所有108组的checkpoint、基因、条件与treated/control全局row IDs已核验。", "",
              "## 全部control池的源表示", "",
              "| Cell line | Frozen SE方差 | 19k LoRA源方差 | LoRA / frozen |",
              "|---|---:|---:|---:|"]
    source_frame=pd.DataFrame(source_rows)
    for line in CELL_LINES:
        group=source_frame[source_frame.cell_line==line]
        f=group[group.representation=="frozen_cached_SE"].iloc[0]
        t=group[group.representation=="trained19k_LoRA_tail_CPU_FP32"].iloc[0]
        lines.append(f"| {line} | {f.mean_feature_variance:.8g} | {t.mean_feature_variance:.8g} | {t.variance_ratio_to_frozen:.6%} |")
    lines += ["", "source_variance.csv 同时记录原base tail的CPU FP32重建对照与缓存差异，以区分精度差异。"
              "源LoRA和端点两阶段的比较为观测诊断，不等同于可加的因果分解。", "",
              "## OT和路径噪声实测", "",
              "实现使用未除维度的平方欧氏距离，随后除以整张cost矩阵的平均值，再使用epsilon=0.01。"
              "因此原始cost单位的等效epsilon是0.01×cost.mean，不能直接将0.01与未归一化cost比较。", "",
              f"在12个预先固定的训练条件（每细胞系4个，原排序等距抽取）上，"
              f"raw cost.mean范围 {ot_frame.cost_mean.min():.6g}–{ot_frame.cost_mean.max():.6g}，"
              f"等效raw epsilon范围 {ot_frame.effective_epsilon_raw.min():.6g}–{ot_frame.effective_epsilon_raw.max():.6g}。"
              f"条件OT熵占uniform熵范围 {ot_frame.entropy_fraction_of_uniform.min():.6%}–{ot_frame.entropy_fraction_of_uniform.max():.6%}，"
              f"每source的有效target数 {ot_frame.effective_targets_per_source.min():.3f}–{ot_frame.effective_targets_per_source.max():.3f}/256；"
              f"平均最大target概率范围 {ot_frame.mean_conditional_max_probability.min():.6g}–{ot_frame.mean_conditional_max_probability.max():.6g}。", "",
              f"constant_noise_path 在所有t直接加sigma=0.01的独立逐维高斯噪声；"
              f"2048维噪声RMS L2={.01*np.sqrt(2048):.6f}，SE输出经L2归一化，单细胞范数约1。"
              f"噪声sigma/源条件内RMS标准差范围 {ot_frame.noise_to_source_within_std.min():.3f}–{ot_frame.noise_to_source_within_std.max():.3f}，"
              f"对应treated范围 {ot_frame.noise_to_target_within_std.min():.3f}–{ot_frame.noise_to_target_within_std.max():.3f}。", "",
              "OT测量使用CPU FP32 tail与256候选，属于训练分布诊断，不承诺逐位复现训练GPU BF16的离散配对。"
              "这些观测能支持选择固定架构的续训对照，但不能单凭相关性认定OT或噪声是收缩的唯一原因。",
              ""]
    trained_rows = source_frame[source_frame.representation=="trained19k_LoRA_tail_CPU_FP32"]
    lines += ["## 固定架构的最小干预建议", "",
              "优先方案1：增加训练control的源表示保真损失。令同一control细胞的原冻结SE为s0、"
              "当前LoRA输出为s，对细胞系内同批control计算 "
              "L_src = mean(((s-mean_cells(s))-(s0-mean_cells(s0)))^2)。"
              "s0只作训练监督并停止梯度，保持现有rank=8、tail和Flow结构；推理仍只用control与扰动条件，"
              "不读取target SE。该中心化锚定允许源均值适应，同时保留细胞之间的原生差异和方向，"
              "比仅把一个方差标量拉大更能约束真实信息。", "",
              f"当前全control池的原SE方差为 {source_frame[source_frame.representation=='frozen_cached_SE'].mean_feature_variance.min():.6g}–"
              f"{source_frame[source_frame.representation=='frozen_cached_SE'].mean_feature_variance.max():.6g}；"
              f"实测centered anchor原始均方损失为 {trained_rows.centered_mse_vs_frozen.min():.6g}–"
              f"{trained_rows.centered_mse_vs_frozen.max():.6g}，"
              f"完整source MSE anchor为 {trained_rows.mse_vs_frozen.min():.6g}–{trained_rows.mse_vs_frozen.max():.6g}。"
              "实现时必须保留按2048维取mean的约定；如以冻结源方差归一化，anchor量级约为1，"
              "其系数不能直接照搬未归一化版本。系数和渐增速度应按训练内源保真与梯度量级设定，"
              "不通过外测Pearson或wMSE反推。", "",
              "可选方案2：为现有decoder增加训练treated真实frozen-SE的普通重建MSE锚定 "
              "L_dec_anchor = mean((D(z_SE_train)-x_train)^2)，与原预测端点decoder损失混合。"
              "此前同一真实SE输入下，19k decoder的外测重建wMSE是原预训练decoder的4.50倍，"
              "这支持检查并维持decoder的原始SE解码能力。该方案不加head、无需更改架构；"
              "真实treated SE仅出现在训练监督分支，正式预测仍然只解码Flow预测端点。"
              "优先先单独检验方案1，再决定是否加入方案2，避免同时改动而无法归因。", "",
              "A/B/C短续训均从已经收缩的19k状态开始；降低LoRA学习率或冻结当前LoRA，只能检验"
              "此后是否继续漂移，不能据此声称已恢复被压缩的原生细胞差异。源保真恢复可能使Flow输入分布改变，"
              "应在训练内监测源方差、锚定误差和残差损失后按既定步数评估。", "",
              "12条件的OT近独立匹配是该19k源表示下的实测现象，不是epsilon=0.01普遍无效的结论。"
              "原cost已按mean归一化，仅改变cost是否除2048不会改变理想计算中的plan。"
              "当前路径噪声相对源变异很大，但源变异本身已收缩；应先区分源表示恢复与噪声调整，"
              "不能把sigma或epsilon单独认定为唯一原因。", ""]
    (output/"followup_zh.md").write_text("\n".join(lines))
    provenance_path=output/"provenance.json"
    provenance=json.loads(provenance_path.read_text()) if provenance_path.exists() else {}
    provenance["followup"]=dict(created_utc=datetime.now(timezone.utc).isoformat(),
        checkpoint=file_identity(checkpoint_path),tail_initialization=file_identity(tail_path),
        frozen_tail_hash_unchanged=original_frozen_hash,
        cpu_only=True,cuda_initialized=torch.cuda.is_initialized(),trained=False,reran_flow=False,
        target_SE_used_only_as_diagnostic=True,threads=args.threads,
        formal_seed=4243,formal_conditions=108,formal_archives=formal_provenance,
        formal_config_example=formal_config,prefixes=prefix_provenance,
        ot_conditions=[r["condition"] for r in chosen],ot_selection="4 original-order evenly spaced training conditions per line",
        ot_candidates=256,ot_epsilon_normalized=.01,path_sigma=.01,
        source_code={str(path):file_identity(path) for path in
                     [Path(__file__),"devflow/flow/matching.py","devflow/flow/paths.py",
                      "devflow/networks/state_encoder.py","devflow/runtime/sciplex3.py"]},
        outputs={name:file_identity(output/name) for name in
                 ["variance_formal80.csv","source_variance.csv","ot_training_scale.csv","followup_zh.md"]},
        seconds=time.time()-started)
    if torch.cuda.is_initialized():
        raise RuntimeError("Unexpected CUDA initialization")
    atomic_json(provenance_path,provenance)
    print((output/"followup_zh.md").read_text(),flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--baseline-dir")
    parser.add_argument("--output")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--followup-only", action="store_true")
    parser.add_argument("--formal-dir")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    followup(args) if args.followup_only else run(args)


if __name__ == "__main__":
    main()

