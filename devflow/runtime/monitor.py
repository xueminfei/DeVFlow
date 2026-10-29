"""Read-only training and evaluation monitors for all DeVFlow experiments.

The same loader, plots, HTML, notebook and watcher support flat SciPlex3
summaries, Combo head/raw/final summaries, and historical attention runs.
ACTIVE_RUN.json joins continuation stages without modifying their source logs.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import html
import json
import os
from pathlib import Path
import time
import traceback

import numpy as np
import pandas as pd

from benchmark.metrics import DE50_METRICS, HVG_METRICS
from .artifacts import atomic_json

RAW_LOSSES = (
    "residual_flow_loss", "mean_shift_loss", "head_fit_loss", "head_data_loss",
    "head_regularization_loss", "head_reg_loss", "head_loss", "decoder_mmd_loss",
    "set_mmd_loss", "decoder_residual_loss", "decoder_mean_loss", "bulk_loss",
    "oracle_loss", "source_anchor_loss", "decoder_true_se_loss",
)
WEIGHTED_LOSSES = (
    "weighted_residual_flow_loss", "weighted_mean_shift_loss", "flow_loss",
    "weighted_head_loss", "weighted_decoder_mmd_loss", "weighted_set_mmd_loss",
    "weighted_decoder_mean_loss", "decoder_loss", "weighted_bulk_loss",
    "weighted_oracle_loss", "weighted_source_anchor_loss", "weighted_decoder_true_se_loss", "total_loss",
)
METRICS = HVG_METRICS
BRANCHES = {
    "head": ("Mean head", "#9467bd", ":"),
    "raw": ("Raw decoder", "#e68613", "--"),
    "final": ("Final scaled prediction", "#2474b5", "-"),
}
MINIMIZE = {"pseudobulk_mae", "pseudobulk_mse", "weighted_mse", "endpoint_mmd",
            "emd", "e_distance", "legacy_wmse"}


def _json(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def _save_figure(fig, path):
    path = Path(path)
    temporary = path.with_name(f".{path.stem}.tmp.{os.getpid()}.png")
    fig.savefig(temporary, dpi=130, format="png")
    temporary.replace(path)


def _is_mean_family(config):
    family = config.get("model", {}).get("family", config.get("model_family", ""))
    return "mlp" in str(family).lower() and "mean" in str(family).lower()


def _stage_layout(root):
    root = Path(root).resolve()
    pointer = _json(root / "ACTIVE_RUN.json", {})
    def resolve(value):
        path = Path(value).expanduser()
        return (path if path.is_absolute() else root / path).resolve()
    active = resolve(pointer.get("active_run", str(root)))
    stages = [root] + [resolve(value) for value in pointer.get("stage_runs", [])] + [active]
    return root, active, list(dict.fromkeys(stages))


def _configuration(item):
    # Source snapshots can change across a continuation while sampling and the
    # scientific evaluation remain identical. Keep source hashes in the table;
    # split plot lines only when the actual recorded evaluation settings differ.
    protocol = item.get("protocol", {})
    if not isinstance(protocol, dict):
        protocol = {"name": protocol}
    values = {key: item.get(key, protocol.get(key)) for key in
              ("precision", "time_points", "cells_per_condition", "n_conditions", "chunk")}
    values.update({key: protocol[key] for key in
                   ("response_profile", "expression", "mmd", "allow_tf32") if key in protocol})
    return json.dumps(values, sort_keys=True, separators=(",", ":"))


def load_run(root):
    """Return root config, active status, merged logs, summaries and warnings.

    Stages retain their source paths in the returned tables. A later stage wins
    a duplicated optimizer update or evaluation identity; source CSV/JSON files
    are never rewritten. Four-seed benchmark rows remain separate from online
    single-seed rows.
    """
    root, active, stages = _stage_layout(root)
    config = _json(root / "run_config.json", _json(active / "run_config.json", {}))
    config = {**config, "monitor_root": str(root), "monitor_active_run": str(active),
              "monitor_stage_runs": [str(path) for path in stages]}
    status = _json(active / "status.json", {})
    rows, frames, warnings = [], [], []
    for stage in stages:
        path = stage / "train_metrics.csv"
        if path.exists():
            try:
                frame = pd.read_csv(path)
                if not frame.empty:
                    if "optimizer_step" not in frame:
                        if "update" in frame:
                            frame["optimizer_step"] = frame["update"]
                        elif "step" in frame:
                            frame["optimizer_step"] = frame["step"]
                    frame["source_run"] = str(stage)
                    frames.append(frame)
            except (pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
                warnings.append("Training CSV is being written; refresh again: " + str(path) + ": " + str(exc))
        for section in ("mean_diagnostics", "evaluations", "benchmark"):
            paths = sorted((stage / section).glob("step_*/summary.json"))
            paths += sorted((stage / section).glob("update_*/summary.json"))
            for path in paths:
                item = _json(path)
                seeds = item.get("seeds", [item["seed"]] if "seed" in item else [])
                base = dict(update=int(item.get("update", item.get("optimizer_step", 0))),
                    split=item.get("split", "external_test"),
                    mode=item.get("mode", "mean_only_diagnostic" if section == "mean_diagnostics" else "full_prediction"),
                    purpose=item.get("purpose", "benchmark" if section == "benchmark" else "monitor"),
                    seeds=",".join(map(str, seeds)), repeats=item.get("n_sampling_repeats", len(seeds)),
                    checkpoint_sha256=item.get("checkpoint_sha256", "unavailable"),
                    protocol_sha256=item.get("protocol_sha256", "unavailable"),
                    configuration_key=_configuration(item),
                    precision=item.get("precision"), time_points=item.get("time_points"),
                    n_conditions=item.get("n_conditions"), cells_per_condition=item.get("cells_per_condition"),
                    elapsed_seconds=item.get("elapsed_seconds"), section=section, source_run=str(stage))
                branches = item.get("branches") or {
                    "head" if base["mode"] == "mean_only_diagnostic" else "final": item}
                for branch, result in branches.items():
                    row = {**base, "branch": branch, **result.get("metrics", {})}
                    row.update({key + "_std": value for key, value in result.get("metrics_std", {}).items()})
                    rows.append(row)
    train = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
    if not train.empty and "optimizer_step" in train:
        train["optimizer_step"] = pd.to_numeric(train["optimizer_step"], errors="coerce")
        train = train.dropna(subset=["optimizer_step"]).drop_duplicates("optimizer_step", keep="last")
        train = train.sort_values("optimizer_step").reset_index(drop=True)
    evaluation = pd.DataFrame(rows)
    if not evaluation.empty:
        identity = ["section", "update", "split", "mode", "seeds", "branch"]
        evaluation = evaluation.drop_duplicates(identity, keep="last").sort_values(["update", "section", "branch"]).reset_index(drop=True)
    return config, status, train, evaluation, warnings


def _loss_curves(train, step_key, curves):
    import matplotlib.pyplot as plt
    generated = []
    if train.empty or step_key not in train:
        return generated
    raw = tuple(dict.fromkeys(RAW_LOSSES + ("decoder_residual_loss", "decoder_mean_loss",
                                          "bulk_loss", "oracle_loss", "source_anchor_loss")))
    weighted = tuple(dict.fromkeys(WEIGHTED_LOSSES + ("weighted_decoder_mean_loss", "decoder_loss",
        "weighted_bulk_loss", "weighted_oracle_loss", "weighted_source_anchor_loss")))
    for title, names, filename in (("Raw training losses", raw, "monitor_raw_losses.png"),
                                  ("Weighted training losses and totals", weighted, "monitor_weighted_losses.png")):
        present = [name for name in names if name in train]
        if not present:
            continue
        fig, ax = plt.subplots(figsize=(11, 4))
        for name in present:
            ax.plot(train[step_key], train[name].rolling(20, min_periods=1).mean(), label=name)
        ax.set(xlabel="Optimizer step", ylabel="20-record rolling mean", title=title)
        ax.set_yscale("symlog", linthresh=1e-6)
        ax.legend(fontsize=7, ncol=2)
        fig.tight_layout()
        path = curves / filename
        _save_figure(fig, path)
        plt.close(fig)
        generated.append(path)
    names = [name for name in train if "gradient_norm" in name]
    if names:
        fig, ax = plt.subplots(figsize=(10, 3))
        for name in names:
            ax.plot(train[step_key], train[name], label=name)
        ax.set(xlabel="Optimizer step", ylabel="Pre-clip gradient norm")
        ax.set_yscale("symlog", linthresh=1e-8)
        ax.legend(fontsize=7)
        fig.tight_layout()
        path = curves / "monitor_gradient_norms.png"
        _save_figure(fig, path)
        plt.close(fig)
        generated.append(path)
    if "zero_decoder_mean_fraction" in train:
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.plot(train[step_key], train["zero_decoder_mean_fraction"])
        ax.set(xlabel="Optimizer step", ylabel="Genes using constant-mean fallback", ylim=(0, 1))
        fig.tight_layout()
        path = curves / "monitor_zero_decoder_mean_fraction.png"
        _save_figure(fig, path)
        plt.close(fig)
        generated.append(path)
    return generated


def _metric_grid(frame, metrics, curves, filename, title, *, prefix="", ncols=5, attention=False):
    """One line per branch, split and actual sampling/protocol configuration."""
    import matplotlib.pyplot as plt
    nrows = (len(metrics) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.0 * ncols, 3.1 * nrows), squeeze=False)
    handles, labels = {}, {}
    for ax, name in zip(axes.flat, metrics):
        metric = prefix + name
        ax.set_title(name + (" ↓" if name in MINIMIZE else " ↑"), fontsize=9)
        ax.set_xlabel("Optimizer step", fontsize=8)
        ax.tick_params(labelsize=7)
        if frame.empty or metric not in frame or frame[metric].isna().all():
            ax.text(.5, .5, "Unavailable", transform=ax.transAxes, ha="center", va="center", fontsize=9)
            continue
        keys = ["split", "branch", "seeds", "configuration_key"]
        groups = list(frame.groupby(keys, sort=False, dropna=False))
        protocol_count = frame["configuration_key"].nunique()
        for identity, group in groups:
            split, branch, seeds, protocol = identity
            group = group.sort_values("update").dropna(subset=[metric])
            if group.empty:
                continue
            label, color, style = BRANCHES.get(branch, (str(branch), "#555555", "-"))
            if attention and branch == "final":
                label = "Attention prediction"
            label += f" · {split} · seeds {seeds}"
            if protocol_count > 1:
                settings = json.loads(str(protocol))
                label += (f" · {settings.get('precision')} / RK4-{settings.get('time_points')}"
                          " · " + hashlib.sha256(str(protocol).encode()).hexdigest()[:8])
            line, = ax.plot(group["update"], group[metric], marker="o", markersize=3,
                            color=color, linestyle=style, label=label)
            handles[label], labels[label] = line, label
            if metric + "_std" in group:
                deviations = pd.to_numeric(group[metric + "_std"], errors="coerce").to_numpy(float)
                values = group[metric].to_numpy(float)
                finite = np.isfinite(deviations)
                if finite.any():
                    steps = group["update"].to_numpy()[finite]
                    ax.fill_between(steps, values[finite] - deviations[finite],
                                    values[finite] + deviations[finite], color=color, alpha=.12)
        ax.grid(alpha=.2)
    for ax in list(axes.flat)[len(metrics):]:
        ax.set_visible(False)
    fig.suptitle(title, fontsize=13)
    if handles:
        fig.legend([handles[key] for key in handles], [labels[key] for key in handles],
                   loc="lower center", ncol=3, fontsize=8)
    fig.tight_layout(rect=(0, .07, 1, .96))
    path = curves / filename
    _save_figure(fig, path)
    plt.close(fig)
    return path


def render(root, *, display=False):
    """Reload logs and render; never execute the model or change training state."""
    import matplotlib
    matplotlib.use("Agg")
    root = Path(root).resolve()
    config, status, train, evaluation, warnings = load_run(root)
    active = Path(config["monitor_active_run"])
    is_mean = _is_mean_family(config)
    task = config.get("task", "sciplex3")
    title = {"combosciplex": "ComboSciPlex", "sciplex3": "SciPlex3", "norman": "Norman19", "norman19": "Norman19"}.get(task, str(task))
    title += " mean / Flow / decoder" if is_mean else " attention"
    curves = root / "curves"
    curves.mkdir(parents=True, exist_ok=True)
    step_key = "optimizer_step" if "optimizer_step" in train else "update"
    generated = _loss_curves(train, step_key, curves)
    settings = config.get("training", {})
    group_count = config.get("train_conditions", 24 if task == "combosciplex" else None)
    group_count = int(group_count) if group_count is not None else None
    conditions_per_update = int(config.get("conditions_per_update", settings.get("gradient_accumulation", 4)))
    accumulation = int(settings.get("gradient_accumulation", 4))
    current = status.get("optimizer_step", status.get("update"))
    if current is None and not train.empty and step_key in train:
        current = int(train[step_key].iloc[-1])
    online_seed = int(config.get("evaluation", {}).get("monitor_seeds", [4243 if is_mean else 4242])[0])
    final_seeds = list(config.get("prediction", {}).get("seeds", [4243, 4244, 4245, 4246]))
    full, mean = evaluation, evaluation
    if not evaluation.empty:
        full = evaluation[(evaluation["section"] == "evaluations") &
                          (evaluation["mode"] == "full_prediction") &
                          (evaluation["seeds"] == str(online_seed))]
        mean = evaluation[evaluation["mode"] == "mean_only_diagnostic"]
    if not full.empty:
        count = full["n_conditions"].dropna()
        count = int(count.iloc[-1]) if len(count) else "recorded"
        generated.append(_metric_grid(full, HVG_METRICS, curves, "monitor_online_hvg_3x5.png",
            f"{title} · {count} conditions · online seed {online_seed} · HVG", attention=not is_mean))
        generated.append(_metric_grid(full, DE50_METRICS, curves, "monitor_online_de50_3x4.png",
            f"{title} · fixed DE50 · online seed {online_seed}", prefix="de50_", ncols=4, attention=not is_mean))
    if not mean.empty:
        names = [name for name in HVG_METRICS if name in mean and not mean[name].isna().all()]
        if names:
            generated.append(_metric_grid(mean, names, curves, "monitor_mean_diagnostics.png",
                "Mean-only diagnostic · no Flow integration or distribution evaluation", ncols=4))
    # Historical attention summaries can include multiple splits or alternative
    # sampling settings. Preserve every recorded metric curve, including records
    # whose seed was not stored; never relabel them as the formal online seed.
    if not is_mean and not evaluation.empty:
        generated.append(_metric_grid(evaluation[evaluation["section"] == "evaluations"], HVG_METRICS,
            curves, "monitor_attention_recorded_hvg.png", "Attention · all recorded splits and configurations", attention=True))
    passes = float(current) * conditions_per_update / group_count if current is not None and group_count else None
    latest = dict(status=status.get("status", "unknown"), training_optimizer_step=current,
        micro_step=status.get("micro_step", int(current) * accumulation if current is not None else None),
        condition_passes=passes, train_condition_count=group_count,
        conditions_per_optimizer_update=conditions_per_update,
        condition_pass_definition="optimizer_step * conditions_per_update / train_conditions; not a cell epoch",
        online_seed=online_seed, formal_sampling_seeds=final_seeds,
        latest_mean_diagnostic_step=int(mean["update"].max()) if not mean.empty else None,
        latest_full_evaluation_step=int(full["update"].max()) if not full.empty else None,
        selection_scope=config.get("selection", {}).get("protocol", "external_test_exploratory"),
        selection_primary_metric=config.get("selection", {}).get("primary_metric"),
        configured_final_update=config.get("final_update", settings.get("max_updates", settings.get("steps"))),
        active_run=str(active), stage_runs=config["monitor_stage_runs"],
        training_log_rows=len(train))
    notes = [
        "Run All reloads CSV/JSON files. This monitor never executes the model or changes training state.",
        f"Condition passes use {group_count} training conditions and {conditions_per_update} conditions per optimizer update; they are not cell epochs.",
        "Continuation stages are joined by optimizer step. Later stages replace duplicated records; source logs remain unchanged.",
        f"Online seed: {online_seed}. Formal sampling seeds: {', '.join(map(str, final_seeds))}. Benchmark values remain separate from online curves.",
        "Each branch keeps the same colour across refreshes. Scientific settings, seeds and splits remain distinct; source hashes remain in the table.",
        "Mean diagnostics omit distribution metrics. Negative unbiased MMD estimates are retained.",
        "Training ot_* convergence is separate from formal 128-cell OT/EMD.",
        "Sampling-repeat SD describes one fixed model and does not measure independent retraining uncertainty.",
    ]
    if is_mean:
        notes += ["Purple: mean head. Orange: raw decoder. Blue: final scaled output.",
            "Full evaluations retain their native condition counts and use 128 cells, FP32 / TF32 off / classical RK4-80.",
            "External-test checkpoint selection is exploratory; one checkpoint supplies the final reported row."]
    else:
        notes.append("Historical attention curves retain their recorded precision, RK4 points and chunk settings; they are not relabelled as the formal benchmark.")
    if task == "combosciplex":
        notes += ["Pseudobulk targets are full treated-condition means. Distribution metrics use locked 128-cell target draws.",
            "Training MMD uses real training treated-cell bandwidth; formal Combo MMD retains its historical pooled-bandwidth scorer."]
    splits = set(evaluation["split"]) if not evaluation.empty else set()
    for split in ("internal_test", "external_test"):
        if split not in splits:
            notes.append(split + ": unavailable; no curve is invented.")
    notes.extend(warnings)
    missing = [name for name in HVG_METRICS if full.empty or name not in full or full[name].isna().all()]
    notes.append("Unavailable online HVG metrics: " + (", ".join(missing) if missing else "none"))
    selected = _json(active / "SELECTED_CHECKPOINT.json", _json(active / "checkpoint_candidates.json",
        _json(root / "SELECTED_CHECKPOINT.json", _json(root / "checkpoint_candidates.json", {}))))
    table = evaluation.to_html(index=False, escape=True) if not evaluation.empty else "<p>No completed evaluation yet.</p>"
    images = "".join('<img style="max-width:100%" alt="' + html.escape(path.stem) + '" src="data:image/png;base64,'
        + base64.b64encode(path.read_bytes()).decode() + '">' for path in generated)
    page = '<!doctype html><html><head><meta charset="utf-8"><meta http-equiv="refresh" content="60"><title>DeVFlow monitor</title></head><body>'
    page += '<h1>DeVFlow ' + html.escape(title) + ' monitor</h1><pre>' + html.escape(json.dumps(latest, indent=2)) + '</pre>'
    page += '<ul>' + ''.join('<li>' + html.escape(note) + '</li>' for note in notes) + '</ul>'
    page += '<h2>Checkpoint selection</h2><pre>' + html.escape(json.dumps(selected, indent=2)) + '</pre>'
    page += images + '<h2>All completed evaluation values</h2>' + table + '</body></html>'
    temporary = root / f"monitor.html.tmp.{os.getpid()}"
    temporary.write_text(page)
    temporary.replace(root / "monitor.html")
    atomic_json(root / "monitor_status.json", latest)
    if display:
        from IPython.display import display as show, Image
        print(json.dumps(latest, indent=2))
        for note in notes:
            print(note)
        print("Checkpoint selection:", json.dumps(selected, indent=2))
        show(train.tail(10))
        show(evaluation)
        for path in generated:
            show(Image(filename=str(path)))
    return latest


def _build_monitor(output):
    import nbformat
    root = Path(output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    latest = render(root)
    code = f"""from pathlib import Path
import os
from devflow.runtime.monitor import render
# Support kernels started in the run folder or the project root. The explicit
# override also supports experiments moved to a different machine or folder.
candidates = []
if os.environ.get('DEVFLOW_RUN_DIR'):
    candidates.append(Path(os.environ['DEVFLOW_RUN_DIR']).expanduser().resolve())
candidates.extend([Path.cwd(), Path.cwd() / 'outputs' / {root.name!r}, Path({str(root)!r})])
ROOT = next((path for path in candidates if (path / 'run_config.json').is_file()), None)
if ROOT is None:
    raise FileNotFoundError('Cannot locate this experiment: tried ' + ', '.join(map(str, candidates)))
render(ROOT, display=True)
"""
    cell = nbformat.v4.new_code_cell(code, execution_count=1)
    cell.outputs = [nbformat.v4.new_output("stream", name="stdout", text=json.dumps(latest, indent=2) + "\n")]
    for path in sorted((root / "curves").glob("monitor_*.png")):
        cell.outputs.append(nbformat.v4.new_output("display_data", data={
            "image/png": base64.b64encode(path.read_bytes()).decode()}, metadata={}))
    notebook = nbformat.v4.new_notebook(cells=[
        nbformat.v4.new_markdown_cell("# DeVFlow live monitor\nRun All reloads training and evaluation artifacts. No training-state changes or model prediction occur.\n"),
        cell,
    ], metadata={"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}})
    for name in ("monitor.ipynb", "monitor_live.ipynb"):
        temporary = root / f"{name}.tmp.{os.getpid()}"
        nbformat.write(notebook, temporary)
        temporary.replace(root / name)
    return root / "monitor_live.ipynb"


def build_attention_monitor(output):
    """Historical attention entry; recorded protocols stay visible and intact."""
    return _build_monitor(output)


def build_monitor(output):
    config, _, _, _, _ = load_run(output)
    return _build_monitor(output) if _is_mean_family(config) else build_attention_monitor(output)


def _terminal(root):
    _, active, _ = _stage_layout(root)
    status = _json(active / "status.json", {})
    return status.get("status") in {"complete", "failed", "paused"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval", type=float, default=60)
    args = parser.parse_args()
    if args.interval <= 0:
        parser.error("--interval must be positive")
    def refresh():
        try:
            build_monitor(args.output)
            (args.output / "MONITOR_WATCH_ERROR.json").unlink(missing_ok=True)
        except Exception as exc:
            atomic_json(args.output / "MONITOR_WATCH_ERROR.json", {
                "error": repr(exc), "traceback": traceback.format_exc(), "process_id": os.getpid()})
            if not args.watch:
                raise
    refresh()
    while args.watch and not _terminal(args.output):
        time.sleep(args.interval)
        refresh()


if __name__ == "__main__":
    main()
