"""Read-only live views for MLP, mean-head and decoder-MMD experiments."""
from __future__ import annotations

import argparse
import base64
import html
import json
import os
from pathlib import Path
import time
import traceback

import numpy as np
import pandas as pd

from .artifacts import atomic_json

RAW_LOSSES = (
    "residual_flow_loss", "mean_shift_loss", "head_fit_loss", "head_data_loss",
    "head_regularization_loss", "head_reg_loss", "head_loss", "decoder_mmd_loss",
    "set_mmd_loss",
)
WEIGHTED_LOSSES = (
    "weighted_residual_flow_loss", "weighted_mean_shift_loss", "flow_loss",
    "weighted_head_loss", "weighted_decoder_mmd_loss", "weighted_set_mmd_loss", "total_loss",
)
METRICS = (
    "pseudobulk_mae", "pseudobulk_mse", "pearson_delta", "pearson_deltapert",
    "weighted_mse", "weighted_r2_deltactrl", "de_spearman", "spearman_delta",
    "endpoint_mmd", "emd", "e_distance", "nir", "pds", "legacy_wmse",
    "legacy_weighted_r2_delta",
)


def _save_figure(fig, path):
    path = Path(path)
    temporary = path.with_name(f".{path.stem}.tmp.{os.getpid()}.png")
    fig.savefig(temporary, dpi=130, format="png")
    temporary.replace(path)


def _terminal(root):
    root = Path(root)
    if (root / "FAILED.json").exists() or (root / "TRAINING_COMPLETE.json").exists():
        return True
    path = root / "status.json"
    return path.exists() and json.loads(path.read_text()).get("status") in {"complete", "failed"}


def load_run(root):
    root = Path(root)
    config = json.loads((root / "run_config.json").read_text()) if (root / "run_config.json").exists() else {}
    status = json.loads((root / "status.json").read_text()) if (root / "status.json").exists() else {}
    train = pd.DataFrame()
    warning = None
    if (root / "train_metrics.csv").exists():
        try:
            train = pd.read_csv(root / "train_metrics.csv")
        except (pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
            warning = "Training CSV is being written; refresh again: " + str(exc)
    rows = []
    patterns = ("mean_diagnostics/step_*/summary.json", "evaluations/step_*/summary.json",
                "benchmark/step_*/summary.json")
    for pattern in patterns:
        for path in sorted(root.glob(pattern)):
            item = json.loads(path.read_text())
            row = dict(update=item["update"], split=item.get("split", "unknown"),
                       mode=item.get("mode", "unknown"), purpose=item.get("purpose", "monitor"),
                       seeds=",".join(map(str, item.get("seeds", []))),
                       repeats=item.get("n_sampling_repeats", 0),
                       checkpoint_sha256=item.get("checkpoint_sha256", "unknown"),
                       protocol_sha256=item.get("protocol_sha256", "unknown"),
                       precision=item.get("precision"), time_points=item.get("time_points"),
                       n_conditions=item.get("n_conditions"),
                       elapsed_seconds=item.get("elapsed_seconds"),
                       **item.get("metrics", {}))
            row.update({name + "_std": value for name, value in item.get("metrics_std", {}).items()})
            rows.append(row)
    return config, status, train, pd.DataFrame(rows), warning


def render(root, *, display=False):
    """Reload current artifacts and render; never touch training state/weights."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root = Path(root).resolve()
    config, status, train, evaluation, warning = load_run(root)
    curves = root / "curves"
    curves.mkdir(parents=True, exist_ok=True)
    generated = []
    step_key = "optimizer_step" if "optimizer_step" in train else "update"
    if not train.empty and step_key in train:
        train = train.dropna(subset=[step_key])
        for title, names, filename in (("Raw losses", RAW_LOSSES, "mlp_raw_losses.png"),
                                       ("Weighted losses and total", WEIGHTED_LOSSES, "mlp_weighted_losses.png")):
            present = [name for name in names if name in train]
            if not present:
                continue
            fig, ax = plt.subplots(figsize=(10, 4))
            for name in present:
                ax.plot(train[step_key], train[name].rolling(20, min_periods=1).mean(), label=name)
            # Unbiased-form MMD estimates may be negative: log() would hide them.
            ax.set(xlabel="Optimizer step", ylabel="20-record rolling mean", yscale="symlog", title=title)
            ax.set_yscale("symlog", linthresh=1e-6)
            ax.legend(fontsize=7)
            fig.tight_layout()
            path = curves / filename
            _save_figure(fig, path)
            plt.close(fig)
            generated.append(path)
        gradients = [name for name in train if "gradient_norm" in name and
                     ("before_clip" in name or name in {"flow_gradient_norm", "decoder_gradient_norm", "head_gradient_norm"})]
        if gradients:
            fig, ax = plt.subplots(figsize=(10, 3))
            for name in gradients:
                ax.plot(train[step_key], train[name], label=name)
            ax.set(xlabel="Optimizer step", ylabel="Gradient norm before separate clipping", yscale="symlog")
            ax.legend(fontsize=7)
            fig.tight_layout()
            path = curves / "mlp_gradient_norms.png"
            _save_figure(fig, path)
            plt.close(fig)
            generated.append(path)
        if "zero_decoder_mean_fraction" in train:
            fig, ax = plt.subplots(figsize=(9, 3))
            ax.plot(train[step_key], train["zero_decoder_mean_fraction"], label="Mean across training condition batches/ranks")
            ax.set(xlabel="Optimizer step", ylabel="Fraction of genes using constant-mean fallback", ylim=(0, 1))
            ax.legend(fontsize=7)
            fig.tight_layout()
            path = curves / "mlp_zero_decoder_mean_fraction.png"
            _save_figure(fig, path)
            plt.close(fig)
            generated.append(path)
    metrics = list(METRICS) + ["de50_" + name for name in METRICS]
    if not evaluation.empty:
        for metric in metrics:
            if metric not in evaluation or evaluation[metric].isna().all():
                continue
            fig, ax = plt.subplots(figsize=(9, 3.8))
            keys = ["split", "mode", "purpose", "seeds", "protocol_sha256"]
            for identity, group in evaluation.groupby(keys, dropna=False, sort=False):
                group = group.sort_values("update").dropna(subset=[metric])
                if group.empty:
                    continue
                split, mode, purpose, seeds, _ = identity
                count = len(str(seeds).split(",")) if seeds else 0
                label = f"{split}; {mode}; {purpose}; seeds {seeds} ({count}/4)"
                values = group[metric].to_numpy(dtype=float)
                deviations = group.get(metric + "_std", pd.Series(np.nan, index=group.index)).to_numpy(dtype=float)
                ax.plot(group["update"], values, marker="o", label=label)
                finite = np.isfinite(deviations)
                if finite.any():
                    steps = group["update"].to_numpy()[finite]
                    ax.fill_between(steps, values[finite] - deviations[finite], values[finite] + deviations[finite], alpha=.14)
            ax.set(xlabel="Evaluated checkpoint optimizer step", ylabel=metric)
            ax.legend(fontsize=6)
            fig.tight_layout()
            path = curves / ("mlp_" + metric + ".png")
            _save_figure(fig, path)
            plt.close(fig)
            generated.append(path)
    groups = int(config.get("train_conditions", 2148))
    conditions = int(config.get("conditions_per_update", 4))
    current = status.get("optimizer_step", status.get("update"))
    if current is None and not train.empty and step_key in train:
        current = int(train[step_key].iloc[-1])
    passes = float(current) * conditions / groups if current is not None else None
    latest = dict(training_optimizer_step=current, micro_step=status.get("micro_step"),
                  condition_passes=passes, train_condition_count=groups,
                  conditions_per_optimizer_update=conditions,
                  condition_pass_definition="optimizer_step * conditions_per_update / train_conditions; not a cell epoch",
                  latest_mean_diagnostic_step=None, latest_full_evaluation_step=None,
                  selection_scope="external_test_exploratory", status=status.get("status", "unknown"))
    if not evaluation.empty:
        for mode, field in (("mean_only_diagnostic", "latest_mean_diagnostic_step"),
                            ("full_prediction", "latest_full_evaluation_step")):
            values = evaluation.loc[evaluation["mode"] == mode, "update"]
            if len(values):
                latest[field] = int(values.max())
    notes = [
        "This page reloads every 60 seconds; its producer reloads CSV/JSON artifacts.",
        "Mean diagnostics do not run the Flow and cannot evaluate cell distributions.",
        "Full evaluations use all 108 conditions, 128 cells, FP32 and RK4-80.",
        "Seed 4243 is one sampling repeat; four-seed SD is sampling uncertainty, not retraining uncertainty.",
        "The original internal_test cells belong to training. No held-out validation/internal-test curve is fabricated.",
        "External-test checkpoint selection is exploratory. Metrics from different checkpoints must not be combined into one reported model.",
        "Negative MMD estimates are retained. Training OT convergence is separate from expression-distribution metrics.",
    ]
    if warning:
        notes.append(warning)
    missing = [metric for metric in METRICS if evaluation.empty or metric not in evaluation or evaluation[metric].isna().all()]
    notes.append("Unavailable evaluation metrics: " + (", ".join(missing) if missing else "none"))
    table = evaluation.to_html(index=False, escape=True) if not evaluation.empty else "<p>No completed evaluation yet.</p>"
    images = "".join('<img style="max-width:100%" alt="' + html.escape(path.stem) + '" src="data:image/png;base64,'
                     + base64.b64encode(path.read_bytes()).decode() + '">' for path in generated)
    page = '<!doctype html><html><head><meta charset="utf-8"><meta http-equiv="refresh" content="60"><title>DeVFlow monitor</title></head><body>'
    page += '<h1>DeVFlow MLP / mean / decoder monitor</h1><pre>' + html.escape(json.dumps(latest, indent=2)) + '</pre>'
    page += '<ul>' + ''.join('<li>' + html.escape(note) + '</li>' for note in notes) + '</ul>'
    page += images + '<h2>All completed evaluation values</h2>' + table + '</body></html>'
    temp = root / f"monitor.html.tmp.{os.getpid()}"
    temp.write_text(page)
    temp.replace(root / "monitor.html")
    atomic_json(root / "monitor_status.json", latest)
    if display:
        from IPython.display import display as show, Image
        print(json.dumps(latest, indent=2))
        for note in notes:
            print(note)
        show(train.tail(10))
        show(evaluation)
        for path in generated:
            show(Image(filename=str(path)))
    return latest


def build_monitor(output):
    import nbformat
    root = Path(output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    latest = render(root)
    source = """from pathlib import Path
import os
from devflow.runtime.monitor_mlp import render
# Jupyter normally starts the kernel in this notebook's directory.
# For a relocated kernel, set DEVFLOW_RUN_DIR to the experiment directory.
ROOT = Path(os.environ.get('DEVFLOW_RUN_DIR', '.')).resolve()
if not (ROOT / 'run_config.json').exists():
    raise FileNotFoundError('Start the kernel in this experiment directory or set DEVFLOW_RUN_DIR.')
render(ROOT, display=True)
"""
    cell = nbformat.v4.new_code_cell(source, execution_count=1)
    cell.outputs = [nbformat.v4.new_output("stream", name="stdout", text=json.dumps(latest, indent=2) + "\n")]
    for path in sorted((root / "curves").glob("mlp_*.png")):
        cell.outputs.append(nbformat.v4.new_output("display_data", data={
            "image/png": base64.b64encode(path.read_bytes()).decode()}, metadata={}))
    notebook = nbformat.v4.new_notebook(cells=[
        nbformat.v4.new_markdown_cell("# DeVFlow live experiment monitor\nRun All reloads the current training and evaluation artifacts. No model execution or training-state changes occur.\n"),
        cell,
    ], metadata={"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}})
    temp = root / f"monitor.ipynb.tmp.{os.getpid()}"
    nbformat.write(notebook, temp)
    temp.replace(root / "monitor.ipynb")
    return root / "monitor.ipynb"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval", type=float, default=60)
    args = parser.parse_args()
    if args.interval <= 0:
        parser.error("--interval must be positive")
    def refresh(initial=False):
        try:
            build_monitor(args.output) if initial else render(args.output)
            (args.output / "MONITOR_WATCH_ERROR.json").unlink(missing_ok=True)
        except Exception as exc:
            atomic_json(args.output / "MONITOR_WATCH_ERROR.json", dict(
                error=repr(exc), traceback=traceback.format_exc(), process_id=os.getpid()))
            if not args.watch:
                raise
    refresh(initial=True)
    while args.watch and not _terminal(args.output):
        time.sleep(args.interval)
        refresh()


if __name__ == "__main__":
    main()
