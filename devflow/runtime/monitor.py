"""Read-only live notebook and HTML views of experiment artifacts."""
from pathlib import Path
import base64
import contextlib
import io
import json


def build_monitor(output):
    # The MLP mean/MMD recipe has different losses, sampling and pass counts.
    # Keep the historical attention monitor behavior for existing runs.
    config_path=Path(output)/"run_config.json"
    if config_path.exists():
        config=json.loads(config_path.read_text())
        family=config.get("model",{}).get("family",config.get("model_family",""))
        if "mlp" in str(family).lower() and "mean" in str(family).lower():
            from .monitor_mlp import build_monitor as build_mlp_monitor
            return build_mlp_monitor(output)
    import nbformat
    root=Path(output).resolve()
    root.mkdir(parents=True,exist_ok=True)
    header="""# DeVFlow live experiment monitor
Run All reloads current files. Loss components retain their recorded weighting.
Online OT convergence is a training diagnostic, distinct from 128-cell formal
benchmark transport metrics. External online seed 4242 is exploratory. Missing
splits and metrics are reported explicitly; no interpolation is performed.
"""
    cells=[nbformat.v4.new_markdown_cell(header)]
    setup=f"""from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path({str(root)!r})
CURVES=ROOT/'curves'
CURVES.mkdir(exist_ok=True)
config=json.loads((ROOT/'run_config.json').read_text()) if (ROOT/'run_config.json').exists() else {{}}
status=json.loads((ROOT/'status.json').read_text()) if (ROOT/'status.json').exists() else {{}}
print('Status:',status.get('status','not started'),'Optimizer step:',status.get('optimizer_step',status.get('update')))
print('Micro step:',status.get('micro_step'),'Condition passes:',status.get('condition_passes'))
print('Groups used for pass conversion:',config.get('train_conditions','unavailable'))
print('Loss weights:',config.get('loss_weights',config.get('model',{{}}).get('loss_weights')))
print('Expression gradients to Flow/LoRA:',config.get('expression_gradient_to_flow',False))
metrics_path=ROOT/'train_metrics.csv'
train=pd.read_csv(metrics_path) if metrics_path.exists() else pd.DataFrame()
"""
    losses="""raw=['residual_flow_loss','mean_shift_loss','decoder_residual_loss','decoder_mean_loss','bulk_loss','oracle_loss','set_mmd_loss','source_anchor_loss','decoder_true_se_loss']
weighted=['weighted_mean_shift_loss','weighted_decoder_mean_loss','weighted_bulk_loss','weighted_oracle_loss','weighted_set_mmd_loss','weighted_source_anchor_loss','weighted_decoder_true_se_loss','flow_loss','decoder_loss','total_loss']
if train.empty:
    print('No training rows yet.')
else:
    print(train.tail(5).to_string(index=False))
    fig,axes=plt.subplots(1,2,figsize=(13,4))
    for ax,names,title in zip(axes,[raw,weighted],['Raw loss components','Weighted components and totals']):
        for name in names:
            if name in train:ax.plot(train['update'],train[name].rolling(50,min_periods=1).mean(),label=name)
        ax.set(xlabel='Optimizer update',ylabel='50-update rolling mean (log scale)',yscale='log',title=title)
        ax.legend(fontsize=7)
    fig.tight_layout();fig.savefig(CURVES/'loss_components.png',dpi=130);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,3))
    for name in ['gradient_norm_before_clip','decoder_gradient_norm_before_clip']:
        if name in train:ax.plot(train['update'],train[name].clip(lower=1e-12),label=name)
    ax.set(yscale='log',xlabel='Optimizer update',ylabel='Pre-clip norm');ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(CURVES/'gradients.png',dpi=130);plt.close(fig)
"""
    evaluation="""rows=[]
for path in sorted((ROOT/'evaluations').glob('update_*/summary.json')):
    item=json.loads(path.read_text())
    row={'update':item['update'],'split':item.get('split','unknown'),'time_points':item.get('time_points'),'precision':item.get('precision'),'chunk':item.get('chunk'),**item.get('metrics',{})}
    rows.append(row)
evaluation=pd.DataFrame(rows)
print('Evaluation records:')
print(evaluation.to_string(index=False) if not evaluation.empty else 'No evaluation records yet.')
for split in ['internal_test','external_test']:
    if evaluation.empty or split not in set(evaluation['split']):print(split+': unavailable')
requested=['pseudobulk_mae','pseudobulk_mse','pearson_delta','pearson_deltapert','weighted_mse','de_spearman','endpoint_mmd','emd','e_distance']
for metric in requested:
    if evaluation.empty or metric not in evaluation:print(metric+': unavailable');continue
    fig,ax=plt.subplots(figsize=(7,3))
    for keys,group in evaluation.groupby(['split','time_points','precision','chunk'],dropna=False):
        ax.plot(group['update'],group[metric],marker='o',label=str(keys))
    ax.set(xlabel='Optimizer update',ylabel=metric);ax.legend(fontsize=7)
    fig.tight_layout();fig.savefig(CURVES/(metric+'.png'),dpi=130);plt.close(fig)
"""
    namespace={}
    for index,source in enumerate([setup,losses,evaluation],1):
        cell=nbformat.v4.new_code_cell(source,execution_count=index)
        stream=io.StringIO()
        with contextlib.redirect_stdout(stream):exec(compile(source,str(root/'monitor.ipynb'),'exec'),namespace)
        cell.outputs=[nbformat.v4.new_output('stream',name='stdout',text=stream.getvalue())]
        if index>1:
            images=sorted((root/'curves').glob('*.png'))
            if index==2:images=[x for x in images if x.stem in ('loss_components','gradients')]
            else:images=[x for x in images if x.stem not in ('loss_components','gradients')]
            for image in images:cell.outputs.append(nbformat.v4.new_output('display_data',data={'image/png':base64.b64encode(image.read_bytes()).decode()},metadata={}))
        cells.append(cell)
    notebook=nbformat.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}})
    nbformat.write(notebook,root/'monitor.ipynb')
    table=namespace['evaluation'].to_html(index=False) if not namespace['evaluation'].empty else '<p>No evaluation records yet.</p>'
    images=''.join('<img style="max-width:100%" src="data:image/png;base64,'+base64.b64encode(path.read_bytes()).decode()+'">' for path in sorted((root/'curves').glob('*.png')))
    (root/'monitor.html').write_text('<meta charset="utf-8"><title>DeVFlow monitor</title><h1>DeVFlow live artifacts</h1>'+table+images)
    return root/'monitor.ipynb'
