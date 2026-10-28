# Minimal model-loading and prediction examples

Install the package using [the project README](../README.md), then configure the
prepared data and weights in the relevant JSON template.
Each task retains its conditioning and decoder.

Load the current SciPlex3 MLP + mean model, or a ComboSciPlex/Norman model:

```bash
python examples/load_model.py sciplex3 CHECKPOINT.pt --family mlp_mean
python examples/load_model.py combosciplex CHECKPOINT.pt
python examples/load_model.py norman CHECKPOINT.pt
```

SciPlex3 defaults to `mlp_mean`. Its checkpoint contains the mean head and
decoder, so this model-loading example does not require a separate SE tail.
For a historical SciPlex3 attention checkpoint, select that family explicitly:

```bash
python examples/load_model.py sciplex3 ATTENTION_CHECKPOINT.pt \
  --family attention --tail-checkpoint SE_TAIL.pt
```

The `--family` and `--tail-checkpoint` options apply only to SciPlex3.
Loading defaults to CPU and starts neither training nor prediction.

For prediction, fill the relevant JSON template and set `prediction.checkpoint`
to the saved model. The input data, features and sampling manifests must also
be configured:

```bash
python examples/predict.py sciplex3 --config configs/sciplex3_mlp_mean_mmd_50k.json
python examples/predict.py combosciplex --config configs/combosciplex.json
python examples/predict.py norman --config configs/norman_additive.json
```

The same prediction operations are available through the installed `devflow`
command. Prepared data and checkpoints are not bundled.
