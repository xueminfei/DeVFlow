# DeVFlow

DeVFlow predicts single-cell responses to chemical and genetic perturbations
using conditional flow matching.

## Installation

From the repository root, with Python 3.9 and a CUDA-enabled PyTorch installation:

```bash
python -m pip install -r requirements.txt
```

## Data and pretrained weights

Set the locations of your prepared data, feature assets and pretrained weights:

```bash
export DEVFLOW_DATA_ROOT=/path/to/prepared_data
export DEVFLOW_ASSET_ROOT=/path/to/prepared_assets
export DEVFLOW_WEIGHT_ROOT=/path/to/pretrained_weights
```

Download links will be added here when the assets are released.

## Preprocessing

Training uses prepared expression data and cached SE embeddings. The migrated
preprocessing sources include dataset preparation and SE/cache extraction:

```bash
python -m preprocessing --help
python -m preprocessing sciplex3 --help
python -m preprocessing combosciplex --help
python -m preprocessing norman --help
python -m preprocessing verify
```

Dependencies are included in the shared `requirements.txt`. Raw data, pretrained
SE weights and gene/drug feature assets are explicit inputs to those commands.

## Usage

Choose a configuration and fill in the required input paths:

| Dataset | Configuration |
| --- | --- |
| SciPlex3 | [MLP + trainable mean + MMD, 50k steps](configs/sciplex3_mlp_mean_mmd_50k.json) |
| ComboSciPlex | [MLP + trainable mean + MMD, 13k steps](configs/combosciplex_mlp_mean_mmd_13k.json) |
| Norman | [Additive split + trainable mean + MMD, 50k steps](configs/norman_additive.json) |

SciPlex3, ComboSciPlex and Norman use the trainable mean/MMD recipe. Norman uses a fixed gene-prior kernel mean head and a 2,054-gene axis: the union of 2,000 training HVGs and perturbation target
genes. Set `data.target_se_assets` to the prepared training-control target gene tokens. The historical Combo recipe remains in
`configs/combosciplex.json`.

For SciPlex3, prepare the assets and start training on eight GPUs:

```bash
python -m devflow.cli sciplex3 --config configs/sciplex3_mlp_mean_mmd_50k.json prepare
python -m torch.distributed.run --standalone --nproc_per_node=8 \
  -m devflow.cli sciplex3 --config configs/sciplex3_mlp_mean_mmd_50k.json train
```

For ComboSciPlex, prepare the training-only mean assets and start on four GPUs:

```bash
python -m devflow.cli combosciplex --config configs/combosciplex_mlp_mean_mmd_13k.json prepare
python -m torch.distributed.run --standalone --nproc_per_node=4 \
  -m devflow.cli combosciplex --config configs/combosciplex_mlp_mean_mmd_13k.json train
```

The new Combo recipe monitors sampling seed 4243 and reports four repeats
(4243–4246), reusing matching online predictions. Checkpoint selection by
external-test HVG wMSE is exploratory.

For prediction or evaluation, set `prediction.checkpoint` in the configuration
to the saved model:

```bash
python -m devflow.cli sciplex3 --config YOUR_CONFIG.json predict
python -m devflow.cli sciplex3 --config YOUR_CONFIG.json evaluate
```

Use `python -m devflow.cli TASK --help` for each dataset's available commands,
where `TASK` is `sciplex3`, `combosciplex` or `norman`.
Checkpoints, metrics and predictions are saved under `outputs/`.
Mean/MMD training also generates live monitor notebooks and HTML in its run directory.

## Paper

<!-- Paper title and link to be added. -->

## Citation

<!-- BibTeX to be added. -->
