# MM-MARAS

MM-MARAS is a multi-modal spatiotemporal model for Bay of Bengal chlorophyll-a reconstruction, short-range forecasting, uncertainty estimation, and ecosystem risk indexing.

The repository contains:

- A complete PyTorch model in `model.py`
- Dataset loading from `.npz` patches in `dataset.py`
- Training with AMP, checkpointing, TensorBoard, and optional DDP in `Train.py`
- Test-time evaluation and figure generation in `eval.py`

## What The Model Does

Given a temporal patch of satellite and environmental inputs, the model predicts:

- `recon`: reconstructed chlorophyll-a for the current timestep
- `forecast`: chlorophyll-a for the next 5 timesteps
- `uncertainty`: per-pixel log-variance for reconstruction
- `eri`: 5-level ecosystem risk logits

The model fuses five input streams:

- Optical observations: `chl_obs`, `obs_mask`
- Physics and forcing: `physics`, `wind`, `static`
- Missingness structure: `obs_mask`, `mcar_mask`, `mnar_mask`, `bloom_mask`
- Biogeochemistry: `bgc_aux`
- River forcing: `discharge`

High-level pipeline:

```text
optical + obs_mask -> OpticalEncoder
physics + wind + static -> PhysicsEncoder
mask stack -> MaskNet
bgc_aux -> BGCAuxEncoder
discharge -> DischargeEncoder

5 encoded streams -> FusionModule (Perceiver-style fusion)
fused sequence -> TemporalModule (ConvLSTM stack)
state -> MoEDecoder (4 experts)
decoded state -> heads for recon / forecast / uncertainty / eri
```

## Repository Layout

```text
.
|-- Train.py
|-- eval.py
|-- model.py
|-- dataset.py
|-- loss.py
|-- fusion.py
|-- temporal.py
|-- masknet.py
|-- optical_encoder.py
|-- physics_encoder.py
|-- bgc_encoder.py
|-- discharge_encoder.py
`-- moe_decoder.py
```

## Input Contract

`MARASSModel.forward()` expects a batch dict with these tensors:

| Key | Shape | Notes |
|---|---|---|
| `chl_obs` | `(B, T, H, W)` | observed chlorophyll-a |
| `obs_mask` | `(B, T, H, W)` | 1 = valid observed pixel |
| `mcar_mask` | `(B, T, H, W)` | missing completely at random |
| `mnar_mask` | `(B, T, H, W)` | missing not at random |
| `bloom_mask` | `(B, T, H, W)` | bloom supervision signal |
| `physics` | `(B, T, 6, H, W)` | `thetao, uo, vo, mlotst, zos, so` |
| `wind` | `(B, T, 4, H, W)` | `u10, v10, msl, tp` |
| `static` | `(B, 2, H, W)` | bathymetry, distance-to-coast |
| `discharge` | `(B, T, 2, H, W)` | `dis24, rowe` |
| `bgc_aux` | `(B, T, 5, H, W)` | `o2, no3, po4, si, nppv` |

Default config in `ModelConfig`:

- `T = 10`
- `H = W = 64`
- `H_fcast = 5`
- `embed_dim = 256`
- `n_experts = 4`

## Dataset Format

`dataset.py` expects patch files here:

```text
data/
`-- patches/
    |-- train/
    |-- val/
    `-- test/
```

Each split directory should contain `.npz` files with these required keys:

- `chl_obs`
- `obs_mask`
- `mcar_mask`
- `mnar_mask`
- `physics`
- `wind`
- `discharge`
- `bgc_aux`
- `static`
- `bloom_mask`
- `target_chl`

The loader derives:

- `land_mask` from `static`
- `target_mask` from `target_chl`

Expected per-file shapes:

```text
chl_obs     (10, 64, 64)
obs_mask    (10, 64, 64)
mcar_mask   (10, 64, 64)
mnar_mask   (10, 64, 64)
physics     (10, 6, 64, 64)
wind        (10, 4, 64, 64)
discharge   (10, 2, 64, 64)
bgc_aux     (10, 5, 64, 64)
static      (2, 64, 64)
bloom_mask  (10, 64, 64)
target_chl  (5, 64, 64)
```

## Losses

Training uses `MARASSLoss` from `loss.py`:

- Reconstruction: heteroscedastic NLL
- Holdout reconstruction: heteroscedastic NLL on artificially hidden observed pixels
- Forecast: masked Huber loss
- ERI: focal-weighted ordinal classification loss
- MoE auxiliary loss: load-balancing regularization

Default weights:

```text
recon    1.0
forecast 0.5
eri      0.3
aux      0.001
holdout  0.5
```

Forecast and ERI losses are ramped in over the first 20% of training steps.

## Setup

There is no pinned environment file in the repo. At minimum you need:

```bash
pip install torch tensorboard numpy
```

For evaluation figures:

```bash
pip install matplotlib
```

## Training

Single GPU:

```bash
python Train.py --patch-dir data/patches
```

CPU debug:

```bash
python Train.py --patch-dir data/patches --device cpu --batch-size 2 --num-workers 0 --no-amp
```

Multi-GPU with `torchrun`:

```bash
torchrun --nproc_per_node=2 Train.py --patch-dir data/patches --batch-size 8
```

Resume from a checkpoint:

```bash
python Train.py --patch-dir data/patches --resume checkpoints/last.pt
```

Useful outputs:

- Best checkpoint: `checkpoints/best.pt`
- Latest checkpoint: `checkpoints/last.pt`
- TensorBoard logs: `runs/`

## Evaluation

Run evaluation on the test split:

```bash
python eval.py --ckpt checkpoints/best.pt --patch-dir data/patches --out-dir eval_results
```

Optional flags:

- `--batch-size`
- `--num-workers`
- `--device cpu|cuda`
- `--no-amp`
- `--n-figures 0`
- `--no-figures`

Evaluation writes:

- `metrics.json`
- `confusion_matrix.csv`
- `calibration.csv`
- `figures/` with reconstruction, forecast, calibration, and routing plots

Reported metrics include:

- Reconstruction: RMSE, MAE, bias, R2, SSIM, CRPS
- Forecast: RMSE, MAE, SSIM by horizon
- ERI: accuracy, macro-F1, per-class F1, ordinal MAE
- Uncertainty: ECE and variance-error correlation
- Routing: expert utilization and entropy

## Smoke Tests

Several modules expose a direct smoke test entry point:

```bash
python model.py
python loss.py
python fusion.py
python masknet.py
python moe_decoder.py
```

`dataset.py` also includes a sanity-check runner, but its `__main__` block currently points at a Kaggle-style example path. These are shape and forward-pass checks, not full training runs.

## Notes

- `Train.py` builds loaders differently for single-process and DDP runs; both paths expect the same patch layout.
- During training, the model randomly holds out a fraction of observed pixels to create explicit gap-filling supervision.
- `eval.py` forces routing collection even in eval mode so MoE usage can be analyzed.
