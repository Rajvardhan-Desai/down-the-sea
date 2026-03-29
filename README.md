# MM-MARAS

**Multi-Modal Mask-Aware Regime-Adaptive Spatiotemporal Model**

MM-MARAS is a deep learning system for Bay of Bengal chlorophyll-a (Chl-a) reconstruction, 5-step forecasting, pixel-level uncertainty estimation, algal bloom early warning, and ecosystem risk assessment. It fuses five heterogeneous input streams through a Perceiver IO cross-attention module, processes the temporal sequence with a two-layer ConvLSTM, and decodes through a soft-routing Mixture-of-Experts decoder.

The project has two parts: a preprocessing pipeline that downloads, aligns, normalizes, and patches multi-source oceanographic data, and a PyTorch model (~42.5M parameters) that consumes those patches.

## Results (v2)

Trained on 1,264 patches from the Bay of Bengal (2021-2024), evaluated on 260 test patches (2024-2025). Training used 2x T4 GPUs with DDP, AMP, batch size 4 per GPU. Phase 1: 60 epochs at lr=1e-4; Phase 2: 25 epochs at lr=5e-5. Total training time ~7.5 hours.

### Reconstruction

| Subset | Pixels | RMSE | MAE | Bias | R² | SSIM |
|---|---|---|---|---|---|---|
| All ocean | 1,031,940 | 0.1457 | 0.0670 | -0.0264 | 0.881 | 0.736 |
| Valid (observed) | 848,900 | 0.1018 | 0.0317 | +0.0051 | 0.952 | 0.927 |
| Gap (missing) | 183,040 | 0.2675 | 0.2310 | -0.1728 | -- | 0.000 |

CRPS: 0.0228.

### Forecast

| Horizon | RMSE | MAE | SSIM |
|---|---|---|---|
| +1 day | 0.1315 | 0.0465 | 0.889 |
| +2 days | 0.1572 | 0.0573 | 0.829 |
| +3 days | 0.1791 | 0.0677 | 0.754 |
| +4 days | 0.1946 | 0.0750 | 0.711 |
| +5 days | 0.2046 | 0.0827 | 0.694 |

### ERI classification

Accuracy: 0.9958. Macro F1: 0.687. Ordinal MAE: 0.004. Per-class F1: class 0 = 0.998, class 1 = 0.189, class 2 = 0.553, class 3 = 0.759, class 4 = 0.935.

### Uncertainty calibration

ECE: 0.0002. Variance-error correlation: 0.581.

### MoE routing

Expert utilization: 1.000 (perfect). Routing entropy: 1.386 / 1.386 (maximum). All four experts equally utilized (24.8%, 25.0%, 25.0%, 25.2%).

### v1 to v2 comparison

| Metric | v1 | v2 | Change |
|---|---|---|---|
| All RMSE | 0.2311 | **0.1457** | -37% |
| All R² | 0.700 | **0.881** | +26% |
| Valid RMSE | 0.1111 | **0.1018** | -8% |
| Gap RMSE | 0.4940 | **0.2675** | -46% |
| CRPS | 0.0264 | **0.0228** | -14% |
| Forecast +1 SSIM | 0.862 | **0.889** | +3% |
| Forecast +5 SSIM | 0.671 | **0.694** | +3% |
| ECE | 0.0016 | **0.0002** | -88% |
| Var-Err corr | 0.547 | **0.581** | +6% |
| Expert utilization | 0.849 | **1.000** | +18% |
| Expert 0 weight | 5.3% | **24.8%** | fixed |

## Model outputs

Given a temporal patch of 10 satellite and environmental time steps, the model predicts:

| Output | Shape | Description |
|---|---|---|
| `recon` | `(B, 1, H, W)` | Gap-filled Chl-a for the last input time step |
| `forecast` | `(B, 5, H, W)` | Chl-a predictions for the next 5 time steps |
| `uncertainty` | `(B, 1, H, W)` | Per-pixel log-variance (aleatoric uncertainty) |
| `eri` | `(B, 5, H, W)` | Ecosystem Risk Index ordinal logits (5 levels: 0-4) |
| `bloom_forecast` | `(B, 5, H, W)` | Per-step bloom probability logits |
| `routing_weights` | `(B, 4)` | MoE expert blend weights (training only) |
| `holdout_mask` | `(B, H, W)` | Pixels held out for gap-filling supervision (training only) |

Additionally, `compute_ecosystem_impact()` derives a per-pixel 0-1 ecosystem impact score by combining bloom probability, forecast intensity, uncertainty, and coastal proximity.

## Architecture

```
optical (chl_obs + obs_mask)      --> OpticalEncoder   (Swin-UNet, 2ch)  --\
physics + wind + static          --> PhysicsEncoder    (Swin-UNet, 12ch) --\
masks (obs/mcar/mnar/bloom)      --> MaskNet           (GNN + temporal)  --> FusionModule --> TemporalModule --> MoEDecoder --> Heads
bgc_aux                          --> BGCAuxEncoder     (Swin-UNet, 5ch)  --/   (Perceiver IO)  (ConvLSTM x2)    (4 experts)
discharge                        --> DischargeEncoder  (Swin-UNet, 2ch)  --/
```

**Encoders.** All four spatial encoders reuse the same Swin-UNet backbone (patch embed, 3-stage encoder, bottleneck, 3-stage decoder with skip connections) with independent weights. Stage dimensions: 64, 128, 256; window sizes: 8, 8, 4. The physics encoder concatenates ocean state (6ch), wind/atmosphere (4ch), and static context (2ch, broadcast over time) into 12 channels. MaskNet classifies pixels into five missingness types via learned embeddings, propagates context through two rounds of masked grid-graph convolution, and mixes across time with depthwise temporal convolution.

**Fusion.** Perceiver IO with 64 latent queries cross-attending to spatially pooled tokens from all five streams (1280 KV tokens). Self-attention refinement, then decode back to full resolution via position-aware spatial queries with residual blend.

**Temporal.** Two stacked ConvLSTM layers. Layer 1 returns the full hidden sequence; layer 2 processes it with global-context bias. Final hidden state + sequence mean residual produces (B, D, H, W).

**Decoder.** Four expert ConvNets soft-blended per sample via global routing (average pool, MLP, softmax). Load-balancing auxiliary loss at weight 0.01.

**Heads.** Reconstruction (1x1 conv), forecast (shared 2-layer trunk + per-step projections), uncertainty (1x1 conv), ERI ordinal logits (1x1 conv), bloom forecast (shared trunk + per-step binary outputs). Total: ~42.5M parameters.

## Repository layout

```
Preprocessing pipeline
├── pipeline.py           Main orchestration script
├── config.py             All settings, paths, variable lists, norm stats
├── loader.py             Format detection, downloaders (CMEMS, CDS, CEMS)
├── aligner.py            Coordinate standardization, regridding, time alignment
├── masker.py             Observation, land, bloom, MCAR, MNAR mask generation
├── normalizer.py         log1p, z-score, min-max normalization
├── patcher.py            Spatiotemporal patch extraction, train/val/test split

Model
├── model.py              Top-level MARASSModel + ModelConfig + output heads
├── optical_encoder.py    Swin-UNet backbone (shared architecture for all spatial encoders)
├── physics_encoder.py    Ocean state + wind + static encoder (12ch input)
├── bgc_encoder.py        BGC auxiliary encoder (o2, no3, po4, si, nppv)
├── discharge_encoder.py  River discharge + runoff encoder (dis24, rowe)
├── masknet.py            MaskNet: missingness type embedding + grid GNN + temporal mixer
├── fusion.py             Perceiver IO cross-modal fusion (5 streams)
├── temporal.py           Two-layer ConvLSTM temporal module
├── moe_decoder.py        Soft-routing Mixture-of-Experts decoder (4 experts)
├── augment.py            Spatial data augmentation (flips + 90° rotations)
├── dataset.py            PyTorch Dataset + DataLoader factory
├── loss.py               MARASSLoss: 6 loss terms with curriculum scheduling
├── Train.py              Training loop (single-GPU, DDP, AMP, curriculum)
├── eval.py               Test-set evaluation, bloom forecast metrics, ecosystem impact

Data (generated by pipeline)
└── data/
    ├── raw/              Downloaded source files
    ├── patches/
    │   ├── train/        train_000000.npz ... (1,264 patches)
    │   ├── val/          val_000000.npz ...   (260 patches)
    │   └── test/         test_000000.npz ...  (260 patches)
    └── stats/
        └── norm_stats_bay_of_bengal.json
```

---

## Preprocessing pipeline

### Domain

**Bay of Bengal** (IHO S-23 definition): 79.5-95.5°E, 5.5-22.5°N. Temporal coverage: 2021-01-01 to 2025-12-31 (5 years, daily).

### Data sources

| Stream | Product | Resolution | Variables |
|---|---|---|---|
| BGC Chl-a + nutrients | CMEMS `GLOBAL_MULTIYEAR_BGC_001_029` | 0.25°, daily | chl, o2, no3, po4, si, nppv |
| Ocean physics | CMEMS `GLOBAL_MULTIYEAR_PHY_001_030` | 0.083°, daily | thetao, uo, vo, mlotst, zos, so |
| Atmosphere | ERA5 `derived-era5-single-levels-daily-statistics` | 0.25°, daily | u10, v10, msl (daily mean), tp (daily sum) |
| Freshwater | GloFAS `cems-glofas-historical` | 0.05°, daily | dis24 (discharge), rowe (runoff) |
| Bathymetry | GEBCO 2025 Global Grid | 15 arc-second | elevation (negative = ocean) |

All modalities are regridded to the BGC 0.25° grid via bilinear interpolation. Discharge and precipitation use conservative regridding when xesmf is available. Time axes are aligned to the Chl-a reference.

### Pipeline steps

1. **Download** raw data from CMEMS, CDS, and CEMS (resumable)
2. **Load and align**: standardize coordinates, clip to domain, extract surface level, resample sub-daily to daily, regrid to BGC grid, align time axes
3. **Build masks**: obs_mask, land_mask, bloom_mask (10 mg/m³ threshold), mcar_mask, mnar_mask
4. **Normalize**: log1p + z-score for skewed variables, plain z-score for Gaussian variables, min-max for static context. Statistics from training split only.
5. **Build static context**: bathymetry + distance-to-coast, min-max normalized
6. **Extract patches**: (T=10, H=64, W=64) windows, stride 32, 5-step forecast horizon, 70/15/15 temporal split

### Running the pipeline

```bash
python pipeline.py --bathy data/raw/gebco_2025_n22.5_s5.5_w79.5_e95.5.nc
```

Skip downloads: add `--no-download` with `--chl`, `--physics`, `--era5-wind`, `--discharge` paths. Reuse stats: add `--load-stats`.

### Authentication

CMEMS: `copernicusmarine login`. CDS: `~/.cdsapirc` with API key.

### Installation (pipeline)

```bash
pip install copernicusmarine cdsapi xarray netCDF4 numpy scipy dask pandas torch
```

Optional: `pip install xesmf` for conservative regridding.

---

## Model

### Input contract

| Key | Shape | Description |
|---|---|---|
| `chl_obs` | `(B, 10, 64, 64)` | Log Chl-a, NaN-filled with 0.0 |
| `obs_mask` | `(B, 10, 64, 64)` | 1 = valid observed pixel |
| `mcar_mask` | `(B, 10, 64, 64)` | 1 = missing completely at random |
| `mnar_mask` | `(B, 10, 64, 64)` | 1 = missing not at random |
| `bloom_mask` | `(B, 10, 64, 64)` | Bloom event labels |
| `physics` | `(B, 10, 6, 64, 64)` | thetao, uo, vo, mlotst, zos, so |
| `wind` | `(B, 10, 4, 64, 64)` | u10, v10, msl, tp |
| `static` | `(B, 2, 64, 64)` | Bathymetry, distance-to-coast |
| `discharge` | `(B, 10, 2, 64, 64)` | dis24, rowe |
| `bgc_aux` | `(B, 10, 5, 64, 64)` | o2, no3, po4, si, nppv |

### Losses

Six loss terms with curriculum scheduling (forecast, ERI, and bloom losses ramped in over first 20% of training):

| Loss | Weight | Description |
|---|---|---|
| Reconstruction | 1.0 | Heteroscedastic NLL on observed ocean pixels |
| Holdout recon | 0.5 | NLL + Laplacian gradient matching on held-out pixels |
| Forecast | 0.5 | Masked Huber (delta=0.5) over forecast window |
| ERI | 0.3 | Focal ordinal cross-entropy (gamma=2.0) |
| Bloom forecast | 0.3 | Binary CE with pos_weight=20 per forecast step |
| MoE auxiliary | 0.01 | Load-balancing (Switch Transformer) |

### Data augmentation

Random flips (horizontal p=0.5, vertical p=0.5) and 90° rotations (p=0.5) applied consistently across all batch tensors during training. Gives up to 8x effective data diversity.

### Training

```bash
# Phase 1: 60 epochs at lr=1e-4
torchrun --nproc_per_node=2 Train.py \
    --patch-dir data/patches --batch-size 4 --epochs 60 --lr 1e-4 --warmup-epochs 10

# Phase 2: fine-tune 25 more at lr=5e-5
torchrun --nproc_per_node=2 Train.py \
    --patch-dir data/patches --resume checkpoints/best.pt \
    --batch-size 4 --epochs 85 --lr 5e-5 --warmup-epochs 5
```

Peak VRAM: ~10.6 GB per T4 GPU. ~218 seconds per epoch.

### Evaluation

```bash
python eval.py --ckpt checkpoints/best.pt --patch-dir data/patches --out-dir eval_results
```

Outputs: `metrics.json`, `confusion_matrix.csv`, `calibration.csv`, and `figures/` containing reconstruction panels, forecast panels, bloom probability + ecosystem impact maps, calibration diagram, and routing bar chart.

Reported metrics: reconstruction (RMSE, MAE, bias, R², SSIM, CRPS over all/valid/gap subsets), forecast (RMSE, MAE, SSIM per horizon), ERI (accuracy, macro-F1, per-class F1, ordinal MAE), uncertainty (ECE, variance-error correlation), MoE routing (per-expert weights, entropy, utilization), bloom forecast (per-step precision, recall, F1), and ecosystem impact (mean, percentiles, high-impact fraction).

### Smoke tests

```bash
python model.py        python loss.py          python augment.py
python fusion.py       python temporal.py      python moe_decoder.py
python masknet.py      python optical_encoder.py
python physics_encoder.py   python bgc_encoder.py   python discharge_encoder.py
```

---

## Bloom early warning and ecosystem impact analysis

### Bloom lead-time prediction

The `BloomForecastHead` predicts bloom probability at each of the 5 forecast steps (t+1 through t+5 days). Trained with binary cross-entropy (pos_weight=20 for extreme bloom rarity). Targets derived from forecast Chl-a at 10 mg/m³ (10.85 in normalized space).

```python
bloom_probs = torch.sigmoid(outputs["bloom_forecast"])  # (B, 5, H, W)
# bloom_probs[:, 0] = P(bloom at +1 day)
# bloom_probs[:, 4] = P(bloom at +5 days)
```

### Ecosystem impact scoring

Post-processing function combining four model outputs:

| Component | Weight | Source |
|---|---|---|
| Bloom severity | 0.40 | Max bloom probability across 5 steps |
| Chl-a intensity | 0.25 | Max forecast Chl-a (tanh-saturated) |
| Coastal proximity | 0.20 | 1 - distance_to_coast |
| Uncertainty flag | 0.15 | Sigmoid of log-variance |

```python
from model import compute_ecosystem_impact
impact = compute_ecosystem_impact(
    torch.sigmoid(outputs["bloom_forecast"]),
    outputs["forecast"], outputs["uncertainty"],
    batch["static"], batch["land_mask"],
)  # (B, H, W) in [0, 1]
```

### Interpreting outputs

A single inference pass produces: bloom probability timeline (5 maps, one per future day), ecosystem impact map (values above 0.6 indicate high risk), uncertainty map (high uncertainty near bloom threshold is a warning signal), and ERI classification (5-level ordinal risk: none/low/moderate/high/extreme).

---

## Notes

- Training loss goes negative because heteroscedastic NLL drives below zero with well-calibrated uncertainty. This is expected.
- 30% of observed pixels are randomly held out during training for gap-filling supervision.
- Validation gap RMSE uses a deterministic holdout mask (SHA-1 hash seed) for reproducibility.
- `eval.py` forces routing weight collection in eval mode for post-training analysis.
- The optical encoder supports optional SatMAE weight initialization via `load_satmae_patch_embed()`.
- The contrastive pre-alignment loss in `fusion.py` is available for an optional pretraining phase.
- All pipeline downloads are resumable. Normalization statistics are training-split-only.
- v2 checkpoints are not compatible with v1 (new heads, different forecast head architecture).
- cuFFT/cuDNN/cuBLAS registration warnings during Kaggle DDP training are harmless.