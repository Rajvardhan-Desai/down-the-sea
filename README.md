# MM-MARAS — Multi-Modal Mask-Aware Regime-Adaptive Spatiotemporal Model

Satellite-derived chlorophyll-a (Chl-a) in the Bay of Bengal is routinely obscured by clouds, sun glint, and algal bloom events, leaving gaps that can span weeks over the same location. MM-MARAS reconstructs those gaps and forecasts future Chl-a by fusing five concurrent data streams: ocean optics, physical oceanography, wind and atmospheric forcing, biogeochemical tracers, and river discharge — along with explicit spatial encoding of the missing data pattern itself.

---

## Architecture overview

```
optical     (B, T, 2, H, W)   ──► OpticalEncoder  ──────────────────────────┐
physics     (B, T, 6, H, W)   ──► PhysicsEncoder  ─────────────────────────►│
masks       (B, T, 4, H, W)   ──► MaskNet         ────────────────────────► FusionModule (Perceiver IO)
bgc_aux     (B, T, 5, H, W)   ──► BGCAuxEncoder   ─────────────────────────►│    │
discharge   (B, T, 2, H, W)   ──► DischargeEncoder ────────────────────────►│    │
                                                                              │    ▼
                                                                              │  TemporalModule (ConvLSTM ×2)
                                                                              │    │
                                                                              │    ▼
                                                                              │  MoEDecoder (4 experts)
                                                                              │    │
                                                                              └───►┼──► recon       (B, 1, H, W)
                                                                                   ├──► forecast    (B, 5, H, W)
                                                                                   ├──► uncertainty (B, 1, H, W)
                                                                                   └──► eri         (B, 5, H, W)
```

### Encoders

| Module | Input | Output | Notes |
|---|---|---|---|
| `OpticalEncoder` | `(B, T, 2, H, W)` | `(B, T, 256, H, W)` | Swin-UNet; chl_obs + obs_mask |
| `PhysicsEncoder` | physics + wind + static | `(B, T, 256, H, W)` | Same backbone, separate weights; thetao, uo, vo, mlotst, zos, so + u10, v10, msl, tp |
| `MaskNet` | `(B, T, 4, H, W)` | `(B, T, 256, H, W)` | Type embedder + grid GNN + temporal mixer |
| `BGCAuxEncoder` | `(B, T, 5, H, W)` | `(B, T, 256, H, W)` | Swin-UNet; o2, no3, po4, si, nppv |
| `DischargeEncoder` | `(B, T, 2, H, W)` | `(B, T, 256, H, W)` | Swin-UNet; dis24, rowe |

### Fusion — `FusionModule` (Perceiver IO)

All five encoder outputs are fused via cross-attention. Learned latent vectors (`n_latents=64`) attend to KV tokens from all five streams, pooled to 16×16 before concatenation — giving 1280 KV tokens rather than a full 20,480, keeping memory tractable on a single GPU.

### Temporal — `TemporalModule` (ConvLSTM ×2)

Two stacked ConvLSTM layers with residual connection collapse the `(B, T, D, H, W)` fused sequence into a single spatial hidden state `(B, D, H, W)`.

### Decoder — `MoEDecoder` (Mixture of Experts)

Four expert decoders with global soft routing. Designed around the four dominant Bay of Bengal oceanographic regimes (pre-monsoon, summer monsoon, post-monsoon, winter). Load-balancing auxiliary loss prevents expert collapse during training.

### Output heads

| Head | Output | Description |
|---|---|---|
| `ReconHead` | `(B, 1, H, W)` | Reconstructed Chl-a for current timestep |
| `ForecastHead` | `(B, 5, H, W)` | Predicted Chl-a for next 5 timesteps |
| `UncertaintyHead` | `(B, 1, H, W)` | Per-pixel aleatoric log-variance |
| `ERIHead` | `(B, 5, H, W)` | Ecosystem Risk Index ordinal logits (5 levels) |

---

## Input specification

All tensors are float32. The batch dict expected by `MARASSModel.forward()`:

| Key | Shape | Variables |
|---|---|---|
| `chl_obs` | `(B, T, H, W)` | Observed Chl-a (log-space, normalised) |
| `obs_mask` | `(B, T, H, W)` | 1 = valid pixel |
| `mcar_mask` | `(B, T, H, W)` | 1 = MCAR missing |
| `mnar_mask` | `(B, T, H, W)` | 1 = MNAR missing |
| `bloom_mask` | `(B, T, H, W)` | 1 = bloom pixel |
| `physics` | `(B, T, 6, H, W)` | thetao, uo, vo, mlotst, zos, so |
| `wind` | `(B, T, 4, H, W)` | u10, v10, msl, tp |
| `static` | `(B, 2, H, W)` | Bathymetry, distance-to-coast |
| `discharge` | `(B, T, 2, H, W)` | dis24, rowe |
| `bgc_aux` | `(B, T, 5, H, W)` | o2, no3, po4, si, nppv |

Default: `T=10`, `H=W=64`, `D=256`.

---

## Loss

```
total = 1.0 × recon_loss
      + 0.5 × forecast_loss    (curriculum ramp over first 20% of steps)
      + 0.3 × eri_loss         (curriculum ramp over first 20% of steps)
      + 0.01 × aux_loss        (MoE load-balancing)
```

`recon_loss` is a heteroscedastic NLL that jointly supervises Chl-a prediction and uncertainty calibration. `eri_loss` uses an ordinal cross-entropy with penalty weighting for off-by-N errors. The curriculum ramp lets the model learn gap filling before being asked to forecast and classify.

---

## File structure

```
├── model.py              Top-level model + ModelConfig
├── optical_encoder.py    Swin-UNet optical backbone
├── physics_encoder.py    Physics + wind encoder
├── bgc_encoder.py        BGC auxiliary encoder
├── discharge_encoder.py  Discharge + runoff encoder
├── masknet.py            Structured missingness encoder
├── fusion.py             Perceiver IO cross-modal fusion
├── temporal.py           ConvLSTM temporal module
├── moe_decoder.py        Mixture-of-Experts decoder
├── loss.py               All loss functions + combined MARASSLoss
└── Train.py              Training loop (AMP, curriculum, TensorBoard)
```

Each file has a `run_smoke_test()` function. Run any of them directly to verify shapes and forward pass:

```bash
python model.py
python fusion.py
python masknet.py
```

---

## Training on Kaggle

```python
# Cell 1 — clone repo
!git clone https://github.com/Rajvardhan-Desai/down-the-sea.git /kaggle/working/maras
import sys
sys.path.insert(0, "/kaggle/working/maras")

# Cell 2 — launch training
!python /kaggle/working/maras/Train.py \
    --patch-dir /kaggle/input/your-dataset/patches \
    --ckpt-dir  /kaggle/working/checkpoints \
    --log-dir   /kaggle/working/runs \
    --epochs 50 \
    --batch-size 4
```

To pick up code changes mid-session:

```python
!git -C /kaggle/working/maras pull
# then restart kernel
```

### Recommended hardware

| GPU | Batch size | AMP |
|---|---|---|
| T4 / P100 (16 GB) | 4 | on |
| T4 (16 GB), tight | 2 | on |
| CPU debug | 2 | off (`--no-amp`) |

---

## Dependencies

```
torch >= 2.0
tensorboard
```

No other non-standard dependencies. All attention, convolution, and normalisation layers use native PyTorch.

---

## What's missing

`dataset.py` — a `build_dataloaders(patch_dir, batch_size, num_workers, pin_memory)` function returning `{"train": DataLoader, "val": DataLoader}`. Each batch must contain the keys listed in the input specification above, plus `land_mask (B, H, W)`, `target_chl (B, 5, H, W)`, and `target_mask (B, 5, H, W)` for the loss computation.