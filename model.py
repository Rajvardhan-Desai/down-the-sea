"""
model.py — MM-MARAS top-level model (v2)

Changes from v1:
    [PERF 1] holdout_frac 0.20 → 0.30 — stronger gap-filling supervision
    [PERF 2] ForecastHead: single 1×1 conv → multi-layer ConvNet with
             per-step refinement. Shared trunk + step-specific output layers.
    [FEAT 1] BloomForecastHead: predicts bloom probability at each of the
             5 forecast steps. Binary sigmoid output. This directly answers
             "can we predict algae blooms in advance?"
    [FEAT 2] compute_ecosystem_impact(): post-processing function that
             combines bloom probability, forecast magnitude, uncertainty,
             and coastal proximity into a single ecosystem impact score.
             Not a trained head — a derived metric for downstream reporting.

All other components (encoders, fusion, temporal, MoE decoder) are unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from masknet import MaskNet
from optical_encoder import OpticalEncoder
from physics_encoder import PhysicsEncoder
from bgc_encoder import BGCAuxEncoder
from discharge_encoder import DischargeEncoder
from fusion import FusionModule
from temporal import TemporalModule
from moe_decoder import MoEDecoder, compute_aux_loss


# ======================================================================
# Config
# ======================================================================

@dataclass
class ModelConfig:
    # Spatial / temporal dims
    T: int = 10             # input time steps
    H: int = 64             # patch height
    W: int = 64             # patch width
    H_fcast: int = 5        # forecast horizon

    # Channel counts (inputs)
    C_optical: int = 2      # chl_obs + obs_mask
    C_physics: int = 6      # thetao, uo, vo, mlotst, zos, so
    C_wind: int = 4         # u10, v10, msl, tp
    C_static: int = 2       # bathymetry, distance-to-coast
    C_masks: int = 4        # obs, mcar, mnar, bloom
    C_discharge: int = 2    # dis24, rowe
    C_bgc: int = 5          # o2, no3, po4, si, nppv

    # Internal feature dimension
    embed_dim: int = 256

    # MoE
    n_experts: int = 4

    # ERI ordinal levels
    n_eri_levels: int = 5

    # [PERF 1] Holdout fraction — increased from 0.20 to 0.30
    # Gap SSIM was 0.000 with 20% holdout. 30% forces the model to
    # fill more pixels from context, producing stronger gap-filling gradients.
    holdout_frac: float = 0.30

    # [FEAT 1] Bloom forecast — predict bloom probability at each forecast step
    bloom_threshold: float = 0.0    # threshold on NORMALIZED log-Chl-a
                                     # (set at runtime from norm_stats;
                                     #  default 0.0 is a placeholder)


# ======================================================================
# Output heads
# ======================================================================

class ReconHead(nn.Module):
    """Reconstruction: filled Chl-a map for the current time step."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.head = nn.Conv2d(cfg.embed_dim, 1, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        return self.head(x)                  # (B, 1, H, W)


class ForecastHead(nn.Module):
    """
    [PERF 2] Multi-layer forecast head with per-step refinement.

    Previous version: single Conv2d(D, H_fcast, 1) — one linear projection
    per pixel for all 5 steps. This forces all forecast horizons to share
    the same feature-to-output mapping, which is too constrained.

    New version:
        1. Shared trunk: Conv3×3 → GELU → Conv3×3 → GELU (spatial context)
        2. Per-step projection: separate Conv1×1 for each horizon step
        3. Each step gets a residual from the shared features

    This lets step +1 and step +5 use different output mappings while
    sharing the spatial feature extraction.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        D = cfg.embed_dim

        # Shared spatial trunk
        self.trunk = nn.Sequential(
            nn.Conv2d(D, D, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, D),
            nn.GELU(),
            nn.Conv2d(D, D // 2, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, D // 2),
            nn.GELU(),
        )

        # Per-step output layers
        self.step_heads = nn.ModuleList([
            nn.Conv2d(D // 2, 1, kernel_size=1)
            for _ in range(cfg.H_fcast)
        ])

    def forward(self, x: Tensor) -> Tensor:
        shared = self.trunk(x)                      # (B, D//2, H, W)
        steps = [head(shared) for head in self.step_heads]  # list of (B, 1, H, W)
        return torch.cat(steps, dim=1)              # (B, H_fcast, H, W)


class UncertaintyHead(nn.Module):
    """
    Aleatoric uncertainty: per-pixel log-variance.

    Output is log-variance (not std), consistent with heteroscedastic NLL loss.
    Exponentiate to get variance: uncertainty.exp()
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.head = nn.Conv2d(cfg.embed_dim, 1, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        return self.head(x)                  # (B, 1, H, W) — log-variance


class ERIHead(nn.Module):
    """
    Ecosystem Risk Index: ordinal classification into 5 risk levels (0–4).

    Output is raw logits (B, n_eri_levels, H, W).
    Apply cumulative link / ordinal softmax at loss time.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.head = nn.Conv2d(cfg.embed_dim, cfg.n_eri_levels, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        return self.head(x)                  # (B, 5, H, W)


class BloomForecastHead(nn.Module):
    """
    [FEAT 1] Bloom lead-time prediction.

    Predicts the probability of algal bloom at each of the H_fcast forecast
    steps. This tells you: "In 1/2/3/4/5 days, will this pixel be in bloom?"

    Output: (B, H_fcast, H, W) — raw logits, apply sigmoid for probabilities.

    Supervision: binary targets derived from target_chl at bloom threshold.
    Loss: binary cross-entropy with positive class weighting (blooms are rare).

    Why a separate head instead of thresholding the forecast?
        The forecast head predicts continuous Chl-a. Thresholding it at the
        bloom level produces poor binary predictions because:
        (a) the forecast is trained with Huber loss that down-weights bloom
            extremes (by design, for RMSE stability)
        (b) the decision boundary at the bloom threshold is never explicitly
            optimized
        A dedicated binary head with BCE loss directly optimizes the bloom
        detection decision boundary.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        D = cfg.embed_dim

        self.trunk = nn.Sequential(
            nn.Conv2d(D, D // 2, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, D // 2),
            nn.GELU(),
        )

        # Per-step binary prediction
        self.step_heads = nn.ModuleList([
            nn.Conv2d(D // 2, 1, kernel_size=1)
            for _ in range(cfg.H_fcast)
        ])

    def forward(self, x: Tensor) -> Tensor:
        shared = self.trunk(x)                      # (B, D//2, H, W)
        steps = [head(shared) for head in self.step_heads]
        return torch.cat(steps, dim=1)              # (B, H_fcast, H, W) — logits


# ======================================================================
# Ecosystem impact scoring (post-processing, not a trained head)
# ======================================================================

def compute_ecosystem_impact(
    bloom_probs: Tensor,
    forecast: Tensor,
    uncertainty: Tensor,
    static: Tensor,
    land_mask: Tensor,
) -> Tensor:
    """
    [FEAT 2] Compute per-pixel ecosystem impact score from model outputs.

    This is a derived metric, not a trained head. It combines:
        - Bloom probability (how likely is a bloom at each future step?)
        - Forecast magnitude (how intense is the predicted Chl-a?)
        - Uncertainty (high uncertainty near bloom boundary = early warning)
        - Coastal proximity (impacts are worse near shore: fisheries,
          aquaculture, tourism, coastal ecosystems)

    Score formula (each component 0-1, weighted sum):
        bloom_severity = max bloom probability across forecast steps
        intensity      = max normalized Chl-a forecast
        coastal_weight = 1 - distance_to_coast (from static channel 1)
        uncertainty_flag = sigmoid(log_var) — high variance near bloom = warning

        impact = 0.40 * bloom_severity
               + 0.25 * intensity
               + 0.20 * coastal_weight
               + 0.15 * uncertainty_flag

    These weights reflect domain priorities: bloom occurrence matters most,
    then intensity, then proximity to vulnerable coastal ecosystems, then
    the model's own confidence. The weights can be tuned for specific
    management applications.

    Args:
        bloom_probs: (B, H_fcast, H, W)  sigmoid bloom probabilities
        forecast:    (B, H_fcast, H, W)  predicted Chl-a (normalized)
        uncertainty: (B, 1, H, W)        log-variance
        static:      (B, 2, H, W)        channel 1 = distance-to-coast (0-1)
        land_mask:   (B, H, W)           1 = land

    Returns:
        impact: (B, H, W)  score in [0, 1], 0 = no impact, 1 = severe impact
    """
    ocean = 1.0 - land_mask                                # (B, H, W)

    # 1. Bloom severity: max probability across all forecast steps
    bloom_severity = bloom_probs.max(dim=1).values         # (B, H, W)

    # 2. Forecast intensity: max predicted Chl-a, clamped to [0, 1]
    intensity = forecast.clamp(min=0).max(dim=1).values    # (B, H, W)
    # Normalize to [0, 1] using a soft saturation
    intensity = torch.tanh(intensity)                      # smooth 0-1

    # 3. Coastal proximity weight: static[:, 1] is distance-to-coast (0-1)
    # Invert: closer to coast = higher weight
    dist_coast = static[:, 1].clamp(0, 1)                 # (B, H, W)
    coastal_weight = 1.0 - dist_coast                      # close to coast = high

    # 4. Uncertainty flag: high log-variance → high risk
    uncertainty_flag = torch.sigmoid(uncertainty.squeeze(1))  # (B, H, W) in [0, 1]

    # Weighted composite
    impact = (
        0.40 * bloom_severity +
        0.25 * intensity +
        0.20 * coastal_weight +
        0.15 * uncertainty_flag
    )

    # Zero out land pixels
    impact = impact * ocean

    return impact.clamp(0, 1)


# ======================================================================
# Top-level model
# ======================================================================

class MARASSModel(nn.Module):
    """
    MM-MARAS v2 — Multi-Modal Mask-Aware Regime-Adaptive Spatiotemporal Model.

    Changes from v1:
        - holdout_frac 0.20 → 0.30
        - ForecastHead: multi-layer with per-step refinement
        - BloomForecastHead: per-step bloom probability prediction
        - compute_ecosystem_impact() available for post-processing

    Forward output keys:
        recon            (B, 1, H, W)          filled Chl-a (current step)
        forecast         (B, H_fcast, H, W)    future Chl-a
        uncertainty      (B, 1, H, W)          log-variance (aleatoric)
        eri              (B, 5, H, W)          ERI ordinal logits
        bloom_forecast   (B, H_fcast, H, W)    bloom logits (new)
        routing_weights  (B, n_experts)         MoE weights — training only
        holdout_mask     (B, H, W)             held-out pixels — training only
    """

    def __init__(self, cfg: ModelConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or ModelConfig()
        cfg = self.cfg

        self.masknet       = MaskNet(cfg.embed_dim, cfg.T)
        self.opt_enc       = OpticalEncoder(in_channels=cfg.C_optical, embed_dim=cfg.embed_dim)
        self.phy_enc       = PhysicsEncoder(
            C_physics=cfg.C_physics,
            C_wind=cfg.C_wind,
            C_static=cfg.C_static,
            embed_dim=cfg.embed_dim,
        )
        self.bgc_enc       = BGCAuxEncoder(C_bgc=cfg.C_bgc, embed_dim=cfg.embed_dim)
        self.discharge_enc = DischargeEncoder(C_discharge=cfg.C_discharge, embed_dim=cfg.embed_dim)
        self.fusion        = FusionModule(embed_dim=cfg.embed_dim, H=cfg.H, W=cfg.W)
        self.temporal      = TemporalModule(embed_dim=cfg.embed_dim)
        self.decoder       = MoEDecoder(embed_dim=cfg.embed_dim, n_experts=cfg.n_experts)

        self.recon_head        = ReconHead(cfg)
        self.forecast_head     = ForecastHead(cfg)          # [PERF 2] deeper
        self.uncertainty_head  = UncertaintyHead(cfg)
        self.eri_head          = ERIHead(cfg)
        self.bloom_fcast_head  = BloomForecastHead(cfg)     # [FEAT 1] new

    def forward(self, batch: dict) -> dict[str, Tensor]:
        cfg = self.cfg

        # ------------------------------------------------------------------
        # Unpack and validate
        # ------------------------------------------------------------------
        chl_obs    = batch["chl_obs"]       # (B, T, H, W)
        obs_mask   = batch["obs_mask"]      # (B, T, H, W)
        mcar_mask  = batch["mcar_mask"]     # (B, T, H, W)
        mnar_mask  = batch["mnar_mask"]     # (B, T, H, W)
        bloom_mask = batch["bloom_mask"]    # (B, T, H, W)
        physics    = batch["physics"]       # (B, T, 6, H, W)
        wind       = batch["wind"]          # (B, T, 4, H, W)
        static     = batch["static"]        # (B, 2, H, W)
        discharge  = batch["discharge"]     # (B, T, 2, H, W)
        bgc_aux    = batch["bgc_aux"]       # (B, T, 5, H, W)

        B, T, H, W = chl_obs.shape
        if T != cfg.T:
            raise RuntimeError(f"Expected T={cfg.T}, got {T}")
        if H != cfg.H or W != cfg.W:
            raise RuntimeError(f"Expected ({cfg.H},{cfg.W}), got ({H},{W})")

        # ------------------------------------------------------------------
        # 1. Prepare multi-stream inputs
        # ------------------------------------------------------------------
        optical = torch.stack([chl_obs, obs_mask], dim=2)
        masks   = torch.stack([obs_mask, mcar_mask, mnar_mask, bloom_mask], dim=2)

        # [PERF 1] Holdout — 30% of observed pixels zeroed during training
        holdout_mask = None
        if self.training and cfg.holdout_frac > 0:
            seq_holdout_mask = (
                (obs_mask > 0.5) & (torch.rand_like(obs_mask) < cfg.holdout_frac)
            ).float()
            optical = optical.clone()
            keep_mask = 1.0 - seq_holdout_mask
            optical[:, :, 0] = optical[:, :, 0] * keep_mask
            optical[:, :, 1] = optical[:, :, 1] * keep_mask
            holdout_mask = seq_holdout_mask[:, -1]

        # ------------------------------------------------------------------
        # 2. Encode (five independent streams)
        # ------------------------------------------------------------------
        mask_emb = self.masknet(masks)
        opt_feat = self.opt_enc(optical)
        phy_feat = self.phy_enc(physics, wind, static)
        bgc_feat = self.bgc_enc(bgc_aux)
        dis_feat = self.discharge_enc(discharge)

        # ------------------------------------------------------------------
        # 3. Fuse (Perceiver IO)
        # ------------------------------------------------------------------
        fused = self.fusion(opt_feat, phy_feat, mask_emb, bgc_feat, dis_feat)

        # ------------------------------------------------------------------
        # 4. Temporal propagation
        # ------------------------------------------------------------------
        state = self.temporal(fused)

        # ------------------------------------------------------------------
        # 5. Decode (regime-adaptive MoE)
        # ------------------------------------------------------------------
        if self.training:
            decoded, routing_weights = self.decoder(state, return_routing=True)
        else:
            decoded = self.decoder(state)
            routing_weights = None

        # ------------------------------------------------------------------
        # 6. Output heads
        # ------------------------------------------------------------------
        outputs = {
            "recon":          self.recon_head(decoded),
            "forecast":       self.forecast_head(decoded),
            "uncertainty":    self.uncertainty_head(decoded),
            "eri":            self.eri_head(decoded),
            "bloom_forecast": self.bloom_fcast_head(decoded),   # [FEAT 1]
        }
        if routing_weights is not None:
            outputs["routing_weights"] = routing_weights
        if holdout_mask is not None:
            outputs["holdout_mask"] = holdout_mask

        return outputs

    def param_count(self) -> dict[str, int]:
        """Return parameter counts per sub-module and total."""
        counts = {}
        for name, module in self.named_children():
            counts[name] = sum(p.numel() for p in module.parameters())
        counts["total"] = sum(p.numel() for p in self.parameters())
        return counts


# ======================================================================
# Smoke test
# ======================================================================

def run_smoke_test() -> None:
    """
    python model.py
    """
    import logging
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    cfg   = ModelConfig()
    model = MARASSModel(cfg)

    B = 2
    fake_batch = {
        "chl_obs":    torch.randn(B, cfg.T, cfg.H, cfg.W),
        "obs_mask":   torch.randint(0, 2, (B, cfg.T, cfg.H, cfg.W)).float(),
        "mcar_mask":  torch.zeros(B, cfg.T, cfg.H, cfg.W),
        "mnar_mask":  torch.zeros(B, cfg.T, cfg.H, cfg.W),
        "bloom_mask": torch.zeros(B, cfg.T, cfg.H, cfg.W),
        "physics":    torch.randn(B, cfg.T, cfg.C_physics,   cfg.H, cfg.W),
        "wind":       torch.randn(B, cfg.T, cfg.C_wind,      cfg.H, cfg.W),
        "static":     torch.randn(B, cfg.C_static,           cfg.H, cfg.W),
        "discharge":  torch.randn(B, cfg.T, cfg.C_discharge, cfg.H, cfg.W),
        "bgc_aux":    torch.randn(B, cfg.T, cfg.C_bgc,       cfg.H, cfg.W),
    }

    expected_shapes = {
        "recon":          (B, 1,                cfg.H, cfg.W),
        "forecast":       (B, cfg.H_fcast,      cfg.H, cfg.W),
        "uncertainty":    (B, 1,                cfg.H, cfg.W),
        "eri":            (B, cfg.n_eri_levels, cfg.H, cfg.W),
        "bloom_forecast": (B, cfg.H_fcast,      cfg.H, cfg.W),
    }

    all_ok = True

    # --- Eval mode ---
    model.eval()
    with torch.no_grad():
        out_eval = model(fake_batch)

    print("\n--- Output shapes (eval) ---")
    for key, tensor in out_eval.items():
        exp    = expected_shapes.get(key)
        status = "OK" if exp is None or tuple(tensor.shape) == exp else f"MISMATCH"
        print(f"  {key:<18} {str(tuple(tensor.shape)):<30} {status}")
        if "MISMATCH" in status:
            all_ok = False

    assert "routing_weights" not in out_eval, "routing_weights should be absent in eval mode"

    # --- Train mode ---
    model.train()
    out_train = model(fake_batch)

    print("\n--- Output shapes (train) ---")
    for key, tensor in out_train.items():
        exp    = expected_shapes.get(key)
        status = "OK" if exp is None or tuple(tensor.shape) == exp else f"MISMATCH"
        print(f"  {key:<18} {str(tuple(tensor.shape)):<30} {status}")
        if "MISMATCH" in status:
            all_ok = False

    assert "routing_weights" in out_train
    assert "bloom_forecast" in out_train, "bloom_forecast missing from outputs"

    # Bloom forecast should be logits (can be any sign)
    bf = out_train["bloom_forecast"]
    print(f"\n  bloom_forecast range: [{bf.min().item():.3f}, {bf.max().item():.3f}]")

    # Ecosystem impact scoring (post-processing)
    print("\n--- Ecosystem impact scoring ---")
    bloom_probs = torch.sigmoid(out_eval["bloom_forecast"])
    impact = compute_ecosystem_impact(
        bloom_probs   = bloom_probs,
        forecast      = out_eval["forecast"],
        uncertainty   = out_eval["uncertainty"],
        static        = fake_batch["static"],
        land_mask     = torch.zeros(B, cfg.H, cfg.W),
    )
    print(f"  impact shape: {tuple(impact.shape)}  (expected {(B, cfg.H, cfg.W)})")
    print(f"  impact range: [{impact.min().item():.3f}, {impact.max().item():.3f}]")
    assert tuple(impact.shape) == (B, cfg.H, cfg.W)

    aux = compute_aux_loss(out_train["routing_weights"])
    print(f"\n  aux_loss: {aux.item():.4f}")

    print("\n--- Parameter counts ---")
    for name, n in model.param_count().items():
        print(f"  {name:<22} {n:>10,}")

    if all_ok:
        print("\nSmoke test passed.")
    else:
        raise RuntimeError("Shape mismatches detected.")


if __name__ == "__main__":
    run_smoke_test()