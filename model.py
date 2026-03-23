"""
model.py — MM-MARAS top-level model

All sub-components are fully implemented. No stubs remain.

Internal feature dimension flows:
    Inputs
      optical:    (B, T, 2, H, W)      chl_obs + obs_mask stacked
      physics:    (B, T, 6, H, W)      thetao, uo, vo, mlotst, zos, so
      wind:       (B, T, 4, H, W)      u10, v10, msl, tp
      static:     (B, 2, H, W)
      masks:      (B, T, 4, H, W)      obs/mcar/mnar/bloom stacked
      discharge:  (B, T, 2, H, W)      dis24, rowe   ← new stream
      bgc_aux:    (B, T, 5, H, W)      o2, no3, po4, si, nppv   ← new stream

    After encoders → (B, T, D, H, W)   D = cfg.embed_dim  (5 separate encoders)
    After fusion   → (B, T, D, H, W)   Perceiver IO — 5 streams, 5×P² KV tokens
    After temporal → (B, D, H, W)      last hidden state
    After decoder  → (B, D, H, W)

    Heads
      recon:       (B, 1, H, W)
      forecast:    (B, H_fcast, H, W)
      uncertainty: (B, 1, H, W)        log-variance (aleatoric)
      eri:         (B, 5, H, W)        ordinal logits (5 levels)

    Training-only outputs
      routing_weights: (B, n_experts)  MoE routing — use for aux loss
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
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
    C_discharge: int = 2    # dis24, rowe  (new stream)
    C_bgc: int = 5          # o2, no3, po4, si, nppv  (new stream)

    # Internal feature dimension
    embed_dim: int = 256

    # MoE
    n_experts: int = 4

    # ERI ordinal levels
    n_eri_levels: int = 5


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
    """Forecast: predicted Chl-a for H_fcast future time steps."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.head = nn.Conv2d(cfg.embed_dim, cfg.H_fcast, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        return self.head(x)                  # (B, H_fcast, H, W)


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


# ======================================================================
# Top-level model
# ======================================================================

class MARASSModel(nn.Module):
    """
    MM-MARAS — Multi-Modal Mask-Aware Regime-Adaptive Spatiotemporal Model.

    Accepts a batch dict from MARASSDataset and returns a dict of outputs.

    Forward input keys (all float32 tensors):
        chl_obs     (B, T, H, W)
        obs_mask    (B, T, H, W)
        mcar_mask   (B, T, H, W)
        mnar_mask   (B, T, H, W)
        bloom_mask  (B, T, H, W)
        physics     (B, T, 6, H, W)   thetao, uo, vo, mlotst, zos, so
        wind        (B, T, 4, H, W)   u10, v10, msl, tp
        static      (B, 2, H, W)
        discharge   (B, T, 2, H, W)   dis24, rowe   ← new
        bgc_aux     (B, T, 5, H, W)   o2, no3, po4, si, nppv   ← new

    Forward output keys:
        recon            (B, 1, H, W)          filled Chl-a (current step)
        forecast         (B, H_fcast, H, W)    future Chl-a
        uncertainty      (B, 1, H, W)          log-variance (aleatoric)
        eri              (B, 5, H, W)          ERI ordinal logits
        routing_weights  (B, n_experts)         MoE weights — training only

    Training loop example:
        outputs  = model(batch)
        task_loss = compute_task_loss(outputs, batch)
        aux_loss  = compute_aux_loss(outputs["routing_weights"])
        loss      = task_loss + 0.01 * aux_loss
        loss.backward()
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

        self.recon_head       = ReconHead(cfg)
        self.forecast_head    = ForecastHead(cfg)
        self.uncertainty_head = UncertaintyHead(cfg)
        self.eri_head         = ERIHead(cfg)

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
        wind       = batch["wind"]          # (B, T, 4, H, W)  u10,v10,msl,tp
        static     = batch["static"]        # (B, 2, H, W)
        discharge  = batch["discharge"]     # (B, T, 2, H, W)  dis24, rowe
        bgc_aux    = batch["bgc_aux"]       # (B, T, 5, H, W)  o2,no3,po4,si,nppv

        B, T, H, W = chl_obs.shape
        assert T == cfg.T, f"Expected T={cfg.T}, got {T}"
        assert H == cfg.H and W == cfg.W, f"Expected ({cfg.H},{cfg.W}), got ({H},{W})"

        # ------------------------------------------------------------------
        # 1. Prepare multi-stream inputs
        # ------------------------------------------------------------------
        optical = torch.stack([chl_obs, obs_mask], dim=2)                           # (B, T, 2, H, W)
        masks   = torch.stack([obs_mask, mcar_mask, mnar_mask, bloom_mask], dim=2)  # (B, T, 4, H, W)

        # ------------------------------------------------------------------
        # 2. Encode (five independent streams)
        # ------------------------------------------------------------------
        mask_emb = self.masknet(masks)                      # (B, T, D, H, W)
        opt_feat = self.opt_enc(optical)                    # (B, T, D, H, W)
        phy_feat = self.phy_enc(physics, wind, static)      # (B, T, D, H, W)
        bgc_feat = self.bgc_enc(bgc_aux)                    # (B, T, D, H, W)
        dis_feat = self.discharge_enc(discharge)            # (B, T, D, H, W)

        # ------------------------------------------------------------------
        # 3. Fuse (Perceiver IO — 5 streams, 5×P² KV tokens)
        # ------------------------------------------------------------------
        fused = self.fusion(opt_feat, phy_feat, mask_emb, bgc_feat, dis_feat)  # (B, T, D, H, W)

        # ------------------------------------------------------------------
        # 4. Temporal propagation
        # ------------------------------------------------------------------
        state = self.temporal(fused)                       # (B, D, H, W)

        # ------------------------------------------------------------------
        # 5. Decode (regime-adaptive)
        #    Return routing weights during training for the aux loss.
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
            "recon":       self.recon_head(decoded),        # (B, 1, H, W)
            "forecast":    self.forecast_head(decoded),     # (B, H_fcast, H, W)
            "uncertainty": self.uncertainty_head(decoded),  # (B, 1, H, W)
            "eri":         self.eri_head(decoded),          # (B, 5, H, W)
        }
        if routing_weights is not None:
            outputs["routing_weights"] = routing_weights    # (B, n_experts)

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
    Verify the forward pass runs end-to-end with correct output shapes,
    in both eval mode (no routing_weights) and train mode (with routing_weights).

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
        "physics":    torch.randn(B, cfg.T, cfg.C_physics,   cfg.H, cfg.W),   # 6ch
        "wind":       torch.randn(B, cfg.T, cfg.C_wind,      cfg.H, cfg.W),   # 4ch
        "static":     torch.randn(B, cfg.C_static,           cfg.H, cfg.W),
        "discharge":  torch.randn(B, cfg.T, cfg.C_discharge, cfg.H, cfg.W),   # 2ch new
        "bgc_aux":    torch.randn(B, cfg.T, cfg.C_bgc,       cfg.H, cfg.W),   # 5ch new
    }

    expected_shapes = {
        "recon":       (B, 1,                cfg.H, cfg.W),
        "forecast":    (B, cfg.H_fcast,      cfg.H, cfg.W),
        "uncertainty": (B, 1,                cfg.H, cfg.W),
        "eri":         (B, cfg.n_eri_levels, cfg.H, cfg.W),
    }

    all_ok = True

    # --- Eval mode ---
    model.eval()
    with torch.no_grad():
        out_eval = model(fake_batch)

    print("\n--- Output shapes (eval) ---")
    for key, tensor in out_eval.items():
        exp    = expected_shapes.get(key)
        status = "OK" if exp is None or tuple(tensor.shape) == exp else f"MISMATCH — expected {exp}"
        print(f"  {key:<18} {str(tuple(tensor.shape)):<30} {status}")
        if "MISMATCH" in status:
            all_ok = False

    assert "routing_weights" not in out_eval, "routing_weights should be absent in eval mode"
    print(f"  {'routing_weights':<18} {'absent in eval mode':<30} OK")

    # --- Train mode ---
    model.train()
    out_train = model(fake_batch)

    print("\n--- Output shapes (train) ---")
    for key, tensor in out_train.items():
        exp    = expected_shapes.get(key)
        status = "OK" if exp is None or tuple(tensor.shape) == exp else f"MISMATCH — expected {exp}"
        print(f"  {key:<18} {str(tuple(tensor.shape)):<30} {status}")
        if "MISMATCH" in status:
            all_ok = False

    assert "routing_weights" in out_train, "routing_weights should be present in train mode"

    aux = compute_aux_loss(out_train["routing_weights"])
    print(f"\n  aux_loss: {aux.item():.4f}  (1.0 = uniform, {cfg.n_experts}.0 = collapsed)")

    print("\n--- Parameter counts ---")
    for name, n in model.param_count().items():
        print(f"  {name:<22} {n:>10,}")

    if all_ok:
        print("\nSmoke test passed.")
    else:
        raise RuntimeError("Shape mismatches detected — see above.")


if __name__ == "__main__":
    run_smoke_test()