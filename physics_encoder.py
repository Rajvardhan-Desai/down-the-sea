"""
physics_encoder.py — Physics + forcing encoder

Processes ocean state variables, wind forcing, and static context
into spatial feature maps matching the optical encoder output.

Input streams (concatenated before encoding):
    physics:  (B, T, 6, H, W)   thetao, uo, vo, mlotst, zos, so
    wind:     (B, T, 4, H, W)   u10, v10, msl, tp
    static:   (B, 2, H, W)      bathymetry, distance-to-coast (no time dim)
    ──────────────────────────────────────────────
    total:    (B, T, 12, H, W)  after broadcasting static over T

Output: (B, T, embed_dim=256, H, W)

Pipeline changes vs previous version:
    physics gains one channel: so (sea water salinity) → C_physics 5→6
    wind variables change: tp+dis24 → msl+tp (same count, different vars)
    discharge (dis24, rowe) is now a dedicated separate stream → DischargeEncoder

Architecture:
    Reuses the OpticalEncoder Swin-UNet backbone with in_channels=12.
    No weight sharing with the optical encoder — the two streams encode
    fundamentally different physical quantities and should learn separate
    feature representations. The cross-modal fusion module downstream
    (Perceiver IO) is where their features are brought together.

    Static variables (bathymetry, distance-to-coast) are time-invariant
    but spatially informative — broadcasting them over T lets the encoder
    condition every time step on the underlying geography without any
    special handling.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from optical_encoder import OpticalEncoder


class PhysicsEncoder(nn.Module):
    """
    Physics + forcing encoder.

    Replaces PhysicsEncoderStub in model.py.

    Args:
        C_physics:  Number of physics channels (default 6: thetao, uo, vo, mlotst, zos, so).
        C_wind:     Number of wind channels (default 4: u10, v10, msl, tp).
        C_static:   Number of static channels (default 2).
        embed_dim:  Output feature dimension (default 256).

    To swap into model.py:
        from physics_encoder import PhysicsEncoder
        self.phy_enc = PhysicsEncoder(
            C_physics=cfg.C_physics,
            C_wind=cfg.C_wind,
            C_static=cfg.C_static,
            embed_dim=cfg.embed_dim,
        )
    """

    def __init__(
        self,
        C_physics: int = 6,
        C_wind: int = 4,
        C_static: int = 2,
        embed_dim: int = 256,
    ) -> None:
        super().__init__()
        self.C_physics = C_physics
        self.C_wind    = C_wind
        self.C_static  = C_static
        self.embed_dim = embed_dim

        C_in = C_physics + C_wind + C_static     # 9 by default

        # Reuse the Swin-UNet backbone — different instance, separate weights
        self.backbone = OpticalEncoder(
            in_channels=C_in,
            embed_dim=embed_dim,
        )

    def forward(self, physics: Tensor, wind: Tensor, static: Tensor) -> Tensor:
        """
        Args:
            physics: (B, T, C_physics, H, W)
            wind:    (B, T, C_wind,    H, W)
            static:  (B, C_static,     H, W)

        Returns:
            feat:    (B, T, embed_dim, H, W)
        """
        B, T, _, H, W = physics.shape

        # Broadcast static over T — geography is constant across time steps
        static_t = static.unsqueeze(1).expand(B, T, self.C_static, H, W)

        # Concatenate all streams along channel dim
        x = torch.cat([physics, wind, static_t], dim=2)   # (B, T, 9, H, W)

        return self.backbone(x)                            # (B, T, embed_dim, H, W)


# ======================================================================
# Smoke test
# ======================================================================

def run_smoke_test() -> None:
    """
    python physics_encoder.py
    """
    import logging
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    B, T, H, W = 2, 10, 64, 64
    embed_dim   = 256

    physics = torch.randn(B, T, 6, H, W)   # thetao, uo, vo, mlotst, zos, so
    wind    = torch.randn(B, T, 4, H, W)   # u10, v10, msl, tp
    static  = torch.randn(B, 2,    H, W)

    enc = PhysicsEncoder(embed_dim=embed_dim)
    enc.eval()

    with torch.no_grad():
        out = enc(physics, wind, static)

    expected = (B, T, embed_dim, H, W)
    status = "OK" if tuple(out.shape) == expected else f"MISMATCH — expected {expected}"
    print(f"Output shape: {tuple(out.shape)}  {status}")

    n_params = sum(p.numel() for p in enc.parameters())
    print(f"Parameters:   {n_params:,}")

    nan_count = torch.isnan(out).sum().item()
    print(f"NaNs in output: {nan_count}")

    if tuple(out.shape) == expected and nan_count == 0:
        print("\nSmoke test passed.")
    else:
        raise RuntimeError("Smoke test failed — see above.")


if __name__ == "__main__":
    run_smoke_test()