"""
discharge_encoder.py — Discharge and runoff encoder

Processes river discharge and runoff variables into spatial feature maps
matching the other encoder outputs.

Input stream:
    discharge: (B, T, 2, H, W)   dis24 (river discharge), rowe (runoff)

Output: (B, T, embed_dim=256, H, W)

Architecture:
    Reuses the OpticalEncoder Swin-UNet backbone with in_channels=2.
    Separate weights from all other encoders.

Why a separate stream (Option B)?
    Discharge and runoff are spatially concentrated near river mouths and
    coastal margins (Ganges-Brahmaputra delta is the dominant signal in the
    Bay of Bengal). Their spatial structure is fundamentally different from
    both optical and physics features — most of the domain is zero or near-zero,
    with sharp gradients near river outflow points. Folding this into the wind
    stream would force the backbone to simultaneously represent uniform wind
    fields and sparse point-source freshwater inputs, which compete for the
    same convolutional filters.

    A dedicated backbone learns to amplify the near-zero/high-gradient pattern
    characteristic of discharge without interference from wind features.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from optical_encoder import OpticalEncoder


class DischargeEncoder(nn.Module):
    """
    Discharge + runoff encoder.

    New encoder stream introduced in Option B pipeline update.

    Args:
        C_discharge: Number of discharge channels (default 2: dis24, rowe).
        embed_dim:   Output feature dimension (default 256).

    Usage in model.py:
        from discharge_encoder import DischargeEncoder
        self.discharge_enc = DischargeEncoder(
            C_discharge=cfg.C_discharge,
            embed_dim=cfg.embed_dim,
        )
    """

    def __init__(self, C_discharge: int = 2, embed_dim: int = 256) -> None:
        super().__init__()
        self.C_discharge = C_discharge
        self.embed_dim   = embed_dim
        # Reuse Swin-UNet backbone — separate instance, independent weights
        self.backbone    = OpticalEncoder(in_channels=C_discharge, embed_dim=embed_dim)

    def forward(self, discharge: Tensor) -> Tensor:
        """
        Args:
            discharge: (B, T, C_discharge, H, W)
        Returns:
            feat:      (B, T, embed_dim, H, W)
        """
        return self.backbone(discharge)


# ======================================================================
# Smoke test
# ======================================================================

def run_smoke_test() -> None:
    """
    python discharge_encoder.py
    """
    import logging
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    B, T, H, W  = 2, 10, 64, 64
    embed_dim   = 256
    C_discharge = 2   # dis24, rowe

    discharge = torch.randn(B, T, C_discharge, H, W)

    enc = DischargeEncoder(C_discharge=C_discharge, embed_dim=embed_dim)
    enc.eval()

    with torch.no_grad():
        out = enc(discharge)

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
