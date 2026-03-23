"""
bgc_encoder.py — BGC auxiliary encoder

Processes biogeochemical auxiliary variables into spatial feature maps
matching the optical and physics encoder outputs.

Input stream:
    bgc_aux: (B, T, 5, H, W)   o2, no3, po4, si, nppv

Output: (B, T, embed_dim=256, H, W)

Architecture:
    Reuses the OpticalEncoder Swin-UNet backbone with in_channels=5.
    Separate weights from all other encoders — BGC tracers encode
    biogeochemical state that is physically distinct from both optical
    radiance and physical forcing. Cross-modal interaction with the
    other streams happens downstream in the Perceiver IO fusion module.

Why a separate stream (Option B) rather than folding into optical?
    BGC tracers (oxygen, nutrients, primary productivity) and optical
    Chl-a share biological context but have very different dynamic
    ranges, spatial patterns, and temporal autocorrelation. Keeping
    them in separate backbone instances lets each stream learn its own
    patch statistics before fusion, rather than forcing the optical
    backbone to simultaneously handle radiance and nutrient fields.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from optical_encoder import OpticalEncoder


class BGCAuxEncoder(nn.Module):
    """
    BGC auxiliary encoder.

    New encoder stream introduced in Option B pipeline update.

    Args:
        C_bgc:      Number of BGC channels (default 5: o2, no3, po4, si, nppv).
        embed_dim:  Output feature dimension (default 256).

    Usage in model.py:
        from bgc_encoder import BGCAuxEncoder
        self.bgc_enc = BGCAuxEncoder(
            C_bgc=cfg.C_bgc,
            embed_dim=cfg.embed_dim,
        )
    """

    def __init__(self, C_bgc: int = 5, embed_dim: int = 256) -> None:
        super().__init__()
        self.C_bgc     = C_bgc
        self.embed_dim = embed_dim
        # Reuse Swin-UNet backbone — separate instance, independent weights
        self.backbone  = OpticalEncoder(in_channels=C_bgc, embed_dim=embed_dim)

    def forward(self, bgc_aux: Tensor) -> Tensor:
        """
        Args:
            bgc_aux: (B, T, C_bgc, H, W)
        Returns:
            feat:    (B, T, embed_dim, H, W)
        """
        return self.backbone(bgc_aux)


# ======================================================================
# Smoke test
# ======================================================================

def run_smoke_test() -> None:
    """
    python bgc_encoder.py
    """
    import logging
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    B, T, H, W = 2, 10, 64, 64
    embed_dim   = 256
    C_bgc       = 5   # o2, no3, po4, si, nppv

    bgc_aux = torch.randn(B, T, C_bgc, H, W)

    enc = BGCAuxEncoder(C_bgc=C_bgc, embed_dim=embed_dim)
    enc.eval()

    with torch.no_grad():
        out = enc(bgc_aux)

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
