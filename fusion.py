"""
fusion.py — Perceiver IO cross-modal fusion

Fuses optical, physics, mask, BGC, and discharge embeddings into a single
feature map via cross-attention between learned latent vectors and the five
input streams.

Architecture per (B*T) frame:
    Input streams (each D=256, H×W=4096 tokens after flattening):
        opt_feat   (N, D, H, W)   optical (chl_obs + obs_mask)
        phy_feat   (N, D, H, W)   physics + wind forcing
        mask_emb   (N, D, H, W)   missingness
        bgc_feat   (N, D, H, W)   BGC auxiliaries (o2, no3, po4, si, nppv)
        dis_feat   (N, D, H, W)   discharge + runoff (dis24, rowe)

    Step 1 — Contrastive pre-alignment (optional, used during pretraining):
        Aligns optical and physics feature spaces via NT-Xent loss so that
        corresponding spatial positions are close before fusion begins.
        Not used in the forward pass during supervised training.

    Step 2 — Cross-attention (Perceiver IO style):
        n_latents learned query vectors attend to all five streams
        (concatenated as keys/values along the token dimension).
        Latents: (n_latents, D)
        Keys/Values: (5 * P * P, D)  — all five streams pooled to P×P
        Output: (n_latents, D)

    Step 3 — Spatial broadcast:
        Latent output is projected back to (H*W, D) via a linear layer,
        then reshaped to (N, D, H, W).

    Output: (N, D, H, W) → reshaped to (B, T, D, H, W)

Why Perceiver IO here?
    - Full self-attention over 5 × H × W = 20,480 tokens is O(N²) — too costly.
    - Simple concatenation + 1×1 conv has no cross-modal interaction.
    - n_latents << H×W, so cross-attention is O(n_latents × 5HW) — linear in tokens.
    - Latents learn to extract the most task-relevant cross-modal patterns.
    - Going from 3 to 5 streams only scales KV tokens from 3×P² to 5×P²
      (768 → 1280 for P=16), which stays well within GPU memory limits.

Contrastive pre-alignment:
    Before fusion, optical and physics features should live in aligned spaces
    so the latents can meaningfully compare them. We use NT-Xent (SimCLR-style)
    treating spatially-averaged features from the same (B, T) location as
    positives and other batch elements as negatives.
    Call compute_contrastive_loss() during the pretraining phase, then disable.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ======================================================================
# Multi-head cross-attention
# ======================================================================

class CrossAttention(nn.Module):
    """
    Multi-head cross-attention: queries attend to keys and values.

    Args:
        query_dim:  Dimension of query vectors.
        kv_dim:     Dimension of key/value vectors (may differ from query_dim).
        num_heads:  Number of attention heads.
        out_dim:    Output projection dimension.
    """

    def __init__(
        self,
        query_dim: int,
        kv_dim: int,
        num_heads: int = 8,
        out_dim: int | None = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        out_dim = out_dim or query_dim
        self.num_heads = num_heads
        self.head_dim  = query_dim // num_heads
        self.scale     = self.head_dim ** -0.5

        self.q_proj  = nn.Linear(query_dim, query_dim, bias=False)
        self.k_proj  = nn.Linear(kv_dim,    query_dim, bias=False)
        self.v_proj  = nn.Linear(kv_dim,    query_dim, bias=False)
        self.out_proj = nn.Linear(query_dim, out_dim)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, query: Tensor, kv: Tensor) -> Tensor:
        """
        Args:
            query: (N, Lq, query_dim)
            kv:    (N, Lkv, kv_dim)
        Returns:
            (N, Lq, out_dim)
        """
        N, Lq, _ = query.shape
        Lkv = kv.shape[1]

        q = self.q_proj(query).view(N, Lq,  self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(kv).view(   N, Lkv, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(kv).view(   N, Lkv, self.num_heads, self.head_dim).transpose(1, 2)

        attn = (q * self.scale) @ k.transpose(-2, -1)     # (N, nH, Lq, Lkv)
        attn = self.attn_drop(attn.softmax(dim=-1))

        out = (attn @ v).transpose(1, 2).reshape(N, Lq, -1)  # (N, Lq, query_dim)
        return self.proj_drop(self.out_proj(out))


# ======================================================================
# Perceiver IO fusion block
# ======================================================================

class PerceiverFusionBlock(nn.Module):
    """
    One Perceiver IO block:
        1. Cross-attend latents → spatially-pooled input tokens
        2. Self-attend latents (refine)
        3. Decode latents → full-resolution spatial tokens

    Args:
        embed_dim:    Feature dimension D.
        n_latents:    Number of learned query latents (default 64).
        num_heads:    Attention heads for both cross- and self-attention.
        H, W:         Full spatial dimensions of the feature map (e.g. 64, 64).
        kv_pool_size: Spatial size to pool KV tokens to before cross-attention.
                      Default 16 → 5 × 16 × 16 = 1280 KV tokens (up from 768
                      with 3 streams). Still well within GPU memory limits.
                      The latents still capture full cross-modal content;
                      fine spatial detail is preserved by the skip-blend in
                      FusionModule.forward().
        n_streams:    Number of input feature streams (default 5).
    """

    def __init__(
        self,
        embed_dim: int,
        n_latents: int = 64,
        num_heads: int = 8,
        H: int = 64,
        W: int = 64,
        kv_pool_size: int = 16,
        n_streams: int = 5,
        drop_rate: float = 0.1,
    ) -> None:
        super().__init__()
        self.embed_dim    = embed_dim
        self.n_latents    = n_latents
        self.H            = H
        self.W            = W
        self.kv_pool_size = kv_pool_size
        self.n_streams    = n_streams

        # Spatial average pool: reduces KV sequence from H*W to kv_pool_size^2
        self.kv_pool = nn.AdaptiveAvgPool2d(kv_pool_size)

        # Learned latent queries: (n_latents, D)
        self.latents = nn.Parameter(torch.randn(n_latents, embed_dim) * 0.02)

        # Cross-attention: latents attend to pooled input streams
        self.cross_attn = CrossAttention(
            query_dim=embed_dim,
            kv_dim=embed_dim,
            num_heads=num_heads,
            attn_drop=drop_rate,
            proj_drop=drop_rate,
        )
        self.cross_norm_q  = nn.LayerNorm(embed_dim)
        self.cross_norm_kv = nn.LayerNorm(embed_dim)

        # Self-attention on latents: refine cross-modal representation
        self.self_attn = nn.MultiheadAttention(
            embed_dim, num_heads=num_heads, dropout=drop_rate, batch_first=True
        )
        self.self_norm  = nn.LayerNorm(embed_dim)

        # MLP after self-attention
        self.mlp = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(drop_rate),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(drop_rate),
        )

        # Decode with explicit spatial queries so each output location carries
        # its own positional signal before attending back to the latent set.
        self.row_embed = nn.Parameter(torch.randn(H, embed_dim) * 0.02)
        self.col_embed = nn.Parameter(torch.randn(W, embed_dim) * 0.02)
        self.decode_attn = CrossAttention(
            query_dim=embed_dim,
            kv_dim=embed_dim,
            num_heads=num_heads,
            attn_drop=drop_rate,
            proj_drop=drop_rate,
        )
        self.decode_norm_q = nn.LayerNorm(embed_dim)
        self.decode_norm_kv = nn.LayerNorm(embed_dim)
        self.decode_norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        opt_feat: Tensor,
        phy_feat: Tensor,
        mask_emb: Tensor,
        bgc_feat: Tensor,
        dis_feat: Tensor,
    ) -> Tensor:
        """
        Args:
            opt_feat:  (N, D, H, W)   optical
            phy_feat:  (N, D, H, W)   physics + wind
            mask_emb:  (N, D, H, W)   missingness
            bgc_feat:  (N, D, H, W)   BGC auxiliaries
            dis_feat:  (N, D, H, W)   discharge + runoff
        Returns:
            fused:     (N, D, H, W)
        """
        N, D, H, W = opt_feat.shape
        assert H == self.H and W == self.W

        # Pool each stream to kv_pool_size × kv_pool_size before flattening.
        # 5 streams × P² tokens (e.g. 5 × 256 = 1280 for P=16) stays tractable.
        def pool_flat(t: Tensor) -> Tensor:
            return self.kv_pool(t).flatten(2).transpose(1, 2)  # (N, P*P, D)

        opt_tok  = pool_flat(opt_feat)                      # (N, P*P, D)
        phy_tok  = pool_flat(phy_feat)
        mask_tok = pool_flat(mask_emb)
        bgc_tok  = pool_flat(bgc_feat)
        dis_tok  = pool_flat(dis_feat)

        # Concatenate all five streams as key-value tokens: (N, 5*P*P, D)
        kv_tokens = torch.cat([opt_tok, phy_tok, mask_tok, bgc_tok, dis_tok], dim=1)

        # Expand latents to batch: (N, n_latents, D)
        latents = self.latents.unsqueeze(0).expand(N, -1, -1)

        # Cross-attention: latents attend to pooled input tokens
        latents = latents + self.cross_attn(
            self.cross_norm_q(latents),
            self.cross_norm_kv(kv_tokens),
        )                                                   # (N, n_latents, D)

        # Self-attention on latents
        lat_norm = self.self_norm(latents)
        lat_sa, _ = self.self_attn(lat_norm, lat_norm, lat_norm)
        latents = latents + lat_sa                         # (N, n_latents, D)

        # MLP
        latents = latents + self.mlp(latents)              # (N, n_latents, D)

        # Decode latents → full-resolution spatial tokens via learned
        # position-aware spatial queries instead of a flat transpose projection.
        spatial_queries = (
            self.row_embed[:, None, :] + self.col_embed[None, :, :]
        ).view(1, H * W, D).expand(N, -1, -1)
        spatial = spatial_queries + self.decode_attn(
            self.decode_norm_q(spatial_queries),
            self.decode_norm_kv(latents),
        )                                                  # (N, H*W, D)
        spatial = self.decode_norm(spatial)
        spatial = spatial.transpose(1, 2).view(N, D, H, W)

        return spatial


# ======================================================================
# Contrastive pre-alignment loss
# ======================================================================

def compute_contrastive_loss(
    opt_feat: Tensor,
    phy_feat: Tensor,
    temperature: float = 0.07,
) -> Tensor:
    """
    NT-Xent (SimCLR) contrastive loss between spatially-averaged optical
    and physics features.

    Use during a contrastive pretraining phase to align the two feature
    spaces before supervised training begins. Not called in the main
    forward pass.

    Args:
        opt_feat:    (B*T, D, H, W)
        phy_feat:    (B*T, D, H, W)
        temperature: Softmax temperature (lower = sharper, default 0.07).

    Returns:
        Scalar loss.
    """
    # Global average pool to get one vector per sample
    z_opt = opt_feat.mean(dim=(2, 3))                       # (N, D)
    z_phy = phy_feat.mean(dim=(2, 3))                       # (N, D)

    # L2 normalize
    z_opt = F.normalize(z_opt, dim=-1)
    z_phy = F.normalize(z_phy, dim=-1)

    N = z_opt.shape[0]

    # Similarity matrix: (N, N) — rows are optical, cols are physics
    sim = torch.mm(z_opt, z_phy.T) / temperature           # (N, N)

    # Positives on the diagonal (same sample), all others are negatives
    labels = torch.arange(N, device=sim.device)
    loss = (F.cross_entropy(sim, labels) + F.cross_entropy(sim.T, labels)) / 2
    return loss


# ======================================================================
# Top-level fusion module
# ======================================================================

class FusionModule(nn.Module):
    """
    Cross-modal fusion via Perceiver IO — five input streams.

    Replaces FusionModuleStub in model.py.

    Streams:
        opt_feat:  optical (chl_obs + obs_mask)
        phy_feat:  physics + wind forcing
        mask_emb:  missingness
        bgc_feat:  BGC auxiliaries (o2, no3, po4, si, nppv)
        dis_feat:  discharge + runoff (dis24, rowe)

    Args:
        embed_dim:  Feature dimension (default 256).
        n_latents:  Perceiver latent count (default 64).
        num_heads:  Attention heads (default 8).
        H, W:       Spatial size of input feature maps (default 64, 64).

    To swap into model.py:
        from fusion import FusionModule
        self.fusion = FusionModule(
            embed_dim=cfg.embed_dim,
            H=cfg.H,
            W=cfg.W,
        )
    """

    def __init__(
        self,
        embed_dim: int = 256,
        n_latents: int = 64,
        num_heads: int = 8,
        H: int = 64,
        W: int = 64,
        drop_rate: float = 0.1,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim

        self.block = PerceiverFusionBlock(
            embed_dim=embed_dim,
            n_latents=n_latents,
            num_heads=num_heads,
            H=H,
            W=W,
            n_streams=5,
            drop_rate=drop_rate,
        )

        # Residual blend: fused output + mean of all five inputs
        self.blend = nn.Conv2d(embed_dim * 2, embed_dim, kernel_size=1)

    def forward(
        self,
        opt_feat: Tensor,
        phy_feat: Tensor,
        mask_emb: Tensor,
        bgc_feat: Tensor,
        dis_feat: Tensor,
    ) -> Tensor:
        """
        Args:
            opt_feat:  (B, T, D, H, W)   optical
            phy_feat:  (B, T, D, H, W)   physics + wind
            mask_emb:  (B, T, D, H, W)   missingness
            bgc_feat:  (B, T, D, H, W)   BGC auxiliaries
            dis_feat:  (B, T, D, H, W)   discharge + runoff
        Returns:
            fused:     (B, T, D, H, W)
        """
        B, T, D, H, W = opt_feat.shape

        # Flatten B×T for per-frame processing
        opt_f  = opt_feat.view(B * T, D, H, W)
        phy_f  = phy_feat.view(B * T, D, H, W)
        mask_f = mask_emb.view(B * T, D, H, W)
        bgc_f  = bgc_feat.view(B * T, D, H, W)
        dis_f  = dis_feat.view(B * T, D, H, W)

        # Perceiver IO cross-attention fusion across all five streams
        fused = self.block(opt_f, phy_f, mask_f, bgc_f, dis_f)   # (B*T, D, H, W)

        # Blend fused output with mean of all inputs (residual stabilises training)
        mean_in = (opt_f + phy_f + mask_f + bgc_f + dis_f) / 5.0
        fused = self.blend(torch.cat([fused, mean_in], dim=1))    # (B*T, D, H, W)

        return fused.view(B, T, D, H, W)


# ======================================================================
# Smoke test
# ======================================================================

def run_smoke_test() -> None:
    """
    python fusion.py
    """
    import logging
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    B, T, D, H, W = 2, 10, 256, 64, 64

    opt_feat  = torch.randn(B, T, D, H, W)
    phy_feat  = torch.randn(B, T, D, H, W)
    mask_emb  = torch.randn(B, T, D, H, W)
    bgc_feat  = torch.randn(B, T, D, H, W)
    dis_feat  = torch.randn(B, T, D, H, W)

    module = FusionModule(embed_dim=D, H=H, W=W)
    module.eval()

    with torch.no_grad():
        out = module(opt_feat, phy_feat, mask_emb, bgc_feat, dis_feat)

    expected = (B, T, D, H, W)
    status = "OK" if tuple(out.shape) == expected else f"MISMATCH — expected {expected}"
    print(f"Output shape:   {tuple(out.shape)}  {status}")

    n_params = sum(p.numel() for p in module.parameters())
    print(f"Parameters:     {n_params:,}")

    nan_count = torch.isnan(out).sum().item()
    print(f"NaNs in output: {nan_count}")

    # Contrastive loss check (still uses opt/phy pair only)
    opt_flat = opt_feat.view(B * T, D, H, W)
    phy_flat = phy_feat.view(B * T, D, H, W)
    loss = compute_contrastive_loss(opt_flat, phy_flat)
    print(f"Contrastive loss (sanity): {loss.item():.4f}  (expect ~log(B*T) ~= {math.log(B*T):.2f} at init)")

    if tuple(out.shape) == expected and nan_count == 0:
        print("\nSmoke test passed.")
    else:
        raise RuntimeError("Smoke test failed — see above.")


if __name__ == "__main__":
    import math
    run_smoke_test()
