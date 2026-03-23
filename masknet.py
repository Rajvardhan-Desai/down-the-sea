"""
masknet.py — MaskNet: structured missingness encoder

Encodes the four mask channels (obs, mcar, mnar, bloom) into a spatial
embedding that captures:
  1. Per-pixel missingness type (MCAR vs MNAR vs bloom vs valid)
  2. Spatial structure of missing regions via graph propagation

Architecture:
    masks (B, T, 4, H, W)
        │
        ├── MissTypeEmbedder   pixel-wise learned embeddings per missingness type
        │                      output: (B, T, D, H, W)
        │
        ├── SpatialGNN         message passing across pixel grid
        │   (GraphConv ×2)     propagates valid-pixel context into gap interiors
        │                      output: (B, T, D, H, W)
        │
        └── TemporalMixer      lightweight depthwise conv across T
                               shares missingness context across time steps
                               output: (B, T, D, H, W)

GNN design:
    - Grid graph: each pixel connected to its 4 cardinal neighbors
    - Edge weights: 1 if both endpoints are valid (obs_mask=1), 0 otherwise
      so information only flows through valid pixels into gap boundaries
    - Two message-passing rounds are enough to reach ~2 pixels into gaps
    - Implemented as depthwise separable convolutions with masked aggregation
      (equivalent to graph conv on a regular grid, no sparse tensor overhead)

Missingness type classification per pixel per timestep:
    valid  : obs_mask == 1
    mcar   : obs_mask == 0 AND mcar_mask == 1
    mnar   : obs_mask == 0 AND mnar_mask == 1
    bloom  : bloom_mask == 1  (may overlap with valid/missing)
    unknown: obs_mask == 0 AND mcar_mask == 0 AND mnar_mask == 0
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ======================================================================
# Missingness type embedder
# ======================================================================

class MissTypeEmbedder(nn.Module):
    """
    Assigns a learned embedding to each pixel based on its missingness type.

    Types (5 total):
        0 = valid          obs_mask == 1
        1 = MCAR           obs_mask == 0, mcar_mask == 1
        2 = MNAR           obs_mask == 0, mnar_mask == 1
        3 = bloom          bloom_mask == 1
        4 = unknown        obs_mask == 0, no type assigned

    Bloom pixels get their own embedding because bloom events are a
    distinct physical regime, not just a data-quality issue.

    A pixel can be both bloom AND missing — in that case bloom takes priority.

    Input:  masks  (B*T, 4, H, W)   obs, mcar, mnar, bloom
    Output: emb    (B*T, D, H, W)
    """

    N_TYPES = 5

    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        # Learned type embeddings: (N_TYPES, D)
        self.type_emb = nn.Embedding(self.N_TYPES, embed_dim)

        # Refine per-pixel after type assignment
        self.refine = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=1),
            nn.GELU(),
        )

    def _classify(self, masks: Tensor) -> Tensor:
        """
        Classify each pixel into one of 5 missingness types.

        Input:  masks  (N, 4, H, W)   obs, mcar, mnar, bloom  (float, 0/1)
        Output: types  (N, H, W)      long, values 0–4
        """
        obs, mcar, mnar, bloom = masks[:, 0], masks[:, 1], masks[:, 2], masks[:, 3]

        # Start: everything is unknown (4)
        types = torch.full(obs.shape, 4, dtype=torch.long, device=masks.device)

        # Apply in reverse priority order so high-priority overwrites low
        types[obs == 1]   = 0   # valid
        types[mcar == 1]  = 1   # MCAR missing
        types[mnar == 1]  = 2   # MNAR missing
        types[bloom == 1] = 3   # bloom (highest priority — physical regime)

        return types  # (N, H, W)

    def forward(self, masks: Tensor) -> Tensor:
        N, _, H, W = masks.shape  # N = B*T

        types = self._classify(masks)                   # (N, H, W)
        emb = self.type_emb(types)                      # (N, H, W, D)
        emb = emb.permute(0, 3, 1, 2).contiguous()     # (N, D, H, W)
        return self.refine(emb)


# ======================================================================
# Spatial GNN (grid graph conv)
# ======================================================================

class GridGraphConv(nn.Module):
    """
    One round of message passing on a regular pixel grid.

    Each pixel aggregates features from its 4 cardinal neighbors,
    weighted by whether those neighbors are valid (obs_mask == 1).
    This lets valid-pixel context flow into gap interiors while
    preventing spurious flow between two missing pixels.

    This is equivalent to a masked graph convolution on the grid graph,
    implemented as convolutions (no sparse tensor overhead).

    Input:
        x        (N, D, H, W)   node features
        obs_mask (N, 1, H, W)   1 = valid pixel (float)
    Output:
        (N, D, H, W)
    """

    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        neighbor_kernel = torch.zeros(embed_dim, 1, 3, 3)
        neighbor_kernel[:, 0, 0, 1] = 1.0  # top
        neighbor_kernel[:, 0, 1, 0] = 1.0  # left
        neighbor_kernel[:, 0, 1, 2] = 1.0  # right
        neighbor_kernel[:, 0, 2, 1] = 1.0  # bottom
        self.register_buffer("neighbor_kernel", neighbor_kernel, persistent=False)
        self.register_buffer("count_kernel", neighbor_kernel[:1].clone(), persistent=False)

        # Update rule: combine self + aggregated neighbors
        self.update = nn.Sequential(
            nn.Conv2d(embed_dim * 2, embed_dim, kernel_size=1),
            nn.GroupNorm(8, embed_dim),
            nn.GELU(),
        )

    def forward(self, x: Tensor, obs_mask: Tensor) -> Tensor:
        # Weight neighbor features by their validity before aggregating.
        # Missing pixels contribute 0 so information flows valid → gap only.
        x_valid = x * obs_mask                              # (N, D, H, W)

        x_valid_padded = F.pad(x_valid, (1, 1, 1, 1), mode="replicate")
        neighbor_kernel = self.neighbor_kernel
        if neighbor_kernel.dtype != x_valid.dtype:
            neighbor_kernel = neighbor_kernel.to(dtype=x_valid.dtype)
        neighbor_sum = F.conv2d(
            x_valid_padded,
            weight=neighbor_kernel,
            groups=neighbor_kernel.shape[0],
        )                                                   # (N, D, H, W)

        # Count valid neighbors per pixel for normalization
        obs_padded = F.pad(obs_mask, (1, 1, 1, 1), mode="replicate")
        count_kernel = self.count_kernel
        if count_kernel.dtype != obs_mask.dtype:
            count_kernel = count_kernel.to(dtype=obs_mask.dtype)
        valid_count = F.conv2d(
            obs_padded,
            weight=count_kernel,
        ).clamp(min=1.0)                                    # (N, 1, H, W)

        neighbor_agg = neighbor_sum / valid_count           # (N, D, H, W)

        # Update: combine self-feature with aggregated neighbor context
        return self.update(torch.cat([x, neighbor_agg], dim=1))

class SpatialGNN(nn.Module):
    """
    Two rounds of masked graph convolution on the pixel grid.

    Two rounds are sufficient to propagate valid-pixel context ~2 pixels
    into the interior of cloud gaps, which covers most gap boundaries.

    Input:
        x        (N, D, H, W)
        obs_mask (N, 1, H, W)
    Output:
        (N, D, H, W)
    """

    def __init__(self, embed_dim: int, n_rounds: int = 2) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            GridGraphConv(embed_dim) for _ in range(n_rounds)
        ])

    def forward(self, x: Tensor, obs_mask: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x, obs_mask)
        return x


# ======================================================================
# Temporal mixer
# ======================================================================

class TemporalMixer(nn.Module):
    """
    Share missingness context across the T time steps with a lightweight
    depthwise Conv1d along the time axis.

    Treats each (D, H, W) spatial location independently and mixes its
    T-length feature sequence with a small temporal kernel.

    Input:  (B, T, D, H, W)
    Output: (B, T, D, H, W)
    """

    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        # Depthwise Conv1d along T: each of the D channels gets its own kernel.
        # This is cheap and keeps spatial positions independent.
        self.t_conv = nn.Conv1d(
            in_channels=embed_dim,
            out_channels=embed_dim,
            kernel_size=3,
            padding=1,
            groups=embed_dim,   # depthwise
            bias=False,
        )
        self.norm = nn.GroupNorm(8, embed_dim)

    def forward(self, x: Tensor) -> Tensor:
        B, T, D, H, W = x.shape

        # Merge spatial dims into batch so Conv1d sees (N, D, T)
        # N = B * H * W, sequence length = T, channels = D
        x_r = x.permute(0, 3, 4, 1, 2)            # (B, H, W, T, D)
        x_r = x_r.reshape(B * H * W, T, D)         # (B*H*W, T, D)
        x_r = x_r.permute(0, 2, 1)                 # (B*H*W, D, T)  ← Conv1d format

        x_r = self.t_conv(x_r)                     # (B*H*W, D, T)

        x_r = x_r.permute(0, 2, 1)                 # (B*H*W, T, D)
        x_r = x_r.reshape(B, H, W, T, D)
        x_r = x_r.permute(0, 3, 4, 1, 2)           # (B, T, D, H, W)

        # GroupNorm on the D dimension
        x_r = self.norm(x_r.reshape(B * T, D, H, W)).view(B, T, D, H, W)

        return x + x_r  # residual


# ======================================================================
# MaskNet (top-level)
# ======================================================================

class MaskNet(nn.Module):
    """
    Full MaskNet: structured missingness encoder.

    Replaces MaskNetStub in model.py.

    Input:  masks  (B, T, 4, H, W)   obs, mcar, mnar, bloom (float 0/1)
    Output: emb    (B, T, D, H, W)

    To swap into model.py:
        self.masknet = MaskNet(cfg.embed_dim, cfg.T)
    """

    def __init__(self, embed_dim: int, T: int, gnn_rounds: int = 2) -> None:
        super().__init__()
        self.embed_dim = embed_dim

        self.type_embedder = MissTypeEmbedder(embed_dim)
        self.spatial_gnn   = SpatialGNN(embed_dim, n_rounds=gnn_rounds)
        self.temporal_mix  = TemporalMixer(embed_dim)

    def forward(self, masks: Tensor) -> Tensor:
        B, T, C, H, W = masks.shape
        assert C == 4, f"Expected 4 mask channels, got {C}"

        # Flatten B×T for per-frame processing
        masks_flat = masks.view(B * T, C, H, W)

        # 1. Per-pixel missingness type embedding
        emb = self.type_embedder(masks_flat)            # (B*T, D, H, W)

        # 2. Spatial propagation (GNN)
        obs_mask_flat = masks_flat[:, 0:1]              # (B*T, 1, H, W)
        emb = self.spatial_gnn(emb, obs_mask_flat)      # (B*T, D, H, W)

        # 3. Reshape back, then mix across time
        emb = emb.view(B, T, self.embed_dim, H, W)
        emb = self.temporal_mix(emb)                    # (B, T, D, H, W)

        return emb


# ======================================================================
# Smoke test
# ======================================================================

def run_smoke_test() -> None:
    """
    python masknet.py
    """
    import logging
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    B, T, H, W, D = 2, 10, 64, 64, 256

    # Simulate realistic masks: ~30% missing, all structured
    obs = (torch.rand(B, T, H, W) > 0.30).float()
    mcar  = torch.zeros(B, T, H, W)   # MCAR: 0% as per Bay of Bengal stats
    mnar  = torch.zeros(B, T, H, W)   # MNAR: 0%
    bloom = (torch.rand(B, T, H, W) > 0.95).float()  # ~5% bloom pixels
    masks = torch.stack([obs, mcar, mnar, bloom], dim=2)   # (B, T, 4, H, W)

    net = MaskNet(embed_dim=D, T=T)
    net.eval()

    with torch.no_grad():
        out = net(masks)

    expected = (B, T, D, H, W)
    status = "OK" if tuple(out.shape) == expected else f"MISMATCH — expected {expected}"
    print(f"Output shape: {tuple(out.shape)}  {status}")

    n_params = sum(p.numel() for p in net.parameters())
    print(f"Parameters:   {n_params:,}")

    nan_count = torch.isnan(out).sum().item()
    print(f"NaNs in output: {nan_count}")

    if tuple(out.shape) == expected and nan_count == 0:
        print("\nSmoke test passed.")
    else:
        raise RuntimeError("Smoke test failed — see above.")


if __name__ == "__main__":
    run_smoke_test()
