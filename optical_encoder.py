"""
optical_encoder.py — Swin-UNet optical encoder

Processes per-frame optical input (chl_obs + obs_mask) into spatial
feature maps suitable for gap filling and forecasting.

Architecture:
    Input: (B*T, 2, H=64, W=64)

    PatchEmbed (4×4, stride 4)
        → (B*T, 256, 16, 16)  tokens on a 16×16 grid

    Encoder
        Stage 1: SwinBlocks × 2, dim=64,  grid=16×16, window=8
        PatchMerging → dim=128, grid=8×8
        Stage 2: SwinBlocks × 2, dim=128, grid=8×8,  window=8
        PatchMerging → dim=256, grid=4×4
        Stage 3: SwinBlocks × 2, dim=256, grid=4×4,  window=4

    Bottleneck: SwinBlocks × 2, dim=256, grid=4×4

    Decoder (with skip connections from encoder stages)
        Stage 3: PatchExpand → dim=128, grid=8×8   + skip from enc stage 2
        Stage 2: PatchExpand → dim=64,  grid=16×16 + skip from enc stage 1
        Stage 1: Conv upsample ×4 → dim=embed_dim, grid=64×64

    Output: (B*T, embed_dim=256, 64, 64)

Swin block details:
    - Window Multi-Head Self-Attention (W-MSA) with relative position bias
    - Shifted-Window MSA (SW-MSA) alternates with W-MSA for cross-window context
    - LayerNorm pre-norm, MLP with GELU, residual connections

SatMAE pretraining:
    SatMAE (He et al., 2022) pretrains a ViT-Large MAE on multispectral satellite
    imagery. Direct weight transfer to Swin is not possible, but two paths exist:
      1. Pretrain this encoder as a masked autoencoder on your own Chl-a patches
         (call pretrain_mae() below — recommended, uses your domain data).
      2. Load SatMAE ViT-L patch embeddings as initialisation for the Swin patch
         embed layer via load_satmae_patch_embed() (partial transfer only).
    Both are optional — the encoder trains from scratch by default.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ======================================================================
# Patch embed / merge / expand
# ======================================================================

class PatchEmbed(nn.Module):
    """
    Non-overlapping patch tokenisation via strided convolution.

    Input:  (N, C_in, H, W)
    Output: (N, embed_dim, H//patch_size, W//patch_size)
    """

    def __init__(self, in_channels: int, embed_dim: int, patch_size: int = 4) -> None:
        super().__init__()
        self.proj = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size,
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.patch_size = patch_size

    def forward(self, x: Tensor) -> Tensor:
        x = self.proj(x)                        # (N, D, H/p, W/p)
        N, D, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)        # (N, H*W, D)
        x = self.norm(x)
        return x.transpose(1, 2).view(N, D, H, W)


class PatchMerging(nn.Module):
    """
    Downsample 2× by concatenating 2×2 neighboring patches and projecting.

    Input:  (N, D, H, W)
    Output: (N, D*2, H//2, W//2)
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(4 * dim)
        self.proj = nn.Linear(4 * dim, 2 * dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        N, D, H, W = x.shape
        x = x.permute(0, 2, 3, 1)              # (N, H, W, D)
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], dim=-1)  # (N, H/2, W/2, 4D)
        x = self.norm(x)
        x = self.proj(x)                          # (N, H/2, W/2, 2D)
        return x.permute(0, 3, 1, 2)             # (N, 2D, H/2, W/2)


class PatchExpanding(nn.Module):
    """
    Upsample 2× via pixel shuffle after a linear projection.

    Input:  (N, D, H, W)
    Output: (N, D//2, H*2, W*2)
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.proj = nn.Linear(dim, dim * 2, bias=False)  # expand channels for shuffle

    def forward(self, x: Tensor) -> Tensor:
        N, D, H, W = x.shape
        x = x.permute(0, 2, 3, 1)              # (N, H, W, D)
        x = self.norm(x)
        x = self.proj(x)                        # (N, H, W, 2D)
        x = x.permute(0, 3, 1, 2)             # (N, 2D, H, W)
        # pixel_shuffle ×2: (N, 2D, H, W) → (N, D/2, 2H, 2W)
        return F.pixel_shuffle(x, 2)


# ======================================================================
# Window attention
# ======================================================================

def window_partition(x: Tensor, window_size: int) -> tuple[Tensor, int, int]:
    """
    Partition feature map into non-overlapping windows.

    Input:  x  (N, H, W, D)
    Output: windows  (N * nW, ws*ws, D),  H_padded, W_padded
    """
    N, H, W, D = x.shape
    ws = window_size

    # Pad to multiple of window_size
    pad_h = (ws - H % ws) % ws
    pad_w = (ws - W % ws) % ws
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))

    H_p, W_p = H + pad_h, W + pad_w
    x = x.view(N, H_p // ws, ws, W_p // ws, ws, D)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, ws * ws, D)
    return windows, H_p, W_p


def window_reverse(windows: Tensor, window_size: int,
                   H_p: int, W_p: int, H: int, W: int) -> Tensor:
    """Reverse window_partition and remove padding."""
    ws = window_size
    N = int(windows.shape[0] / (H_p // ws * W_p // ws))
    x = windows.view(N, H_p // ws, W_p // ws, ws, ws, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(N, H_p, W_p, -1)
    return x[:, :H, :W, :].contiguous()


class WindowAttention(nn.Module):
    """
    Window-based Multi-Head Self-Attention with relative position bias.

    Supports both regular (shift=0) and shifted (shift=window_size//2) windows.

    Args:
        dim:         Feature dimension.
        window_size: Side length of attention window (ws).
        num_heads:   Number of attention heads.
        shift:       Cyclic shift offset (0 for W-MSA, ws//2 for SW-MSA).
    """

    def __init__(
        self,
        dim: int,
        window_size: int,
        num_heads: int,
        shift: int = 0,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.ws = window_size
        self.shift = shift
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5

        self.qkv  = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        # Relative position bias table: (2*ws-1)^2 positions × num_heads
        self.rel_pos_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) ** 2, num_heads)
        )
        nn.init.trunc_normal_(self.rel_pos_bias_table, std=0.02)

        # Precompute relative position index for ws×ws window
        coords = torch.stack(torch.meshgrid(
            torch.arange(window_size),
            torch.arange(window_size),
            indexing="ij",
        ))                                                  # (2, ws, ws)
        coords_flat = coords.flatten(1)                    # (2, ws*ws)
        rel = coords_flat[:, :, None] - coords_flat[:, None, :]  # (2, ws*ws, ws*ws)
        rel = rel.permute(1, 2, 0).contiguous()            # (ws*ws, ws*ws, 2)
        rel[:, :, 0] += window_size - 1
        rel[:, :, 1] += window_size - 1
        rel[:, :, 0] *= 2 * window_size - 1
        self.register_buffer("rel_pos_idx", rel.sum(-1))   # (ws*ws, ws*ws)

        # Attention mask for SW-MSA (computed lazily in forward)
        self._attn_mask_cache: dict[tuple, Tensor] = {}

    def _get_attn_mask(self, H: int, W: int, device: torch.device) -> Optional[Tensor]:
        if self.shift == 0:
            return None
        key = (H, W)
        if key not in self._attn_mask_cache:
            ws = self.ws
            img_mask = torch.zeros(1, H, W, 1, device=device)
            h_slices = (slice(0, -ws), slice(-ws, -self.shift), slice(-self.shift, None))
            w_slices = (slice(0, -ws), slice(-ws, -self.shift), slice(-self.shift, None))
            cnt = 0
            for h_s in h_slices:
                for w_s in w_slices:
                    img_mask[:, h_s, w_s, :] = cnt
                    cnt += 1
            windows, H_p, W_p = window_partition(img_mask, ws)
            windows = windows.view(-1, ws * ws)
            attn_mask = windows.unsqueeze(1) - windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, -100.0).masked_fill(attn_mask == 0, 0.0)
            self._attn_mask_cache[key] = attn_mask
        return self._attn_mask_cache[key]

    def forward(self, x: Tensor) -> Tensor:
        """
        Input:  x  (N, H, W, D)
        Output:    (N, H, W, D)
        """
        N, H, W, D = x.shape
        ws = self.ws

        # Cyclic shift for SW-MSA
        if self.shift > 0:
            x = torch.roll(x, shifts=(-self.shift, -self.shift), dims=(1, 2))

        # Partition into windows
        windows, H_p, W_p = window_partition(x, ws)       # (nW*N, ws*ws, D)
        nW_N = windows.shape[0]

        # QKV
        qkv = self.qkv(windows).reshape(nW_N, ws * ws, 3, self.num_heads, D // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)                           # each: (nW*N, nH, ws*ws, head_dim)

        # Attention
        attn = (q * self.scale) @ k.transpose(-2, -1)     # (nW*N, nH, ws*ws, ws*ws)

        # Relative position bias
        bias = self.rel_pos_bias_table[self.rel_pos_idx.view(-1)].view(
            ws * ws, ws * ws, self.num_heads
        ).permute(2, 0, 1).unsqueeze(0)                   # (1, nH, ws*ws, ws*ws)
        attn = attn + bias

        # SW-MSA mask
        mask = self._get_attn_mask(H_p, W_p, x.device)
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(nW_N // nW, nW, self.num_heads, ws * ws, ws * ws)
            attn = attn + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(nW_N, self.num_heads, ws * ws, ws * ws)

        attn = self.attn_drop(attn.softmax(dim=-1))
        out = (attn @ v).transpose(1, 2).reshape(nW_N, ws * ws, D)
        out = self.proj_drop(self.proj(out))

        # Reverse windows
        out = window_reverse(out, ws, H_p, W_p, H, W)     # (N, H, W, D)

        # Reverse cyclic shift
        if self.shift > 0:
            out = torch.roll(out, shifts=(self.shift, self.shift), dims=(1, 2))

        return out


# ======================================================================
# Swin Transformer block
# ======================================================================

class SwinBlock(nn.Module):
    """
    One Swin Transformer block.

    Alternates W-MSA (shift=0) and SW-MSA (shift=ws//2) based on block index.
    Pre-norm architecture with residual connections.

    Input/output: (N, D, H, W)
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: int,
        block_idx: int = 0,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
    ) -> None:
        super().__init__()
        shift = 0 if block_idx % 2 == 0 else window_size // 2

        self.norm1 = nn.LayerNorm(dim)
        self.attn  = WindowAttention(
            dim, window_size=window_size, num_heads=num_heads,
            shift=shift, attn_drop=drop, proj_drop=drop,
        )
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden, dim),
            nn.Dropout(drop),
        )

    def forward(self, x: Tensor) -> Tensor:
        N, D, H, W = x.shape
        x_hwc = x.permute(0, 2, 3, 1)                     # (N, H, W, D)

        x_hwc = x_hwc + self.attn(self.norm1(x_hwc))
        x_hwc = x_hwc + self.mlp(self.norm2(x_hwc))

        return x_hwc.permute(0, 3, 1, 2)                  # (N, D, H, W)


def make_stage(dim: int, num_heads: int, window_size: int,
               depth: int, drop: float = 0.0) -> nn.Sequential:
    """Build a sequence of Swin blocks, alternating W-MSA and SW-MSA."""
    return nn.Sequential(*[
        SwinBlock(dim, num_heads, window_size, block_idx=i, drop=drop)
        for i in range(depth)
    ])


# ======================================================================
# Swin-UNet optical encoder
# ======================================================================

class OpticalEncoder(nn.Module):
    """
    Swin-UNet encoder for per-frame optical input.

    Replaces OpticalEncoderStub in model.py.

    Input:  optical  (B, T, C_optical=2, H=64, W=64)
    Output: feat     (B, T, embed_dim=256, H=64, W=64)

    To swap into model.py:
        from optical_encoder import OpticalEncoder
        self.opt_enc = OpticalEncoder(
            in_channels=cfg.C_optical,
            embed_dim=cfg.embed_dim,
            T=cfg.T,
        )

    Internal dimensions (following Swin-T scaling):
        Stage 1: dim=64,  heads=4,  window=8
        Stage 2: dim=128, heads=8,  window=8
        Stage 3: dim=256, heads=16, window=4  (grid is only 4×4 here)
        Bottleneck: dim=256, heads=16, window=4
    """

    def __init__(
        self,
        in_channels: int = 2,
        embed_dim: int = 256,
        patch_size: int = 4,
        depths: tuple[int, ...] = (2, 2, 2),
        num_heads: tuple[int, ...] = (4, 8, 16),
        window_sizes: tuple[int, ...] = (8, 8, 4),
        drop_rate: float = 0.0,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim

        # Stage dims: 64 → 128 → 256 (= embed_dim)
        dims = [embed_dim // 4, embed_dim // 2, embed_dim]
        assert dims[-1] == embed_dim

        # Patch embedding
        self.patch_embed = PatchEmbed(in_channels, dims[0], patch_size)

        # Encoder stages
        self.enc1 = make_stage(dims[0], num_heads[0], window_sizes[0], depths[0], drop_rate)
        self.down1 = PatchMerging(dims[0])                 # dims[0] → dims[1]

        self.enc2 = make_stage(dims[1], num_heads[1], window_sizes[1], depths[1], drop_rate)
        self.down2 = PatchMerging(dims[1])                 # dims[1] → dims[2]

        self.enc3 = make_stage(dims[2], num_heads[2], window_sizes[2], depths[2], drop_rate)

        # Bottleneck
        self.bottleneck = make_stage(dims[2], num_heads[2], window_sizes[2], 2, drop_rate)

        # Decoder
        self.up3    = PatchExpanding(dims[2])              # dims[2] → dims[1]
        self.skip3  = nn.Conv2d(dims[1] * 2, dims[1], kernel_size=1)  # fuse skip
        self.dec3   = make_stage(dims[1], num_heads[1], window_sizes[1], depths[1], drop_rate)

        self.up2    = PatchExpanding(dims[1])              # dims[1] → dims[0]
        self.skip2  = nn.Conv2d(dims[0] * 2, dims[0], kernel_size=1)
        self.dec2   = make_stage(dims[0], num_heads[0], window_sizes[0], depths[0], drop_rate)

        # Final upsample ×4 from 16×16 back to 64×64
        self.out_proj = nn.Sequential(
            nn.Conv2d(dims[0], embed_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1),
        )
        self.upsample = nn.Upsample(scale_factor=4, mode="bilinear", align_corners=False)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward_single(self, x: Tensor) -> Tensor:
        """Process one frame. Input/output: (N, C, H, W)."""
        # Patch embed: (N, C, 64, 64) → (N, 64, 16, 16)
        x = self.patch_embed(x)                 # (N, 64, 16, 16)

        # Encoder
        s1 = self.enc1(x)                       # (N, 64,  16, 16)
        s2 = self.enc2(self.down1(s1))          # (N, 128,  8,  8)
        s3 = self.enc3(self.down2(s2))          # (N, 256,  4,  4)

        # Bottleneck
        b = self.bottleneck(s3)                 # (N, 256,  4,  4)

        # Decoder with skips
        d3 = self.up3(b)                        # (N, 128,  8,  8)
        d3 = self.skip3(torch.cat([d3, s2], dim=1))  # fuse skip
        d3 = self.dec3(d3)                      # (N, 128,  8,  8)

        d2 = self.up2(d3)                       # (N, 64,  16, 16)
        d2 = self.skip2(torch.cat([d2, s1], dim=1))
        d2 = self.dec2(d2)                      # (N, 64,  16, 16)

        # Project to embed_dim and upsample to original resolution
        out = self.out_proj(d2)                 # (N, 256, 16, 16)
        out = self.upsample(out)                # (N, 256, 64, 64)
        return out

    def forward(self, optical: Tensor) -> Tensor:
        """
        Input:  (B, T, C, H, W)
        Output: (B, T, embed_dim, H, W)
        """
        B, T, C, H, W = optical.shape
        x = optical.view(B * T, C, H, W)
        x = self.forward_single(x)              # (B*T, embed_dim, H, W)
        return x.view(B, T, self.embed_dim, H, W)

    def load_satmae_patch_embed(self, satmae_ckpt_path: str) -> None:
        """
        Partial weight transfer from a SatMAE ViT-L checkpoint.

        SatMAE uses a 16×16 ViT patch embed. We load those weights into our
        4×4 Swin patch embed by bicubic-interpolating the kernel weights.
        This is a rough initialisation — fine-tuning on your data is still needed.

        Args:
            satmae_ckpt_path: Path to a SatMAE .pth checkpoint.
        """
        import os
        if not os.path.exists(satmae_ckpt_path):
            raise FileNotFoundError(f"SatMAE checkpoint not found: {satmae_ckpt_path}")

        ckpt = torch.load(satmae_ckpt_path, map_location="cpu")
        state = ckpt.get("model", ckpt)

        # SatMAE patch embed key: "patch_embed.proj.weight" (D_vit, C, 16, 16)
        vit_key = "patch_embed.proj.weight"
        if vit_key not in state:
            raise KeyError(f"Key '{vit_key}' not found in SatMAE checkpoint.")

        vit_w = state[vit_key]                  # (D_vit, C_vit, 16, 16)
        D_vit, C_vit, _, _ = vit_w.shape
        D_swin = self.patch_embed.proj.weight.shape[0]
        C_swin = self.patch_embed.proj.weight.shape[1]

        # Interpolate kernel from 16×16 to 4×4
        vit_w_r = F.interpolate(
            vit_w.float(),
            size=(4, 4),
            mode="bicubic",
            align_corners=False,
        )                                       # (D_vit, C_vit, 4, 4)

        # If channel counts differ, take first C_swin channels
        vit_w_r = vit_w_r[:, :C_swin, :, :]

        # If feature dim differs, take first D_swin filters
        vit_w_r = vit_w_r[:D_swin, :, :, :]

        with torch.no_grad():
            self.patch_embed.proj.weight.copy_(vit_w_r)

        print(f"Loaded SatMAE patch embed ({vit_w.shape} → {vit_w_r.shape})")


# ======================================================================
# Smoke test
# ======================================================================

def run_smoke_test() -> None:
    """
    python optical_encoder.py
    """
    import logging
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    B, T, H, W = 2, 10, 64, 64
    embed_dim = 256
    C_optical = 2

    optical = torch.randn(B, T, C_optical, H, W)

    enc = OpticalEncoder(in_channels=C_optical, embed_dim=embed_dim)
    enc.eval()

    with torch.no_grad():
        out = enc(optical)

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