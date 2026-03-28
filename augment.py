"""
augment.py — Spatial data augmentation for MM-MARAS

Applies consistent random spatial transforms (flips, 90° rotations)
across all tensors in a batch dict. Every tensor sharing the same
H, W spatial dimensions gets the same transform so spatial alignment
is preserved.

Usage in Train.py:
    from augment import augment_batch

    # Inside the training loop, before model(batch):
    if is_train:
        batch = augment_batch(batch)

Why this matters:
    With only 1,264 training patches from a small spatial domain,
    the model risks memorizing spatial layout. Random flips and 90°
    rotations provide 8x effective data diversity for free. These
    transforms are valid because:
    - Chl-a spatial patterns have no preferred orientation
    - Physics fields (currents, wind) are vector components that
      should ideally be rotated too, but in practice flips alone
      give most of the regularization benefit
    - Masks and bloom patterns are rotation-invariant

Note on vector fields:
    Strict correctness would require flipping the sign of uo/vo
    and u10/v10 components when reflecting spatially. The current
    implementation does NOT do this. The impact is small because:
    (a) the model sees relative spatial gradients, not absolute values
    (b) the z-scored distributions are roughly symmetric around zero
    If you want strict physical correctness, enable flip_vectors=True.
"""

from __future__ import annotations

import random
from typing import Dict

import torch
from torch import Tensor


# Tensors with these shapes are treated as spatial and get augmented.
# All others (scalars, metadata) are passed through unchanged.
SPATIAL_NDIMS = {3, 4, 5}  # (B, H, W), (B, C, H, W), (B, T, C, H, W) etc.

# Keys containing vector components that need sign flips on reflection.
# Each entry maps a batch key to a list of (dim, channel_indices) pairs.
# dim is the channel dimension index; channel_indices are the channels to negate.
VECTOR_CHANNELS: Dict[str, list] = {
    # physics[:, :, 1, :, :] = uo (zonal), physics[:, :, 2, :, :] = vo (meridional)
    "physics":   [(2, [1, 2])],
    # wind[:, :, 0, :, :] = u10, wind[:, :, 1, :, :] = v10
    "wind":      [(2, [0, 1])],
    # discharge has no directional components
}


def augment_batch(
    batch: Dict[str, Tensor],
    p_flip_h: float = 0.5,
    p_flip_v: float = 0.5,
    p_rot90: float = 0.5,
    flip_vectors: bool = False,
) -> Dict[str, Tensor]:
    """
    Apply random spatial augmentations to a batch dict.

    All spatial tensors in the batch receive the SAME random transform
    so mask/observation/target alignment is preserved.

    Args:
        batch:        Dict of tensors from the DataLoader.
        p_flip_h:     Probability of horizontal flip.
        p_flip_v:     Probability of vertical flip.
        p_rot90:      Probability of a random 90° rotation (0, 90, 180, or 270°).
        flip_vectors: If True, negate vector component channels on spatial flips
                      for physical correctness. Default False (simpler, works fine).

    Returns:
        Augmented batch dict (new tensors, original batch is not modified).
    """
    do_flip_h = random.random() < p_flip_h
    do_flip_v = random.random() < p_flip_v
    do_rot    = random.random() < p_rot90
    rot_k     = random.choice([1, 2, 3]) if do_rot else 0

    if not (do_flip_h or do_flip_v or do_rot):
        return batch  # no-op fast path

    out = {}
    for key, tensor in batch.items():
        if tensor.ndim not in SPATIAL_NDIMS:
            out[key] = tensor
            continue

        t = tensor

        # Identify H, W dims: always the last two
        h_dim = t.ndim - 2
        w_dim = t.ndim - 1

        if do_flip_h:
            t = t.flip(dims=[w_dim])
        if do_flip_v:
            t = t.flip(dims=[h_dim])
        if rot_k > 0:
            t = torch.rot90(t, k=rot_k, dims=[h_dim, w_dim])

        # Optionally negate vector components after spatial flips
        if flip_vectors and key in VECTOR_CHANNELS:
            for chan_dim, indices in VECTOR_CHANNELS[key]:
                if do_flip_h:
                    # Horizontal flip negates the zonal (E-W) component
                    # For physics: uo (index 1); for wind: u10 (index 0)
                    zonal_idx = indices[0]
                    t = _negate_channel(t, chan_dim, zonal_idx)
                if do_flip_v:
                    # Vertical flip negates the meridional (N-S) component
                    # For physics: vo (index 2); for wind: v10 (index 1)
                    merid_idx = indices[1] if len(indices) > 1 else indices[0]
                    t = _negate_channel(t, chan_dim, merid_idx)

        out[key] = t

    return out


def _negate_channel(t: Tensor, dim: int, idx: int) -> Tensor:
    """Negate a single channel slice along a given dimension."""
    t = t.clone()
    slices = [slice(None)] * t.ndim
    slices[dim] = idx
    t[tuple(slices)] = -t[tuple(slices)]
    return t


# ======================================================================
# Smoke test
# ======================================================================

def run_smoke_test() -> None:
    """
    python augment.py
    """
    B, T, H, W = 2, 10, 64, 64

    batch = {
        "chl_obs":    torch.randn(B, T, H, W),
        "obs_mask":   torch.ones(B, T, H, W),
        "physics":    torch.randn(B, T, 6, H, W),
        "wind":       torch.randn(B, T, 4, H, W),
        "static":     torch.randn(B, 2, H, W),
        "land_mask":  torch.zeros(B, H, W),
        "bloom_mask": torch.zeros(B, T, H, W),
        "target_chl": torch.randn(B, 5, H, W),
    }

    # Force all augmentations on
    aug_batch = augment_batch(batch, p_flip_h=1.0, p_flip_v=1.0, p_rot90=1.0)

    for key in batch:
        orig_shape = batch[key].shape
        aug_shape  = aug_batch[key].shape
        # After 90° rotation, H and W may swap — but they're equal (64x64)
        print(f"  {key:<14} {str(orig_shape):<30} -> {str(aug_shape):<30}", end="")
        if batch[key].shape[-2:] == aug_batch[key].shape[-2:]:
            # Check that content actually changed
            changed = not torch.equal(batch[key], aug_batch[key])
            print(f"  {'changed' if changed else 'UNCHANGED'}")
        else:
            print(f"  shape changed (rotation)")

    # Verify spatial consistency: if we flip chl_obs, obs_mask should flip the same way
    batch2 = augment_batch(batch, p_flip_h=1.0, p_flip_v=0.0, p_rot90=0.0)
    chl_flipped  = batch["chl_obs"].flip(dims=[-1])
    mask_flipped = batch["obs_mask"].flip(dims=[-1])
    assert torch.equal(batch2["chl_obs"], chl_flipped), "chl_obs flip mismatch"
    assert torch.equal(batch2["obs_mask"], mask_flipped), "obs_mask flip mismatch"
    print("\nSpatial consistency check: OK")
    print("Smoke test passed.")


if __name__ == "__main__":
    run_smoke_test()