"""
dataset.py — MM-MARAS patch loader

Reads .npz patch files from data/patches/{train,val,test}/
and returns typed tensors for all model inputs and targets.

Expected .npz keys and shapes (per patch):

    Core Chl-a
    ----------
    chl_obs     (T=10, H=64, W=64)       log-Chl-a, NaN where missing
    obs_mask    (T=10, H=64, W=64)       1 = valid pixel
    mcar_mask   (T=10, H=64, W=64)       1 = MCAR missing
    mnar_mask   (T=10, H=64, W=64)       1 = MNAR missing

    Physics (CMEMS 0.083°)
    ----------------------
    physics     (T=10, C=6, H=64, W=64)  thetao, uo, vo, mlotst, zos, so

    Atmospheric + precipitation forcing (ERA5 0.25°)
    ------------------------------------------------
    wind        (T=10, C=4, H=64, W=64)  u10, v10, msl, tp
                                          ch0-1: ERA5 daily_mean wind
                                          ch2  : ERA5 daily_mean sea-level pressure
                                          ch3  : ERA5 daily_sum  precipitation

    Freshwater / land-surface forcing (GloFAS 0.05°)
    -------------------------------------------------
    discharge   (T=10, C=2, H=64, W=64)  dis24, rowe
                                          ch0: river discharge (m³/s), log1p-norm
                                          ch1: runoff water equivalent (m/day), log1p-norm
                                          Note: swvl removed — not in consolidated GloFAS product

    BGC auxiliary state (CMEMS BGC 0.25°)
    ---------------------------------------
    bgc_aux     (T=10, C=5, H=64, W=64)  o2, no3, po4, si, nppv

    Static context
    --------------
    static      (C=2, H=64, W=64)        bathymetry, distance-to-coast

    Labels / targets
    ----------------
    bloom_mask  (T=10, H=64, W=64)       bloom event labels (ERI supervision)
    target_chl  (H_fcast=5, H=64, W=64)  future Chl-a (forecast target)

Derived tensors added at load time:
    land_mask   (H=64, W=64)             1 = land pixel (from static NaNs)
    target_mask (H_fcast=5, H=64, W=64)  1 = valid observable ocean pixel

Land pixels are filled with 0.0 in static, physics, and target_chl.
Use land_mask and target_mask to exclude them from loss computation:
    valid = target_mask * (1 - land_mask).unsqueeze(0)   # (5, H, W)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

log = logging.getLogger(__name__)

# -------------------------------------------------------------------
# Constants
# -------------------------------------------------------------------

SPLIT = Literal["train", "val", "test"]

EXPECTED_SHAPES: dict[str, tuple] = {
    "chl_obs":    (10, 64, 64),
    "obs_mask":   (10, 64, 64),
    "mcar_mask":  (10, 64, 64),
    "mnar_mask":  (10, 64, 64),
    "physics":    (10,  6, 64, 64),   # thetao, uo, vo, mlotst, zos, so
    "wind":       (10,  4, 64, 64),   # u10, v10, msl, tp
    "discharge":  (10,  2, 64, 64),   # dis24, rowe
    "bgc_aux":    (10,  5, 64, 64),   # o2, no3, po4, si, nppv
    "static":     ( 2, 64, 64),
    "bloom_mask": (10, 64, 64),
    "target_chl": ( 5, 64, 64),
}

REQUIRED_KEYS = set(EXPECTED_SHAPES.keys())

# Keys that may contain NaN for expected reasons (land or cloud masking)
# — filled with 0.0; corresponding mask tensors track valid pixels.
SILENT_NAN_KEYS = {"static", "physics", "discharge", "bgc_aux", "target_chl"}


# -------------------------------------------------------------------
# Dataset
# -------------------------------------------------------------------

class MARASSDataset(Dataset):
    """
    PyTorch Dataset for MM-MARAS .npz patch files.

    Args:
        patch_dir : Root of the patches directory (data/patches/).
        split     : One of "train", "val", "test".
        validate  : If True, shape-check every patch on first access.
        nan_fill  : Fill value for missing Chl-a pixels in chl_obs.
                    The obs_mask already encodes which pixels were originally
                    missing so the model can learn to ignore this value.
    """

    def __init__(
        self,
        patch_dir: str | Path,
        split: SPLIT,
        validate: bool = True,
        nan_fill: float = 0.0,
    ) -> None:
        self.split    = split
        self.validate = validate
        self.nan_fill = nan_fill

        split_dir = Path(patch_dir) / split
        if not split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {split_dir}")

        self.files = sorted(split_dir.glob("*.npz"))
        if not self.files:
            raise RuntimeError(f"No .npz files found in {split_dir}")

        log.info(f"[{split}] Loaded {len(self.files)} patches from {split_dir}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        path = self.files[idx]

        try:
            data = np.load(path, allow_pickle=False)
        except Exception as exc:
            raise RuntimeError(f"Failed to load patch: {path}") from exc

        missing = REQUIRED_KEYS - set(data.keys())
        if missing:
            raise KeyError(f"Patch {path.name} is missing keys: {missing}")

        sample: dict[str, Tensor] = {}

        # --- Derive land_mask from static NaNs (before any filling) ----------
        static_raw    = data["static"].astype(np.float32)
        land_mask_np  = np.isnan(static_raw).any(axis=0)          # (H, W)
        sample["land_mask"] = torch.from_numpy(land_mask_np.astype(np.float32))

        # --- Derive target_mask from target_chl NaNs -------------------------
        target_raw       = data["target_chl"].astype(np.float32)  # (5, H, W)
        target_mask_np   = (~np.isnan(target_raw)).astype(np.float32)
        sample["target_mask"] = torch.from_numpy(target_mask_np)

        # --- Load all required tensors ---------------------------------------
        for key in REQUIRED_KEYS:
            arr    = data[key].astype(np.float32)

            if self.validate:
                _check_shape(key, arr, path.name)

            tensor = torch.from_numpy(arr)

            if key == "chl_obs":
                # Expected NaNs (cloud/glint) — obs_mask encodes their location
                tensor = torch.nan_to_num(tensor, nan=self.nan_fill)

            elif key in SILENT_NAN_KEYS:
                # Expected NaNs (land for physics/discharge/bgc_aux/static,
                # or cloud/land for target_chl).
                # land_mask and target_mask exclude these pixels from losses.
                tensor = torch.nan_to_num(tensor, nan=0.0)

            elif torch.isnan(tensor).any():
                nan_count = torch.isnan(tensor).sum().item()
                log.warning(
                    f"[{path.name}] Unexpected NaNs in '{key}': {nan_count} pixels"
                )

            sample[key] = tensor

        return sample

    def __repr__(self) -> str:
        return (
            f"MARASSDataset(split={self.split!r}, "
            f"n_patches={len(self.files)}, "
            f"validate={self.validate})"
        )


# -------------------------------------------------------------------
# Shape validator
# -------------------------------------------------------------------

def _check_shape(key: str, arr: np.ndarray, fname: str) -> None:
    expected = EXPECTED_SHAPES[key]
    if arr.shape != expected:
        raise ValueError(
            f"[{fname}] Shape mismatch for '{key}': "
            f"expected {expected}, got {arr.shape}"
        )


# -------------------------------------------------------------------
# DataLoader factory
# -------------------------------------------------------------------

def build_dataloaders(
    patch_dir: str | Path,
    batch_size: int = 8,
    num_workers: int = 2,
    pin_memory: bool = True,
    validate: bool = True,
    nan_fill: float = 0.0,
) -> dict[SPLIT, DataLoader]:
    """
    Build train / val / test DataLoaders.

    Args:
        patch_dir   : Root patches directory (data/patches/).
        batch_size  : Samples per batch.
        num_workers : Parallel workers (0 for Windows debugging).
        pin_memory  : Speeds up CPU → GPU transfers (disable on CPU-only).
        validate    : Shape-check patches on load.
        nan_fill    : Fill value for missing Chl-a pixels.

    Returns:
        Dict with keys "train", "val", "test".
    """
    patch_dir = Path(patch_dir)
    loaders: dict[str, DataLoader] = {}

    for split in ("train", "val", "test"):
        ds = MARASSDataset(
            patch_dir = patch_dir,
            split     = split,      # type: ignore[arg-type]
            validate  = validate,
            nan_fill  = nan_fill,
        )
        loaders[split] = DataLoader(
            ds,
            batch_size  = batch_size,
            shuffle     = (split == "train"),
            num_workers = num_workers,
            pin_memory  = pin_memory,
            drop_last   = (split == "train"),
        )
        log.info(
            f"[{split}] DataLoader: {len(ds)} samples, "
            f"batch_size={batch_size}, "
            f"{len(loaders[split])} batches"
        )

    return loaders


# -------------------------------------------------------------------
# Sanity check
# -------------------------------------------------------------------

def run_sanity_check(patch_dir: str | Path, batch_size: int = 4) -> None:
    """
    Quick smoke test. Run from the repo root:

        python dataset.py
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    loaders = build_dataloaders(
        patch_dir   = patch_dir,
        batch_size  = batch_size,
        num_workers = 0,
        pin_memory  = False,
        validate    = True,
    )

    for split, loader in loaders.items():
        batch = next(iter(loader))
        print(f"\n--- {split.upper()} ---")
        for key, tensor in batch.items():
            nan_count = torch.isnan(tensor).sum().item()
            nan_note  = f"  *** {nan_count} NaNs ***" if nan_count else ""
            extra = ""
            if key == "land_mask":
                extra = f"  ({tensor.mean().item()*100:.1f}% land)"
            elif key == "target_mask":
                extra = f"  ({tensor.mean().item()*100:.1f}% supervisable)"
            print(
                f"  {key:<14} {str(tuple(tensor.shape)):<32} "
                f"dtype={tensor.dtype}{nan_note}{extra}"
            )

    print("\nSanity check passed.")


if __name__ == "__main__":
    run_sanity_check(patch_dir="/kaggle/input/datasets/rajvardhandesai27/down-the-sea/patches", batch_size=4)