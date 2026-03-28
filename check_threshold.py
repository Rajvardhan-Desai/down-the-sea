"""
Quick diagnostic: check the actual distribution of target_chl values
in the test patches to find the right bloom threshold.

Run on Kaggle:
    python /kaggle/working/maras/check_threshold.py
"""
import numpy as np
from pathlib import Path

patch_dir = Path("/kaggle/input/datasets/rajvardhandesai27/down-the-sea/patches/test")
files = sorted(patch_dir.glob("*.npz"))

all_target = []
all_chl = []
for f in files:
    d = np.load(f)
    all_target.append(d["target_chl"].flatten())
    all_chl.append(d["chl_obs"].flatten())

target = np.concatenate(all_target)
chl = np.concatenate(all_chl)

# Remove NaN-filled zeros (land/missing pixels)
target_valid = target[np.isfinite(target) & (target != 0.0)]
chl_valid = chl[np.isfinite(chl) & (chl != 0.0)]

print(f"target_chl — valid pixels: {len(target_valid):,}")
print(f"  min:  {target_valid.min():.4f}")
print(f"  max:  {target_valid.max():.4f}")
print(f"  mean: {target_valid.mean():.4f}")
print(f"  std:  {target_valid.std():.4f}")
print(f"  p90:  {np.percentile(target_valid, 90):.4f}")
print(f"  p95:  {np.percentile(target_valid, 95):.4f}")
print(f"  p99:  {np.percentile(target_valid, 99):.4f}")
print(f"  p99.5:{np.percentile(target_valid, 99.5):.4f}")
print(f"  p99.9:{np.percentile(target_valid, 99.9):.4f}")

print(f"\nchl_obs — valid pixels: {len(chl_valid):,}")
print(f"  min:  {chl_valid.min():.4f}")
print(f"  max:  {chl_valid.max():.4f}")
print(f"  p99:  {np.percentile(chl_valid, 99):.4f}")
print(f"  p99.5:{np.percentile(chl_valid, 99.5):.4f}")
print(f"  p99.9:{np.percentile(chl_valid, 99.9):.4f}")

# Test different thresholds
print("\nBloom pixel counts at different thresholds:")
for t in [1.0, 1.5, 2.0, 2.5, 3.0, 5.0, 10.0, 10.85]:
    n = (target_valid > t).sum()
    pct = 100 * n / len(target_valid)
    print(f"  threshold={t:>5.2f}  →  {n:>8,} pixels ({pct:.4f}%)")