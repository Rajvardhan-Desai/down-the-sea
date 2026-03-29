"""
eval.py — MM-MARAS test set evaluation

Metrics:
    Reconstruction (three-way pixel split):
        all / valid / gap  — RMSE, MAE, bias, R², SSIM per subset
    CRPS                   — probabilistic calibration (uses uncertainty head)
    Forecast               — RMSE, MAE, SSIM per horizon step (1–5)
    ERI classification     — accuracy, macro-F1, per-class F1, ordinal MAE,
                             confusion matrix
    Uncertainty calibration — ECE, reliability diagram, var-err correlation
    MoE routing            — per-expert mean weights, entropy, utilisation

Outputs written to --out-dir:
    metrics.json          All scalar metrics
    confusion_matrix.csv  ERI confusion matrix (rows=true, cols=pred)
    calibration.csv       ECE reliability bin data
    figures/
        recon_rmse.png       RMSE bar: all / valid / gap
        calibration.png      Reliability diagram
        routing.png          MoE expert weight bar chart
        recon_NNNN.png       Sample gap-filling panels (n-figures samples)
        forecast_NNNN.png    Sample forecast panels

Usage:
    python eval.py --ckpt /kaggle/working/checkpoints/best.pt \\
                   --patch-dir /kaggle/input/.../patches \\
                   --out-dir /kaggle/working/eval_results
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

log = logging.getLogger(__name__)


# ======================================================================
# Args
# ======================================================================

def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate MM-MARAS on test set")
    p.add_argument("--ckpt",        required=True,          help="Path to checkpoint (.pt)")
    p.add_argument("--patch-dir",   required=True,          help="Root patches directory")
    p.add_argument("--out-dir",     default="eval_results",  help="Output directory")
    p.add_argument("--batch-size",  type=int, default=8)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--n-figures",   type=int, default=8,    help="Sample figures to save (0 = none)")
    p.add_argument("--no-figures",  action="store_true",    help="Skip all figure generation")
    p.add_argument("--no-amp",      action="store_true")
    p.add_argument("--device",      default=None)
    return p.parse_args()


# ======================================================================
# Standalone metric functions (used inside accumulators and per-sample)
# ======================================================================

def _masked_ssim(pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    """Simplified SSIM computed over valid (mask==1) pixels."""
    valid = mask.astype(bool)
    if valid.sum() < 2:
        return float("nan")
    p = pred[valid].astype(float)
    t = target[valid].astype(float)
    mu_p, mu_t = p.mean(), t.mean()
    var_p = ((p - mu_p) ** 2).mean()
    var_t = ((t - mu_t) ** 2).mean()
    cov   = ((p - mu_p) * (t - mu_t)).mean()
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    num = (2 * mu_p * mu_t + C1) * (2 * cov + C2)
    den = (mu_p ** 2 + mu_t ** 2 + C1) * (var_p + var_t + C2)
    return float(num / den) if den != 0 else float("nan")


def _compute_crps_batch(
    pred:    torch.Tensor,   # (B, H, W)
    log_var: torch.Tensor,   # (B, H, W)
    target:  torch.Tensor,   # (B, H, W)
    mask:    torch.Tensor,   # (B, H, W)  1 = supervise
) -> float:
    """
    CRPS for a Gaussian predictive distribution N(mu, sigma²).

    CRPS(N(mu,sigma), y) = sigma * (z*(2Phi(z)-1) + 2phi(z) - 1/sqrt(pi))
    where z = (y - mu) / sigma.

    Lower = better. A well-calibrated unbiased model → CRPS ≈ 0.
    """
    valid = mask.bool()
    if not valid.any():
        return float("nan")
    mu    = pred[valid].float()
    sigma = (log_var[valid].float() * 0.5).exp().clamp(min=1e-6)
    y     = target[valid].float()
    z     = (y - mu) / sigma
    phi   = torch.exp(-0.5 * z ** 2) / math.sqrt(2 * math.pi)
    Phi   = 0.5 * (1 + torch.erf(z / math.sqrt(2)))
    crps  = sigma * (z * (2 * Phi - 1) + 2 * phi - 1 / math.sqrt(math.pi))
    return crps.mean().item()


# ======================================================================
# Metric accumulators
# ======================================================================

class ReconAccumulator:
    """
    Accumulates pixel-level reconstruction errors in three pixel subsets:
        all   — all ocean pixels
        valid — observed ocean pixels (obs_mask == 1)
        gap   — missing ocean pixels (obs_mask == 0)

    Also accumulates per-sample SSIM for all and gap subsets,
    and CRPS using the uncertainty head log-variance.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self._px   = {s: {"se": [], "ae": [], "pred": [], "true": []}
                      for s in ("all", "valid", "gap")}
        self._ssim = {"all": [], "valid": [], "gap": []}
        self._crps = []

    def update(
        self,
        pred:      torch.Tensor,   # (B, 1, H, W)
        log_var:   torch.Tensor,   # (B, 1, H, W)
        target:    torch.Tensor,   # (B, H, W)   last chl_obs timestep
        obs_mask:  torch.Tensor,   # (B, H, W)   1 = observed
        land_mask: torch.Tensor,   # (B, H, W)   1 = land
    ) -> None:
        pred_sq   = pred.squeeze(1).float().cpu()
        lv_sq     = log_var.squeeze(1).float()           # keep on device for CRPS
        target_sq = target.float().cpu()
        obs_sq    = obs_mask.float().cpu()
        ocean     = (1.0 - land_mask.float()).cpu()

        se = (pred_sq - target_sq).pow(2)
        ae = (pred_sq - target_sq).abs()

        masks_np = {
            "all":   ocean.bool(),
            "valid": (obs_sq * ocean).bool(),
            "gap":   ((1 - obs_sq) * ocean).bool(),
        }

        # Pixel-level accumulators
        for name, m in masks_np.items():
            if m.any():
                store = self._px[name]
                store["se"].append(se[m].numpy())
                store["ae"].append(ae[m].numpy())
                store["pred"].append(pred_sq[m].numpy())
                store["true"].append(target_sq[m].numpy())

        # Per-sample SSIM (numpy)
        B = pred_sq.shape[0]
        pred_np   = pred_sq.numpy()
        target_np = target_sq.numpy()
        obs_np    = obs_sq.numpy()
        ocean_np  = ocean.numpy()
        for i in range(B):
            for name, m_fn in [
                ("all",   ocean_np[i]),
                ("valid", (obs_np[i] * ocean_np[i])),
                ("gap",   ((1 - obs_np[i]) * ocean_np[i])),
            ]:
                self._ssim[name].append(
                    _masked_ssim(pred_np[i], target_np[i], m_fn)
                )

        # CRPS (batch-level, on device)
        obs_device  = obs_mask.float()
        land_device = land_mask.float()
        valid_mask  = (obs_device * (1.0 - land_device)).bool()
        crps = _compute_crps_batch(
            pred.squeeze(1).float(), lv_sq, target.float(), valid_mask
        )
        if not math.isnan(crps):
            self._crps.append(crps)

    def compute(self) -> dict:
        results = {}
        for name, store in self._px.items():
            if not store["se"]:
                results[name] = {}
                continue
            se    = np.concatenate(store["se"])
            ae    = np.concatenate(store["ae"])
            pred  = np.concatenate(store["pred"])
            true_ = np.concatenate(store["true"])
            ss_res = se.sum()
            ss_tot = ((true_ - true_.mean()) ** 2).sum()
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
            ssim_vals = [x for x in self._ssim[name] if not math.isnan(x)]
            results[name] = {
                "rmse":  float(np.sqrt(se.mean())),
                "mae":   float(ae.mean()),
                "bias":  float((pred - true_).mean()),
                "r2":    float(r2),
                "ssim":  float(np.mean(ssim_vals)) if ssim_vals else float("nan"),
                "n_pix": int(len(se)),
            }
        crps_vals = [x for x in self._crps if not math.isnan(x)]
        results["crps"] = float(np.mean(crps_vals)) if crps_vals else float("nan")
        return results


class ForecastAccumulator:
    """Accumulates per-horizon RMSE, MAE, and SSIM."""

    def __init__(self, h_fcast: int):
        self.h_fcast = h_fcast
        self.se   = [[] for _ in range(h_fcast)]
        self.ae   = [[] for _ in range(h_fcast)]
        self.ssim = [[] for _ in range(h_fcast)]

    def update(
        self,
        pred:        torch.Tensor,   # (B, H_fcast, H, W)
        target:      torch.Tensor,   # (B, H_fcast, H, W)
        target_mask: torch.Tensor,   # (B, H_fcast, H, W)
        land_mask:   torch.Tensor,   # (B, H, W)
    ) -> None:
        ocean  = (1.0 - land_mask.float()).unsqueeze(1)   # keep on device
        valid  = (target_mask.float() * ocean).bool().cpu()
        pred_c = pred.float().cpu()
        tgt_c  = target.float().cpu()

        for h in range(self.h_fcast):
            m = valid[:, h]
            if m.any():
                diff = pred_c[:, h][m] - tgt_c[:, h][m]
                self.se[h].append(diff.pow(2).numpy())
                self.ae[h].append(diff.abs().numpy())
            # SSIM per sample per step
            B = pred_c.shape[0]
            for i in range(B):
                self.ssim[h].append(
                    _masked_ssim(
                        pred_c[i, h].numpy(),
                        tgt_c[i, h].numpy(),
                        valid[i, h].numpy(),
                    )
                )

    def compute(self) -> dict:
        results = {}
        for h in range(self.h_fcast):
            if not self.se[h]:
                results[f"step_{h+1}"] = {}
                continue
            se = np.concatenate(self.se[h])
            ae = np.concatenate(self.ae[h])
            ssim_vals = [x for x in self.ssim[h] if not math.isnan(x)]
            results[f"step_{h+1}"] = {
                "rmse": float(np.sqrt(se.mean())),
                "mae":  float(ae.mean()),
                "ssim": float(np.mean(ssim_vals)) if ssim_vals else float("nan"),
            }
        return results


class ERIAccumulator:
    """Accumulates ERI predictions for accuracy, per-class F1, and confusion matrix."""

    def __init__(self, n_levels: int = 5):
        self.n_levels = n_levels
        self.preds    = []
        self.labels   = []

    def update(
        self,
        logits:    torch.Tensor,   # (B, 5, H, W)
        target:    torch.Tensor,   # (B, H, W)  integer 0-4
        land_mask: torch.Tensor,   # (B, H, W)
    ) -> None:
        ocean    = (1.0 - land_mask.float()).bool().cpu()
        pred_cls = logits.argmax(dim=1).cpu()
        target_c = target.long().cpu()
        self.preds.append(pred_cls[ocean].numpy())
        self.labels.append(target_c[ocean].numpy())

    def compute(self) -> tuple[dict, np.ndarray]:
        preds  = np.concatenate(self.preds)
        labels = np.concatenate(self.labels)
        cm = np.zeros((self.n_levels, self.n_levels), dtype=np.int64)
        for p, l in zip(preds, labels):
            cm[int(l), int(p)] += 1
        acc = float((preds == labels).mean())
        f1s = []
        for c in range(self.n_levels):
            tp = cm[c, c]
            fp = cm[:, c].sum() - tp
            fn = cm[c, :].sum() - tp
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            f1s.append(f1)
        mae_ord = float(np.abs(preds.astype(float) - labels.astype(float)).mean())
        metrics = {
            "accuracy":     acc,
            "macro_f1":     float(np.mean(f1s)),
            "per_class_f1": {str(c): float(f1s[c]) for c in range(self.n_levels)},
            "mae_ordinal":  mae_ord,
        }
        return metrics, cm


class UncertaintyAccumulator:
    """ECE reliability diagram and variance-error correlation."""

    def __init__(self, n_bins: int = 10):
        self.n_bins   = n_bins
        self.log_vars = []
        self.sq_errs  = []

    def update(
        self,
        log_var:   torch.Tensor,   # (B, 1, H, W)
        pred:      torch.Tensor,   # (B, 1, H, W)
        target:    torch.Tensor,   # (B, H, W)
        obs_mask:  torch.Tensor,   # (B, H, W)
        land_mask: torch.Tensor,   # (B, H, W)
    ) -> None:
        ocean = (1.0 - land_mask.float())   # keep on device
        valid = (obs_mask.float() * ocean).bool().cpu()
        lv_sq   = log_var.squeeze(1).float().cpu()
        pred_sq = pred.squeeze(1).float().cpu()
        tgt_sq  = target.float().cpu()
        if valid.any():
            self.log_vars.append(lv_sq[valid].numpy())
            self.sq_errs.append((pred_sq[valid] - tgt_sq[valid]).pow(2).numpy())

    def compute(self) -> tuple[dict, list]:
        if not self.log_vars:
            return {}, []
        log_vars  = np.concatenate(self.log_vars)
        sq_errs   = np.concatenate(self.sq_errs)
        variances = np.exp(log_vars.clip(-10, 10))
        order  = np.argsort(variances)
        n      = len(order)
        bin_sz = n // self.n_bins
        bins, ece_terms = [], []
        for b in range(self.n_bins):
            idx = order[b * bin_sz: (b + 1) * bin_sz]
            mpv = float(variances[idx].mean())
            mse = float(sq_errs[idx].mean())
            bins.append({
                "bin":         b,
                "pred_std":    float(math.sqrt(max(mpv, 0))),
                "actual_rmse": float(math.sqrt(max(mse, 0))),
                "pred_var":    mpv,
                "actual_mse":  mse,
            })
            ece_terms.append(abs(mpv - mse))
        corr = float(np.corrcoef(variances, sq_errs)[0, 1])
        return {"ece": float(np.mean(ece_terms)), "var_err_corr": corr}, bins


class RoutingAccumulator:
    """Collects MoE routing weights."""

    def __init__(self, n_experts: int):
        self.n_experts = n_experts
        self.weights   = []

    def update(self, routing_weights: torch.Tensor) -> None:
        self.weights.append(routing_weights.float().cpu().numpy())

    def compute(self) -> dict:
        if not self.weights:
            return {}
        all_w = np.concatenate(self.weights, axis=0)
        mean_w = all_w.mean(axis=0)
        std_w  = all_w.std(axis=0)
        p = mean_w + 1e-8
        entropy = float(-(p * np.log(p)).sum())
        max_ent = float(math.log(self.n_experts))
        return {
            "mean_weight":  {f"expert_{e}": float(mean_w[e]) for e in range(self.n_experts)},
            "std_weight":   {f"expert_{e}": float(std_w[e])  for e in range(self.n_experts)},
            "entropy":      entropy,
            "max_entropy":  max_ent,
            "utilisation":  entropy / max_ent,
        }


# ======================================================================
# Per-sample figures
# ======================================================================

def _plt():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        return None


def save_recon_figure(
    chl_obs:   np.ndarray,   # (T, H, W)
    obs_mask:  np.ndarray,   # (T, H, W)
    pred:      np.ndarray,   # (H, W)
    land_mask: np.ndarray,   # (H, W)
    path: Path,
    idx:  int,
) -> None:
    plt = _plt()
    if plt is None:
        return
    last_obs = np.where(obs_mask[-1].astype(bool), chl_obs[-1], np.nan)
    diff     = np.where(obs_mask[-1].astype(bool), pred - chl_obs[-1], np.nan)
    pred_vis = np.where(land_mask.astype(bool), np.nan, pred)
    vmin = float(np.nanpercentile(chl_obs, 2))
    vmax = float(np.nanpercentile(chl_obs, 98))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].imshow(last_obs,  cmap="viridis", vmin=vmin, vmax=vmax, origin="lower")
    axes[0].set_title("Observed (last step)")
    axes[1].imshow(pred_vis,  cmap="viridis", vmin=vmin, vmax=vmax, origin="lower")
    axes[1].set_title("Reconstruction")
    im = axes[2].imshow(diff, cmap="RdBu_r",  vmin=-0.5, vmax=0.5,  origin="lower")
    axes[2].set_title("Difference (pred − obs)")
    plt.colorbar(im, ax=axes[2], shrink=0.8)
    for ax in axes:
        ax.axis("off")
    plt.suptitle(f"Test sample {idx}", y=1.02)
    plt.tight_layout()
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()


def save_forecast_figure(
    forecast_pred: np.ndarray,   # (H_fcast, H, W)
    target_chl:    np.ndarray,   # (H_fcast, H, W)
    target_mask:   np.ndarray,   # (H_fcast, H, W)
    land_mask:     np.ndarray,   # (H, W)
    path: Path,
    idx:  int,
) -> None:
    plt = _plt()
    if plt is None:
        return
    H_fcast = forecast_pred.shape[0]
    fig, axes = plt.subplots(2, H_fcast, figsize=(4 * H_fcast, 7))
    valid_all = target_mask.astype(bool) & ~land_mask[None].astype(bool)
    vals = target_chl[valid_all]
    vmin = float(np.nanpercentile(vals, 2))  if len(vals) else -2.0
    vmax = float(np.nanpercentile(vals, 98)) if len(vals) else  2.0
    for t in range(H_fcast):
        v = target_mask[t].astype(bool) & ~land_mask.astype(bool)
        p = np.where(v, forecast_pred[t], np.nan)
        g = np.where(v, target_chl[t],    np.nan)
        axes[0, t].imshow(p, cmap="viridis", vmin=vmin, vmax=vmax, origin="lower")
        axes[0, t].set_title(f"Pred t+{t+1}")
        axes[1, t].imshow(g, cmap="viridis", vmin=vmin, vmax=vmax, origin="lower")
        axes[1, t].set_title(f"Target t+{t+1}")
        axes[0, t].axis("off")
        axes[1, t].axis("off")
    plt.suptitle(f"Forecast — test sample {idx}", y=1.01)
    plt.tight_layout()
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()


def save_summary_figures(
    recon_result:   dict,
    calib_bins:     list,
    routing_result: dict,
    out_dir:        Path,
) -> None:
    plt = _plt()
    if plt is None:
        return
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    # RMSE bar: all / valid / gap
    cats     = ["all", "valid", "gap"]
    rmse_v   = [recon_result.get(c, {}).get("rmse", 0) for c in cats]
    ssim_v   = [recon_result.get(c, {}).get("ssim", 0) for c in cats]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    bars = axes[0].bar(cats, rmse_v, color=["steelblue", "seagreen", "tomato"])
    for bar, v in zip(bars, rmse_v):
        axes[0].text(bar.get_x() + bar.get_width() / 2, v + 0.001,
                     f"{v:.4f}", ha="center", va="bottom", fontsize=9)
    axes[0].set_ylabel("RMSE")
    axes[0].set_title("Reconstruction RMSE by pixel type")
    bars2 = axes[1].bar(cats, ssim_v, color=["steelblue", "seagreen", "tomato"])
    for bar, v in zip(bars2, ssim_v):
        axes[1].text(bar.get_x() + bar.get_width() / 2, v + 0.001,
                     f"{v:.4f}", ha="center", va="bottom", fontsize=9)
    axes[1].set_ylabel("SSIM")
    axes[1].set_title("Reconstruction SSIM by pixel type")
    fig.tight_layout()
    fig.savefig(fig_dir / "recon_metrics.png", dpi=150)
    plt.close(fig)

    # Calibration reliability diagram
    if calib_bins:
        pred_stds   = [b["pred_std"]    for b in calib_bins]
        actual_rmse = [b["actual_rmse"] for b in calib_bins]
        lim = max(max(pred_stds), max(actual_rmse)) * 1.1
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot([0, lim], [0, lim], "k--", lw=1, label="Perfect calibration")
        ax.plot(pred_stds, actual_rmse, "o-", color="steelblue", label="Model")
        ax.set_xlabel("Predicted std")
        ax.set_ylabel("Actual RMSE")
        ax.set_title("Uncertainty calibration")
        ax.legend()
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        fig.tight_layout()
        fig.savefig(fig_dir / "calibration.png", dpi=150)
        plt.close(fig)

    # MoE routing bar chart
    if routing_result and "mean_weight" in routing_result:
        experts = sorted(routing_result["mean_weight"].keys())
        means   = [routing_result["mean_weight"][e] for e in experts]
        stds    = [routing_result["std_weight"][e]  for e in experts]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(experts, means, yerr=stds, capsize=4, color="steelblue", alpha=0.8)
        ax.axhline(1.0 / len(experts), color="red", linestyle="--", label="Uniform")
        ax.set_ylabel("Mean routing weight")
        ax.set_title(
            f"MoE routing  util={routing_result['utilisation']:.3f}"
            f"  H={routing_result['entropy']:.3f}"
        )
        ax.set_ylim(0, 0.6)
        ax.legend()
        fig.tight_layout()
        fig.savefig(fig_dir / "routing.png", dpi=150)
        plt.close(fig)

    log.info(f"Summary figures saved to {fig_dir}")


# ======================================================================
# Bloom forecast accumulator
# ======================================================================

class BloomForecastAccumulator:
    """
    Accumulates per-step bloom detection metrics: precision, recall, F1.
    Bloom targets are derived from target_chl at a normalized threshold.
    """

    def __init__(self, h_fcast: int, bloom_threshold: float = 2.5):
        self.h_fcast = h_fcast
        self.bloom_threshold = bloom_threshold
        self.tp = [0] * h_fcast
        self.fp = [0] * h_fcast
        self.fn = [0] * h_fcast
        self.tn = [0] * h_fcast
        self.n_pixels = [0] * h_fcast

    def update(
        self,
        bloom_logits: torch.Tensor,   # (B, H_fcast, H, W)
        target_chl:   torch.Tensor,   # (B, H_fcast, H, W)
        target_mask:  torch.Tensor,   # (B, H_fcast, H, W)
        land_mask:    torch.Tensor,   # (B, H, W)
    ) -> None:
        ocean = (1.0 - land_mask.float()).unsqueeze(1)
        valid = (target_mask.float() * ocean).bool().cpu()
        pred_bloom = (bloom_logits.float().cpu() > 0.0)   # logit > 0 → prob > 0.5
        true_bloom = (target_chl.float().cpu() > self.bloom_threshold)

        for h in range(self.h_fcast):
            m = valid[:, h]
            if not m.any():
                continue
            p = pred_bloom[:, h][m]
            t = true_bloom[:, h][m]
            self.tp[h] += int((p & t).sum())
            self.fp[h] += int((p & ~t).sum())
            self.fn[h] += int((~p & t).sum())
            self.tn[h] += int((~p & ~t).sum())
            self.n_pixels[h] += int(m.sum())

    def compute(self) -> dict:
        results = {}
        all_f1 = []
        for h in range(self.h_fcast):
            tp, fp, fn = self.tp[h], self.fp[h], self.fn[h]
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            bloom_rate = (tp + fn) / self.n_pixels[h] if self.n_pixels[h] > 0 else 0.0
            results[f"step_{h+1}"] = {
                "precision":  float(prec),
                "recall":     float(rec),
                "f1":         float(f1),
                "tp": tp, "fp": fp, "fn": fn,
                "bloom_rate": float(bloom_rate),
            }
            all_f1.append(f1)
        results["macro_f1"] = float(np.mean(all_f1)) if all_f1 else 0.0
        return results


# ======================================================================
# Ecosystem impact accumulator
# ======================================================================

class EcosystemImpactAccumulator:
    """Accumulates ecosystem impact score statistics."""

    def __init__(self):
        self.scores = []
        self.high_impact_count = 0
        self.total_ocean_pixels = 0

    def update(self, impact: torch.Tensor, land_mask: torch.Tensor) -> None:
        ocean = (1.0 - land_mask.float()).bool().cpu()
        vals = impact.float().cpu()[ocean].numpy()
        if len(vals) > 0:
            self.scores.append(vals)
            self.high_impact_count += int((vals > 0.6).sum())
            self.total_ocean_pixels += len(vals)

    def compute(self) -> dict:
        if not self.scores:
            return {}
        all_scores = np.concatenate(self.scores)
        return {
            "mean":           float(np.mean(all_scores)),
            "std":            float(np.std(all_scores)),
            "median":         float(np.median(all_scores)),
            "p90":            float(np.percentile(all_scores, 90)),
            "p95":            float(np.percentile(all_scores, 95)),
            "p99":            float(np.percentile(all_scores, 99)),
            "max":            float(np.max(all_scores)),
            "high_impact_frac": float(self.high_impact_count / max(self.total_ocean_pixels, 1)),
        }


def save_bloom_forecast_figure(
    bloom_probs:  np.ndarray,   # (H_fcast, H, W)  — sigmoid probabilities
    target_chl:   np.ndarray,   # (H_fcast, H, W)
    target_mask:  np.ndarray,   # (H_fcast, H, W)
    land_mask:    np.ndarray,   # (H, W)
    impact:       np.ndarray,   # (H, W)
    path: Path,
    idx:  int,
    bloom_threshold: float = 2.5,
) -> None:
    """Save bloom probability maps + ecosystem impact for one sample."""
    plt = _plt()
    if plt is None:
        return
    H_fcast = bloom_probs.shape[0]
    fig, axes = plt.subplots(2, H_fcast + 1, figsize=(4 * (H_fcast + 1), 7))

    land = land_mask.astype(bool)

    # Row 1: bloom probability per step
    for t in range(H_fcast):
        prob_vis = np.where(land, np.nan, bloom_probs[t])
        axes[0, t].imshow(prob_vis, cmap="YlOrRd", vmin=0, vmax=1, origin="lower")
        axes[0, t].set_title(f"Bloom prob t+{t+1}")
        axes[0, t].axis("off")

    # Row 1, last column: ecosystem impact
    impact_vis = np.where(land, np.nan, impact)
    im = axes[0, -1].imshow(impact_vis, cmap="hot_r", vmin=0, vmax=1, origin="lower")
    axes[0, -1].set_title("Ecosystem impact")
    axes[0, -1].axis("off")
    plt.colorbar(im, ax=axes[0, -1], shrink=0.8)

    # Row 2: ground truth bloom (binary from target_chl)
    for t in range(H_fcast):
        v = target_mask[t].astype(bool) & ~land
        truth = np.where(v, (target_chl[t] > bloom_threshold).astype(float), np.nan)
        axes[1, t].imshow(truth, cmap="YlOrRd", vmin=0, vmax=1, origin="lower")
        axes[1, t].set_title(f"True bloom t+{t+1}")
        axes[1, t].axis("off")

    # Row 2, last column: max bloom prob across steps
    max_prob = np.where(land, np.nan, bloom_probs.max(axis=0))
    axes[1, -1].imshow(max_prob, cmap="YlOrRd", vmin=0, vmax=1, origin="lower")
    axes[1, -1].set_title("Max bloom prob")
    axes[1, -1].axis("off")

    plt.suptitle(f"Bloom forecast + Ecosystem impact — sample {idx}", y=1.01)
    plt.tight_layout()
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()


# ======================================================================
# Forward with routing (eval mode)
# ======================================================================

def forward_with_routing(model, batch):
    """Run model eval-mode forward and always collect routing weights."""
    m = model.module if hasattr(model, "module") else model
    import torch as _t
    cfg = m.cfg

    chl_obs    = batch["chl_obs"]
    obs_mask   = batch["obs_mask"]
    mcar_mask  = batch["mcar_mask"]
    mnar_mask  = batch["mnar_mask"]
    bloom_mask = batch["bloom_mask"]
    physics    = batch["physics"]
    wind       = batch["wind"]
    static_    = batch["static"]
    discharge  = batch["discharge"]
    bgc_aux    = batch["bgc_aux"]

    optical = _t.stack([chl_obs, obs_mask], dim=2)
    masks   = _t.stack([obs_mask, mcar_mask, mnar_mask, bloom_mask], dim=2)

    mask_emb = m.masknet(masks)
    opt_feat = m.opt_enc(optical)
    phy_feat = m.phy_enc(physics, wind, static_)
    bgc_feat = m.bgc_enc(bgc_aux)
    dis_feat = m.discharge_enc(discharge)

    fused   = m.fusion(opt_feat, phy_feat, mask_emb, bgc_feat, dis_feat)
    state   = m.temporal(fused)
    decoded, routing_w = m.decoder(state, return_routing=True)

    return {
        "recon":          m.recon_head(decoded),
        "forecast":       m.forecast_head(decoded),
        "uncertainty":    m.uncertainty_head(decoded),
        "eri":            m.eri_head(decoded),
        "bloom_forecast": m.bloom_fcast_head(decoded),
    }, routing_w


# ======================================================================
# Main
# ======================================================================

def evaluate(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir / "figures"
    if not args.no_figures:
        fig_dir.mkdir(exist_ok=True)

    device = (
        torch.device(args.device) if args.device
        else torch.device("cuda") if torch.cuda.is_available()
        else torch.device("cpu")
    )
    use_amp = not args.no_amp and device.type == "cuda"
    log.info(f"Device: {device}  |  AMP: {use_amp}")

    sys.path.insert(0, str(Path(__file__).parent))
    from model import MARASSModel, ModelConfig
    from loss import build_eri_target
    from dataset import build_dataloaders
    from model import compute_ecosystem_impact

    cfg   = ModelConfig()
    model = MARASSModel(cfg).to(device)
    ckpt  = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    log.info(
        f"Loaded: {args.ckpt}  "
        f"(epoch {ckpt.get('epoch','?')}, "
        f"val_loss {ckpt.get('val_loss', float('nan')):.4f})"
    )

    loaders = build_dataloaders(
        patch_dir=args.patch_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    test_loader = loaders["test"]
    log.info(f"Test set: {len(test_loader)} batches")

    recon_acc   = ReconAccumulator()
    fcast_acc   = ForecastAccumulator(cfg.H_fcast)
    eri_acc     = ERIAccumulator(cfg.n_eri_levels)
    uncert_acc  = UncertaintyAccumulator()
    routing_acc = RoutingAccumulator(cfg.n_experts)
    bloom_acc   = BloomForecastAccumulator(cfg.H_fcast, bloom_threshold=2.5)
    impact_acc  = EcosystemImpactAccumulator()

    n_figs = 0
    n_batches = len(test_loader)

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            if i % 10 == 0:
                log.info(f"  Batch {i}/{n_batches}")

            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                outputs, routing_w = forward_with_routing(model, batch)

            land_mask  = batch["land_mask"]
            obs_mask_t = batch["obs_mask"][:, -1]
            target     = batch["chl_obs"][:, -1]
            target_chl = batch["target_chl"]
            tgt_mask   = batch["target_mask"]
            bloom_mask = batch["bloom_mask"]

            recon_acc.update(
                outputs["recon"], outputs["uncertainty"],
                target, obs_mask_t, land_mask,
            )
            fcast_acc.update(outputs["forecast"], target_chl, tgt_mask, land_mask)

            eri_target = build_eri_target(bloom_mask)
            eri_acc.update(outputs["eri"], eri_target, land_mask)

            uncert_acc.update(
                outputs["uncertainty"], outputs["recon"],
                target, obs_mask_t, land_mask,
            )
            routing_acc.update(routing_w)

            # Bloom forecast metrics
            if "bloom_forecast" in outputs:
                bloom_acc.update(
                    outputs["bloom_forecast"], target_chl, tgt_mask, land_mask,
                )

                # Ecosystem impact scoring
                bloom_probs = torch.sigmoid(outputs["bloom_forecast"])
                impact = compute_ecosystem_impact(
                    bloom_probs, outputs["forecast"],
                    outputs["uncertainty"], batch["static"], land_mask,
                )
                impact_acc.update(impact, land_mask)

            # Per-sample figures
            if not args.no_figures and n_figs < args.n_figures:
                B = outputs["recon"].shape[0]
                for si in range(B):
                    if n_figs >= args.n_figures:
                        break
                    gidx = i * args.batch_size + si
                    save_recon_figure(
                        chl_obs   = batch["chl_obs"][si].cpu().float().numpy(),
                        obs_mask  = batch["obs_mask"][si].cpu().float().numpy(),
                        pred      = outputs["recon"][si].squeeze(0).cpu().float().numpy(),
                        land_mask = land_mask[si].cpu().float().numpy(),
                        path      = fig_dir / f"recon_{gidx:04d}.png",
                        idx       = gidx,
                    )
                    save_forecast_figure(
                        forecast_pred = outputs["forecast"][si].cpu().float().numpy(),
                        target_chl    = target_chl[si].cpu().float().numpy(),
                        target_mask   = tgt_mask[si].cpu().float().numpy(),
                        land_mask     = land_mask[si].cpu().float().numpy(),
                        path          = fig_dir / f"forecast_{gidx:04d}.png",
                        idx           = gidx,
                    )
                    # Bloom forecast + ecosystem impact figure
                    if "bloom_forecast" in outputs:
                        bf_probs = torch.sigmoid(outputs["bloom_forecast"][si]).cpu().float().numpy()
                        sample_impact = compute_ecosystem_impact(
                            torch.sigmoid(outputs["bloom_forecast"][si:si+1]),
                            outputs["forecast"][si:si+1],
                            outputs["uncertainty"][si:si+1],
                            batch["static"][si:si+1],
                            land_mask[si:si+1],
                        )[0].cpu().float().numpy()
                        save_bloom_forecast_figure(
                            bloom_probs   = bf_probs,
                            target_chl    = target_chl[si].cpu().float().numpy(),
                            target_mask   = tgt_mask[si].cpu().float().numpy(),
                            land_mask     = land_mask[si].cpu().float().numpy(),
                            impact        = sample_impact,
                            path          = fig_dir / f"bloom_impact_{gidx:04d}.png",
                            idx           = gidx,
                        )
                    n_figs += 1

    log.info("Computing metrics...")

    recon_result            = recon_acc.compute()
    fcast_result            = fcast_acc.compute()
    eri_result, cm          = eri_acc.compute()
    uncert_result, cal_bins = uncert_acc.compute()
    routing_result          = routing_acc.compute()
    bloom_result            = bloom_acc.compute()
    impact_result           = impact_acc.compute()

    # --- Print ---
    print("\n" + "=" * 62)
    print("RECONSTRUCTION METRICS")
    print("=" * 62)
    for subset in ["all", "valid", "gap"]:
        m = recon_result.get(subset, {})
        if not m:
            continue
        print(f"\n  [{subset.upper()} — {m.get('n_pix',0):,} pixels]")
        print(f"    RMSE : {m['rmse']:.4f}")
        print(f"    MAE  : {m['mae']:.4f}")
        print(f"    Bias : {m['bias']:+.4f}")
        print(f"    R²   : {m['r2']:.4f}")
        print(f"    SSIM : {m['ssim']:.4f}")
    print(f"\n  CRPS   : {recon_result.get('crps', float('nan')):.4f}  (lower = better)")

    print("\n" + "=" * 62)
    print("FORECAST METRICS")
    print("=" * 62)
    for step, m in fcast_result.items():
        if m:
            print(f"  {step}:  RMSE {m['rmse']:.4f}   MAE {m['mae']:.4f}   SSIM {m['ssim']:.4f}")

    print("\n" + "=" * 62)
    print("ERI CLASSIFICATION")
    print("=" * 62)
    print(f"  Accuracy     : {eri_result.get('accuracy', 0):.4f}")
    print(f"  Macro F1     : {eri_result.get('macro_f1', 0):.4f}")
    print(f"  Ordinal MAE  : {eri_result.get('mae_ordinal', 0):.4f}")
    print(f"  Per-class F1 : {eri_result.get('per_class_f1', {})}")

    print("\n" + "=" * 62)
    print("UNCERTAINTY CALIBRATION")
    print("=" * 62)
    print(f"  ECE          : {uncert_result.get('ece', float('nan')):.4f}")
    print(f"  Var-Err corr : {uncert_result.get('var_err_corr', float('nan')):.4f}")

    print("\n" + "=" * 62)
    print("MOE ROUTING")
    print("=" * 62)
    if routing_result:
        for e, w in routing_result.get("mean_weight", {}).items():
            print(f"  {e}: {w:.4f}")
        print(f"  Entropy      : {routing_result['entropy']:.4f} / {routing_result['max_entropy']:.4f}")
        print(f"  Utilisation  : {routing_result['utilisation']:.4f}")
    print("=" * 62)

    print("\n" + "=" * 62)
    print("BLOOM LEAD-TIME PREDICTION")
    print("=" * 62)
    if bloom_result:
        for step_key in sorted(k for k in bloom_result if k.startswith("step_")):
            m = bloom_result[step_key]
            print(f"  {step_key}:  Prec {m['precision']:.4f}   Rec {m['recall']:.4f}   "
                  f"F1 {m['f1']:.4f}   bloom_rate {m['bloom_rate']:.6f}")
        print(f"\n  Macro F1     : {bloom_result.get('macro_f1', 0):.4f}")
    print("=" * 62)

    print("\n" + "=" * 62)
    print("ECOSYSTEM IMPACT ANALYSIS")
    print("=" * 62)
    if impact_result:
        print(f"  Mean score   : {impact_result['mean']:.4f}")
        print(f"  Median       : {impact_result['median']:.4f}")
        print(f"  Std          : {impact_result['std']:.4f}")
        print(f"  P90          : {impact_result['p90']:.4f}")
        print(f"  P95          : {impact_result['p95']:.4f}")
        print(f"  P99          : {impact_result['p99']:.4f}")
        print(f"  Max          : {impact_result['max']:.4f}")
        print(f"  High impact  : {impact_result['high_impact_frac']*100:.2f}% of ocean pixels (score > 0.6)")
    print("=" * 62)

    # --- Save outputs ---
    all_metrics = {
        "checkpoint":     str(args.ckpt),
        "epoch":          ckpt.get("epoch"),
        "val_loss":       ckpt.get("val_loss"),
        "reconstruction": recon_result,
        "forecast":       fcast_result,
        "eri":            eri_result,
        "uncertainty":    uncert_result,
        "routing":        routing_result,
        "bloom_forecast": bloom_result,
        "ecosystem_impact": impact_result,
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)

    np.savetxt(
        out_dir / "confusion_matrix.csv", cm, fmt="%d", delimiter=",",
        header=",".join(f"pred_{c}" for c in range(cfg.n_eri_levels)),
    )

    if cal_bins:
        import csv
        with open(out_dir / "calibration.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cal_bins[0].keys())
            w.writeheader()
            w.writerows(cal_bins)

    if not args.no_figures:
        save_summary_figures(recon_result, cal_bins, routing_result, out_dir)
        log.info(f"Per-sample figures: {n_figs} saved to {fig_dir}")

    log.info(f"Results saved to {out_dir}")
    log.info("Evaluation complete.")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    evaluate(get_args())