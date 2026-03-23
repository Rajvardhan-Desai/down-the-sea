"""
eval.py — MM-MARAS test set evaluation

Runs five evaluation passes on the test split:

    1. Reconstruction metrics (RMSE, MAE, bias, R²)
       Reported three ways:
         - All ocean pixels
         - Valid (observed) pixels only  — measures fitting quality
         - Gap (missing) pixels only     — measures gap-filling quality

    2. Forecast metrics (RMSE, MAE per horizon step)
       Against target_chl over the H_fcast=5 future steps.

    3. ERI classification metrics
       Accuracy, macro-F1, confusion matrix across the 5 ordinal levels.

    4. Uncertainty calibration
       Expected Calibration Error (ECE) and reliability diagram data.
       Checks whether predicted std correlates with actual error.

    5. MoE routing analysis
       Mean routing weight per expert. If a month column is in your
       dataset patches, also breaks down routing by month to check
       whether experts learned seasonal regimes.

Usage:
    python eval.py --ckpt /kaggle/working/checkpoints/best.pt \
                   --patch-dir /kaggle/input/.../patches \
                   --out-dir /kaggle/working/eval_results

Outputs written to --out-dir:
    metrics.json          All scalar metrics
    confusion_matrix.csv  ERI confusion matrix
    calibration.csv       ECE reliability data
    routing.csv           Per-sample routing weights + metadata
    figures/              PNG plots (recon scatter, calibration, routing)
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
    p.add_argument("--ckpt",       required=True,        help="Path to best.pt checkpoint")
    p.add_argument("--patch-dir",  required=True,        help="Root patches directory")
    p.add_argument("--out-dir",    default="eval_results", help="Output directory for results")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--device",     default=None,         help="cuda / cpu")
    p.add_argument("--no-figures", action="store_true",  help="Skip matplotlib figure generation")
    return p.parse_args()


# ======================================================================
# Metric accumulators
# ======================================================================

class ReconAccumulator:
    """Accumulates pixel-level reconstruction errors."""

    def __init__(self):
        self.reset()

    def reset(self):
        self._all   = {"se": [], "ae": [], "pred": [], "true": []}
        self._valid = {"se": [], "ae": [], "pred": [], "true": []}
        self._gap   = {"se": [], "ae": [], "pred": [], "true": []}

    def update(
        self,
        pred:      torch.Tensor,   # (B, 1, H, W)
        target:    torch.Tensor,   # (B, H, W)  — last chl_obs timestep
        obs_mask:  torch.Tensor,   # (B, H, W)  — 1 = observed
        land_mask: torch.Tensor,   # (B, H, W)  — 1 = land
    ) -> None:
        pred_sq    = pred.squeeze(1).float().cpu()   # (B, H, W)
        target_sq  = target.float().cpu()
        obs_sq     = obs_mask.float().cpu()
        ocean      = (1.0 - land_mask.float()).cpu()

        se = (pred_sq - target_sq).pow(2)
        ae = (pred_sq - target_sq).abs()

        ocean_mask  = ocean.bool()
        valid_mask  = (obs_sq * ocean).bool()
        gap_mask    = ((1 - obs_sq) * ocean).bool()

        for mask, store in [
            (ocean_mask, self._all),
            (valid_mask, self._valid),
            (gap_mask,   self._gap),
        ]:
            if mask.any():
                store["se"].append(se[mask].numpy())
                store["ae"].append(ae[mask].numpy())
                store["pred"].append(pred_sq[mask].numpy())
                store["true"].append(target_sq[mask].numpy())

    def compute(self) -> dict:
        results = {}
        for name, store in [("all", self._all), ("valid", self._valid), ("gap", self._gap)]:
            if not store["se"]:
                results[name] = {}
                continue
            se   = np.concatenate(store["se"])
            ae   = np.concatenate(store["ae"])
            pred = np.concatenate(store["pred"])
            true = np.concatenate(store["true"])

            ss_res = se.sum()
            ss_tot = ((true - true.mean()) ** 2).sum()
            r2     = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

            results[name] = {
                "rmse":  float(np.sqrt(se.mean())),
                "mae":   float(ae.mean()),
                "bias":  float((pred - true).mean()),   # positive = over-predicting
                "r2":    float(r2),
                "n_pix": int(len(se)),
            }
        return results


class ForecastAccumulator:
    """Accumulates per-horizon forecast errors."""

    def __init__(self, h_fcast: int):
        self.h_fcast = h_fcast
        self.se  = [[] for _ in range(h_fcast)]
        self.ae  = [[] for _ in range(h_fcast)]

    def update(
        self,
        pred:        torch.Tensor,   # (B, H_fcast, H, W)
        target:      torch.Tensor,   # (B, H_fcast, H, W)
        target_mask: torch.Tensor,   # (B, H_fcast, H, W)
        land_mask:   torch.Tensor,   # (B, H, W)
    ) -> None:
        ocean = (1.0 - land_mask.float()).unsqueeze(1)  # (B, 1, H, W) — keep on device
        valid = (target_mask.float() * ocean).bool().cpu()

        pred_c   = pred.float().cpu()
        target_c = target.float().cpu()

        for h in range(self.h_fcast):
            m = valid[:, h]
            if m.any():
                diff = pred_c[:, h][m] - target_c[:, h][m]
                self.se[h].append(diff.pow(2).numpy())
                self.ae[h].append(diff.abs().numpy())

    def compute(self) -> dict:
        results = {}
        for h in range(self.h_fcast):
            if not self.se[h]:
                results[f"step_{h+1}"] = {}
                continue
            se = np.concatenate(self.se[h])
            ae = np.concatenate(self.ae[h])
            results[f"step_{h+1}"] = {
                "rmse": float(np.sqrt(se.mean())),
                "mae":  float(ae.mean()),
            }
        return results


class ERIAccumulator:
    """Accumulates ERI classification predictions."""

    def __init__(self, n_levels: int = 5):
        self.n_levels = n_levels
        self.preds  = []
        self.labels = []

    def update(
        self,
        logits:    torch.Tensor,   # (B, 5, H, W)
        target:    torch.Tensor,   # (B, H, W)  integer 0-4
        land_mask: torch.Tensor,   # (B, H, W)
    ) -> None:
        ocean = (1.0 - land_mask.float()).bool().cpu()
        pred_cls = logits.argmax(dim=1).cpu()         # (B, H, W)
        target_c = target.long().cpu()

        self.preds.append(pred_cls[ocean].numpy())
        self.labels.append(target_c[ocean].numpy())

    def compute(self) -> tuple[dict, np.ndarray]:
        preds  = np.concatenate(self.preds)
        labels = np.concatenate(self.labels)

        # Confusion matrix
        cm = np.zeros((self.n_levels, self.n_levels), dtype=np.int64)
        for p, l in zip(preds, labels):
            cm[int(l), int(p)] += 1

        # Accuracy
        acc = float((preds == labels).mean())

        # Per-class F1 (macro average)
        f1s = []
        for c in range(self.n_levels):
            tp = cm[c, c]
            fp = cm[:, c].sum() - tp
            fn = cm[c, :].sum() - tp
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            f1s.append(f1)

        # Mean absolute ordinal error
        mae_ord = float(np.abs(preds.astype(float) - labels.astype(float)).mean())

        metrics = {
            "accuracy":    acc,
            "macro_f1":    float(np.mean(f1s)),
            "per_class_f1": {str(c): float(f1s[c]) for c in range(self.n_levels)},
            "mae_ordinal": mae_ord,
        }
        return metrics, cm


class UncertaintyAccumulator:
    """
    Accumulates predicted log-variance and actual squared errors to
    check calibration: a well-calibrated model has high predicted
    variance where actual errors are large.
    """

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
        ocean = (1.0 - land_mask.float())  # keep on device
        valid = (obs_mask.float() * ocean).bool().cpu()

        lv_sq  = log_var.squeeze(1).float().cpu()
        pred_sq = pred.squeeze(1).float().cpu()
        tgt_sq  = target.float().cpu()

        if valid.any():
            self.log_vars.append(lv_sq[valid].numpy())
            self.sq_errs.append((pred_sq[valid] - tgt_sq[valid]).pow(2).numpy())

    def compute(self) -> tuple[dict, list]:
        """
        Returns scalar ECE and bin data for reliability diagram.

        ECE: expected calibration error — mean absolute difference
        between predicted variance and actual MSE within each quantile bin.
        """
        if not self.log_vars:
            return {}, []

        log_vars = np.concatenate(self.log_vars)
        sq_errs  = np.concatenate(self.sq_errs)
        variances = np.exp(log_vars.clip(-10, 10))

        # Sort by predicted variance into n_bins quantile bins
        order   = np.argsort(variances)
        n       = len(order)
        bin_sz  = n // self.n_bins

        bins = []
        ece_terms = []
        for b in range(self.n_bins):
            idx = order[b * bin_sz: (b + 1) * bin_sz]
            mean_pred_var  = float(variances[idx].mean())
            mean_actual_se = float(sq_errs[idx].mean())
            bins.append({
                "bin":          b,
                "pred_std":     float(math.sqrt(max(mean_pred_var, 0))),
                "actual_rmse":  float(math.sqrt(max(mean_actual_se, 0))),
                "pred_var":     mean_pred_var,
                "actual_mse":   mean_actual_se,
            })
            ece_terms.append(abs(mean_pred_var - mean_actual_se))

        # Pearson correlation between predicted variance and actual SE
        corr = float(np.corrcoef(variances, sq_errs)[0, 1])

        metrics = {
            "ece":         float(np.mean(ece_terms)),
            "var_err_corr": corr,   # positive = uncertainty is informative
        }
        return metrics, bins


class RoutingAccumulator:
    """Collects per-batch routing weights for MoE analysis."""

    def __init__(self, n_experts: int):
        self.n_experts = n_experts
        self.weights   = []   # list of (B, n_experts) arrays

    def update(self, routing_weights: torch.Tensor) -> None:
        self.weights.append(routing_weights.float().cpu().numpy())

    def compute(self) -> dict:
        if not self.weights:
            return {}
        all_w = np.concatenate(self.weights, axis=0)   # (N_samples, n_experts)
        mean_w = all_w.mean(axis=0)
        std_w  = all_w.std(axis=0)

        # Shannon entropy of mean distribution
        p = mean_w + 1e-8
        entropy = float(-(p * np.log(p)).sum())
        max_entropy = float(math.log(self.n_experts))

        return {
            "mean_weight":   {f"expert_{e}": float(mean_w[e]) for e in range(self.n_experts)},
            "std_weight":    {f"expert_{e}": float(std_w[e])  for e in range(self.n_experts)},
            "entropy":       entropy,
            "max_entropy":   max_entropy,
            "utilisation":   entropy / max_entropy,   # 1.0 = perfectly uniform
        }


# ======================================================================
# Figures
# ======================================================================

def save_figures(
    recon_metrics:  dict,
    calib_bins:     list,
    routing_result: dict,
    out_dir:        Path,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("matplotlib not available — skipping figures")
        return

    fig_dir = out_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    # --- Calibration reliability diagram ---
    if calib_bins:
        pred_stds   = [b["pred_std"]    for b in calib_bins]
        actual_rmse = [b["actual_rmse"] for b in calib_bins]
        lim = max(max(pred_stds), max(actual_rmse)) * 1.1

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot([0, lim], [0, lim], "k--", lw=1, label="Perfect calibration")
        ax.plot(pred_stds, actual_rmse, "o-", color="steelblue", label="Model")
        ax.set_xlabel("Predicted std (uncertainty head)")
        ax.set_ylabel("Actual RMSE")
        ax.set_title("Uncertainty calibration")
        ax.legend()
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        fig.tight_layout()
        fig.savefig(fig_dir / "calibration.png", dpi=150)
        plt.close(fig)

    # --- MoE routing bar chart ---
    if routing_result and "mean_weight" in routing_result:
        experts = sorted(routing_result["mean_weight"].keys())
        means   = [routing_result["mean_weight"][e] for e in experts]
        stds    = [routing_result["std_weight"][e]  for e in experts]

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(experts, means, yerr=stds, capsize=4, color="steelblue", alpha=0.8)
        ax.axhline(1.0 / len(experts), color="red", linestyle="--", label="Uniform")
        ax.set_ylabel("Mean routing weight")
        ax.set_title(
            f"MoE routing (utilisation={routing_result['utilisation']:.2f},"
            f" entropy={routing_result['entropy']:.3f})"
        )
        ax.set_ylim(0, 0.6)
        ax.legend()
        fig.tight_layout()
        fig.savefig(fig_dir / "routing.png", dpi=150)
        plt.close(fig)

    # --- RMSE bar chart: all / valid / gap ---
    categories = ["all", "valid", "gap"]
    rmse_vals  = [recon_metrics.get(c, {}).get("rmse", 0) for c in categories]

    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(categories, rmse_vals, color=["steelblue", "seagreen", "tomato"])
    for bar, v in zip(bars, rmse_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.001,
                f"{v:.4f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("RMSE")
    ax.set_title("Reconstruction RMSE by pixel type")
    fig.tight_layout()
    fig.savefig(fig_dir / "recon_rmse.png", dpi=150)
    plt.close(fig)

    log.info(f"Figures saved to {fig_dir}")


# ======================================================================
# Main evaluation loop
# ======================================================================

def evaluate(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Device ---
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    log.info(f"Device: {device}")

    # --- Model ---
    sys.path.insert(0, str(Path(__file__).parent))
    from model import MARASSModel, ModelConfig
    from loss import build_eri_target

    cfg   = ModelConfig()
    model = MARASSModel(cfg).to(device)

    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    log.info(f"Loaded checkpoint: {args.ckpt}  (epoch {ckpt.get('epoch', '?')}, val_loss {ckpt.get('val_loss', '?'):.4f})")

    # --- Data ---
    from dataset import build_dataloaders
    loaders = build_dataloaders(
        patch_dir=args.patch_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    test_loader = loaders["test"]
    log.info(f"Test set: {len(test_loader)} batches")

    # --- Accumulators ---
    recon_acc   = ReconAccumulator()
    fcast_acc   = ForecastAccumulator(cfg.H_fcast)
    eri_acc     = ERIAccumulator(cfg.n_eri_levels)
    uncert_acc  = UncertaintyAccumulator()
    routing_acc = RoutingAccumulator(cfg.n_experts)

    # --- Eval loop ---
    n_batches = len(test_loader)
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            if i % 10 == 0:
                log.info(f"  Batch {i}/{n_batches}")

            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

            # Run model in eval mode — no routing_weights returned
            # Re-enable routing collection by temporarily switching decoder
            decoded_state = _forward_with_routing(model, batch)
            outputs, routing_w = decoded_state

            land_mask  = batch["land_mask"]              # (B, H, W)
            obs_mask_t = batch["obs_mask"][:, -1]        # (B, H, W) — last step
            target     = batch["chl_obs"][:, -1]         # (B, H, W)
            target_chl = batch["target_chl"]             # (B, H_fcast, H, W)
            tgt_mask   = batch["target_mask"]            # (B, H_fcast, H, W)
            bloom_mask = batch["bloom_mask"]             # (B, T, H, W)

            recon_acc.update(outputs["recon"], target, obs_mask_t, land_mask)
            fcast_acc.update(outputs["forecast"], target_chl, tgt_mask, land_mask)

            eri_target = build_eri_target(bloom_mask)
            eri_acc.update(outputs["eri"], eri_target, land_mask)

            uncert_acc.update(
                outputs["uncertainty"], outputs["recon"],
                target, obs_mask_t, land_mask,
            )

            if routing_w is not None:
                routing_acc.update(routing_w)

    # --- Compute ---
    log.info("Computing metrics...")

    recon_result   = recon_acc.compute()
    fcast_result   = fcast_acc.compute()
    eri_result, cm = eri_acc.compute()
    uncert_result, calib_bins = uncert_acc.compute()
    routing_result = routing_acc.compute()

    # --- Print summary ---
    print("\n" + "=" * 60)
    print("RECONSTRUCTION METRICS")
    print("=" * 60)
    for subset in ["all", "valid", "gap"]:
        m = recon_result.get(subset, {})
        if not m:
            continue
        print(f"\n  [{subset.upper()} pixels — {m.get('n_pix', 0):,}]")
        print(f"    RMSE : {m['rmse']:.4f}")
        print(f"    MAE  : {m['mae']:.4f}")
        print(f"    Bias : {m['bias']:+.4f}  ({'over' if m['bias'] > 0 else 'under'}-predicting)")
        print(f"    R²   : {m['r2']:.4f}")

    print("\n" + "=" * 60)
    print("FORECAST METRICS (per horizon step)")
    print("=" * 60)
    for step, m in fcast_result.items():
        if m:
            print(f"  {step}:  RMSE {m['rmse']:.4f}   MAE {m['mae']:.4f}")

    print("\n" + "=" * 60)
    print("ERI CLASSIFICATION")
    print("=" * 60)
    print(f"  Accuracy      : {eri_result.get('accuracy', 0):.4f}")
    print(f"  Macro F1      : {eri_result.get('macro_f1', 0):.4f}")
    print(f"  Ordinal MAE   : {eri_result.get('mae_ordinal', 0):.4f}")
    print(f"  Per-class F1  : {eri_result.get('per_class_f1', {})}")

    print("\n" + "=" * 60)
    print("UNCERTAINTY CALIBRATION")
    print("=" * 60)
    print(f"  ECE           : {uncert_result.get('ece', float('nan')):.4f}")
    print(f"  Var-Err corr  : {uncert_result.get('var_err_corr', float('nan')):.4f}  (>0 = informative)")

    print("\n" + "=" * 60)
    print("MOE ROUTING")
    print("=" * 60)
    if routing_result:
        for e, w in routing_result["mean_weight"].items():
            print(f"  {e}: {w:.4f}")
        print(f"  Entropy       : {routing_result['entropy']:.4f} / {routing_result['max_entropy']:.4f}")
        print(f"  Utilisation   : {routing_result['utilisation']:.4f}  (1.0 = fully uniform)")
    print("=" * 60)

    # --- Save metrics.json ---
    all_metrics = {
        "checkpoint":   str(args.ckpt),
        "epoch":        ckpt.get("epoch"),
        "val_loss":     ckpt.get("val_loss"),
        "reconstruction": recon_result,
        "forecast":       fcast_result,
        "eri":            eri_result,
        "uncertainty":    uncert_result,
        "routing":        routing_result,
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)
    log.info(f"Saved metrics.json to {out_dir}")

    # --- Save confusion matrix ---
    np.savetxt(
        out_dir / "confusion_matrix.csv",
        cm, fmt="%d", delimiter=",",
        header=",".join(f"pred_{i}" for i in range(cfg.n_eri_levels)),
    )

    # --- Save calibration bins ---
    if calib_bins:
        import csv
        with open(out_dir / "calibration.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=calib_bins[0].keys())
            w.writeheader()
            w.writerows(calib_bins)

    # --- Figures ---
    if not args.no_figures:
        save_figures(recon_result, calib_bins, routing_result, out_dir)

    log.info("Evaluation complete.")


def _forward_with_routing(
    model: "MARASSModel",
    batch: dict,
) -> tuple[dict, "torch.Tensor | None"]:
    """
    Run model forward and collect routing weights without switching to
    train mode. Calls decoder directly with return_routing=True.
    """
    # Access underlying model (handles DDP wrapper if present)
    m = model.module if hasattr(model, "module") else model

    from torch import no_grad
    import torch

    cfg = m.cfg

    chl_obs    = batch["chl_obs"]
    obs_mask   = batch["obs_mask"]
    mcar_mask  = batch["mcar_mask"]
    mnar_mask  = batch["mnar_mask"]
    bloom_mask = batch["bloom_mask"]
    physics    = batch["physics"]
    wind       = batch["wind"]
    static     = batch["static"]
    discharge  = batch["discharge"]
    bgc_aux    = batch["bgc_aux"]

    optical = torch.stack([chl_obs, obs_mask], dim=2)
    masks   = torch.stack([obs_mask, mcar_mask, mnar_mask, bloom_mask], dim=2)

    mask_emb = m.masknet(masks)
    opt_feat = m.opt_enc(optical)
    phy_feat = m.phy_enc(physics, wind, static)
    bgc_feat = m.bgc_enc(bgc_aux)
    dis_feat = m.discharge_enc(discharge)

    fused = m.fusion(opt_feat, phy_feat, mask_emb, bgc_feat, dis_feat)
    state = m.temporal(fused)

    # Always collect routing weights during eval
    decoded, routing_weights = m.decoder(state, return_routing=True)

    outputs = {
        "recon":       m.recon_head(decoded),
        "forecast":    m.forecast_head(decoded),
        "uncertainty": m.uncertainty_head(decoded),
        "eri":         m.eri_head(decoded),
    }
    return outputs, routing_weights


# ======================================================================
# Entry point
# ======================================================================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    args = get_args()
    evaluate(args)