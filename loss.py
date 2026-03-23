"""
loss.py — MM-MARAS loss functions

Four task losses + combined training loss.

Loss summary:
    recon_loss      Heteroscedastic NLL over masked ocean pixels
                    Jointly optimises Chl-a prediction and uncertainty
                    calibration. Penalises both wrong predictions and
                    miscalibrated confidence.

    forecast_loss   Masked Huber (smooth-L1) over target_mask * (1 - land_mask)
                    Switched from MSE to Huber for robustness against
                    bloom-event outliers that spike Chl-a.

    eri_loss        Ordinal cross-entropy via cumulative link model
                    Respects the ordinal structure of ERI levels (0–4):
                    predicting level 2 when truth is 3 is less wrong than
                    predicting level 0.

    aux_loss        MoE load-balancing (from moe_decoder.py)
                    Prevents expert collapse.

Combined:
    loss = (
        w_recon    * recon_loss
      + w_forecast * forecast_loss
      + w_eri      * eri_loss
      + w_aux      * aux_loss
    )

Default weights:
    w_recon    = 1.0   primary task
    w_forecast = 0.5   secondary task (harder, lower weight early in training)
    w_eri      = 0.3   auxiliary classification task
    w_aux      = 0.001  load-balancing regulariser (reduced from 0.01)

Curriculum (optional):
    Pass step / total_steps to MARASSLoss.forward() to ramp up forecast
    and ERI weights from 0 over the first 20% of training. This lets the
    model first learn gap filling before being asked to forecast.

Changes vs previous version
----------------------------
[FIX 1] eri_loss — ordinal penalty now uses soft expected level (differentiable)
    Previously: `logits.argmax(dim=1)` — hard non-differentiable assignment.
    Now: `(softmax(logits) * level_indices).sum(dim=1)` — smooth expected level.
    Impact: cleaner gradient signal for the ordinal structure.

[FIX 2] eri_loss — focal modulation + rebalanced class weights
    Previously: class_weights = [0.20, 2.0, 3.0, 4.0, 5.0]
    Now: class_weights = [0.15, 5.0, 4.0, 4.0, 5.0]
    + focal factor (1 - p_correct)^gamma applied per pixel.
    Impact: addresses class 1 F1=0.378 — focal loss concentrates gradient on
    hard examples at the class 0/1 decision boundary.

[FIX 3] forecast_loss — switched from MSE to Huber (smooth-L1, delta=0.5)
    Previously: plain `(pred - target).pow(2)`.
    Now: `F.huber_loss(..., delta=0.5)` per valid pixel.
    Impact: outlier bloom events no longer dominate the forecast gradient,
    improving step +4/+5 RMSE.

[FIX 4] holdout_recon_loss — uses heteroscedastic NLL instead of plain MSE
    Previously: raw MSE on held-out pixels, inconsistent with recon_loss.
    Now: same NLL formula as recon_loss, using the uncertainty head output.
    Impact: gap-filling and uncertainty are jointly optimised on held-out pixels,
    not just on observed pixels. Variance-error correlation should improve.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from moe_decoder import compute_aux_loss


# ======================================================================
# Reconstruction loss — heteroscedastic NLL
# ======================================================================

def recon_loss(
    pred: Tensor,
    log_var: Tensor,
    target: Tensor,
    obs_mask: Tensor,
    land_mask: Tensor,
    holdout_mask: Tensor | None = None,
) -> Tensor:
    """
    Heteroscedastic negative log-likelihood loss for Chl-a reconstruction.

    Jointly optimises prediction accuracy and uncertainty calibration:

        NLL = 0.5 * (log_var + (pred - target)^2 / exp(log_var))

    A pixel where the model is uncertain (high log_var) gets a smaller
    gradient from the squared error term, but pays a price via the log_var
    term — forcing the model to be uncertain only when it truly needs to be.

    Supervision is restricted to pixels that are:
        - Valid in the input (obs_mask == 1): we know the true value
        - Not land (land_mask == 0)

    We supervise on OBSERVED pixels, not missing ones. The model learns
    to reconstruct by predicting values it can verify. Gap filling quality
    is evaluated separately (held-out spatial blocks).

    Args:
        pred:      (B, 1, H, W)   predicted Chl-a (log-space, normalised)
        log_var:   (B, 1, H, W)   predicted log-variance (from uncertainty head)
        target:    (B, T, H, W)   chl_obs — use last time step as reconstruction target
        obs_mask:  (B, T, H, W)   1 = valid pixel
        land_mask: (B, H, W)      1 = land pixel

    Returns:
        Scalar loss (mean over valid pixels).
    """
    # Use the last observed time step as the reconstruction target
    target_t  = target[:, -1]                              # (B, H, W)
    valid_t   = obs_mask[:, -1]                            # (B, H, W)

    # Valid supervision mask: observed ocean pixels only
    # Exclude pixels that were held out (they are supervised by holdout_recon_loss)
    ocean = 1.0 - land_mask                                # (B, H, W)
    sup_mask = valid_t * ocean                             # (B, H, W)
    if holdout_mask is not None:
        sup_mask = sup_mask * (1.0 - holdout_mask)        # remove held-out pixels

    pred_sq   = pred.squeeze(1)                            # (B, H, W)
    lv_sq     = log_var.squeeze(1)                         # (B, H, W)

    # Clamp log_var for numerical stability — avoids exp() blowup
    lv_clamped = lv_sq.clamp(min=-10.0, max=10.0)

    nll = 0.5 * (lv_clamped + (pred_sq - target_t).pow(2) / lv_clamped.exp())

    # Mean over valid pixels only
    n_valid = sup_mask.sum().clamp(min=1.0)
    return (nll * sup_mask).sum() / n_valid


# ======================================================================
# Holdout reconstruction loss — gap-filling supervision
# [FIX 4] Uses heteroscedastic NLL instead of plain MSE for consistency
# ======================================================================

def holdout_recon_loss(
    pred:         Tensor,
    log_var:      Tensor,
    target:       Tensor,
    holdout_mask: Tensor,
    land_mask:    Tensor,
) -> Tensor:
    """
    Heteroscedastic NLL loss on pixels that were artificially held out during
    the forward pass.

    These pixels were observed (ground truth available) but zeroed in the
    optical input so the model had to infer them from context. Supervising
    on them directly trains the gap-filling pathway.

    [FIX 4] Previously used plain MSE. Now uses the same NLL formula as
    recon_loss so that:
      (a) gap-filling and uncertainty are jointly supervised on held-out pixels,
      (b) the loss scale is consistent with recon_loss,
      (c) the uncertainty head gets gradient from both observed and gap pixels.

    Args:
        pred:         (B, 1, H, W)  reconstruction output
        log_var:      (B, 1, H, W)  log-variance from uncertainty head
        target:       (B, H, W)     chl_obs last timestep (ground truth)
        holdout_mask: (B, H, W)     1 = pixel was held out this forward pass
        land_mask:    (B, H, W)     1 = land

    Returns:
        Scalar loss.
    """
    ocean    = 1.0 - land_mask                             # (B, H, W)
    sup_mask = holdout_mask * ocean                        # only ocean hold-outs
    n_valid  = sup_mask.sum().clamp(min=1.0)

    pred_sq = pred.squeeze(1)                              # (B, H, W)
    lv_sq   = log_var.squeeze(1).clamp(min=-10.0, max=10.0)

    nll = 0.5 * (lv_sq + (pred_sq - target).pow(2) / lv_sq.exp())
    return (nll * sup_mask).sum() / n_valid


# ======================================================================
# Forecast loss — masked Huber (smooth-L1)
# [FIX 3] Switched from MSE to Huber for bloom-outlier robustness
# ======================================================================

def forecast_loss(
    pred: Tensor,
    target: Tensor,
    target_mask: Tensor,
    land_mask: Tensor,
    delta: float = 0.5,
) -> Tensor:
    """
    Masked Huber (smooth-L1) loss for Chl-a forecasting.

    [FIX 3] Previously used plain MSE. Bloom events produce Chl-a spikes
    that are legitimate extremes, not noise — but they cause MSE gradients
    to be dominated by rare high-error pixels at t+4/t+5, destabilising
    training. Huber with delta=0.5 behaves like MSE for errors < 0.5
    (typical range) and like MAE for larger errors (bloom spikes), keeping
    gradients bounded.

    delta=0.5: For log-normalised Chl-a, errors > 0.5 correspond roughly
    to >1.6× the typical interquartile range — a reasonable threshold for
    "this is an outlier bloom spike, not a normal prediction error".

    Only supervises pixels that are both:
        - Observable in the forecast window (target_mask == 1)
        - Not land (land_mask == 0)

    Args:
        pred:        (B, H_fcast, H, W)   predicted future Chl-a
        target:      (B, H_fcast, H, W)   true future Chl-a (target_chl)
        target_mask: (B, H_fcast, H, W)   1 = valid observable pixel
        land_mask:   (B, H, W)            1 = land pixel
        delta:       Huber threshold (default 0.5)

    Returns:
        Scalar loss (mean over valid pixels).
    """
    ocean = (1.0 - land_mask).unsqueeze(1)                 # (B, 1, H, W)
    valid = target_mask * ocean                            # (B, H_fcast, H, W)

    # F.huber_loss computes element-wise; we mask manually for correct mean
    huber = F.huber_loss(pred, target, reduction="none", delta=delta)
    n_valid = valid.sum().clamp(min=1.0)
    return (huber * valid).sum() / n_valid


# ======================================================================
# ERI loss — ordinal cross-entropy (cumulative link model)
# [FIX 1] Soft ordinal penalty (was hard argmax)
# [FIX 2] Focal modulation + rebalanced class weights
# ======================================================================

def eri_loss(
    logits: Tensor,
    target: Tensor,
    land_mask: Tensor,
    bloom_mask: Tensor | None = None,
    focal_gamma: float = 2.0,
) -> Tensor:
    """
    Ordinal cross-entropy for ERI classification (5 levels: 0–4).

    Uses a cumulative link model (CLM), also called proportional-odds model:
        P(Y <= k) = sigmoid(threshold_k - score)   for k = 0..3
        P(Y == k) = P(Y <= k) - P(Y <= k-1)

    The thresholds are implicit in the 5-channel logits: we treat logits
    as scores per level and derive cumulative probabilities via a log-softmax
    over the ordinal levels.

    In practice we use a simplified but effective approach:
        - Convert logits to log-probabilities via log-softmax
        - Apply standard NLL loss with integer class labels
        - [FIX 1] Weight pixel losses by (1 + |soft_pred_level - true_level|)
          where soft_pred_level = expected level under softmax (differentiable)
        - [FIX 2] Apply focal modulation: multiply by (1-p_correct)^gamma
          so the model focuses gradient on hard examples at class boundaries

    [FIX 1] — Soft ordinal penalty:
        Previously used `logits.argmax(dim=1)` (hard, non-differentiable).
        Now uses `(softmax * level_indices).sum(dim=1)` — the expected level
        under the current predicted distribution. This is smooth and gives
        the model a continuous gradient signal to pull predictions toward
        the ordinal midpoint.

    [FIX 2] — Focal loss + rebalanced weights:
        Previously class_weights = [0.20, 2.0, 3.0, 4.0, 5.0].
        Class 1 (1–2 bloom steps) produced F1=0.378, indicating the weight
        was far too low for the true class frequency (~2%). The model was
        defaulting to class 0 on ambiguous low-bloom pixels.

        New weights = [0.15, 5.0, 4.0, 4.0, 5.0]:
          - Class 0 slightly reduced (0.15) to give more headroom to rare classes
          - Class 1 increased from 2.0 → 5.0 to force learning the 0/1 boundary
          - Classes 2–4 rebalanced to avoid drowning class 1

        Focal modulation multiplies each pixel's loss by (1-p_correct)^gamma
        (gamma=2.0 is the standard value from Lin et al. 2017). This means:
          - Easy class-0 pixels (p_correct ≈ 0.99) get × 0.0001 — ignored
          - Hard class-1 pixels (p_correct ≈ 0.5) get × 0.25 — trained hard
          - Misclassified pixels (p_correct ≈ 0.1) get × 0.81 — trained hardest

    Supervision is restricted to:
        - Ocean pixels (land_mask == 0)
        - If bloom_mask provided: upweight bloom pixels (class imbalance)

    Args:
        logits:      (B, 5, H, W)    raw ERI logits from ERIHead
        target:      (B, H, W)       integer ERI labels (0–4), from bloom_mask
                                     (0 = no risk, 4 = extreme bloom)
        land_mask:   (B, H, W)       1 = land
        bloom_mask:  (B, T, H, W) | None   if provided, any-time bloom pixels
                                     get upweighted (bloom events are rare)
        focal_gamma: Focal modulation exponent (default 2.0, 0 = no focal loss)

    Returns:
        Scalar loss.
    """
    B, n_levels, H, W = logits.shape
    ocean = 1.0 - land_mask                                # (B, H, W)

    # Log-probabilities and probabilities per level
    log_probs = F.log_softmax(logits, dim=1)               # (B, 5, H, W)
    probs     = log_probs.exp()                            # (B, 5, H, W)

    # Standard NLL: (B, H, W)
    target_long = target.long().clamp(0, n_levels - 1)
    nll = F.nll_loss(log_probs, target_long, reduction="none")  # (B, H, W)

    # [FIX 2] Rebalanced per-class inverse-frequency weights.
    # Class 0 (~95%) slightly reduced; class 1 (~2%) substantially raised.
    # Approximate priors for Bay of Bengal bloom statistics:
    #   level 0 (~95%), level 1 (~2%), level 2 (~1%), level 3 (~1%), level 4 (~1%)
    class_weights = torch.tensor(
        [0.15, 5.0, 4.0, 4.0, 5.0], device=logits.device
    )
    sample_weight = class_weights[target_long]             # (B, H, W)

    # [FIX 1] Soft ordinal penalty: expected level under predicted distribution.
    # Shape: (1, 5, 1, 1) for broadcast over (B, 5, H, W)
    level_idx  = torch.arange(n_levels, device=logits.device, dtype=torch.float)
    level_idx  = level_idx.view(1, n_levels, 1, 1)
    soft_level = (probs * level_idx).sum(dim=1)            # (B, H, W) — differentiable
    ord_penalty = 1.0 + (soft_level - target.float()).abs()

    # [FIX 2] Focal modulation: (1 - p_correct)^gamma
    # Gather the probability assigned to the true class at each pixel
    p_correct = probs.gather(
        dim=1, index=target_long.unsqueeze(1)
    ).squeeze(1)                                           # (B, H, W)
    focal_weight = (1.0 - p_correct).clamp(min=0.0).pow(focal_gamma)

    pixel_loss = nll * ord_penalty * sample_weight * focal_weight  # (B, H, W)

    # Bloom upweighting — bloom events are rare so they'd be overwhelmed
    # by the majority "no risk" class without this
    weight = ocean.clone()
    if bloom_mask is not None:
        # Any timestep with bloom activity → upweight that pixel
        bloom_any = (bloom_mask.sum(dim=1) > 0).float()   # (B, H, W)
        weight = weight * (1.0 + 5.0 * bloom_any)         # 6× weight on bloom pixels

    n_valid = weight.sum().clamp(min=1.0)
    return (pixel_loss * weight).sum() / n_valid


# ======================================================================
# ERI target builder
# ======================================================================

def build_eri_target(bloom_mask: Tensor) -> Tensor:
    """
    Derive integer ERI target labels (0–4) from bloom_mask.

    ERI level is proportional to the fraction of T timesteps with active bloom:
        0 bloom steps  → ERI 0 (no risk)
        1–2 steps      → ERI 1 (low)
        3–4 steps      → ERI 2 (moderate)
        5–7 steps      → ERI 3 (high)
        8–10 steps     → ERI 4 (extreme)

    Args:
        bloom_mask: (B, T, H, W)   binary bloom labels

    Returns:
        eri_target: (B, H, W)      integer labels 0–4
    """
    bloom_count = bloom_mask.sum(dim=1)                    # (B, H, W)  0–10

    eri = torch.zeros_like(bloom_count, dtype=torch.long)
    eri[bloom_count >= 1]  = 1
    eri[bloom_count >= 3]  = 2
    eri[bloom_count >= 5]  = 3
    eri[bloom_count >= 8]  = 4

    return eri


# ======================================================================
# Combined loss
# ======================================================================

@dataclass
class LossWeights:
    recon:    float = 1.0
    forecast: float = 0.5
    eri:      float = 0.3
    aux:      float = 0.001   # reduced from 0.01 — allows expert specialisation
    holdout:  float = 0.5     # gap-filling supervision on held-out observed pixels


class MARASSLoss(nn.Module):
    """
    Combined MM-MARAS training loss.

    Wraps all four task losses with configurable weights and optional
    curriculum ramp-up for forecast and ERI losses.

    Curriculum:
        For the first `curriculum_frac` of training (default 20%),
        forecast and ERI weights are linearly ramped from 0 to their
        full values. This lets the model first learn gap filling
        (reconstruction) before being asked to forecast and classify.

        Pass `step` and `total_steps` to forward() to enable curriculum.
        If neither is passed, full weights are applied from step 0.

    Usage:
        criterion = MARASSLoss()

        # Training loop
        outputs = model(batch)
        loss, breakdown = criterion(
            outputs, batch,
            step=global_step,
            total_steps=total_training_steps,
        )
        loss.backward()

        # breakdown is a dict of scalar losses for logging:
        # {"total", "recon", "forecast", "eri", "aux", "holdout", "curriculum_scale"}
    """

    def __init__(
        self,
        weights: LossWeights | None = None,
        curriculum_frac: float = 0.20,
        forecast_delta: float = 0.5,
        eri_focal_gamma: float = 2.0,
    ) -> None:
        super().__init__()
        self.w = weights or LossWeights()
        self.curriculum_frac  = curriculum_frac
        self.forecast_delta   = forecast_delta
        self.eri_focal_gamma  = eri_focal_gamma

    def _curriculum_scale(self, step: int | None, total_steps: int | None) -> float:
        """Returns a scale in [0, 1] for forecast/ERI based on training progress."""
        if step is None or total_steps is None:
            return 1.0
        warmup = int(total_steps * self.curriculum_frac)
        if warmup == 0:
            return 1.0
        return min(1.0, step / warmup)

    def forward(
        self,
        outputs: dict,
        batch: dict,
        step: int | None = None,
        total_steps: int | None = None,
    ) -> tuple[Tensor, dict[str, float]]:
        """
        Args:
            outputs:     Model output dict (from MARASSModel.forward).
            batch:       Dataset batch dict (from MARASSDataset).
            step:        Current global training step (for curriculum).
            total_steps: Total training steps (for curriculum).

        Returns:
            loss:      Scalar total loss (call .backward() on this).
            breakdown: Dict of individual loss values for logging.
        """
        land_mask   = batch["land_mask"]                   # (B, H, W)
        obs_mask    = batch["obs_mask"]                    # (B, T, H, W)
        chl_obs     = batch["chl_obs"]                     # (B, T, H, W)
        target_chl  = batch["target_chl"]                  # (B, H_fcast, H, W)
        target_mask = batch["target_mask"]                 # (B, H_fcast, H, W)
        bloom_mask  = batch["bloom_mask"]                  # (B, T, H, W)

        # --- Reconstruction ---
        # Pass holdout_mask so NLL supervision excludes held-out pixels —
        # those are supervised separately by holdout_recon_loss.
        l_recon = recon_loss(
            pred         = outputs["recon"],
            log_var      = outputs["uncertainty"],
            target       = chl_obs,
            obs_mask     = obs_mask,
            land_mask    = land_mask,
            holdout_mask = outputs.get("holdout_mask"),
        )

        # --- Forecast (Huber) ---
        l_forecast = forecast_loss(
            pred        = outputs["forecast"],
            target      = target_chl,
            target_mask = target_mask,
            land_mask   = land_mask,
            delta       = self.forecast_delta,
        )

        # --- ERI (focal + soft ordinal) ---
        eri_target = build_eri_target(bloom_mask)          # (B, H, W)
        l_eri = eri_loss(
            logits      = outputs["eri"],
            target      = eri_target,
            land_mask   = land_mask,
            bloom_mask  = bloom_mask,
            focal_gamma = self.eri_focal_gamma,
        )

        # --- Aux (MoE load-balancing) ---
        if "routing_weights" in outputs:
            l_aux = compute_aux_loss(outputs["routing_weights"])
        else:
            l_aux = torch.tensor(0.0, device=l_recon.device)

        # --- Holdout reconstruction loss (gap-filling supervision, NLL) ---
        if "holdout_mask" in outputs and outputs["holdout_mask"] is not None:
            l_holdout = holdout_recon_loss(
                pred         = outputs["recon"],
                log_var      = outputs["uncertainty"],
                target       = chl_obs[:, -1],
                holdout_mask = outputs["holdout_mask"],
                land_mask    = land_mask,
            )
        else:
            l_holdout = torch.tensor(0.0, device=l_recon.device)

        # --- Curriculum scaling for secondary tasks ---
        scale = self._curriculum_scale(step, total_steps)

        total = (
            self.w.recon    * l_recon
          + self.w.forecast * scale * l_forecast
          + self.w.eri      * scale * l_eri
          + self.w.aux      * l_aux
          + self.w.holdout  * l_holdout
        )

        breakdown = {
            "total":    total.item(),
            "recon":    l_recon.item(),
            "forecast": l_forecast.item(),
            "eri":      l_eri.item(),
            "aux":      l_aux.item(),
            "holdout":  l_holdout.item(),
            "curriculum_scale": scale,
        }

        return total, breakdown


# ======================================================================
# Smoke test
# ======================================================================

def run_smoke_test() -> None:
    """
    python loss.py
    """
    import logging
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    from model import MARASSModel, ModelConfig

    cfg   = ModelConfig()
    model = MARASSModel(cfg)
    model.train()

    B = 4
    T, H, W, H_fcast = cfg.T, cfg.H, cfg.W, cfg.H_fcast

    # Simulate a realistic batch
    obs = (torch.rand(B, T, H, W) > 0.30).float()   # 70% valid
    batch = {
        "chl_obs":    torch.randn(B, T, H, W) * obs,
        "obs_mask":   obs,
        "mcar_mask":  torch.zeros(B, T, H, W),
        "mnar_mask":  torch.zeros(B, T, H, W),
        "bloom_mask": (torch.rand(B, T, H, W) > 0.95).float(),
        "physics":    torch.randn(B, T, 6, H, W),
        "wind":       torch.randn(B, T, 4, H, W),
        "static":     torch.randn(B, 2, H, W),
        "discharge":  torch.randn(B, T, 2, H, W),
        "bgc_aux":    torch.randn(B, T, 5, H, W),
        "land_mask":  (torch.rand(B, H, W) > 0.97).float(),
        "target_chl": torch.randn(B, H_fcast, H, W),
        "target_mask": (torch.rand(B, H_fcast, H, W) > 0.30).float(),
    }

    outputs = model(batch)
    criterion = MARASSLoss()

    # Step 0 of 1000 — curriculum scale should be 0 for forecast/ERI
    loss, breakdown = criterion(outputs, batch, step=0, total_steps=1000)

    print("\n--- Loss breakdown (step=0, curriculum=0%) ---")
    for k, v in breakdown.items():
        print(f"  {k:<22} {v:.4f}")

    assert torch.isfinite(loss), "Loss is not finite"
    print(f"\nTotal loss: {loss.item():.4f}  (finite: OK)")

    # Mid-training — curriculum scale should be ~0.5
    loss2, breakdown2 = criterion(outputs, batch, step=100, total_steps=1000)
    print(f"\n--- Curriculum scale at step 100/1000: {breakdown2['curriculum_scale']:.2f} ---")
    assert breakdown2["curriculum_scale"] == 0.5, "Expected 0.5 at 10% of warmup (20% of total)"

    # Full training — scale = 1.0
    loss3, breakdown3 = criterion(outputs, batch, step=500, total_steps=1000)
    print(f"--- Curriculum scale at step 500/1000: {breakdown3['curriculum_scale']:.2f} ---")
    assert breakdown3["curriculum_scale"] == 1.0

    # Backward pass check
    loss.backward()
    grad_norms = {
        name: p.grad.norm().item()
        for name, p in model.named_parameters()
        if p.grad is not None
    }
    n_params_with_grad = len(grad_norms)
    all_finite = all(torch.isfinite(torch.tensor(v)) for v in grad_norms.values())
    print(f"\nBackward pass: {n_params_with_grad} params with gradients, all finite: {all_finite}")

    # Verify focal loss reduces loss on easy examples
    # (sanity: ERI loss with all-class-0 predictions should be small)
    print("\n--- Focal loss sanity check ---")
    easy_logits = torch.zeros(B, 5, H, W)
    easy_logits[:, 0] = 10.0          # very confident class 0
    easy_target = torch.zeros(B, H, W, dtype=torch.long)
    land = torch.zeros(B, H, W)
    l_easy = eri_loss(easy_logits, easy_target, land, focal_gamma=2.0)
    l_easy_nofocal = eri_loss(easy_logits, easy_target, land, focal_gamma=0.0)
    print(f"  ERI loss (easy correct, focal γ=2): {l_easy.item():.6f}")
    print(f"  ERI loss (easy correct, no focal):  {l_easy_nofocal.item():.6f}")
    assert l_easy.item() < l_easy_nofocal.item(), "Focal loss should reduce easy-example loss"
    print("  Focal suppression confirmed: easy examples correctly down-weighted")

    print("\nSmoke test passed.")


if __name__ == "__main__":
    run_smoke_test()