"""
train.py — MM-MARAS training loop

Features:
    - Mixed precision training (torch.cuda.amp) — halves VRAM usage
    - Train / val loop with per-epoch metrics
    - Curriculum scheduling (forecast + ERI ramp over first 20% of steps)
    - AdamW optimiser + cosine LR schedule with linear warmup
    - Gradient clipping (max norm 1.0)
    - Checkpoint saving: best val loss + periodic every N epochs
    - TensorBoard logging (losses, LR, grad norm, MoE routing entropy)
    - Resume from checkpoint
    - Device auto-detection (CUDA > MPS > CPU)

Usage:
    # Fresh run (RTX 3060 6GB — default batch size 4 with AMP)
    python train.py

    # Resume from checkpoint
    python train.py --resume checkpoints/last.pt

    # Custom config
    python train.py --epochs 100 --batch-size 4 --lr 3e-4

    # CPU-only (for debugging)
    python train.py --device cpu --batch-size 2 --num-workers 0
"""

from __future__ import annotations

import argparse
import logging
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter

from dataset import build_dataloaders
from loss import MARASSLoss, LossWeights
from model import MARASSModel, ModelConfig

log = logging.getLogger(__name__)


# ======================================================================
# Config
# ======================================================================

def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train MM-MARAS")

    # Paths
    p.add_argument("--patch-dir",    default="data/patches",  help="Root patches directory")
    p.add_argument("--ckpt-dir",     default="checkpoints",   help="Checkpoint output directory")
    p.add_argument("--log-dir",      default="runs",          help="TensorBoard log directory")
    p.add_argument("--resume",       default=None,            help="Path to checkpoint to resume from")

    # Training
    p.add_argument("--epochs",        type=int,   default=50)
    p.add_argument("--batch-size",    type=int,   default=4,   help="Per-GPU batch size (default 4 for 6GB VRAM)")
    p.add_argument("--lr",            type=float, default=1e-4)
    p.add_argument("--weight-decay",  type=float, default=1e-2)
    p.add_argument("--grad-clip",     type=float, default=1.0, help="Max gradient norm")
    p.add_argument("--warmup-epochs", type=int,   default=5,   help="LR linear warmup epochs")
    p.add_argument("--save-every",    type=int,   default=10,  help="Save periodic checkpoint every N epochs")
    p.add_argument("--no-amp",        action="store_true",     help="Disable mixed precision (use if NaN losses appear)")

    # Loss weights
    p.add_argument("--w-recon",    type=float, default=1.0)
    p.add_argument("--w-forecast", type=float, default=0.5)
    p.add_argument("--w-eri",      type=float, default=0.3)
    p.add_argument("--w-aux",      type=float, default=0.01)

    # DataLoader
    p.add_argument("--num-workers", type=int, default=2)

    # Hardware
    p.add_argument("--device", default=None, help="cuda / mps / cpu (auto-detected if not set)")

    return p.parse_args()


# ======================================================================
# LR schedule: linear warmup + cosine decay
# ======================================================================

def build_scheduler(
    optimizer: AdamW,
    warmup_steps: int,
    total_steps: int,
) -> LambdaLR:
    """Linear warmup then cosine decay to 0."""
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return LambdaLR(optimizer, lr_lambda)


# ======================================================================
# Metrics helpers
# ======================================================================

def masked_rmse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> float:
    """RMSE over valid (mask==1) pixels. Returns Python float."""
    valid = mask.bool()
    if not valid.any():
        return float("nan")
    return (pred[valid] - target[valid]).pow(2).mean().sqrt().item()


def routing_entropy(routing_weights: torch.Tensor) -> float:
    """Shannon entropy of mean routing distribution. Max = log(n_experts)."""
    mean_w = routing_weights.mean(dim=0)
    return -(mean_w * (mean_w + 1e-8).log()).sum().item()


# ======================================================================
# One epoch
# ======================================================================

def run_epoch(
    model: MARASSModel,
    loader,
    criterion: MARASSLoss,
    optimizer: AdamW | None,
    scheduler: LambdaLR | None,
    scaler: GradScaler | None,
    device: torch.device,
    global_step: int,
    total_steps: int,
    grad_clip: float,
    writer: SummaryWriter | None,
    phase: str,
    use_amp: bool,
) -> tuple[dict[str, float], int]:
    """
    Run one full epoch.

    Returns:
        metrics:     dict of averaged scalar metrics for this epoch
        global_step: updated step counter (only increments during train)
    """
    is_train = (phase == "train")
    model.train(is_train)

    # AMP only on CUDA; autocast device string must match
    amp_device = device.type if device.type in ("cuda", "cpu") else "cpu"
    amp_enabled = use_amp and (device.type == "cuda")

    totals: dict[str, float] = {}
    n_batches = 0
    t0 = time.time()

    grad_ctx = torch.enable_grad() if is_train else torch.no_grad()

    with grad_ctx:
        for batch in loader:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

            # Forward under autocast
            with autocast(device_type=amp_device, enabled=amp_enabled):
                outputs = model(batch)
                loss, breakdown = criterion(
                    outputs, batch,
                    step=global_step if is_train else None,
                    total_steps=total_steps,
                )

            if is_train:
                optimizer.zero_grad(set_to_none=True)

                if amp_enabled and scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    grad_norm = nn.utils.clip_grad_norm_(
                        model.parameters(), max_norm=grad_clip
                    ).item()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    grad_norm = nn.utils.clip_grad_norm_(
                        model.parameters(), max_norm=grad_clip
                    ).item()
                    optimizer.step()

                scheduler.step()
                global_step += 1

                if writer:
                    writer.add_scalar("train/loss",              breakdown["total"],    global_step)
                    writer.add_scalar("train/recon",             breakdown["recon"],    global_step)
                    writer.add_scalar("train/forecast",          breakdown["forecast"], global_step)
                    writer.add_scalar("train/eri",               breakdown["eri"],      global_step)
                    writer.add_scalar("train/aux",               breakdown["aux"],      global_step)
                    writer.add_scalar("train/grad_norm",         grad_norm,             global_step)
                    writer.add_scalar("train/lr",                scheduler.get_last_lr()[0], global_step)
                    writer.add_scalar("train/curriculum_scale",  breakdown["curriculum_scale"], global_step)
                    if amp_enabled and scaler is not None:
                        writer.add_scalar("train/amp_scale", scaler.get_scale(), global_step)
                    if "routing_weights" in outputs:
                        ent = routing_entropy(outputs["routing_weights"].detach())
                        writer.add_scalar("train/routing_entropy", ent, global_step)

            for k, v in breakdown.items():
                totals[k] = totals.get(k, 0.0) + v

            # RMSE on observed pixels (cast to float32 for stable metrics)
            with torch.no_grad():
                last_obs_mask = batch["obs_mask"][:, -1]
                last_chl      = batch["chl_obs"][:, -1]
                pred_recon    = outputs["recon"].squeeze(1).float()
                rmse = masked_rmse(pred_recon, last_chl.float(), last_obs_mask)
                totals["recon_rmse"] = totals.get("recon_rmse", 0.0) + rmse

            n_batches += 1

    elapsed = time.time() - t0
    metrics = {k: v / n_batches for k, v in totals.items()}
    metrics["epoch_time_s"] = elapsed

    return metrics, global_step


# ======================================================================
# Checkpoint helpers
# ======================================================================

def save_checkpoint(
    path: Path,
    model: MARASSModel,
    optimizer: AdamW,
    scheduler: LambdaLR,
    scaler: GradScaler | None,
    epoch: int,
    global_step: int,
    val_loss: float,
) -> None:
    ckpt = {
        "epoch":       epoch,
        "global_step": global_step,
        "val_loss":    val_loss,
        "model":       model.state_dict(),
        "optimizer":   optimizer.state_dict(),
        "scheduler":   scheduler.state_dict(),
    }
    if scaler is not None:
        ckpt["scaler"] = scaler.state_dict()
    torch.save(ckpt, path)
    log.info(f"Saved checkpoint: {path}")


def load_checkpoint(
    path: Path,
    model: MARASSModel,
    optimizer: AdamW,
    scheduler: LambdaLR,
    scaler: GradScaler | None,
    device: torch.device,
) -> tuple[int, int, float]:
    """Returns (start_epoch, global_step, best_val_loss)."""
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    if scaler is not None and "scaler" in ckpt:
        scaler.load_state_dict(ckpt["scaler"])
    log.info(
        f"Resumed from {path} "
        f"(epoch {ckpt['epoch']}, step {ckpt['global_step']}, "
        f"val_loss {ckpt['val_loss']:.4f})"
    )
    return ckpt["epoch"] + 1, ckpt["global_step"], ckpt["val_loss"]


# ======================================================================
# Main
# ======================================================================

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    args = get_args()

    # --- Device ---
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    use_amp = not args.no_amp and device.type == "cuda"
    log.info(f"Device: {device}  |  AMP: {'enabled' if use_amp else 'disabled'}")

    # --- Directories ---
    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)

    # --- Data ---
    loaders = build_dataloaders(
        patch_dir=args.patch_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    steps_per_epoch = len(loaders["train"])
    total_steps     = steps_per_epoch * args.epochs
    warmup_steps    = steps_per_epoch * args.warmup_epochs
    log.info(
        f"Data: {steps_per_epoch} batches/epoch × {args.epochs} epochs "
        f"= {total_steps} steps  ({warmup_steps} warmup)"
    )

    # --- Model ---
    cfg   = ModelConfig()
    model = MARASSModel(cfg).to(device)
    log.info(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # --- Optimiser (selective weight decay) ---
    decay_params = [
        p for n, p in model.named_parameters()
        if p.requires_grad and "bias" not in n and "norm" not in n and "bn" not in n
    ]
    no_decay_params = [
        p for n, p in model.named_parameters()
        if p.requires_grad and ("bias" in n or "norm" in n or "bn" in n)
    ]
    optimizer = AdamW([
        {"params": decay_params,    "weight_decay": args.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ], lr=args.lr)

    scheduler = build_scheduler(optimizer, warmup_steps, total_steps)

    # --- AMP scaler (no-op when AMP disabled) ---
    scaler = GradScaler(device="cuda") if use_amp else None

    # --- Loss ---
    criterion = MARASSLoss(
        weights=LossWeights(
            recon=args.w_recon,
            forecast=args.w_forecast,
            eri=args.w_eri,
            aux=args.w_aux,
        )
    ).to(device)

    # --- Resume ---
    start_epoch   = 0
    global_step   = 0
    best_val_loss = float("inf")

    if args.resume:
        start_epoch, global_step, best_val_loss = load_checkpoint(
            Path(args.resume), model, optimizer, scheduler, scaler, device
        )

    # --- TensorBoard ---
    writer = SummaryWriter(log_dir=args.log_dir)

    # --- Log VRAM baseline ---
    if device.type == "cuda":
        allocated = torch.cuda.memory_allocated(device) / 1024**3
        reserved  = torch.cuda.memory_reserved(device)  / 1024**3
        log.info(f"VRAM at start: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")

    # --- Training loop ---
    log.info(f"Starting training: epochs {start_epoch}–{args.epochs - 1}")

    for epoch in range(start_epoch, args.epochs):
        train_metrics, global_step = run_epoch(
            model=model, loader=loaders["train"],
            criterion=criterion, optimizer=optimizer,
            scheduler=scheduler, scaler=scaler,
            device=device, global_step=global_step,
            total_steps=total_steps, grad_clip=args.grad_clip,
            writer=writer, phase="train", use_amp=use_amp,
        )

        val_metrics, _ = run_epoch(
            model=model, loader=loaders["val"],
            criterion=criterion, optimizer=None,
            scheduler=None, scaler=None,
            device=device, global_step=global_step,
            total_steps=total_steps, grad_clip=args.grad_clip,
            writer=None, phase="val", use_amp=use_amp,
        )

        val_loss = val_metrics["total"]

        writer.add_scalar("val/loss",       val_metrics["total"],      epoch)
        writer.add_scalar("val/recon",      val_metrics["recon"],      epoch)
        writer.add_scalar("val/forecast",   val_metrics["forecast"],   epoch)
        writer.add_scalar("val/eri",        val_metrics["eri"],        epoch)
        writer.add_scalar("val/recon_rmse", val_metrics["recon_rmse"], epoch)

        vram_str = ""
        if device.type == "cuda":
            vram_gb = torch.cuda.max_memory_allocated(device) / 1024**3
            torch.cuda.reset_peak_memory_stats(device)
            vram_str = f"  VRAM {vram_gb:.1f}GB"

        log.info(
            f"Epoch {epoch:03d} | "
            f"train {train_metrics['total']:.4f} "
            f"(R {train_metrics['recon']:.4f} "
            f"F {train_metrics['forecast']:.4f} "
            f"E {train_metrics['eri']:.4f}) | "
            f"val {val_loss:.4f} rmse {val_metrics['recon_rmse']:.4f} | "
            f"{train_metrics['epoch_time_s']:.0f}s{vram_str}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                ckpt_dir / "best.pt", model, optimizer,
                scheduler, scaler, epoch, global_step, best_val_loss,
            )
            log.info(f"  → New best val loss: {best_val_loss:.4f}")

        save_checkpoint(
            ckpt_dir / "last.pt", model, optimizer,
            scheduler, scaler, epoch, global_step, val_loss,
        )

        if (epoch + 1) % args.save_every == 0:
            save_checkpoint(
                ckpt_dir / f"epoch_{epoch:03d}.pt", model, optimizer,
                scheduler, scaler, epoch, global_step, val_loss,
            )

    writer.close()
    log.info(f"Training complete. Best val loss: {best_val_loss:.4f}")
    log.info(f"Best checkpoint: {ckpt_dir / 'best.pt'}")


if __name__ == "__main__":
    main()