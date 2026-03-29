"""
Train.py — MM-MARAS training loop

Features:
    - Multi-GPU training via DDP (torchrun) — auto-detected, no flag needed
    - Single-GPU fallback when launched with plain python
    - Mixed precision training (torch.cuda.amp) — halves VRAM usage
    - Train / val loop with per-epoch metrics
    - Curriculum scheduling (forecast + ERI ramp over first 20% of steps)
    - AdamW optimiser + cosine LR schedule with linear warmup
    - Gradient clipping (max norm 1.0)
    - Checkpoint saving: best val loss + periodic every N epochs
    - TensorBoard logging on rank 0 only
    - Resume from checkpoint
    - DistributedSampler with per-epoch shuffle for DDP correctness

Usage:
    # Single GPU
    python Train.py --patch-dir data/patches

    # Multi-GPU (2 GPUs) — Kaggle T4 x2
    torchrun --nproc_per_node=2 Train.py --patch-dir data/patches --batch-size 8

    # Resume
    torchrun --nproc_per_node=2 Train.py --resume checkpoints/last.pt

    # CPU debug
    python Train.py --device cpu --batch-size 2 --num-workers 0

DDP note:
    dataset.py must export a MARASSDataset class that accepts patch_dir and
    a split argument ("train" / "val" / "test"). build_dataloaders() is used
    for single-GPU runs. For DDP, Train.py builds loaders directly so it can
    attach a DistributedSampler.

    If your MARASSDataset has a different constructor signature, update
    _build_ddp_loaders() below accordingly.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import math
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Sampler
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter

from augment import augment_batch
from dataset import build_dataloaders, MARASSDataset
from loss import MARASSLoss, LossWeights
from model import MARASSModel, ModelConfig

log = logging.getLogger(__name__)


# ======================================================================
# DDP helpers
# ======================================================================

def is_ddp_run() -> bool:
    """True when launched via torchrun (LOCAL_RANK env var is set)."""
    return "LOCAL_RANK" in os.environ


def ddp_setup() -> tuple[int, int, torch.device]:
    """
    Initialise the NCCL process group and return (local_rank, world_size, device).
    Call once at the start of main().
    """
    local_rank  = int(os.environ["LOCAL_RANK"])
    world_size  = int(os.environ.get("WORLD_SIZE", 1))
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    return local_rank, world_size, device


def ddp_cleanup() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def reduce_sum_tensor(tensor: torch.Tensor, world_size: int) -> torch.Tensor:
    """Sum-reduce a tensor across DDP ranks."""
    if world_size == 1:
        return tensor
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor


def reduce_max_tensor(tensor: torch.Tensor, world_size: int) -> torch.Tensor:
    """Max-reduce a tensor across DDP ranks."""
    if world_size == 1:
        return tensor
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return tensor


def unwrap_model(model: nn.Module) -> MARASSModel:
    """Strip DDP wrapper to access the underlying MARASSModel."""
    return model.module if isinstance(model, DDP) else model


class DistributedEvalSampler(Sampler[int]):
    """Shard eval datasets across ranks without padding or duplicated samples."""

    def __init__(self, dataset: MARASSDataset, rank: int, world_size: int) -> None:
        self.start = (len(dataset) * rank) // world_size
        self.end = (len(dataset) * (rank + 1)) // world_size

    def __iter__(self):
        return iter(range(self.start, self.end))

    def __len__(self) -> int:
        return self.end - self.start


# ======================================================================
# DataLoader builders
# ======================================================================

def _build_ddp_loaders(
    patch_dir: str,
    batch_size: int,
    num_workers: int,
    rank: int,
    world_size: int,
) -> dict[str, DataLoader]:
    """
    Build train/val/test loaders with distributed sharding for DDP.

    Train loader uses a DistributedSampler (shuffle handled per-epoch via
    sampler.set_epoch). Val/test loaders use a non-padding sampler so
    every rank evaluates a non-overlapping shard — metrics are then averaged
    across ranks with weighted metric reduction.
    """
    loaders = {}
    for split in ("train", "val", "test"):
        dataset = MARASSDataset(patch_dir=patch_dir, split=split)
        if split == "train":
            sampler = DistributedSampler(
                dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=True,
                drop_last=True,
            )
        else:
            sampler = DistributedEvalSampler(dataset, rank=rank, world_size=world_size)
        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=(num_workers > 0),
        )
    return loaders


# ======================================================================
# Args
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
    p.add_argument("--batch-size",    type=int,   default=4,
                   help="Per-GPU batch size. With 2xT4 and AMP, 8 fits comfortably.")
    p.add_argument("--lr",            type=float, default=1e-4)
    p.add_argument("--weight-decay",  type=float, default=1e-2)
    p.add_argument("--grad-clip",     type=float, default=1.0)
    p.add_argument("--warmup-epochs", type=int,   default=5)
    p.add_argument("--save-every",    type=int,   default=10)
    p.add_argument("--no-amp",        action="store_true",
                   help="Disable mixed precision (use if NaN losses appear)")

    # Loss weights
    p.add_argument("--w-recon",    type=float, default=1.0)
    p.add_argument("--w-forecast", type=float, default=0.5)
    p.add_argument("--w-eri",      type=float, default=0.3)
    p.add_argument("--w-aux",      type=float, default=0.01)
    p.add_argument("--w-holdout",  type=float, default=0.5)

    # DataLoader
    p.add_argument("--num-workers", type=int, default=2)

    # Hardware (ignored when running under torchrun — rank determines device)
    p.add_argument("--device", default=None, help="cuda / mps / cpu (single-GPU only)")

    return p.parse_args()


# ======================================================================
# LR schedule: linear warmup + cosine decay
# ======================================================================

def build_scheduler(optimizer: AdamW, warmup_steps: int, total_steps: int) -> LambdaLR:
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return LambdaLR(optimizer, lr_lambda)


# ======================================================================
# Metrics helpers
# ======================================================================

def routing_entropy(routing_weights: torch.Tensor) -> float:
    mean_w = routing_weights if routing_weights.ndim == 1 else routing_weights.mean(dim=0)
    return -(mean_w * (mean_w + 1e-8).log()).sum().item()


def stable_holdout_mask(
    obs_mask: torch.Tensor,
    land_mask: torch.Tensor,
    holdout_frac: float,
) -> torch.Tensor:
    """Build a deterministic validation holdout mask from observed ocean pixels."""
    if holdout_frac <= 0:
        return torch.zeros_like(obs_mask)

    obs_cpu = obs_mask.detach().to(device="cpu", dtype=torch.float32).contiguous()
    digest = hashlib.sha1(obs_cpu.numpy().tobytes()).digest()
    seed = int.from_bytes(digest[:8], byteorder="little", signed=False)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    rand = torch.rand(obs_mask.shape, generator=generator).to(obs_mask.device)
    ocean = 1.0 - land_mask
    return ((obs_mask > 0.5) & (ocean > 0.5) & (rand < holdout_frac)).float()


def compute_masked_rmse_stats(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[float, float]:
    """Return (sum_squared_error, count) over a binary mask."""
    valid = mask.bool()
    if not valid.any():
        return 0.0, 0.0
    diff = pred[valid].float() - target[valid].float()
    return diff.pow(2).sum().item(), float(valid.sum().item())


def build_gap_eval_batch(
    batch: dict[str, torch.Tensor],
    holdout_mask: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Zero held-out last-step inputs for synthetic validation gap evaluation."""
    eval_batch = dict(batch)
    eval_batch["chl_obs"] = batch["chl_obs"].clone()
    eval_batch["obs_mask"] = batch["obs_mask"].clone()
    eval_batch["chl_obs"][:, -1] *= (1.0 - holdout_mask)
    eval_batch["obs_mask"][:, -1] *= (1.0 - holdout_mask)
    return eval_batch


# ======================================================================
# One epoch
# ======================================================================

def run_epoch(
    model: nn.Module,
    loader: DataLoader,
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
    world_size: int = 1,
    is_main: bool = True,
) -> tuple[dict[str, float], int]:
    is_train = (phase == "train")
    amp_device = "cuda" if device.type == "cuda" else "cpu"
    amp_enabled = use_amp and (device.type == "cuda")

    model.train(is_train)

    metric_keys = ("aux", "bloom_fcast", "curriculum_scale", "eri", "forecast", "holdout", "recon", "total")
    totals = {k: 0.0 for k in metric_keys}
    model_cfg = unwrap_model(model).cfg
    n_examples = 0.0
    gap_sse = 0.0
    gap_count = 0.0
    routing_weight_sum = torch.zeros(model_cfg.n_experts, dtype=torch.float64, device=device)
    routing_count = 0.0
    t0 = time.time()

    grad_ctx = torch.enable_grad() if is_train else torch.no_grad()

    with grad_ctx:
        for batch in loader:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            batch_size = batch["chl_obs"].shape[0]

            # Spatial augmentation (flips + 90° rotations) — training only
            if is_train:
                batch = augment_batch(batch)

            # zero_grad before forward so step ordering is unambiguous
            if is_train:
                optimizer.zero_grad(set_to_none=True)

            with autocast(device_type=amp_device, enabled=amp_enabled):
                outputs = model(batch)
                loss, breakdown = criterion(
                    outputs, batch,
                    step=global_step if is_train else None,
                    total_steps=total_steps,
                )

            if is_train:
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

                # TensorBoard — rank 0 only
                if writer and is_main:
                    writer.add_scalar("train/loss",             breakdown["total"],    global_step)
                    writer.add_scalar("train/recon",            breakdown["recon"],    global_step)
                    writer.add_scalar("train/forecast",         breakdown["forecast"], global_step)
                    writer.add_scalar("train/eri",              breakdown["eri"],      global_step)
                    writer.add_scalar("train/aux",              breakdown["aux"],      global_step)
                    writer.add_scalar("train/grad_norm",        grad_norm,             global_step)
                    writer.add_scalar("train/lr",               scheduler.get_last_lr()[0], global_step)
                    writer.add_scalar("train/curriculum_scale", breakdown["curriculum_scale"], global_step)
                    writer.add_scalar("train/holdout",           breakdown.get("holdout", 0.0),  global_step)
                    writer.add_scalar("train/bloom_fcast",       breakdown.get("bloom_fcast", 0.0), global_step)
                    if amp_enabled and scaler is not None:
                        writer.add_scalar("train/amp_scale", scaler.get_scale(), global_step)

            for k, v in breakdown.items():
                totals[k] = totals.get(k, 0.0) + (v * batch_size)
            n_examples += batch_size

            with torch.no_grad():
                last_chl = batch["chl_obs"][:, -1]
                pred_recon = outputs["recon"].squeeze(1).float()

                if is_train and "holdout_mask" in outputs:
                    sse, count = compute_masked_rmse_stats(
                        pred_recon, last_chl, outputs["holdout_mask"]
                    )
                    gap_sse += sse
                    gap_count += count
                elif not is_train:
                    holdout_mask = stable_holdout_mask(
                        batch["obs_mask"][:, -1],
                        batch["land_mask"],
                        model_cfg.holdout_frac,
                    )
                    if holdout_mask.any():
                        eval_batch = build_gap_eval_batch(batch, holdout_mask)
                        with autocast(device_type=amp_device, enabled=amp_enabled):
                            gap_outputs = model(eval_batch)
                        gap_pred = gap_outputs["recon"].squeeze(1).float()
                        sse, count = compute_masked_rmse_stats(gap_pred, last_chl, holdout_mask)
                        gap_sse += sse
                        gap_count += count

                if "routing_weights" in outputs:
                    batch_routing_sum = outputs["routing_weights"].detach().sum(dim=0, dtype=torch.float64)
                    routing_weight_sum += batch_routing_sum
                    routing_count += batch_size

    elapsed = time.time() - t0
    keys = list(metric_keys)
    totals_tensor = torch.tensor([totals[k] for k in keys], dtype=torch.float64, device=device)
    totals_tensor = reduce_sum_tensor(totals_tensor, world_size)

    counts_tensor = torch.tensor(
        [n_examples, gap_sse, gap_count, routing_count],
        dtype=torch.float64,
        device=device,
    )
    counts_tensor = reduce_sum_tensor(counts_tensor, world_size)
    n_examples, gap_sse, gap_count, routing_count = counts_tensor.tolist()

    elapsed_tensor = torch.tensor(elapsed, dtype=torch.float64, device=device)
    elapsed_tensor = reduce_max_tensor(elapsed_tensor, world_size)

    metrics = {k: totals_tensor[i].item() / max(n_examples, 1.0) for i, k in enumerate(keys)}
    metrics["epoch_time_s"] = elapsed_tensor.item()
    metrics["gap_rmse"] = math.sqrt(gap_sse / gap_count) if gap_count > 0 else float("nan")

    routing_weight_sum = reduce_sum_tensor(routing_weight_sum, world_size)
    if routing_count > 0:
        metrics["routing_entropy"] = routing_entropy((routing_weight_sum / routing_count).float())

    return metrics, global_step


# ======================================================================
# Checkpoint helpers
# ======================================================================

def save_checkpoint(
    path: Path,
    model: nn.Module,
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
        "model":       unwrap_model(model).state_dict(),   # strip DDP wrapper
        "optimizer":   optimizer.state_dict(),
        "scheduler":   scheduler.state_dict(),
    }
    if scaler is not None:
        ckpt["scaler"] = scaler.state_dict()
    torch.save(ckpt, path)
    log.info(f"Saved checkpoint: {path}")


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: AdamW,
    scheduler: LambdaLR,
    scaler: GradScaler | None,
    device: torch.device,
) -> tuple[int, int, float]:
    """Returns (start_epoch, global_step, best_val_loss)."""
    ckpt = torch.load(path, map_location=device)
    unwrap_model(model).load_state_dict(ckpt["model"])
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
    args = get_args()

    # ------------------------------------------------------------------
    # DDP vs single-GPU setup
    # ------------------------------------------------------------------
    using_ddp = is_ddp_run()

    if using_ddp:
        local_rank, world_size, device = ddp_setup()
        is_main = (local_rank == 0)
    else:
        local_rank  = 0
        world_size  = 1
        is_main     = True
        if args.device:
            device = torch.device(args.device)
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

    # Only rank 0 configures logging — avoids duplicate lines with 2 GPUs
    if is_main:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s  %(levelname)s  %(message)s",
            datefmt="%H:%M:%S",
        )

    use_amp = not args.no_amp and device.type == "cuda"

    if is_main:
        mode_str = f"DDP world_size={world_size}" if using_ddp else "single-GPU"
        log.info(f"Device: {device}  |  Mode: {mode_str}  |  AMP: {'enabled' if use_amp else 'disabled'}")

    # ------------------------------------------------------------------
    # Directories (rank 0 only creates them)
    # ------------------------------------------------------------------
    ckpt_dir = Path(args.ckpt_dir)
    if is_main:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        Path(args.log_dir).mkdir(parents=True, exist_ok=True)

    # Barrier: make sure rank 0 has created dirs before other ranks proceed
    if using_ddp:
        dist.barrier()

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    if using_ddp:
        loaders = _build_ddp_loaders(
            patch_dir=args.patch_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            rank=local_rank,
            world_size=world_size,
        )
        # steps_per_epoch is the shard size seen by each rank
        steps_per_epoch = len(loaders["train"])
        # total_steps is global — multiply up by world_size for scheduler
        total_steps  = steps_per_epoch * world_size * args.epochs
        warmup_steps = steps_per_epoch * world_size * args.warmup_epochs
    else:
        loaders = build_dataloaders(
            patch_dir=args.patch_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
        )
        steps_per_epoch = len(loaders["train"])
        total_steps     = steps_per_epoch * args.epochs
        warmup_steps    = steps_per_epoch * args.warmup_epochs

    if is_main:
        log.info(
            f"Data: {steps_per_epoch} batches/rank/epoch x "
            f"{world_size} rank(s) x {args.epochs} epochs "
            f"= {total_steps} steps  ({warmup_steps} warmup)"
        )

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    cfg   = ModelConfig()
    model = MARASSModel(cfg).to(device)

    if using_ddp:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    if is_main:
        log.info(f"Parameters: {sum(p.numel() for p in unwrap_model(model).parameters()):,}")

    # ------------------------------------------------------------------
    # Optimiser (selective weight decay — skip bias / norm params)
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # AMP scaler
    # ------------------------------------------------------------------
    scaler = GradScaler(device="cuda") if use_amp else None

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------
    criterion = MARASSLoss(
        weights=LossWeights(
            recon=args.w_recon,
            forecast=args.w_forecast,
            eri=args.w_eri,
            bloom_fcast=0.3,
            aux=args.w_aux,
            holdout=args.w_holdout,
        ),
        bloom_threshold=2.5,
    ).to(device)

    # ------------------------------------------------------------------
    # Resume
    # ------------------------------------------------------------------
    start_epoch   = 0
    global_step   = 0
    best_val_loss = float("inf")

    if args.resume:
        start_epoch, global_step, best_val_loss = load_checkpoint(
            Path(args.resume), model, optimizer, scheduler, scaler, device
        )

    # ------------------------------------------------------------------
    # TensorBoard (rank 0 only)
    # ------------------------------------------------------------------
    writer = SummaryWriter(log_dir=args.log_dir) if is_main else None

    # ------------------------------------------------------------------
    # VRAM baseline
    # ------------------------------------------------------------------
    if is_main and device.type == "cuda":
        allocated = torch.cuda.memory_allocated(device) / 1024**3
        reserved  = torch.cuda.memory_reserved(device)  / 1024**3
        log.info(f"VRAM at start: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    if is_main:
        log.info(f"Starting training: epochs {start_epoch}-{args.epochs - 1}")

    for epoch in range(start_epoch, args.epochs):

        # DistributedSampler must know the epoch for correct per-epoch shuffle
        if using_ddp:
            loaders["train"].sampler.set_epoch(epoch)

        train_metrics, global_step = run_epoch(
            model=model, loader=loaders["train"],
            criterion=criterion, optimizer=optimizer,
            scheduler=scheduler, scaler=scaler,
            device=device, global_step=global_step,
            total_steps=total_steps, grad_clip=args.grad_clip,
            writer=writer, phase="train", use_amp=use_amp,
            world_size=world_size, is_main=is_main,
        )

        val_metrics, _ = run_epoch(
            model=model, loader=loaders["val"],
            criterion=criterion, optimizer=None,
            scheduler=None, scaler=None,
            device=device, global_step=global_step,
            total_steps=total_steps, grad_clip=args.grad_clip,
            writer=None, phase="val", use_amp=use_amp,
            world_size=world_size, is_main=is_main,
        )

        val_loss = val_metrics["total"]

        # All operations below are rank 0 only
        if is_main:
            if writer:
                writer.add_scalar("val/loss",       val_metrics["total"],      epoch)
                writer.add_scalar("val/recon",      val_metrics["recon"],      epoch)
                writer.add_scalar("val/forecast",   val_metrics["forecast"],   epoch)
                writer.add_scalar("val/eri",        val_metrics["eri"],        epoch)
                writer.add_scalar("val/bloom_fcast", val_metrics.get("bloom_fcast", 0.0), epoch)
                writer.add_scalar("train_epoch/routing_entropy", train_metrics["routing_entropy"], epoch)
                writer.add_scalar("train_epoch/gap_rmse",        train_metrics["gap_rmse"],        epoch)
                writer.add_scalar("val/gap_rmse",                val_metrics["gap_rmse"],          epoch)

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
                f"E {train_metrics['eri']:.4f} "
                f"B {train_metrics.get('bloom_fcast', 0.0):.4f}) | "
                f"val {val_loss:.4f} gap_rmse {val_metrics['gap_rmse']:.4f} | "
                f"{train_metrics['epoch_time_s']:.0f}s{vram_str}"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(
                    ckpt_dir / "best.pt", model, optimizer,
                    scheduler, scaler, epoch, global_step, best_val_loss,
                )
                log.info(f"  -> New best val loss: {best_val_loss:.4f}")

            save_checkpoint(
                ckpt_dir / "last.pt", model, optimizer,
                scheduler, scaler, epoch, global_step, val_loss,
            )

            if (epoch + 1) % args.save_every == 0:
                save_checkpoint(
                    ckpt_dir / f"epoch_{epoch:03d}.pt", model, optimizer,
                    scheduler, scaler, epoch, global_step, val_loss,
                )

        # All ranks sync before the next epoch
        if using_ddp:
            dist.barrier()

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------
    if writer:
        writer.close()

    ddp_cleanup()

    if is_main:
        log.info(f"Training complete. Best val loss: {best_val_loss:.4f}")
        log.info(f"Best checkpoint: {ckpt_dir / 'best.pt'}")


if __name__ == "__main__":
    main()