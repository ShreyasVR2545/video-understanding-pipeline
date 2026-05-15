"""
Training Loop for Video Action Recognition

Implements two-phase training with mixed precision:
  Phase 1 (epochs 1–5):  Backbone frozen, only head trains
  Phase 2 (epochs 6–20): Full model fine-tunes with differential LR

Features:
  - torch.cuda.amp mixed precision (fp16) for memory efficiency
  - Cosine annealing LR schedule with linear warmup
  - Gradient clipping for stability during backbone unfreezing
  - Periodic checkpointing and metric logging
  - Top-1 and Top-5 accuracy evaluation

Usage:
    python train.py [--data_root DATA_PATH] [--epochs N] [--batch_size B]
"""

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from config import PipelineConfig
from dataset import HMDB51Dataset
from evaluate import evaluate_model
from model import VideoActionClassifier


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_steps: int,
    min_lr: float = 1e-6,
):
    """
    Cosine annealing with linear warmup.

    The warmup phase linearly increases LR from 0 to the target over
    `warmup_steps`. After warmup, cosine decay smoothly reduces LR
    to `min_lr`. This schedule is standard in vision transformer
    fine-tuning (MAE, DINO, etc.).
    """
    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            # Linear warmup
            return current_step / max(1, warmup_steps)
        else:
            # Cosine annealing
            progress = (current_step - warmup_steps) / max(
                1, total_steps - warmup_steps
            )
            return max(min_lr, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_one_epoch(
    model: VideoActionClassifier,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    scaler: GradScaler,
    device: torch.device,
    epoch: int,
    config: PipelineConfig,
) -> dict:
    """
    Train for one epoch with mixed precision.

    Returns:
        dict with "loss" (average), "lr", "time_sec"
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    start_time = time.time()

    for batch_idx, batch in enumerate(dataloader):
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()

        # ── Mixed precision forward pass ──
        # autocast runs matmuls in fp16 for ~2x speedup and ~50% memory reduction
        with autocast(device_type="cuda", enabled=config.training.use_amp):
            outputs = model(pixel_values=pixel_values, labels=labels)
            loss = outputs["loss"]

        # ── Scaled backward pass ──
        # GradScaler prevents fp16 underflow by dynamically scaling gradients
        scaler.scale(loss).backward()

        # Gradient clipping (important during backbone unfreezing)
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(
            model.parameters(), config.training.max_grad_norm
        )

        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        total_loss += loss.item()
        num_batches += 1

        # Progress logging
        if (batch_idx + 1) % 20 == 0 or (batch_idx + 1) == len(dataloader):
            avg_loss = total_loss / num_batches
            lr = optimizer.param_groups[-1]["lr"]  # Head LR
            print(
                f"  Epoch {epoch} [{batch_idx + 1}/{len(dataloader)}] "
                f"loss={avg_loss:.4f} lr={lr:.2e}"
            )

    elapsed = time.time() - start_time

    return {
        "loss": total_loss / max(num_batches, 1),
        "lr": optimizer.param_groups[-1]["lr"],
        "time_sec": elapsed,
    }


def train(config: PipelineConfig):
    """
    Full training pipeline with two-phase schedule.

    Phase 1 (epochs 1 to warmup_epochs):
        Backbone parameters are frozen. Only the classification head
        learns to map VideoMAE features → action classes. This prevents
        catastrophic forgetting of the pretrained representations.

    Phase 2 (epochs warmup_epochs+1 to total_epochs):
        All parameters are unfrozen. The backbone adapts its features
        to HMDB51's action domain with a 10x lower learning rate than
        the head (differential LR).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Train] Using device: {device}")

    if device.type == "cuda":
        print(f"[Train] GPU: {torch.cuda.get_device_name()}")
        print(f"[Train] VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ── Reproducibility ──
    torch.manual_seed(config.training.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.training.seed)

    # ── Dataset & DataLoader ──
    print("\n[Train] Loading HMDB51 dataset...")
    train_dataset = HMDB51Dataset(config.data, split="train", use_jitter=True)
    test_dataset = HMDB51Dataset(config.data, split="test", use_jitter=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.training.num_workers,
        pin_memory=True,
        drop_last=True,  # Avoid small last batch with BatchNorm
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
        pin_memory=True,
    )

    # ── Model ──
    print("\n[Train] Initializing VideoMAE classifier...")
    model = VideoActionClassifier(config.model).to(device)

    # Phase 1: Freeze backbone
    model.freeze_backbone()

    # ── Optimizer (differential LR applied in Phase 2) ──
    optimizer = torch.optim.AdamW(
        model.get_param_groups(
            lr=config.training.learning_rate,
            backbone_lr_scale=0.1,
        ),
        weight_decay=config.training.weight_decay,
        betas=config.training.betas,
    )

    # ── LR Scheduler ──
    total_steps = config.training.total_epochs * len(train_loader)
    warmup_steps = config.training.warmup_epochs * len(train_loader)
    scheduler = create_scheduler(
        optimizer, total_steps, warmup_steps, config.training.min_lr
    )

    # ── Mixed Precision Scaler ──
    scaler = GradScaler("cuda", enabled=config.training.use_amp)

    # ── Training History ──
    history = {
        "train_loss": [],
        "test_top1": [],
        "test_top5": [],
        "epoch_times": [],
    }
    best_top1 = 0.0

    # ══════════════════════════════════════════════════════════════
    # Training Loop
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"Starting training: {config.training.total_epochs} epochs")
    print(f"  Phase 1 (frozen backbone): epochs 1–{config.training.warmup_epochs}")
    print(f"  Phase 2 (full fine-tune):  epochs {config.training.warmup_epochs + 1}–{config.training.total_epochs}")
    print(f"  Mixed precision: {'ON' if config.training.use_amp else 'OFF'}")
    print(f"  Batch size: {config.training.batch_size}")
    print(f"{'='*60}\n")

    for epoch in range(1, config.training.total_epochs + 1):
        # ── Phase transition: unfreeze backbone after warmup ──
        if epoch == config.training.warmup_epochs + 1:
            print(f"\n{'─'*40}")
            print(f"PHASE 2: Unfreezing backbone at epoch {epoch}")
            print(f"{'─'*40}\n")
            model.unfreeze_backbone()

        # ── Train ──
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, scheduler, scaler,
            device, epoch, config,
        )

        # ── Evaluate ──
        eval_metrics = evaluate_model(model, test_loader, device, config)

        # ── Record history ──
        history["train_loss"].append(train_metrics["loss"])
        history["test_top1"].append(eval_metrics["top1"])
        history["test_top5"].append(eval_metrics["top5"])
        history["epoch_times"].append(train_metrics["time_sec"])

        # ── Epoch summary ──
        phase = "FROZEN" if epoch <= config.training.warmup_epochs else "FULL"
        print(
            f"\nEpoch {epoch}/{config.training.total_epochs} [{phase}] "
            f"loss={train_metrics['loss']:.4f} "
            f"top1={eval_metrics['top1']:.2f}% "
            f"top5={eval_metrics['top5']:.2f}% "
            f"({train_metrics['time_sec']:.0f}s)\n"
        )

        # ── Checkpointing (save best + latest) ──
        if eval_metrics["top1"] > best_top1:
            best_top1 = eval_metrics["top1"]
            save_checkpoint(
                model, optimizer, epoch, eval_metrics,
                config.checkpoint_dir / "best_model.pt",
            )
            print(f"  ★ New best model saved (top1={best_top1:.2f}%)")

        # Save latest every 5 epochs
        if epoch % 5 == 0:
            save_checkpoint(
                model, optimizer, epoch, eval_metrics,
                config.checkpoint_dir / f"checkpoint_epoch{epoch}.pt",
            )

    # ── Save training history ──
    with open(config.log_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"\n[Train] History saved to {config.log_dir / 'training_history.json'}")

    # ── Final summary ──
    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"  Best Top-1 accuracy: {best_top1:.2f}%")
    print(f"  Best Top-5 accuracy: {max(history['test_top5']):.2f}%")
    print(f"  Total training time: {sum(history['epoch_times']):.0f}s")
    print(f"{'='*60}")

    return model, history


def save_checkpoint(
    model: VideoActionClassifier,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict,
    path: Path,
) -> None:
    """Save model checkpoint with training state for resumption."""
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
        },
        path,
    )


def load_checkpoint(
    model: VideoActionClassifier,
    path: Path,
    device: torch.device,
) -> dict:
    """Load model checkpoint (inference only — no optimizer state)."""
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"[Checkpoint] Loaded from {path} (epoch {checkpoint['epoch']})")
    return checkpoint.get("metrics", {})


# ──────────────────────────────────────────────────────────────────────
# CLI Entry Point
# ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train VideoMAE on HMDB51")
    parser.add_argument("--data_root", type=str, default="data/hmdb51")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--no_amp", action="store_true", help="Disable mixed precision")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = PipelineConfig()
    config.data.data_root = Path(args.data_root)
    config.training.total_epochs = args.epochs
    config.training.batch_size = args.batch_size
    config.training.learning_rate = args.lr
    config.training.warmup_epochs = args.warmup_epochs
    config.training.use_amp = not args.no_amp
    config.training.seed = args.seed

    train(config)
