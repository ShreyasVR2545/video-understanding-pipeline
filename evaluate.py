"""
Evaluation Module

Computes Top-1 and Top-5 accuracy on the HMDB51 test split.
Also supports zero-shot evaluation for baseline comparison.

Metrics reported:
  - Top-1 Accuracy: correct class is the model's highest-confidence prediction
  - Top-5 Accuracy: correct class is among the 5 highest-confidence predictions
"""

from typing import Dict

import torch
from torch.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import PipelineConfig


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    config: PipelineConfig,
) -> Dict[str, float]:
    """
    Evaluate model on a dataset split.

    Uses mixed precision inference for consistency with training.
    Accumulates predictions across all batches before computing
    final metrics to avoid averaging-of-averages bias.

    Args:
        model: VideoActionClassifier in eval mode
        dataloader: Test/val DataLoader
        device: Compute device
        config: Pipeline config for AMP settings

    Returns:
        dict with "top1", "top5" (percentages), "num_samples"
    """
    model.eval()

    correct_top1 = 0
    correct_top5 = 0
    total = 0

    for batch in dataloader:
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)

        with autocast(device_type="cuda", enabled=config.training.use_amp):
            outputs = model(pixel_values=pixel_values)

        logits = outputs["logits"]  # (B, num_classes)

        # Top-1: highest logit matches label
        _, pred_top1 = logits.topk(1, dim=1)
        correct_top1 += (pred_top1.squeeze(1) == labels).sum().item()

        # Top-5: label is among 5 highest logits
        _, pred_top5 = logits.topk(min(5, logits.size(1)), dim=1)
        correct_top5 += sum(
            labels[i].item() in pred_top5[i].tolist()
            for i in range(labels.size(0))
        )

        total += labels.size(0)

    top1_acc = 100.0 * correct_top1 / max(total, 1)
    top5_acc = 100.0 * correct_top5 / max(total, 1)

    return {
        "top1": top1_acc,
        "top5": top5_acc,
        "num_samples": total,
    }


@torch.no_grad()
def evaluate_zero_shot(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """
    Evaluate zero-shot performance (random head, pretrained backbone).

    This establishes a baseline: how well does the pretrained VideoMAE
    backbone perform with a randomly initialized classification head?
    The expected Top-1 for 51 classes is ~2% (random chance), but the
    backbone's features may push it slightly higher.

    Used in the README results table to show fine-tuning improvement.
    """
    model.eval()

    correct_top1 = 0
    correct_top5 = 0
    total = 0

    for batch in dataloader:
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(pixel_values=pixel_values)
        logits = outputs["logits"]

        _, pred_top1 = logits.topk(1, dim=1)
        correct_top1 += (pred_top1.squeeze(1) == labels).sum().item()

        _, pred_top5 = logits.topk(min(5, logits.size(1)), dim=1)
        correct_top5 += sum(
            labels[i].item() in pred_top5[i].tolist()
            for i in range(labels.size(0))
        )

        total += labels.size(0)

    return {
        "top1": 100.0 * correct_top1 / max(total, 1),
        "top5": 100.0 * correct_top5 / max(total, 1),
        "num_samples": total,
    }


def compute_per_class_accuracy(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    class_names: list,
) -> Dict[str, float]:
    """
    Per-class Top-1 accuracy for error analysis.

    Useful for identifying which action classes the model struggles with
    (e.g., visually similar actions like "drink" vs "eat").
    """
    model.eval()

    class_correct = {name: 0 for name in class_names}
    class_total = {name: 0 for name in class_names}

    with torch.no_grad():
        for batch in dataloader:
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(pixel_values=pixel_values)
            _, preds = outputs["logits"].topk(1, dim=1)
            preds = preds.squeeze(1)

            for i in range(labels.size(0)):
                label_idx = labels[i].item()
                cls_name = class_names[label_idx]
                class_total[cls_name] += 1
                if preds[i].item() == label_idx:
                    class_correct[cls_name] += 1

    per_class = {}
    for name in class_names:
        if class_total[name] > 0:
            per_class[name] = 100.0 * class_correct[name] / class_total[name]
        else:
            per_class[name] = 0.0

    return per_class
