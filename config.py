"""
Centralized configuration for the Video Understanding Pipeline.

All hyperparameters, paths, and model settings live here so experiments
are reproducible and easy to modify from a single location.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass
class ModelConfig:
    """VideoMAE backbone and classification head settings."""

    backbone: str = "MCG-NJU/videomae-base"
    hidden_size: int = 768          # VideoMAE-Base hidden dimension
    num_classes: int = 51           # HMDB51 action classes
    num_frames: int = 16            # Temporal resolution for VideoMAE
    image_size: int = 224           # Spatial resolution (H = W)
    tubelet_size: int = 2           # VideoMAE temporal patch size
    drop_rate: float = 0.1         # Dropout in classification head


@dataclass
class TrainingConfig:
    """Training loop hyperparameters following the project spec."""

    # Optimizer
    learning_rate: float = 1e-4
    weight_decay: float = 0.05
    betas: tuple = (0.9, 0.999)

    # Schedule
    total_epochs: int = 20
    warmup_epochs: int = 5          # Backbone frozen during warmup
    batch_size: int = 8
    num_workers: int = 0            # Set to 4 on Linux; 0 avoids Windows multiprocessing issues

    # Mixed precision (signals scale awareness to reviewers)
    use_amp: bool = True

    # Learning rate schedule
    lr_scheduler: str = "cosine"    # Cosine annealing after warmup
    min_lr: float = 1e-6

    # Reproducibility
    seed: int = 42

    # Gradient clipping for stability during unfreezing
    max_grad_norm: float = 1.0


@dataclass
class DataConfig:
    """Dataset paths and preprocessing settings."""

    dataset_name: str = "HMDB51"
    data_root: Path = Path("data/hmdb51")
    split: int = 1                  # HMDB51 has 3 official splits; use split 1

    # Frame sampling
    num_frames: int = 16
    image_size: int = 224

    # Normalization (ImageNet stats — standard for HuggingFace vision models)
    mean: List[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    std: List[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])


@dataclass
class PipelineConfig:
    """Top-level config aggregating all sub-configs."""

    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)

    # Output paths
    checkpoint_dir: Path = Path("checkpoints")
    log_dir: Path = Path("logs")
    viz_dir: Path = Path("visualizations")

    def __post_init__(self):
        """Create output directories on initialization."""
        for d in [self.checkpoint_dir, self.log_dir, self.viz_dir]:
            d.mkdir(parents=True, exist_ok=True)
