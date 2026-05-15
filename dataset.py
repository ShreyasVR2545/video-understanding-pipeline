"""
Frame Sampler & HMDB51 Dataset (Direct HuggingFace Download)

Handles the full video-to-tensor pipeline:
  1. Download HMDB51 zip from HuggingFace Hub via huggingface_hub
  2. Extract to class-organized folder structure
  3. Uniformly sample `num_frames` frames per clip with OpenCV
  4. Resize to 224×224 and normalize with ImageNet statistics
  5. Package as tensors compatible with VideoMAE's expected input shape

Design decisions:
  - Direct zip download via huggingface_hub (bypasses datasets lib issues)
  - Uniform segment-midpoint sampling for deterministic evaluation
  - Temporal jittering only during training for augmentation
  - Stratified 80/20 train/test split with fixed seed
"""

import os
import random
import shutil
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import VideoMAEImageProcessor

from config import DataConfig


# ──────────────────────────────────────────────────────────────────────
# HMDB51 Label Mapping (alphabetical — 51 action classes)
# ──────────────────────────────────────────────────────────────────────

HMDB51_CLASSES = [
    "brush_hair", "cartwheel", "catch", "chew", "clap", "climb",
    "climb_stairs", "dive", "draw_sword", "dribble", "drink", "eat",
    "fall_floor", "fencing", "flic_flac", "golf", "handstand", "hit",
    "hug", "jump", "kick", "kick_ball", "kiss", "laugh", "pick",
    "pour", "pullup", "punch", "push", "pushup", "ride_bike",
    "ride_horse", "run", "shake_hands", "shoot_ball", "shoot_bow",
    "shoot_gun", "sit", "situp", "smile", "smoke", "somersault",
    "stand", "swing_baseball", "sword", "sword_exercise", "talk",
    "throw", "turn", "walk", "wave",
]


# ──────────────────────────────────────────────────────────────────────
# Frame Sampling Strategies
# ──────────────────────────────────────────────────────────────────────

def uniform_sample_frames(
    video_path: str,
    num_frames: int = 16,
    image_size: int = 224,
) -> np.ndarray:
    """
    Uniformly sample `num_frames` from a video file.

    Strategy: Divide video into `num_frames` equal segments, take the
    middle frame of each segment. This gives temporally uniform coverage
    regardless of video length or FPS.

    Args:
        video_path: Path to video file (supports .avi, .mp4, etc.)
        num_frames: Number of frames to sample (default 16 for VideoMAE)
        image_size: Target spatial resolution (H = W)

    Returns:
        np.ndarray of shape (num_frames, H, W, 3) in RGB uint8
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    # Read all frames (HMDB51 clips are short, ~2-5s, so this is fine)
    frames_list = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames_list.append(frame)
    cap.release()

    total_frames = len(frames_list)
    if total_frames == 0:
        raise ValueError(f"Video has no readable frames: {video_path}")

    indices = _compute_sample_indices(total_frames, num_frames)
    sampled = [frames_list[i] for i in indices]

    # Resize and convert BGR → RGB
    processed = []
    for frame in sampled:
        frame = cv2.resize(frame, (image_size, image_size))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        processed.append(frame)

    return np.stack(processed, axis=0)  # (T, H, W, 3)


def _compute_sample_indices(total: int, num_samples: int) -> List[int]:
    """
    Compute uniform sample indices with segment-midpoint strategy.

    For a 100-frame video sampled to 16 frames, each segment is ~6.25
    frames wide. We take the midpoint of each segment.

    If the video has fewer frames than requested, we repeat frames
    (padding strategy used by most video transformers).
    """
    if total >= num_samples:
        segment_size = total / num_samples
        return [int(segment_size * i + segment_size / 2) for i in range(num_samples)]
    else:
        indices = list(range(total))
        while len(indices) < num_samples:
            indices.extend(range(total))
        return sorted(indices[:num_samples])


def sample_frames_with_jitter(
    video_path: str,
    num_frames: int = 16,
    image_size: int = 224,
) -> np.ndarray:
    """
    Uniform sampling with per-segment temporal jitter (training augmentation).

    Instead of always taking the segment midpoint, randomly sample one
    frame within each segment. This provides temporal augmentation without
    losing uniform coverage.
    """
    cap = cv2.VideoCapture(str(video_path))
    frames_list = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames_list.append(frame)
    cap.release()

    total_frames = len(frames_list)
    if total_frames == 0:
        return uniform_sample_frames(video_path, num_frames, image_size)

    segment_size = total_frames / num_frames
    indices = []
    for i in range(num_frames):
        start = int(segment_size * i)
        end = min(int(segment_size * (i + 1)), total_frames) - 1
        idx = random.randint(start, max(start, end))
        indices.append(idx)

    sampled = []
    for idx in indices:
        frame = frames_list[idx]
        frame = cv2.resize(frame, (image_size, image_size))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        sampled.append(frame)

    while len(sampled) < num_frames:
        sampled.append(sampled[-1] if sampled else np.zeros((image_size, image_size, 3), dtype=np.uint8))

    return np.stack(sampled[:num_frames], axis=0)


# ──────────────────────────────────────────────────────────────────────
# HMDB51 Dataset Download & Setup
# ──────────────────────────────────────────────────────────────────────

def _download_and_extract_hmdb51(cache_dir: Path) -> None:
    """
    Download HMDB51 zip from HuggingFace Hub and extract video folders.

    Uses huggingface_hub.hf_hub_download for reliable, resumable downloads
    (bypasses the `datasets` library which has label-parsing issues on
    some HMDB51 mirrors).

    The zip contains per-class folders, each with .avi video clips:
        hmdb51/brush_hair/clip_00001.avi
        hmdb51/cartwheel/clip_00002.avi
        ...
    """
    from huggingface_hub import hf_hub_download

    print("[HMDB51] Downloading from HuggingFace Hub (first time only)...")
    print("[HMDB51] This will take a few minutes (~2 GB).")

    # Download the zip file (cached by huggingface_hub)
    zip_path = hf_hub_download(
        repo_id="jili5044/hmdb51",
        filename="hmdb51.zip",
        repo_type="dataset",
    )

    print(f"[HMDB51] Extracting to {cache_dir}...")
    cache_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zf:
        # Extract all files
        total = len(zf.namelist())
        for i, member in enumerate(zf.namelist()):
            zf.extract(member, cache_dir)
            if (i + 1) % 1000 == 0:
                print(f"  Extracted {i + 1}/{total} files...")

    print(f"[HMDB51] Extraction complete.")

    # The zip may nest files under a subfolder — find where the class dirs are
    _flatten_if_needed(cache_dir)


def _flatten_if_needed(cache_dir: Path) -> None:
    """
    If the zip extracted into a nested subfolder (e.g., cache_dir/hmdb51/brush_hair/),
    move class folders up to cache_dir level.
    """
    # Check if class directories are directly in cache_dir
    direct_classes = [
        d for d in cache_dir.iterdir()
        if d.is_dir() and d.name in HMDB51_CLASSES
    ]

    if len(direct_classes) >= 10:
        return  # Already flat

    # Look one level deeper
    for subdir in cache_dir.iterdir():
        if not subdir.is_dir():
            continue
        nested_classes = [
            d for d in subdir.iterdir()
            if d.is_dir() and d.name in HMDB51_CLASSES
        ]
        if len(nested_classes) >= 10:
            print(f"[HMDB51] Moving class folders from {subdir.name}/ up to root...")
            for cls_dir in nested_classes:
                target = cache_dir / cls_dir.name
                if not target.exists():
                    shutil.move(str(cls_dir), str(target))
            return


def _build_split(cache_dir: Path) -> None:
    """
    Create stratified 80/20 train/test split from class-organized folders.

    Split is deterministic (seed=42) so results are reproducible across runs.
    Saves split files to cache_dir for instant loading on subsequent runs.
    """
    rng = random.Random(42)

    # Discover all video files organized by class
    class_groups: Dict[str, List[str]] = {}
    video_extensions = {".avi", ".mp4", ".mkv", ".mov"}

    for cls_name in HMDB51_CLASSES:
        cls_dir = cache_dir / cls_name
        if not cls_dir.exists():
            continue
        videos = [
            str(v) for v in sorted(cls_dir.iterdir())
            if v.suffix.lower() in video_extensions and v.stat().st_size > 0
        ]
        if videos:
            class_groups[cls_name] = videos

    if not class_groups:
        # Try finding .avi files in any subdirectory structure
        for cls_dir in sorted(cache_dir.iterdir()):
            if not cls_dir.is_dir():
                continue
            videos = [
                str(v) for v in sorted(cls_dir.rglob("*.avi"))
                if v.stat().st_size > 0
            ]
            if videos:
                class_groups[cls_dir.name] = videos

    print(f"[HMDB51] Found {len(class_groups)} classes with video files")
    total_vids = sum(len(v) for v in class_groups.values())
    print(f"[HMDB51] Total videos: {total_vids}")

    # Build the class-to-index mapping based on what we actually found
    found_classes = sorted(class_groups.keys())
    class_to_idx = {cls: idx for idx, cls in enumerate(found_classes)}

    # Stratified split
    train_lines = []
    test_lines = []

    for cls_name in found_classes:
        videos = class_groups[cls_name]
        shuffled = videos.copy()
        rng.shuffle(shuffled)
        split_idx = int(len(shuffled) * 0.8)
        label = class_to_idx[cls_name]

        for v in shuffled[:split_idx]:
            train_lines.append(f"{v}\t{label}\n")
        for v in shuffled[split_idx:]:
            test_lines.append(f"{v}\t{label}\n")

    # Save split files
    with open(cache_dir / "train_split.txt", "w") as f:
        f.writelines(train_lines)
    with open(cache_dir / "test_split.txt", "w") as f:
        f.writelines(test_lines)

    # Save class mapping
    with open(cache_dir / "classes.txt", "w") as f:
        for cls_name in found_classes:
            f.write(f"{cls_name}\n")

    print(f"[HMDB51] Split: {len(train_lines)} train, {len(test_lines)} test")


# ──────────────────────────────────────────────────────────────────────
# HMDB51 Dataset Class
# ──────────────────────────────────────────────────────────────────────

class HMDB51Dataset(Dataset):
    """
    HMDB51 action recognition dataset.

    Downloads from HuggingFace Hub on first use and caches locally.
    Creates a stratified 80/20 train/test split.

    Args:
        config: DataConfig with preprocessing settings
        split: "train" or "test"
        use_jitter: Whether to apply temporal jitter augmentation
        processor: Optional HuggingFace VideoMAEImageProcessor
    """

    def __init__(
        self,
        config: DataConfig,
        split: str = "train",
        use_jitter: bool = False,
        processor: Optional[VideoMAEImageProcessor] = None,
    ):
        self.config = config
        self.split = split
        self.use_jitter = use_jitter and (split == "train")

        # Initialize HuggingFace processor for normalization
        self.processor = processor or VideoMAEImageProcessor.from_pretrained(
            "MCG-NJU/videomae-base"
        )

        # Load dataset (download if needed)
        cache_dir = Path("data/hmdb51_cache")
        self._ensure_data_ready(cache_dir)
        self._load_split(cache_dir)

        print(
            f"[HMDB51] Loaded {len(self.video_paths)} {split} samples "
            f"across {len(self.classes)} classes"
        )

    def _ensure_data_ready(self, cache_dir: Path) -> None:
        """Download and extract if split files don't exist yet."""
        split_file = cache_dir / f"{self.split}_split.txt"

        if split_file.exists():
            return  # Already prepared

        # Check if videos are extracted but splits not built
        classes_found = [
            d for d in cache_dir.iterdir()
            if d.is_dir() and any(d.glob("*.avi"))
        ] if cache_dir.exists() else []

        if len(classes_found) < 10:
            _download_and_extract_hmdb51(cache_dir)

        _build_split(cache_dir)

    def _load_split(self, cache_dir: Path) -> None:
        """Load video paths and labels from cached split file."""
        # Load class names
        classes_file = cache_dir / "classes.txt"
        if classes_file.exists():
            with open(classes_file, "r") as f:
                self.classes = [line.strip() for line in f if line.strip()]
        else:
            self.classes = HMDB51_CLASSES

        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        # Load split
        split_file = cache_dir / f"{self.split}_split.txt"
        self.video_paths = []
        self.labels = []

        with open(split_file, "r") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) == 2:
                    path, label = parts[0], int(parts[1])
                    if Path(path).exists():
                        self.video_paths.append(path)
                        self.labels.append(label)

    def __len__(self) -> int:
        return len(self.video_paths)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns:
            dict with:
                "pixel_values": (num_frames, 3, H, W) float tensor
                "labels": scalar int tensor
        """
        video_path = self.video_paths[idx]
        label = self.labels[idx]

        try:
            if self.use_jitter:
                frames = sample_frames_with_jitter(
                    video_path, self.config.num_frames, self.config.image_size
                )
            else:
                frames = uniform_sample_frames(
                    video_path, self.config.num_frames, self.config.image_size
                )
        except Exception as e:
            # Fallback: return black frames rather than crashing training
            print(f"[Warning] Failed to load {video_path}: {e}")
            frames = np.zeros(
                (self.config.num_frames, self.config.image_size, self.config.image_size, 3),
                dtype=np.uint8,
            )

        # frames: (T, H, W, 3) uint8 → list of arrays for processor
        frames_list = [frames[i] for i in range(frames.shape[0])]

        # HuggingFace processor handles normalization and tensor conversion
        inputs = self.processor.preprocess(
            videos=frames_list,
            return_tensors="pt",
        )

        pixel_values = inputs["pixel_values"].squeeze(0)  # (T, C, H, W)

        return {
            "pixel_values": pixel_values,
            "labels": torch.tensor(label, dtype=torch.long),
        }

    def get_class_name(self, idx: int) -> str:
        """Map class index back to human-readable action name."""
        return self.classes[idx]
