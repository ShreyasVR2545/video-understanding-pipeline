"""
HMDB51 Dataset Download & Setup

Downloads the HMDB51 action recognition dataset and organizes it into
the expected directory structure for training.

HMDB51 Stats:
  - 6,766 video clips (train + test)
  - 51 action classes
  - ~2 GB download
  - Mean clip duration: ~3 seconds

Usage:
    python download_data.py [--data_root data/hmdb51]
"""

import argparse
import os
import tarfile
import zipfile
from pathlib import Path
from urllib.request import urlretrieve


# HMDB51 official download URLs
HMDB51_DATA_URL = "http://serre-lab.clps.brown.edu/wp-content/uploads/2013/10/hmdb51_org.rar"
HMDB51_SPLITS_URL = "http://serre-lab.clps.brown.edu/wp-content/uploads/2013/10/test_train_splits.rar"

# Alternative: mirror links (if official URLs are slow)
HMDB51_DATA_ALT = "https://serre-lab.clps.brown.edu/wp-content/uploads/2013/10/hmdb51_org.rar"


def download_file(url: str, dest: Path, desc: str = "") -> Path:
    """Download a file with progress reporting."""
    if dest.exists():
        print(f"  [Skip] {desc} already exists at {dest}")
        return dest

    print(f"  [Download] {desc}: {url}")
    dest.parent.mkdir(parents=True, exist_ok=True)

    def progress_hook(count, block_size, total_size):
        if total_size > 0:
            pct = min(100, count * block_size * 100 // total_size)
            print(f"\r  Progress: {pct}%", end="", flush=True)

    urlretrieve(url, str(dest), reporthook=progress_hook)
    print()
    return dest


def extract_archive(archive_path: Path, dest_dir: Path) -> None:
    """Extract .rar, .tar.gz, or .zip archives."""
    dest_dir.mkdir(parents=True, exist_ok=True)

    suffix = "".join(archive_path.suffixes).lower()

    if ".tar" in suffix or ".tgz" in suffix:
        with tarfile.open(archive_path) as tar:
            tar.extractall(dest_dir)
    elif suffix.endswith(".zip"):
        with zipfile.ZipFile(archive_path) as zf:
            zf.extractall(dest_dir)
    elif suffix.endswith(".rar"):
        # Requires `unrar` system command
        os.system(f"unrar x -o- '{archive_path}' '{dest_dir}/'")
    else:
        print(f"  [Warning] Unknown archive format: {archive_path}")
        return

    print(f"  [Extracted] {archive_path.name} → {dest_dir}")


def setup_hmdb51(data_root: Path) -> None:
    """
    Download and organize HMDB51 dataset.

    Final structure:
        data_root/
        ├── brush_hair/
        │   ├── *.avi
        ├── cartwheel/
        │   ├── *.avi
        ├── ... (51 classes)
        └── splits/
            ├── brush_hair_test_split1.txt
            └── ...
    """
    data_root.mkdir(parents=True, exist_ok=True)
    tmp_dir = data_root / "_downloads"
    tmp_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("HMDB51 Dataset Setup")
    print("=" * 60)

    # ── Step 1: Download video data ──
    print("\n[1/3] Downloading HMDB51 video clips (~2 GB)...")
    data_archive = tmp_dir / "hmdb51_org.rar"
    try:
        download_file(HMDB51_DATA_URL, data_archive, "HMDB51 videos")
    except Exception as e:
        print(f"  Primary URL failed: {e}")
        print("  Trying alternative mirror...")
        download_file(HMDB51_DATA_ALT, data_archive, "HMDB51 videos (mirror)")

    # ── Step 2: Download split annotations ──
    print("\n[2/3] Downloading train/test split files...")
    splits_archive = tmp_dir / "test_train_splits.rar"
    download_file(HMDB51_SPLITS_URL, splits_archive, "Split annotations")

    # ── Step 3: Extract and organize ──
    print("\n[3/3] Extracting and organizing...")

    # Extract video archives (HMDB51 is a .rar of .rar files)
    extract_archive(data_archive, tmp_dir / "videos")

    # The inner structure has per-class .rar files
    inner_dir = tmp_dir / "videos"
    for rar_file in sorted(inner_dir.glob("*.rar")):
        class_name = rar_file.stem
        class_dir = data_root / class_name
        if not class_dir.exists() or not any(class_dir.glob("*.avi")):
            extract_archive(rar_file, class_dir)

    # Extract split files
    splits_dir = data_root / "splits"
    extract_archive(splits_archive, splits_dir)

    # ── Verify ──
    classes = [d.name for d in data_root.iterdir() if d.is_dir() and d.name not in ("splits", "_downloads")]
    total_videos = sum(len(list((data_root / c).glob("*.avi"))) for c in classes)

    print(f"\n{'='*60}")
    print(f"Setup complete!")
    print(f"  Classes: {len(classes)}")
    print(f"  Total videos: {total_videos}")
    print(f"  Location: {data_root.resolve()}")
    print(f"{'='*60}")

    if len(classes) < 51:
        print(
            "\n[Note] Expected 51 classes. If you see fewer, you may need"
            "\n       to install 'unrar' (apt install unrar) and re-run."
        )


def create_mini_split(data_root: Path, num_per_class: int = 10) -> None:
    """
    Create a mini version of HMDB51 for quick debugging.

    Symlinks a small subset of videos per class for fast iteration
    without waiting for full dataset training.
    """
    mini_root = data_root.parent / "hmdb51_mini"
    mini_root.mkdir(parents=True, exist_ok=True)

    classes = sorted([
        d.name for d in data_root.iterdir()
        if d.is_dir() and d.name not in ("splits", "_downloads")
    ])

    for cls in classes:
        src_dir = data_root / cls
        dst_dir = mini_root / cls
        dst_dir.mkdir(exist_ok=True)

        videos = sorted(src_dir.glob("*.avi"))[:num_per_class]
        for v in videos:
            dst = dst_dir / v.name
            if not dst.exists():
                os.symlink(v.resolve(), dst)

    print(f"[Mini] Created mini dataset at {mini_root} ({num_per_class} clips/class)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download HMDB51 dataset")
    parser.add_argument("--data_root", type=str, default="data/hmdb51")
    parser.add_argument(
        "--mini", action="store_true",
        help="Also create a mini split for debugging",
    )
    args = parser.parse_args()

    data_root = Path(args.data_root)
    setup_hmdb51(data_root)

    if args.mini:
        create_mini_split(data_root)
