"""
Temporal Attention Visualization

The visual centerpiece of this project. Extracts attention rollout from
VideoMAE and overlays it onto the original video frames as heatmaps,
revealing which frames and spatial regions the model attends to for
each action prediction.

Visualization types:
  1. Temporal attention bar — per-frame attention magnitude
  2. Spatial heatmap overlay — attention projected onto original frames
  3. Combined strip — annotated frame grid with heatmaps + bar chart

These visualizations answer: "Why did the model predict this action?"
by showing the temporal and spatial evidence it relied on.
"""

import cv2
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server/headless use
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
from pathlib import Path
from typing import List, Optional, Tuple

from dataset import uniform_sample_frames
from model import VideoActionClassifier


def generate_attention_overlay(
    model: VideoActionClassifier,
    video_path: str,
    device: torch.device,
    class_names: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    top_k: int = 3,
) -> Tuple[np.ndarray, dict]:
    """
    Generate full attention visualization for a video clip.

    Pipeline:
      1. Sample 16 frames from the video
      2. Run inference to get prediction + attention rollout
      3. Reshape attention to (temporal, spatial_h, spatial_w)
      4. Upsample spatial attention to frame resolution
      5. Overlay as colored heatmap on original frames
      6. Compose into annotated visualization figure

    Args:
        model: Trained VideoActionClassifier
        video_path: Path to input video
        device: Compute device
        class_names: Optional list mapping indices → action names
        save_path: If provided, save figure to this path
        top_k: Number of top predictions to show

    Returns:
        figure_array: Rendered figure as numpy array (H, W, 3)
        info: dict with prediction details and attention stats
    """
    from transformers import VideoMAEImageProcessor

    processor = VideoMAEImageProcessor.from_pretrained("MCG-NJU/videomae-base")

    # ── Step 1: Sample and preprocess frames ──
    raw_frames = uniform_sample_frames(video_path, num_frames=16, image_size=224)
    # raw_frames: (16, 224, 224, 3) uint8 RGB

    frames_list = [raw_frames[i] for i in range(raw_frames.shape[0])]
    inputs = processor.preprocess(videos=frames_list, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)  # (1, 16, 3, 224, 224)

    # ── Step 2: Get attention maps and predictions ──
    attention_map, logits = model.get_attention_maps(pixel_values)
    # attention_map: (T_patches, H_patches, W_patches), normalized [0,1]
    # For VideoMAE-Base: (8, 14, 14)

    probs = torch.softmax(logits, dim=-1).cpu().squeeze(0)
    top_probs, top_indices = probs.topk(top_k)

    # ── Step 3: Compute per-frame temporal attention ──
    # Sum spatial attention within each temporal patch
    temporal_attention = attention_map.sum(dim=(1, 2))  # (T_patches,)
    temporal_attention = temporal_attention / temporal_attention.sum()  # normalize

    # Map temporal patches back to frames, handling any temporal dimension
    ta = temporal_attention.numpy()
    frame_attention = np.interp(
        np.linspace(0, len(ta) - 1, 16),
        np.arange(len(ta)),
        ta,
    )  # always exactly (16,)

    # ── Step 4: Upsample spatial attention to frame resolution ──
    spatial_heatmaps = _upsample_attention_to_frames(attention_map.numpy(), target_size=224, num_frames=16)
    # spatial_heatmaps: (16, 224, 224) float32

    # ── Step 5: Compose visualization ──
    prediction_info = {
        "top_predictions": [],
        "frame_attention": frame_attention.tolist(),
    }
    for i in range(top_k):
        idx = top_indices[i].item()
        name = class_names[idx] if class_names else f"class_{idx}"
        prediction_info["top_predictions"].append({
            "class": name,
            "confidence": f"{100 * top_probs[i].item():.1f}%",
        })

    fig = _compose_visualization(
        raw_frames, spatial_heatmaps, frame_attention, prediction_info
    )

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="black")
        print(f"[Viz] Saved attention visualization to {save_path}")

    # Convert figure to numpy array for Gradio display
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())
    figure_array = buf[:, :, :3].copy()  # RGBA → RGB
    plt.close(fig)

    return figure_array, prediction_info


def _upsample_attention_to_frames(
    attention_map: np.ndarray,
    target_size: int = 224,
    num_frames: int = 16,
) -> np.ndarray:
    """
    Upsample attention to (num_frames, H, W).

    Uses bilinear interpolation for smooth heatmaps. Temporal patches
    are repeated to cover their corresponding frames, then padded or
    truncated to exactly num_frames.
    """
    if attention_map.ndim == 3:
        t_patches, h_patches, w_patches = attention_map.shape
    elif attention_map.ndim == 2:
        t_patches = 1
        h_patches, w_patches = attention_map.shape
        attention_map = attention_map[np.newaxis, ...]
    else:
        # 1D fallback
        side = int(np.sqrt(attention_map.shape[0]))
        attention_map = attention_map.reshape(1, side, -1)
        t_patches = 1
        h_patches, w_patches = attention_map.shape[1], attention_map.shape[2]

    heatmaps = []
    frames_per_patch = max(1, num_frames // t_patches)

    for t in range(t_patches):
        spatial = attention_map[t]
        upsampled = cv2.resize(
            spatial, (target_size, target_size),
            interpolation=cv2.INTER_LINEAR,
        )
        for _ in range(frames_per_patch):
            heatmaps.append(upsampled)

    # Pad or truncate to exactly num_frames
    while len(heatmaps) < num_frames:
        heatmaps.append(heatmaps[-1])

    return np.stack(heatmaps[:num_frames], axis=0)

def overlay_heatmap_on_frame(
    frame: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.5,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """
    Blend a normalized heatmap onto an RGB frame.

    Args:
        frame: (H, W, 3) uint8 RGB
        heatmap: (H, W) float32 in [0, 1]
        alpha: Blend factor (0 = original frame, 1 = full heatmap)
        colormap: OpenCV colormap for the heatmap

    Returns:
        (H, W, 3) uint8 RGB blended image
    """
    heatmap_uint8 = (heatmap * 255).astype(np.uint8)
    colored = cv2.applyColorMap(heatmap_uint8, colormap)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)

    blended = (
        (1 - alpha) * frame.astype(np.float32)
        + alpha * colored.astype(np.float32)
    ).astype(np.uint8)

    return blended


def _compose_visualization(
    frames: np.ndarray,
    heatmaps: np.ndarray,
    frame_attention: np.ndarray,
    prediction_info: dict,
) -> plt.Figure:
    """
    Compose the full attention visualization figure.

    Layout:
    ┌──────────────────────────────────────────────────┐
    │  Title: Predicted action + confidence scores      │
    ├──────────────────────────────────────────────────┤
    │  Row 1: Original frames (8 frames, every other)  │
    │  Row 2: Attention overlay on same frames          │
    ├──────────────────────────────────────────────────┤
    │  Bottom: Temporal attention bar chart (16 frames) │
    └──────────────────────────────────────────────────┘
    """
    fig = plt.figure(figsize=(20, 10), facecolor="black")
    gs = gridspec.GridSpec(
        3, 8, figure=fig,
        height_ratios=[1, 1, 0.6],
        hspace=0.3, wspace=0.05,
    )

    # ── Title with predictions ──
    pred_text = " | ".join(
        f"{p['class']}: {p['confidence']}"
        for p in prediction_info["top_predictions"]
    )
    fig.suptitle(
        f"Predicted Action: {pred_text}",
        color="white", fontsize=14, fontweight="bold", y=0.98,
    )

    # ── Display 8 frames (every other) with attention overlays ──
    display_indices = list(range(0, 16, 2))  # [0, 2, 4, 6, 8, 10, 12, 14]

    for col, frame_idx in enumerate(display_indices):
        # Row 1: Original frames
        ax_orig = fig.add_subplot(gs[0, col])
        ax_orig.imshow(frames[frame_idx])
        ax_orig.set_title(f"t={frame_idx}", color="white", fontsize=9)
        ax_orig.axis("off")

        # Row 2: Attention heatmap overlay
        ax_heat = fig.add_subplot(gs[1, col])
        overlay = overlay_heatmap_on_frame(frames[frame_idx], heatmaps[frame_idx])
        ax_heat.imshow(overlay)
        # Annotate with attention magnitude
        attn_val = frame_attention[frame_idx]
        ax_heat.set_title(f"attn={attn_val:.3f}", color="cyan", fontsize=9)
        ax_heat.axis("off")

    # ── Bottom: Temporal attention bar chart ──
    ax_bar = fig.add_subplot(gs[2, :])
    bars = ax_bar.bar(
        range(16), frame_attention,
        color=plt.cm.plasma(frame_attention / frame_attention.max()),
        edgecolor="none",
    )
    ax_bar.set_xlabel("Frame Index", color="white", fontsize=11)
    ax_bar.set_ylabel("Attention", color="white", fontsize=11)
    ax_bar.set_title("Temporal Attention Distribution", color="white", fontsize=12)
    ax_bar.set_xticks(range(16))
    ax_bar.tick_params(colors="white")
    ax_bar.set_facecolor("#1a1a2e")
    ax_bar.spines["bottom"].set_color("white")
    ax_bar.spines["left"].set_color("white")
    ax_bar.spines["top"].set_visible(False)
    ax_bar.spines["right"].set_visible(False)

    # Highlight the peak-attention frame
    peak_idx = np.argmax(frame_attention)
    bars[peak_idx].set_edgecolor("cyan")
    bars[peak_idx].set_linewidth(2)
    ax_bar.annotate(
        "peak", xy=(peak_idx, frame_attention[peak_idx]),
        xytext=(peak_idx, frame_attention[peak_idx] + 0.02),
        color="cyan", fontsize=9, ha="center",
    )

    return fig


def generate_attention_gif(
    model: VideoActionClassifier,
    video_path: str,
    device: torch.device,
    save_path: str = "attention.gif",
    fps: int = 4,
    class_names: Optional[List[str]] = None,
) -> str:
    """
    Generate an animated GIF showing attention heatmap evolving over frames.

    This is the GIF deliverable for the README — it shows the attention
    overlay smoothly transitioning across the 16 sampled frames, making
    temporal attention patterns visually intuitive.

    Args:
        model: Trained VideoActionClassifier
        video_path: Path to input video
        device: Compute device
        save_path: Output GIF path
        fps: Frames per second in the GIF
        class_names: Optional class name mapping

    Returns:
        save_path for convenience
    """
    import imageio
    from transformers import VideoMAEImageProcessor

    processor = VideoMAEImageProcessor.from_pretrained("MCG-NJU/videomae-base")

    # Sample and preprocess
    raw_frames = uniform_sample_frames(video_path, num_frames=16, image_size=224)
    frames_list = [raw_frames[i] for i in range(raw_frames.shape[0])]
    inputs = processor.preprocess(videos=frames_list, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)

    # Get attention
    attention_map, logits = model.get_attention_maps(pixel_values)
    probs = torch.softmax(logits, dim=-1).cpu().squeeze(0)
    top_prob, top_idx = probs.max(dim=0)
    pred_name = class_names[top_idx.item()] if class_names else f"class_{top_idx.item()}"

    heatmaps = _upsample_attention_to_frames(attention_map.numpy(), target_size=224, num_frames=16)

    # Generate annotated frames for GIF
    gif_frames = []
    for i in range(16):
        overlay = overlay_heatmap_on_frame(raw_frames[i], heatmaps[i], alpha=0.5)

        # Add text annotation
        frame_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
        cv2.putText(
            frame_bgr, f"Frame {i}/15",
            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
        )
        cv2.putText(
            frame_bgr, f"{pred_name} ({100*top_prob.item():.1f}%)",
            (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1,
        )
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        gif_frames.append(frame_rgb)

    imageio.mimsave(save_path, gif_frames, fps=fps, loop=0)
    print(f"[Viz] Saved attention GIF to {save_path}")

    return save_path
