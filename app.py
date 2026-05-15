"""
Gradio Inference Demo — Video Action Recognition with Attention Visualization

Upload a short video clip → get the predicted action class, confidence scores,
and a temporal attention heatmap showing which frames and regions drove the
model's decision.

Usage:
    python app.py [--checkpoint PATH] [--share]
"""

import argparse
from pathlib import Path
from typing import Optional, Tuple

import gradio as gr
import numpy as np
import torch

from attention_viz import generate_attention_overlay, generate_attention_gif
from config import PipelineConfig, ModelConfig
from dataset import uniform_sample_frames
from model import VideoActionClassifier
from train import load_checkpoint


# ──────────────────────────────────────────────────────────────────────
# HMDB51 Class Names (alphabetical — matches dataset.py ordering)
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


def load_model(
    checkpoint_path: Optional[str] = None,
    device: Optional[torch.device] = None,
) -> Tuple[VideoActionClassifier, torch.device]:
    """Load model from checkpoint or initialize fresh for demo."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = ModelConfig()
    model = VideoActionClassifier(config).to(device)

    if checkpoint_path and Path(checkpoint_path).exists():
        load_checkpoint(model, Path(checkpoint_path), device)
        print(f"[App] Loaded fine-tuned model from {checkpoint_path}")
    else:
        print("[App] No checkpoint found — using pretrained backbone with random head")
        print("[App] Run train.py first for meaningful predictions")

    model.eval()
    return model, device


def predict_action(
    video_path: str,
    model: VideoActionClassifier,
    device: torch.device,
) -> Tuple[dict, np.ndarray, Optional[str]]:
    """
    Run inference on a video and generate attention visualization.

    Returns:
        confidences: dict mapping class names → confidence scores (for Gradio label)
        viz_image: attention visualization as numpy array
        gif_path: path to attention GIF (or None)
    """
    if video_path is None:
        return {}, np.zeros((224, 224, 3), dtype=np.uint8), None

    # Generate attention visualization (includes prediction)
    viz_image, info = generate_attention_overlay(
        model=model,
        video_path=video_path,
        device=device,
        class_names=HMDB51_CLASSES,
        top_k=5,
    )

    # Build confidence dict for Gradio's Label component
    confidences = {}
    for pred in info["top_predictions"]:
        conf_str = pred["confidence"].replace("%", "")
        confidences[pred["class"]] = float(conf_str) / 100.0

    # Generate GIF for download
    gif_path = None
    try:
        gif_path = generate_attention_gif(
            model=model,
            video_path=video_path,
            device=device,
            save_path="/tmp/attention_overlay.gif",
            class_names=HMDB51_CLASSES,
        )
    except ImportError:
        print("[App] imageio not installed — skipping GIF generation")

    return confidences, viz_image, gif_path


def build_demo(
    model: VideoActionClassifier,
    device: torch.device,
) -> gr.Blocks:
    """Construct the Gradio interface."""

    with gr.Blocks(
        title="Video Action Recognition — VideoMAE + HMDB51",
        theme=gr.themes.Soft(),
    ) as demo:
        gr.Markdown(
            """
            # 🎬 Video Action Recognition with Temporal Attention

            Upload a short video clip to classify the action and visualize
            which frames and regions the VideoMAE model attends to.

            **Model:** VideoMAE-Base fine-tuned on HMDB51 (51 action classes)
            **Input:** 16 uniformly sampled frames at 224×224
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                video_input = gr.Video(label="Upload Video Clip")
                predict_btn = gr.Button("🔍 Analyze", variant="primary", size="lg")

                gr.Markdown(
                    """
                    ### Supported Actions
                    Examples: `brush_hair`, `cartwheel`, `climb_stairs`,
                    `dribble`, `fencing`, `golf`, `handstand`, `kick_ball`,
                    `ride_bike`, `run`, `swing_baseball`, `walk` ...
                    (51 classes total)
                    """
                )

            with gr.Column(scale=2):
                label_output = gr.Label(
                    label="Top-5 Predictions",
                    num_top_classes=5,
                )
                viz_output = gr.Image(
                    label="Temporal Attention Visualization",
                    type="numpy",
                )
                gif_output = gr.File(label="Download Attention GIF")

        # ── Event handler ──
        def on_predict(video):
            if video is None:
                return {}, None, None
            confidences, viz, gif = predict_action(video, model, device)
            return confidences, viz, gif

        predict_btn.click(
            fn=on_predict,
            inputs=[video_input],
            outputs=[label_output, viz_output, gif_output],
        )

        # ── Example clips (if available) ──
        example_dir = Path("examples")
        if example_dir.exists():
            examples = [str(p) for p in sorted(example_dir.glob("*.avi"))[:4]]
            if not examples:
                examples = [str(p) for p in sorted(example_dir.glob("*.mp4"))[:4]]
            if examples:
                gr.Examples(
                    examples=[[e] for e in examples],
                    inputs=[video_input],
                )

    return demo


# ──────────────────────────────────────────────────────────────────────
# Entry Point
# ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Video Action Recognition Demo")
    parser.add_argument(
        "--checkpoint", type=str, default="checkpoints/best_model.pt",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--share", action="store_true",
        help="Create public Gradio link",
    )
    parser.add_argument(
        "--port", type=int, default=7860,
        help="Port for the Gradio server",
    )
    args = parser.parse_args()

    model, device = load_model(args.checkpoint)
    demo = build_demo(model, device)
    demo.launch(server_port=args.port, share=args.share)
