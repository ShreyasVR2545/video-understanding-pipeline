"""
VideoMAE Model for Action Recognition

Architecture:
  ┌──────────────────────────────────────────────────┐
  │  VideoMAE-Base Backbone (ViT encoder)            │
  │  Input: 16 frames × 224×224 → 1568 patch tokens  │
  │  Output: 768-dim token embeddings                 │
  └──────────────────┬───────────────────────────────┘
                     │ CLS token (768-dim)
  ┌──────────────────▼───────────────────────────────┐
  │  Classification Head                              │
  │  LayerNorm → Dropout → Linear(768, 51)           │
  └──────────────────────────────────────────────────┘

Key design choices:
  - Separate backbone freezing/unfreezing for two-phase training
  - Custom classification head (not using HF's built-in) for flexibility
  - Attention weight extraction hooks for visualization
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from transformers import VideoMAEModel

from config import ModelConfig


class VideoActionClassifier(nn.Module):
    """
    VideoMAE-based action classifier with two-phase training support.

    Phase 1 (warmup): Backbone frozen, only classification head trains.
        → Prevents catastrophic forgetting of pretrained representations.
    Phase 2 (fine-tune): Full model trains with lower learning rate.
        → Adapts spatial-temporal features to HMDB51 domain.

    Args:
        config: ModelConfig with backbone name, hidden size, num classes
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # ── Load pretrained VideoMAE backbone ──
        self.backbone = VideoMAEModel.from_pretrained(
            config.backbone,
            ignore_mismatched_sizes=True,
        )

        # ── Classification head ──
        # LayerNorm → Dropout → Linear, operating on the CLS token
        self.classifier = nn.Sequential(
            nn.LayerNorm(config.hidden_size),
            nn.Dropout(config.drop_rate),
            nn.Linear(config.hidden_size, config.num_classes),
        )

    def forward(
        self,
        pixel_values: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through VideoMAE + classification head.

        Args:
            pixel_values: (B, T, C, H, W) video tensor
            labels: (B,) class indices for loss computation
            output_attentions: If True, return attention weights from all layers

        Returns:
            dict with "logits", optionally "loss" and "attentions"
        """
        # VideoMAE forward pass
        outputs = self.backbone(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
        )

        # Use CLS token representation for classification
        # outputs.last_hidden_state shape: (B, num_patches + 1, hidden_size)
        # The CLS token is at index 0
        cls_token = outputs.last_hidden_state[:, 0, :]  # (B, 768)

        logits = self.classifier(cls_token)  # (B, num_classes)

        result = {"logits": logits}

        # Compute cross-entropy loss if labels provided
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            result["loss"] = loss_fn(logits, labels)

        # Pass through attention weights for visualization
        if output_attentions and outputs.attentions is not None:
            result["attentions"] = outputs.attentions

        return result

    # ──────────────────────────────────────────────────────────────
    # Backbone Freeze / Unfreeze for Two-Phase Training
    # ──────────────────────────────────────────────────────────────

    def freeze_backbone(self) -> None:
        """
        Freeze all backbone parameters (Phase 1: warmup).

        Only the classification head will receive gradients.
        This prevents the pretrained VideoMAE representations from
        being destroyed before the head learns a reasonable mapping.
        """
        for param in self.backbone.parameters():
            param.requires_grad = False

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(
            f"[Model] Backbone FROZEN — "
            f"trainable: {trainable:,} / {total:,} params "
            f"({100 * trainable / total:.1f}%)"
        )

    def unfreeze_backbone(self) -> None:
        """
        Unfreeze all backbone parameters (Phase 2: full fine-tuning).

        Called after warmup epochs. The head has learned a reasonable
        mapping, so backbone updates won't cause divergence.
        """
        for param in self.backbone.parameters():
            param.requires_grad = True

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(
            f"[Model] Backbone UNFROZEN — "
            f"trainable: {trainable:,} / {total:,} params "
            f"({100 * trainable / total:.1f}%)"
        )

    # ──────────────────────────────────────────────────────────────
    # Attention Extraction for Visualization
    # ──────────────────────────────────────────────────────────────

    def get_attention_maps(
        self,
        pixel_values: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract attention rollout across all transformer layers.

        Attention rollout multiplies attention matrices across layers to
        approximate how much each input token contributes to the final
        CLS token representation. This gives us a spatial-temporal
        attention map showing which frames and regions matter most.

        Uses a separate "eager" backbone instance to extract attention
        weights, since the default SDPA implementation doesn't support
        output_attentions in newer transformers versions.

        Args:
            pixel_values: (1, T, C, H, W) single video tensor

        Returns:
            attention_map: (T, H_patches, W_patches) normalized attention
            logits: (1, num_classes) prediction logits
        """
        self.eval()

        # ── Step 1: Get logits from the main model ──
        with torch.no_grad():
            outputs = self.forward(pixel_values=pixel_values)
        logits = outputs["logits"]

        # ── Step 2: Load a separate eager-attention backbone for viz ──
        # The default SDPA attention doesn't return attention weights,
        # so we need an "eager" implementation just for visualization.
        eager_backbone = VideoMAEModel.from_pretrained(
            self.config.backbone,
            attn_implementation="eager",
            ignore_mismatched_sizes=True,
        )

        # Copy our fine-tuned weights into the eager backbone
        # (only copy matching keys to handle minor arch differences)
        trained_state = self.backbone.state_dict()
        eager_state = eager_backbone.state_dict()
        for key in eager_state:
            if key in trained_state and eager_state[key].shape == trained_state[key].shape:
                eager_state[key] = trained_state[key]
        eager_backbone.load_state_dict(eager_state)
        eager_backbone = eager_backbone.to(pixel_values.device)
        eager_backbone.eval()

        # ── Step 3: Forward pass with attention output ──
        with torch.no_grad():
            eager_outputs = eager_backbone(
                pixel_values=pixel_values,
                output_attentions=True,
            )

        attentions = eager_outputs.attentions  # tuple of (1, heads, seq, seq)

        if attentions is None or len(attentions) == 0:
            # Fallback: return uniform attention
            print("[Warning] Could not extract attention weights, using uniform attention")
            num_temporal = self.config.num_frames // self.config.tubelet_size
            attention_map = torch.ones(num_temporal, 14, 14) / (14 * 14)
            return attention_map, logits

        # ── Step 4: Attention rollout ──
        # Multiply attention matrices across layers to approximate
        # how much each input token contributes to the CLS token.
        rollout = None
        for attn in attentions:
            # attn: (1, num_heads, seq_len, seq_len)
            attn_mean = attn.mean(dim=1)  # (1, seq_len, seq_len)

            # Add residual connection (identity) as in original rollout paper
            attn_mean = 0.5 * attn_mean + 0.5 * torch.eye(
                attn_mean.size(-1), device=attn_mean.device
            ).unsqueeze(0)

            # Re-normalize rows
            attn_mean = attn_mean / attn_mean.sum(dim=-1, keepdim=True)

            if rollout is None:
                rollout = attn_mean
            else:
                rollout = torch.bmm(rollout, attn_mean)

        # Extract CLS token's attention to all patch tokens
        # rollout: (1, seq_len, seq_len) — row 0 is CLS attending to everything
        cls_attention = rollout[0, 0, 1:]  # exclude CLS-to-CLS

        # ── Step 5: Reshape to spatial-temporal grid ──
        # VideoMAE with tubelet_size=2: temporal patches = T/2 = 8
        # Spatial patches per frame: (224/16)² = 196
        # Total: 8 * 196 = 1568 visible patches
        num_temporal = self.config.num_frames // self.config.tubelet_size  # 8
        num_spatial_h = self.config.image_size // 16  # 14
        num_spatial_w = self.config.image_size // 16  # 14
        expected_patches = num_temporal * num_spatial_h * num_spatial_w

        if cls_attention.shape[0] == expected_patches:
            attention_map = cls_attention.reshape(
                num_temporal, num_spatial_h, num_spatial_w
            )
        else:
            # Fallback: adaptive reshape if patch count differs
            n = cls_attention.shape[0]
            side = int(n ** 0.5)
            if side * side == n:
                attention_map = cls_attention.reshape(1, side, side)
            else:
                attention_map = cls_attention.reshape(1, 1, -1)

        # Normalize to [0, 1]
        attention_map = (attention_map - attention_map.min()) / (
            attention_map.max() - attention_map.min() + 1e-8
        )

        # Clean up the eager backbone to free VRAM
        del eager_backbone
        if pixel_values.is_cuda:
            torch.cuda.empty_cache()

        return attention_map.cpu(), logits

    def get_param_groups(self, lr: float, backbone_lr_scale: float = 0.1):
        """
        Create parameter groups with differential learning rates.

        The backbone uses a scaled-down learning rate to preserve
        pretrained features, while the head trains at full LR.

        Args:
            lr: Base learning rate for the classification head
            backbone_lr_scale: Multiplier for backbone LR (default 0.1x)

        Returns:
            List of param group dicts for torch.optim
        """
        return [
            {
                "params": self.backbone.parameters(),
                "lr": lr * backbone_lr_scale,
                "name": "backbone",
            },
            {
                "params": self.classifier.parameters(),
                "lr": lr,
                "name": "classifier",
            },
        ]
