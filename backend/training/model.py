"""
Minimal defect-detection model for the hybrid-reward training pipeline.

The project's training scaffold (`HybridRewardTrainer`) expects a model whose
forward pass returns a dict with::

    {
        "logits":      (B, num_classes)   # classification logits
        "boxes":       (B, 4)             # [x, y, w, h] in input-pixel coords
        "classes":     (B,)               # argmax(logits)
        "confidences": (B,)               # max softmax probability
    }

`SimpleDefectDetector` is a deliberately small single-stage detector: a ResNet
backbone produces a global feature, and two linear heads predict (a) the image's
dominant defect class and (b) one bounding box. This keeps the model fully
compatible with the existing reward dimensions and loss functions while still
learning to classify and localize the 20 Tianchi defect categories.

For multi-defect-per-image training (e.g. several boxes per photo) this
single-box model is a simplification; see the Ultralytics YOLOv8 path for
full multi-object detection.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn


def _build_backbone() -> nn.Module:
    """Return a ResNet18 feature extractor (outputs a 512-d global vector)."""
    try:
        from torchvision.models import ResNet18_Weights, resnet18

        backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
    except Exception:
        try:
            from torchvision.models import resnet18

            backbone = resnet18(pretrained=True)
        except Exception:
            from torchvision.models import resnet18

            backbone = resnet18(pretrained=False)

    # Drop the classification head, keep everything up to (and including) avgpool.
    return nn.Sequential(*list(backbone.children())[:-1])


class SimpleDefectDetector(nn.Module):
    """Single-stage defect classifier + single-box localizer."""

    def __init__(
        self,
        num_classes: int = 20,
        image_size: Tuple[int, int] = (640, 640),
        hidden_dim: int = 512,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.image_size = image_size
        self.hidden_dim = hidden_dim

        self.backbone = _build_backbone()

        # Heads operate on the 512-d global feature.
        self.class_head = nn.Linear(hidden_dim, num_classes)
        self.box_head = nn.Linear(hidden_dim, 4)

        # ImageNet normalization applied before the pretrained backbone.
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Predict class logits and a single bounding box per image.

        Args:
            images: (B, 3, H, W) float tensor in [0, 1].

        Returns:
            Dict with logits/boxes/classes/confidences (see module docstring).
        """
        x = (images - self.mean) / self.std
        feat = self.backbone(x).flatten(1)  # (B, 512)

        logits = self.class_head(feat)  # (B, num_classes)

        # Box head predicts normalized (cx, cy, w, h), converted to [x, y, w, h].
        raw = torch.sigmoid(self.box_head(feat))  # (B, 4) in [0, 1]
        h, w = self.image_size
        cx = raw[:, 0] * w
        cy = raw[:, 1] * h
        bw = raw[:, 2] * w
        bh = raw[:, 3] * h
        boxes = torch.stack([cx - bw / 2.0, cy - bh / 2.0, bw, bh], dim=-1)

        classes = logits.argmax(dim=-1)  # (B,)
        probs = logits.softmax(dim=-1)
        confidences = probs.max(dim=-1).values  # (B,)

        return {
            "logits": logits,
            "boxes": boxes,
            "classes": classes,
            "confidences": confidences,
        }
