"""
Data augmentation pipeline for fabric defect detection training.

Matches Section 5.4 augmentation config:
  - random_flip: 0.5
  - random_rotate: 15 degrees
  - color_jitter: [0.3, 0.3, 0.3]
  - mosaic: 0.5
  - mixup: 0.3
"""

import random
from typing import List, Optional, Tuple

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF


class DefectAugmentation:
    """
    Composable augmentation pipeline for fabric defect images.

    Supports image-level (flip, rotate, color) and batch-level (mosaic, mixup)
    augmentations as specified in Section 5.4.
    """

    def __init__(
        self,
        flip_prob: float = 0.5,
        rotate_degrees: int = 15,
        color_jitter: Tuple[float, float, float] = (0.3, 0.3, 0.3),
        mosaic_prob: float = 0.5,
        mixup_prob: float = 0.3,
        mixup_alpha: float = 1.0,
    ):
        self.flip_prob = flip_prob
        self.rotate_degrees = rotate_degrees
        self.mosaic_prob = mosaic_prob
        self.mixup_prob = mixup_prob
        self.mixup_alpha = mixup_alpha

        # Color jitter transform
        self.color_jitter = (
            T.ColorJitter(
                brightness=color_jitter[0],
                contrast=color_jitter[1],
                saturation=color_jitter[2],
            )
            if any(c > 0 for c in color_jitter)
            else None
        )

    def __call__(
        self,
        images: torch.Tensor,
        targets: Optional[List[dict]] = None,
    ) -> Tuple[torch.Tensor, Optional[List[dict]]]:
        """
        Apply augmentations to a batch of images.

        Args:
            images: Batch tensor (B, C, H, W).
            targets: List of target dicts with 'boxes' and 'labels'.

        Returns:
            (augmented_images, augmented_targets).
        """
        batch_size = images.size(0)

        for i in range(batch_size):
            img = images[i]
            target = targets[i] if targets else None

            # Random horizontal flip
            if random.random() < self.flip_prob:
                img = TF.hflip(img)
                if target and "boxes" in target:
                    w = img.size(-1)
                    for box in target["boxes"]:
                        box[0] = w - box[0] - box[2]

            # Random rotation
            angle = random.uniform(-self.rotate_degrees, self.rotate_degrees)
            img = TF.rotate(img.unsqueeze(0), angle).squeeze(0)

            # Color jitter
            if self.color_jitter is not None:
                img = self.color_jitter(img)

            images[i] = img

        # Batch-level: Mosaic (Section 5.4)
        if random.random() < self.mosaic_prob and batch_size >= 4:
            images = self._apply_mosaic(images)

        # Batch-level: MixUp (Section 5.4)
        if random.random() < self.mixup_prob and batch_size >= 2:
            images, targets = self._apply_mixup(images, targets)

        return images, targets

    def _apply_mosaic(self, images: torch.Tensor) -> torch.Tensor:
        """
        Apply Mosaic augmentation: combine 4 images into one grid.

        Each quadrant of the output comes from a different image.
        """
        if images.size(0) < 4:
            return images

        _, _, H, W = images.shape
        # Take first 4 images
        a, b, c, d = images[0], images[1], images[2], images[3]

        # Random cut point
        cx = random.randint(W // 4, 3 * W // 4)
        cy = random.randint(H // 4, 3 * H // 4)

        mosaic = images[0].clone()
        mosaic[:, :cy, :cx] = a[:, :cy, :cx]  # top-left
        mosaic[:, :cy, cx:] = b[:, :cy, W - cx :]  # top-right
        mosaic[:, cy:, :cx] = c[:, H - cy :, :cx]  # bottom-left
        mosaic[:, cy:, cx:] = d[:, H - cy :, W - cx :]  # bottom-right

        # Put the mosaic back at index 0 and keep the full batch (previously
        # this collapsed the batch to a single image via unsqueeze(0)).
        out = images.clone()
        out[0] = mosaic
        return out

    def _apply_mixup(
        self,
        images: torch.Tensor,
        targets: Optional[List[dict]],
    ) -> Tuple[torch.Tensor, Optional[List[dict]]]:
        """
        Apply MixUp augmentation: linearly interpolate two images.
        """
        if images.size(0) < 2:
            return images, targets

        # Sample lambda from Beta distribution
        lam = random.betavariate(self.mixup_alpha, self.mixup_alpha)

        # Mix first two images
        mixed = lam * images[0] + (1 - lam) * images[1]
        images[0] = mixed

        # Mix targets if available
        if targets and len(targets) >= 2:
            targets[0] = {
                **targets[0],
                "_mixup_lambda": lam,
                "_mixup_target_2": targets[1],
            }

        return images, targets


def build_augmentation(config: Optional[dict] = None) -> DefectAugmentation:
    """
    Build augmentation pipeline from config dict (Section 5.4 format).

    Args:
        config: Dict with keys matching Section 5.4 augmentation block.

    Returns:
        Configured DefectAugmentation instance.
    """
    if config is None:
        return DefectAugmentation()

    return DefectAugmentation(
        flip_prob=config.get("random_flip", 0.5),
        rotate_degrees=config.get("random_rotate", 15),
        color_jitter=config.get("color_jitter", [0.3, 0.3, 0.3]),
        mosaic_prob=config.get("mosaic", 0.5),
        mixup_prob=config.get("mixup", 0.3),
    )
