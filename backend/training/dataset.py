"""
Custom dataset loader for fabric defect detection training.

Supports:
  - COCO-format JSON annotations
  - YOLO-format text annotations
  - Custom directory structures
  - Train/val/test split configuration
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import logging
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split

logger = logging.getLogger(__name__)


@dataclass
class DatasetConfig:
    """Configuration for training dataset."""

    # Data paths
    train_images: str = "./data/train/images"
    train_labels: str = "./data/train/labels"
    val_images: str = "./data/val/images"
    val_labels: str = "./data/val/labels"

    # Annotation format: "coco", "yolo", "custom"
    format: str = "yolo"

    # Split ratios (used when no separate val set)
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    # Image settings
    image_size: Tuple[int, int] = (640, 640)
    num_classes: int = 10

    # Augmentation
    augment: bool = True

    # Class names
    class_names: List[str] = field(
        default_factory=lambda: [
            "broken_yarn",
            "missing_stitch",
            "skip_stitch",
            "hole",
            "stain",
            "color_diff",
            "thick_yarn",
            "thin_yarn",
            "crease",
            "other",
        ]
    )


class FabricDataset(Dataset):
    """
    PyTorch Dataset for fabric defect images.

    Supports YOLO format (images/*.jpg + labels/*.txt) and
    COCO format (single JSON annotation file).
    """

    def __init__(
        self,
        image_dir: str,
        label_dir: str = "",
        annotation_file: str = "",
        image_size: Tuple[int, int] = (640, 640),
        format: str = "yolo",
        class_names: Optional[List[str]] = None,
        augment: bool = False,
    ):
        self.image_dir = image_dir
        self.label_dir = label_dir or image_dir.replace("images", "labels")
        self.annotation_file = annotation_file
        self.image_size = image_size
        self.format = format
        self.class_names = class_names or [
            "broken_yarn",
            "missing_stitch",
            "skip_stitch",
            "hole",
            "stain",
            "color_diff",
            "thick_yarn",
            "thin_yarn",
            "crease",
            "other",
        ]
        self.augment = augment

        # Load annotations
        if format == "coco" and annotation_file:
            self._annotations = self._load_coco(annotation_file)
        elif format == "yolo":
            self._annotations = self._load_yolo()
        else:
            self._annotations = []

        # Image file list
        self._image_files = self._find_images()

    def __len__(self) -> int:
        return len(self._image_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict]:
        """Returns (image_tensor, target_dict)."""
        img_path = self._image_files[idx]
        try:
            import cv2

            img = cv2.imread(img_path)
            if img is None:
                img = np.zeros((*self.image_size, 3), dtype=np.uint8)
            img = cv2.resize(img, self.image_size)
            img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        except Exception:
            img_tensor = torch.zeros(3, *self.image_size)

        # Load corresponding labels
        target = self._get_labels(idx)

        return img_tensor, target

    def _find_images(self) -> List[str]:
        """Find all image files in the image directory."""
        if not os.path.isdir(self.image_dir):
            return []
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
        files = sorted(
            [
                os.path.join(self.image_dir, f)
                for f in os.listdir(self.image_dir)
                if os.path.splitext(f)[1].lower() in exts
            ]
        )
        return files

    def _get_labels(self, idx: int) -> Dict:
        """Get labels for a single image."""
        target = {"boxes": [], "labels": []}

        if idx < len(self._annotations):
            ann = self._annotations[idx]
            target["boxes"] = ann.get("boxes", [])
            target["labels"] = ann.get("labels", [])

        return target

    # ------------------------------------------------------------------
    # YOLO format
    # ------------------------------------------------------------------

    def _load_yolo(self) -> List[Dict]:
        """Load YOLO-format annotations (one .txt per image)."""
        annotations = []
        for img_path in self._image_files:
            base = os.path.splitext(os.path.basename(img_path))[0]
            label_path = os.path.join(self.label_dir, base + ".txt")
            boxes, labels = [], []
            if os.path.exists(label_path):
                with open(label_path, "r") as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            cls = int(parts[0])
                            # YOLO: [class, cx, cy, w, h] normalized
                            cx, cy, w, h = map(float, parts[1:5])
                            # Convert to [x, y, w, h] in pixel coords
                            iw, ih = self.image_size
                            boxes.append(
                                [
                                    int((cx - w / 2) * iw),
                                    int((cy - h / 2) * ih),
                                    int(w * iw),
                                    int(h * ih),
                                ]
                            )
                            labels.append(cls)
            annotations.append({"boxes": boxes, "labels": labels})
        return annotations

    # ------------------------------------------------------------------
    # COCO format
    # ------------------------------------------------------------------

    def _load_coco(self, annotation_file: str) -> List[Dict]:
        """Load COCO-format JSON annotations."""
        if not os.path.exists(annotation_file):
            return []

        with open(annotation_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Build image_id → annotations map
        img_anns: Dict[int, List] = {}
        for ann in data.get("annotations", []):
            img_id = ann["image_id"]
            if img_id not in img_anns:
                img_anns[img_id] = []
            img_anns[img_id].append(ann)

        # Build image_id → file_name map
        img_map = {img["id"]: img["file_name"] for img in data.get("images", [])}

        # Match to our image files
        file_to_id = {os.path.basename(f): img_id for img_id, fname in img_map.items()}

        annotations = []
        for img_path in self._image_files:
            img_id = file_to_id.get(os.path.basename(img_path))
            anns = img_anns.get(img_id, []) if img_id else []
            boxes = []
            labels = []
            for ann in anns:
                # COCO bbox: [x, y, w, h]
                boxes.append(ann["bbox"])
                labels.append(ann["category_id"] - 1)  # 1-indexed → 0-indexed
            annotations.append({"boxes": boxes, "labels": labels})

        return annotations


def create_dataloaders(
    config: DatasetConfig,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train/val/test DataLoaders from config.

    Returns:
        (train_loader, val_loader, test_loader).
    """
    # Check if separate val set exists
    has_val = os.path.isdir(config.val_images) and os.listdir(config.val_images)

    if has_val:
        # Use separate directories
        train_dataset = FabricDataset(
            image_dir=config.train_images,
            label_dir=config.train_labels,
            image_size=config.image_size,
            format=config.format,
            class_names=config.class_names,
            augment=config.augment,
        )
        val_dataset = FabricDataset(
            image_dir=config.val_images,
            label_dir=config.val_labels,
            image_size=config.image_size,
            format=config.format,
            class_names=config.class_names,
            augment=False,
        )
        test_dataset = None
    else:
        # Split from single directory
        full_dataset = FabricDataset(
            image_dir=config.train_images,
            label_dir=config.train_labels,
            image_size=config.image_size,
            format=config.format,
            class_names=config.class_names,
            augment=config.augment,
        )
        n = len(full_dataset)
        n_train = int(n * config.train_ratio)
        n_val = int(n * config.val_ratio)
        n_test = n - n_train - n_val
        train_dataset, val_dataset, test_dataset = random_split(
            full_dataset, [n_train, n_val, n_test]
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset or val_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader


def auto_discover_dataset(root_dir: str) -> DatasetConfig:
    """
    Auto-discover dataset structure and create config.

    Supports common directory layouts:
      root/
        train/images/ + train/labels/
        val/images/ + val/labels/
      root/
        images/ + labels/  (auto-split)

    Returns configured DatasetConfig.
    """
    config = DatasetConfig()

    # Check for train/val split
    train_img = os.path.join(root_dir, "train", "images")
    val_img = os.path.join(root_dir, "val", "images")

    if os.path.isdir(train_img):
        config.train_images = train_img
        config.train_labels = os.path.join(root_dir, "train", "labels")
    if os.path.isdir(val_img):
        config.val_images = val_img
        config.val_labels = os.path.join(root_dir, "val", "labels")

    # Auto-detect format
    sample_txt = (
        os.path.join(config.train_labels, os.listdir(config.train_labels)[0])
        if os.path.isdir(config.train_labels) and os.listdir(config.train_labels)
        else ""
    )
    if sample_txt and sample_txt.endswith(".txt"):
        config.format = "yolo"
    elif sample_txt and sample_txt.endswith(".json"):
        config.format = "coco"

    logger.info(f"Auto-discovered dataset: {root_dir}")
    logger.info(f"  Train: {config.train_images}")
    logger.info(f"  Val:   {config.val_images}")
    logger.info(f"  Format: {config.format}")

    return config
