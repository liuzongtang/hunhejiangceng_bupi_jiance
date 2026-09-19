"""
Train the minimal 20-class defect detector with the hybrid-reward trainer.

This is "Path B": wire the project's own `HybridRewardTrainer` (7-dimension
hybrid reward) to the Tianchi dataset via `SimpleDefectDetector`.

Run after converting the dataset (scripts/convert_tianchi_to_yolo.py):

    python scripts/train_defect_detector.py --data data/tianchi --epochs 20

The model is a single-box detector (dominant defect per image) — a
demonstration of the hybrid-reward pipeline. For full multi-object detection,
see the Ultralytics YOLOv8 path (data/tianchi/data.yaml).
"""

from __future__ import annotations

import argparse

import torch

from backend.training.dataset import auto_discover_dataset, create_dataloaders
from backend.training.hybrid_reward_trainer import HybridRewardTrainer
from backend.training.model import SimpleDefectDetector

# Augmentation is disabled for a first clean run (the scaffold's mosaic/mixup
# target handling is incomplete). Set all probabilities to 0.
_NO_AUGMENTATION = {
    "random_flip": 0.0,
    "random_rotate": 0,
    "color_jitter": [0.0, 0.0, 0.0],
    "mosaic": 0.0,
    "mixup": 0.0,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/tianchi", help="Converted dataset root")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--device", default=None, help="cuda / cpu (auto if omitted)")
    parser.add_argument("--checkpoint", default="checkpoints/defect_detector.pt")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 1. Discover the converted dataset (train/val split + yolo format).
    config = auto_discover_dataset(args.data)
    print(
        f"Dataset: {config.train_images} "
        f"(format={config.format}, classes={config.num_classes})"
    )

    # 2. Build dataloaders (collate_fn handles variable-length boxes).
    train_loader, val_loader, _ = create_dataloaders(config, batch_size=args.batch_size)

    # 3. Build the model and trainer.
    model = SimpleDefectDetector(
        num_classes=config.num_classes, image_size=config.image_size
    )
    trainer = HybridRewardTrainer(
        model,
        train_loader,
        val_loader,
        learning_rate=args.lr,
        num_classes=config.num_classes,
        augmentation_config=_NO_AUGMENTATION,
        device=device,
    )

    # 4. Train.
    metrics = None
    try:
        metrics = trainer.fit(epochs=args.epochs)
    except KeyboardInterrupt:
        print("\nInterrupted — saving checkpoint anyway.")

    # 5. Save checkpoint.
    import os

    os.makedirs(os.path.dirname(args.checkpoint) or ".", exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "num_classes": config.num_classes,
            "class_names": config.class_names,
            "image_size": config.image_size,
        },
        args.checkpoint,
    )
    print(f"\nSaved checkpoint to {args.checkpoint}")

    if metrics is not None:
        print(f"Best scalar reward: {trainer.best_val_reward:.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
