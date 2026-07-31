#!/usr/bin/env python3
"""
Handwritten Digit Recognition -- Training Script.

Trains a CNN on MNIST with full training process visualization.
Supports WandB (cloud) and TensorBoard (local) for live monitoring.

Usage:
    # Quick training (3 epochs, CPU, TensorBoard only):
    python scripts/train_digit_recognizer.py --epochs 3

    # Full training with WandB:
    python scripts/train_digit_recognizer.py --epochs 10 --wandb

    # Custom batch size and learning rate:
    python scripts/train_digit_recognizer.py --epochs 5 --batch-size 128 --lr 0.002

    # CPU-only (force even if CUDA available):
    python scripts/train_digit_recognizer.py --epochs 5 --cpu

Monitoring:
    TensorBoard: tensorboard --logdir ./runs/digit_recognition
"""

import argparse
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from mixed_reward.digit_recognition import (
    DigitCNN,
    DigitTrainer,
    count_parameters,
    get_mnist_dataloaders,
)
from mixed_reward.visualizer import MixedRewardVisualizer


def main():
    parser = argparse.ArgumentParser(
        description="Train a CNN for handwritten digit recognition (MNIST)"
    )

    # Training
    parser.add_argument(
        "--epochs", type=int, default=5, help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for training and evaluation",
    )
    parser.add_argument("--lr", type=float, default=0.001, help="Initial learning rate")
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-5,
        help="Weight decay for Adam optimizer",
    )
    parser.add_argument(
        "--dropout", type=float, default=0.5, help="Dropout rate in the FC layer"
    )

    # Hardware
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU training even if CUDA is available",
    )
    parser.add_argument(
        "--num-workers", type=int, default=0, help="Number of data loading workers"
    )

    # Data
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./data",
        help="Directory to store/download MNIST data",
    )

    # Visualization
    parser.add_argument("--wandb", action="store_true", help="Enable WandB logging")
    parser.add_argument(
        "--tensorboard", action="store_true", help="Enable TensorBoard logging"
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="./runs/digit_recognition",
        help="Log directory for TensorBoard",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=50,
        help="Log training metrics every N batches",
    )
    parser.add_argument(
        "--viz-interval",
        type=int,
        default=2,
        help="Generate prediction grid every N epochs",
    )
    parser.add_argument("--run-name", type=str, default=None, help="Run name for WandB")

    # Checkpoint
    parser.add_argument(
        "--save-path",
        type=str,
        default="./checkpoints/digit_cnn_best.pt",
        help="Path to save best model checkpoint",
    )
    parser.add_argument(
        "--no-save", action="store_true", help="Disable model checkpoint saving"
    )

    # Seed
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )

    args = parser.parse_args()

    # ========================================================================
    # Setup
    # ========================================================================

    torch.manual_seed(args.seed)

    device = "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("  Handwritten Digit Recognition -- Training")
    print("=" * 60)
    print(f"  Device:      {device}")
    print(f"  Epochs:      {args.epochs}")
    print(f"  Batch size:  {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  WandB:       {args.wandb}")
    print(f"  TensorBoard: {args.tensorboard}")
    print("=" * 60)
    print()

    # ========================================================================
    # Data
    # ========================================================================

    print("[1/4] Loading MNIST data...")
    train_loader, test_loader = get_mnist_dataloaders(
        batch_size=args.batch_size,
        data_dir=args.data_dir,
        num_workers=args.num_workers,
    )
    print(f"      Train samples: {len(train_loader.dataset)}")
    print(f"      Test samples:  {len(test_loader.dataset)}")
    print(f"      Batches/train: {len(train_loader)}")

    # ========================================================================
    # Model
    # ========================================================================

    print("[2/4] Creating model...")
    model = DigitCNN(dropout=args.dropout)
    n_params = count_parameters(model)
    print("      Architecture: DigitCNN (2 conv + 2 fc)")
    print(f"      Parameters:   {n_params:,}")

    # ========================================================================
    # Visualizer
    # ========================================================================

    print("[3/4] Setting up visualization...")
    viz = MixedRewardVisualizer(
        use_wandb=args.wandb,
        use_tensorboard=args.tensorboard,
        log_dir=args.log_dir,
        project_name="digit-recognition",
        run_name=args.run_name,
        config={
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "dropout": args.dropout,
            "model_params": n_params,
            "device": device,
        },
    )

    # ========================================================================
    # Trainer
    # ========================================================================

    print("[4/4] Starting training...")

    trainer = DigitTrainer(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        visualizer=viz,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        device=device,
        log_interval=args.log_interval,
        viz_interval=args.viz_interval,
    )

    # ========================================================================
    # Train
    # ========================================================================

    trainer.fit(
        epochs=args.epochs,
        save_best=not args.no_save,
        save_path=args.save_path,
    )

    # ========================================================================
    # Done
    # ========================================================================

    viz.close()

    print()
    print("To view training curves:")
    if args.tensorboard:
        print(f"  tensorboard --logdir {args.log_dir}")
    if args.wandb:
        print("  Visit https://wandb.ai to view your run")
    print()
    print("Per-class accuracy chart saved to:")
    print(f"  {args.log_dir}/per_class_accuracy.png")


if __name__ == "__main__":
    main()
