"""
Training loop for handwritten digit recognition with live visualization.

Integrates with MixedRewardVisualizer for dual WandB + TensorBoard logging.
Designed to train on CPU (MNIST CNN is lightweight).

Usage:
    from mixed_reward.digit_recognition import DigitCNN, DigitTrainer, get_mnist_dataloaders
    from mixed_reward.visualizer import MixedRewardVisualizer

    model = DigitCNN()
    train_loader, test_loader = get_mnist_dataloaders(batch_size=64)
    viz = MixedRewardVisualizer(use_wandb=False, use_tensorboard=True)
    trainer = DigitTrainer(model, train_loader, test_loader, visualizer=viz)
    trainer.fit(epochs=5)
"""

from __future__ import annotations

import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from mixed_reward.digit_recognition.data import (
    compute_per_class_accuracy,
    create_prediction_grid,
)
from mixed_reward.visualizer import MixedRewardVisualizer

# Optional imports
try:
    import wandb

    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False

try:
    from torch.utils.tensorboard import SummaryWriter  # noqa: F401

    _TENSORBOARD_AVAILABLE = True
except ImportError:
    _TENSORBOARD_AVAILABLE = False


class DigitTrainer:
    """
    Trainer for handwritten digit recognition with live training visualization.

    Features:
    - Real-time loss and accuracy logging per batch
    - Per-class accuracy after each epoch
    - Prediction grid visualization every N epochs
    - Model checkpointing (best accuracy)
    - Learning rate scheduling
    - Console progress output with live-updating stats
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        visualizer: Optional[MixedRewardVisualizer] = None,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-5,
        device: Optional[str] = None,
        log_interval: int = 50,
        viz_interval: int = 1,
    ):
        """
        Args:
            model: The CNN model to train.
            train_loader: Training data DataLoader.
            test_loader: Test data DataLoader.
            visualizer: MixedRewardVisualizer instance (creates one if None).
            learning_rate: Initial learning rate for Adam optimizer.
            weight_decay: Weight decay for Adam.
            device: 'cpu', 'cuda', or None (auto-detect).
            log_interval: Log metrics every N batches.
            viz_interval: Generate prediction grids every N epochs.
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.log_interval = log_interval
        self.viz_interval = viz_interval

        # Optimizer
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, step_size=5, gamma=0.5
        )

        # Loss function
        self.criterion = nn.CrossEntropyLoss()

        # Visualizer (create default if not provided)
        if visualizer is None:
            self.viz = MixedRewardVisualizer(
                use_wandb=False,
                use_tensorboard=True,
                log_dir="./runs/digit_recognition",
                project_name="digit-recognition",
            )
            self._own_viz = True
        else:
            self.viz = visualizer
            self._own_viz = False

        # Tracking state
        self.current_epoch = 0
        self.global_step = 0
        self.best_accuracy = 0.0
        self.best_epoch = 0

        # Metrics history for post-training analysis
        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "train_acc": [],
            "test_loss": [],
            "test_acc": [],
        }
        self.per_class_history: List[Dict[int, float]] = []

        # Fixed samples for consistent visualization across epochs
        self._viz_images: Optional[torch.Tensor] = None
        self._viz_labels: Optional[torch.Tensor] = None

    # ========================================================================
    # Training
    # ========================================================================

    def train_epoch(self) -> Tuple[float, float]:
        """
        Train for one epoch.

        Returns:
            (average_loss, accuracy) for the epoch.
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        epoch_start = time.time()
        n_batches = len(self.train_loader)

        for batch_idx, (images, labels) in enumerate(self.train_loader):
            images, labels = images.to(self.device), labels.to(self.device)

            # Forward
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            # Backward
            loss.backward()
            self.optimizer.step()

            # Track stats
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
            self.global_step += 1

            # Log at intervals
            if batch_idx % self.log_interval == 0:
                running_loss = total_loss / (batch_idx + 1)
                running_acc = 100.0 * correct / total
                lr = self.optimizer.param_groups[0]["lr"]

                self.viz.log_training_metrics(
                    {
                        "train/loss": running_loss,
                        "train/accuracy": running_acc,
                        "train/lr": lr,
                    },
                    step=self.global_step,
                )

                # Console progress
                pct = 100.0 * (batch_idx + 1) / n_batches
                print(
                    f"\r  Epoch {self.current_epoch:3d} | "
                    f"{pct:5.1f}% | "
                    f"Loss: {running_loss:.4f} | "
                    f"Acc: {running_acc:5.1f}% | "
                    f"LR: {lr:.6f}",
                    end="",
                    file=sys.stderr,
                )

        avg_loss = total_loss / n_batches
        accuracy = 100.0 * correct / total
        elapsed = time.time() - epoch_start

        print(
            f"\r  Epoch {self.current_epoch:3d} | "
            f"100.0% | "
            f"Loss: {avg_loss:.4f} | "
            f"Acc: {accuracy:5.1f}% | "
            f"Time: {elapsed:.1f}s  ",
            file=sys.stderr,
        )

        return avg_loss, accuracy

    # ========================================================================
    # Evaluation
    # ========================================================================

    @torch.no_grad()
    def evaluate(self) -> Tuple[float, float, np.ndarray, np.ndarray]:
        """
        Evaluate on the test set.

        Returns:
            (average_loss, accuracy, all_predictions, all_labels).
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []

        for images, labels in self.test_loader:
            images, labels = images.to(self.device), labels.to(self.device)

            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

            all_preds.append(predicted.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

        avg_loss = total_loss / len(self.test_loader)
        accuracy = 100.0 * correct / total
        all_preds_arr = np.concatenate(all_preds)
        all_labels_arr = np.concatenate(all_labels)

        return avg_loss, accuracy, all_preds_arr, all_labels_arr

    # ========================================================================
    # Visualization
    # ========================================================================

    def _log_epoch_metrics(
        self,
        train_loss: float,
        train_acc: float,
        test_loss: float,
        test_acc: float,
        per_class: Dict[int, float],
    ):
        """Log all metrics for an epoch completion."""
        metrics = {
            "epoch/train_loss": train_loss,
            "epoch/train_accuracy": train_acc,
            "epoch/test_loss": test_loss,
            "epoch/test_accuracy": test_acc,
        }
        # Per-class accuracy
        for cls_idx, cls_acc in per_class.items():
            metrics[f"class_accuracy/digit_{cls_idx}"] = cls_acc * 100.0

        self.viz.log_training_metrics(metrics, step=self.current_epoch)

    def _log_prediction_grid(self):
        """Generate and log a prediction visualization grid."""
        # Get a fixed batch for consistent visualization
        if self._viz_images is None:
            batch = next(iter(self.test_loader))
            self._viz_images, self._viz_labels = batch
            self._viz_images = self._viz_images[:16]
            self._viz_labels = self._viz_labels[:16]

        images = self._viz_images.to(self.device)
        labels = self._viz_labels
        predictions = self.model.predict(images).cpu()

        fig = create_prediction_grid(images.cpu(), labels, predictions, max_samples=16)

        # Log to WandB if available
        if _WANDB_AVAILABLE and self.viz.use_wandb:
            wandb.log(
                {
                    "predictions/grid": wandb.Image(fig),
                },
                step=self.current_epoch,
            )

        # Log to TensorBoard if available
        if _TENSORBOARD_AVAILABLE and self.viz.use_tensorboard:
            # Convert figure to numpy array for TensorBoard
            fig.canvas.draw()
            # Use buffer_rgba for matplotlib 3.8+ compatibility
            img_arr = np.asarray(fig.canvas.buffer_rgba())
            self.viz.writer.add_image(
                "predictions/grid",
                img_arr[:, :, :3].transpose(2, 0, 1),
                self.current_epoch,
            )

        import matplotlib.pyplot as plt

        plt.close(fig)

    def _log_weight_histograms(self):
        """Log model weight histograms (TensorBoard only)."""
        if not _TENSORBOARD_AVAILABLE or not self.viz.use_tensorboard:
            return

        for name, param in self.model.named_parameters():
            if param.requires_grad and "weight" in name:
                self.viz.writer.add_histogram(
                    f"weights/{name}",
                    param.data.cpu(),
                    self.current_epoch,
                )
                if param.grad is not None:
                    self.viz.writer.add_histogram(
                        f"gradients/{name}",
                        param.grad.data.cpu(),
                        self.current_epoch,
                    )

    def _log_per_class_chart(self, per_class: Dict[int, float], save_path: str = ""):
        """Generate and log a per-class accuracy bar chart.

        Args:
            per_class: Dict mapping class index -> accuracy in [0, 1].
            save_path: If non-empty, also save PNG to this file path.
        """
        import matplotlib.pyplot as plt

        digits = list(range(10))
        accuracies = [per_class.get(d, 0.0) * 100.0 for d in digits]

        # Color bars: green >= 90%, yellow >= 70%, red < 70%
        colors = [
            "#2ecc71" if a >= 90 else "#f1c40f" if a >= 70 else "#e74c3c"
            for a in accuracies
        ]

        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.bar(
            digits, accuracies, color=colors, edgecolor="white", linewidth=1.2
        )

        # Value labels on bars
        for bar, acc in zip(bars, accuracies, strict=False):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1.5,
                f"{acc:.1f}%",
                ha="center",
                fontsize=11,
                fontweight="bold",
                color="#2c3e50",
            )

        ax.set_xlabel("Digit Class", fontsize=13)
        ax.set_ylabel("Accuracy (%)", fontsize=13)
        ax.set_title(
            f"Per-Class Accuracy -- Epoch {self.current_epoch}",
            fontsize=14,
            fontweight="bold",
        )
        ax.set_xticks(digits)
        ax.set_xticklabels([str(d) for d in digits], fontsize=12)
        ax.set_ylim(0, 105)
        ax.axhline(
            y=90,
            color="#bdc3c7",
            linestyle="--",
            linewidth=1,
            alpha=0.7,
            label="90% threshold",
        )
        ax.legend(fontsize=10)
        ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()

        # Save to file if path provided
        if save_path:
            import os

            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            fig.savefig(save_path, dpi=120, bbox_inches="tight")
            print(f"\n  Per-class chart saved to: {save_path}")

        # Log to WandB
        if _WANDB_AVAILABLE and self.viz.use_wandb:
            wandb.log(
                {"per_class_accuracy/chart": wandb.Image(fig)}, step=self.current_epoch
            )

        # Log to TensorBoard
        if _TENSORBOARD_AVAILABLE and self.viz.use_tensorboard:
            fig.canvas.draw()
            img_arr = np.asarray(fig.canvas.buffer_rgba())
            self.viz.writer.add_image(
                "per_class_accuracy/chart",
                img_arr[:, :, :3].transpose(2, 0, 1),
                self.current_epoch,
            )

        plt.close(fig)

    # ========================================================================
    # Main Loop
    # ========================================================================

    def fit(
        self,
        epochs: int = 10,
        save_best: bool = True,
        save_path: str = "./checkpoints/digit_cnn_best.pt",
    ) -> Dict[str, List[float]]:
        """
        Full training loop with live visualization.

        Args:
            epochs: Number of epochs to train.
            save_best: Save best model checkpoint.
            save_path: Path for best model checkpoint.

        Returns:
            Training history dict with loss and accuracy curves.
        """
        print(f"\n{'=' * 60}")
        print("  Digit Recognition Training")
        print(f"{'=' * 60}")
        print(f"  Device:      {self.device}")
        print(f"  Epochs:      {epochs}")
        print(f"  Train size:  {len(self.train_loader.dataset)}")
        print(f"  Test size:   {len(self.test_loader.dataset)}")
        print(f"  Batch size:  {self.train_loader.batch_size}")
        print("  Optimizer:   Adam")
        print(f"{'=' * 60}\n")

        import os

        total_start = time.time()

        for epoch in range(1, epochs + 1):
            self.current_epoch = epoch

            # Train
            train_loss, train_acc = self.train_epoch()

            # Step scheduler
            self.scheduler.step()

            # Evaluate
            test_loss, test_acc, all_preds, all_labels = self.evaluate()

            # Per-class accuracy
            per_class = compute_per_class_accuracy(all_preds, all_labels)

            # Store history
            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["test_loss"].append(test_loss)
            self.history["test_acc"].append(test_acc)
            self.per_class_history.append(per_class)

            # Log epoch metrics
            self._log_epoch_metrics(
                train_loss, train_acc, test_loss, test_acc, per_class
            )

            # Log weight histograms (every epoch for small model)
            self._log_weight_histograms()

            # Log per-class accuracy chart (every epoch)
            self._log_per_class_chart(per_class)

            # Log prediction grid at intervals
            if epoch % self.viz_interval == 0:
                self._log_prediction_grid()

            # Best model tracking
            if test_acc > self.best_accuracy:
                self.best_accuracy = test_acc
                self.best_epoch = epoch
                if save_best:
                    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
                    torch.save(
                        {
                            "epoch": epoch,
                            "model_state_dict": self.model.state_dict(),
                            "optimizer_state_dict": self.optimizer.state_dict(),
                            "accuracy": test_acc,
                            "history": self.history,
                        },
                        save_path,
                    )

            # Console summary for this epoch
            is_best = " *" if test_acc >= self.best_accuracy else ""
            print(
                f"  Test  | Loss: {test_loss:.4f} | Acc: {test_acc:5.1f}%{is_best}",
                file=sys.stderr,
            )
            print(file=sys.stderr)

        total_elapsed = time.time() - total_start

        # Final summary
        print(f"\n{'=' * 60}")
        print("  Training Complete!")
        print(f"{'=' * 60}")
        print(f"  Total time:     {total_elapsed:.1f}s")
        print(f"  Best accuracy:  {self.best_accuracy:.2f}% (epoch {self.best_epoch})")
        print(f"  Final accuracy: {self.history['test_acc'][-1]:.2f}%")
        print(f"{'=' * 60}")

        # Per-class final accuracy
        print("\n  Per-Class Accuracy:")
        final_per_class = self.per_class_history[-1]
        for cls_idx in range(10):
            acc = final_per_class.get(cls_idx, 0.0) * 100.0
            bar = "#" * int(acc / 5) + "-" * (20 - int(acc / 5))
            print(f"    Digit {cls_idx}: {bar} {acc:.1f}%")

        # Save per-class chart to file alongside TensorBoard logs
        import os as _os

        chart_path = _os.path.join(self.viz.log_dir, "per_class_accuracy.png")
        self.current_epoch = epochs
        self._log_per_class_chart(final_per_class, save_path=chart_path)

        if save_best:
            print(f"\n  Best model saved to: {save_path}")

        # Close visualizer if we own it
        if self._own_viz:
            self.viz.close()

        return self.history

    def close(self):
        """Clean up visualizer resources."""
        if self._own_viz:
            self.viz.close()
