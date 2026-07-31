"""
Hybrid Reward Trainer for fabric defect detection.

Section 5.3 of the project document.

Combines 7-dimension rewards with scalar aggregation and consistency loss
for multi-objective defect detection optimization.

Key formula:
  L_total = L_detection + beta * L_scalar + gamma * L_consistency

Where L_scalar = -R_scalar, R_scalar = sum(w_k * R_dim_k) + alpha * R_consistency
"""

from __future__ import annotations

import sys
import time
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from backend.training.augmentation import build_augmentation
from backend.training.dimension_rewards import (
    DIMENSION_NAMES,
    NUM_DIMENSIONS,
    DimensionRewardComputer,
)
from backend.training.loss_functions import total_loss
from backend.training.metrics import TrainingMetrics


class HybridRewardTrainer:
    """
    Multi-dimensional hybrid reward trainer for defect detection.

    Implements the full training loop described in Section 5.3:
    1. Forward pass through model
    2. Compute 7 dimension rewards
    3. Compute scalar reward (weighted + consistency)
    4. Compute total loss: detection + scalar + consistency
    5. Backward pass + optimizer step
    6. Adaptive weight update per epoch
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        visualizer=None,
        # Hyperparameters (Section 5.4)
        learning_rate: float = 0.001,
        weight_decay: float = 1e-4,
        lambda_miss: float = 1.5,
        lambda_fp: float = 1.0,
        beta: float = 0.1,
        gamma: float = 0.05,
        alpha: float = 0.3,
        weight_update_freq: int = 5,
        # Augmentation
        augmentation_config: Optional[dict] = None,
        # Device
        device: Optional[str] = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.visualizer = visualizer

        # Hyperparameters
        self.beta = beta
        self.gamma = gamma
        self.alpha = alpha
        self.weight_update_freq = weight_update_freq

        # Learnable dimension weights (Section 5.2.2)
        self.dim_weights = nn.Parameter(torch.ones(NUM_DIMENSIONS) / NUM_DIMENSIONS)

        # Optimizer
        self.optimizer = optim.AdamW(
            [*list(model.parameters()), self.dim_weights],
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        # Scheduler (cosine with warmup — Section 5.4)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=200)

        # Reward computer
        self.reward_computer = DimensionRewardComputer(
            lambda_miss=lambda_miss,
            lambda_fp=lambda_fp,
        )

        # Augmentation
        self.augmentation = build_augmentation(augmentation_config)

        # Metrics tracking
        self.metrics = TrainingMetrics()

        # State
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_reward = float("-inf")
        self.dim_history: List[torch.Tensor] = []  # Per-epoch dim reward history

    # ========================================================================
    # Training Step
    # ========================================================================

    def train_step(
        self,
        images: torch.Tensor,
        targets: List[Dict],
    ) -> Dict[str, float]:
        """
        Single training step (Section 5.3 train_step).

        Returns:
            Dict of all loss and reward values for logging.
        """
        images = images.to(self.device)
        self.model.train()

        # Apply augmentation
        images, targets = self.augmentation(images, targets)

        # Forward pass
        predictions = self.model(images)

        # Ensure predictions dict has required keys
        predictions = self._normalize_predictions(predictions)

        # 1. Compute 7 dimension rewards
        dim_rewards = self.reward_computer.compute_all(predictions, targets)

        # 2. Compute total loss
        losses = total_loss(
            predictions=predictions,
            targets=targets,
            dim_rewards=dim_rewards,
            dim_weights=self.dim_weights,
            beta=self.beta,
            gamma=self.gamma,
            alpha=self.alpha,
        )

        # 3. Backward
        self.optimizer.zero_grad()
        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        self.global_step += 1

        # Collect scalar values for logging
        result = {
            "total_loss": losses["total"].item(),
            "det_loss": losses["detection"].item(),
            "scalar_loss": losses["scalar"].item(),
            "consistency_loss": losses["consistency"].item(),
        }
        for name in DIMENSION_NAMES:
            result[f"dim_{name}"] = dim_rewards[name].item()

        return result

    # ========================================================================
    # Epoch Training
    # ========================================================================

    def train_epoch(self) -> Dict[str, float]:
        """Train for one full epoch."""
        epoch_losses = dict.fromkeys(DIMENSION_NAMES, 0.0)
        total_loss_sum = 0.0
        det_loss_sum = 0.0
        n_batches = len(self.train_loader)

        epoch_start = time.time()

        for batch_idx, (images, targets) in enumerate(self.train_loader):
            step_result = self.train_step(images, targets)

            # Accumulate
            total_loss_sum += step_result["total_loss"]
            det_loss_sum += step_result["det_loss"]
            for name in DIMENSION_NAMES:
                epoch_losses[name] += step_result[f"dim_{name}"]

            # Log at intervals
            if batch_idx % 50 == 0 and self.visualizer is not None:
                self.visualizer.log_training_metrics(step_result, step=self.global_step)
                pct = 100.0 * (batch_idx + 1) / n_batches
                print(
                    f"\r  Epoch {self.current_epoch:4d} | {pct:5.1f}% | "
                    f"Loss: {step_result['total_loss']:.4f} | "
                    f"Dim: [{step_result['dim_loc']:.2f}, {step_result['dim_cls']:.2f}, "
                    f"{step_result['dim_broken']:.2f}, {step_result['dim_stitch']:.2f}]",
                    end="",
                    file=sys.stderr,
                )

        elapsed = time.time() - epoch_start

        # Average
        avg = {
            "total_loss": total_loss_sum / n_batches,
            "det_loss": det_loss_sum / n_batches,
        }
        for name in DIMENSION_NAMES:
            avg[f"dim_{name}"] = epoch_losses[name] / n_batches

        avg["time"] = elapsed
        return avg

    # ========================================================================
    # Validation
    # ========================================================================

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Run validation on the val loader."""
        if self.val_loader is None:
            return {}

        self.model.eval()
        val_results = dict.fromkeys(DIMENSION_NAMES, 0.0)
        val_total = 0.0
        n_batches = len(self.val_loader)

        for images, targets in self.val_loader:
            images = images.to(self.device)
            predictions = self.model(images)
            predictions = self._normalize_predictions(predictions)

            dim_rewards = self.reward_computer.compute_all(predictions, targets)
            losses = total_loss(
                predictions,
                targets,
                dim_rewards,
                self.dim_weights,
                self.beta,
                self.gamma,
                self.alpha,
            )

            val_total += losses["total"].item()
            for name in DIMENSION_NAMES:
                val_results[name] += dim_rewards[name].item()

        avg = {"val_total_loss": val_total / n_batches}
        for name in DIMENSION_NAMES:
            avg[f"val_dim_{name}"] = val_results[name] / n_batches

        return avg

    # ========================================================================
    # Full Training Loop
    # ========================================================================

    def fit(self, epochs: int = 200) -> TrainingMetrics:
        """
        Full training loop (Section 5.3).

        Args:
            epochs: Number of epochs to train.

        Returns:
            TrainingMetrics with full history.
        """
        print(f"\n{'=' * 60}")
        print("  Hybrid Reward Trainer — Fabric Defect Detection")
        print(f"{'=' * 60}")
        print(f"  Device:     {self.device}")
        print(f"  Epochs:     {epochs}")
        print(f"  Dimensions: {DIMENSION_NAMES}")
        print(f"  beta:       {self.beta}")
        print(f"  gamma:      {self.gamma}")
        print(f"{'=' * 60}\n")

        total_start = time.time()

        for epoch in range(1, epochs + 1):
            self.current_epoch = epoch

            # Train
            train_avg = self.train_epoch()
            self.scheduler.step()

            # Validate
            val_avg = self.validate()

            # Track per-epoch dim rewards for adaptive weighting
            reward_tensor = torch.tensor(
                [train_avg[f"dim_{name}"] for name in DIMENSION_NAMES]
            )
            self.dim_history.append(reward_tensor)
            self.metrics.update(train_avg, val_avg)

            # Adaptive weight update (Section 5.2.2)
            if epoch % self.weight_update_freq == 0:
                self._update_adaptive_weights()

            # Log epoch summary
            self._log_epoch(train_avg, val_avg)

            # Print epoch summary
            print(
                f"\r  Epoch {epoch:4d} | "
                f"Loss: {train_avg['total_loss']:.4f} | "
                f"Loc: {train_avg['dim_loc']:.2f} | "
                f"Cls: {train_avg['dim_cls']:.2f} | "
                f"Broken: {train_avg['dim_broken']:.2f} | "
                f"Stitch: {train_avg['dim_stitch']:.2f} | "
                f"Time: {train_avg['time']:.1f}s",
                file=sys.stderr,
            )
            print(file=sys.stderr)

            # Best model tracking
            scalar_reward = sum(train_avg[f"dim_{name}"] for name in DIMENSION_NAMES)
            if scalar_reward > self.best_val_reward:
                self.best_val_reward = scalar_reward

        total_elapsed = time.time() - total_start

        print(f"\n{'=' * 60}")
        print("  Training Complete!")
        print(f"{'=' * 60}")
        print(f"  Total time:      {total_elapsed:.1f}s")
        print(f"  Best dim reward: {self.best_val_reward:.4f}")
        print(f"{'=' * 60}")

        return self.metrics

    # ========================================================================
    # Adaptive Weights (Section 5.2.2)
    # ========================================================================

    def _update_adaptive_weights(self):
        """
        Update dimension weights based on convergence rates.

        Dimensions converging slowly get higher weight.
        """
        if len(self.dim_history) < 5:
            return

        recent = torch.stack(self.dim_history[-5:])  # (5, 7)
        convergence_rates = []
        for dim in range(NUM_DIMENSIONS):
            # Rate of change over last 5 epochs
            rate = abs((recent[-1, dim] - recent[0, dim]).item()) / 5.0
            convergence_rates.append(rate)

        # Softmax: slow-converging → higher weight
        rates_tensor = torch.tensor(convergence_rates)
        new_weights = torch.softmax(rates_tensor, dim=0)

        with torch.no_grad():
            self.dim_weights.copy_(new_weights)

    # ========================================================================
    # Logging
    # ========================================================================

    def _log_epoch(self, train_avg: dict, val_avg: dict):
        """Log epoch metrics to visualizer."""
        if self.visualizer is None:
            return

        metrics = {
            "epoch/train_loss": train_avg["total_loss"],
            "epoch/det_loss": train_avg["det_loss"],
        }
        for name in DIMENSION_NAMES:
            metrics[f"epoch/dim_{name}"] = train_avg[f"dim_{name}"]

        if val_avg:
            metrics["epoch/val_loss"] = val_avg.get("val_total_loss", 0)
            for name in DIMENSION_NAMES:
                metrics[f"epoch/val_dim_{name}"] = val_avg.get(f"val_dim_{name}", 0)

        # Current dimension weights
        weights = torch.softmax(self.dim_weights, dim=0)
        for i, name in enumerate(DIMENSION_NAMES):
            metrics[f"weights/{name}"] = weights[i].item()

        # Learning rate
        metrics["train/lr"] = self.optimizer.param_groups[0]["lr"]

        self.visualizer.log_training_metrics(metrics, step=self.current_epoch)

    # ========================================================================
    # Utilities
    # ========================================================================

    def _normalize_predictions(self, predictions) -> Dict[str, torch.Tensor]:
        """
        Ensure predictions dict has expected keys: logits, boxes, classes, confidences.

        Handles different model output formats.
        """
        if not isinstance(predictions, dict):
            # Model returns raw tensor → wrap as logits
            predictions = {"logits": predictions}

        # Default missing keys
        for key in ["logits", "boxes", "classes", "confidences"]:
            if key not in predictions:
                if key == "logits":
                    predictions[key] = torch.zeros(1, 10, device=self.device)
                elif key == "boxes":
                    predictions[key] = torch.zeros(0, 4, device=self.device)
                elif key == "classes":
                    # Derive from logits
                    if "logits" in predictions and predictions["logits"].numel() > 0:
                        predictions[key] = predictions["logits"].argmax(dim=-1)
                    else:
                        predictions[key] = torch.zeros(
                            0, dtype=torch.long, device=self.device
                        )
                elif key == "confidences":
                    if "logits" in predictions and predictions["logits"].numel() > 0:
                        predictions[key] = (
                            torch.softmax(predictions["logits"], dim=-1)
                            .max(dim=-1)
                            .values
                        )
                    else:
                        predictions[key] = torch.zeros(0, device=self.device)

        return predictions

    def state_dict(self) -> dict:
        """Get trainer state for checkpointing."""
        return {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "dim_weights": self.dim_weights.data.clone(),
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            "best_val_reward": self.best_val_reward,
            "dim_history": [d.clone() for d in self.dim_history[-100:]],
        }

    def load_state_dict(self, state_dict: dict):
        """Restore trainer state from checkpoint."""
        self.model.load_state_dict(state_dict["model"])
        self.optimizer.load_state_dict(state_dict["optimizer"])
        with torch.no_grad():
            self.dim_weights.copy_(state_dict["dim_weights"])
        self.current_epoch = state_dict["epoch"]
        self.global_step = state_dict["global_step"]
        self.best_val_reward = state_dict["best_val_reward"]
        self.dim_history = state_dict["dim_history"]
