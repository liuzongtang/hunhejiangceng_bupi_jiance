"""
Training process visualization.

Generates real-time charts during training:
  - Loss curves (train/val)
  - Per-dimension reward trends
  - Scalar reward evolution
  - Learning rate schedule
  - Weight distribution

Outputs: PNG charts + WandB/TensorBoard logging + console progress.
"""

from __future__ import annotations

import os
import sys
import time
from typing import Dict, List, Optional


class TrainingVisualizer:
    """
    Real-time training process visualizer.

    Generates charts and console output during training.
    """

    def __init__(
        self,
        output_dir: str = "./runs/training",
        dimension_names: Optional[List[str]] = None,
        external_visualizer=None,
    ):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.dimension_names = dimension_names or [
            "loc",
            "cls",
            "cal",
            "miss",
            "fp",
            "broken",
            "skip",
        ]
        self.external = external_visualizer

        # History for chart generation
        self.history: Dict[str, List[float]] = {
            "epoch": [],
            "train_loss": [],
            "val_loss": [],
            "det_loss": [],
            "scalar_reward": [],
            "consistency_loss": [],
            "learning_rate": [],
        }
        for name in self.dimension_names:
            self.history[f"dim_{name}"] = []

        self._start_time = time.time()

    def record_epoch(
        self,
        epoch: int,
        train_metrics: dict,
        val_metrics: Optional[dict] = None,
        lr: float = 0.0,
    ):
        """Record metrics for one epoch."""
        self.history["epoch"].append(epoch)
        self.history["train_loss"].append(train_metrics.get("total_loss", 0))
        self.history["det_loss"].append(train_metrics.get("det_loss", 0))

        if val_metrics:
            self.history["val_loss"].append(val_metrics.get("val_total_loss", 0))

        self.history["learning_rate"].append(lr)

        # Scalar reward = sum of all dimension rewards
        scalar = sum(
            train_metrics.get(f"dim_{name}", 0)
            for name in ["loc", "cls", "cal", "miss", "fp", "broken", "skip"]
        )
        self.history["scalar_reward"].append(scalar)

        # Per-dimension
        for i, name in enumerate(["loc", "cls", "cal", "miss", "fp", "broken", "skip"]):
            dim_name = (
                self.dimension_names[i] if i < len(self.dimension_names) else name
            )
            self.history[f"dim_{dim_name}"].append(train_metrics.get(f"dim_{name}", 0))

        # Generate charts
        if epoch % 5 == 0 or epoch == 1:
            self._save_charts(epoch)

        # Log to external visualizer
        if self.external:
            self.external.log_training_metrics(
                {
                    "epoch/train_loss": train_metrics.get("total_loss", 0),
                    "epoch/det_loss": train_metrics.get("det_loss", 0),
                    "epoch/scalar_reward": scalar,
                    "epoch/learning_rate": lr,
                },
                step=epoch,
            )

    def _save_charts(self, epoch: int):
        """Generate and save training charts as PNG."""
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            epochs = self.history["epoch"]

            # 2x2 subplot layout
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            fig.suptitle(
                f"训练过程可视化 — Epoch {epoch}", fontsize=14, fontweight="bold"
            )

            # 1. Loss curves
            ax = axes[0, 0]
            ax.plot(
                epochs,
                self.history["train_loss"],
                "b-",
                label="训练损失",
                linewidth=1.5,
            )
            if any(v > 0 for v in self.history["val_loss"]):
                ax.plot(
                    epochs,
                    self.history["val_loss"],
                    "r--",
                    label="验证损失",
                    linewidth=1.5,
                )
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.set_title("损失曲线")
            ax.legend()
            ax.grid(True, alpha=0.3)

            # 2. Scalar reward
            ax = axes[0, 1]
            ax.plot(epochs, self.history["scalar_reward"], "g-", linewidth=2)
            ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Scalar Reward")
            ax.set_title("标量奖惩")
            ax.grid(True, alpha=0.3)

            # 3. Dimension rewards
            ax = axes[1, 0]
            colors = [
                "#ef4444",
                "#f97316",
                "#eab308",
                "#22c55e",
                "#38bdf8",
                "#8b5cf6",
                "#ec4899",
            ]
            for _i, dim_name in enumerate(self.dimension_names):
                key = f"dim_{dim_name}"
                if self.history.get(key):
                    ax.plot(
                        epochs,
                        self.history[key],
                        color=colors[i % 7],
                        label=dim_name,
                        linewidth=1,
                        alpha=0.8,
                    )
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Dimension Reward")
            ax.set_title("维度奖惩")
            ax.legend(fontsize=8, ncol=2)
            ax.grid(True, alpha=0.3)

            # 4. Learning rate
            ax = axes[1, 1]
            ax.plot(epochs, self.history["learning_rate"], "m-", linewidth=1.5)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Learning Rate")
            ax.set_title("学习率")
            ax.grid(True, alpha=0.3)
            ax.set_yscale("log")

            plt.tight_layout()
            path = os.path.join(self.output_dir, f"training_epoch_{epoch:04d}.png")
            fig.savefig(path, dpi=120, bbox_inches="tight")
            plt.close(fig)

        except ImportError:
            pass  # matplotlib not available

    def print_epoch_summary(self, epoch: int, train_metrics: dict, elapsed: float):
        """Print a formatted epoch summary to console."""
        dims = " ".join(
            f"{name}={train_metrics.get(f'dim_{n}', 0):+.2f}"
            for n, name in zip(
                ["loc", "cls", "cal", "miss", "fp", "broken", "skip"],
                self.dimension_names,
                strict=False,
            )
        )

        total_elapsed = time.time() - self._start_time
        print(
            f"\r  Epoch {epoch:4d} | "
            f"Loss: {train_metrics['total_loss']:.4f} | "
            f"{dims} | "
            f"{elapsed:.1f}s/ep | "
            f"Total: {total_elapsed / 60:.0f}min",
            file=sys.stderr,
        )
        print(file=sys.stderr)

    def generate_summary_report(self) -> str:
        """Generate a markdown summary report after training."""
        epochs = self.history["epoch"]
        if not epochs:
            return "No training data available."

        lines = [
            "# 训练报告",
            "",
            f"**总 Epoch 数**: {len(epochs)}",
            f"**训练时长**: {(time.time() - self._start_time) / 60:.1f} 分钟",
            "",
            "## 最终指标",
            "",
            "| 指标 | 初始值 | 最终值 | 变化 |",
            "|------|--------|--------|------|",
        ]

        for key, label in [
            ("train_loss", "训练损失"),
            ("scalar_reward", "标量奖惩"),
        ]:
            if key in self.history and len(self.history[key]) >= 2:
                start, end = self.history[key][0], self.history[key][-1]
                change = "▲" if end > start else "▼"
                lines.append(
                    f"| {label} | {start:.4f} | {end:.4f} | {change} {abs(end - start):.4f} |"
                )

        lines.extend(["", "## 维度奖惩", ""])
        for _i, dim_name in enumerate(self.dimension_names):
            key = f"dim_{dim_name}"
            if key in self.history and len(self.history[key]) >= 2:
                start, end = self.history[key][0], self.history[key][-1]
                lines.append(f"- **{dim_name}**: {start:+.3f} → {end:+.3f}")

        lines.extend(["", f"图表目录: `{self.output_dir}/`"])
        return "\n".join(lines)
