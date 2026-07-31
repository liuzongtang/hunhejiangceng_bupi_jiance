"""
Mixed Reward Visualizer.

Dual-backend visualization system supporting both WandB (cloud collaboration)
and TensorBoard (local, lightweight). Provides comprehensive logging for:
- Per-dimension reward curves
- Dimension state (D vector) evolution
- Heatmaps of D over time
- Gradient scale distributions
- Pareto frontier plots

Usage:
    viz = MixedRewardVisualizer(
        use_wandb=True,
        use_tensorboard=True,
        log_dir="./runs/mixed_reward",
        project_name="mixed-reward-rlhf",
    )
    viz.set_dimensions(["accuracy", "safety", "completeness", "format"])

    # In training loop:
    viz.log_reward_vector(reward_dict, step=100)
    viz.log_dimension_state(D, step=100)
    viz.log_gradient_metrics({"grad_norm": 0.5}, step=100)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)

# Optional imports
try:
    import wandb

    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False

try:
    from torch.utils.tensorboard import SummaryWriter

    _TENSORBOARD_AVAILABLE = True
except ImportError:
    _TENSORBOARD_AVAILABLE = False

try:
    import matplotlib

    matplotlib.use("Agg")  # Non-interactive backend
    import matplotlib.pyplot as plt

    _MPL_AVAILABLE = True
except ImportError:
    _MPL_AVAILABLE = False


class MixedRewardVisualizer:
    """
    Dual-backend visualizer for the mixed reward mechanism.

    Supports WandB, TensorBoard, or both simultaneously.
    All methods are safe to call regardless of backend availability.
    """

    def __init__(
        self,
        use_wandb: bool = True,
        use_tensorboard: bool = True,
        log_dir: str = "./runs/mixed_reward",
        project_name: str = "mixed-reward-rlhf",
        wandb_entity: Optional[str] = None,
        run_name: Optional[str] = None,
        log_interval: int = 10,
        heatmap_interval: int = 100,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Args:
            use_wandb: Enable WandB logging.
            use_tensorboard: Enable TensorBoard logging.
            log_dir: Directory for TensorBoard logs.
            project_name: WandB project name.
            wandb_entity: WandB entity (team/user).
            run_name: WandB run name.
            log_interval: Log scalar metrics every N steps.
            heatmap_interval: Generate heatmap figure every N steps.
            config: Optional config dict to log to WandB.
        """
        self.use_wandb = use_wandb and _WANDB_AVAILABLE
        self.use_tensorboard = use_tensorboard and _TENSORBOARD_AVAILABLE
        self.log_interval = log_interval
        self.heatmap_interval = heatmap_interval

        if use_wandb and not _WANDB_AVAILABLE:
            logger.warning(
                "WandB requested but not installed. Disabling WandB logging."
            )

        if use_tensorboard and not _TENSORBOARD_AVAILABLE:
            logger.warning(
                "TensorBoard requested but not installed. Disabling TB logging."
            )

        # Initialize WandB
        if self.use_wandb:
            wandb.init(
                project=project_name,
                entity=wandb_entity,
                name=run_name,
                config=config or {},
                reinit=True,
            )

        # Initialize TensorBoard
        self.log_dir = log_dir
        if self.use_tensorboard:
            self.writer = SummaryWriter(log_dir=log_dir)
        else:
            self.writer = None

        # State
        self.dimensions: Optional[List[str]] = None
        self._step = 0

    def set_dimensions(self, dimensions: List[str]):
        """Set dimension names for labeling."""
        self.dimensions = dimensions

    def _dim_label(self, idx: int) -> str:
        """Get a human-readable label for dimension idx."""
        if self.dimensions and idx < len(self.dimensions):
            return self.dimensions[idx]
        return f"dim_{idx}"

    # ========================================================================
    # Reward Logging
    # ========================================================================

    def log_reward_vector(
        self,
        reward_dict: Dict[str, float],
        step: Optional[int] = None,
        prefix: str = "reward",
    ):
        """
        Log per-dimension and total rewards.

        Args:
            reward_dict: Dict mapping dimension name → score.
            step: Training step (uses internal counter if None).
            prefix: Metric prefix for grouping.
        """
        if step is None:
            step = self._step

        total = sum(reward_dict.values())
        metrics = {f"{prefix}/{k}": v for k, v in reward_dict.items()}
        metrics[f"{prefix}/total"] = total
        self._log_metrics(metrics, step)

    def log_reward_batch(
        self,
        batch_rewards: List[Dict[str, float]],
        step: Optional[int] = None,
    ):
        """
        Log batch-aggregated reward statistics.

        Args:
            batch_rewards: List of per-response reward dicts.
            step: Training step.
        """
        if step is None:
            step = self._step

        if not batch_rewards:
            return

        # Aggregate per dimension
        all_dims = set()
        for rd in batch_rewards:
            all_dims.update(rd.keys())

        metrics = {}
        for dim in sorted(all_dims):
            values = [rd.get(dim, 0.0) for rd in batch_rewards]
            arr = np.array(values)
            metrics[f"reward_batch/{dim}_mean"] = float(arr.mean())
            metrics[f"reward_batch/{dim}_std"] = float(arr.std())
            metrics[f"reward_batch/{dim}_min"] = float(arr.min())
            metrics[f"reward_batch/{dim}_max"] = float(arr.max())

        # Total reward stats
        totals = [sum(rd.values()) for rd in batch_rewards]
        totals_arr = np.array(totals)
        metrics["reward_batch/total_mean"] = float(totals_arr.mean())
        metrics["reward_batch/total_std"] = float(totals_arr.std())

        self._log_metrics(metrics, step)

    # ========================================================================
    # Dimension State Logging
    # ========================================================================

    def log_dimension_state(
        self,
        D: Union[np.ndarray, "torch.Tensor", List[float]],
        step: Optional[int] = None,
    ):
        """
        Log current dimension state values.

        Args:
            D: Dimension state vector of shape (num_dimensions,).
            step: Training step.
        """
        if step is None:
            step = self._step

        # Convert to numpy
        if hasattr(D, "detach"):
            D = D.detach().numpy()
        D_arr = np.asarray(D).flatten()

        metrics = {}
        for i, val in enumerate(D_arr):
            metrics[f"dim_state/{self._dim_label(i)}"] = float(val)

        self._log_metrics(metrics, step)

        # Log histogram distribution
        if self.use_tensorboard and self.writer is not None:
            self.writer.add_histogram("dim_state/distribution", D_arr, step)

    def log_dimension_state_heatmap(
        self,
        D: Union[np.ndarray, "torch.Tensor", List[float]],
        step: Optional[int] = None,
        history: Optional[np.ndarray] = None,
    ):
        """
        Log a heatmap visualization of dimension state.

        Two modes:
        - Single D vector: renders as 1×N heatmap bar
        - History array: renders as T×N heatmap over time

        Args:
            D: Current dimension state vector.
            step: Training step.
            history: Optional full history (steps, num_dimensions).
        """
        if step is None:
            step = self._step

        if not self.use_wandb or not _MPL_AVAILABLE:
            return

        # Convert to numpy
        if hasattr(D, "detach"):
            D = D.detach().numpy()
        D_arr = np.asarray(D).flatten()

        fig, ax = plt.subplots(figsize=(10, 3))

        if history is not None and len(history) > 1:
            # Full history heatmap
            ax.imshow(
                history.T,
                aspect="auto",
                cmap="RdYlGn",
                vmin=0.5,
                vmax=2.0,
                interpolation="nearest",
            )
            ax.set_ylabel("Dimension")
            ax.set_xlabel("Step")
            # Set y-tick labels
            if self.dimensions:
                ax.set_yticks(range(len(self.dimensions)))
                ax.set_yticklabels(self.dimensions)
        else:
            # Single-step bar
            x = np.arange(len(D_arr))
            colors = [
                "#2ecc71" if v > 1.0 else "#e74c3c" if v < 1.0 else "#95a5a6"
                for v in D_arr
            ]
            ax.bar(x, D_arr, color=colors, edgecolor="white", linewidth=1)
            ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=1, alpha=0.5)
            ax.set_xticks(x)
            if self.dimensions:
                ax.set_xticklabels(self.dimensions, rotation=45, ha="right")
            ax.set_ylabel("D Value")
            ax.set_ylim(0.4, 2.1)
            # Add value labels
            for i, v in enumerate(D_arr):
                ax.text(i, v + 0.05, f"{v:.3f}", ha="center", fontsize=9)

        ax.set_title(f"Dimension State at Step {step}")
        plt.tight_layout()

        wandb.log({"dim_state/heatmap": wandb.Image(fig)}, step=step)
        plt.close(fig)

    # ========================================================================
    # Gradient Metrics
    # ========================================================================

    def log_gradient_metrics(
        self,
        grad_metrics: Dict[str, float],
        step: Optional[int] = None,
    ):
        """
        Log gradient-related metrics.

        Args:
            grad_metrics: Dict of metric name → value.
                          Typical keys: grad_norm, grad_scale_mean, clip_ratio.
            step: Training step.
        """
        if step is None:
            step = self._step

        metrics = {f"grad/{k}": v for k, v in grad_metrics.items()}
        self._log_metrics(metrics, step)

    def log_gradient_scales(
        self,
        scales: List[float],
        step: Optional[int] = None,
    ):
        """
        Log per-parameter-group gradient modulation scales.

        Args:
            scales: List of scale factors per group.
            step: Training step.
        """
        if step is None:
            step = self._step

        if not scales:
            return

        metrics = {}
        for i, scale in enumerate(scales):
            metrics[f"grad_scale/group_{i}"] = scale
        metrics["grad_scale/mean"] = float(np.mean(scales))
        metrics["grad_scale/std"] = float(np.std(scales))
        metrics["grad_scale/min"] = float(np.min(scales))
        metrics["grad_scale/max"] = float(np.max(scales))

        self._log_metrics(metrics, step)

    # ========================================================================
    # Pareto Frontier
    # ========================================================================

    def log_pareto_frontier(
        self,
        reward_vectors: np.ndarray,
        step: Optional[int] = None,
        dim_x: int = 0,
        dim_y: int = 1,
    ):
        """
        Render and log a 2D Pareto frontier scatter plot.

        Args:
            reward_vectors: Array of shape (N, num_dimensions).
            step: Training step.
            dim_x: Dimension index for x-axis.
            dim_y: Dimension index for y-axis.
        """
        if step is None:
            step = self._step

        if not self.use_wandb or not _MPL_AVAILABLE:
            return

        if reward_vectors.shape[0] < 2:
            return

        fig, ax = plt.subplots(figsize=(8, 6))

        x = reward_vectors[:, dim_x]
        y = reward_vectors[:, dim_y]

        # Plot all points
        ax.scatter(x, y, alpha=0.5, s=30, label="All samples", color="#3498db")

        # Compute and plot Pareto frontier (maximization in both dimensions)
        pareto_mask = _compute_pareto_frontier(x, y)
        if pareto_mask.any():
            ax.scatter(
                x[pareto_mask],
                y[pareto_mask],
                alpha=0.9,
                s=60,
                label="Pareto frontier",
                color="#e74c3c",
                edgecolors="black",
                linewidth=1,
            )
            # Connect Pareto points
            pareto_order = np.argsort(x[pareto_mask])
            px = x[pareto_mask][pareto_order]
            py = y[pareto_mask][pareto_order]
            ax.plot(px, py, "r--", alpha=0.5, linewidth=1)

        x_label = self._dim_label(dim_x)
        y_label = self._dim_label(dim_y)
        ax.set_xlabel(f"{x_label} Reward")
        ax.set_ylabel(f"{y_label} Reward")
        ax.set_title(f"Pareto Frontier ({x_label} vs {y_label}) at Step {step}")
        ax.legend()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)

        wandb.log({f"pareto/{x_label}_vs_{y_label}": wandb.Image(fig)}, step=step)
        plt.close(fig)

    # ========================================================================
    # Training Metrics
    # ========================================================================

    def log_training_metrics(
        self,
        metrics: Dict[str, Any],
        step: Optional[int] = None,
    ):
        """
        Log arbitrary training metrics.

        Filters out non-scalar values and logs only scalars to WandB/TB.
        Non-scalar values are silently skipped.

        Args:
            metrics: Dict of metric name → scalar value.
            step: Training step.
        """
        if step is None:
            step = self._step

        scalar_metrics = {}
        for k, v in metrics.items():
            if isinstance(v, (int, float, np.floating, np.integer)):
                scalar_metrics[k] = float(v)
            elif hasattr(v, "item"):
                import contextlib

                with contextlib.suppress(ValueError, TypeError):
                    scalar_metrics[k] = float(v.item())

        self._log_metrics(scalar_metrics, step)

    # ========================================================================
    # Internal
    # ========================================================================

    def _log_metrics(self, metrics: Dict[str, float], step: int):
        """Log metrics to both backends."""
        if self.use_wandb:
            wandb.log(metrics, step=step)

        if self.use_tensorboard and self.writer is not None:
            for key, value in metrics.items():
                self.writer.add_scalar(key, value, step)

    def step_counter(self):
        """Increment and return the internal step counter."""
        self._step += 1
        return self._step

    def close(self):
        """Clean up visualization backends."""
        if self.use_wandb:
            wandb.finish()
        if self.use_tensorboard and self.writer is not None:
            self.writer.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# ============================================================================
# Helper: Pareto Frontier Computation
# ============================================================================


def _compute_pareto_frontier(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Compute the Pareto frontier (maximization) for 2D points.

    A point dominates another if it has higher or equal values in BOTH dimensions
    and strictly higher in at least one.

    Returns:
        Boolean array where True indicates a point on the Pareto frontier.
    """
    n = len(x)
    is_pareto = np.ones(n, dtype=bool)

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            # Point j dominates point i if: j_x >= i_x and j_y >= i_y
            # AND (j_x > i_x or j_y > i_y)
            if x[j] >= x[i] and y[j] >= y[i] and (x[j] > x[i] or y[j] > y[i]):
                is_pareto[i] = False
                break

    return is_pareto


# ============================================================================
# Standalone Plot Generators (for programmatic use)
# ============================================================================


def generate_dimension_heatmap(
    history: np.ndarray,
    dimension_names: List[str],
    step: int,
) -> "plt.Figure":
    """Generate a dimension state heatmap figure (returns matplotlib Figure)."""
    if not _MPL_AVAILABLE:
        raise ImportError("matplotlib is required for plot generation")

    fig, ax = plt.subplots(figsize=(12, 4))
    im = ax.imshow(
        history.T,
        aspect="auto",
        cmap="RdYlGn",
        vmin=0.5,
        vmax=2.0,
        interpolation="nearest",
    )
    ax.set_yticks(range(len(dimension_names)))
    ax.set_yticklabels(dimension_names)
    ax.set_xlabel("Training Step")
    ax.set_title(f"Dimension State Evolution (Step {step})")
    plt.colorbar(im, ax=ax, label="D Value")
    plt.tight_layout()
    return fig


def generate_reward_curves(
    steps: List[int],
    reward_history: Dict[str, List[float]],
    title: str = "Reward Curves",
) -> "plt.Figure":
    """Generate reward curves over training (returns matplotlib Figure)."""
    if not _MPL_AVAILABLE:
        raise ImportError("matplotlib is required for plot generation")

    fig, ax = plt.subplots(figsize=(10, 5))
    for dim_name, values in reward_history.items():
        ax.plot(steps[: len(values)], values, label=dim_name, linewidth=1.5, alpha=0.8)

    ax.set_xlabel("Training Step")
    ax.set_ylabel("Reward Score")
    ax.set_title(title)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    return fig
