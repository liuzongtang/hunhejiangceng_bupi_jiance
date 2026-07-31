"""
Training monitoring metrics (Section 5.5).

Tracks:
  - train_loss, val_loss
  - scalar_reward (maximizing toward 1.0)
  - Per-dimension rewards (loc, cls, cal, miss, fp, broken, stitch)
  - consistency_loss (minimizing)
  - mAP@0.5, recall (target >= 0.95)
  - Per-class accuracy via confusion matrix
"""

from typing import Dict, List, Optional

from backend.training.dimension_rewards import DIMENSION_NAMES


class TrainingMetrics:
    """
    Tracks and aggregates training metrics across epochs.

    Matches the monitoring table in Section 5.5.
    """

    def __init__(self):
        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "det_loss": [],
            "val_loss": [],
            "scalar_reward": [],
            "consistency_loss": [],
            "dim_loc": [],
            "dim_cls": [],
            "dim_cal": [],
            "dim_miss": [],
            "dim_fp": [],
            "dim_broken": [],
            "dim_stitch": [],
            "learning_rate": [],
        }
        self.best_epoch: Dict[str, int] = {}
        self.best_value: Dict[str, float] = {}

    def update(self, train_avg: dict, val_avg: Optional[dict] = None):
        """Record metrics for one epoch."""
        self.history["train_loss"].append(train_avg.get("total_loss", 0))
        self.history["det_loss"].append(train_avg.get("det_loss", 0))

        if val_avg:
            self.history["val_loss"].append(val_avg.get("val_total_loss", 0))

        for dim_name in DIMENSION_NAMES:
            key = f"dim_{dim_name}"
            self.history[key].append(train_avg.get(key, 0))

    def get_latest(self) -> dict:
        """Get latest epoch metrics."""
        return {k: v[-1] if v else 0 for k, v in self.history.items()}

    def get_best(self, metric: str, mode: str = "max") -> float:
        """
        Get best value for a metric.

        Args:
            metric: Metric name from history.
            mode: 'max' for metrics to maximize, 'min' for metrics to minimize.
        """
        values = self.history.get(metric, [])
        if not values:
            return 0.0
        if mode == "max":
            return max(values)
        return min(values)

    def get_scalar_reward_trend(self) -> float:
        """Compute scalar reward trend over last 5 epochs."""
        loc = self.history["dim_loc"]
        if len(loc) < 5:
            return 0.0
        recent = [
            sum(self.history[f"dim_{d}"][i] for d in DIMENSION_NAMES)
            for i in range(-5, 0)
        ]
        return (recent[-1] - recent[0]) / 5.0

    def summary(self) -> str:
        """Human-readable training summary."""
        latest = self.get_latest()
        lines = ["Training Metrics Summary", "-" * 40]
        lines.append(f"  Train Loss:        {latest['train_loss']:.4f}")
        lines.append(f"  Val Loss:          {latest['val_loss']:.4f}")
        lines.append(f"  Scalar Reward:     {latest['scalar_reward']:.4f}")
        lines.append(f"  Consistency Loss:  {latest['consistency_loss']:.4f}")
        for dim in DIMENSION_NAMES:
            val = latest[f"dim_{dim}"]
            bar = "#" * int((val + 2) * 10) + "-" * (20 - int((val + 2) * 10))
            lines.append(f"  Dim {dim:8s}:   {val:+.3f}  {bar}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Convert full history to dict for serialization."""
        return dict(self.history)
