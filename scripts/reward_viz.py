"""
Visualization helpers for reward-injected RT-DETR training runs.

Reads the Ultralytics ``results.csv`` (per-epoch supervised losses + val mAP)
and the reward ``reward_history.csv`` (per-step reward stats written by
``train_rtdetr_reward.py``) and renders a single figure into the run directory.

Kept dependency-light: only ``csv`` + ``matplotlib`` (matplotlib is already a
transitive dependency of ultralytics).
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import List, Optional

import matplotlib

matplotlib.use("Agg")  # headless backend; safe under the training loop
import matplotlib.pyplot as plt


def _read_results_csv(save_dir: Path) -> Optional[dict]:
    """Return per-epoch columns from ``results.csv`` (or None if absent)."""
    path = save_dir / "results.csv"
    if not path.exists():
        return None
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    out: dict = {}
    for key in rows[0]:
        out[key.strip()] = [float(r[key]) for r in rows]
    return out


def _read_reward_csv(save_dir: Path) -> Optional[dict]:
    """Return per-step reward columns from ``reward_history.csv`` (or None)."""
    path = save_dir / "reward_history.csv"
    if not path.exists():
        return None
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    return {key.strip(): [float(r[key]) for r in rows] for key in rows[0]}


def _col(data: dict, *needles: str) -> Optional[List[float]]:
    """First column whose name contains every needle (e.g. "giou", "train")."""
    for name, values in data.items():
        if all(n in name for n in needles):
            return values
    return None


def _rolling(x: List[float], window: int) -> List[float]:
    """Running mean over ``window`` (head padded with partial means)."""
    res: List[float] = []
    run = 0.0
    for i, v in enumerate(x):
        run += v
        if i >= window:
            run -= x[i - window]
            res.append(run / window)
        else:
            res.append(run / (i + 1))
    return res


def plot_reward_training(save_dir: str | Path, window: int = 50) -> Optional[Path]:
    """
    Render loss/reward/mAP curves for a run into ``save_dir/reward_training.png``.

    Args:
        save_dir: Ultralytics run directory (contains ``results.csv`` and, for
            reward runs, ``reward_history.csv``).
        window: Rolling-mean window for the per-step reward curves.

    Returns:
        Path to the written PNG, or None if there is nothing to plot.
    """
    save_dir = Path(save_dir)
    results = _read_results_csv(save_dir)
    reward = _read_reward_csv(save_dir)
    if results is None and reward is None:
        return None

    nrows = 2
    fig, axes = plt.subplots(
        nrows, 2, figsize=(14, 9), squeeze=False, constrained_layout=True
    )
    ax_loss, ax_reward, ax_mean, ax_map = (
        axes[0][0],
        axes[0][1],
        axes[1][0],
        axes[1][1],
    )

    # (0,0) supervised loss curves vs epoch
    if results is not None:
        epochs = results.get("epoch")
        if epochs is not None:
            for label, needles in (
                ("giou", ("train", "giou")),
                ("cls", ("train", "cls")),
                ("l1", ("train", "l1")),
            ):
                vals = _col(results, *needles)
                if vals:
                    ax_loss.plot(epochs, vals, label=label)
            ax_loss.set_title("Supervised losses (per epoch)")
            ax_loss.set_xlabel("epoch")
            ax_loss.set_ylabel("loss")
            ax_loss.legend()
            ax_loss.grid(alpha=0.3)
    if not ax_loss.has_data():
        ax_loss.text(0.5, 0.5, "no results.csv", ha="center", va="center")

    # (0,1) loss_reward vs step (raw + rolling)
    if reward is not None:
        steps = list(range(len(reward["loss_reward"])))
        lr = reward["loss_reward"]
        ax_reward.plot(
            steps, lr, lw=0.6, alpha=0.5, color="tab:red", label="loss_reward"
        )
        ax_reward.plot(
            steps, _rolling(lr, window), color="tab:red", label=f"rolling({window})"
        )
        ax_reward.set_title("loss_reward = -beta * mean(R) (per step)")
        ax_reward.set_xlabel("step")
        ax_reward.set_ylabel("loss_reward")
        ax_reward.legend()
        ax_reward.grid(alpha=0.3)
    else:
        ax_reward.text(0.5, 0.5, "no reward_history.csv", ha="center", va="center")

    # (1,0) mean scalar reward R vs step (interpretable: +1 good, -1 bad)
    if reward is not None:
        steps = list(range(len(reward["mean_reward"])))
        mr = reward["mean_reward"]
        ax_mean.axhline(0, color="k", lw=0.5, ls="--")
        ax_mean.plot(steps, mr, lw=0.6, alpha=0.5, color="tab:blue", label="mean(R)")
        ax_mean.plot(
            steps, _rolling(mr, window), color="tab:blue", label=f"rolling({window})"
        )
        ax_mean.set_title("7-dim scalar reward mean(R) (per step)")
        ax_mean.set_xlabel("step")
        ax_mean.set_ylabel("mean reward")
        ax_mean.set_ylim(-1.05, 1.05)
        ax_mean.legend()
        ax_mean.grid(alpha=0.3)
    else:
        ax_mean.text(0.5, 0.5, "no reward_history.csv", ha="center", va="center")

    # (1,1) val mAP vs epoch
    if results is not None:
        epochs = results.get("epoch")
        if epochs is not None:
            for label, needles in (("mAP50", ("mAP50",)), ("mAP50-95", ("mAP50-95",))):
                vals = _col(results, *needles)
                if vals:
                    ax_map.plot(epochs, vals, label=label)
            ax_map.set_title("Validation mAP (per epoch)")
            ax_map.set_xlabel("epoch")
            ax_map.set_ylabel("mAP")
            ax_map.legend()
            ax_map.grid(alpha=0.3)
    if not ax_map.has_data():
        ax_map.text(0.5, 0.5, "no results.csv", ha="center", va="center")

    out = save_dir / "reward_training.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out
