"""
Generate comparison-ready training curves and a metrics summary for a Tianchi
RT-DETR training run.

Reads runs/rtdetr/<name>/results.csv (written by Ultralytics) and, unless
--no-val is passed, runs a fresh validation pass to obtain per-class mAP50.
Outputs into <run_dir>/analysis/:

  - training_curves.png   loss / mAP / precision-recall / LR curves
  - per_class_map.png     per-class mAP50 bar chart
  - metrics_summary.json  final-epoch + per-class metrics

Usage:
    python scripts/visualize_training.py --run runs/rtdetr/tianchi20
    python scripts/visualize_training.py --run runs/rtdetr/tianchi20 --no-val
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _load_results(csv_path: str) -> Optional[pd.DataFrame]:
    if not os.path.isfile(csv_path):
        print(f"[warn] results.csv not found: {csv_path}")
        return None
    df = pd.read_csv(csv_path)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _run_val(weights_pt: str, data_yaml: str, imgsz: int) -> Optional[Dict]:
    """Run a validation pass and return overall + per-class metrics."""
    if not os.path.isfile(weights_pt):
        print(f"[warn] weights not found: {weights_pt}; skipping validation")
        return None
    try:
        from ultralytics import YOLO

        model = YOLO(weights_pt)
        res = model.val(data=data_yaml, imgsz=imgsz, split="val", plots=False, verbose=False)
        box = res.box
        names = list(model.names.values()) if model.names else []

        per_class: Dict[str, float] = {}
        class_ids = getattr(box, "ap_class_index", None)
        if class_ids is not None and hasattr(box, "ap50"):
            ap50 = box.ap50  # per-class AP50, aligned with ap_class_index
            for j, cid in enumerate(class_ids):
                name = names[int(cid)] if int(cid) < len(names) else str(cid)
                per_class[name] = float(ap50[j])

        return {
            "map50": float(box.map50),
            "map": float(box.map),
            "map75": float(getattr(box, "map75", None)) if hasattr(box, "map75") else None,
            "precision": float(box.mp),
            "recall": float(box.mr),
            "per_class_ap50": per_class,
        }
    except Exception as e:  # pragma: no cover - defensive
        print(f"[warn] validation failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _line(ax, df, cols, ylabel, title=None):
    """Plot one or more columns against epoch."""
    for col in cols:
        if col in df.columns:
            ax.plot(df["epoch"], df[col], label=col, linewidth=1.8)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("epoch")
    ax.grid(alpha=0.3)
    if title:
        ax.set_title(title)
    ax.legend(fontsize=8, loc="best")


def _plot_curves(df: pd.DataFrame, out_png: str) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("RT-DETR Tianchi-20 training curves", fontsize=14, fontweight="bold")

    _line(axes[0, 0], df, ["train/giou_loss", "train/cls_loss", "train/l1_loss"],
          "train loss", "Training losses")
    _line(axes[0, 1], df, ["val/giou_loss", "val/cls_loss", "val/l1_loss"],
          "val loss", "Validation losses")
    _line(axes[0, 2], df, ["metrics/mAP50(B)", "metrics/mAP50-95(B)"],
          "mAP", "Detection mAP")
    _line(axes[1, 0], df, ["metrics/precision(B)", "metrics/recall(B)"],
          "score", "Precision / Recall")
    _line(axes[1, 1], df, ["lr/pg0"], "learning rate", "Learning rate")

    # Summary text panel
    ax = axes[1, 2]
    ax.axis("off")
    final = df.iloc[-1]
    lines = [
        "Final epoch metrics",
        f"mAP50      = {final.get('metrics/mAP50(B)', float('nan')):.4f}",
        f"mAP50-95   = {final.get('metrics/mAP50-95(B)', float('nan')):.4f}",
        f"Precision  = {final.get('metrics/precision(B)', float('nan')):.4f}",
        f"Recall     = {final.get('metrics/recall(B)', float('nan')):.4f}",
        "",
        f"epochs     = {len(df)}",
        f"best mAP50 = {df['metrics/mAP50(B)'].max():.4f}",
    ]
    ax.text(0.05, 0.95, "\n".join(lines), va="top", fontsize=11, family="monospace")

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"  saved {out_png}")


def _plot_per_class(per_class: Dict[str, float], out_png: str) -> None:
    if not per_class:
        print("  [warn] no per-class metrics to plot")
        return
    items = sorted(per_class.items(), key=lambda kv: kv[1])
    names = [k for k, _ in items]
    vals = [v for _, v in items]

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = ["#dc2626" if v < 0.5 else "#16a34a" for v in vals]
    ax.barh(names, vals, color=colors)
    ax.axvline(0.5, color="gray", linestyle="--", linewidth=1, label="mAP50 = 0.5")
    ax.set_xlabel("mAP50")
    ax.set_title("Per-class mAP50 (RT-DETR, Tianchi-20)", fontweight="bold")
    for i, v in enumerate(vals):
        ax.text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=9)
    ax.set_xlim(0, 1.0)
    ax.grid(axis="x", alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"  saved {out_png}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="runs/rtdetr/tianchi20", help="Run directory")
    parser.add_argument("--data", default="data/tianchi/data.yaml", help="Dataset yaml")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--no-val", action="store_true", help="Skip the validation pass")
    parser.add_argument("--out", default="", help="Output dir (default <run>/analysis)")
    args = parser.parse_args()

    run_dir = os.path.abspath(args.run)
    out_dir = args.out or os.path.join(run_dir, "analysis")
    os.makedirs(out_dir, exist_ok=True)

    df = _load_results(os.path.join(run_dir, "results.csv"))
    if df is not None:
        _plot_curves(df, os.path.join(out_dir, "training_curves.png"))

    val_metrics = None
    if not args.no_val:
        weights_pt = os.path.join(run_dir, "weights", "best.pt")
        val_metrics = _run_val(weights_pt, args.data, args.imgsz)
        if val_metrics:
            _plot_per_class(val_metrics["per_class_ap50"], os.path.join(out_dir, "per_class_map.png"))

    summary: Dict = {}
    if df is not None:
        final = df.iloc[-1]
        summary["final_epoch"] = int(final["epoch"])
        summary["map50"] = round(float(final["metrics/mAP50(B)"]), 4)
        summary["map50_95"] = round(float(final["metrics/mAP50-95(B)"]), 4)
        summary["precision"] = round(float(final["metrics/precision(B)"]), 4)
        summary["recall"] = round(float(final["metrics/recall(B)"]), 4)
        summary["best_map50"] = round(float(df["metrics/mAP50(B)"].max()), 4)
        summary["best_map50_epoch"] = int(df["metrics/mAP50(B)"].idxmax() + 1)
    if val_metrics:
        summary["val"] = {k: v for k, v in val_metrics.items() if k != "per_class_ap50"}
        summary["per_class_ap50"] = val_metrics["per_class_ap50"]

    with open(os.path.join(out_dir, "metrics_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"  saved {os.path.join(out_dir, 'metrics_summary.json')}")

    print(f"\nDone. Analysis artifacts in {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
