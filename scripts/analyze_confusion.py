"""
Diagnose WHY critical-class recall is stuck: misclassification vs miss.

The 18-class RT-DETR baseline's critical classes (hole / broken_warp /
broken_spandex) have recall stuck at ~0.48–0.72 despite four failed
training-signal interventions (Layer-1 reward, Layer-3 GRPO, severity full-term,
severity positive-only). A per-class instance count showed these classes are
*not* the long tail (they are mid-frequency: 273 / 681 / 492 instances); the
true tail is flower_board / hundred_feet / hanging_warp / hair_particle /
coarse_warp. So the bottleneck is likely *confusability* (label noise /
inter-class ambiguity), not rarity.

This script runs the baseline on ``val`` and extracts the ultralytics confusion
matrix to answer, per class: how many GT instances were (a) detected correctly,
(b) misclassified as another class, (c) missed entirely (predicted as
background).

Usage:
    python scripts/analyze_confusion.py
    python scripts/analyze_confusion.py --weights runs/rtdetr-cost/severity_pos/weights/best.pt
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ultralytics import YOLO
from ultralytics.models.rtdetr.val import RTDETRValidator

CRITICAL_IDS = [0, 9, 15]  # hole, broken_warp, broken_spandex
CRITICAL_NAMES = ["hole", "broken_warp", "broken_spandex"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--weights", default="runs/rtdetr/tianchi18_full/weights/best.pt"
    )
    parser.add_argument("--data", default="data/tianchi/data.yaml")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default="0")
    parser.add_argument("--project", default="runs/confusion")
    args = parser.parse_args()

    save_dir = Path(args.project)
    save_dir.mkdir(parents=True, exist_ok=True)

    val_args = {
        "data": args.data,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "workers": 0,
        "plots": True,  # ensures the confusion matrix is populated
        "rect": True,
        "mode": "val",
        "verbose": False,
    }
    validator = RTDETRValidator(args=val_args, save_dir=save_dir)
    model = YOLO(args.weights)
    validator(model=model.model)

    nc = validator.nc
    names = validator.names
    cm = validator.metrics.confusion_matrix.matrix  # (nc+1) x (nc+1), last idx = bg
    bg = nc  # background index

    print(
        f"\nConfusion matrix shape: {cm.shape} (rows=GT, cols=pred, idx {nc}=background)\n"
    )

    # Full matrix dump for reference.
    with open(save_dir / "confusion_matrix.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(
            ["gt_class"] + [names[i] if i < nc else "background" for i in range(nc + 1)]
        )
        for i in range(nc + 1):
            gt_name = names[i] if i < nc else "background"
            w.writerow([gt_name] + [int(cm[i, j]) for j in range(nc + 1)])

    # Per-class breakdown, sorted by misclassification rate.
    print(
        f"{'class':<14} {'correct':>8} {'miss':>6} {'confused':>8}  top confusions (pred <- count)"
    )
    rows = []
    for c in range(nc):
        row = cm[c]
        total = int(row.sum())
        if total == 0:
            continue
        correct = int(row[c])
        missed = int(row[bg])
        confused = total - correct - missed
        conf = sorted(
            ((j, int(row[j])) for j in range(nc) if j != c and row[j] > 0),
            key=lambda x: -x[1],
        )
        conf_str = "  ".join(f"{names[j]}<-{v}" for j, v in conf[:4])
        rows.append((c, total, correct, missed, confused, conf_str))
    rows.sort(key=lambda x: -(x[3] + x[4]) / x[1])  # worst recall first

    for c, _total, correct, missed, confused, conf_str in rows:
        tag = " *CRIT*" if c in CRITICAL_IDS else ""
        print(f"{names[c]:<14} {correct:>8} {missed:>6} {confused:>8}  {conf_str}{tag}")

    print("\nCritical-class summary:")
    for c in CRITICAL_IDS:
        row = cm[c]
        total = int(row.sum())
        if total == 0:
            print(f"  {names[c]}: no GT instances in val")
            continue
        correct = int(row[c])
        missed = int(row[bg])
        confused = total - correct - missed
        print(
            f"  {names[c]}: recall={correct / total:.3f}  correct={correct}  "
            f"missed(bg)={missed}  misclassified={confused}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
