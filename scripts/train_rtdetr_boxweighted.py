"""
Train an RT-DETR detector with box-loss severity weighting — Direction B (R2).

The classification severity weighting (``train_rtdetr_costsensitive.py``) was a
fourth negative: no weighting beats uniform fine-tune. The confusion-matrix +
threshold-calibration diagnosis then showed the critical classes are *missed
below threshold* with systematically low confidence — a localization problem,
not a classification one (RT-DETR's varifocal ``gt_score`` == predicted IoU).

This experiment therefore up-weights the *box* arm (L1 + GIoU) of matched
queries by the severity of their assigned class, via
``backend/training/boxweighted.py``. If critical-class localization (and hence
predicted confidence) improves, the recall/precision tradeoff at low thresholds
should stop collapsing.

Usage:
    # Severity box-weighted fine-tune (default: 5 epochs)
    python scripts/train_rtdetr_boxweighted.py --epochs 5 --preset severity

    # Control arm — identical run but with uniform box weights
    python scripts/train_rtdetr_boxweighted.py --epochs 5 --preset none --name ctl_uniform
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from ultralytics import YOLO
from ultralytics.models.rtdetr.val import RTDETRValidator
import ultralytics.models.utils.loss as _detr_loss

from backend.training.boxweighted import (
    BoxWeightedRTDETRDetectionLoss,
    set_default_box_weights,
)

# Critical / secondary class ids under the 18-class taxonomy (see data.yaml).
CRITICAL_CLASS_IDS = [0, 9, 15]  # hole, broken_warp, broken_spandex
CRITICAL_CLASS_NAMES = ["hole", "broken_warp", "broken_spandex"]
SECONDARY_CLASS_IDS = [14, 17]  # star_skip, weave_defect
TRACKED_NAMES = [*CRITICAL_CLASS_NAMES, "star_skip", "weave_defect"]

# Reference baseline (18-class tianchi18_full best.pt), measured with the same
# RTDETRValidator + per-class recall extraction used here.
BASELINE = {
    "map50": 0.541,
    "critical_recall": 0.5743,
    "per_class": {
        "hole": 0.718,
        "broken_warp": 0.521,
        "broken_spandex": 0.484,
        "star_skip": 0.686,
        "weave_defect": 0.487,
    },
}


def severity_weights(
    nc: int = 18, critical_w: float = 2.0, secondary_w: float = 1.5, base_w: float = 1.0
) -> list:
    """Build a length-``nc`` weight vector: critical > secondary > base."""
    w = [base_w] * nc
    for c in CRITICAL_CLASS_IDS:
        w[c] = critical_w
    for c in SECONDARY_CLASS_IDS:
        w[c] = secondary_w
    return w


def validate(
    rtdetr: torch.nn.Module, args: argparse.Namespace, device_str: str
) -> dict:
    """Run noise-free RT-DETR validation and return overall + per-class recall."""
    val_args = {
        "data": args.data,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": device_str,
        "workers": 0,
        "plots": False,
        "rect": True,
        "mode": "val",
        "verbose": False,
    }
    validator = RTDETRValidator(
        args=val_args, save_dir=Path(args.project) / args.name / "val"
    )
    validator(model=rtdetr)

    results = validator.metrics.results_dict
    box = validator.metrics.box
    ap_class_index = list(box.ap_class_index)
    r = list(box.r)
    per_class: dict = {}
    for c in range(validator.nc):
        name = validator.names[c]
        per_class[name] = (
            float(r[ap_class_index.index(c)]) if c in ap_class_index else 0.0
        )
    crit = [per_class[validator.names[c]] for c in CRITICAL_CLASS_IDS]
    return {
        "map50": float(results["metrics/mAP50(B)"]),
        "map50_95": float(results["metrics/mAP50-95(B)"]),
        "precision": float(results["metrics/precision(B)"]),
        "recall": float(results["metrics/recall(B)"]),
        "per_class": per_class,
        "critical_recall": sum(crit) / len(crit),
    }


def _print_metrics(tag: str, m: dict) -> None:
    """Print a compact validation summary (overall + tracked-class recall)."""
    tracked = "  ".join(f"{n}={m['per_class'][n]:.3f}" for n in TRACKED_NAMES)
    print(
        f"[box] {tag}: mAP50={m['map50']:.4f} mAP50-95={m['map50_95']:.4f} "
        f"P={m['precision']:.4f} R={m['recall']:.4f} crit_recall={m['critical_recall']:.4f}\n"
        f"        {tracked}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--weights", default="runs/rtdetr/tianchi18_full/weights/best.pt"
    )
    parser.add_argument("--data", default="data/tianchi/data.yaml")
    parser.add_argument(
        "--preset",
        default="severity",
        choices=["severity", "none"],
        help="severity = critical 2.0 / secondary 1.5 / base 1.0 on the box arm; none = uniform (control)",
    )
    parser.add_argument(
        "--critical-w", type=float, default=2.0, help="Weight for critical classes"
    )
    parser.add_argument(
        "--secondary-w", type=float, default=1.5, help="Weight for secondary classes"
    )
    parser.add_argument("--epochs", type=int, default=5, help="Fine-tune epochs")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument(
        "--lr0",
        type=float,
        default=0.0001,
        help="Fine-tune LR (lower than from-scratch)",
    )
    parser.add_argument(
        "--amp",
        type=lambda s: str(s).lower() in ("true", "1", "yes", "on"),
        default=False,
    )
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument(
        "--project", default="runs/rtdetr-box", help="Output project dir"
    )
    parser.add_argument("--name", default="severity", help="Run name")
    parser.add_argument(
        "--no-val", action="store_true", help="Skip the final per-class recall report"
    )
    args = parser.parse_args()

    # Resolve project to an absolute path so Ultralytics does not insert the
    # task name (runs/detect/<project>) for relative paths.
    args.project = os.path.abspath(args.project)

    # Build severity weights and inject the weighted loss class.
    if args.preset == "severity":
        weights = severity_weights(18, args.critical_w, args.secondary_w, 1.0)
    else:
        weights = [1.0] * 18
    set_default_box_weights(weights)
    _detr_loss.RTDETRDetectionLoss = BoxWeightedRTDETRDetectionLoss

    print(
        f"[box] preset={args.preset} box_class_weights={[round(w, 2) for w in weights]}"
    )
    print(f"[box] weights: {dict(zip(range(18), weights, strict=True))}")

    model = YOLO(args.weights)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        lr0=args.lr0,
        amp=args.amp,
        mosaic=0.0,
        close_mosaic=0,
        patience=20,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=args.name,
        exist_ok=True,
        seed=42,
    )

    if args.no_val:
        return 0

    # Final per-class recall on the best checkpoint, vs the 18-class baseline.
    best = Path(args.project) / args.name / "weights" / "best.pt"
    final_model = YOLO(str(best))
    rtdetr = final_model.model
    rtdetr.eval()
    m = validate(rtdetr, args, args.device)
    _print_metrics("best", m)

    print("\n[box] baseline vs box-weighted (per-class recall @ max-F1):")
    print("class            baseline   final     delta")
    for n in TRACKED_NAMES:
        base = BASELINE["per_class"][n]
        final = m["per_class"][n]
        print(f"{n:<16}   {base:.3f}    {final:.3f}   {final - base:+.3f}")
    print(
        f"mAP50           {BASELINE['map50']:.4f}    {m['map50']:.4f}   {m['map50'] - BASELINE['map50']:+.4f}"
    )
    print(
        f"crit_recall     {BASELINE['critical_recall']:.4f}    "
        f"{m['critical_recall']:.4f}   {m['critical_recall'] - BASELINE['critical_recall']:+.4f}"
    )
    print(
        "\n[box] next: run `python scripts/eval_threshold_calibration.py "
        f"--weights {best}` to check whether the critical-class confidence/precision "
        "tradeoff improved."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
