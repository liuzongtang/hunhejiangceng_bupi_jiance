"""
Train an RT-DETR detector with an NWD box loss — Direction B refinement (R3).

R2 box-loss *weighting* (``train_rtdetr_boxweighted.py``) was the sixth negative:
weighting the same IoU-based box arm did not beat uniform fine-tune and did not
stop the low-confidence precision collapse. The literature-backed fix is to
*change the geometry of the box loss itself*: NWD (Normalized Wasserstein
Distance) models boxes as 2D Gaussians and replaces IoU's pathological
sensitivity to tiny offsets on thin defects with a smooth distance. It also
replaces the varifocal ``gt_score`` (== predicted IoU) with NWD, so the
confidence target stays high under small localization errors — the actual lever
the diagnosis pointed at.

Usage:
    # NWD box loss + NWD gt_score (default: 5 epochs, C estimated from labels)
    python scripts/train_rtdetr_nwd.py --epochs 5 --preset nwd

    # Control arm — identical run but keep IoU gt_score (only the box term changes)
    python scripts/train_rtdetr_nwd.py --epochs 5 --preset nwd-boxonly --name nwd_boxonly

    # Override the NWD constant C explicitly
    python scripts/train_rtdetr_nwd.py --nwd-c 0.03
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

from backend.training.nwd import (
    BoxNWDRTDETRDetectionLoss,
    estimate_nwd_c,
    set_default_nwd,
)

# Critical / secondary class ids under the 18-class taxonomy (see data.yaml).
CRITICAL_CLASS_IDS = [0, 9, 15]  # hole, broken_warp, broken_spandex
CRITICAL_CLASS_NAMES = ["hole", "broken_warp", "broken_spandex"]
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
        f"[nwd] {tag}: mAP50={m['map50']:.4f} mAP50-95={m['map50_95']:.4f} "
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
        "--labels-dir",
        default="data/tianchi/train/labels",
        help="Dir to estimate NWD C from",
    )
    parser.add_argument(
        "--preset",
        default="nwd",
        choices=["nwd", "nwd-boxonly"],
        help="nwd = NWD box loss + NWD gt_score; nwd-boxonly = NWD box loss only, IoU gt_score",
    )
    parser.add_argument(
        "--nwd-c",
        type=float,
        default=None,
        help="NWD constant C (default: estimate from labels)",
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
        "--project", default="runs/rtdetr-nwd", help="Output project dir"
    )
    parser.add_argument("--name", default="nwd", help="Run name")
    parser.add_argument(
        "--no-val", action="store_true", help="Skip the final per-class recall report"
    )
    args = parser.parse_args()

    # Resolve project to an absolute path so Ultralytics does not insert the
    # task name (runs/detect/<project>) for relative paths.
    args.project = os.path.abspath(args.project)

    use_gt_score = args.preset == "nwd"
    c = args.nwd_c if args.nwd_c is not None else estimate_nwd_c(args.labels_dir)
    set_default_nwd(c=c, use_gt_score=use_gt_score)
    _detr_loss.RTDETRDetectionLoss = BoxNWDRTDETRDetectionLoss

    print(f"[nwd] preset={args.preset} nwd_c={c:.4f} use_nwd_gt_score={use_gt_score}")

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

    print("\n[nwd] baseline vs NWD (per-class recall @ max-F1):")
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
        "\n[nwd] next: run `python scripts/eval_threshold_calibration.py "
        f"--weights {best}` to check whether the critical-class confidence/precision "
        "tradeoff improved."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
