#!/usr/bin/env python
"""eval_multIou.py -- official-pipeline mAP at IoU 0.1/0.3/0.5 (phase 0.2).

Runs the unmodified ultralytics val pipeline (same rect letterbox preprocessing,
conf=0.001, max_det=300, official ap_per_class) and only swaps the IoU matching
vector from linspace(0.5, 0.95, 10) to [0.1, 0.3, 0.5], so the resulting
mAP@0.1/0.3/0.5 are directly comparable with the official contest metric.

Sanity contract: mAP@0.5 must reproduce the training results.csv value
(tianchi18_full best.pt -> ~0.541). If it does not, the pipeline diverged and
the 0.1/0.3 numbers must not be trusted.

Usage:
    python scripts/eval_multIou.py --weights runs/rtdetr/tianchi18_full/weights/best.pt \
        --data data/tianchi/data.yaml --out runs/multi_iou/tianchi18_best.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import sys

import numpy as np
import torch

IOUS = (0.1, 0.3, 0.5)

# ---------------------------------------------------------------------------
# Patch 1: swap the IoU matching vector. Everything else in the official
# pipeline (dataloader, letterbox, conf=0.001, max_det=300, ap_per_class)
# stays untouched so numbers remain official-grade.
# ---------------------------------------------------------------------------
from ultralytics.models.rtdetr.val import RTDETRValidator
from ultralytics.utils.metrics import Metric

_orig_init = RTDETRValidator.__init__


def _patched_init(self, *args, **kwargs):
    _orig_init(self, *args, **kwargs)
    self.iouv = torch.tensor(IOUS)
    self.niou = (
        self.iouv.numel()
    )  # keep niou in sync: empty-pred fast path in update_metrics uses it


RTDETRValidator.__init__ = _patched_init

# Patch 2: Metric.map75 indexes all_ap[:, 5]; guard it so summary() cannot
# crash when niou < 6 (we never read map75 anyway).
Metric.map75 = property(
    lambda self: (
        float(self.all_ap[:, 5].mean())
        if getattr(self.all_ap, "shape", (0, 0))[1] > 5
        else float("nan")
    )
    if len(self.all_ap)
    else 0.0
)

CAPTURE: dict = {}


def _capture(validator):
    m = validator.metrics.box
    CAPTURE.clear()
    CAPTURE.update(
        iouv=[float(x) for x in validator.iouv],
        all_ap=np.asarray(m.all_ap).copy(),  # (n_classes_present, len(IOUS))
        ap_class_index=list(np.asarray(m.ap_class_index)),
        p=np.asarray(m.p).copy(),
        r=np.asarray(m.r).copy(),
        f1=np.asarray(m.f1).copy(),
        nt_per_class=np.asarray(validator.metrics.nt_per_class).copy(),
    )


_orig_init_reg = RTDETRValidator.__init__


def _patched_init_reg(self, *args, **kwargs):
    _orig_init_reg(self, *args, **kwargs)
    self.add_callback("on_val_end", _capture)


RTDETRValidator.__init__ = _patched_init_reg


def main() -> int:
    ap = argparse.ArgumentParser(description="Official-pipeline mAP@IoU 0.1/0.3/0.5")
    ap.add_argument("--weights", required=True, help="path to .pt checkpoint")
    ap.add_argument(
        "--data", default="data/tianchi/data.yaml", help="dataset yaml (val split)"
    )
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="0")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--name", default=None, help="run name for runs/val artifacts")
    ap.add_argument("--out", default=None, help="where to write the JSON report")
    args = ap.parse_args()

    import ultralytics
    from ultralytics import RTDETR

    model = RTDETR(args.weights)
    metrics = model.val(
        data=args.data,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        plots=False,
        verbose=False,
        save=False,
        name=args.name or f"multIou_{Path(args.weights).parent.parent.name}",
    )

    if not CAPTURE:
        print("ERROR: on_val_end capture failed", file=sys.stderr)
        return 2

    all_ap = CAPTURE["all_ap"]  # (n_present, 3)
    idx = CAPTURE["ap_class_index"]
    nt = CAPTURE["nt_per_class"]
    names = model.names  # dict id -> name (from checkpoint)

    map_at = {f"mAP@{t}": float(all_ap[:, i].mean()) for i, t in enumerate(IOUS)}

    per_class = []
    for row, c in enumerate(idx):
        per_class.append(
            {
                "id": int(c),
                "name": names[int(c)],
                "n_inst": int(nt[c]),
                "AP@0.1": round(float(all_ap[row, 0]), 4),
                "AP@0.3": round(float(all_ap[row, 1]), 4),
                "AP@0.5": round(float(all_ap[row, 2]), 4),
                "P@maxF1": round(float(CAPTURE["p"][row]), 4),
                "R@maxF1": round(float(CAPTURE["r"][row]), 4),
            }
        )

    report = {
        "weights": str(Path(args.weights).resolve()),
        "data": str(Path(args.data).resolve()),
        "imgsz": args.imgsz,
        "ious": list(IOUS),
        "val_conf": 0.001,
        "max_det": 300,
        "n_classes_present": len(idx),
        "n_instances_total": int(nt.sum()),
        "mAP": {k: round(v, 4) for k, v in map_at.items()},
        "note": "sanity: mAP@0.5 should match training results.csv (tianchi18_full best.pt ~= 0.541)",
        "per_class": per_class,
        "env": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "ultralytics": ultralytics.__version__,
            "cuda": torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else None,
        },
    }

    print("\n================ multi-IoU evaluation ================")
    print(f"weights : {report['weights']}")
    print(f"data    : {report['data']}  (imgsz={args.imgsz}, conf=0.001, max_det=300)")
    print(
        f"instances: {report['n_instances_total']} across {report['n_classes_present']} classes"
    )
    for k, v in report["mAP"].items():
        print(f"  {k:10s} = {v:.4f}")
    print("  sanity  : mAP@0.5 must match results.csv best (0.541 for tianchi18_full)")
    print("\n  class            n_inst   AP@0.1  AP@0.3  AP@0.5   P@F1   R@F1")
    for r in per_class:
        print(
            f"  {r['name']:<16s} {r['n_inst']:>6d}  {r['AP@0.1']:>6.3f} {r['AP@0.3']:>6.3f} "
            f"{r['AP@0.5']:>6.3f} {r['P@maxF1']:>6.3f} {r['R@maxF1']:>6.3f}"
        )
    crit = {"hole", "broken_warp", "broken_spandex"}
    print("\n  critical classes (hole/broken_warp/broken_spandex):")
    for r in per_class:
        if r["name"] in crit:
            print(
                f"    {r['name']:<16s} AP@0.1={r['AP@0.1']:.3f}  AP@0.3={r['AP@0.3']:.3f}  "
                f"AP@0.5={r['AP@0.5']:.3f}  R@maxF1={r['R@maxF1']:.3f}"
            )
    print("=====================================================\n")

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"report written to {out.resolve()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
