"""
Per-class confidence-threshold calibration for cost-sensitive detection.

The confusion-matrix diagnosis (``analyze_confusion.py``) showed the critical
classes (hole / broken_warp / broken_spandex) are *missed below threshold*, not
misclassified — their detections carry systematically low confidence. So the
cheap, zero-training lever is to stop using one global confidence threshold and
give the critical classes their own (lower) threshold.

This script extracts the per-class precision/recall/F1 curves over the
confidence axis already computed by the RT-DETR validator, and reports, for the
critical classes, the recall/precision tradeoff across a confidence sweep, the
per-class max-F1 operating point, and the resulting cost-sensitive critical
recall (mean over the three critical classes) at three regimes:

1. single global max-F1 threshold (the reported baseline, crit_recall ~0.574);
2. per-class max-F1 threshold (each class at its own optimum);
3. recall-prioritized threshold (fixed low conf, e.g. 0.10) — favours catch
   over false-alarm, matching the ``漏检代价 > 误检代价`` requirement.

Usage:
    python scripts/eval_threshold_calibration.py
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from ultralytics import YOLO
from ultralytics.models.rtdetr.val import RTDETRValidator

CRITICAL_IDS = [0, 9, 15]  # hole, broken_warp, broken_spandex
CONF_SWEEP = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
RECALL_PRIORITIZED_CONF = 0.10


def _curve_pos(ap_class_index: list, class_id: int) -> int:
    """Map a class id to its row position in the (18, 1000) metric curves."""
    return ap_class_index.index(class_id)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--weights", default="runs/rtdetr/tianchi18_full/weights/best.pt"
    )
    parser.add_argument("--data", default="data/tianchi/data.yaml")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default="0")
    parser.add_argument("--project", default="runs/threshold_calibration")
    args = parser.parse_args()

    save_dir = Path(args.project)
    save_dir.mkdir(parents=True, exist_ok=True)

    val_args = {
        "data": args.data,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "workers": 0,
        "plots": False,
        "rect": True,
        "mode": "val",
        "verbose": False,
    }
    validator = RTDETRValidator(args=val_args, save_dir=save_dir)
    model = YOLO(args.weights)
    validator(model=model.model)

    box = validator.metrics.box
    names = validator.names
    x = np.asarray(box.px, dtype=float)  # confidence axis, 1000 points
    ap_idx = list(box.ap_class_index)

    def at_conf(cid: int, tau: float) -> tuple:
        """Return (recall, precision, f1) for class ``cid`` at confidence ``tau``."""
        p = _curve_pos(ap_idx, cid)
        j = int(np.argmin(np.abs(x - tau)))
        return (
            float(box.r_curve[p, j]),
            float(box.p_curve[p, j]),
            float(box.f1_curve[p, j]),
        )

    # --- confidence sweep for critical classes ---
    print("\nCritical-class recall/precision vs confidence threshold:")
    print(f"{'class':<14}" + "".join(f"{tau:>8.2f}" for tau in CONF_SWEEP))
    for cid in CRITICAL_IDS:
        r_vals, p_vals = [], []
        for tau in CONF_SWEEP:
            r, p, _ = at_conf(cid, tau)
            r_vals.append(r)
            p_vals.append(p)
        print(f"{names[cid]:<14}  R:" + "".join(f"{r:>8.3f}" for r in r_vals))
        print(f"{'':<14}  P:" + "".join(f"{p:>8.3f}" for p in p_vals))

    # --- per-class max-F1 operating point ---
    print("\nPer-class max-F1 operating point (vs the single global threshold):")
    i_global = int(np.argmax(_smooth(box.f1_curve.mean(0), 0.1)))
    print(f"  global max-F1 threshold conf = {x[i_global]:.3f}\n")
    print(
        f"{'class':<14} {'global conf':>11} {'global R':>9} {'global P':>9} | "
        f"{'own conf':>8} {'own R':>8} {'own P':>8}"
    )
    per_class_maxf1 = {}
    for cid in range(validator.nc):
        if cid not in ap_idx:
            continue
        p = _curve_pos(ap_idx, cid)
        i_own = int(np.argmax(box.f1_curve[p]))
        g_r = float(box.r_curve[p, i_global])
        g_p = float(box.p_curve[p, i_global])
        o_r = float(box.r_curve[p, i_own])
        o_p = float(box.p_curve[p, i_own])
        per_class_maxf1[cid] = (o_r, o_p)
        tag = " *CRIT*" if cid in CRITICAL_IDS else ""
        print(
            f"{names[cid]:<14} {x[i_global]:>11.3f} {g_r:>9.3f} {g_p:>9.3f} | "
            f"{x[i_own]:>8.3f} {o_r:>8.3f} {o_p:>8.3f}{tag}"
        )

    # --- cost-sensitive critical recall at three regimes ---
    def crit_recall(get_r) -> tuple:
        rs = [get_r(cid) for cid in CRITICAL_IDS]
        return float(np.mean(rs))

    global_crit = crit_recall(lambda cid: at_conf(cid, x[i_global])[0])
    perclass_crit = crit_recall(lambda cid: per_class_maxf1[cid][0])
    prioritized_crit = crit_recall(lambda cid: at_conf(cid, RECALL_PRIORITIZED_CONF)[0])

    print(
        "\nCost-sensitive critical recall (mean over hole/broken_warp/broken_spandex):"
    )
    print(
        f"  1. single global threshold (conf={x[i_global]:.3f}):  crit_recall = {global_crit:.4f}"
    )
    print(
        f"  2. per-class max-F1 threshold:                       crit_recall = {perclass_crit:.4f}"
    )
    print(
        f"  3. recall-prioritized (conf={RECALL_PRIORITIZED_CONF:.2f}):            "
        f"crit_recall = {prioritized_crit:.4f}"
    )

    # --- save sweep CSV ---
    with open(save_dir / "threshold_sweep.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["class_id", "class", "conf", "recall", "precision", "f1"])
        for cid in range(validator.nc):
            if cid not in ap_idx:
                continue
            for tau in CONF_SWEEP:
                r, p, f = at_conf(cid, tau)
                w.writerow(
                    [cid, names[cid], f"{tau:.2f}", f"{r:.4f}", f"{p:.4f}", f"{f:.4f}"]
                )

    print(f"\nSweep saved to {save_dir / 'threshold_sweep.csv'}")
    return 0


def _smooth(y: np.ndarray, f: float = 0.1) -> np.ndarray:
    """Box filter of fraction ``f`` of the curve, mirroring ultralytics ``smooth``."""
    nf = round(len(y) * f * 2) // 2 + 1
    p = np.ones(nf // 2)
    yp = np.concatenate((p * y[0], y, p * y[-1]), 0)
    return np.convolve(yp, np.ones(nf) / nf, mode="same")[nf // 2 : 1 - nf // 2]


if __name__ == "__main__":
    raise SystemExit(main())
