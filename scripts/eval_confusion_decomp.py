#!/usr/bin/env python
"""
eval_confusion_decomp.py -- GT-instance-level confusion decomposition for critical classes.

For each GT instance of a class, at confidence threshold tau and IoU 0.5:
  (a) correct       -- some same-class prediction with IoU>=iou_thr and conf>=tau.
                       This is exactly the numerator of the ultralytics r_curve
                       recall at conf=tau (utils/metrics.py ap_per_class uses
                       recall[:, 0], i.e. the IoU=0.5 column of the class-aware
                       match_predictions output), so correct/total must agree with
                       runs/threshold_calibration/threshold_sweep.csv.
  (b) misclassified -- no same-class match, but some different-class prediction with
                       IoU>=iou_thr and conf>=tau (the box was detected, wrong class).
                       Also records which class "stole" it (best-overlap wrong class).
  (c) missed        -- no prediction at all with IoU>=iou_thr and conf>=tau.
                       missed_with_subtau counts those with an any-class sub-tau
                       detection (a query fired on the box below tau);
                       missed_subtau_samecls restricts to same-class fires (the
                       sharper "detected below threshold" claim).

correct + misclassified + missed == total (val instance counts) by construction.

Motivation: paper 4.6 table 6 counts could not be reproduced from any saved artifact
(the saved runs/confusion matrix is matched-only at val conf=0.001 and its totals
disagree with the paper's). This script is the authoritative source.

Protocol mirrors the official val exactly:
  - RTDETRDataset with rect=False (square letterbox) -- ultralytics
    models/rtdetr/val.py build_dataset hardcodes "rect=False,  # no rect",
    i.e. the official RT-DETR val protocol is SQUARE letterbox regardless of
    any rect flag passed elsewhere (the earlier rect=True runs of this script
    were a protocol mismatch and produced garbage recalls).
  - raw model output (B, 300, 6) is NORMALIZED cxcywh; postprocess does
    xywh2xyxy(bboxes) * args.imgsz.
  - GT batch["bboxes"] are normalized cxcywh w.r.t. the same letterboxed image
    (_prepare_batch: xywh2xyxy(bbox) * imgsz).

Usage:
    python scripts/eval_confusion_decomp.py --weights runs/rtdetr/tianchi18_full/weights/best.pt --tau 0.25
    python scripts/eval_confusion_decomp.py --weights runs/rtdetr/tianchi18_full/weights/best.pt --tau 0.001
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.data import build_dataloader
from ultralytics.data.utils import check_det_dataset
from ultralytics.models.rtdetr.val import RTDETRDataset

CRIT = ["hole", "broken_warp", "broken_spandex"]


def iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """IoU between two sets of xyxy boxes. (n,m)"""
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)))
    area_a = (boxes_a[:, 2] - boxes_a[:, 0]).clip(0) * (
        boxes_a[:, 3] - boxes_a[:, 1]
    ).clip(0)
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]).clip(0) * (
        boxes_b[:, 3] - boxes_b[:, 1]
    ).clip(0)
    lt = np.maximum(boxes_a[:, None, :2], boxes_b[None, :, :2])
    rb = np.minimum(boxes_a[:, None, 2:], boxes_b[None, :, 2:])
    wh = (rb - lt).clip(0)
    inter = wh[..., 0] * wh[..., 1]
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def cxcywh_to_xyxy(b: np.ndarray) -> np.ndarray:
    out = np.empty_like(b)
    out[:, 0] = b[:, 0] - b[:, 2] / 2
    out[:, 1] = b[:, 1] - b[:, 3] / 2
    out[:, 2] = b[:, 0] + b[:, 2] / 2
    out[:, 3] = b[:, 1] + b[:, 3] / 2
    return out


@torch.no_grad()
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", default="data/tianchi/data.yaml")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--tau", type=float, default=0.25, help="confidence threshold")
    ap.add_argument("--iou-thr", type=float, default=0.5)
    ap.add_argument("--device", default="0")
    ap.add_argument(
        "--rect",
        type=int,
        default=0,
        help="1=rect letterbox, 0=square (default 0 = official RT-DETR val protocol)",
    )
    ap.add_argument(
        "--sweep-csv",
        default="runs/threshold_calibration/threshold_sweep.csv",
        help="r_curve sweep for cross-check (from eval_threshold_calibration.py)",
    )
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    model = YOLO(args.weights)
    device = torch.device(
        args.device
        if args.device.startswith("cuda") or args.device == "cpu"
        else f"cuda:{args.device}"
    )
    rtdetr = model.model.to(device).eval()
    names = model.names

    data_dict = check_det_dataset(args.data)
    cfg = get_cfg(
        overrides={"imgsz": args.imgsz, "rect": bool(args.rect), "cache": False}
    )
    dataset = RTDETRDataset(
        img_path=data_dict["val"],
        imgsz=args.imgsz,
        batch_size=args.batch,
        augment=False,
        hyp=cfg,
        rect=bool(args.rect),
        cache=None,
        prefix="val: ",
        data=data_dict,
    )
    loader = build_dataloader(dataset, args.batch, workers=0, shuffle=False)

    # per class: correct / misclassified / missed / missed_with_subtau /
    # missed_subtau_samecls / total; plus "stolen by" histogram for misclassified
    stats = {
        c: dict(
            correct=0,
            misclassified=0,
            missed=0,
            missed_with_subtau=0,
            missed_subtau_samecls=0,
            total=0,
            stolen_by={},
        )
        for c in names.values()
    }

    for batch in loader:
        img = batch["img"].to(device).float() / 255.0
        out = rtdetr(img)
        preds = (
            (out[0] if isinstance(out, (tuple, list)) else out).cpu().numpy()
        )  # (B,300,6)
        B = preds.shape[0]
        H, W = img.shape[2], img.shape[3]  # square letterbox: H == W == imgsz
        scale = np.array([W, H, W, H], dtype=float)
        for j in range(B):
            p = preds[j]  # (300,6): cx,cy,w,h,conf,cls (normalized cxcywh)
            pboxes = cxcywh_to_xyxy(p[:, :4]) * scale
            pconf = p[:, 4]
            pcls = p[:, 5].astype(int)
            # GT for image j in this batch
            sel = batch["batch_idx"].cpu().numpy() == j
            gcls = batch["cls"].cpu().numpy().reshape(-1)[sel].astype(int)
            gboxes_n = (
                batch["bboxes"].cpu().numpy().reshape(-1, 4)[sel]
            )  # normalized cxcywh
            if len(gcls) == 0:
                continue
            gxyxy = cxcywh_to_xyxy(gboxes_n) * scale

            ious = iou_matrix(pboxes, gxyxy)  # (300, n_gt)
            above = pconf >= args.tau
            for gi in range(len(gxyxy)):
                gc = int(gcls[gi])
                cname = names[gc]
                st = stats[cname]
                st["total"] += 1
                ov = ious[:, gi] >= args.iou_thr  # predictions overlapping this GT
                same_hit = ov & above & (pcls == gc)
                diff_hit = ov & above & (pcls != gc)
                if same_hit.any():
                    st["correct"] += 1
                elif diff_hit.any():
                    st["misclassified"] += 1
                    thief = int(pcls[diff_hit][np.argmax(pconf[diff_hit])])
                    tname = names[thief]
                    st["stolen_by"][tname] = st["stolen_by"].get(tname, 0) + 1
                else:
                    st["missed"] += 1
                    if ov.any():  # some prediction (any class, any conf) on the box
                        st["missed_with_subtau"] += 1
                        if (ov & (pcls == gc)).any():
                            st["missed_subtau_samecls"] += 1

    print(f"\n=== confusion decomposition: {args.weights} ===")
    print(
        f"protocol: imgsz={args.imgsz} rect={bool(args.rect)} tau={args.tau} iou={args.iou_thr}"
    )
    print(
        f"{'class':<18} {'total':>6} {'correct':>8} {'misclf':>7} {'missed':>7} "
        f"{'miss_subtau':>12} {'sub_same':>9} {'recall':>7}"
    )
    for c in CRIT:
        st = stats[c]
        rec = st["correct"] / st["total"] if st["total"] else float("nan")
        print(
            f"{c:<18} {st['total']:>6} {st['correct']:>8} {st['misclassified']:>7} {st['missed']:>7} "
            f"{st['missed_with_subtau']:>12} {st['missed_subtau_samecls']:>9} {rec:>7.3f}"
        )
        if st["stolen_by"]:
            print(f"{'':<18}   stolen_by: {st['stolen_by']}")
    crit_recall = np.mean([stats[c]["correct"] / stats[c]["total"] for c in CRIT])
    print(f"critical-class mean recall = {crit_recall:.4f}")

    # --- cross-check vs ultralytics r_curve sweep (class-aware recall at same tau) ---
    sweep_path = Path(args.sweep_csv)
    if sweep_path.exists():
        want = {}
        with open(sweep_path, newline="") as fh:
            for row in csv.DictReader(fh):
                if abs(float(row["conf"]) - args.tau) < 1e-9:
                    want[row["class"]] = float(row["recall"])
        print(
            "\ncross-check vs r_curve sweep (must match; validates scaling + matching):"
        )
        ok = True
        for c in CRIT:
            if c not in want:
                continue
            st = stats[c]
            rec = st["correct"] / st["total"] if st["total"] else float("nan")
            flag = "OK" if abs(rec - want[c]) < 0.02 else "MISMATCH"
            if flag != "OK":
                ok = False
            print(
                f"  {c:<18} decomp recall {rec:.4f}  vs r_curve {want[c]:.4f}  [{flag}]"
            )
        if not ok:
            print(
                "  !! recall mismatch >0.02 -- investigate before using these numbers"
            )
    else:
        print(f"(sweep csv not found at {sweep_path}; skipped cross-check)")

    if args.out:
        rep = {
            "weights": str(Path(args.weights).resolve()),
            "tau": args.tau,
            "iou_thr": args.iou_thr,
            "imgsz": args.imgsz,
            "rect": bool(args.rect),
            "critical_recall": round(float(crit_recall), 4),
            "per_class": {c: stats[c] for c in stats},
        }
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(
            json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"report written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
