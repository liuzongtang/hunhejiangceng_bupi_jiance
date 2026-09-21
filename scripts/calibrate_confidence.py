"""
Per-class confidence calibration: does boosting critical-class logits recover recall?

The P1 diagnostic showed query-noise GRPO is structurally dead. The Layer-1 finding
was that "miss" = a critical GT force-assigned to a low-confidence query. This script
tests the cheapest direct remedy: add a per-class *logit bias* to the critical classes
(broken_warp 9 / star_skip 15 / broken_spandex 16 / weave_defect 19) at inference time,
and measure the recall-vs-precision tradeoff — no training, pure post-processing.

It runs the raw RT-DETR decoder (``dec_logits`` shape (B,300,20), pre-sigmoid), sweeps
the bias, re-sigmoids, re-argmaxes, and evaluates per-class recall + overall
precision/recall + mAP50 against the val GTs.

Usage:
    python scripts/calibrate_confidence.py --limit 0
    python scripts/calibrate_confidence.py --biases 0,0.5,1,1.5,2,3 --conf 0.25
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.data import build_dataloader
from ultralytics.data.utils import check_det_dataset
from ultralytics.models.rtdetr import RTDETRValidator
from ultralytics.utils.metrics import compute_ap

from backend.inference.class_mapping import TIANCHI_CLASS_NAMES

CRITICAL = [9, 15, 16, 19]  # broken_warp / star_skip / broken_spandex / weave_defect


def _resolve_device(dev: str) -> torch.device:
    if dev == "cpu":
        return torch.device("cpu")
    if dev.isdigit():
        return torch.device(f"cuda:{dev}" if torch.cuda.is_available() else "cpu")
    return torch.device(dev)


def _xywh2xyxy(boxes: np.ndarray) -> np.ndarray:
    """Normalized xywh (N,4) -> xyxy (N,4)."""
    x, y, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    return np.stack([x - w / 2, y - h / 2, x + w / 2, y + h / 2], axis=1)


def _box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between two sets of xyxy boxes: (N,4) x (M,4) -> (N,M)."""
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[:, :, 0] * wh[:, :, 1]
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / np.clip(union, 1e-9, None)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", default="runs/rtdetr/tianchi20/weights/best.pt")
    parser.add_argument("--data", default="data/tianchi/data.yaml")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--biases", default="0,0.5,1,1.5,2,3")
    parser.add_argument("--critical", default="9,15,16,19")
    parser.add_argument("--limit", type=int, default=0, help="Max val images (0 = all)")
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--project", default="runs/rtdetr-calib")
    parser.add_argument("--name", default="calib")
    args = parser.parse_args()
    biases = [float(x) for x in args.biases.split(",")]
    critical = [int(x) for x in args.critical.split(",")]

    device = _resolve_device(args.device)
    args.project = os.path.abspath(args.project)
    save_dir = Path(args.project) / args.name
    save_dir.mkdir(parents=True, exist_ok=True)

    # ---- Model -------------------------------------------------------------
    yolo = YOLO(args.weights)
    rtdetr = yolo.model
    rtdetr.to(device)
    rtdetr.eval()

    # ---- Val data ----------------------------------------------------------
    # Use the SAME dataset class as ``model.val()`` (RTDETRDataset), which
    # stretches images to a square imgsz (rect_mode=False) rather than
    # aspect-preserving letterbox. Reproducing the val mAP requires this exact
    # preprocessing; letterbox gives a systematically lower mAP (~0.41 vs 0.60).
    data_dict = check_det_dataset(args.data)
    stride = int(rtdetr.stride.max())
    val_args = {
        "model": args.weights,
        "data": args.data,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "conf": 0.001,
        "device": str(args.device),
    }
    validator = RTDETRValidator(args=val_args)
    validator.data = data_dict
    validator.stride = stride
    dataset = validator.build_dataset(data_dict["val"], batch=args.batch)
    loader = build_dataloader(dataset, args.batch, workers=args.workers, shuffle=False)

    # ---- Pass 1: collect raw predictions + GTs -----------------------------
    all_pred_boxes: list[np.ndarray] = []  # per-image (300,4) xyxy normalized
    all_pred_logits: list[np.ndarray] = []  # per-image (300,20) raw logits
    all_gt_cls: list[np.ndarray] = []  # per-image (n,)
    all_gt_boxes: list[np.ndarray] = []  # per-image (n,4) xyxy normalized

    seen = 0
    with torch.no_grad():
        for batch in loader:
            bs = batch["img"].shape[0]
            if args.limit and seen >= args.limit:
                break
            img = batch["img"].to(device).float() / 255.0
            out = rtdetr.predict(img)
            raw = out[1]
            dec_boxes = raw[0][-1].cpu().numpy()  # (B,300,4) xywh normalized
            dec_logits = raw[1][-1].cpu().numpy()  # (B,300,20) logits

            cls = batch["cls"].cpu().numpy().reshape(-1).astype(int)
            bboxes = batch["bboxes"].cpu().numpy().reshape(-1, 4)
            batch_idx = batch["batch_idx"].cpu().numpy().reshape(-1).astype(int)

            for i in range(bs):
                mask = batch_idx == i
                all_pred_boxes.append(_xywh2xyxy(dec_boxes[i]))
                all_pred_logits.append(dec_logits[i])
                all_gt_cls.append(cls[mask])
                all_gt_boxes.append(_xywh2xyxy(bboxes[mask]))
            seen += bs
            if args.limit and seen >= args.limit:
                break

    n_imgs = len(all_pred_boxes)
    n_cls = all_pred_logits[0].shape[1]
    print(
        f"[calib] {n_imgs} val images, {n_cls} classes, "
        f"critical={[TIANCHI_CLASS_NAMES[c] for c in critical]}"
    )

    # ---- Pass 2: bias sweep (numpy, fast) ----------------------------------
    bias_vec = np.zeros(n_cls, dtype=np.float32)
    rows = []
    for bias in biases:
        bias_vec[:] = 0.0
        for c in critical:
            bias_vec[c] = bias

        # AP replicates the RTDETRValidator pipeline exactly: IoU-desc greedy
        # matching (``match_predictions``) + ultralytics' 101-point AP. The
        # val filters ``scores > conf`` at conf=0.001, which keeps essentially
        # all 300 queries, so no effective conf threshold is applied to AP.
        # P/R uses a single conf>=args.conf operating point.
        conf_all: list[np.ndarray] = []
        cls_all: list[np.ndarray] = []
        tp_all: list[np.ndarray] = []
        gt_all: list[np.ndarray] = []
        ft_tp = ft_fp = 0
        ft_gt = 0
        crit_recall = {c: [0, 0] for c in critical}  # tp, total

        for img_i in range(n_imgs):
            logits = all_pred_logits[img_i] + bias_vec[None, :]
            scores = 1.0 / (1.0 + np.exp(-logits))  # (300, 20)
            # Replicate ultralytics RT-DETR postprocess: top-300 (query, class)
            # PAIRS over the flattened score matrix, not per-query argmax.
            n_q = scores.shape[0]
            flat = scores.ravel()  # (300*20,)
            order = np.argsort(-flat)[:n_q]
            pred_cls = order % n_cls  # (300,)
            q_idx = order // n_cls  # (300,)
            conf = flat[order]  # (300,) already desc
            pred_box = all_pred_boxes[img_i][q_idx]  # (300, 4)

            gt_cls = all_gt_cls[img_i]
            gt_box = all_gt_boxes[img_i]

            # ---- IoU-desc greedy matching (mirrors match_predictions) --------
            tp = np.zeros(len(pred_cls), dtype=bool)
            if len(gt_cls) and len(pred_cls):
                iou = _box_iou(gt_box, pred_box)  # (n_gt, n_pred)
                iou_masked = iou * (gt_cls[:, None] == pred_cls[None, :])
                matches = np.argwhere(iou_masked >= 0.5)  # (n, 2) [gt, pred]
                if matches.shape[0]:
                    if matches.shape[0] > 1:
                        matches = matches[
                            iou_masked[matches[:, 0], matches[:, 1]].argsort()[::-1]
                        ]
                        matches = matches[
                            np.unique(matches[:, 1], return_index=True)[1]
                        ]
                        matches = matches[
                            np.unique(matches[:, 0], return_index=True)[1]
                        ]
                    tp[matches[:, 1].astype(int)] = True
            conf_all.append(conf)
            cls_all.append(pred_cls)
            tp_all.append(tp)
            gt_all.append(gt_cls)

            # ---- fixed-threshold pass: conf >= args.conf ---------------------
            keep = conf >= args.conf
            p_cls = pred_cls[keep]
            p_box = pred_box[keep]
            p_conf = conf[keep]
            matched_ft = set()
            for j in np.argsort(-p_conf):
                c = int(p_cls[j])
                cand = [
                    k
                    for k in range(len(gt_cls))
                    if gt_cls[k] == c and k not in matched_ft
                ]
                if cand:
                    ious = _box_iou(p_box[j : j + 1], gt_box[cand])[0]
                    b = int(np.argmax(ious))
                    if ious[b] >= 0.5:
                        matched_ft.add(cand[b])
                        ft_tp += 1
                        if c in critical:
                            crit_recall[c][0] += 1
                    else:
                        ft_fp += 1
                else:
                    ft_fp += 1
            ft_gt += len(gt_cls)
            for c in critical:
                crit_recall[c][1] += int((gt_cls == c).sum())

        # Fixed-threshold metrics.
        prec = ft_tp / max(ft_tp + ft_fp, 1)
        rec = ft_tp / max(ft_gt, 1)
        # mAP50 via ultralytics' own compute_ap (101-point), mirroring
        # ap_per_class: conf-desc sort, per-GT-class grouping, mean over
        # classes present in the targets.
        conf_cat = np.concatenate(conf_all)
        cls_cat = np.concatenate(cls_all)
        tp_cat = np.concatenate(tp_all).astype(np.float64)
        gt_cls_cat = np.concatenate(gt_all)
        order = np.argsort(-conf_cat)
        cls_cat = cls_cat[order]
        tp_cat = tp_cat[order]
        unique_cls, nt = np.unique(gt_cls_cat, return_counts=True)
        aps = []
        for ci, c in enumerate(unique_cls):
            m = cls_cat == c
            n_l = int(nt[ci])
            n_p = int(m.sum())
            if n_p == 0 or n_l == 0:
                aps.append(0.0)
                continue
            tpc = tp_cat[m].cumsum()
            fpc = (1.0 - tp_cat[m]).cumsum()
            recall = tpc / (n_l + 1e-16)
            precision = tpc / (tpc + fpc + 1e-16)
            ap_c, _, _ = compute_ap(recall, precision)
            aps.append(float(ap_c))
        map50 = float(np.mean(aps))

        row = {
            "bias": bias,
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "mAP50": round(map50, 4),
        }
        for c in critical:
            tp, tot = crit_recall[c]
            row[f"recall_{TIANCHI_CLASS_NAMES[c]}"] = round(tp / max(tot, 1), 4)
        rows.append(row)

        crit_str = "  ".join(
            f"{TIANCHI_CLASS_NAMES[c]}={row[f'recall_{TIANCHI_CLASS_NAMES[c]}']}"
            for c in critical
        )
        print(
            f"  bias={bias:>4.1f}  P={prec:.4f}  R={rec:.4f}  mAP50={map50:.4f}  |  {crit_str}"
        )

    # ---- Save --------------------------------------------------------------
    csv_path = save_dir / "calibration.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[calib] wrote {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
