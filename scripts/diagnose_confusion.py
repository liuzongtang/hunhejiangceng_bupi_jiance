"""
Per-class confusion diagnostic: what does each GT get predicted as?

Reuses the RT-DETR val data pipeline (RTDETRDataset stretch-to-square) to collect
predictions and GTs, then for each GT box finds the highest-IoU prediction (any
class) and records (gt_cls -> pred_cls | miss). Mirrors ultralytics' confusion
matrix but prints a compact, human-readable table focused on the critical/warp
classes.

Usage:
    python scripts/diagnose_confusion.py --limit 0
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.data import build_dataloader
from ultralytics.data.utils import check_det_dataset
from ultralytics.models.rtdetr import RTDETRValidator

from backend.inference.class_mapping import TIANCHI_CLASS_NAMES

CRITICAL = [9, 15, 16, 19]  # broken_warp / star_skip / broken_spandex / weave_defect


def _resolve_device(dev: str) -> torch.device:
    if dev == "cpu":
        return torch.device("cpu")
    if dev.isdigit():
        return torch.device(f"cuda:{dev}" if torch.cuda.is_available() else "cpu")
    return torch.device(dev)


def _xywh2xyxy(b: np.ndarray) -> np.ndarray:
    x, y, w, h = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    return np.stack([x - w / 2, y - h / 2, x + w / 2, y + h / 2], axis=1)


def _box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[:, :, 0] * wh[:, :, 1]
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.clip(area_a[:, None] + area_b[None, :] - inter, 1e-9, None)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", default="runs/rtdetr/tianchi20/weights/best.pt")
    parser.add_argument("--data", default="data/tianchi/data.yaml")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0, help="Max val images (0=all)")
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=0)
    args = parser.parse_args()

    device = _resolve_device(args.device)
    yolo = YOLO(args.weights)
    rtdetr = yolo.model
    rtdetr.to(device)
    rtdetr.eval()

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

    nc = len(TIANCHI_CLASS_NAMES)
    # confusion[gt_cls, pred_cls] with a final "miss" column at index nc.
    confusion = np.zeros((nc, nc + 1), dtype=np.int64)

    seen = 0
    with torch.no_grad():
        for batch in loader:
            bs = batch["img"].shape[0]
            if args.limit and seen >= args.limit:
                break
            img = batch["img"].to(device).float() / 255.0
            out = rtdetr.predict(img)
            raw = out[1]
            dec_boxes = raw[0][-1].cpu().numpy()  # (B,300,4) xywh norm
            dec_logits = raw[1][-1].cpu().numpy()  # (B,300,20) logits

            cls = batch["cls"].cpu().numpy().reshape(-1).astype(int)
            bboxes = batch["bboxes"].cpu().numpy().reshape(-1, 4)
            batch_idx = batch["batch_idx"].cpu().numpy().reshape(-1).astype(int)

            for i in range(bs):
                # predictions: top-300 (query,class) pairs (mirrors postprocess)
                logits = dec_logits[i]
                scores = 1.0 / (1.0 + np.exp(-logits))
                n_q = scores.shape[0]
                flat = scores.ravel()
                order = np.argsort(-flat)[:n_q]
                pred_cls = order % nc
                pred_box = _xywh2xyxy(dec_boxes[i])[order // nc]

                m = batch_idx == i
                gt_cls = cls[m]
                gt_box = _xywh2xyxy(bboxes[m])

                if len(gt_cls) and len(pred_cls):
                    iou = _box_iou(gt_box, pred_box)  # (n_gt, n_pred)
                    # argmax IoU prediction per GT (any class)
                    best = iou.argmax(axis=1)
                    best_iou = iou[np.arange(len(gt_cls)), best]
                    for g in range(len(gt_cls)):
                        if best_iou[g] >= 0.5:
                            confusion[gt_cls[g], pred_cls[best[g]]] += 1
                        else:
                            confusion[gt_cls[g], nc] += 1
                elif len(gt_cls):
                    confusion[gt_cls, nc] += 1
            seen += bs
            if args.limit and seen >= args.limit:
                break

    # ---- Report -----------------------------------------------------------
    focus = list(range(nc))  # all classes, so both directions of each pair are visible
    print("\nGT class  ->  [pred class (count)] ...  |  miss  |  total  |  recall")
    for g in focus:
        row = confusion[g]
        total = int(row.sum())
        if total == 0:
            continue
        hit = int(row[g])
        miss = int(row[nc])
        # top off-diagonal confusion targets
        off = np.argsort(-row[:nc])
        off = [c for c in off if c != g][:4]
        conf_str = "  ".join(
            f"{TIANCHI_CLASS_NAMES[c]}({int(row[c])})" for c in off if row[c] > 0
        )
        print(
            f"{TIANCHI_CLASS_NAMES[g]:>16}  ->  {conf_str:<48} | miss {miss:3d} | tot {total:3d} | self {hit}  recall={hit / total:.2f}"
        )
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
