"""
Crop the actual confusion cases from val images for visual inspection.

For each (A -> B) confusion pair, find val GTs of class A whose highest-IoU
prediction is class B, crop the padded GT box from the *original* image, and save
it alongside a few correctly-predicted crops of A and B for side-by-side comparison.
Answers "is this a labeling ambiguity or a model limitation?" before committing to a
fine-tune.

Usage:
    python scripts/visualize_confusion_pairs.py --pairs 9,18 12,11 19,18 --per-pair 6
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.data import build_dataloader
from ultralytics.data.utils import check_det_dataset
from ultralytics.models.rtdetr import RTDETRValidator

from backend.inference.class_mapping import TIANCHI_CLASS_NAMES


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


def _crop(im_file: str, box_norm_xyxy: np.ndarray, pad: float = 0.5) -> np.ndarray:
    """Crop a padded box (normalized xyxy) from the original image."""
    img = cv2.imread(im_file)
    if img is None:
        return np.zeros((128, 128, 3), dtype=np.uint8)
    h, w = img.shape[:2]
    x1, y1, x2, y2 = box_norm_xyxy * np.array([w, h, w, h])
    bw, bh = x2 - x1, y2 - y1
    x1 -= pad * bw
    y1 -= pad * bh
    x2 += pad * bw
    y2 += pad * bh
    x1, y1, x2, y2 = int(max(0, x1)), int(max(0, y1)), int(min(w, x2)), int(min(h, y2))
    return img[y1:y2, x1:x2]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", default="runs/rtdetr/tianchi20/weights/best.pt")
    parser.add_argument("--data", default="data/tianchi/data.yaml")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument(
        "--pairs",
        nargs="+",
        default=["9,18", "12,11", "19,18"],
        help="Space-separated 'A,B' confusion pairs",
    )
    parser.add_argument("--per-pair", type=int, default=6)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--out", default="runs/confusion-pairs")
    args = parser.parse_args()

    device = _resolve_device(args.device)
    pairs = [tuple(int(x) for x in p.split(",")) for p in args.pairs]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

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
    # Collect crops: pair -> (predicted-as-B crops, correct-A crops, correct-B crops)
    crops: dict[tuple, dict[str, list[tuple[np.ndarray, str]]]] = {
        p: {"confuse": [], "correct_a": [], "correct_b": []} for p in pairs
    }

    def _enough(p) -> bool:
        c = crops[p]
        return (
            len(c["confuse"]) >= args.per_pair
            and len(c["correct_a"]) >= 3
            and len(c["correct_b"]) >= 3
        )

    seen = 0
    with torch.no_grad():
        for batch in loader:
            bs = batch["img"].shape[0]
            if args.limit and seen >= args.limit:
                break
            im_files = batch.get("im_file") or [""] * bs
            img = batch["img"].to(device).float() / 255.0
            out = rtdetr.predict(img)
            raw = out[1]
            dec_boxes = raw[0][-1].cpu().numpy()
            dec_logits = raw[1][-1].cpu().numpy()

            cls = batch["cls"].cpu().numpy().reshape(-1).astype(int)
            bboxes = batch["bboxes"].cpu().numpy().reshape(-1, 4)
            batch_idx = batch["batch_idx"].cpu().numpy().reshape(-1).astype(int)

            for i in range(bs):
                logits = dec_logits[i]
                scores = 1.0 / (1.0 + np.exp(-logits))
                n_q = scores.shape[0]
                flat = scores.ravel()
                order = np.argsort(-flat)[:n_q]
                pred_cls = order % nc
                pred_box = _xywh2xyxy(dec_boxes[i])[order // nc]

                m = batch_idx == i
                gt_cls = cls[m]
                gt_box_norm = _xywh2xyxy(bboxes[m])

                if len(gt_cls) and len(pred_cls):
                    iou = _box_iou(gt_box_norm, pred_box)
                    best = iou.argmax(axis=1)
                    best_iou = iou[np.arange(len(gt_cls)), best]
                    for g in range(len(gt_cls)):
                        if best_iou[g] < 0.5:
                            continue
                        a, b = int(gt_cls[g]), int(pred_cls[best[g]])
                        for p in pairs:
                            if (a, b) == p and len(crops[p]["confuse"]) < args.per_pair:
                                crop = _crop(im_files[i], gt_box_norm[g])
                                crops[p]["confuse"].append(
                                    (
                                        crop,
                                        f"{TIANCHI_CLASS_NAMES[a]}_as_{TIANCHI_CLASS_NAMES[b]}",
                                    )
                                )
                            elif a == b == p[0] and len(crops[p]["correct_a"]) < 3:
                                crops[p]["correct_a"].append(
                                    (
                                        _crop(im_files[i], gt_box_norm[g]),
                                        TIANCHI_CLASS_NAMES[a],
                                    )
                                )
                            elif a == b == p[1] and len(crops[p]["correct_b"]) < 3:
                                crops[p]["correct_b"].append(
                                    (
                                        _crop(im_files[i], gt_box_norm[g]),
                                        TIANCHI_CLASS_NAMES[b],
                                    )
                                )
                if all(_enough(p) for p in pairs):
                    break
            if all(_enough(p) for p in pairs):
                break
            seen += bs
            if args.limit and seen >= args.limit:
                break

    # ---- Save -------------------------------------------------------------
    for p in pairs:
        a, b = p
        d = out_dir / f"{TIANCHI_CLASS_NAMES[a]}_{TIANCHI_CLASS_NAMES[b]}"
        d.mkdir(parents=True, exist_ok=True)
        for key, prefix in (
            ("confuse", "AS"),
            ("correct_a", f"CORRECT_{TIANCHI_CLASS_NAMES[a]}"),
            ("correct_b", f"CORRECT_{TIANCHI_CLASS_NAMES[b]}"),
        ):
            for k, (crop, _) in enumerate(crops[p][key]):
                cv2.imwrite(str(d / f"{prefix}_{k}.jpg"), crop)
        c = crops[p]
        print(
            f"[viz] {TIANCHI_CLASS_NAMES[a]} -> {TIANCHI_CLASS_NAMES[b]}: "
            f"{len(c['confuse'])} confuse, {len(c['correct_a'])} correct-A, {len(c['correct_b'])} correct-B  -> {d}"
        )
    print(f"[viz] done -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
