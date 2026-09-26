"""
Audit confusable-class predictions to separate label noise from label ambiguity.

For each ``(A, B)`` confusion pair, run the trained RT-DETR over a dataset split
and export every case where a ground-truth box of A is matched (IoU >= 0.5) to a
prediction of B (and vice versa). Each sample is rendered as a side-by-side
image -- full frame on the left, zoomed crop of the ground-truth box on the
right -- with ground-truth boxes in green and model boxes in blue/red, so a
human can eyeball whether the label is simply wrong (noise) or the two classes
are pixel-identical (ambiguity).

A CSV manifest lists one row per sample with an empty ``verdict`` column to fill
in (``noise`` | ``ambiguous`` | ``model_error`` | ``correct``). A few correctly
predicted samples of A and B are also exported for contrast.

Usage:
    python scripts/audit_confusion.py --pairs 9,18 12,11 --per-pair 30
    python scripts/audit_confusion.py --split train --limit 2000 --conf 0.25
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.data import build_dataloader
from ultralytics.data.utils import check_det_dataset
from ultralytics.models.rtdetr import RTDETRValidator

from backend.inference.class_mapping import TIANCHI_CLASS_NAMES, TIANCHI_CLASS_NAMES_ZH

GT_COLOR = (0, 200, 0)
PRED_COLOR = (200, 100, 0)
FOCUS_GT_COLOR = (0, 255, 0)
FOCUS_PRED_COLOR = (0, 0, 255)


def _resolve_device(dev: str) -> torch.device:
    """Resolve a ``--device`` string to a torch device."""
    if dev == "cpu":
        return torch.device("cpu")
    if dev.isdigit():
        return torch.device(f"cuda:{dev}" if torch.cuda.is_available() else "cpu")
    return torch.device(dev)


def _xywh2xyxy(b: np.ndarray) -> np.ndarray:
    """Convert normalized [cx, cy, w, h] boxes to [x1, y1, x2, y2]."""
    x, y, w, h = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    return np.stack([x - w / 2, y - h / 2, x + w / 2, y + h / 2], axis=1)


def _box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between two sets of [x1, y1, x2, y2] boxes."""
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[:, :, 0] * wh[:, :, 1]
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.clip(area_a[:, None] + area_b[None, :] - inter, 1e-9, None)


def _read_img(path: str) -> Optional[np.ndarray]:
    """Read an image, safe for Windows paths containing non-ASCII characters."""
    data = np.fromfile(path, dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def _draw_box(
    img: np.ndarray,
    box_px: np.ndarray,
    color: Tuple[int, int, int],
    label: str = "",
    thickness: int = 2,
) -> None:
    """Draw a rectangle and (optionally) a filled label tag onto ``img`` in-place."""
    x1, y1, x2, y2 = [round(v) for v in box_px]
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    if label:
        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), _ = cv2.getTextSize(label, font, 0.5, 1)
        ty = y1 - th - 6
        if ty < 0:
            ty = y2 + th + 6
        cv2.rectangle(img, (x1, ty - th - 4), (x1 + tw + 6, ty + 4), color, -1)
        cv2.putText(
            img, label, (x1 + 3, ty), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA
        )


def _render_sample(
    img: np.ndarray,
    gt_boxes_norm: np.ndarray,
    gt_cls: np.ndarray,
    pred_boxes_norm: np.ndarray,
    pred_cls: np.ndarray,
    pred_conf: np.ndarray,
    focus_g: int,
    matched_p: Optional[int],
    conf_thresh: float,
) -> np.ndarray:
    """Render one sample as a (full image | zoomed crop) side-by-side composite."""
    h, w = img.shape[:2]
    scale = np.array([w, h, w, h], dtype=np.float32)
    gt_px = gt_boxes_norm * scale
    pred_px = pred_boxes_norm * scale

    canvas = img.copy()
    for k in range(len(gt_cls)):
        _draw_box(canvas, gt_px[k], GT_COLOR, f"GT:{TIANCHI_CLASS_NAMES[gt_cls[k]]}", 2)
    for k in range(len(pred_cls)):
        if pred_conf[k] >= conf_thresh:
            _draw_box(
                canvas,
                pred_px[k],
                PRED_COLOR,
                f"{TIANCHI_CLASS_NAMES[pred_cls[k]]} {pred_conf[k]:.2f}",
                1,
            )
    _draw_box(
        canvas,
        gt_px[focus_g],
        FOCUS_GT_COLOR,
        f"GT:{TIANCHI_CLASS_NAMES[gt_cls[focus_g]]}",
        3,
    )
    if matched_p is not None:
        _draw_box(
            canvas,
            pred_px[matched_p],
            FOCUS_PRED_COLOR,
            f"{TIANCHI_CLASS_NAMES[pred_cls[matched_p]]} {pred_conf[matched_p]:.2f}",
            3,
        )

    # Zoomed crop around the focus GT box.
    fx1, fy1, fx2, fy2 = gt_px[focus_g]
    bw, bh = fx2 - fx1, fy2 - fy1
    pad = 2.0
    cx1, cy1 = max(0, int(fx1 - pad * bw)), max(0, int(fy1 - pad * bh))
    cx2, cy2 = min(w, int(fx2 + pad * bw)), min(h, int(fy2 + pad * bh))
    crop = img[cy1:cy2, cx1:cx2].copy()
    if crop.size == 0:
        crop = img.copy()
        cx1 = cy1 = 0
    off = np.array([cx1, cy1, cx1, cy1], dtype=np.float32)
    for k in range(len(gt_cls)):
        _draw_box(
            crop, gt_px[k] - off, GT_COLOR, f"GT:{TIANCHI_CLASS_NAMES[gt_cls[k]]}", 2
        )
    for k in range(len(pred_cls)):
        if pred_conf[k] >= conf_thresh:
            _draw_box(
                crop,
                pred_px[k] - off,
                PRED_COLOR,
                f"{TIANCHI_CLASS_NAMES[pred_cls[k]]} {pred_conf[k]:.2f}",
                1,
            )
    if matched_p is not None:
        _draw_box(
            crop,
            pred_px[matched_p] - off,
            FOCUS_PRED_COLOR,
            f"{TIANCHI_CLASS_NAMES[pred_cls[matched_p]]} {pred_conf[matched_p]:.2f}",
            2,
        )

    # Common height, side by side.
    max_w = 1200
    full = (
        cv2.resize(canvas, (max_w, int(h * max_w / w)), interpolation=cv2.INTER_AREA)
        if w > max_w
        else canvas
    )
    fh = full.shape[0]
    ch = max(1, crop.shape[0])
    crop_rs = cv2.resize(
        crop, (max(1, int(fh * crop.shape[1] / ch)), fh), interpolation=cv2.INTER_LINEAR
    )
    min_cw = 420
    if crop_rs.shape[1] < min_cw:
        crop_rs = np.pad(
            crop_rs,
            ((0, 0), (min_cw - crop_rs.shape[1], 0), (0, 0)),
            constant_values=128,
        )
    return np.hstack([full, crop_rs])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", default="runs/rtdetr/tianchi20/weights/best.pt")
    parser.add_argument("--data", default="data/tianchi/data.yaml")
    parser.add_argument("--split", default="val", help="Dataset split: val | train")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument(
        "--conf", type=float, default=0.25, help="Draw preds above this score"
    )
    parser.add_argument(
        "--iou", type=float, default=0.5, help="GT-pred match threshold"
    )
    parser.add_argument(
        "--pairs",
        nargs="+",
        default=["9,18", "12,11"],
        help="Space-separated 'A,B' confusion pairs (class ids)",
    )
    parser.add_argument(
        "--per-pair", type=int, default=30, help="Max samples per direction"
    )
    parser.add_argument(
        "--contrast", type=int, default=5, help="Correct samples per class"
    )
    parser.add_argument(
        "--limit", type=int, default=0, help="Max images to scan (0=all)"
    )
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--out", default="runs/confusion-audit")
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
    if args.split not in data_dict:
        raise SystemExit(
            f"split '{args.split}' not in data.yaml (has {sorted(data_dict)})"
        )
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
    dataset = validator.build_dataset(data_dict[args.split], batch=args.batch)
    loader = build_dataloader(dataset, args.batch, workers=args.workers, shuffle=False)

    # pair -> direction -> list of sample dicts.
    samples: Dict[Tuple[int, int], Dict[str, list]] = {
        p: {"a_as_b": [], "b_as_a": [], "correct_a": [], "correct_b": []} for p in pairs
    }
    directions = ("a_as_b", "b_as_a")

    def _cap(key: str) -> int:
        return args.per_pair if key in directions else args.contrast

    def _full(p: Tuple[int, int]) -> bool:
        return all(len(samples[p][key]) >= _cap(key) for key in samples[p])

    seen = 0
    with torch.no_grad():
        for batch in loader:
            bs = batch["img"].shape[0]
            if args.limit and seen >= args.limit:
                break
            im_files = batch.get("im_file") or [""] * bs
            img = batch["img"].to(device).float() / 255.0
            raw = rtdetr.predict(img)[1]
            dec_boxes = raw[0][-1].cpu().numpy()  # (B, 300, 4) xywh norm
            dec_logits = raw[1][-1].cpu().numpy()  # (B, 300, 20)

            cls = batch["cls"].cpu().numpy().reshape(-1).astype(int)
            bboxes = batch["bboxes"].cpu().numpy().reshape(-1, 4)
            batch_idx = batch["batch_idx"].cpu().numpy().reshape(-1).astype(int)

            for i in range(bs):
                scores = 1.0 / (1.0 + np.exp(-dec_logits[i]))  # (300, 20)
                pred_cls = scores.argmax(axis=1)
                pred_conf = scores[np.arange(scores.shape[0]), pred_cls]
                keep = pred_conf >= args.conf
                pred_box_norm = _xywh2xyxy(dec_boxes[i])[keep]
                pred_cls = pred_cls[keep]
                pred_conf = pred_conf[keep]

                m = batch_idx == i
                gt_cls = cls[m]
                gt_box_norm = _xywh2xyxy(bboxes[m])
                if len(gt_cls) == 0 or len(pred_cls) == 0:
                    continue

                iou = _box_iou(gt_box_norm, pred_box_norm)
                best = iou.argmax(axis=1)
                best_iou = iou[np.arange(len(gt_cls)), best]

                for g in range(len(gt_cls)):
                    if best_iou[g] < args.iou:
                        continue
                    a, b = int(gt_cls[g]), int(pred_cls[best[g]])
                    for p in pairs:
                        key = None
                        if (a, b) == p:
                            key = "a_as_b"
                        elif (b, a) == p:
                            key = "b_as_a"
                        elif a == b == p[0]:
                            key = "correct_a"
                        elif a == b == p[1]:
                            key = "correct_b"
                        if key is not None and len(samples[p][key]) < _cap(key):
                            samples[p][key].append(
                                {
                                    "im_file": im_files[i],
                                    "focus_g": g,
                                    "matched_p": int(best[g]),
                                    "gt_boxes": gt_box_norm,
                                    "gt_cls": gt_cls,
                                    "pred_boxes": pred_box_norm,
                                    "pred_cls": pred_cls,
                                    "pred_conf": pred_conf,
                                    "gt_id": a,
                                    "pred_id": b,
                                    "conf": float(pred_conf[best[g]]),
                                    "iou": float(best_iou[g]),
                                }
                            )
                if all(_full(p) for p in pairs):
                    break
            if all(_full(p) for p in pairs):
                break
            seen += bs
            if args.limit and seen >= args.limit:
                break

    # ---- Render + manifest ------------------------------------------------
    manifest_path = out_dir / "manifest.csv"
    rows: List[Dict[str, str]] = []
    for p in pairs:
        a, b = p
        pair_dir = (
            out_dir / f"{a}_{b}_{TIANCHI_CLASS_NAMES[a]}_{TIANCHI_CLASS_NAMES[b]}"
        )
        pair_dir.mkdir(parents=True, exist_ok=True)
        for key, direction in (
            ("a_as_b", f"{TIANCHI_CLASS_NAMES[a]}_as_{TIANCHI_CLASS_NAMES[b]}"),
            ("b_as_a", f"{TIANCHI_CLASS_NAMES[b]}_as_{TIANCHI_CLASS_NAMES[a]}"),
            ("correct_a", f"correct_{TIANCHI_CLASS_NAMES[a]}"),
            ("correct_b", f"correct_{TIANCHI_CLASS_NAMES[b]}"),
        ):
            for k, s in enumerate(samples[p][key]):
                im = _read_img(s["im_file"])
                if im is None:
                    print(f"[skip] unreadable: {s['im_file']}")
                    continue
                rendered = _render_sample(
                    im,
                    s["gt_boxes"],
                    s["gt_cls"],
                    s["pred_boxes"],
                    s["pred_cls"],
                    s["pred_conf"],
                    s["focus_g"],
                    s["matched_p"],
                    args.conf,
                )
                stem = Path(s["im_file"]).stem
                out_name = f"{direction}_{k:03d}_{stem}.jpg"
                cv2.imwrite(
                    str(pair_dir / out_name), rendered, [cv2.IMWRITE_JPEG_QUALITY, 90]
                )
                rows.append(
                    {
                        "pair": f"{TIANCHI_CLASS_NAMES[a]}/{TIANCHI_CLASS_NAMES[b]}",
                        "direction": direction,
                        "filename": out_name,
                        "source": Path(s["im_file"]).name,
                        "gt_class": TIANCHI_CLASS_NAMES[s["gt_id"]],
                        "gt_zh": TIANCHI_CLASS_NAMES_ZH[s["gt_id"]],
                        "pred_class": TIANCHI_CLASS_NAMES[s["pred_id"]],
                        "pred_zh": TIANCHI_CLASS_NAMES_ZH[s["pred_id"]],
                        "pred_conf": f"{s['conf']:.3f}",
                        "iou": f"{s['iou']:.3f}",
                        "verdict": "",
                    }
                )

    with open(manifest_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "pair",
                "direction",
                "filename",
                "source",
                "gt_class",
                "gt_zh",
                "pred_class",
                "pred_zh",
                "pred_conf",
                "iou",
                "verdict",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    for p in pairs:
        a, b = p
        s = samples[p]
        print(
            f"[audit] {TIANCHI_CLASS_NAMES[a]}({a}) <-> {TIANCHI_CLASS_NAMES[b]}({b}): "
            f"a_as_b={len(s['a_as_b'])} b_as_a={len(s['b_as_a'])} "
            f"correct_a={len(s['correct_a'])} correct_b={len(s['correct_b'])}"
        )
    print(f"[audit] {len(rows)} samples -> {out_dir}")
    print(f"[audit] manifest -> {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
