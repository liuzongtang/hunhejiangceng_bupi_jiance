"""
Dump per-query detections for the inference-side alarm-scorer replay.

Captures, for every val image, the full per-query class-score matrix
(1, 300, 18) plus boxes (1, 300, 4) in cxcywh normalized coords — the
intermediate tensors inside ``RTDETRDecoder.postprocess``, which are NOT part
of the (1, 300, 6) final output (the final output keeps only the global top-300
(query, class) pairs). The alarm scorer needs within-group normalized class
distributions, hence the hook.

Preprocessing matches the official mAP eval (``eval_multIou.py``): the
unmodified ultralytics RTDETRDataset val dataloader (rect=False square
letterbox, augment=False, imgsz=1280).

Output: runs/alarm_replay/detections.jsonl, one JSON object per image:
  {"image": "<stem>", "im_file": "<abs path>", "gt_defective": bool,
   "gt_boxes": [[cx, cy, w, h, cls], ...],
   "detections": [[cx, cy, w, h, conf, cls, s0..s17], ...]}   # top-K by conf

Only the top-K detections per image (by max class score) are stored — the
alarm dimensions (severity / confidence / uncertainty / group conflict / size /
clustering) are all monotone in score, so low-score queries never contribute.

Usage:
    python scripts/alarm_replay_dump.py \
        --weights runs/rtdetr/tianchi18_mosaic1280/weights/best.pt --topk 30
"""

from __future__ import annotations

import argparse
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


def has_box(label_path: Path) -> bool:
    try:
        with open(label_path, "r", encoding="utf-8") as f:
            return any(line.strip() for line in f)
    except FileNotFoundError:
        return False


def img_to_label(im_file: str) -> Path:
    p = Path(im_file)
    parts = list(p.parts)
    for i, seg in enumerate(parts):
        if seg == "images":
            parts[i] = "labels"
            break
    return Path(*parts).with_suffix(".txt")


def read_gt(label_path: Path) -> list:
    """GT boxes in YOLO cxcywh + class id."""
    boxes = []
    try:
        with open(label_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 5:
                    boxes.append([float(v) for v in parts[:5]])
    except FileNotFoundError:
        pass
    return boxes


@torch.no_grad()
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", default="data/tianchi/data.yaml")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="0")
    ap.add_argument(
        "--topk", type=int, default=30, help="detections kept per image (by conf)"
    )
    ap.add_argument("--out", default="runs/alarm_replay/detections.jsonl")
    ap.add_argument(
        "--image-dir",
        default=None,
        help="Cross-domain negative mode: recursively dump every image under this "
        "directory with gt_defective=False (e.g. shujuji_AITEX/NODefect_images). "
        "Skips the val dataloader.",
    )
    args = ap.parse_args()

    device = torch.device("cpu")
    if args.device.isdigit() and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.device}")

    yolo = YOLO(args.weights)
    rtdetr = yolo.model
    rtdetr.to(device)
    rtdetr.eval()

    # ---- hook: capture per-query (boxes, scores) inside postprocess ----
    decoder = rtdetr.model[-1]  # RTDETRDecoder
    capture: dict = {}
    orig_post = decoder.postprocess

    def wrapped_postprocess(boxes, scores):
        capture["boxes"] = boxes.detach()
        capture["scores"] = scores.detach()  # (1, 300, nc) sigmoid
        return orig_post(boxes, scores)

    decoder.postprocess = wrapped_postprocess

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_img = 0
    with out_path.open("w", encoding="utf-8") as fout:
        if args.image_dir:
            # ---- cross-domain negative mode (no GT; all clean) ----
            import cv2

            root = Path(args.image_dir)
            paths = sorted(
                p
                for p in root.rglob("*")
                if p.suffix.lower() in (".png", ".jpg", ".jpeg")
            )
            for p in paths:
                im = cv2.imread(str(p), cv2.IMREAD_COLOR)
                if im is None:
                    continue
                im = cv2.resize(
                    im, (args.imgsz, args.imgsz), interpolation=cv2.INTER_LINEAR
                )
                t = (
                    torch.from_numpy(im)
                    .permute(2, 0, 1)
                    .contiguous()
                    .unsqueeze(0)
                    .float()
                    / 255.0
                )
                rtdetr(t.to(device))
                boxes = capture["boxes"].float().cpu().numpy()
                scores = capture["scores"].float().cpu().numpy()
                b_j, s_j = boxes[0], scores[0]
                conf = s_j.max(axis=1)
                cls = s_j.argmax(axis=1)
                order = np.argsort(-conf)[: args.topk]
                dets = []
                for q in order:
                    if conf[q] <= 0.0:
                        continue
                    row = [
                        round(float(b_j[q, 0]), 5),
                        round(float(b_j[q, 1]), 5),
                        round(float(b_j[q, 2]), 5),
                        round(float(b_j[q, 3]), 5),
                        round(float(conf[q]), 5),
                        int(cls[q]),
                    ]
                    row.extend(round(float(s), 5) for s in s_j[q])
                    dets.append(row)
                rec = {
                    "image": p.stem,
                    "im_file": str(p),
                    "gt_defective": False,
                    "gt_boxes": [],
                    "detections": dets,
                    "cross_domain": True,
                }
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_img += 1
                print(f"\r[dump:xdomain] {n_img} images", end="", flush=True)
        else:
            data_dict = check_det_dataset(args.data)
            cfg = get_cfg(
                overrides={"imgsz": args.imgsz, "rect": False, "cache": False}
            )
            dataset = RTDETRDataset(
                img_path=data_dict["val"],
                imgsz=args.imgsz,
                batch_size=args.batch,
                augment=False,
                hyp=cfg,
                rect=False,
                cache=None,
                prefix="val: ",
                data=data_dict,
            )
            loader = build_dataloader(dataset, args.batch, workers=0, shuffle=False)
            for batch in loader:
                img = batch["img"].float() / 255.0
                rtdetr(img.to(device))  # hook fires inside
                boxes = capture["boxes"].float().cpu().numpy()  # (B, 300, 4) cxcywh
                scores = capture["scores"].float().cpu().numpy()  # (B, 300, 18)
                B = boxes.shape[0]
                for j in range(B):
                    b_j, s_j = boxes[j], scores[j]
                    conf = s_j.max(axis=1)
                    cls = s_j.argmax(axis=1)
                    order = np.argsort(-conf)[: args.topk]
                    im_file = batch["im_file"][j]
                    lbl = img_to_label(im_file)
                    dets = []
                    for q in order:
                        if conf[q] <= 0.0:
                            continue
                        row = [
                            round(float(b_j[q, 0]), 5),
                            round(float(b_j[q, 1]), 5),
                            round(float(b_j[q, 2]), 5),
                            round(float(b_j[q, 3]), 5),
                            round(float(conf[q]), 5),
                            int(cls[q]),
                        ]
                        row.extend(round(float(s), 5) for s in s_j[q])
                        dets.append(row)
                    rec = {
                        "image": Path(im_file).stem,
                        "im_file": str(im_file),
                        "gt_defective": has_box(lbl),
                        "gt_boxes": read_gt(lbl),
                        "detections": dets,
                    }
                    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    n_img += 1
                print(f"\r[dump] {n_img} images", end="", flush=True)
    print(f"\n[dump] wrote {out_path} ({n_img} images)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
