"""
Targeted fine-tune of the trained RT-DETR on the oversampled confusion-class list.

Resumes from ``runs/rtdetr/tianchi20/weights/best.pt`` at a low LR and trains on
``data/tianchi/data_oversampled.yaml`` (images containing the line/weft confusion
classes repeated), to sharpen the broken_warp<->surface_mark / weave_defect<->
surface_mark / weft_shrink<->coarse_weft decision boundaries.

Usage:
    python scripts/finetune_confusion.py --epochs 20 --lr0 0.0001
"""

from __future__ import annotations

import argparse

from ultralytics import YOLO


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", default="runs/rtdetr/tianchi20/weights/best.pt")
    parser.add_argument("--data", default="data/tianchi/data_oversampled.yaml")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr0", type=float, default=0.0001)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--project", default="runs/rtdetr-finetune")
    parser.add_argument("--name", default="confusion_v1")
    parser.add_argument("--device", default="0")
    args = parser.parse_args()

    model = YOLO(args.weights)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        lr0=args.lr0,
        patience=args.patience,
        workers=0,
        device=args.device,
        seed=42,
        deterministic=True,
        project=args.project,
        name=args.name,
        exist_ok=True,
        val=True,
        plots=True,
        amp=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
