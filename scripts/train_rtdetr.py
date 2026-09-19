"""
Train an RT-DETR detector on the converted Tianchi cloth-defect dataset.

RT-DETR is DETR-based (one-to-one matching, no NMS) and is strong on small
objects — a good fit for fabric defects when trained at high input resolution.
Uses the Ultralytics unified API; the COCO-pretrained head (80 classes) is
auto-replaced with the 20 Tianchi classes from data.yaml.

Usage:
    # 1-epoch sanity check (validates the data pipeline + baseline mAP)
    python scripts/train_rtdetr.py --epochs 1 --imgsz 1280 --batch 8 --name sanity

    # Full run
    python scripts/train_rtdetr.py --epochs 60 --imgsz 1280 --batch 8

Requires the dataset converted by scripts/convert_tianchi_to_yolo.py first.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Allow `python scripts/train_rtdetr.py` to import the `backend` package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ultralytics import YOLO

from backend.logging_config import get_training_log


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="rtdetr-l.pt", help="Pretrained weights")
    parser.add_argument("--data", default="data/tianchi/data.yaml", help="Dataset yaml")
    parser.add_argument("--epochs", type=int, default=60, help="Epochs to train")
    parser.add_argument("--imgsz", type=int, default=1280, help="Input resolution")
    parser.add_argument("--batch", type=int, default=8, help="Batch size")
    parser.add_argument("--lr0", type=float, default=0.0005, help="Initial learning rate")
    parser.add_argument("--mosaic", type=float, default=0.0, help="Mosaic aug prob (RT-DETR: 0)")
    parser.add_argument("--patience", type=int, default=20, help="Early-stop patience (epochs)")
    parser.add_argument("--save-period", type=int, default=5, help="Save last.pt every N epochs (resume safety)")
    parser.add_argument("--device", default="0", help="cuda device index")
    parser.add_argument("--workers", type=int, default=0, help="DataLoader workers")
    parser.add_argument("--project", default="runs/rtdetr", help="Output project dir")
    parser.add_argument("--name", default="tianchi20", help="Run name")
    args = parser.parse_args()

    # Resolve project to an absolute path so ultralytics' get_save_dir does not
    # insert the task name (e.g. runs/detect/<project>) for relative paths.
    args.project = os.path.abspath(args.project)

    model = YOLO(args.model)

    log = get_training_log()

    def _scalar(value):
        """Convert a metric value to a JSON-safe scalar."""
        if isinstance(value, (int, float)):
            return round(float(value), 4) if isinstance(value, float) else value
        if hasattr(value, "item"):  # torch tensor
            return round(float(value.item()), 4)
        return str(value)

    def _on_fit_epoch_end(trainer):
        metrics = trainer.metrics or {}
        log.info(
            "epoch_metrics",
            data={
                "epoch": int(getattr(trainer, "epoch", 0)) + 1,
                "metrics": {str(k): _scalar(v) for k, v in metrics.items()},
            },
        )

    def _on_train_end(trainer):
        log.info(
            "training_complete",
            data={
                "epoch": int(getattr(trainer, "epoch", 0)) + 1,
                "best_fitness": _scalar(getattr(trainer, "best_fitness", None)),
                "save_dir": str(getattr(trainer, "save_dir", "")),
            },
        )

    model.add_callback("on_fit_epoch_end", _on_fit_epoch_end)
    model.add_callback("on_train_end", _on_train_end)
    log.info(
        "training_start",
        data={
            "model": args.model,
            "data": args.data,
            "epochs": args.epochs,
            "imgsz": args.imgsz,
            "batch": args.batch,
        },
    )

    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        lr0=args.lr0,
        mosaic=args.mosaic,
        close_mosaic=0 if args.mosaic == 0.0 else 10,
        patience=args.patience,
        save_period=args.save_period,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=args.name,
        exist_ok=True,
        seed=42,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
