"""
Train the RT-DETR-l "fabric" variant on the Tianchi dataset.

Three neck upgrades over vanilla RT-DETR-l, all injected via
``ultralytics.nn.tasks`` so the custom YAML
(``backend/training/rtdetr-l-fabric.yaml``) can resolve them:

  1. CARAFE content-aware upsampling (replaces the two nearest upsampling ops).
  2. DeformableAIFI — deformable intra-scale attention (replaces AIFI).
  3. ContextFusion — adaptive cross-scale fusion (replaces the CCFM RepC3 blocks).

Unchanged layers (backbone, decoder, lateral/downsample convs) can be seeded
from the COCO-pretrained ``rtdetr-l.pt`` via ``--weights``.

Usage:
    # 1-epoch sanity check (validates injection + YAML + data pipeline)
    python scripts/train_rtdetr_fabric.py --epochs 1 --imgsz 1280 --batch 4

    # Real run with partial pretrain
    python scripts/train_rtdetr_fabric.py --weights rtdetr-l.pt --epochs 20
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ultralytics import RTDETR
import ultralytics.nn.tasks as tasks

from backend.training.modules import CARAFE, ContextFusion, DeformableAIFI

# Inject the three custom neck modules so parse_model resolves them:
#   * CARAFE / DeformableAIFI are channel-preserving and referenced by new names.
#   * ContextFusion changes 2C -> C channels, so it is injected by replacing
#     tasks.RepC3; the YAML still writes "RepC3" and channel accounting is fine.
tasks.CARAFE = CARAFE
tasks.DeformableAIFI = DeformableAIFI
tasks.RepC3 = ContextFusion


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--yaml", default="backend/training/rtdetr-l-fabric.yaml", help="Model YAML"
    )
    parser.add_argument("--data", default="data/tianchi/data.yaml", help="Dataset yaml")
    parser.add_argument(
        "--weights", default="", help="Optional .pt for partial pretrain init"
    )
    parser.add_argument("--epochs", type=int, default=1, help="Epochs to train")
    parser.add_argument("--imgsz", type=int, default=1280, help="Input resolution")
    parser.add_argument("--batch", type=int, default=4, help="Batch size")
    parser.add_argument("--lr0", type=float, default=0.0005, help="Initial LR")
    parser.add_argument("--device", default="0", help="cuda device index")
    parser.add_argument("--workers", type=int, default=0, help="DataLoader workers")
    parser.add_argument("--project", default="runs/rtdetr-fabric", help="Output dir")
    parser.add_argument("--name", default="carafe_sanity", help="Run name")
    args = parser.parse_args()

    # Resolve project to an absolute path so ultralytics does not insert the
    # task name (runs/detect/<project>) for relative paths.
    args.project = str(Path(args.project).resolve())

    # RTDETR (not YOLO) builds from the custom YAML: YOLO('...yaml') cannot
    # guess the task for an "rtdetrdecoder" head (cfg2task only matches
    # "detect" in the head name), so the task must be pinned via the RTDETR
    # entry point, which forces task='detect' and selects RTDETRDetectionModel.
    model = RTDETR(args.yaml)
    if args.weights:
        model.load(args.weights)  # partial pretrain (unchanged layers only)

    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        lr0=args.lr0,
        mosaic=0.0,
        close_mosaic=0,
        patience=20,
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
