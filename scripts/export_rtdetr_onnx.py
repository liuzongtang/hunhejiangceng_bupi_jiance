"""
Export the trained RT-DETR checkpoint to ONNX for production inference.

RT-DETR's ONNX output differs from YOLO: the decoder head applies its own
top-k selection and emits a single tensor of shape (1, 300, 6) with format
[cx, cy, w, h, score, class] (normalized 0..1), so a dedicated parser is
required downstream (see backend/inference/rtdetr_onnx.py).

Usage:
    python scripts/export_rtdetr_onnx.py
    python scripts/export_rtdetr_onnx.py --weights runs/rtdetr/tianchi20/weights/best.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow `python scripts/export_rtdetr_onnx.py` to run without an editable install.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ultralytics import YOLO


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", default="runs/rtdetr/tianchi20/weights/best.pt")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument(
        "--no-simplify", action="store_true", help="Skip onnxsim (RT-DETR fallback)"
    )
    args = parser.parse_args()

    model = YOLO(args.weights)
    simplify = not args.no_simplify
    try:
        path = model.export(format="onnx", imgsz=args.imgsz, opset=args.opset, simplify=simplify)
    except Exception as e:
        # RT-DETR's deformable-attention graph can break onnxsim; fall back.
        if simplify:
            print(f"[warn] simplify=True failed ({e}); retrying simplify=False")
            path = model.export(format="onnx", imgsz=args.imgsz, opset=args.opset, simplify=False)
        else:
            raise
    print(f"Exported ONNX to: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
