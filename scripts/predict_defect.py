"""
Run the trained RT-DETR defect detector on a folder of images.

Supports both the PyTorch checkpoint (.pt) and the exported ONNX model (.onnx);
the backend is auto-detected from the weights extension. Each detection is
reported with its Tianchi class (English + Chinese), severity, defect code,
confidence and bbox.

Usage:
    python scripts/predict_defect.py --limit 10
    python scripts/predict_defect.py --weights runs/rtdetr/tianchi20/weights/best.onnx
    python scripts/predict_defect.py --images path/to/dir --out detections.json
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Dict, List, Tuple

# Allow `python scripts/predict_defect.py` to import the `backend` package
# without requiring an editable install.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.inference.class_mapping import (
    TIANCHI_CLASS_NAMES,
    TIANCHI_CLASS_NAMES_ZH,
    project_code,
    project_severity,
)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}


def make_item(
    class_id: int, conf: float, xyxy: Tuple[float, float, float, float]
) -> Dict:
    """Build one detection record in the project's reporting format."""
    name = TIANCHI_CLASS_NAMES[class_id]
    x1, y1, x2, y2 = xyxy
    return {
        "class_id": class_id,
        "class_name": name,
        "class_zh": TIANCHI_CLASS_NAMES_ZH[class_id],
        "severity": project_severity(name),
        "code": project_code(name),
        "confidence": round(float(conf), 4),
        "bbox_xyxy": [
            round(float(x1), 1),
            round(float(y1), 1),
            round(float(x2), 1),
            round(float(y2), 1),
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", default="runs/rtdetr/tianchi20/weights/best.pt")
    parser.add_argument("--images", default="data/tianchi/test/images")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--limit", type=int, default=0, help="Max images (0 = all)")
    parser.add_argument("--out", default="", help="Optional JSON output path")
    args = parser.parse_args()

    if not os.path.isdir(args.images):
        raise SystemExit(f"Image dir not found: {args.images}")
    images = sorted(
        os.path.join(args.images, f)
        for f in os.listdir(args.images)
        if os.path.splitext(f)[1].lower() in IMAGE_EXTS
    )
    if not images:
        raise SystemExit(f"No images found in {args.images}")
    if args.limit:
        images = images[: args.limit]

    # Auto-detect backend from the weights extension.
    use_onnx = args.weights.lower().endswith(".onnx")
    if use_onnx:
        import cv2

        from backend.inference.rtdetr_onnx import RTDETRONNXEngine

        engine = RTDETRONNXEngine(
            args.weights,
            imgsz=args.imgsz,
            conf_threshold=args.conf,
            class_names=TIANCHI_CLASS_NAMES,
        ).load()
        print(f"Backend: ONNX ({args.weights})")
    else:
        from ultralytics import YOLO

        model = YOLO(args.weights)
        print(f"Backend: PyTorch ({args.weights})")

    detections: Dict[str, List[Dict]] = {}
    for path in images:
        name = os.path.basename(path)
        items: List[Dict] = []
        if use_onnx:
            img = cv2.imread(path)
            d = engine.predict([img])[0]
            for c, s, b in zip(d["classes"], d["scores"], d["boxes"], strict=False):
                items.append(make_item(int(c), float(s), tuple(b)))
        else:
            r = model.predict(
                source=path, imgsz=args.imgsz, conf=args.conf, verbose=False
            )[0]
            if r.boxes is not None and len(r.boxes) > 0:
                for c, s, b in zip(
                    r.boxes.cls, r.boxes.conf, r.boxes.xyxy, strict=False
                ):
                    items.append(make_item(int(c), float(s), tuple(b.cpu().numpy())))
        detections[name] = items

    total = sum(len(v) for v in detections.values())
    print(f"Images: {len(detections)}  Detections: {total}\n")
    for name, items in detections.items():
        if not items:
            continue
        print(f"{name}:")
        for it in items:
            print(
                f"  [{it['confidence']:.2f}] {it['class_zh']}({it['class_name']}) "
                f"-> {it['severity']} {it['code']} "
                f"bbox={it['bbox_xyxy']}"
            )

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(detections, f, ensure_ascii=False, indent=2)
        print(f"\nSaved to {args.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
