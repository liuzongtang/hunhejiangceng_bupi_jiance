"""
Render the RT-DETR detector's predictions on sample images into
comparison-ready annotated images (single frames + a mosaic grid).

Each box is labeled with the Tianchi class (Chinese + English), confidence,
and the mapped project severity, color-coded by severity. Supports both .pt
and .onnx weights (auto-detected).

Usage:
    python scripts/visualize_predictions.py --limit 12
    python scripts/visualize_predictions.py --weights .../best.onnx --limit 12
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
from typing import List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image, ImageDraw, ImageFont

from backend.inference.class_mapping import (
    TIANCHI_CLASS_NAMES,
    TIANCHI_CLASS_NAMES_ZH,
    project_severity,
)

SEVERITY_COLORS = {
    "critical": (220, 38, 38),  # red
    "major": (234, 88, 12),  # orange
    "medium": (202, 138, 4),  # amber
    "minor": (37, 99, 235),  # blue
    "info": (107, 114, 128),  # gray
}

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
CJK_FONTS = [
    "C:/Windows/Fonts/msyh.ttc",
    "C:/Windows/Fonts/simhei.ttf",
    "C:/Windows/Fonts/simsun.ttc",
]


def _load_font(size: int) -> ImageFont.FreeTypeFont:
    for p in CJK_FONTS:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size=size)
            except Exception:
                continue
    return ImageFont.load_default()


def _get_backend(weights: str, imgsz: int, conf: float):
    """Return (kind, model) where kind is 'onnx' or 'pt'."""
    if weights.lower().endswith(".onnx"):
        from backend.inference.rtdetr_onnx import RTDETRONNXEngine

        engine = RTDETRONNXEngine(
            weights, imgsz=imgsz, conf_threshold=conf, class_names=TIANCHI_CLASS_NAMES
        ).load()
        return ("onnx", engine)
    from ultralytics import YOLO

    return ("pt", YOLO(weights))


def _detect(
    backend, image_path: str, imgsz: int, conf: float
) -> List[Tuple[int, float, Tuple]]:
    """Return detections as [(class_id, score, (x1, y1, x2, y2)), ...]."""
    kind, model = backend
    if kind == "onnx":
        import cv2

        img = cv2.imread(image_path)
        d = model.predict([img])[0]
        return [
            (int(c), float(s), tuple(b))
            for c, s, b in zip(d["classes"], d["scores"], d["boxes"], strict=False)
        ]

    r = model.predict(source=image_path, imgsz=imgsz, conf=conf, verbose=False)[0]
    out: List[Tuple[int, float, Tuple]] = []
    if r.boxes is not None and len(r.boxes) > 0:
        for c, s, b in zip(r.boxes.cls, r.boxes.conf, r.boxes.xyxy, strict=False):
            out.append((int(c), float(s), tuple(b.cpu().numpy())))
    return out


def _annotate(
    image_path: str, detections: List[Tuple], font: ImageFont.FreeTypeFont
) -> Image.Image:
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    lw = max(3, int(img.width * 0.004))

    for cid, score, xyxy in detections:
        x1, y1, x2, y2 = (int(v) for v in xyxy)
        x1 = max(0, min(x1, img.width - 1))
        x2 = max(0, min(x2, img.width - 1))
        y1 = max(0, min(y1, img.height - 1))
        y2 = max(0, min(y2, img.height - 1))

        name = TIANCHI_CLASS_NAMES[cid]
        color = SEVERITY_COLORS[project_severity(name)]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=lw)

        label = f"{TIANCHI_CLASS_NAMES_ZH[cid]} {TIANCHI_CLASS_NAMES[cid]} {score:.2f}"
        # Try to measure text; fall back to a fixed width estimate.
        try:
            bbox = draw.textbbox((0, 0), label, font=font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        except Exception:
            tw, th = len(label) * int(font.size * 0.6), font.size
        ty = y1 - th - 4 if y1 - th - 4 >= 0 else y1 + 2
        draw.rectangle([x1, ty, x1 + tw + 6, ty + th + 4], fill=color)
        draw.text((x1 + 3, ty + 2), label, fill=(255, 255, 255), font=font)

    return img


def _mosaic(images: List[Image.Image], ncols: int, cell: int) -> Image.Image:
    nrows = (len(images) + ncols - 1) // ncols
    canvas = Image.new("RGB", (ncols * cell, nrows * cell), (255, 255, 255))
    for i, im in enumerate(images):
        im = im.copy()
        im.thumbnail((cell, cell), Image.LANCZOS)
        r, c = divmod(i, ncols)
        canvas.paste(im, (c * cell, r * cell))
    return canvas


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", default="runs/rtdetr/tianchi20/weights/best.pt")
    parser.add_argument("--images", default="data/tianchi/test/images")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument(
        "--limit", type=int, default=12, help="Number of images to annotate"
    )
    parser.add_argument("--ncols", type=int, default=4, help="Mosaic columns")
    parser.add_argument("--cell", type=int, default=640, help="Mosaic cell size (px)")
    parser.add_argument("--out", default="", help="Output dir")
    args = parser.parse_args()

    if not os.path.isdir(args.images):
        raise SystemExit(f"Image dir not found: {args.images}")
    images = sorted(
        os.path.join(args.images, f)
        for f in os.listdir(args.images)
        if os.path.splitext(f)[1].lower() in IMAGE_EXTS
    )[: args.limit]
    if not images:
        raise SystemExit(f"No images found in {args.images}")

    out_dir = args.out or os.path.join(
        os.path.dirname(os.path.abspath(args.weights)), "..", "analysis", "predictions"
    )
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    backend = _get_backend(args.weights, args.imgsz, args.conf)
    print(f"Backend: {backend[0]} | {len(images)} images -> {out_dir}")

    font = _load_font(max(16, int(args.cell * 0.03)))
    annotated: List[Image.Image] = []
    for path in images:
        dets = _detect(backend, path, args.imgsz, args.conf)
        img = _annotate(path, dets, font)
        annotated.append(img)
        stem = os.path.splitext(os.path.basename(path))[0]
        img.save(os.path.join(out_dir, f"{stem}_pred.jpg"), quality=90)
        print(f"  {stem}: {len(dets)} detections")

    mosaic = _mosaic(annotated, args.ncols, args.cell)
    mosaic_path = os.path.join(out_dir, "mosaic.jpg")
    mosaic.save(mosaic_path, quality=90)
    print(f"\n  saved {mosaic_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
