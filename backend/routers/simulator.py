"""
Simulator data router — real Tianchi test images + RT-DETR detections.

Serves the production-line simulator frontend with real fabric images from the
Tianchi test split and the trained RT-DETR model's predictions, so the simulator
shows a realistic camera feed instead of procedural placeholder graphics.

Endpoints:
    GET /api/v1/simulator/images       -> list of test image filenames
    GET /api/v1/simulator/image/{name} -> raw JPEG
    GET /api/v1/simulator/detect/{name}-> cached RT-DETR predictions
"""

from __future__ import annotations

import glob
import os
import threading
import time
from typing import Dict, List

from fastapi import APIRouter, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse

from backend.schemas.defect import (
    DEFECT_CODES,
    DEFECT_SEVERITY,
    TIANCHI_CLASS_NAMES,
    TIANCHI_CLASS_NAMES_ZH,
)

router = APIRouter(prefix="/api/v1/simulator", tags=["simulator"])

_PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
_TEST_IMAGE_DIR = os.path.join(_PROJECT_ROOT, "data", "tianchi", "test", "images")
_MODEL_PATH = os.path.join(
    _PROJECT_ROOT, "runs", "rtdetr", "tianchi20", "weights", "best.onnx"
)

_ZH = dict(zip(TIANCHI_CLASS_NAMES, TIANCHI_CLASS_NAMES_ZH, strict=True))
_SEV_BY_NAME = {dt.value: DEFECT_SEVERITY[dt].value for dt in DEFECT_SEVERITY}
_CODE_BY_NAME = {dt.value: DEFECT_CODES[dt].value for dt in DEFECT_CODES}

_engine = None
_engine_lock = threading.Lock()
_detect_cache: Dict[str, List[Dict]] = {}
_cache_lock = threading.Lock()


def _get_engine():
    """Lazily load the RT-DETR ONNX engine (singleton)."""
    global _engine
    if _engine is None:
        with _engine_lock:
            if _engine is None:
                from backend.inference.rtdetr_onnx import RTDETRONNXEngine

                _engine = RTDETRONNXEngine(
                    _MODEL_PATH,
                    imgsz=1280,
                    conf_threshold=0.25,
                    class_names=list(TIANCHI_CLASS_NAMES),
                ).load()
    return _engine


def _image_path(name: str) -> str:
    """Resolve a test image path, guarding against path traversal."""
    safe = os.path.basename(name)
    path = os.path.join(_TEST_IMAGE_DIR, safe)
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail=f"image not found: {safe}")
    return path


def _detect(name: str) -> List[Dict]:
    """Run RT-DETR on one test image (cached) and return detections."""
    with _cache_lock:
        cached = _detect_cache.get(name)
        if cached is not None:
            return cached

    import cv2
    import numpy as np

    # np.fromfile + imdecode is unicode-safe on Windows (cv2.imread uses the
    # ANSI codepage and fails on paths with non-ASCII characters like 维度模型).
    data = np.fromfile(_image_path(name), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=500, detail=f"failed to read image: {name}")

    raw = _get_engine().predict([img])[0]
    detections: List[Dict] = []
    for j in range(len(raw["boxes"])):
        cls_name = raw["class_names"][j]
        x1, y1, x2, y2 = (float(v) for v in raw["boxes"][j])
        detections.append(
            {
                "type": cls_name,
                "zh": _ZH.get(cls_name, cls_name),
                "code": _CODE_BY_NAME.get(cls_name, "OT-01"),
                "severity": _SEV_BY_NAME.get(cls_name, "info"),
                "bbox": [
                    round(x1, 1),
                    round(y1, 1),
                    round(x2 - x1, 1),
                    round(y2 - y1, 1),
                ],
                "confidence": round(float(raw["scores"][j]), 3),
            }
        )

    with _cache_lock:
        _detect_cache[name] = detections
    return detections


@router.get("/images")
async def list_images():
    """List available test images (sorted, deterministic feed)."""
    if not os.path.isdir(_TEST_IMAGE_DIR):
        return {"total": 0, "images": []}
    names = sorted(
        os.path.basename(p) for p in glob.glob(os.path.join(_TEST_IMAGE_DIR, "*.jpg"))
    )
    return {"total": len(names), "images": names}


@router.get("/image/{name}")
async def get_image(name: str):
    """Serve a raw test image."""
    return FileResponse(_image_path(name), media_type="image/jpeg")


@router.get("/detect/{name}")
async def detect(name: str):
    """Run the RT-DETR model on one test image (cached) and return predictions."""
    t0 = time.perf_counter()
    detections = await run_in_threadpool(_detect, name)
    ms = round((time.perf_counter() - t0) * 1000, 1)
    return {"name": name, "detections": detections, "inference_ms": ms}
