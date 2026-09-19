"""
Fabric Defect Detector — high-level inference API.

Ties together the inference engine, preprocessing, and postprocessing
into a single detect() call suitable for API integration.
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Dict, List, Optional, Tuple

import numpy as np

from backend.inference.class_mapping import project_code, project_severity
from backend.inference.engine import InferenceEngine, create_engine
from backend.inference.postprocessing import DetectionPostprocessor
from backend.inference.preprocessing import ImagePreprocessor
from backend.schemas.api import DetectionReportRequest
from backend.schemas.defect import TIANCHI_CLASS_NAMES

logger = logging.getLogger(__name__)


class FabricDefectDetector:
    """
    High-level fabric defect detector.

    Usage:
        detector = FabricDefectDetector(backend="dummy")
        detector.load()

        # Single image
        detections = detector.detect(image_array)

        # Build API report
        request = detector.build_report(
            device_id="CAM-001",
            batch_id="BATCH-01",
            images=[image],
        )
    """

    DEFECT_CLASSES = list(TIANCHI_CLASS_NAMES)

    def __init__(
        self,
        backend: str = "dummy",
        model_path: str = "",
        input_size: Tuple[int, int] = (640, 640),
        confidence_threshold: float = 0.5,
        nms_threshold: float = 0.45,
        device: str = "cpu",
        normalize: bool = True,
        seed: int = 42,
    ):
        """
        Args:
            backend: "dummy" | "onnx" | "pytorch" | "rtdetr"
            model_path: Path to model file.
            input_size: Model input size (W, H).
            confidence_threshold: Minimum confidence for detections.
            nms_threshold: IoU threshold for NMS.
            device: "cpu" | "cuda"
            normalize: Apply ImageNet normalization.
            seed: Random seed for dummy backend.
        """
        self.backend = backend

        # RT-DETR is trained at 1280; default to that resolution unless the
        # caller explicitly requested a different input_size.
        if backend == "rtdetr" and input_size == (640, 640):
            input_size = (1280, 1280)
        self.input_size = input_size

        # Create engine
        self.engine: InferenceEngine = create_engine(
            backend=backend,
            model_path=model_path,
            input_size=input_size,
            confidence_threshold=confidence_threshold,
            nms_threshold=nms_threshold,
            device=device,
            seed=seed,
        )

        # Pre/post processors (unused on the RT-DETR path, which preprocesses
        # and parses inside the engine).
        self.preprocessor = ImagePreprocessor(
            input_size=input_size,
            normalize=normalize,
        )
        self.postprocessor = DetectionPostprocessor(
            confidence_threshold=confidence_threshold,
            nms_threshold=nms_threshold,
            input_size=input_size[0],
        )

        self._loaded = False

    def load(self):
        """Load the model and warm up."""
        if not self._loaded and self.backend != "dummy":
            self.engine.load()
            self.engine.warmup()
        self._loaded = True
        logger.info(f"Detector ready (backend={self.backend})")

    def detect(
        self,
        images: List[np.ndarray],
    ) -> List[List[Dict]]:
        """
        Detect defects in a batch of images.

        Args:
            images: List of (H, W, 3) uint8 numpy arrays.

        Returns:
            List of per-image lists of detection dicts with:
              {defect_id, type, type_code, severity, bbox, confidence}
        """
        if not images:
            return []

        if self.backend == "rtdetr":
            return self._detect_rtdetr(images)

        # Record original sizes for bbox scaling
        original_sizes = [(img.shape[0], img.shape[1]) for img in images]

        # Preprocess
        preprocessed = self.preprocessor.preprocess_batch(images)

        # Infer
        t0 = time.perf_counter()
        raw = self.engine.infer(preprocessed)
        inference_time_ms = (time.perf_counter() - t0) * 1000

        # Postprocess
        detections = self.postprocessor.process(raw, original_sizes)

        # Attach inference time
        for det_list in detections:
            for det in det_list:
                det["inference_time_ms"] = round(inference_time_ms / len(images), 1)

        return detections

    def _detect_rtdetr(self, images: List[np.ndarray]) -> List[List[Dict]]:
        """
        RT-DETR path: send raw images straight to the engine.

        Skips the YOLO-style preprocessor (ImageNet letterbox) and
        postprocessor (NMS + input-size scaling); ``RTDETRONNXEngine.predict``
        performs its own scale-fill resize and returns xyxy boxes in original
        pixels, which we convert to the report convention of xywh.
        """
        t0 = time.perf_counter()
        raw = self.engine.predict(images)
        inference_time_ms = (time.perf_counter() - t0) * 1000
        per_image_ms = round(inference_time_ms / len(images), 1)

        detections: List[List[Dict]] = []
        for d in raw:
            items: List[Dict] = []
            boxes = d.get("boxes", np.zeros((0, 4), dtype=np.float32))
            scores = d.get("scores", np.zeros(0, dtype=np.float32))
            names = d.get("class_names", [])
            for j in range(len(boxes)):
                name = names[j] if names and j < len(names) else "other"
                x1, y1, x2, y2 = (float(v) for v in boxes[j])
                items.append(
                    {
                        "defect_id": f"D-{uuid.uuid4().hex[:8].upper()}",
                        "type": name,
                        "type_code": project_code(name),
                        "severity": project_severity(name),
                        "bbox": [
                            round(x1, 1),
                            round(y1, 1),
                            round(x2 - x1, 1),
                            round(y2 - y1, 1),
                        ],
                        "confidence": float(scores[j]),
                        "inference_time_ms": per_image_ms,
                    }
                )
            detections.append(items)
        return detections

    def detect_single(self, image: np.ndarray) -> List[Dict]:
        """Detect defects in a single image."""
        results = self.detect([image])
        return results[0] if results else []

    def build_report(
        self,
        device_id: str,
        batch_id: str,
        images: List[np.ndarray],
        frame_seq: int = 0,
    ) -> DetectionReportRequest:
        """
        Run detection and build an API-compatible report.

        Full pipeline: images → detect → DetectionReportRequest

        Args:
            device_id: Camera device ID.
            batch_id: Production batch ID.
            images: Raw images to analyze.
            frame_seq: Frame sequence number.

        Returns:
            DetectionReportRequest ready for POST /api/v1/detection/report.
        """
        detections_list = self.detect(images)

        # Flatten all detections
        all_defects = []
        by_type = {}
        by_severity = {}
        alarm_triggered = False
        alarm_type = ""

        for det_list in detections_list:
            for det in det_list:
                all_defects.append(
                    {
                        "type": det["type"],
                        "bbox": det["bbox"],
                        "confidence": det["confidence"],
                        "severity": det["severity"],
                    }
                )
                by_type[det["type"]] = by_type.get(det["type"], 0) + 1
                by_severity[det["severity"]] = by_severity.get(det["severity"], 0) + 1

                if det["severity"] in ("critical", "major"):
                    alarm_triggered = True
                    if det["severity"] == "critical":
                        alarm_type = "critical_defect"

        from datetime import datetime, timezone

        return DetectionReportRequest(
            device_id=device_id,
            batch_id=batch_id,
            timestamp=datetime.now(timezone.utc),
            results={
                "total_defects": len(all_defects),
                "defect_list": all_defects,
                "by_type": by_type,
                "by_severity": by_severity,
                "alarm_triggered": alarm_triggered,
                "alarm_type": alarm_type,
            },
        )

    def benchmark(self, batch_size: int = 1, iterations: int = 50) -> Dict:
        """Benchmark the full pipeline."""
        results = {"backend": self.backend, "batch_size": batch_size}
        engine_bench = self.engine.benchmark(batch_size, iterations)
        results.update(engine_bench)
        return results


# Singleton detector for the application
_detector: Optional[FabricDefectDetector] = None


def get_detector(backend: str = "dummy", **kwargs) -> FabricDefectDetector:
    """Get or create the singleton detector."""
    global _detector
    if _detector is None:
        _detector = FabricDefectDetector(backend=backend, **kwargs)
        _detector.load()
    return _detector
