"""
Postprocessing for detection model outputs.

Pipeline:
  1. Confidence threshold filtering
  2. NMS (Non-Maximum Suppression)
  3. Bbox coordinate conversion (input coords → original image coords)
  4. Defect type mapping to standard codes
"""

from typing import Dict, List, Optional, Tuple

import numpy as np

from backend.schemas.defect import DEFECT_CODES, DEFECT_SEVERITY, TIANCHI_CLASS_NAMES, DefectType


class DetectionPostprocessor:
    """
    Postprocesses raw model outputs into structured detection results.
    """

    DEFECT_CLASSES = list(TIANCHI_CLASS_NAMES)

    def __init__(
        self,
        confidence_threshold: float = 0.5,
        nms_threshold: float = 0.45,
        max_detections: int = 100,
        input_size: int = 640,
    ):
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.max_detections = max_detections
        self.input_size = input_size

    def process(
        self,
        raw_detections: List[Dict],
        original_sizes: Optional[List[Tuple[int, int]]] = None,
    ) -> List[List[Dict]]:
        """
        Postprocess batch of raw detections.

        Args:
            raw_detections: List of per-image raw outputs from engine.infer().
            original_sizes: Optional list of (H, W) original image sizes
                           for coordinate scaling.

        Returns:
            List of per-image lists of filtered detection dicts:
              [{defect_id, type, type_code, severity, bbox, confidence}, ...]
        """
        results = []
        for i, raw in enumerate(raw_detections):
            detections = self._process_single(raw)
            # Scale boxes to original image if sizes provided
            if original_sizes and i < len(original_sizes):
                detections = self._scale_boxes(detections, original_sizes[i])
            results.append(detections[: self.max_detections])
        return results

    def _process_single(self, raw: Dict) -> List[Dict]:
        """Process a single image's raw detections."""
        boxes = raw.get("boxes", np.zeros((0, 4)))
        scores = raw.get("scores", np.zeros(0))
        classes = raw.get("classes", np.zeros(0, dtype=np.int64))
        class_names = raw.get("class_names", [])

        if len(boxes) == 0:
            return []

        # 1. Confidence filtering
        keep = scores >= self.confidence_threshold
        boxes = boxes[keep]
        scores = scores[keep]
        classes = classes[keep]
        if class_names:
            class_names = [class_names[i] for i, k in enumerate(keep) if k]

        if len(boxes) == 0:
            return []

        # 2. NMS
        keep_indices = self._nms(boxes, scores)
        boxes = boxes[keep_indices]
        scores = scores[keep_indices]
        classes = classes[keep_indices]
        if class_names:
            class_names = [class_names[i] for i in keep_indices]

        # 3. Build structured results
        detections = []
        for j in range(len(boxes)):
            cls_idx = int(classes[j])
            cls_name = (
                class_names[j]
                if class_names and j < len(class_names)
                else self._class_name(cls_idx)
            )
            code = self._defect_code(cls_name)
            severity = self._severity(cls_name)
            import uuid

            detections.append(
                {
                    "defect_id": f"D-{uuid.uuid4().hex[:8].upper()}",
                    "type": cls_name,
                    "type_code": code,
                    "severity": severity,
                    "bbox": boxes[j].tolist(),
                    "confidence": float(scores[j]),
                }
            )

        return detections

    def _nms(self, boxes: np.ndarray, scores: np.ndarray) -> np.ndarray:
        """
        Non-Maximum Suppression.

        Args:
            boxes: (N, 4) in [x, y, w, h].
            scores: (N,).

        Returns:
            Indices of kept boxes.
        """
        if len(boxes) == 0:
            return np.array([], dtype=np.int64)

        # Convert [x,y,w,h] → [x1,y1,x2,y2]
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 0] + boxes[:, 2]
        y2 = boxes[:, 1] + boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)

        order = scores.argsort()[::-1]
        keep = []

        while len(order) > 0:
            i = order[0]
            keep.append(i)

            if len(order) == 1:
                break

            # IoU with remaining boxes
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0, xx2 - xx1)
            h = np.maximum(0, yy2 - yy1)
            inter = w * h
            iou = inter / (areas[i] + areas[order[1:]] - inter)

            remain = np.where(iou <= self.nms_threshold)[0]
            order = order[remain + 1]

        return np.array(keep, dtype=np.int64)

    def _scale_boxes(
        self, detections: List[Dict], original_size: Tuple[int, int]
    ) -> List[Dict]:
        """Scale bboxes from model input size back to original image size."""
        oh, ow = original_size
        scale_x = ow / self.input_size
        scale_y = oh / self.input_size
        for det in detections:
            bbox = det["bbox"]
            det["bbox"] = [
                round(bbox[0] * scale_x, 1),
                round(bbox[1] * scale_y, 1),
                round(bbox[2] * scale_x, 1),
                round(bbox[3] * scale_y, 1),
            ]
        return detections

    @staticmethod
    def _class_name(cls_idx: int) -> str:
        """Map class index to name."""
        classes = DetectionPostprocessor.DEFECT_CLASSES
        if 0 <= cls_idx < len(classes):
            return classes[cls_idx]
        return "other"

    @staticmethod
    def _defect_code(type_name: str) -> str:
        """Map defect type name to standard code."""
        try:
            dt = DefectType(type_name)
            return DEFECT_CODES[dt].value
        except (ValueError, KeyError):
            return "OT-01"

    @staticmethod
    def _severity(type_name: str) -> str:
        """Determine severity from defect type via the canonical severity map."""
        try:
            return DEFECT_SEVERITY[DefectType(type_name)].value
        except (ValueError, KeyError):
            return "info"
