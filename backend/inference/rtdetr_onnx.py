"""
Dedicated RT-DETR ONNX inference engine + parser.

The project's legacy `InferenceEngine` backends assume a YOLO-style output of
shape [batch, N, 4 + 1 + nc] (an objectness score at index 4) and ImageNet
mean/std normalization. RT-DETR differs on both axes:

  * Input: stretch-resize to a square (LetterBox scale_fill=True), BGR->RGB,
    divide by 255. No ImageNet mean/std.
  * Output: a single (1, 300, 6) tensor of [cx, cy, w, h, score, class] with
    cx..h normalized 0..1 and NO objectness channel. The decoder already did
    top-k selection, so no NMS is required.

Because the input is scale-filled (stretched, no letterbox padding), normalized
coordinates map back to the original image by simple multiplication by the
original width/height. This mirrors `ultralytics.models.rtdetr.predict`.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np


class RTDETRONNXEngine:
    """Run an exported RT-DETR ONNX model and parse its detections.

    Unlike `backend.inference.engine.InferenceEngine`, this engine is
    self-contained: it performs its own preprocessing on raw BGR uint8 images
    and does not expect the caller to have pre-normalized them.
    """

    def __init__(
        self,
        model_path: str,
        imgsz: int = 1280,
        conf_threshold: float = 0.25,
        class_names: Optional[List[str]] = None,
        providers: Optional[List[str]] = None,
    ):
        self.model_path = model_path
        self.imgsz = imgsz
        self.conf_threshold = conf_threshold
        self.class_names = class_names or []
        self.providers = providers
        self._session = None
        self._input_name = None

    def load(self) -> "RTDETRONNXEngine":
        """Load the ONNX model into an onnxruntime session."""
        import onnxruntime as ort

        wanted = self.providers or ["CUDAExecutionProvider", "CPUExecutionProvider"]
        available = ort.get_available_providers()
        providers = [p for p in wanted if p in available] or ["CPUExecutionProvider"]
        self._session = ort.InferenceSession(self.model_path, providers=providers)
        self._input_name = self._session.get_inputs()[0].name
        return self

    def warmup(self, iterations: int = 1) -> None:
        """No-op warmup for interface parity with ``InferenceEngine``."""

    def infer(self, images: List[np.ndarray]) -> List[Dict]:
        """Alias for :meth:`predict` to match the ``InferenceEngine`` interface."""
        return self.predict(images)

    # -- preprocessing ----------------------------------------------------

    def preprocess(self, image_bgr: np.ndarray) -> np.ndarray:
        """Convert a BGR (H, W, 3) uint8 image to a (1, 3, imgsz, imgsz) tensor.

        Uses scale-fill (stretch to the square, no padding), matching the
        RT-DETR inference path used during training/validation.
        """
        import cv2

        resized = cv2.resize(
            image_bgr, (self.imgsz, self.imgsz), interpolation=cv2.INTER_LINEAR
        )
        rgb = resized[:, :, ::-1]  # BGR -> RGB
        tensor = rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
        return np.ascontiguousarray(tensor[None, ...])

    # -- inference --------------------------------------------------------

    def predict(self, images: List[np.ndarray]) -> List[Dict]:
        """Run inference on a list of BGR (H, W, 3) uint8 images.

        Returns one dict per image (see parse_output for the keys).
        """
        results: List[Dict] = []
        for image in images:
            orig_h, orig_w = image.shape[:2]
            tensor = self.preprocess(image)
            output = self._session.run(None, {self._input_name: tensor})[0]
            results.append(self.parse_output(output, orig_w, orig_h))
        return results

    # -- the dedicated parser ---------------------------------------------

    def parse_output(self, output: np.ndarray, orig_w: int, orig_h: int) -> Dict:
        """Parse RT-DETR ONNX output (1, 300, 6) into structured detections.

        Args:
            output: (1, 300, 6) tensor of [cx, cy, w, h, score, class] with
                cx..h normalized 0..1.
            orig_w: Original image width (pixels).
            orig_h: Original image height (pixels).

        Returns:
            Dict with:
              - boxes: (N, 4) float32 [x1, y1, x2, y2] in original pixels
              - scores: (N,) float32
              - classes: (N,) int64
              - class_names: (N,) str
        """
        preds = output[0]  # (300, 6)
        if preds.size == 0:
            return _empty_detection()

        boxes_n = preds[:, :4]
        scores = preds[:, 4]
        labels = preds[:, 5].astype(np.int64)

        cx, cy, w, h = boxes_n[:, 0], boxes_n[:, 1], boxes_n[:, 2], boxes_n[:, 3]
        x1 = (cx - w / 2) * orig_w
        y1 = (cy - h / 2) * orig_h
        x2 = (cx + w / 2) * orig_w
        y2 = (cy + h / 2) * orig_h

        keep = scores >= self.conf_threshold
        boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)[keep].astype(np.float32)
        scores = scores[keep].astype(np.float32)
        classes = labels[keep]
        class_names = [
            self.class_names[c] if c < len(self.class_names) else str(c)
            for c in classes
        ]

        return {
            "boxes": boxes_xyxy,
            "scores": scores,
            "classes": classes,
            "class_names": class_names,
        }


def _empty_detection() -> Dict:
    return {
        "boxes": np.zeros((0, 4), dtype=np.float32),
        "scores": np.zeros((0,), dtype=np.float32),
        "classes": np.zeros((0,), dtype=np.int64),
        "class_names": [],
    }
