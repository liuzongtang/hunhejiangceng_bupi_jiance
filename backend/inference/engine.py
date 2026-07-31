"""
Inference engine backends for fabric defect detection.

Supports three backends:
  - ONNXInferenceEngine: ONNX Runtime (production, TensorRT compatible)
  - PyTorchInferenceEngine: Native PyTorch (development / debugging)
  - DummyInferenceEngine: Simulated detections (testing without model)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import logging
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class InferenceEngine(ABC):
    """Abstract base class for inference engines."""

    def __init__(
        self,
        model_path: str = "",
        input_size: Tuple[int, int] = (640, 640),
        confidence_threshold: float = 0.5,
        nms_threshold: float = 0.45,
        device: str = "cpu",
    ):
        self.model_path = model_path
        self.input_size = input_size
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.device = device
        self._warmed_up = False

    @abstractmethod
    def infer(self, images: np.ndarray) -> List[Dict]:
        """
        Run inference on a batch of images.

        Args:
            images: (B, C, H, W) normalized float32 array.

        Returns:
            List of detection dicts per image, each with:
              - boxes: (N, 4) [x, y, w, h] in input coordinates
              - scores: (N,) confidence scores
              - classes: (N,) class indices
              - class_names: (N,) class name strings
        """
        ...

    @abstractmethod
    def load(self):
        """Load model into memory."""
        ...

    def warmup(self, iterations: int = 3):
        """Warm up the engine with dummy data."""
        if self._warmed_up:
            return
        dummy = np.random.randn(1, 3, *self.input_size).astype(np.float32)
        for _ in range(iterations):
            self.infer(dummy)
        self._warmed_up = True
        logger.info(f"Engine warmed up ({iterations} iterations)")

    def benchmark(self, batch_size: int = 1, iterations: int = 100) -> Dict[str, float]:
        """Benchmark inference speed."""
        dummy = np.random.randn(batch_size, 3, *self.input_size).astype(np.float32)
        self.warmup(3)

        times = []
        for _ in range(iterations):
            start = time.perf_counter()
            self.infer(dummy)
            times.append(time.perf_counter() - start)

        avg_ms = np.mean(times) * 1000
        fps = 1000 / avg_ms * batch_size
        return {
            "avg_latency_ms": round(avg_ms, 2),
            "fps": round(fps, 1),
            "batch_size": batch_size,
            "iterations": iterations,
        }


class DummyInferenceEngine(InferenceEngine):
    """
    Dummy engine that returns simulated detections.

    Useful for testing the full pipeline without a trained model.
    Returns a random number of detections per image with plausible values.
    """

    DEFECT_CLASSES = [
        "broken_yarn",
        "missing_stitch",
        "skip_stitch",
        "hole",
        "stain",
        "color_diff",
        "thick_yarn",
        "thin_yarn",
        "crease",
    ]

    def __init__(self, seed: int = 42, **kwargs):
        kwargs.pop("model_path", None)  # Dummy doesn't use model_path
        super().__init__(model_path="dummy", **kwargs)
        self.rng = np.random.RandomState(seed)

    def load(self):
        logger.info("Dummy engine loaded (no model required)")

    def infer(self, images: np.ndarray) -> List[Dict]:
        """Generate simulated detections."""
        batch_size = images.shape[0]
        results = []

        for _ in range(batch_size):
            n_detections = self.rng.randint(0, 4)
            if n_detections == 0:
                results.append(
                    {
                        "boxes": np.zeros((0, 4), dtype=np.float32),
                        "scores": np.zeros((0,), dtype=np.float32),
                        "classes": np.zeros((0,), dtype=np.int64),
                        "class_names": [],
                    }
                )
                continue

            h, w = self.input_size
            boxes = np.zeros((n_detections, 4), dtype=np.float32)
            boxes[:, 0] = self.rng.randint(0, w // 2, n_detections)
            boxes[:, 1] = self.rng.randint(0, h // 2, n_detections)
            boxes[:, 2] = self.rng.randint(20, w // 4, n_detections)
            boxes[:, 3] = self.rng.randint(20, h // 4, n_detections)

            scores = self.rng.uniform(0.55, 0.99, n_detections).astype(np.float32)
            class_indices = self.rng.randint(0, len(self.DEFECT_CLASSES), n_detections)
            class_names = [self.DEFECT_CLASSES[i] for i in class_indices]

            results.append(
                {
                    "boxes": boxes,
                    "scores": scores,
                    "classes": class_indices.astype(np.int64),
                    "class_names": class_names,
                }
            )

        return results


class ONNXInferenceEngine(InferenceEngine):
    """
    ONNX Runtime inference engine.

    Loads an ONNX model and runs inference via onnxruntime.
    """

    def __init__(self, model_path: str, **kwargs):
        kwargs.pop("model_path", None)
        super().__init__(model_path=model_path, **kwargs)
        self.session = None

    def load(self):
        """Load ONNX model."""
        try:
            import onnxruntime as ort

            providers = (
                ["CUDAExecutionProvider", "CPUExecutionProvider"]
                if self.device == "cuda"
                else ["CPUExecutionProvider"]
            )
            self.session = ort.InferenceSession(self.model_path, providers=providers)
            logger.info(f"ONNX model loaded: {self.model_path}")
            logger.info(f"  Providers: {self.session.get_providers()}")
            logger.info(f"  Input: {[i.name for i in self.session.get_inputs()]}")
        except ImportError as e:
            raise ImportError(
                "onnxruntime not installed. Run: pip install onnxruntime"
            ) from e
        except Exception as e:
            raise RuntimeError(f"Failed to load ONNX model: {e}") from e

    def infer(self, images: np.ndarray) -> List[Dict]:
        """Run ONNX inference."""
        if self.session is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        input_name = self.session.get_inputs()[0].name
        outputs = self.session.run(None, {input_name: images})
        return self._parse_outputs(outputs, images.shape[0])

    def _parse_outputs(self, outputs: list, batch_size: int) -> List[Dict]:
        """
        Parse raw ONNX outputs into structured detections.

        Supports common YOLO-style output: [batch, num_dets, 4 + 1 + num_classes]
        where the last dim is [x, y, w, h, confidence, class_probs...].
        """
        results = []
        for i in range(batch_size):
            if outputs and len(outputs) > 0 and isinstance(outputs[0], np.ndarray):
                out = outputs[0]  # First output tensor
                if out.ndim == 3 and out.shape[0] > i:
                    dets = out[i]  # (num_dets, 5+num_classes)
                    scores = dets[:, 4]  # objectness
                    class_probs = dets[:, 5:]  # class probabilities
                    classes = np.argmax(class_probs, axis=1)
                    boxes = dets[:, :4]

                    # Filter by confidence
                    keep = scores >= self.confidence_threshold
                    results.append(
                        {
                            "boxes": boxes[keep].astype(np.float32),
                            "scores": scores[keep].astype(np.float32),
                            "classes": classes[keep].astype(np.int64),
                            "class_names": [
                                self._class_name(int(c)) for c in classes[keep]
                            ],
                        }
                    )
                    continue

            # Fallback: return empty
            results.append(
                {
                    "boxes": np.zeros((0, 4), dtype=np.float32),
                    "scores": np.zeros((0,), dtype=np.float32),
                    "classes": np.zeros((0,), dtype=np.int64),
                    "class_names": [],
                }
            )
        return results


class PyTorchInferenceEngine(InferenceEngine):
    """
    PyTorch inference engine.

    Loads a PyTorch model and runs inference natively.
    """

    def __init__(self, model_path: str, model: Optional[object] = None, **kwargs):
        kwargs.pop("model_path", None)
        super().__init__(model_path=model_path, **kwargs)
        self._model = model

    def load(self):
        """Load PyTorch model."""
        try:
            import torch

            if self._model is not None:
                self.model = self._model.to(self.device)
            elif self.model_path:
                self.model = torch.load(
                    self.model_path, map_location=self.device, weights_only=False
                )
            else:
                raise ValueError("Either model_path or model must be provided")
            self.model.eval()
            logger.info(f"PyTorch model loaded: {self.model_path or 'in-memory'}")
        except Exception as e:
            raise RuntimeError(f"Failed to load PyTorch model: {e}") from e

    def infer(self, images: np.ndarray) -> List[Dict]:
        """Run PyTorch inference."""
        import torch

        with torch.no_grad():
            x = torch.from_numpy(images).to(self.device)
            outputs = self.model(x)
            return self._parse_outputs(outputs, images.shape[0])

    def _parse_outputs(self, outputs, batch_size: int) -> List[Dict]:
        """
        Parse PyTorch model outputs into structured detections.

        Supports tuple/list output: (boxes, scores, classes) or single tensor.
        """
        results = []
        for i in range(batch_size):
            try:
                if isinstance(outputs, (tuple, list)) and len(outputs) >= 3:
                    # (boxes, scores, classes) format
                    boxes = outputs[0]
                    scores = outputs[1]
                    classes = outputs[2]
                    if isinstance(boxes, torch.Tensor) and boxes.dim() >= 2:
                        if boxes.dim() == 3:
                            bi, si, ci = (
                                boxes[i].cpu().numpy(),
                                scores[i].cpu().numpy(),
                                classes[i].cpu().numpy(),
                            )
                        else:
                            bi, si, ci = (
                                boxes.cpu().numpy(),
                                scores.cpu().numpy(),
                                classes.cpu().numpy(),
                            )
                        keep = si >= self.confidence_threshold
                        results.append(
                            {
                                "boxes": bi[keep].astype(np.float32),
                                "scores": si[keep].astype(np.float32),
                                "classes": ci[keep].astype(np.int64),
                                "class_names": [
                                    self._class_name(int(c)) for c in ci[keep]
                                ],
                            }
                        )
                        continue

                elif isinstance(outputs, torch.Tensor) and outputs.dim() == 3:
                    # Single tensor: (batch, num_dets, 5+num_classes)
                    dets = outputs[i].cpu().numpy()
                    boxes = dets[:, :4]
                    scores = dets[:, 4]
                    class_probs = dets[:, 5:]
                    classes = np.argmax(class_probs, axis=1)
                    keep = scores >= self.confidence_threshold
                    results.append(
                        {
                            "boxes": boxes[keep].astype(np.float32),
                            "scores": scores[keep].astype(np.float32),
                            "classes": classes[keep].astype(np.int64),
                            "class_names": [
                                self._class_name(int(c)) for c in classes[keep]
                            ],
                        }
                    )
                    continue

            except Exception:
                pass

            # Fallback: empty
            results.append(
                {
                    "boxes": np.zeros((0, 4), dtype=np.float32),
                    "scores": np.zeros((0,), dtype=np.float32),
                    "classes": np.zeros((0,), dtype=np.int64),
                    "class_names": [],
                }
            )
        return results


def create_engine(
    backend: str = "dummy",
    model_path: str = "",
    **kwargs,
) -> InferenceEngine:
    """
    Factory: create an inference engine by backend name.

    Args:
        backend: "dummy" | "onnx" | "pytorch"
        model_path: Path to model file (not needed for dummy).
        **kwargs: Passed to engine constructor.

    Returns:
        Configured InferenceEngine instance.
    """
    engines = {
        "dummy": DummyInferenceEngine,
        "onnx": ONNXInferenceEngine,
        "pytorch": PyTorchInferenceEngine,
    }
    engine_cls = engines.get(backend)
    if engine_cls is None:
        raise ValueError(f"Unknown backend: {backend}. Options: {list(engines.keys())}")

    engine = engine_cls(model_path=model_path, **kwargs)
    if backend != "dummy":
        engine.load()
        engine.warmup()
    return engine
