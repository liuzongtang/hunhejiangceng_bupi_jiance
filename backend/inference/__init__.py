"""
Inference engine for fabric defect detection.

Supports ONNX Runtime, PyTorch, and Dummy backends.
"""

from backend.inference.detector import FabricDefectDetector, get_detector
from backend.inference.engine import (
    DummyInferenceEngine,
    InferenceEngine,
    ONNXInferenceEngine,
    PyTorchInferenceEngine,
    create_engine,
)
from backend.inference.postprocessing import DetectionPostprocessor
from backend.inference.preprocessing import ImagePreprocessor

__all__ = [
    "DetectionPostprocessor",
    "DummyInferenceEngine",
    "FabricDefectDetector",
    "ImagePreprocessor",
    "InferenceEngine",
    "ONNXInferenceEngine",
    "PyTorchInferenceEngine",
    "create_engine",
    "get_detector",
]
