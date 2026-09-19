"""
Lightweight bridge to the canonical 20-class taxonomy in ``backend.schemas.defect``.

The trained RT-DETR detector outputs 20 fine-grained Tianchi classes whose ids
(0..19) are now identical to the project's defect classes, so the old lossy
20->10 mapping is gone. This module re-exports the 20 class names (English +
Chinese) and provides name->code / name->severity helpers for the inference
layer and scripts. It stays free of torch/cv2 imports so the lightweight
inference path keeps no heavy dependencies.
"""

from __future__ import annotations

from backend.schemas.defect import (
    DEFECT_CODES,
    DEFECT_SEVERITY,
    DefectType,
    TIANCHI_CLASS_NAMES,
    TIANCHI_CLASS_NAMES_ZH,
)

__all__ = [
    "TIANCHI_CLASS_NAMES",
    "TIANCHI_CLASS_NAMES_ZH",
    "project_code",
    "project_severity",
]


def project_code(defect_type: str) -> str:
    """Map a 20-class type name to its standard defect code (e.g. BW-01)."""
    try:
        return DEFECT_CODES[DefectType(defect_type)].value
    except (ValueError, KeyError):
        return "OT-01"


def project_severity(defect_type: str) -> str:
    """Map a 20-class type name to its severity (critical/major/medium/minor)."""
    try:
        return DEFECT_SEVERITY[DefectType(defect_type)].value
    except (ValueError, KeyError):
        return "info"
