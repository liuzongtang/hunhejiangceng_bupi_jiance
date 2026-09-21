"""
Defect records and annotation database models.

Matches the defect classification standard in Section 4.1 and
the annotation format in Section 4.2 of the project document.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import (
    JSON,
    DateTime,
    Enum as SAEnum,
    Float,
    ForeignKey,
    Integer,
    String,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.database import Base
from backend.schemas.defect import DEFECT_CODES, DefectType, Severity

# Re-export for backward compatibility
DEFECT_CODE_MAP = {k: v.value for k, v in DEFECT_CODES.items()}


def _enum_values(enum_cls):
    """Return an enum's string values so SAEnum persists values, not member names."""
    return [member.value for member in enum_cls]


class DefectRecord(Base):
    """
    Individual defect detection record.

    Maps to DetectionOutput.defects[] in Section 3.1.3.
    """

    __tablename__ = "defect_records"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    defect_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    type: Mapped[DefectType] = mapped_column(
        SAEnum(DefectType, values_callable=_enum_values), nullable=False
    )
    type_code: Mapped[str] = mapped_column(String(10), nullable=False)
    severity: Mapped[Severity] = mapped_column(
        SAEnum(Severity, values_callable=_enum_values), nullable=False
    )
    # Alert-only escalation: set when the predicted class is visually confusable
    # with a higher-impact class (e.g. surface_mark ~ broken_warp). ``severity``
    # keeps the canonical class severity; this records the escalated level used
    # to trigger the alarm so the escalation is auditable.
    escalated_severity: Mapped[Optional[Severity]] = mapped_column(
        SAEnum(Severity, values_callable=_enum_values), nullable=True
    )

    # Bounding box: [x, y, width, height]
    bbox: Mapped[dict] = mapped_column(JSON, nullable=False)

    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    dimension_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    scalar_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Segmentation mask (RLE encoded or polygon points)
    segmentation_mask: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)

    # Metadata
    device_id: Mapped[str] = mapped_column(String(32), index=True)
    batch_id: Mapped[str] = mapped_column(String(64), index=True)
    frame_seq: Mapped[int] = mapped_column(Integer, nullable=False)
    inference_time_ms: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Link to detection report
    report_id: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("detection_reports.id"), nullable=True
    )
    report: Mapped[Optional["DetectionReport"]] = relationship(back_populates="defects")

    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, index=True
    )

    def to_dict(self) -> dict:
        return {
            "defect_id": self.defect_id,
            "type": self.type.value,
            "type_code": self.type_code,
            "severity": self.severity.value,
            "escalated_severity": self.escalated_severity.value
            if self.escalated_severity
            else None,
            "bbox": list(self.bbox.values())
            if isinstance(self.bbox, dict)
            else self.bbox,
            "confidence": self.confidence,
            "segmentation_mask": self.segmentation_mask,
            "dimension_score": self.dimension_score,
            "scalar_score": self.scalar_score,
        }


class Annotation(Base):
    """
    Human-annotated defect label.

    Matches the annotation format in Section 4.2.
    """

    __tablename__ = "annotations"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    annotation_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    image_id: Mapped[str] = mapped_column(String(128), index=True, nullable=False)
    image_width: Mapped[int] = mapped_column(Integer, nullable=False)
    image_height: Mapped[int] = mapped_column(Integer, nullable=False)

    defect_type: Mapped[DefectType] = mapped_column(
        SAEnum(DefectType, values_callable=_enum_values), nullable=False
    )
    defect_code: Mapped[str] = mapped_column(String(10), nullable=False)
    severity: Mapped[Severity] = mapped_column(
        SAEnum(Severity, values_callable=_enum_values), nullable=False
    )

    # BBox: stored as JSON {x_min, y_min, x_max, y_max}
    bbox: Mapped[dict] = mapped_column(JSON, nullable=False)

    # Segmentation: polygon points
    segmentation: Mapped[dict] = mapped_column(JSON, nullable=True)

    annotator_id: Mapped[str] = mapped_column(String(32), nullable=False)
    review_status: Mapped[str] = mapped_column(
        String(20), default="pending"
    )  # pending / approved / rejected

    annotated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
