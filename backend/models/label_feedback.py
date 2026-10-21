"""
Human-in-the-loop label feedback model (the #16 relabel loop).

Captures operator corrections to a detector's output — false positives, missed
defects, and misclassifications — so they can be converted back into YOLO label
corrections and fed into an incremental fine-tune. This closes the loop that
the deployment layer opens: alarms and low-confidence detections are reviewed,
and the review verdict becomes training data.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from sqlalchemy import JSON, DateTime, Enum as SAEnum, Float, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from backend.database import Base


class FeedbackType(str, Enum):
    """The three correction kinds an operator can supply."""

    FALSE_POSITIVE = "false_positive"  # 误报：删掉错误的检测框
    MISSED_DEFECT = "missed_defect"  # 漏检：补上漏掉的瑕疵框
    MISCLASSIFICATION = "misclassification"  # 错分：改类别


def _enum_values(enum_cls):
    """Return an enum's string values so SAEnum persists values, not names."""
    return [member.value for member in enum_cls]


class LabelFeedback(Base):
    """
    One operator correction to a detection result.

    The image is referenced by its YOLO label stem (``image_id``), so the
    feedback maps 1:1 onto a train/val label file and can be applied by
    ``scripts/relabel_loop.py``.
    """

    __tablename__ = "label_feedbacks"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    feedback_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    image_id: Mapped[str] = mapped_column(String(128), index=True, nullable=False)

    feedback_type: Mapped[FeedbackType] = mapped_column(
        SAEnum(FeedbackType, values_callable=_enum_values), nullable=False
    )

    # Corrected class (18-class id 0..17). For MISSED_DEFECT this is the class
    # of the box being added; for MISCLASSIFICATION the new class; for
    # FALSE_POSITIVE the class the detector wrongly reported.
    class_id: Mapped[int] = mapped_column(Integer, nullable=False)
    class_name: Mapped[str] = mapped_column(String(32), nullable=False)

    # Normalized YOLO box {cx, cy, w, h} in 0..1 — the box to add (missed),
    # correct (misclassification) or suppress (false positive).
    bbox: Mapped[dict] = mapped_column(JSON, nullable=False)

    # For MISCLASSIFICATION: the class id the detector originally predicted.
    old_class_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    # For FALSE_POSITIVE: the confidence the detector gave the wrong box.
    confidence: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    status: Mapped[str] = mapped_column(
        String(20), default="pending"
    )  # pending / applied / rejected
    source: Mapped[str] = mapped_column(String(20), default="dashboard")
    note: Mapped[Optional[str]] = mapped_column(String(256), nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    applied_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    def to_dict(self) -> dict:
        return {
            "feedback_id": self.feedback_id,
            "image_id": self.image_id,
            "feedback_type": self.feedback_type.value,
            "class_id": self.class_id,
            "class_name": self.class_name,
            "bbox": self.bbox,
            "old_class_id": self.old_class_id,
            "confidence": self.confidence,
            "status": self.status,
            "source": self.source,
            "note": self.note,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
