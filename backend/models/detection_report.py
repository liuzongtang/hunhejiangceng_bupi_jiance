"""
Detection report model for batch-level detection summaries.

Maps to the detection report API in Section 3.2.1.
"""

from __future__ import annotations

from datetime import datetime
from typing import List

from sqlalchemy import JSON, DateTime, Integer, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.database import Base


class DetectionReport(Base):
    """
    Batch-level detection summary report.

    A report aggregates multiple DefectRecords from a single device batch.
    """

    __tablename__ = "detection_reports"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    report_id: Mapped[str] = mapped_column(
        String(64), unique=True, index=True, nullable=False
    )
    device_id: Mapped[str] = mapped_column(String(32), index=True)
    batch_id: Mapped[str] = mapped_column(String(64), index=True)

    # Summary statistics
    total_defects: Mapped[int] = mapped_column(Integer, default=0)
    by_type: Mapped[dict] = mapped_column(JSON, default=dict)
    by_severity: Mapped[dict] = mapped_column(JSON, default=dict)

    # Alarm info
    alarm_triggered: Mapped[bool] = mapped_column(default=False)
    alarm_type: Mapped[str] = mapped_column(String(32), nullable=True, default="")
    alarm_action: Mapped[str] = mapped_column(String(32), nullable=True, default="")

    # Timing
    timestamp: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    # Relationships
    defects: Mapped[List["DefectRecord"]] = relationship(
        back_populates="report", lazy="selectin"
    )

    def to_dict(self) -> dict:
        return {
            "report_id": self.report_id,
            "device_id": self.device_id,
            "batch_id": self.batch_id,
            "total_defects": self.total_defects,
            "by_type": self.by_type,
            "by_severity": self.by_severity,
            "alarm": {
                "triggered": self.alarm_triggered,
                "alarm_type": self.alarm_type,
                "action": self.alarm_action,
            },
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
        }
