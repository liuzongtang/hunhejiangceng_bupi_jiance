"""
Device model for factory-floor inspection units.

Maps to system status API response in Section 3.2.3.
"""

from datetime import datetime

from sqlalchemy import DateTime, Float, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from backend.database import Base


class Device(Base):
    """
    Inspection device/camera unit installed on the production line.
    """

    __tablename__ = "devices"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    device_id: Mapped[str] = mapped_column(
        String(32), unique=True, index=True, nullable=False
    )

    # Device metadata
    device_type: Mapped[str] = mapped_column(String(32), default="line_scan_camera")
    location: Mapped[str] = mapped_column(String(128), nullable=True, default="")
    line_name: Mapped[str] = mapped_column(String(64), nullable=True, default="")

    # Status
    status: Mapped[str] = mapped_column(
        String(16), default="offline"
    )  # online / offline / maintenance
    uptime_hours: Mapped[float] = mapped_column(Float, default=0.0)

    # Counters
    total_detections: Mapped[int] = mapped_column(
        Integer, default=0, server_default="0"
    )
    defect_count: Mapped[int] = mapped_column(Integer, default=0, server_default="0")

    # Line parameters
    line_speed: Mapped[float] = mapped_column(Float, default=0.0)  # m/min
    tension_value: Mapped[float] = mapped_column(Float, default=0.0)
    yarn_count: Mapped[int] = mapped_column(Integer, default=0)

    # Timestamps
    last_heartbeat: Mapped[datetime] = mapped_column(DateTime, nullable=True)
    first_seen: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )

    def to_dict(self) -> dict:
        return {
            "device_id": self.device_id,
            "status": self.status,
            "uptime_hours": self.uptime_hours,
            "total_detections": self.total_detections,
            "defect_count": self.defect_count,
            "last_heartbeat": self.last_heartbeat.isoformat()
            if self.last_heartbeat
            else None,
        }
