"""
Model version registry for tracking deployed defect detection models.

Maps to the model update API in Section 3.2.2.
"""

from datetime import datetime

from sqlalchemy import JSON, DateTime, Float, String
from sqlalchemy.orm import Mapped, mapped_column

from backend.database import Base


class ModelVersion(Base):
    """
    Registry entry for a deployed or deployable model version.
    """

    __tablename__ = "model_versions"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    model_id: Mapped[str] = mapped_column(
        String(32), unique=True, index=True, nullable=False
    )
    model_type: Mapped[str] = mapped_column(String(32), default="yolov8_defect")
    model_url: Mapped[str] = mapped_column(String(512), nullable=False)
    checksum: Mapped[str] = mapped_column(String(128), nullable=False)

    # Version info (JSON)
    version_info: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)

    # Metrics
    accuracy: Mapped[float] = mapped_column(Float, default=0.0)
    mAP50: Mapped[float] = mapped_column(Float, default=0.0)
    recall: Mapped[float] = mapped_column(Float, default=0.0)

    # Deployment status
    status: Mapped[str] = mapped_column(
        String(20), default="pending"
    )  # pending / deploying / active / failed / retired
    is_current: Mapped[bool] = mapped_column(default=False)

    # Timing
    release_date: Mapped[datetime] = mapped_column(DateTime, nullable=True)
    deployed_at: Mapped[datetime] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    def to_dict(self) -> dict:
        return {
            "model_id": self.model_id,
            "model_type": self.model_type,
            "model_url": self.model_url,
            "checksum": self.checksum,
            "version_info": self.version_info,
            "accuracy": self.accuracy,
            "mAP50": self.mAP50,
            "recall": self.recall,
            "status": self.status,
            "is_current": self.is_current,
            "release_date": self.release_date.isoformat()
            if self.release_date
            else None,
            "deployed_at": self.deployed_at.isoformat() if self.deployed_at else None,
        }
