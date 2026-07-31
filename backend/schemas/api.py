"""
Pydantic schemas for external API endpoints.

Matches Section 3.2 of the project document:
  - 3.2.1 POST /api/v1/detection/report
  - 3.2.2 POST /api/v1/model/update
  - 3.2.3 GET /api/v1/system/status
"""

from datetime import datetime
from typing import Any, List, Optional

from pydantic import BaseModel, Field

# ============================================================================
# Generic API Response Wrapper
# ============================================================================


class APIResponse(BaseModel):
    """Standard API response envelope."""

    code: int = Field(default=0, description="0 = success, non-zero = error")
    message: str = Field(default="success")
    data: Optional[Any] = Field(default=None)


class ErrorResponse(BaseModel):
    """Error response."""

    code: int = Field(..., gt=0)
    message: str = Field(..., description="Human-readable error message")
    detail: Optional[str] = Field(default=None)


# ============================================================================
# 3.2.1 Detection Report API
# ============================================================================


class DefectReportItem(BaseModel):
    """Single defect in a detection report."""

    type: str = Field(..., description="Defect type name")
    bbox: List[float] = Field(
        ..., min_length=4, max_length=4, description="[x, y, w, h]"
    )
    confidence: float = Field(..., ge=0.0, le=1.0)
    severity: str = Field(..., description="critical / major / medium / minor")


class DetectionReportRequest(BaseModel):
    """Request body for POST /api/v1/detection/report (Section 3.2.1)."""

    device_id: str = Field(..., min_length=1, max_length=32)
    batch_id: str = Field(..., min_length=1, max_length=64)
    timestamp: datetime = Field(...)
    results: dict = Field(
        ..., description="Detection results with total_defects and defect_list"
    )


class DetectionReportResponseData(BaseModel):
    """Response data for detection report."""

    report_id: str
    accepted: bool = True


# ============================================================================
# 3.2.2 Model Update API
# ============================================================================


class VersionInfo(BaseModel):
    """Model version metadata."""

    release_date: str
    accuracy: float = Field(..., ge=0.0, le=1.0)
    improvements: List[str] = Field(default_factory=list)


class ModelUpdateRequest(BaseModel):
    """Request body for POST /api/v1/model/update (Section 3.2.2)."""

    model_id: str = Field(..., min_length=1, max_length=32)
    model_type: str = Field(default="yolov8_defect")
    model_url: str = Field(..., description="URL to download the model")
    checksum: str = Field(..., description="sha256:abc123...")
    version_info: VersionInfo


class ModelUpdateResponseData(BaseModel):
    """Response data for model update."""

    update_id: str
    estimated_deploy_time: int = Field(
        default=120, ge=0, description="Seconds until deployment"
    )


# ============================================================================
# 3.2.3 System Status API
# ============================================================================


class DeviceStatus(BaseModel):
    """Status of a single device."""

    device_id: str
    status: str = Field(..., description="online / offline / maintenance")
    uptime_hours: float = Field(default=0.0, ge=0)
    total_detections: int = Field(default=0, ge=0)
    defect_count: int = Field(default=0, ge=0)
    last_heartbeat: Optional[str] = Field(default=None)


class ModelStatus(BaseModel):
    """Status of the current model."""

    current_version: str
    last_updated: Optional[str] = None


class SystemMetrics(BaseModel):
    """System resource metrics."""

    cpu_usage: float = Field(default=0.0, ge=0.0, le=100.0)
    memory_usage: float = Field(default=0.0, ge=0.0, le=100.0)
    gpu_usage: Optional[float] = Field(default=None, ge=0.0, le=100.0)


class SystemStatusResponseData(BaseModel):
    """Response data for GET /api/v1/system/status (Section 3.2.3)."""

    devices: List[DeviceStatus] = Field(default_factory=list)
    model: ModelStatus
    system: SystemMetrics
