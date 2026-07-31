"""
Pydantic schemas for internal module interfaces.

Matches Section 3.1 of the project document:
  - 3.1.1 ImageCaptureOutput
  - 3.1.2 PreprocessOutput
  - 3.1.3 DetectionOutput
  - 3.1.4 BrokenYarnAlert
  - 3.1.5 MissingStitchDetection
"""

from datetime import datetime
from typing import Dict, List, Optional

from pydantic import BaseModel, Field

# ============================================================================
# 3.1.1 Image Capture -> Preprocessing Interface
# ============================================================================


class ImageHeader(BaseModel):
    """Common header for all internal messages."""

    timestamp: datetime = Field(..., description="ISO 8601 timestamp")
    device_id: str = Field(..., description="Camera/device identifier")
    batch_id: str = Field(..., description="Production batch identifier")
    frame_seq: int = Field(..., ge=0, description="Sequential frame number")


class RawImage(BaseModel):
    """Raw image data block."""

    format: str = Field(default="raw_8bit", description="Image pixel format")
    width: int = Field(..., gt=0, description="Image width in pixels")
    height: int = Field(..., gt=0, description="Image height in pixels")
    channels: int = Field(default=1, ge=1, le=4, description="Number of color channels")
    data: str = Field(..., description="Base64-encoded image data")


class CameraMetadata(BaseModel):
    """Camera and environmental metadata."""

    line_speed: float = Field(..., ge=0, description="Line speed in m/min")
    exposure_time: float = Field(..., ge=0, description="Exposure time in ms")
    gain: float = Field(default=1.0, ge=0)
    temperature: Optional[float] = Field(
        default=None, description="Sensor temperature in C"
    )


class ImageCaptureOutput(BaseModel):
    """Output from image capture module (Section 3.1.1)."""

    header: ImageHeader
    image: RawImage
    metadata: CameraMetadata


# ============================================================================
# 3.1.2 Preprocessing -> Detection Interface
# ============================================================================


class ProcessedImage(BaseModel):
    """Preprocessed image ready for inference."""

    format: str = Field(default="rgb_8bit")
    width: int = Field(..., gt=0)
    height: int = Field(..., gt=0)
    channels: int = Field(default=3)
    data: str = Field(..., description="Base64-encoded image data")


class ROI(BaseModel):
    """Region of Interest definition."""

    roi_id: str = Field(..., description="ROI identifier")
    x: int = Field(..., ge=0)
    y: int = Field(..., ge=0)
    width: int = Field(..., gt=0)
    height: int = Field(..., gt=0)


class PreprocessLog(BaseModel):
    """Log of preprocessing operations applied."""

    noise_filter: Optional[str] = Field(default="median")
    contrast_enhance: bool = Field(default=True)
    illumination_correction: bool = Field(default=True)


class PreprocessOutput(BaseModel):
    """Output from preprocessing module (Section 3.1.2)."""

    header: ImageHeader
    processed_image: ProcessedImage
    roi_list: List[ROI] = Field(default_factory=list)
    preprocess_log: PreprocessLog = Field(default_factory=PreprocessLog)


# ============================================================================
# 3.1.3 Detection -> Result Output Interface
# ============================================================================


class DefectItem(BaseModel):
    """Single detected defect."""

    defect_id: str = Field(..., description="Unique defect identifier")
    type: str = Field(..., description="Defect type name, e.g. broken_yarn")
    type_code: str = Field(..., description="Standard defect code, e.g. BY-01")
    severity: str = Field(
        ..., description="Severity: critical / major / medium / minor"
    )
    bbox: List[float] = Field(
        ..., min_length=4, max_length=4, description="[x, y, width, height]"
    )
    confidence: float = Field(..., ge=0.0, le=1.0, description="Detection confidence")
    segmentation_mask: Optional[str] = Field(
        default=None, description="RLE-encoded mask"
    )
    dimension_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    scalar_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)


class AlarmSignal(BaseModel):
    """Alarm signal for detected defects."""

    triggered: bool = Field(default=False)
    alarm_type: Optional[str] = Field(
        default=None, description="critical_defect / major_defect"
    )
    action: Optional[str] = Field(
        default=None, description="stop_machine / mark_roll / log_only"
    )


class Statistics(BaseModel):
    """Per-frame detection statistics."""

    total_defects: int = Field(default=0, ge=0)
    by_type: Dict[str, int] = Field(default_factory=dict)
    by_severity: Dict[str, int] = Field(default_factory=dict)


class DetectionOutput(BaseModel):
    """Output from defect detection module (Section 3.1.3)."""

    header: ImageHeader
    inference_time_ms: Optional[float] = Field(
        default=None, description="Inference time in ms"
    )
    defects: List[DefectItem] = Field(default_factory=list)
    alarm: AlarmSignal = Field(default_factory=AlarmSignal)
    statistics: Statistics = Field(default_factory=Statistics)


# ============================================================================
# 3.1.4 Broken Yarn Alert Interface
# ============================================================================


class BrokenYarnPosition(BaseModel):
    """Position info for broken yarn alert."""

    yarn_id: str = Field(..., description="Yarn identifier (e.g. WARP-023)")
    loom_position: int = Field(..., ge=0, description="Position on the loom")
    image_coords: List[float] = Field(
        ..., min_length=4, max_length=4, description="[x, y, w, h]"
    )


class BrokenYarnAlertHeader(BaseModel):
    """Header for broken yarn alerts."""

    timestamp: datetime
    device_id: str
    batch_id: str


class BrokenYarnAlert(BaseModel):
    """Broken yarn alert (Section 3.1.4)."""

    header: BrokenYarnAlertHeader
    alert: dict = Field(
        ..., description="Alert details with alert_id, type, position, etc."
    )
    context: dict = Field(
        ..., description="Context: line_speed_before, tension_value, yarn_count"
    )


# ============================================================================
# 3.1.5 Missing Stitch Detection Interface
# ============================================================================


class MissingStitchItem(BaseModel):
    """Individual missing stitch detection."""

    position: List[float] = Field(
        ..., min_length=4, max_length=4, description="[x, y, w, h]"
    )
    row: int = Field(..., ge=0)
    col: int = Field(..., ge=0)
    confidence: float = Field(..., ge=0.0, le=1.0)


class MissingStitchHeader(BaseModel):
    """Header for missing stitch detection."""

    timestamp: datetime
    device_id: str
    batch_id: str


class MissingStitchDetection(BaseModel):
    """Missing stitch detection result (Section 3.1.5)."""

    header: MissingStitchHeader
    result: dict = Field(
        ..., description="Detection result with stitch density, missing list, etc."
    )
