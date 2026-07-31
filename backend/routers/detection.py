"""
Detection report router — POST /api/v1/detection/report (Section 3.2.1).
"""

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from backend.database import get_session
from backend.schemas.api import (
    APIResponse,
    DetectionReportRequest,
)
from backend.services.detection_service import DetectionService

router = APIRouter(prefix="/api/v1/detection", tags=["detection"])


@router.post("/report", response_model=APIResponse)
async def submit_detection_report(
    request: DetectionReportRequest,
    session: AsyncSession = Depends(get_session),
):
    """
    Submit a detection report from an edge device.

    Accepts detection results including defect list, bounding boxes,
    confidence scores, and severity levels. Triggers alarms for
    critical defects (broken_yarn, missing_stitch, hole).

    Returns a report_id for tracking.
    """
    try:
        service = DetectionService(session)
        result = await service.process_report(request)
        return APIResponse(
            code=0,
            message="success",
            data=result.model_dump(),
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process detection report: {e!s}",
        ) from e


@router.post("/infer", response_model=APIResponse)
async def run_inference(
    device_id: str = "CAM-001",
    batch_id: str = "",
    session=Depends(get_session),
):
    """
    Run defect detection inference and return results.

    Uses the dummy engine by default. Configure via backend config.
    Returns the same format as /report for consistency.
    """
    from datetime import datetime, timezone

    import numpy as np

    # Create a synthetic test image (in production, this would come from camera)
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    from backend.inference.detector import get_detector

    detector = get_detector(backend="dummy")

    report = detector.build_report(
        device_id=device_id,
        batch_id=batch_id
        or f"INFER-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}",
        images=[test_image],
    )

    # Store in database
    from backend.services.detection_service import DetectionService

    service = DetectionService(session)
    result = await service.process_report(report)

    return APIResponse(
        code=0,
        message="inference complete",
        data={
            "report_id": result.report_id,
            "detections": report.results.get("defect_list", []),
            "total_defects": report.results.get("total_defects", 0),
        },
    )


@router.get("/reports", response_model=APIResponse)
async def list_detection_reports(
    device_id: Optional[str] = None,
    limit: int = 20,
    session: AsyncSession = Depends(get_session),
):
    """List recent detection reports, optionally filtered by device."""
    from sqlalchemy import select

    from backend.models.detection_report import DetectionReport

    query = select(DetectionReport).order_by(DetectionReport.created_at.desc())
    if device_id:
        query = query.where(DetectionReport.device_id == device_id)
    query = query.limit(limit)

    result = await session.execute(query)
    reports = result.scalars().all()
    return APIResponse(
        code=0,
        message="success",
        data=[r.to_dict() for r in reports],
    )
