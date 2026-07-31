"""
System status router — GET /api/v1/system/status (Section 3.2.3).
"""

from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, Query
import psutil
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.database import get_session
from backend.models.device import Device
from backend.models.model_registry import ModelVersion
from backend.schemas.api import (
    APIResponse,
    DeviceStatus,
    ModelStatus as ModelStatusSchema,
    SystemMetrics,
    SystemStatusResponseData,
)

router = APIRouter(prefix="/api/v1/system", tags=["system"])


@router.get("/status", response_model=APIResponse)
async def get_system_status(
    device_id: str = Query(None, description="Filter by device ID"),
    session: AsyncSession = Depends(get_session),
):
    """
    Get system status including devices, current model, and resource usage.

    Matches the response format in Section 3.2.3.
    """
    # Get devices
    query = select(Device)
    if device_id:
        query = query.where(Device.device_id == device_id)
    result = await session.execute(query)
    devices = result.scalars().all()

    device_statuses = [
        DeviceStatus(
            device_id=d.device_id,
            status=d.status,
            uptime_hours=d.uptime_hours,
            total_detections=d.total_detections,
            defect_count=d.defect_count,
            last_heartbeat=d.last_heartbeat.isoformat() if d.last_heartbeat else None,
        )
        for d in devices
    ]

    # Get current model
    model_result = await session.execute(
        select(ModelVersion).where(ModelVersion.is_current)
    )
    current_model = model_result.scalar_one_or_none()

    model_status = ModelStatusSchema(
        current_version=current_model.model_id if current_model else "unknown",
        last_updated=(
            current_model.deployed_at.isoformat()
            if current_model and current_model.deployed_at
            else None
        ),
    )

    # Get system resource metrics
    metrics = SystemMetrics(
        cpu_usage=round(psutil.cpu_percent(interval=0.1), 1),
        memory_usage=round(psutil.virtual_memory().percent, 1),
        gpu_usage=None,  # GPU monitoring requires pynvml
    )

    data = SystemStatusResponseData(
        devices=device_statuses,
        model=model_status,
        system=metrics,
    )

    return APIResponse(code=0, message="success", data=data.model_dump())


@router.get("/stats", response_model=APIResponse)
async def get_system_stats(
    time_range: str = Query("24h", description="Time range: 1h, 24h, 7d, 30d"),
    session: AsyncSession = Depends(get_session),
):
    """
    Get aggregated detection statistics for dashboard charts.

    Returns hourly detection counts, defect type distribution, and alert counts.
    """
    from datetime import datetime, timezone

    from backend.models.detection_report import DetectionReport

    # Parse time range
    range_map = {"1h": 1, "24h": 24, "7d": 168, "30d": 720}
    hours = range_map.get(time_range, 24)
    since = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(hours=hours)

    # Get reports in range
    result = await session.execute(
        select(DetectionReport).where(DetectionReport.created_at >= since)
    )
    reports = result.scalars().all()

    total_defects = sum(r.total_defects for r in reports)
    total_reports = len(reports)
    alerts_triggered = sum(1 for r in reports if r.alarm_triggered)

    # Aggregate by type
    by_type = {}
    for r in reports:
        if r.by_type:
            for t, c in r.by_type.items():
                by_type[t] = by_type.get(t, 0) + c

    # Aggregate by severity
    by_severity = {}
    for r in reports:
        if r.by_severity:
            for s, c in r.by_severity.items():
                by_severity[s] = by_severity.get(s, 0) + c

    # Hourly trend data for charts
    hourly_labels = []
    hourly_counts = []
    for h in range(hours, 0, -1):
        slot = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(hours=h)
        label = slot.strftime("%H:00")
        count = sum(
            r.total_defects
            for r in reports
            if r.created_at
            and r.created_at >= slot
            and r.created_at < slot + timedelta(hours=1)
        )
        hourly_labels.append(label)
        hourly_counts.append(count)

    return APIResponse(
        code=0,
        message="success",
        data={
            "range": time_range,
            "total_reports": total_reports,
            "total_defects": total_defects,
            "alerts_triggered": alerts_triggered,
            "by_type": by_type,
            "by_severity": by_severity,
            "trend": {
                "labels": hourly_labels,
                "data": hourly_counts,
            },
        },
    )


@router.get("/health", response_model=APIResponse)
async def health_check():
    """Simple health check endpoint."""
    return APIResponse(
        code=0,
        message="healthy",
        data={"timestamp": datetime.now(timezone.utc).isoformat(), "status": "ok"},
    )
