"""
Detection report service — handles POST /api/v1/detection/report business logic.

Receives detection results from edge devices, stores them in the database,
and triggers alerts for critical defects.
"""

from datetime import datetime, timezone
from typing import Dict
import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.models.defect import DefectRecord, DefectType, Severity
from backend.models.detection_report import DetectionReport
from backend.models.device import Device
from backend.schemas.api import DetectionReportRequest, DetectionReportResponseData


class DetectionService:
    """Service for processing detection reports from edge devices."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def process_report(
        self, request: DetectionReportRequest
    ) -> DetectionReportResponseData:
        """
        Process an incoming detection report.

        1. Validate device exists (or auto-register)
        2. Store individual defect records
        3. Create a detection report summary
        4. Update device counters
        """
        report_id = f"RPT-{datetime.now(timezone.utc).strftime('%Y%m%d')}-{uuid.uuid4().hex[:6].upper()}"

        results = request.results
        defect_list = results.get("defect_list", [])
        total_defects = results.get("total_defects", len(defect_list))

        # Aggregate by type and severity
        by_type: Dict[str, int] = {}
        by_severity: Dict[str, int] = {}
        alarm_triggered = False
        alarm_type = ""
        alarm_action = "log_only"

        # Store individual defect records
        for item in defect_list:
            defect_id = f"D-{uuid.uuid4().hex[:8].upper()}"
            defect_type = item.get("type", "other")
            severity = item.get("severity", "minor")
            bbox = item.get("bbox", [0, 0, 0, 0])
            confidence = item.get("confidence", 0.0)

            # Update aggregates
            by_type[defect_type] = by_type.get(defect_type, 0) + 1
            by_severity[severity] = by_severity.get(severity, 0) + 1

            # Check for critical defects that need alarm
            if severity in ("critical", "major"):
                alarm_triggered = True
                if severity == "critical":
                    alarm_type = "critical_defect"
                    alarm_action = "stop_machine"
                elif alarm_type == "":
                    alarm_type = "major_defect"
                    alarm_action = "mark_roll"

            # Create defect record
            record = DefectRecord(
                defect_id=defect_id,
                type=DefectType(defect_type)
                if defect_type in DefectType.__members__
                else DefectType.OTHER,
                type_code=self._get_defect_code(defect_type),
                severity=Severity(severity)
                if severity in Severity.__members__
                else Severity.INFO,
                bbox={"x": bbox[0], "y": bbox[1], "w": bbox[2], "h": bbox[3]},
                confidence=confidence,
                device_id=request.device_id,
                batch_id=request.batch_id,
                frame_seq=0,
                created_at=request.timestamp,
            )
            self.session.add(record)

        # Create detection report
        report = DetectionReport(
            report_id=report_id,
            device_id=request.device_id,
            batch_id=request.batch_id,
            total_defects=total_defects,
            by_type=by_type,
            by_severity=by_severity,
            alarm_triggered=alarm_triggered,
            alarm_type=alarm_type,
            alarm_action=alarm_action,
            timestamp=request.timestamp,
        )
        self.session.add(report)

        # Update device counters
        device = await self._get_or_create_device(request.device_id)
        device.total_detections += 1
        device.defect_count += total_defects
        device.last_heartbeat = datetime.now(timezone.utc)
        device.status = "online"

        await self.session.commit()

        return DetectionReportResponseData(report_id=report_id, accepted=True)

    async def _get_or_create_device(self, device_id: str) -> Device:
        """Get an existing device or auto-register a new one."""
        result = await self.session.execute(
            select(Device).where(Device.device_id == device_id)
        )
        device = result.scalar_one_or_none()
        if device is None:
            device = Device(
                device_id=device_id,
                status="online",
                total_detections=0,
                defect_count=0,
                first_seen=datetime.now(timezone.utc),
            )
            self.session.add(device)
        return device

    @staticmethod
    def _get_defect_code(defect_type: str) -> str:
        """Map defect type name to standard code using the canonical mapping."""
        from backend.schemas.defect import DEFECT_CODES, DefectType

        try:
            dt = DefectType(defect_type)
            return DEFECT_CODES[dt].value
        except (ValueError, KeyError):
            return "OT-01"
