"""
Tests for the alert-severity logic.

After the 2026-09-23 class merge (surface_mark -> broken_warp, coarse_weft ->
weft_shrink) the only minor-lookalike-of-critical case is gone, so
``CONFUSABLE_HIGHER_IMPACT`` is empty and ``effective_alert_severity`` is a
canonical pass-through. These tests pin that behaviour: alerting follows each
class's own ``DEFECT_SEVERITY``, and ``escalated_severity`` is never populated.
"""

from __future__ import annotations

import os
import sqlite3

from backend.inference.class_mapping import project_alert_severity
from backend.schemas.defect import (
    DefectType,
    Severity,
    effective_alert_severity,
)


def _broken_warp_request():
    """A detection report containing a single broken_warp (critical) defect."""
    return {
        "device_id": "CAM-BW",
        "batch_id": "B-BW-001",
        "timestamp": "2026-07-12T10:30:15.678Z",
        "results": {
            "total_defects": 1,
            "defect_list": [
                {
                    "type": "broken_warp",
                    "bbox": [320, 450, 80, 120],
                    "confidence": 0.85,
                    "severity": "critical",
                },
            ],
        },
    }


class TestEffectiveAlertSeverity:
    """Unit tests for the (now pass-through) escalation function."""

    def test_broken_warp_is_critical(self):
        assert effective_alert_severity(DefectType.BROKEN_WARP) == Severity.CRITICAL

    def test_weave_defect_is_major(self):
        assert effective_alert_severity(DefectType.WEAVE_DEFECT) == Severity.MAJOR

    def test_stain_is_medium(self):
        assert effective_alert_severity(DefectType.STAIN) == Severity.MEDIUM

    def test_knot_is_minor(self):
        assert effective_alert_severity(DefectType.KNOT) == Severity.MINOR


class TestProjectAlertSeverity:
    """Unit tests for the inference-layer bridge."""

    def test_broken_warp_returns_critical(self):
        assert project_alert_severity("broken_warp") == "critical"

    def test_weave_defect_returns_major(self):
        assert project_alert_severity("weave_defect") == "major"

    def test_unknown_type_returns_info(self):
        assert project_alert_severity("not_a_real_type") == "info"


class TestAlertSeverityAPI:
    """Service-level tests via the detection report endpoint."""

    def test_critical_defect_uses_critical_defect_alarm(self, client):
        """broken_warp (critical) should trigger a stop-machine alarm."""
        response = client.post("/api/v1/detection/report", json=_broken_warp_request())
        assert response.status_code == 200

        reports = client.get("/api/v1/detection/reports").json()["data"]
        assert len(reports) == 1
        alarm = reports[0]["alarm"]
        assert alarm["triggered"] is True
        assert alarm["alarm_type"] == "critical_defect"
        assert alarm["action"] == "stop_machine"

    def test_no_escalated_severity_stored(self, client):
        """With the lookalike map empty, escalated_severity stays NULL."""
        client.post("/api/v1/detection/report", json=_broken_warp_request())

        # Query the SQLite DB directly (dev test DB lives at repo root).
        db_path = os.path.join(
            os.path.dirname(__file__), "..", "..", "fabric_defect.db"
        )
        conn = sqlite3.connect(db_path)
        try:
            row = conn.execute(
                "SELECT severity, escalated_severity FROM defect_records"
            ).fetchone()
        finally:
            conn.close()

        assert row is not None
        assert row[0] == "critical"
        assert row[1] is None
