"""
Tests for the alert-severity escalation logic.

When a predicted class is visually confusable with a higher-impact class
(e.g. surface_mark ~ broken_warp), alerting escalates to the higher severity so
a critical defect mislabelled as a minor lookalike is not silently downgraded.
The stored severity stays canonical; the escalation is recorded separately.
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


def _surface_mark_request():
    """A detection report containing a single surface_mark (minor) defect."""
    return {
        "device_id": "CAM-SM",
        "batch_id": "B-SM-001",
        "timestamp": "2026-07-12T10:30:15.678Z",
        "results": {
            "total_defects": 1,
            "defect_list": [
                {
                    "type": "surface_mark",
                    "bbox": [320, 450, 80, 120],
                    "confidence": 0.85,
                    "severity": "minor",
                },
            ],
        },
    }


class TestEffectiveAlertSeverity:
    """Unit tests for the pure escalation function."""

    def test_surface_mark_escalates_to_critical(self):
        """surface_mark (minor) confusable with broken_warp (critical)."""
        assert effective_alert_severity(DefectType.SURFACE_MARK) == Severity.CRITICAL

    def test_direct_critical_not_escalated(self):
        """broken_warp is already critical; escalation is a no-op."""
        assert effective_alert_severity(DefectType.BROKEN_WARP) == Severity.CRITICAL

    def test_weave_defect_is_major(self):
        """weave_defect maps to major and is not confusable upward."""
        assert effective_alert_severity(DefectType.WEAVE_DEFECT) == Severity.MAJOR

    def test_plain_minor_not_escalated(self):
        """knot has no confusable higher-impact class."""
        assert effective_alert_severity(DefectType.KNOT) == Severity.MINOR

    def test_stain_not_escalated(self):
        """stain (medium) has no confusable higher-impact class."""
        assert effective_alert_severity(DefectType.STAIN) == Severity.MEDIUM


class TestProjectAlertSeverity:
    """Unit tests for the inference-layer bridge."""

    def test_surface_mark_returns_critical(self):
        assert project_alert_severity("surface_mark") == "critical"

    def test_broken_warp_returns_critical(self):
        assert project_alert_severity("broken_warp") == "critical"

    def test_unknown_type_returns_info(self):
        assert project_alert_severity("not_a_real_type") == "info"


class TestAlertEscalationAPI:
    """Service-level tests via the detection report endpoint."""

    def test_surface_mark_triggers_suspected_critical_alarm(self, client):
        """A minor surface_mark should escalate to a stop-machine alarm."""
        response = client.post(
            "/api/v1/detection/report",
            json=_surface_mark_request(),
        )
        assert response.status_code == 200

        reports = client.get("/api/v1/detection/reports").json()["data"]
        assert len(reports) == 1
        alarm = reports[0]["alarm"]
        assert alarm["triggered"] is True
        assert alarm["alarm_type"] == "suspected_critical"
        assert alarm["action"] == "stop_machine"

    def test_direct_critical_uses_critical_defect_alarm(self, client):
        """broken_warp should keep the plain critical_defect alarm type."""
        request = _surface_mark_request()
        request["device_id"] = "CAM-BW"
        request["results"]["defect_list"][0]["type"] = "broken_warp"
        request["results"]["defect_list"][0]["severity"] = "critical"

        client.post("/api/v1/detection/report", json=request)

        reports = client.get("/api/v1/detection/reports").json()["data"]
        alarm = reports[0]["alarm"]
        assert alarm["triggered"] is True
        assert alarm["alarm_type"] == "critical_defect"
        assert alarm["action"] == "stop_machine"

    def test_surface_mark_stores_escalated_severity(self, client):
        """The stored severity stays minor; escalation is recorded separately."""
        client.post("/api/v1/detection/report", json=_surface_mark_request())

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
        assert row[0] == "minor"
        assert row[1] == "critical"
