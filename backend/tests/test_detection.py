"""
Tests for detection report API (Section 3.2.1).
"""


class TestDetectionReport:
    """POST /api/v1/detection/report"""

    def test_submit_report(self, client, sample_detection_request):
        """Should accept a valid detection report."""
        response = client.post(
            "/api/v1/detection/report",
            json=sample_detection_request,
        )
        assert response.status_code == 200
        data = response.json()
        assert data["code"] == 0
        assert data["message"] == "success"
        assert data["data"]["accepted"] is True
        assert "report_id" in data["data"]

    def test_submit_report_missing_required_field(self, client):
        """Should reject a report missing device_id."""
        response = client.post(
            "/api/v1/detection/report",
            json={"batch_id": "test"},
        )
        assert response.status_code == 422  # Validation error

    def test_submit_report_empty_defects(self, client):
        """Should accept a report with no defects."""
        response = client.post(
            "/api/v1/detection/report",
            json={
                "device_id": "CAM-002",
                "batch_id": "B20260712-002",
                "timestamp": "2026-07-12T10:30:15Z",
                "results": {"total_defects": 0, "defect_list": []},
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["data"]["accepted"] is True

    def test_list_reports(self, client, sample_detection_request):
        """Should list recent detection reports."""
        client.post("/api/v1/detection/report", json=sample_detection_request)
        response = client.get("/api/v1/detection/reports")
        assert response.status_code == 200
        data = response.json()
        assert data["code"] == 0
        assert len(data["data"]) >= 1
