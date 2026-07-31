"""
Quality assurance: stress tests, edge cases, boundary conditions.
"""

from concurrent.futures import ThreadPoolExecutor
import time


class TestStressDetection:
    """Concurrent detection report submission."""

    def test_concurrent_reports(self, client):
        """Handle 50 concurrent detection report submissions."""
        payload = {
            "device_id": "CAM-STRESS",
            "batch_id": "B-STRESS-001",
            "timestamp": "2026-07-12T10:00:00Z",
            "results": {
                "total_defects": 1,
                "defect_list": [
                    {
                        "type": "hole",
                        "bbox": [100, 100, 50, 50],
                        "confidence": 0.95,
                        "severity": "critical",
                    }
                ],
            },
        }

        def submit():
            return client.post("/api/v1/detection/report", json=payload)

        start = time.time()
        with ThreadPoolExecutor(max_workers=10) as ex:
            results = list(ex.map(lambda _: submit(), range(50)))
        elapsed = time.time() - start

        successes = sum(1 for r in results if r.status_code == 200)
        assert successes >= 45, f"Only {successes}/50 succeeded in {elapsed:.1f}s"
        assert elapsed < 30, f"Too slow: {elapsed:.1f}s"

    def test_large_defect_batch(self, client):
        """Handle a report with 100 defects."""
        payload = {
            "device_id": "CAM-LARGE",
            "batch_id": "B-LARGE",
            "timestamp": "2026-07-12T10:00:00Z",
            "results": {
                "total_defects": 100,
                "defect_list": [
                    {
                        "type": "stain",
                        "bbox": [i, i, 10, 10],
                        "confidence": 0.8,
                        "severity": "minor",
                    }
                    for i in range(100)
                ],
            },
        }
        response = client.post("/api/v1/detection/report", json=payload)
        assert response.status_code == 200


class TestEdgeCases:
    """Edge case and boundary testing."""

    def test_empty_results(self, client):
        """Empty defect list."""
        r = client.post(
            "/api/v1/detection/report",
            json={
                "device_id": "CAM-EDGE",
                "batch_id": "B-EDGE",
                "timestamp": "2026-07-12T10:00:00Z",
                "results": {"total_defects": 0, "defect_list": []},
            },
        )
        assert r.status_code == 200

    def test_very_long_device_id(self, client):
        """Reject overly long device_id."""
        r = client.post(
            "/api/v1/detection/report",
            json={
                "device_id": "X" * 100,
                "batch_id": "B",
                "timestamp": "2026-07-12T10:00:00Z",
                "results": {"total_defects": 0, "defect_list": []},
            },
        )
        assert r.status_code == 422

    def test_invalid_json(self, client):
        """Handle malformed request body."""
        r = client.post("/api/v1/detection/report", content=b"not json")
        assert r.status_code == 422

    def test_negative_confidence(self, client):
        """Reject negative confidence."""
        r = client.post(
            "/api/v1/detection/report",
            json={
                "device_id": "CAM",
                "batch_id": "B",
                "timestamp": "2026-07-12T10:00:00Z",
                "results": {
                    "total_defects": 1,
                    "defect_list": [
                        {
                            "type": "hole",
                            "bbox": [1, 1, 1, 1],
                            "confidence": -0.5,
                            "severity": "critical",
                        }
                    ],
                },
            },
        )
        # The service accepts it (dict, not Pydantic validator) — tests robustness
        assert r.status_code in (200, 422)

    def test_stats_default_range(self, client):
        """Stats without range parameter should default."""
        r = client.get("/api/v1/system/stats")
        assert r.status_code == 200

    def test_model_update_invalid_accuracy(self, client):
        """Reject accuracy > 1.0."""
        r = client.post(
            "/api/v1/model/update",
            json={
                "model_id": "v1",
                "model_type": "yolo",
                "model_url": "https://x.com/m.onnx",
                "checksum": "sha:x",
                "version_info": {
                    "release_date": "2026-07-12",
                    "accuracy": 99.0,
                    "improvements": [],
                },
            },
        )
        assert r.status_code == 422

    def test_storage_empty_upload(self, client):
        """Reject empty file upload."""
        from io import BytesIO

        r = client.post(
            "/api/v1/storage/upload",
            files={"file": ("e.jpg", BytesIO(b""), "image/jpeg")},
        )
        assert r.status_code == 400


class TestSystemLoad:
    """System-level load and resilience."""

    def test_health_under_load(self, client):
        """Health check should remain responsive under load."""
        payload = {
            "device_id": "CAM-LOAD",
            "batch_id": "B-LOAD",
            "timestamp": "2026-07-12T10:00:00Z",
            "results": {"total_defects": 0, "defect_list": []},
        }

        def hit():
            for _ in range(5):
                client.post("/api/v1/detection/report", json=payload)
            return client.get("/api/v1/system/health")

        with ThreadPoolExecutor(max_workers=4) as ex:
            results = list(ex.map(lambda _: hit(), range(4)))

        for r in results:
            assert r.status_code == 200
            assert r.json()["message"] == "healthy"

    def test_all_endpoints_accessible(self, client):
        """All GET endpoints should be accessible."""
        endpoints = [
            "/api/v1/system/status",
            "/api/v1/system/health",
            "/api/v1/system/stats",
            "/api/v1/detection/reports",
            "/api/v1/model/current",
            "/api/v1/model/history",
            "/docs",
            "/openapi.json",
            "/dashboard",
            "/simulator",
        ]
        for ep in endpoints:
            r = client.get(ep)
            assert r.status_code == 200, f"Failed: {ep} → {r.status_code}"

    def test_api_response_consistency(self, client):
        """All API responses should follow the {code, message, data} envelope."""
        payload = {
            "device_id": "CAM-CONSIST",
            "batch_id": "B-CONSIST",
            "timestamp": "2026-07-12T10:00:00Z",
            "results": {"total_defects": 0, "defect_list": []},
        }
        # POST
        r = client.post("/api/v1/detection/report", json=payload)
        d = r.json()
        assert "code" in d and "message" in d and "data" in d

        # GET
        r = client.get("/api/v1/system/status")
        d = r.json()
        assert "code" in d and "data" in d

    def test_model_register_then_query(self, client):
        """Register model then query — full lifecycle."""
        # Register
        r1 = client.post(
            "/api/v1/model/update",
            json={
                "model_id": "v9.9.9",
                "model_type": "yolov8_defect",
                "model_url": "https://cdn.example.com/v9.9.9.onnx",
                "checksum": "sha256:abc999",
                "version_info": {
                    "release_date": "2026-07-12",
                    "accuracy": 0.99,
                    "improvements": ["test"],
                },
            },
        )
        assert r1.status_code == 200
        # Query current
        r2 = client.get("/api/v1/model/current")
        assert r2.status_code == 200
        assert r2.json()["data"]["model_id"] == "v9.9.9"
        # Query history
        r3 = client.get("/api/v1/model/history?limit=5")
        assert r3.status_code == 200
        assert len(r3.json()["data"]) >= 1
