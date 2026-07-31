"""
Tests for model update API (Section 3.2.2).
"""


class TestModelUpdate:
    """POST /api/v1/model/update"""

    def test_register_model(self, client, sample_model_request):
        """Should register a new model version."""
        response = client.post(
            "/api/v1/model/update",
            json=sample_model_request,
        )
        assert response.status_code == 200
        data = response.json()
        assert data["code"] == 0
        assert data["message"] == "model update accepted"
        assert "update_id" in data["data"]
        assert data["data"]["estimated_deploy_time"] == 120

    def test_get_current_model(self, client, sample_model_request):
        """Should return the current active model."""
        client.post("/api/v1/model/update", json=sample_model_request)
        response = client.get("/api/v1/model/current")
        assert response.status_code == 200
        data = response.json()
        assert data["code"] == 0
        if data["data"]:
            assert data["data"]["model_id"] == sample_model_request["model_id"]

    def test_get_model_history(self, client, sample_model_request):
        """Should return model version history."""
        client.post("/api/v1/model/update", json=sample_model_request)
        response = client.get("/api/v1/model/history")
        assert response.status_code == 200
        data = response.json()
        assert data["code"] == 0
        assert len(data["data"]) >= 1

    def test_register_model_missing_checksum(self, client):
        """Should reject model update without checksum."""
        response = client.post(
            "/api/v1/model/update",
            json={
                "model_id": "v1.0.0",
                "model_type": "yolov8_defect",
                "model_url": "https://example.com/model.onnx",
                "version_info": {
                    "release_date": "2026-07-12",
                    "accuracy": 0.95,
                    "improvements": [],
                },
            },
        )
        assert response.status_code == 422  # Missing required field
