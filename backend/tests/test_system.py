"""
Tests for system status API (Section 3.2.3).
"""


class TestSystemStatus:
    """GET /api/v1/system/status"""

    def test_get_system_status(self, client):
        """Should return system status with devices, model, and metrics."""
        response = client.get("/api/v1/system/status")
        assert response.status_code == 200
        data = response.json()
        assert data["code"] == 0
        assert "devices" in data["data"]
        assert "model" in data["data"]
        assert "system" in data["data"]

    def test_system_status_model_info(self, client):
        """Should return current model version in status."""
        response = client.get("/api/v1/system/status")
        data = response.json()
        assert data["data"]["model"]["current_version"] is not None

    def test_system_status_metrics(self, client):
        """Should return CPU and memory usage."""
        response = client.get("/api/v1/system/status")
        data = response.json()
        metrics = data["data"]["system"]
        assert "cpu_usage" in metrics
        assert "memory_usage" in metrics
        assert 0.0 <= metrics["cpu_usage"] <= 100.0
        assert 0.0 <= metrics["memory_usage"] <= 100.0


class TestHealthCheck:
    """GET /api/v1/system/health"""

    def test_health_check(self, client):
        """Health check should return 200."""
        response = client.get("/api/v1/system/health")
        assert response.status_code == 200
        data = response.json()
        assert data["code"] == 0
        assert data["message"] == "healthy"


class TestOpenAPI:
    """API documentation endpoints."""

    def test_docs_available(self, client):
        """OpenAPI docs should be accessible."""
        response = client.get("/docs")
        assert response.status_code == 200

    def test_openapi_json(self, client):
        """OpenAPI JSON schema should be accessible."""
        response = client.get("/openapi.json")
        assert response.status_code == 200
        schema = response.json()
        assert "paths" in schema
        paths = schema["paths"]
        assert "/api/v1/detection/report" in paths
        assert "/api/v1/model/update" in paths
        assert "/api/v1/system/status" in paths


class TestDashboard:
    """Dashboard endpoint."""

    def test_dashboard_loads(self, client):
        """Dashboard HTML should load at /dashboard."""
        response = client.get("/dashboard")
        assert response.status_code == 200
        assert "text/html" in response.headers.get("content-type", "")
        assert "织物瑕疵" in response.text

    def test_dashboard_has_tabs(self, client):
        """Dashboard should contain all 4 monitoring tabs."""
        response = client.get("/dashboard")
        html = response.text
        assert "总览" in html
        assert "检测" in html
        assert "告警" in html
        assert "模型" in html

    def test_dashboard_has_charts(self, client):
        """Dashboard should include Chart.js for visualization."""
        response = client.get("/dashboard")
        html = response.text
        assert "chart.min.js" in html.lower()
        assert "trendChart" in html
        assert "typeChart" in html

    def test_simulator_loads(self, client):
        """Simulator page should load."""
        response = client.get("/simulator")
        assert response.status_code == 200
        assert "产线模拟器" in response.text
        assert "fabricCanvas" in response.text


class TestSystemStats:
    """GET /api/v1/system/stats"""

    def test_stats_endpoint(self, client):
        """Stats endpoint should return aggregated data."""
        response = client.get("/api/v1/system/stats?range=24h")
        assert response.status_code == 200
        data = response.json()
        assert data["code"] == 0
        assert "total_defects" in data["data"]
        assert "by_type" in data["data"]
        assert "by_severity" in data["data"]
        assert "trend" in data["data"]

    def test_stats_different_ranges(self, client):
        """Stats should accept different time ranges."""
        for r in ["1h", "24h", "7d", "30d"]:
            response = client.get(f"/api/v1/system/stats?range={r}")
            assert response.status_code == 200

    def test_stats_trend_format(self, client):
        """Trend data should have labels and data arrays of equal length."""
        response = client.get("/api/v1/system/stats?range=24h")
        trend = response.json()["data"]["trend"]
        assert len(trend["labels"]) == len(trend["data"])
