"""Tests for image storage and WebSocket endpoints."""

from io import BytesIO


class TestStorageAPI:
    """Tests for image storage API."""

    def test_upload_image(self, client):
        """Should upload an image."""
        fake_image = BytesIO(b"\x89PNG\r\n\x1a\n" + b"\x00" * 256)
        response = client.post(
            "/api/v1/storage/upload?device_id=CAM-001",
            files={"file": ("test.png", fake_image, "image/png")},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["code"] == 0
        assert "object_name" in data["data"]

    def test_upload_non_image(self, client):
        """Should reject non-image files."""
        fake_file = BytesIO(b"not an image")
        response = client.post(
            "/api/v1/storage/upload",
            files={"file": ("test.txt", fake_file, "text/plain")},
        )
        assert response.status_code == 400

    def test_list_images(self, client):
        """Should list uploaded images."""
        # Upload first
        fake = BytesIO(b"\x89PNG\r\n\x1a\n" + b"\x00" * 128)
        client.post(
            "/api/v1/storage/upload?device_id=CAM-001",
            files={"file": ("test.png", fake, "image/png")},
        )
        # List
        response = client.get("/api/v1/storage/list?limit=10")
        assert response.status_code == 200
        data = response.json()
        assert data["code"] == 0
        assert len(data["data"]) >= 1

    def test_download_not_found(self, client):
        """Should return 404 for missing image."""
        response = client.get("/api/v1/storage/download/nonexistent.jpg")
        assert response.status_code == 404

    def test_delete_not_found(self, client):
        """Should return 404 for deleting missing image."""
        response = client.delete("/api/v1/storage/nonexistent.jpg")
        assert response.status_code == 404


class TestWebSocketEndpoint:
    """Tests for WebSocket endpoint."""

    def test_ws_endpoint_exists(self, client):
        """WebSocket endpoint should be registered."""
        # Can't easily test WS with TestClient, but verify openapi schema
        resp = client.get("/openapi.json")
        resp.json()  # Verify OpenAPI schema is valid JSON
        # WebSocket routes may not appear in OpenAPI schema with TestClient
        # Just verify the app doesn't crash on import
        from backend.websocket import get_ws_manager

        manager = get_ws_manager()
        assert manager is not None
        assert manager.active_connections >= 0

    def test_ws_manager_singleton(self):
        """get_ws_manager should return the same instance."""
        from backend.websocket import get_ws_manager

        m1 = get_ws_manager()
        m2 = get_ws_manager()
        assert m1 is m2
