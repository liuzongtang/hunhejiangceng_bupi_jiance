"""
Shared test fixtures for backend tests.

Uses SQLite in-memory database for fast, isolated testing.
Uses synchronous TestClient for simplicity.
"""

import asyncio

from fastapi.testclient import TestClient
import pytest


@pytest.fixture(scope="session")
def event_loop():
    """Create a session-scoped event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(autouse=True)
def clean_database():
    """Remove stale DB and reset engine before each test."""
    import os

    from backend import config as cfg, database as db

    # Reset config and engine for clean state
    cfg._config = None
    db._engine = None
    db._session_factory = None

    import contextlib

    db_path = os.path.join(os.path.dirname(__file__), "..", "..", "fabric_defect.db")
    with contextlib.suppress(FileNotFoundError):
        os.remove(db_path)
    yield
    with contextlib.suppress(FileNotFoundError):
        os.remove(db_path)


@pytest.fixture
def client():
    """Create a synchronous FastAPI test client."""
    from backend import config as cfg

    cfg._config = cfg.BackendConfig(
        environment="development",  # Uses SQLite
        debug=True,
    )
    from backend.main import app

    with TestClient(app) as tc:
        yield tc


@pytest.fixture
def sample_detection_request():
    """Sample detection report matching Section 3.2.1."""
    return {
        "device_id": "CAM-001",
        "batch_id": "B20260712-001",
        "timestamp": "2026-07-12T10:30:15.678Z",
        "results": {
            "total_defects": 2,
            "defect_list": [
                {
                    "type": "broken_yarn",
                    "bbox": [320, 450, 80, 120],
                    "confidence": 0.97,
                    "severity": "critical",
                },
                {
                    "type": "missing_stitch",
                    "bbox": [1250, 780, 45, 55],
                    "confidence": 0.89,
                    "severity": "major",
                },
            ],
        },
    }


@pytest.fixture
def sample_model_request():
    """Sample model update request matching Section 3.2.2."""
    return {
        "model_id": "v1.2.3",
        "model_type": "yolov8_defect",
        "model_url": "https://cdn.example.com/models/v1.2.3.onnx",
        "checksum": "sha256:abc123def456",
        "version_info": {
            "release_date": "2026-07-12",
            "accuracy": 0.962,
            "improvements": ["improved dark_fabric detection"],
        },
    }
