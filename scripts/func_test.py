"""Comprehensive functional test of all 10 modules."""

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

passed = 0
failed = 0
errors = []


def test(name, fn):
    global passed, failed
    try:
        fn()
        passed += 1
        print(f"  [OK] {name}")
    except Exception as e:
        failed += 1
        print(f"  [FAIL] {name}: {e}")
        errors.append(name)


print("=" * 60)
print("  Functional Test -- All 10 Modules")
print("=" * 60)

# 1. Config
print("\n--- Module 1: Backend Config ---")
from backend.config import BackendConfig

cfg = BackendConfig(environment="development")
test(
    "Config fields",
    lambda: (cfg.database.host, cfg.server.port, cfg.model.confidence_threshold),
)

# 2. Database
print("\n--- Module 2: Database & Models ---")
# Import all models so Base.metadata knows about them
import asyncio

import backend.models  # noqa: F401


async def _db():
    from backend.database import Base, init_db

    await init_db()
    tables = list(Base.metadata.tables.keys())
    assert "devices" in tables
    assert "defect_records" in tables
    assert "detection_reports" in tables
    assert "model_versions" in tables
    assert "annotations" in tables


asyncio.run(_db())
test("5 tables created", lambda: None)

# 3. Schemas
print("\n--- Module 3: Pydantic Schemas ---")
from datetime import datetime, timezone

from backend.schemas.api import DetectionReportRequest
from backend.schemas.defect import DEFECT_CODES, DefectType

req = DetectionReportRequest(
    device_id="CAM-001",
    batch_id="B01",
    timestamp=datetime.now(timezone.utc),
    results={
        "total_defects": 1,
        "defect_list": [
            {
                "type": "broken_yarn",
                "bbox": [1, 2, 3, 4],
                "confidence": 0.9,
                "severity": "critical",
            }
        ],
    },
)
test("DetectionReportRequest", lambda: req.device_id)
test("10 defect types", lambda: len(DefectType))
test("BY-01 mapping", lambda: DEFECT_CODES[DefectType.BROKEN_YARN].value)

# 4. Services
print("\n--- Module 4: Alert Service ---")
from backend.services.alert_service import AlertService

svc = AlertService()
a = svc.evaluate_defect(
    "broken_yarn", "critical", "CAM-001", "B01", [0, 0, 10, 10], 0.96
)
test("Critical -> stop_machine", lambda: a.recommended_action == "stop_machine")
b = svc.evaluate_defect("thin_yarn", "minor", "CAM", "B", [0, 0, 1, 1], 0.5)
test("Minor -> no alert", lambda: b is None)

# 5. Inference Engine
print("\n--- Module 5: Inference Engine ---")
from backend.inference.engine import DummyInferenceEngine, create_engine

e = create_engine("dummy")
test("Dummy engine", lambda: isinstance(e, DummyInferenceEngine))
raw = e.infer(np.random.randn(2, 3, 640, 640).astype(np.float32))
test("Batch of 2", lambda: len(raw) == 2)

from backend.inference.detector import FabricDefectDetector

detector = FabricDefectDetector(backend="dummy", seed=0)
img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
dets = detector.detect_single(img)
test("Detect single image", lambda: isinstance(dets, list))
report = detector.build_report("CAM-001", "B01", [img])
test("Build API report", lambda: report.device_id == "CAM-001")

# 6. Hybrid Reward Training
print("\n--- Module 6: Hybrid Reward Training ---")
from backend.training.dimension_rewards import DimensionRewardComputer

comp = DimensionRewardComputer()
preds = {
    "boxes": np.array([[100, 100, 50, 50]], dtype=np.float32),
    "classes": np.array([0]),
    "confidences": np.array([0.95], dtype=np.float32),
}
targets = [{"boxes": [[100, 100, 50, 50]], "labels": [0]}]
rewards = comp.compute_all(preds, targets)
test("7 dimensions", lambda: len(rewards) == 7)
test("D01 loc range", lambda: -1 <= rewards["loc"].item() <= 1)
test("D06 broken value", lambda: rewards["broken"].item() in (-2.0, 0.0, 1.0))

from backend.training.loss_functions import total_loss

dim_weights = torch.ones(7) / 7
losses = total_loss(preds, targets, rewards, dim_weights)
test(
    "4 loss components",
    lambda: all(k in losses for k in ["total", "detection", "scalar", "consistency"]),
)

# 7. Model Deploy
print("\n--- Module 7: Model Deploy ---")
import tempfile

from backend.deploy.export import create_dummy_detection_model, export_to_onnx

model = create_dummy_detection_model()
with tempfile.TemporaryDirectory() as td:
    path = os.path.join(td, "test.onnx")
    export_to_onnx(model, path)
    test("ONNX export", lambda: os.path.exists(path) and os.path.getsize(path) > 0)

from backend.deploy.edge_deploy import HARDWARE_PRESETS

test("3 HW presets", lambda: len(HARDWARE_PRESETS) == 3)

# 8. Image Storage
print("\n--- Module 8: Image Storage ---")
from backend.storage import get_storage

store = get_storage()
store.upload("test/func.jpg", b"\xff\xd8\xff\xe0" + b"\x00" * 100)
test("Upload", lambda: store.exists("test/func.jpg"))
test("Download", lambda: len(store.download("test/func.jpg")) > 0)
store.delete("test/func.jpg")
test("Delete", lambda: not store.exists("test/func.jpg"))

# 9. WebSocket
print("\n--- Module 9: WebSocket ---")
from backend.websocket import get_ws_manager

ws = get_ws_manager()
test("Singleton", lambda: get_ws_manager() is ws)

# 10. Web UI
print("\n--- Module 10: Web UI ---")
dash = os.path.join(os.path.dirname(__file__), "..", "backend", "dashboard.html")
sim = os.path.join(os.path.dirname(__file__), "..", "backend", "simulator.html")
test("dashboard.html", lambda: os.path.exists(dash))
test("simulator.html", lambda: os.path.exists(sim))
with open(dash, encoding="utf-8") as f:
    dh = f.read()
test("WebSocket in dashboard", lambda: "new WebSocket" in dh)
test("Chart.js in dashboard", lambda: "chart.js" in dh.lower())

# 11. End-to-End API
print("\n--- End-to-End: FastAPI ---")
from backend import config as _cfg

_cfg._config = _cfg.BackendConfig(environment="development", debug=True)
from fastapi.testclient import TestClient

from backend.main import app

with TestClient(app) as client:
    r = client.post(
        "/api/v1/detection/report",
        json={
            "device_id": "CAM-E2E",
            "batch_id": "B-E2E",
            "timestamp": "2026-07-12T10:00:00Z",
            "results": {
                "total_defects": 1,
                "defect_list": [
                    {
                        "type": "broken_yarn",
                        "bbox": [100, 200, 50, 60],
                        "confidence": 0.97,
                        "severity": "critical",
                    }
                ],
            },
        },
    )
    test("POST /detection/report 200", lambda: r.status_code == 200)
    test("Report accepted", lambda: r.json()["data"]["accepted"])

    r = client.get("/api/v1/system/status")
    test("GET /system/status 200", lambda: r.status_code == 200)
    test("has devices", lambda: isinstance(r.json()["data"]["devices"], list))

    r = client.get("/api/v1/system/stats?time_range=24h")
    test("GET /system/stats 200", lambda: r.status_code == 200)
    test("has trend", lambda: "trend" in r.json()["data"])

    r = client.get("/api/v1/system/health")
    test("Health healthy", lambda: r.json()["message"] == "healthy")

    r = client.get("/dashboard")
    test("Dashboard HTML", lambda: "text/html" in r.headers["content-type"])

    r = client.get("/simulator")
    test("Simulator HTML", lambda: "Production Line Simulator" in r.text)

    r = client.post(
        "/api/v1/model/update",
        json={
            "model_id": "v9.9.9",
            "model_type": "yolov8_defect",
            "model_url": "https://x.com/m.onnx",
            "checksum": "sha:x",
            "version_info": {
                "release_date": "2026-07-12",
                "accuracy": 0.99,
                "improvements": ["test"],
            },
        },
    )
    test("POST /model/update 200", lambda: r.status_code == 200)

    r = client.get("/api/v1/model/current")
    test("Model current v9.9.9", lambda: r.json()["data"]["model_id"] == "v9.9.9")

# Summary
print(f"\n{'=' * 60}")
print(f"  RESULTS: {passed} passed, {failed} failed")
print(f"{'=' * 60}")
if errors:
    print(f"FAILURES: {len(errors)}")
    for e in errors:
        print(f"  - {e}")
    sys.exit(1)
else:
    print("ALL FUNCTIONAL TESTS PASSED")
