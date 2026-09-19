"""Tests for the FabricDefectDetector."""

import numpy as np
import pytest

from backend.inference.detector import FabricDefectDetector


class TestFabricDefectDetector:
    """Tests for the high-level detector."""

    @pytest.fixture
    def detector(self):
        return FabricDefectDetector(backend="dummy", seed=42)

    def test_create_detector(self, detector):
        """Should create with dummy backend."""
        assert detector.backend == "dummy"
        assert detector.engine is not None

    def test_detect_single_image(self, detector):
        """Should detect defects in a single image."""
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        results = detector.detect([image])
        assert isinstance(results, list)
        assert len(results) == 1
        # Each detection should have required keys
        for det in results[0]:
            assert "defect_id" in det
            assert "type" in det
            assert "type_code" in det
            assert "severity" in det
            assert "bbox" in det
            assert "confidence" in det

    def test_detect_batch(self, detector):
        """Should handle batch of images."""
        images = [
            np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8) for _ in range(4)
        ]
        results = detector.detect(images)
        assert len(results) == 4

    def test_detect_empty(self, detector):
        """Empty image list should return empty."""
        results = detector.detect([])
        assert results == []

    def test_detect_single_convenience(self, detector):
        """detect_single should return a flat list."""
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        results = detector.detect_single(image)
        assert isinstance(results, list)

    def test_build_report(self, detector):
        """build_report should return DetectionReportRequest."""
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        report = detector.build_report(
            device_id="CAM-001",
            batch_id="BATCH-TEST",
            images=[image],
        )
        assert report.device_id == "CAM-001"
        assert report.batch_id == "BATCH-TEST"
        assert "total_defects" in report.results
        assert "defect_list" in report.results

    def test_detections_have_valid_range(self, detector):
        """Detection properties should be in valid ranges."""
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        results = detector.detect_single(image)
        for det in results:
            assert 0 <= det["confidence"] <= 1.0
            bbox = det["bbox"]
            assert len(bbox) == 4
            assert bbox[2] > 0  # width > 0
            assert bbox[3] > 0  # height > 0
            assert det["severity"] in ("critical", "major", "medium", "minor", "info")
            assert det["type_code"].startswith(
                (
                    "HO",
                    "ST",
                    "SL",
                    "KN",
                    "FL",
                    "HF",
                    "HP",
                    "CW",
                    "LW",
                    "BW",
                    "HW",
                    "CF",
                    "WS",
                    "SK",
                    "BS",
                    "DS",
                    "SM",
                    "WD",
                )
            )
            assert "-" in det["type_code"]

    def test_benchmark(self, detector):
        """Benchmark should not crash."""
        metrics = detector.benchmark(batch_size=1, iterations=5)
        assert "avg_latency_ms" in metrics
        assert metrics["backend"] == "dummy"
