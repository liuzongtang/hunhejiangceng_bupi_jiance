"""Tests for detection postprocessing."""

import numpy as np
import pytest

from backend.inference.postprocessing import DetectionPostprocessor


class TestNMS:
    """Tests for Non-Maximum Suppression."""

    @pytest.fixture
    def pp(self):
        return DetectionPostprocessor(confidence_threshold=0.5, nms_threshold=0.5)

    def test_nms_single_box(self, pp):
        """Single box should survive NMS."""
        boxes = np.array([[100, 100, 50, 50]], dtype=np.float32)
        scores = np.array([0.9], dtype=np.float32)
        keep = pp._nms(boxes, scores)
        assert len(keep) == 1

    def test_nms_identical_boxes(self, pp):
        """Identical boxes: lower score should be suppressed."""
        boxes = np.array(
            [
                [100, 100, 50, 50],
                [100, 100, 50, 50],
            ],
            dtype=np.float32,
        )
        scores = np.array([0.9, 0.6], dtype=np.float32)
        keep = pp._nms(boxes, scores)
        assert len(keep) == 1
        assert keep[0] == 0  # Higher score survives

    def test_nms_non_overlapping(self, pp):
        """Non-overlapping boxes should all survive."""
        boxes = np.array(
            [
                [10, 10, 50, 50],
                [200, 200, 50, 50],
                [400, 400, 50, 50],
            ],
            dtype=np.float32,
        )
        scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)
        keep = pp._nms(boxes, scores)
        assert len(keep) == 3

    def test_nms_empty(self, pp):
        """Empty input should return empty."""
        boxes = np.zeros((0, 4), dtype=np.float32)
        scores = np.zeros(0, dtype=np.float32)
        keep = pp._nms(boxes, scores)
        assert len(keep) == 0


class TestPostprocessing:
    """Tests for full postprocessing pipeline."""

    @pytest.fixture
    def pp(self):
        return DetectionPostprocessor(confidence_threshold=0.5, nms_threshold=0.5)

    def test_confidence_filter(self, pp):
        """Low confidence detections should be filtered."""
        raw = [
            {
                "boxes": np.array(
                    [[10, 10, 50, 50], [200, 200, 50, 50]], dtype=np.float32
                ),
                "scores": np.array([0.9, 0.3], dtype=np.float32),  # 0.3 below threshold
                "classes": np.array([0, 1], dtype=np.int64),
                "class_names": ["broken_warp", "stain"],
            }
        ]
        results = pp.process(raw)
        assert len(results[0]) == 1
        assert results[0][0]["type"] == "broken_warp"

    def test_empty_detections(self, pp):
        """Empty raw output should return empty list."""
        raw = [
            {
                "boxes": np.zeros((0, 4)),
                "scores": np.zeros(0),
                "classes": np.zeros(0, dtype=np.int64),
                "class_names": [],
            }
        ]
        results = pp.process(raw)
        assert results[0] == []

    def test_detection_format(self, pp):
        """Each detection should have required fields."""
        raw = [
            {
                "boxes": np.array([[100, 200, 80, 60]], dtype=np.float32),
                "scores": np.array([0.95], dtype=np.float32),
                "classes": np.array([0], dtype=np.int64),
                "class_names": ["broken_warp"],
            }
        ]
        results = pp.process(raw)
        det = results[0][0]
        for key in ["defect_id", "type", "type_code", "severity", "bbox", "confidence"]:
            assert key in det, f"Missing key: {key}"
        assert det["type_code"] == "BW-01"
        assert det["severity"] == "critical"
        assert det["confidence"] == pytest.approx(0.95, abs=0.01)

    def test_defect_code_mapping(self, pp):
        """All 18 defect types should map correctly."""
        expected = {
            "hole": "HO-01",
            "stain": "ST-01",
            "three_silk": "SL-01",
            "knot": "KN-01",
            "flower_board": "FL-01",
            "hundred_feet": "HF-01",
            "hair_particle": "HP-01",
            "coarse_warp": "CW-01",
            "loose_warp": "LW-01",
            "broken_warp": "BW-01",
            "hanging_warp": "HW-01",
            "weft_shrink": "WS-01",
            "size_stain": "ST-02",
            "warping_knot": "KN-02",
            "star_skip": "SK-01",
            "broken_spandex": "BS-01",
            "dense_section": "DS-01",
            "weave_defect": "WD-01",
        }
        for name, code in expected.items():
            assert pp._defect_code(name) == code
