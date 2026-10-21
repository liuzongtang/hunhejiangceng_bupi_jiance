"""
Tests for the label feedback API (#16 relabel loop).

POST /api/v1/feedback, GET /api/v1/feedback, POST /api/v1/feedback/apply.
"""

from backend.services.label_feedback_service import (
    apply_feedback,
    boxes_to_yolo,
    parse_yolo,
)


def _feedback(image_id="img001", **overrides):
    payload = {
        "image_id": image_id,
        "feedback_type": "missed_defect",
        "class_id": 15,
        "bbox": {"cx": 0.5, "cy": 0.5, "w": 0.03, "h": 0.03},
    }
    payload.update(overrides)
    return payload


class TestLabelFeedback:
    """POST /api/v1/feedback"""

    def test_submit_feedback(self, client):
        response = client.post("/api/v1/feedback", json=_feedback())
        assert response.status_code == 200
        data = response.json()
        assert data["code"] == 0
        assert data["data"]["status"] == "pending"
        assert data["data"]["class_name"] == "broken_spandex"
        assert data["data"]["feedback_id"].startswith("FB-")

    def test_submit_invalid_feedback_type(self, client):
        response = client.post(
            "/api/v1/feedback", json=_feedback(feedback_type="bogus")
        )
        assert response.status_code == 422

    def test_submit_out_of_range_class(self, client):
        response = client.post("/api/v1/feedback", json=_feedback(class_id=99))
        assert response.status_code == 422

    def test_misclassification_requires_old_class(self, client):
        response = client.post(
            "/api/v1/feedback", json=_feedback(feedback_type="misclassification")
        )
        assert response.status_code == 422

    def test_list_feedback(self, client):
        client.post("/api/v1/feedback", json=_feedback())
        response = client.get("/api/v1/feedback")
        assert response.status_code == 200
        data = response.json()
        assert data["code"] == 0
        assert len(data["data"]) >= 1


class TestApplyFeedback:
    """POST /api/v1/feedback/apply"""

    def test_apply_unknown_image_is_noop(self, client):
        # A feedback whose image_id maps to no real label file must not touch
        # the dataset — it is reported as no_label and marked applied.
        client.post(
            "/api/v1/feedback", json=_feedback(image_id="definitely_not_a_real_image")
        )
        response = client.post("/api/v1/feedback/apply")
        assert response.status_code == 200
        data = response.json()
        assert data["code"] == 0
        assert data["data"]["no_label"] == 1
        assert data["data"]["applied"] == 0


class TestPureCorrection:
    """Unit tests for the pure feedback -> label correction logic."""

    def test_missed_defect_adds_box(self):
        boxes = parse_yolo(["9 0.1 0.1 0.2 0.2"])
        action, _detail, boxes = apply_feedback(
            boxes,
            {
                "type": "missed_defect",
                "class_id": 15,
                "cx": 0.5,
                "cy": 0.5,
                "w": 0.03,
                "h": 0.03,
            },
        )
        assert action == "added"
        assert boxes[-1][0] == 15

    def test_misclassification_changes_class(self):
        boxes = parse_yolo(["15 0.5 0.5 0.03 0.03"])
        action, _detail, boxes = apply_feedback(
            boxes,
            {
                "type": "misclassification",
                "old_class_id": 15,
                "class_id": 9,
                "cx": 0.5,
                "cy": 0.5,
                "w": 0.03,
                "h": 0.03,
            },
        )
        assert action == "changed"
        assert boxes[0][0] == 9

    def test_false_positive_removes_box(self):
        boxes = parse_yolo(["15 0.5 0.5 0.03 0.03", "4 0.1 0.1 0.2 0.2"])
        action, _detail, boxes = apply_feedback(
            boxes,
            {
                "type": "false_positive",
                "class_id": 15,
                "cx": 0.5,
                "cy": 0.5,
                "w": 0.03,
                "h": 0.03,
            },
        )
        assert action == "removed"
        assert all(b[0] != 15 for b in boxes)

    def test_roundtrip_serialization(self):
        boxes = parse_yolo(["15 0.5 0.5 0.03 0.03"])
        assert boxes_to_yolo(boxes).strip() == "15 0.500000 0.500000 0.030000 0.030000"
