"""
Label feedback service — the server half of the #16 relabel loop.

Captures operator corrections (false positive / missed defect /
misclassification) as ``LabelFeedback`` rows, and applies pending corrections
back to the YOLO label files so they feed the next incremental fine-tune.

The pure correction logic (``iou`` / ``apply_feedback`` / ``parse_yolo``) lives
here as the single source of truth; ``scripts/relabel_loop.py`` imports it for
its standalone demo.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple
import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.models.label_feedback import FeedbackType, LabelFeedback
from backend.schemas.api import LabelFeedbackRequest
from backend.schemas.defect import TIANCHI_CLASS_NAMES

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_TRAIN_LABELS = _PROJECT_ROOT / "data" / "tianchi" / "train" / "labels"
_VAL_LABELS = _PROJECT_ROOT / "data" / "tianchi" / "val" / "labels"

# A correction set is worth an incremental fine-tune once this many have been
# applied. Kept in one place so the service and the demo agree.
FINE_TUNE_THRESHOLD = 50


def parse_yolo(lines: List[str]) -> List[tuple]:
    """Parse YOLO label lines into ``[(cls, cx, cy, w, h)]``."""
    out = []
    for ln in lines:
        ln = ln.strip()
        if not ln:
            continue
        p = ln.split()
        out.append(
            (int(float(p[0])), float(p[1]), float(p[2]), float(p[3]), float(p[4]))
        )
    return out


def boxes_to_yolo(boxes: List[tuple]) -> str:
    """Serialize ``[(cls, cx, cy, w, h)]`` back to YOLO label text."""
    return "\n".join(
        f"{c} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}" for c, cx, cy, w, h in boxes
    ) + ("\n" if boxes else "")


def iou(a: tuple, b: tuple) -> float:
    """IoU of two normalized boxes ``(cls, cx, cy, w, h)``."""
    _, acx, acy, aw, ah = a
    _, bcx, bcy, bw, bh = b
    ax0, ay0, ax1, ay1 = acx - aw / 2, acy - ah / 2, acx + aw / 2, acy + ah / 2
    bx0, by0, bx1, by1 = bcx - bw / 2, bcy - bh / 2, bcx + bw / 2, bcy + bh / 2
    ix = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    iy = max(0.0, min(ay1, by1) - max(ay0, by0))
    inter = ix * iy
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def apply_feedback(
    boxes: List[tuple], fb: dict, iou_thr: float = 0.3
) -> Tuple[str, str, List[tuple]]:
    """Apply one feedback to a list of boxes; return ``(action, detail, new_boxes)``.

    ``fb`` carries ``type`` (feedback_type), ``class_id``, ``cx/cy/w/h``, and
    optionally ``old_class_id``.
    """
    ftype = fb["type"]
    cls = fb["class_id"]
    box = (cls, fb["cx"], fb["cy"], fb["w"], fb["h"])

    if ftype == "missed_defect":
        if any(iou(box, b) >= iou_thr for b in boxes):
            return "skipped", "box already present in label", boxes
        boxes.append(box)
        return "added", f"append {TIANCHI_CLASS_NAMES[cls]} box", boxes

    if ftype == "misclassification":
        old = fb.get("old_class_id")
        old_name = TIANCHI_CLASS_NAMES[old] if old is not None else "unknown"
        for i, b in enumerate(boxes):
            if b[0] == old and iou(box, b) >= iou_thr:
                boxes[i] = (cls, b[1], b[2], b[3], b[4])
                return "changed", f"{old_name} -> {TIANCHI_CLASS_NAMES[cls]}", boxes
        return "skipped", f"no {old_name} box matches", boxes

    if ftype == "false_positive":
        for i, b in enumerate(boxes):
            if b[0] == cls and iou(box, b) >= iou_thr:
                boxes.pop(i)
                return (
                    "removed",
                    f"drop hallucinated {TIANCHI_CLASS_NAMES[cls]} box",
                    boxes,
                )
        return (
            "no-op",
            f"no GT box matches -> image is a hard negative for {TIANCHI_CLASS_NAMES[cls]}",
            boxes,
        )

    return "unknown", ftype, boxes


class LabelFeedbackService:
    """Stores operator corrections and applies them back to the label files."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def create_feedback(self, request: LabelFeedbackRequest) -> LabelFeedback:
        """Validate and persist one correction."""
        if request.feedback_type not in {ft.value for ft in FeedbackType}:
            raise ValueError(f"unknown feedback_type: {request.feedback_type}")
        if request.class_id not in range(len(TIANCHI_CLASS_NAMES)):
            raise ValueError(f"class_id out of range: {request.class_id}")
        if (
            request.feedback_type == "misclassification"
            and request.old_class_id is None
        ):
            raise ValueError("misclassification feedback requires old_class_id")

        fb = LabelFeedback(
            feedback_id=f"FB-{uuid.uuid4().hex[:10].upper()}",
            image_id=request.image_id,
            feedback_type=FeedbackType(request.feedback_type),
            class_id=request.class_id,
            class_name=TIANCHI_CLASS_NAMES[request.class_id],
            bbox=request.bbox,
            old_class_id=request.old_class_id,
            confidence=request.confidence,
            status="pending",
            source=request.source,
            note=request.note,
        )
        self.session.add(fb)
        await self.session.commit()
        await self.session.refresh(fb)
        return fb

    async def list_feedback(
        self, status: Optional[str] = None, limit: int = 50
    ) -> List[LabelFeedback]:
        """List feedback rows, optionally filtered by status."""
        query = (
            select(LabelFeedback).order_by(LabelFeedback.created_at.desc()).limit(limit)
        )
        if status:
            query = query.where(LabelFeedback.status == status)
        result = await self.session.execute(query)
        return list(result.scalars().all())

    async def apply_pending(self, iou_thr: float = 0.3) -> dict:
        """Apply every pending correction to its label file and mark it applied.

        Returns a summary with counts and whether the fine-tune threshold has
        been reached.
        """
        result = await self.session.execute(
            select(LabelFeedback).where(LabelFeedback.status == "pending")
        )
        pending = list(result.scalars().all())

        summary = {"total": len(pending), "applied": 0, "skipped": 0, "no_label": 0}
        for fb in pending:
            label_path = self._locate_label(fb.image_id)
            if label_path is None:
                summary["no_label"] += 1
                continue
            boxes = parse_yolo(label_path.read_text().splitlines())
            action, _detail, boxes = apply_feedback(
                boxes,
                {
                    "type": fb.feedback_type.value,
                    "class_id": fb.class_id,
                    "cx": fb.bbox.get("cx", 0.0),
                    "cy": fb.bbox.get("cy", 0.0),
                    "w": fb.bbox.get("w", 0.0),
                    "h": fb.bbox.get("h", 0.0),
                    "old_class_id": fb.old_class_id,
                },
                iou_thr,
            )
            if action in ("added", "changed", "removed"):
                label_path.write_text(boxes_to_yolo(boxes))
                summary["applied"] += 1
                fb.status = "applied"
                fb.applied_at = datetime.now(timezone.utc)
            else:
                summary["skipped"] += 1
                fb.status = "applied"  # a no-op verdict is still resolved
                fb.applied_at = datetime.now(timezone.utc)

        await self.session.commit()
        summary["should_fine_tune"] = summary["applied"] >= FINE_TUNE_THRESHOLD
        return summary

    @staticmethod
    def _locate_label(image_id: str) -> Optional[Path]:
        """Find the YOLO label file for ``image_id`` (train first, then val)."""
        stem = Path(image_id).stem
        for base in (_TRAIN_LABELS, _VAL_LABELS):
            p = base / f"{stem}.txt"
            if p.exists():
                return p
        return None
