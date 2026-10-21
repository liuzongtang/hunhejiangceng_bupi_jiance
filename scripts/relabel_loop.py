"""
Relabel loop (#16): turn operator feedback into YOLO label corrections.

This is the human-in-the-loop half of the deployment layer. Alarms and
low-confidence detections get reviewed on the dashboard; the verdict (false
positive / missed defect / misclassification) lands as a ``LabelFeedback``
row, and this script converts it into a concrete YOLO label edit and tracks
how many corrections have accumulated — the trigger for an incremental
fine-tune.

This is a *sample* demo: it applies three example feedbacks to a scratch copy
of a real label file and prints the before/after diff. The production path is
``backend/routers/feedback.py`` (``POST /api/v1/feedback``) plus the service's
``apply_pending``, which persists rows and batch-applies them to the labels.

Usage:
    python scripts/relabel_loop.py            # run the 3-example demo
    python scripts/relabel_loop.py --dry-run  # same, but don't write scratch files
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

VAL_LABELS = Path("data/tianchi/val/labels")
SCRATCH = Path("data/tianchi/synth/relabel_demo")

# The pure correction logic is the single source of truth in the service; the
# demo reuses it rather than duplicating it.
from backend.services.label_feedback_service import (
    FINE_TUNE_THRESHOLD,
    apply_feedback,
    boxes_to_yolo,
    parse_yolo,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    # Pick a real val label that has a broken_spandex (class 15) box.
    target = None
    for lbl in sorted(VAL_LABELS.glob("*.txt")):
        if any(
            int(float(ln.split()[0])) == 15
            for ln in lbl.read_text().splitlines()
            if ln.strip()
        ):
            target = lbl
            break
    if target is None:
        print("no val label with broken_spandex found")
        return 1

    orig_lines = target.read_text().splitlines()
    boxes = parse_yolo(orig_lines)
    print(f"== demo on {target.name} ({len(boxes)} boxes) ==\n")

    # Three example feedbacks built from the label's actual boxes so every
    # mechanism fires. In practice these arrive as LabelFeedback rows.
    b15 = next((b for b in boxes if b[0] == 15), boxes[0])
    first = boxes[0]
    examples = [
        {
            "type": "missed_defect",
            "class_id": 15,
            "cx": 0.50,
            "cy": 0.50,
            "w": 0.03,
            "h": 0.03,
        },
        {
            "type": "misclassification",
            "old_class_id": 15,
            "class_id": 9,
            "cx": b15[1],
            "cy": b15[2],
            "w": b15[3],
            "h": b15[4],
        },
        {
            "type": "false_positive",
            "class_id": first[0],
            "cx": first[1],
            "cy": first[2],
            "w": first[3],
            "h": first[4],
        },
    ]

    applied = 0
    for fb in examples:
        action, detail, boxes = apply_feedback(boxes, fb)
        status = "✓" if action in ("added", "changed", "removed") else "·"
        print(f"{status} [{fb['type']:<16}] {action:<9} {detail}")
        applied += action in ("added", "changed", "removed")

    print()
    new_content = boxes_to_yolo(boxes)

    # Show the actual label diff.
    print("== label diff ==")
    before = sorted(orig_lines)
    after = sorted(new_content.splitlines())
    removed = [f"- {l}" for l in before if l not in after]
    added = [f"+ {l}" for l in after if l not in before]
    for l in removed:
        print(l)
    for l in added:
        print(l)
    if not removed and not added:
        print("(no net change)")

    print(
        f"\n[relabel] {applied} correction(s) applied; "
        f"{FINE_TUNE_THRESHOLD - applied} more until incremental fine-tune trigger"
    )
    if not args.dry_run:
        SCRATCH.mkdir(parents=True, exist_ok=True)
        out = SCRATCH / (target.stem + ".corrected.txt")
        out.write_text(new_content)
        print(f"[relabel] corrected label written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
