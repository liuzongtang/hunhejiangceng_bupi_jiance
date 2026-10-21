"""
Compare detector confidence for synthetic vs real broken_spandex (class 15).

The point of the #7 synthesis is to lift the recall of the missed critical
class (broken_spandex / 断氨纶). This script measures whether the *current*
detector detects the synthesized defects — and compares that to how it treats
the real (missed) ones.

Usage:
    python scripts/eval_synth_detection.py
    python scripts/eval_synth_detection.py --conf 0.05 --max-real 16
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

WEIGHTS = Path("runs/rtdetr/tianchi18_full/weights/best.pt")
SYNTH_DIR = Path("data/tianchi/synth/images")
VAL_LABELS = Path("data/tianchi/val/labels")
VAL_IMAGES = Path("data/tianchi/val/images")
CLASS_ID = 15  # broken_spandex


def collect_real(max_n: int) -> list:
    """Return up to ``max_n`` real val image paths that contain class 15."""
    out = []
    for lbl in sorted(VAL_LABELS.glob("*.txt")):
        if any(
            ln.split() and int(float(ln.split()[0])) == CLASS_ID
            for ln in lbl.read_text().splitlines()
            if ln.strip()
        ):
            img = VAL_IMAGES / (lbl.stem + ".jpg")
            if img.exists():
                out.append(img)
        if len(out) >= max_n:
            break
    return out


def detect(model, paths: list, conf: float) -> dict:
    """Run the detector, return per-image best class-15 confidence (or None)."""
    per_image = {}
    results = model.predict(
        list(map(str, paths)), imgsz=640, conf=conf, verbose=False, stream=True
    )
    for p, r in zip(paths, results, strict=False):
        if r.boxes is None or len(r.boxes) == 0:
            per_image[str(p)] = None
            continue
        cls = r.boxes.cls.cpu().numpy().astype(int)
        sc = r.boxes.conf.cpu().numpy()
        hit = sc[cls == CLASS_ID]
        per_image[str(p)] = float(hit.max()) if len(hit) else None
    return per_image


def summarize(tag: str, per_image: dict) -> None:
    vals = [v for v in per_image.values() if v is not None]
    n = len(per_image)
    n_hit = len(vals)
    mean = sum(vals) / len(vals) if vals else 0.0
    print(
        f"[{tag}] {n} images, {n_hit} detected class-15 "
        f"({100.0 * n_hit / max(1, n):.0f}%), mean conf {mean:.3f}"
    )
    if vals:
        print(
            f"           conf values: {', '.join(f'{v:.2f}' for v in sorted(vals, reverse=True))}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conf", type=float, default=0.05)
    parser.add_argument("--max-real", type=int, default=16)
    args = parser.parse_args()

    from ultralytics import YOLO

    model = YOLO(str(WEIGHTS))

    synth_paths = sorted(SYNTH_DIR.glob("*.jpg"))
    real_paths = collect_real(args.max_real)
    if not synth_paths:
        print("no synthetic images — run scripts/synthesize_defects.py first")
        return 1

    synth = detect(model, synth_paths, args.conf)
    summarize("synth", synth)
    summarize("real ", detect(model, real_paths, args.conf))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
