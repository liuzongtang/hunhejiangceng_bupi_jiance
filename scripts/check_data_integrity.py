#!/usr/bin/env python
"""check_data_integrity.py -- R2 data gate for the Tianchi YOLO dataset.

Mirrors handoff section "R2 数据完整性校验", rewritten as a script because the
handoff's bash heredoc is not portable to PowerShell. Any mismatch is a hard
stop: the 2026-10 historical bug was partA/partB annotations being byte-identical
while the image names did not overlap, so partB images silently became empty
label files and polluted the positive/negative ratio.

Expected values for the current data/tianchi build:
    val images 957, defect 583, empty 374, boxes 953, train/val leak 0
    train 8619 images, test present

Exit code 0 = all gates pass, 1 = at least one gate failed.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

IMG_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

EXPECTED = {
    "val_images": 957,
    "val_defect": 583,
    "val_empty": 374,
    "val_boxes": 953,
    "leak": 0,
}


def split_images(root: Path, split: str) -> dict[str, Path]:
    """Return ``stem -> path`` for one split's images."""
    img_dir = root / split / "images"
    if not img_dir.is_dir():
        return {}
    return {
        p.stem: p
        for p in img_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMG_SUFFIXES
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="data/tianchi")
    ap.add_argument("--allow-missing-train", action="store_true")
    args = ap.parse_args()
    root = Path(args.root)
    if not root.is_dir():
        print(f"FAIL: {root} does not exist", file=sys.stderr)
        return 1

    failures: list[str] = []
    stats: dict[str, int] = {}

    for split in ("train", "val", "test"):
        img_dir = root / split / "images"
        lab_dir = root / split / "labels"
        n_img = len(split_images(root, split))
        n_lab = len(list(lab_dir.glob("*.txt"))) if lab_dir.is_dir() else 0
        stats[f"{split}_images"] = n_img
        stats[f"{split}_labels"] = n_lab
        print(f"{split:<5s} images={n_img:>5d} labels={n_lab:>5d}")
        if split == "train" and n_img == 0 and not args.allow_missing_train:
            failures.append("train images = 0 (use --allow-missing-train to test on val only)")
        # test/ is unlabelled by design (competition submission split): parity
        # between images and .txt files only makes sense for train/val.
        if split != "test" and n_img and n_img != n_lab:
            failures.append(f"{split}: images != label files ({n_img} vs {n_lab})")

    val_lab = root / "val" / "labels"
    defect = empty = boxes = 0
    bad_lines = 0
    for lp in val_lab.glob("*.txt") if val_lab.is_dir() else []:
        lines = [ln for ln in lp.read_text(encoding="utf-8").splitlines() if ln.strip()]
        if lines:
            defect += 1
            boxes += len(lines)
            for ln in lines:
                parts = ln.split()
                if len(parts) != 5:
                    bad_lines += 1
                    continue
                cid = int(float(parts[0]))
                if not 0 <= cid < 18:
                    bad_lines += 1
                for v in parts[1:]:
                    f = float(v)
                    if not -0.01 <= f <= 1.01:
                        bad_lines += 1
        else:
            empty += 1
    stats.update(val_defect=defect, val_empty=empty, val_boxes=boxes, bad_label_lines=bad_lines)
    print(f"val defect/empty/boxes = {defect} / {empty} / {boxes}")

    train_stems = set(split_images(root, "train"))
    val_stems = set(split_images(root, "val"))
    leak = train_stems & val_stems
    stats["leak"] = len(leak)
    print(f"train/val 泄漏: {len(leak)}" + (f"  e.g. {sorted(leak)[:5]}" if leak else ""))

    for key, want in EXPECTED.items():
        got = stats.get(key)
        if got != want:
            failures.append(f"{key}: got {got}, expected {want}")

    if bad_lines:
        failures.append(f"{bad_lines} malformed label lines (class id / normalized coords)")

    print()
    if failures:
        print("R2 FAIL:", file=sys.stderr)
        for f in failures:
            print(f"  - {f}", file=sys.stderr)
        print(
            "\nDo NOT proceed to R3. Rebuild the dataset (scripts/convert_tianchi_to_yolo.py) "
            "and check image<->label matching by stem, per handoff pitfall #9.",
            file=sys.stderr,
        )
        return 1
    print("R2 PASS: dataset matches the expected Tianchi build.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
