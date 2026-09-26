"""
Merge confusable classes in the Tianchi label set to recover recall.

Per the human audit verdict (2026-09-23), two pairs are merged so a model no
longer needs to tell apart defects that are visually indistinguishable:

  * surface_mark (18) -> broken_warp (9)   (merged class keeps critical severity)
  * coarse_weft  (11) -> weft_shrink (12)  (label contamination absorbed by merge)

This remaps every YOLO label file in train/val/test (``class cx cy w h`` ->
``new_class cx cy w h``) and rewrites ``data.yaml`` from ``nc=20`` to ``nc=18``.

The mapping is fully specified by ``OLD_NAMES`` / ``NEW_NAMES`` / ``REMAP`` and
is reversible. Run with ``--dry-run`` to preview the change counts only.

Usage:
    python scripts/merge_confusable_classes.py            # apply
    python scripts/merge_confusable_classes.py --dry-run  # report only
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Old 20-class taxonomy (model id 0..19), matching data.yaml before the merge.
OLD_NAMES = [
    "hole",
    "stain",
    "three_silk",
    "knot",
    "flower_board",
    "hundred_feet",
    "hair_particle",
    "coarse_warp",
    "loose_warp",
    "broken_warp",
    "hanging_warp",
    "coarse_weft",
    "weft_shrink",
    "size_stain",
    "warping_knot",
    "star_skip",
    "broken_spandex",
    "dense_section",
    "surface_mark",
    "weave_defect",
]

# New 18-class taxonomy (id 0..17).
NEW_NAMES = [
    "hole",
    "stain",
    "three_silk",
    "knot",
    "flower_board",
    "hundred_feet",
    "hair_particle",
    "coarse_warp",
    "loose_warp",
    "broken_warp",
    "hanging_warp",
    "weft_shrink",
    "size_stain",
    "warping_knot",
    "star_skip",
    "broken_spandex",
    "dense_section",
    "weave_defect",
]

# old id -> new id
REMAP = {
    0: 0,
    1: 1,
    2: 2,
    3: 3,
    4: 4,
    5: 5,
    6: 6,
    7: 7,
    8: 8,
    9: 9,  # broken_warp stays 9
    10: 10,
    11: 11,  # coarse_weft  -> weft_shrink (new id 11)
    12: 11,  # weft_shrink  -> 11
    13: 12,  # size_stain   -> 12
    14: 13,  # warping_knot -> 13
    15: 14,  # star_skip    -> 14
    16: 15,  # broken_spandex -> 15
    17: 16,  # dense_section -> 16
    18: 9,  # surface_mark -> broken_warp (new id 9)
    19: 17,  # weave_defect -> 17
}


def remap_label(path: Path, stats: Counter) -> str:
    """Remap the class ids in one YOLO label file, returning the new content."""
    out: list = []
    for ln in path.read_text().splitlines():
        if not ln.strip():
            continue
        parts = ln.split()
        c = int(float(parts[0]))  # float() tolerates a stray "11.0"
        stats[c] += 1
        out.append(" ".join([str(REMAP[c]), *parts[1:]]))
    return "\n".join(out) + ("\n" if out else "")


def write_data_yaml(root: Path) -> None:
    """Rewrite data.yaml with nc=18 and the new names."""
    yaml_path = root / "data.yaml"
    lines = [
        f"path: {root.resolve().as_posix()}",
        "train: train/images",
        "val: val/images",
        "test: test/images",
    ]
    lines.append("nc: 18")
    lines.append("names:")
    for i, name in enumerate(NEW_NAMES):
        lines.append(f"  {i}: {name}")
    # UTF-8 explicitly: the path contains non-ASCII (Chinese) characters and
    # Path.write_text() defaults to the locale codec (cp936 here), which would
    # corrupt the path and break ultralytics' UTF-8 YAML load.
    yaml_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Report only, no writes")
    args = parser.parse_args()

    root = Path("data/tianchi")
    stats: Counter = Counter()

    for split in ("train", "val", "test"):
        labels_dir = root / split / "labels"
        if not labels_dir.is_dir():
            continue
        n_files = 0
        for p in sorted(labels_dir.glob("*.txt")):
            content = remap_label(p, stats)
            n_files += 1
            if not args.dry_run:
                p.write_text(content)
        print(f"[merge] {split}: {n_files} label files processed")

    print("[merge] per-old-class box counts (changed by this mapping):")
    for c in sorted(stats):
        print(
            f"  old {c:2d} {OLD_NAMES[c]:<16} -> new {REMAP[c]:2d} {NEW_NAMES[REMAP[c]]:<14} : {stats[c]}"
        )

    if args.dry_run:
        print("[merge] DRY RUN — no files written")
        return 0

    write_data_yaml(root)
    # Regenerate classes.txt (one name per line) for consistency.
    (root / "classes.txt").write_text("\n".join(NEW_NAMES) + "\n")
    print(f"[merge] rewrote {root / 'data.yaml'} (nc=18) and classes.txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
