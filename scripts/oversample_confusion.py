"""
Build an oversampled train list targeting the line/weft confusion classes.

The confusion diagnostic found three systematic pairs: broken_warp(9) <-> surface_mark(18),
weave_defect(19) <-> surface_mark(18), and weft_shrink(12) <-> coarse_weft(11). This script
writes a ``.txt`` list of train image paths with images containing any of those classes
repeated ``--repeat`` times (others once), plus a new ``data.yaml`` whose ``train`` points at
that list. Ultralytics reads a ``.txt`` train path as a list of image files
(``data/base.py:get_img_files``), so this oversamples without duplicating files on disk.

Usage:
    python scripts/oversample_confusion.py --repeat 2
"""

from __future__ import annotations

import argparse
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml


def _label_path(img: Path) -> Path:
    """images/...jpg -> labels/...txt (the ultralytics convention)."""
    parts = list(img.parts)
    try:
        i = parts.index("images")
    except ValueError:
        i = -2
    parts[i] = "labels"
    return Path(*parts).with_suffix(".txt")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/tianchi/data.yaml")
    parser.add_argument("--out-txt", default="data/tianchi/train_oversampled.txt")
    parser.add_argument("--out-yaml", default="data/tianchi/data_oversampled.yaml")
    parser.add_argument(
        "--classes", default="9,11,12,18,19", help="Confusion classes to oversample"
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=2,
        help="Repeat factor for images containing a target class",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    target = {int(x) for x in args.classes.split(",")}
    data = yaml.safe_load(Path(args.data).read_text(encoding="utf-8"))
    base = Path(data["path"])
    train_dir = base / data["train"]
    imgs = sorted(train_dir.glob("*.jpg")) + sorted(train_dir.glob("*.png"))
    if not imgs:
        raise SystemExit(f"no images under {train_dir}")

    rng = random.Random(args.seed)
    hits = dict.fromkeys(target, 0)
    n_target_imgs = 0
    out: list[str] = []
    for img in imgs:
        lbl = _label_path(img)
        classes: set[int] = set()
        if lbl.exists():
            for line in lbl.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                classes.add(int(line.split()[0]))
        inter = classes & target
        if inter:
            n_target_imgs += 1
            for c in inter:
                hits[c] += 1
            rep = args.repeat
        else:
            rep = 1
        out.extend([str(img)] * rep)

    rng.shuffle(out)
    Path(args.out_txt).write_text("\n".join(out) + "\n", encoding="utf-8")

    # New yaml: same as source but train -> oversampled list.
    data["train"] = Path(args.out_txt).name  # relative to data path
    Path(args.out_yaml).write_text(
        yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )

    print(
        f"[oversample] {len(imgs)} source images, {len(out)} samples after {args.repeat}x "
        f"on {n_target_imgs} target images"
    )
    print(f"[oversample] target-class image counts: {hits}")
    print(f"[oversample] wrote {args.out_txt} and {args.out_yaml}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
