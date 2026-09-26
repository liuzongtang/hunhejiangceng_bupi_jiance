"""
Create a deterministic subset of the Tianchi train set for fast pipeline smoke tests.

Generates two files under ``data/tianchi``:

  * ``train_subset<N>.txt``  — every 1/fraction-th train image (absolute posix paths)
  * ``data_subset<N>.yaml``  — nc=18, ``train:`` points at the .txt, val/test unchanged

The subset is a deterministic stride over the sorted image list, so it is
reproducible and roughly preserves the class distribution. Val stays full
(957 images) so the mAP signal stays stable and comparable to full runs.

Usage:
    python scripts/make_subset.py --fraction 0.25   # 25% of train images
"""

from __future__ import annotations

import argparse
from pathlib import Path

EXTS = (".jpg", ".jpeg", ".png", ".bmp")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fraction",
        type=float,
        default=0.25,
        help="Fraction of train images to keep (e.g. 0.25 = 25%%).",
    )
    args = parser.parse_args()

    if not 0.0 < args.fraction <= 1.0:
        raise SystemExit(f"fraction must be in (0, 1], got {args.fraction}")

    root = Path("data/tianchi")
    imgs = sorted(
        p for p in (root / "train" / "images").iterdir() if p.suffix.lower() in EXTS
    )
    stride = max(1, round(1.0 / args.fraction))
    subset = imgs[::stride]

    tag = str(round(args.fraction * 100))
    txt = root / f"train_subset{tag}.txt"
    txt.write_text(
        "\n".join(p.resolve().as_posix() for p in subset) + "\n", encoding="utf-8"
    )

    # Reuse the nc/names block from data.yaml (single source of truth).
    names = [
        line.strip()
        for line in (root / "data.yaml").read_text(encoding="utf-8").splitlines()
        if line.strip()[:1].isdigit() and ": " in line
    ]
    yaml = root / f"data_subset{tag}.yaml"
    yaml.write_text(
        f"path: {root.resolve().as_posix()}\n"
        f"train: {txt.name}\n"
        "val: val/images\n"
        "test: test/images\n"
        f"nc: {len(names)}\n"
        "names:\n" + "\n".join(f"  {n}" for n in names) + "\n",
        encoding="utf-8",
    )

    print(
        f"[subset] kept {len(subset)} / {len(imgs)} train images "
        f"(stride {stride}, {args.fraction:.0%})"
    )
    print(f"[subset] wrote {txt.name} and {yaml.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
