#!/usr/bin/env python
"""rebuild_tianchi20_val.py -- rebuild 20-class val labels for fair comparison.

The project merged two confusable class pairs in place (surface_mark ->
broken_warp, coarse_weft -> weft_shrink), overwriting the original 20-class
labels under data/tianchi/val/labels. The public Tianchi solutions
(59.4~65.8% mAP50) are all 20-class, so an apples-to-apples comparison needs
the original 20-class labels for the SAME 957 val images.

This script re-derives them from the untouched raw annotations in
shujuji/*/Annotations/anno_train.json (same mapping table as
scripts/convert_tianchi_to_yolo.py) and writes them to data/tianchi20/.

Output layout:
    data/tianchi20/labels/val/<stem>.txt     # 20-class YOLO labels
    data/tianchi20/images/val/               # junction -> data/tianchi/val/images
    data/tianchi20/data.yaml                 # nc=20

Usage:
    python scripts/rebuild_tianchi20_val.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
SHUJUJI = ROOT / "shujuji"
VAL_IMAGES = ROOT / "data" / "tianchi" / "val" / "images"
OUT = ROOT / "data" / "tianchi20"

TRAIN_DIRS = [
    SHUJUJI / "smartdiagnosisofclothflaw_round1train1_datasets_partA" / "partA",
    SHUJUJI / "smartdiagnosisofclothflaw_round1train1_datasets_partB" / "partB",
    SHUJUJI
    / "smartdiagnosisofclothflaw_round1train2_datasets"
    / "guangdong1_round1_train2_20190828",
]

# Same table as convert_tianchi_to_yolo.py (do not diverge).
NAME_TO_CATEGORY = {
    "破洞": 1,
    "水渍": 2,
    "油渍": 2,
    "污渍": 2,
    "三丝": 3,
    "结头": 4,
    "花板跳": 5,
    "百脚": 6,
    "毛粒": 7,
    "粗经": 8,
    "松经": 9,
    "断经": 10,
    "吊经": 11,
    "粗纬": 12,
    "粗维": 12,
    "纬缩": 13,
    "浆斑": 14,
    "整经结": 15,
    "星跳": 16,
    "跳花": 16,
    "断氨纶": 17,
    "稀密档": 18,
    "浪纹档": 18,
    "色差档": 18,
    "磨痕": 19,
    "轧痕": 19,
    "修痕": 19,
    "烧毛痕": 19,
    "死皱": 20,
    "云织": 20,
    "双纬": 20,
    "双经": 20,
    "跳纱": 20,
    "筘路": 20,
    "纬纱不良": 20,
}

TIANCHI_CATEGORIES = [
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


def main() -> int:
    if not VAL_IMAGES.is_dir():
        print(f"ERROR: val images not found: {VAL_IMAGES}", file=sys.stderr)
        return 1

    # 1. Collect raw annotations from all three train dirs.
    # NOTE: partA/partB/anno_train.json are byte-identical (md5-verified) and
    # jointly annotate all 4774 defect images. Dedupe at JSON-batch level: the
    # first JSON that mentions an image owns its annotations; later identical
    # JSONs contribute nothing (matches the original converter, which staged
    # partA+partB images but never double-applied annotations).
    by_name: dict[str, list[dict]] = {}
    claimed: set[str] = set()
    for ds_dir in TRAIN_DIRS:
        anno = ds_dir / "Annotations" / "anno_train.json"
        if not anno.exists():
            print(f"  [warn] missing {anno}")
            continue
        entries = json.loads(anno.read_text(encoding="utf-8"))
        batch_names = {a["name"] for a in entries}
        fresh = batch_names - claimed
        for ann in entries:
            if ann["name"] in fresh:
                by_name.setdefault(ann["name"], []).append(ann)
        claimed |= batch_names
        print(
            f"  {anno.parent.parent.name}: {len(entries)} entries, {len(fresh)} new images"
        )
    print(f"raw annotations cover {len(by_name)} images")

    # 2. Write 20-class labels for the val split.
    out_labels = OUT / "labels" / "val"
    out_labels.mkdir(parents=True, exist_ok=True)
    val_images = sorted(
        p for p in VAL_IMAGES.iterdir() if p.suffix.lower() in {".jpg", ".jpeg"}
    )
    n_with_defect = 0
    n_boxes = 0
    skipped_names: set[str] = set()
    missing = 0
    for img_path in val_images:
        anns = by_name.get(img_path.name)
        if anns is None:
            # normal image (no defect) -> empty label file
            missing += 1
        with Image.open(img_path) as im:
            w, h = im.size
        lines = []
        for ann in anns or []:
            cat = NAME_TO_CATEGORY.get(ann["defect_name"])
            if cat is None:
                skipped_names.add(ann["defect_name"])
                continue
            xmin, ymin, xmax, ymax = (float(v) for v in ann["bbox"])
            cx = min(max((xmin + xmax) / 2.0 / w, 0.0), 1.0)
            cy = min(max((ymin + ymax) / 2.0 / h, 0.0), 1.0)
            bw = min(max((xmax - xmin) / w, 0.0), 1.0)
            bh = min(max((ymax - ymin) / h, 0.0), 1.0)
            lines.append(f"{cat - 1} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        (out_labels / f"{img_path.stem}.txt").write_text(
            "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
        )
        if lines:
            n_with_defect += 1
            n_boxes += len(lines)

    print(
        f"val images: {len(val_images)} ({missing} normal/empty, {n_with_defect} with defects)"
    )
    print(f"total 20-class boxes in val: {n_boxes}")
    if skipped_names:
        print(f"  [warn] unknown defect_names skipped: {sorted(skipped_names)}")

    # 3. Hard-link images/val -> the existing val images.
    # Hard links (not junctions!): junctions resolve transparently, which makes
    # ultralytics resolve label paths back to the 18-class labels via its
    # "/images/" -> "/labels/" string replacement. Hard links are real
    # directory entries at the new path, so labels resolve to tianchi20/.
    link_dir = OUT / "images" / "val"
    link_dir.mkdir(parents=True, exist_ok=True)
    n_link = 0
    for img_path in val_images:
        dst = link_dir / img_path.name
        if not dst.exists():
            os.link(img_path, dst)
            n_link += 1
    print(f"hard-linked {n_link} val images into {link_dir}")

    # 4. data.yaml (nc=20).
    yaml_lines = [
        f"path: {OUT.as_posix()}",
        "train: train/images",  # unused for val-only eval
        "val: images/val",
        "test: test/images",
        "nc: 20",
        "names:",
    ]
    yaml_lines += [f"  {i}: {n}" for i, n in enumerate(TIANCHI_CATEGORIES)]
    (OUT / "data.yaml").write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")
    print(f"wrote {OUT / 'data.yaml'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
