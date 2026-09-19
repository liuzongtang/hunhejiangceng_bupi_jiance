"""
Convert the extracted Tianchi "smartdiagnosisofclothflaw" dataset to YOLO format.

The dataset ships as extracted folders:

    <ds_root>/
        defect_Images/              # defect images (.jpg)
        normal_Images/              # no-defect images (.jpg, negative samples)
        Annotations/anno_train.json # [{"name", "defect_name", "bbox": [xmin,ymin,xmax,ymax]}, ...]

with Chinese defect names mapping to 20 category ids (plus id 0 = no defect).
This project's `backend.training.dataset.FabricDataset` reads YOLO format
(`images/*.jpg` + `labels/*.txt`), so this script normalizes each bbox to
`class cx cy w h` (all 0..1) and emits one `.txt` per image. Normal images
become empty label files (negative samples).

Usage:
    python scripts/convert_tianchi_to_yolo.py \
        --train-dirs shujuji/smartdiagnosisofclothflaw_round1train1_datasets_partA/partA \
                     shujuji/smartdiagnosisofclothflaw_round1train1_datasets_partB/partB \
                     shujuji/smartdiagnosisofclothflaw_round1train2_datasets/guangdong1_round1_train2_20190828 \
        --test-dirs shujuji/smartdiagnosisofclothflaw_round1testA_datasets/guangdong1_round1_testA_20190818 \
                    shujuji/smartdiagnosisofclothflaw_round1testB_datasets/guangdong1_round1_testB_20190919 \
        --out data/tianchi --class-scheme tianchi20
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from PIL import Image

# ---------------------------------------------------------------------------
# Tianchi class scheme (20 defect categories, id 1..20; id 0 = no defect).
# ---------------------------------------------------------------------------

# Chinese defect_name -> category id (1..20), per the dataset README.
NAME_TO_CATEGORY: Dict[str, int] = {
    "破洞": 1,
    "水渍": 2, "油渍": 2, "污渍": 2,
    "三丝": 3,
    "结头": 4,
    "花板跳": 5,
    "百脚": 6,
    "毛粒": 7,
    "粗经": 8,
    "松经": 9,
    "断经": 10,
    "吊经": 11,
    "粗纬": 12, "粗维": 12,  # 粗纬 (coarse weft); 粗维 is the README's typo
    "纬缩": 13,
    "浆斑": 14,
    "整经结": 15,
    "星跳": 16, "跳花": 16,
    "断氨纶": 17,
    "稀密档": 18, "浪纹档": 18, "色差档": 18,
    "磨痕": 19, "轧痕": 19, "修痕": 19, "烧毛痕": 19,
    "死皱": 20, "云织": 20, "双纬": 20, "双经": 20,
    "跳纱": 20, "筘路": 20, "纬纱不良": 20,
}

# category id (1..20) -> english class name
TIANCHI_CATEGORIES: List[str] = [
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

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}


def category_to_class_id(category: int, scheme: str) -> int:
    """Map a Tianchi category id (1..20) to a 0-indexed class id."""
    if scheme == "tianchi20":
        return category - 1
    raise ValueError(f"Unknown class scheme: {scheme}")


def class_names_for(scheme: str) -> List[str]:
    """Return the class-name list matching the chosen scheme."""
    if scheme == "tianchi20":
        return TIANCHI_CATEGORIES
    raise ValueError(f"Unknown class scheme: {scheme}")


def load_annotations(anno_path: str) -> Dict[str, List[Dict]]:
    """Read anno_train.json and group annotations by image name."""
    by_name: Dict[str, List[Dict]] = defaultdict(list)
    if not os.path.exists(anno_path):
        return by_name
    with open(anno_path, encoding="utf-8") as f:
        for ann in json.load(f):
            by_name[ann["name"]].append(ann)
    return by_name


def xyxy_to_yolo(bbox: List[float], w: int, h: int) -> Tuple[float, float, float, float]:
    """Convert xyxy pixel bbox to normalized (cx, cy, bw, bh)."""
    xmin, ymin, xmax, ymax = (float(v) for v in bbox)
    cx = (xmin + xmax) / 2.0 / w
    cy = (ymin + ymax) / 2.0 / h
    bw = max(0.0, (xmax - xmin)) / w
    bh = max(0.0, (ymax - ymin)) / h
    cx, cy = min(max(cx, 0.0), 1.0), min(max(cy, 0.0), 1.0)
    bw, bh = min(bw, 1.0), min(bh, 1.0)
    return cx, cy, bw, bh


def convert_train_dir(
    ds_dir: str,
    out_images: str,
    out_labels: str,
    scheme: str,
) -> int:
    """
    Convert one extracted train dataset dir to YOLO format.

    Args:
        ds_dir: Directory containing defect_Images/, normal_Images/,
            Annotations/anno_train.json.
        out_images: Flat directory to write images into.
        out_labels: Flat directory to write label .txt files into.
        scheme: "tianchi20".

    Returns:
        Number of images written.
    """
    os.makedirs(out_images, exist_ok=True)
    os.makedirs(out_labels, exist_ok=True)

    by_name = load_annotations(os.path.join(ds_dir, "Annotations", "anno_train.json"))
    count = 0
    skipped = 0

    for subdir in ("defect_Images", "normal_Images"):
        img_dir = os.path.join(ds_dir, subdir)
        if not os.path.isdir(img_dir):
            continue
        for fname in sorted(os.listdir(img_dir)):
            if os.path.splitext(fname)[1].lower() not in IMAGE_EXTS:
                continue
            src = os.path.join(img_dir, fname)

            # Read dimensions to normalize bbox coordinates.
            with Image.open(src) as im:
                w, h = im.size

            dst_img = os.path.join(out_images, fname)
            if os.path.exists(dst_img):
                print(f"  [warn] duplicate image name skipped: {fname}")
                continue
            shutil.copy(src, dst_img)

            # Build YOLO label (empty file for normal images).
            lines: List[str] = []
            for ann in by_name.get(fname, []):
                category = NAME_TO_CATEGORY.get(ann["defect_name"])
                if category is None:
                    skipped += 1
                    continue
                cls = category_to_class_id(category, scheme)
                cx, cy, bw, bh = xyxy_to_yolo(ann["bbox"], w, h)
                lines.append(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

            label_path = os.path.join(out_labels, os.path.splitext(fname)[0] + ".txt")
            with open(label_path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines) + ("\n" if lines else ""))

            count += 1

    if skipped:
        print(f"  [warn] {skipped} annotations had unknown defect_name and were skipped")
    return count


def convert_test_dir(test_dir: str, out_images: str) -> int:
    """Copy an extracted test dir (flat .jpg, no labels) into out_images."""
    os.makedirs(out_images, exist_ok=True)
    count = 0
    for fname in sorted(os.listdir(test_dir)):
        if os.path.splitext(fname)[1].lower() not in IMAGE_EXTS:
            continue
        shutil.copy(os.path.join(test_dir, fname), os.path.join(out_images, fname))
        count += 1
    return count


def split_train_val(root: str, val_ratio: float, seed: int) -> Tuple[int, int]:
    """Split the staging train dir into train/ and val/ by image name."""
    staging_img = os.path.join(root, "_staging", "images")
    staging_lbl = os.path.join(root, "_staging", "labels")
    train_img = os.path.join(root, "train", "images")
    train_lbl = os.path.join(root, "train", "labels")
    val_img = os.path.join(root, "val", "images")
    val_lbl = os.path.join(root, "val", "labels")

    all_images = sorted(os.listdir(staging_img))
    random.seed(seed)
    random.shuffle(all_images)
    n_val = int(len(all_images) * val_ratio)
    val_set = set(all_images[:n_val])

    for d in (train_img, train_lbl, val_img, val_lbl):
        os.makedirs(d, exist_ok=True)

    for name in all_images:
        dst_img_dir = val_img if name in val_set else train_img
        dst_lbl_dir = val_lbl if name in val_set else train_lbl
        os.rename(os.path.join(staging_img, name), os.path.join(dst_img_dir, name))
        lbl = os.path.splitext(name)[0] + ".txt"
        src_lbl = os.path.join(staging_lbl, lbl)
        if os.path.exists(src_lbl):
            os.rename(src_lbl, os.path.join(dst_lbl_dir, lbl))

    shutil.rmtree(os.path.join(root, "_staging"), ignore_errors=True)
    return len(all_images) - n_val, n_val


def write_meta(root: str, scheme: str) -> None:
    """Write classes.txt and data.yaml (Ultralytics) for the output root."""
    names = class_names_for(scheme)
    with open(os.path.join(root, "classes.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(names) + "\n")

    yaml_lines = [
        f"path: {os.path.abspath(root).replace(os.sep, '/')}",
        "train: train/images",
        "val: val/images",
        "test: test/images",
        f"nc: {len(names)}",
        "names:",
    ]
    yaml_lines += [f"  {i}: {n}" for i, n in enumerate(names)]
    with open(os.path.join(root, "data.yaml"), "w", encoding="utf-8") as f:
        f.write("\n".join(yaml_lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dirs", nargs="+", default=[], help="Extracted train dataset dirs")
    parser.add_argument("--test-dirs", nargs="+", default=[], help="Extracted test dirs (no labels)")
    parser.add_argument("--out", default="data/tianchi", help="Output root directory")
    parser.add_argument(
        "--class-scheme", default="tianchi20", choices=["tianchi20"],
        help="Class scheme to emit (default: tianchi20, 20 classes)",
    )
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for val split")
    args = parser.parse_args()

    scheme = args.class_scheme
    root = args.out
    print(f"Class scheme: {scheme} ({len(class_names_for(scheme))} classes)")
    print(f"Output root: {root}")

    # 1. Convert all train dirs into a staging train dir.
    staging_img = os.path.join(root, "_staging", "images")
    staging_lbl = os.path.join(root, "_staging", "labels")
    total = 0
    for ds_dir in args.train_dirs:
        print(f"Converting train dir: {ds_dir}")
        n = convert_train_dir(ds_dir, staging_img, staging_lbl, scheme)
        total += n
        print(f"  wrote {n} images")
    print(f"Total train images: {total}")

    # 2. Split staging -> train/val.
    if os.path.isdir(staging_img) and os.listdir(staging_img):
        n_train, n_val = split_train_val(root, args.val_ratio, args.seed)
        print(f"Split: {n_train} train / {n_val} val")
    else:
        print("  [warn] no train images found; skipping split")

    # 3. Convert test dirs (no labels).
    test_img = os.path.join(root, "test", "images")
    for test_dir in args.test_dirs:
        print(f"Converting test dir: {test_dir}")
        n = convert_test_dir(test_dir, test_img)
        print(f"  wrote {n} test images")

    # 4. Write classes.txt + data.yaml.
    write_meta(root, scheme)

    print("Done.")
    print(f"  train: {os.path.join(root, 'train', 'images')}")
    print(f"  val:   {os.path.join(root, 'val', 'images')}")
    print(f"  test:  {test_img}")
    print(f"  classes: {os.path.join(root, 'classes.txt')}")
    print(f"  data.yaml: {os.path.join(root, 'data.yaml')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
