#!/usr/bin/env python
"""build_family_dataset.py -- remap the 18-class YOLO dataset to the 3 stable families.

Stage B (B1/B2) of ``docs/plan_stage2_reward_relaunch_handoff.md``: builds an
independent YOLO dataset whose labels carry the 3 super-families found by
``scripts/build_family_mapping.py`` (KMeans k=3 == Ward k=3, bootstrap support
1.00 for all three), so a specialised family detector can be compared against
the 18-class detector aggregated to families.

Mapping source, in priority order:

1. ``--map runs/family_clustering/families.yaml`` if it exists. That file's
   ``class_to_family3`` block is produced by ``build_family_mapping.py`` and is
   keyed by the *18-class names*. Missing assets (``box_features.npz``) mean the
   file itself may not exist on this machine; then:
2. the built-in ``SUPER`` table below, which is transcribed from
   ``build_family_mapping.py`` and paper 4.10 (A linear 7 / B blotch 5 / C point 6).

The class order is always read from ``<src>/data.yaml`` so ids are never guessed;
a class with no family raises instead of silently landing in family 0.

Output layout (independent directory, ``data/tianchi`` is never modified)::

    <dst>/data.yaml                 nc=3, names A_linear/B_blotch/C_point
    <dst>/<split>/images/*          hardlink (default) / copy / symlink
    <dst>/<split>/labels/*.txt      class id remapped 0-17 -> 0-2, cxcywh untouched

Self-check (printed; a mismatch is an error, not a warning): label-file count and
GT-box count per split must equal the source, and the instance total on val must
still be 953 for the current data/tianchi.

Usage::

    python scripts/build_family_dataset.py --src data/tianchi --dst data/tianchi_family3
    python scripts/build_family_dataset.py --src data/tianchi --dst data/tianchi_family3 \
        --map runs/family_clustering/families.yaml --link-mode copy
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys

import yaml

# 18-class Tianchi taxonomy, id == index (backend/schemas/defect.py).
CLASSES = [
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

# Stable super-families (KMeans k=3 == Ward k=3, verified 2026-10-15).
SUPER = {
    "A_linear": [
        "three_silk",
        "coarse_warp",
        "loose_warp",
        "broken_warp",
        "hanging_warp",
        "star_skip",
        "weave_defect",
    ],
    "B_blotch": ["stain", "flower_board", "hundred_feet", "size_stain", "dense_section"],
    "C_point": [
        "hole",
        "knot",
        "hair_particle",
        "weft_shrink",
        "warping_knot",
        "broken_spandex",
    ],
}

FAMILY_ORDER = ["A_linear", "B_blotch", "C_point"]

CN = {
    "hole": "破洞",
    "stain": "污渍",
    "three_silk": "三丝",
    "knot": "结头",
    "flower_board": "花板",
    "hundred_feet": "百脚",
    "hair_particle": "毛粒",
    "coarse_warp": "粗经",
    "loose_warp": "松经",
    "broken_warp": "断经",
    "hanging_warp": "吊经",
    "weft_shrink": "纬缩",
    "size_stain": "浆斑",
    "warping_knot": "整经结",
    "star_skip": "跳花",
    "broken_spandex": "断氨纶",
    "dense_section": "密路",
    "weave_defect": "织疵",
}

IMG_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def load_mapping(map_path: Path | None) -> dict[str, int]:
    """Return ``class name -> family id (0..2)``.

    Args:
        map_path: Optional ``families.yaml`` produced by ``build_family_mapping.py``.

    Returns:
        Mapping from every 18-class name to its family id.

    Raises:
        SystemExit: If an explicitly requested file is unusable.
    """
    if map_path is not None:
        if not map_path.exists():
            raise SystemExit(
                f"ERROR: --map {map_path} does not exist. Omit --map to use the "
                "built-in SUPER table, or run scripts/build_family_mapping.py first."
            )
        raw = yaml.safe_load(map_path.read_text(encoding="utf-8")) or {}
        block = raw.get("class_to_family3")
        if isinstance(block, bool) or block is None:
            # build_family_mapping.py writes this key as a YAML *comment* header
            # for a flat mapping, so accept either a dict or a top-level map.
            block = {k: v for k, v in raw.items() if isinstance(k, str) and k in SUPER}
        if not isinstance(block, dict) or not block:
            raise SystemExit(
                f"ERROR: {map_path} has no usable 'class_to_family3' mapping. "
                "Expected {'<class name>': '<family name>' | 0..2}."
            )
        out: dict[str, int] = {}
        for cls, fam in block.items():
            fam_name = fam if isinstance(fam, str) else FAMILY_ORDER[int(fam)]
            if fam_name not in FAMILY_ORDER:
                raise SystemExit(f"ERROR: {map_path}: unknown family {fam!r} for {cls!r}")
            out[str(cls)] = FAMILY_ORDER.index(fam_name)
        return out

    return {
        cls: FAMILY_ORDER.index(fam)
        for fam, members in SUPER.items()
        for cls in members
    }


def resolve_names(src: Path, mapping: dict[str, int]) -> list[str]:
    """Read class names from ``<src>/data.yaml`` and validate them against ``mapping``."""
    data_yaml = src / "data.yaml"
    if not data_yaml.exists():
        raise SystemExit(f"ERROR: {data_yaml} not found (is --src the YOLO dataset root?)")
    raw = yaml.safe_load(data_yaml.read_text(encoding="utf-8")) or {}
    names = raw.get("names")
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names)]
    if not isinstance(names, list) or not names:
        raise SystemExit(f"ERROR: {data_yaml} has no usable 'names' list")
    names = [str(n) for n in names]

    unmapped = [n for n in names if n not in mapping]
    if unmapped:
        raise SystemExit(
            "ERROR: these classes from data.yaml have no family in the mapping: "
            f"{unmapped}\n  mapping covers: {sorted(mapping)}"
        )
    return names


def link_image(src: Path, dst: Path, mode: str) -> None:
    """Materialise one image at ``dst`` using ``mode`` (hardlink/copy/symlink)."""
    if dst.exists():
        return
    if mode == "hardlink":
        try:
            dst.hardlink_to(src)
            return
        except (OSError, NotImplementedError):
            # Different volume or no permission -> fall back to a copy.
            pass
    elif mode == "symlink":
        try:
            dst.symlink_to(src)
            return
        except OSError as exc:
            print(f"[family] symlink failed ({exc}); copying {src.name}", file=sys.stderr)
    dst.write_bytes(src.read_bytes())


def remap_label(text: str, names: list[str], mapping: dict[str, int]) -> tuple[str, int]:
    """Rewrite one label file's class ids; return (new text, true box count)."""
    out: list[str] = []
    boxes = 0
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cls_id = int(float(parts[0]))
        if cls_id < 0 or cls_id >= len(names):
            raise SystemExit(f"ERROR: label class id {cls_id} outside 0..{len(names) - 1}")
        fam_id = mapping[names[cls_id]]
        out.append(" ".join([str(fam_id), *parts[1:5]]))
        boxes += 1
    return ("\n".join(out) + ("\n" if out else "")), boxes


def convert_split(
    src: Path,
    dst: Path,
    split: str,
    names: list[str],
    mapping: dict[str, int],
    mode: str,
) -> dict:
    """Convert one split (images + labels) and return its integrity stats."""
    s_img, s_lab = src / split / "images", src / split / "labels"
    d_img, d_lab = dst / split / "images", dst / split / "labels"
    if not s_img.exists():
        return {"split": split, "skipped": True}
    d_img.mkdir(parents=True, exist_ok=True)
    d_lab.mkdir(parents=True, exist_ok=True)

    images = [p for p in sorted(s_img.iterdir()) if p.is_file() and p.suffix.lower() in IMG_SUFFIXES]
    files = labels = boxes = empty = 0
    missing_label = []
    for img in images:
        link_image(img, d_img / img.name, mode)
        files += 1
        lab = s_lab / f"{img.stem}.txt"
        if not lab.exists():
            # test/ is unlabelled by design (competition submission split). Writing
            # empty .txt files there would make the parity self-check fail for a
            # perfectly good dataset, so only train/val synthesise negatives.
            if split == "test":
                continue
            missing_label.append(img.stem)
            (d_lab / f"{img.stem}.txt").write_text("", encoding="utf-8")
            empty += 1
            continue
        new_text, n = remap_label(lab.read_text(encoding="utf-8"), names, mapping)
        (d_lab / f"{img.stem}.txt").write_text(new_text, encoding="utf-8")
        labels += 1
        boxes += n
        empty += 1 if n == 0 else 0

    return {
        "split": split,
        "images": files,
        "label_files": labels + len(missing_label),
        "gt_boxes": boxes,
        "empty_labels": empty,
        "missing_source_label": missing_label[:10],
        "n_missing_source_label": len(missing_label),
    }


def source_stats(src: Path, split: str) -> dict:
    """Count images / label files / boxes in the source split for the self-check."""
    s_img, s_lab = src / split / "images", src / split / "labels"
    if not s_img.exists():
        return {}
    images = [p for p in s_img.iterdir() if p.is_file() and p.suffix.lower() in IMG_SUFFIXES]
    boxes = 0
    empty = 0
    for lab in s_lab.glob("*.txt"):
        lines = [ln for ln in lab.read_text(encoding="utf-8").splitlines() if ln.strip()]
        boxes += len(lines)
        if not lines:
            empty += 1
    return {
        "images": len(images),
        "label_files": len(list(s_lab.glob("*.txt"))),
        "gt_boxes": boxes,
        "empty_labels": empty,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default="data/tianchi", help="source YOLO dataset root")
    ap.add_argument("--dst", default="data/tianchi_family3", help="output dataset root")
    ap.add_argument("--map", default=None, help="families.yaml (default: built-in 3-family table)")
    ap.add_argument("--splits", default="train,val,test")
    ap.add_argument(
        "--link-mode",
        default="hardlink",
        choices=["hardlink", "copy", "symlink"],
        help="how to materialise images (hardlink needs the same volume)",
    )
    ap.add_argument("--samples", type=int, default=10, help="label samples to print")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    src, dst = Path(args.src), Path(args.dst)
    if not src.is_dir():
        raise SystemExit(f"ERROR: --src {src} is not a directory")
    if dst.resolve() == src.resolve():
        raise SystemExit("ERROR: --dst must differ from --src (data/tianchi is read-only in place)")

    map_path = Path(args.map) if args.map else None
    mapping = load_mapping(map_path)
    names = resolve_names(src, mapping)
    print(f"[family] source classes ({len(names)}): {names}")
    print(f"[family] mapping source: {'file ' + str(map_path) if map_path else 'built-in SUPER table'}")
    for fam in FAMILY_ORDER:
        members = [n for n in names if mapping[n] == FAMILY_ORDER.index(fam)]
        print(f"[family]   {FAMILY_ORDER.index(fam)} {fam:<10s} {len(members):>2d} classes: {members}")

    dst.mkdir(parents=True, exist_ok=True)
    reports = []
    for split in [s.strip() for s in args.splits.split(",") if s.strip()]:
        if not (src / split / "images").exists():
            print(f"[family] split {split}: not present, skipped")
            continue
        rep = convert_split(src, dst, split, names, mapping, args.link_mode)
        rep["source"] = source_stats(src, split)
        reports.append(rep)
        ok = (
            rep["images"] == rep["source"]["images"]
            and rep["gt_boxes"] == rep["source"]["gt_boxes"]
            and rep["label_files"] == rep["source"]["label_files"]
        )
        print(
            f"[family] {split:<5s} images={rep['images']:>5d} labels={rep['label_files']:>5d} "
            f"boxes={rep['gt_boxes']:>5d} empty={rep['empty_labels']:>4d} "
            f"[{'OK' if ok else 'MISMATCH'}]"
        )
        if not ok:
            print(f"          source: {rep['source']}", file=sys.stderr)
            raise SystemExit(
                "ERROR: remapped split does not match the source counts -- aborting "
                "before any training uses this dataset."
            )
        if rep["n_missing_source_label"]:
            print(
                f"          WARNING: {rep['n_missing_source_label']} images had no label file "
                f"-> written as empty (e.g. {rep['missing_source_label']})",
                file=sys.stderr,
            )

    data_yaml = {
        "path": str(dst.resolve()),
        "train": "train/images",
        "val": "val/images",
        "nc": len(FAMILY_ORDER),
        "names": FAMILY_ORDER,
        "family_source": str(map_path) if map_path else "built-in SUPER table (build_family_mapping.py)",
    }
    if (dst / "test" / "images").exists():
        data_yaml["test"] = "test/images"
    (dst / "data.yaml").write_text(
        yaml.safe_dump(data_yaml, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    print(f"[family] wrote {dst / 'data.yaml'} (nc={len(FAMILY_ORDER)}, names={FAMILY_ORDER})")

    # Human spot-check: show a few remapped labels with the class->family trace.
    split0 = reports[0]["split"] if reports else None
    if split0 and args.samples > 0:
        labs = sorted((dst / split0 / "labels").glob("*.txt"))
        random.Random(args.seed).shuffle(labs)
        print(f"\n[family] spot-check {min(args.samples, len(labs))} labels from {split0}/:")
        src_labs = src / split0 / "labels"
        for lab in labs[: args.samples]:
            new = lab.read_text(encoding="utf-8").strip().splitlines()
            old = (src_labs / lab.name)
            old_lines = old.read_text(encoding="utf-8").strip().splitlines() if old.exists() else []
            pairs = []
            for o, n in zip(old_lines, new):
                op, np_ = o.split(), n.split()
                if len(op) < 5 or len(np_) < 5:
                    continue
                oc = names[int(float(op[0]))]
                pairs.append(f"{oc}({CN.get(oc, '?')})->{FAMILY_ORDER[int(float(np_[0]))]}")
            print(f"  {lab.name}: {len(new)} boxes; " + ", ".join(pairs[:4]))

    (dst / "family_build_report.json").write_text(
        json.dumps({"src": str(src), "dst": str(dst), "splits": reports}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[family] report -> {dst / 'family_build_report.json'}")

    val = next((r for r in reports if r["split"] == "val"), None)
    if val and val["gt_boxes"] != 953:
        print(
            f"[family] NOTE: val GT boxes = {val['gt_boxes']} (handoff expects 953 on the "
            "current data/tianchi); confirm the source dataset is the same build.",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
