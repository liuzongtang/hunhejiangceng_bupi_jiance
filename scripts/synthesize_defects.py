"""
Synthesize rare-class defects by copy-paste (Simple Copy-Paste / InstaBoost lineage).

Targets the "missed" critical class — broken_spandex (断氨纶, 18-class id 15) — by
cutting defect patches from source train images, pasting them onto clean fabric
backgrounds (AITEX ``NODefect_images`` + clean crops), and writing matching YOLO
labels. A small gallery montage is saved for visual inspection.

This is the #7 improvement ("生成合成数据补断氨纶这类漏检"). A first simple
sample: default ``--num 16``, no detector needed.

Usage:
    python scripts/synthesize_defects.py                     # 16 samples, class 15
    python scripts/synthesize_defects.py --class-id 9 --num 32
    python scripts/synthesize_defects.py --dry-run           # report sources only
"""

from __future__ import annotations

import argparse
from pathlib import Path
import random
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ROOT = Path("data/tianchi")
AITEX = Path("shujuji_AITEX/NODefect_images")
OUT = ROOT / "synth"

# 18-class taxonomy (id 0..17), matching data/tianchi/data.yaml after the merge.
CLASS_NAMES = [
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


def parse_yolo(path: Path) -> list:
    """Read a YOLO label file into ``[(cls, cx, cy, w, h)]`` normalized boxes."""
    out = []
    for ln in path.read_text().splitlines():
        if not ln.strip():
            continue
        p = ln.split()
        out.append(
            (int(float(p[0])), float(p[1]), float(p[2]), float(p[3]), float(p[4]))
        )
    return out


def load_sources(class_id: int, limit: int) -> list:
    """Return ``[(image_path, [boxes of class_id])]`` for train images containing it."""
    sources = []
    labels_dir = ROOT / "train" / "labels"
    for lbl in sorted(labels_dir.glob("*.txt")):
        hit = [b for b in parse_yolo(lbl) if b[0] == class_id]
        if not hit:
            continue
        img = ROOT / "train" / "images" / (lbl.stem + ".jpg")
        if img.exists():
            sources.append((img, hit))
        if limit and len(sources) >= limit:
            break
    return sources


def load_backgrounds() -> list:
    """Return AITEX clean-fabric image paths (recursively, under per-roll subdirs)."""
    return sorted(
        p
        for p in AITEX.rglob("*")
        if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")
    )


def load_all_train(limit: int) -> list:
    """Return ``[(image_path, all_boxes)]`` for a sample of train images."""
    out = []
    for lbl in sorted((ROOT / "train" / "labels").glob("*.txt")):
        boxes = parse_yolo(lbl)
        if not boxes:
            continue
        img = ROOT / "train" / "images" / (lbl.stem + ".jpg")
        if img.exists():
            out.append((img, boxes))
        if limit and len(out) >= limit:
            break
    return out


def crop_overlaps(
    boxes: list, x0: int, y0: int, x1: int, y1: int, W: int, H: int
) -> bool:
    """True if any box intersects the pixel rectangle ``(x0,y0)-(x1,y1)``."""
    for _, cx, cy, w, h in boxes:
        bx0 = (cx - w / 2) * W
        by0 = (cy - h / 2) * H
        bx1 = (cx + w / 2) * W
        by1 = (cy + h / 2) * H
        if min(x1, bx1) > max(x0, bx0) and min(y1, by1) > max(y0, by0):
            return True
    return False


def tianchi_clean_crop(train: list, rng: random.Random, side: int) -> np.ndarray | None:
    """Crop a defect-free ``side``x``side`` region from a Tianchi train image.

    Texture-matched backgrounds make the copy-paste evaluation fair — pasting a
    Tianchi defect onto foreign AITEX fabric confounds the detector check.
    """
    for _ in range(50):
        img_path, boxes = rng.choice(train)
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        H, W = img.shape[:2]
        if side > W or side > H:
            continue
        x0 = rng.randint(0, W - side)
        y0 = rng.randint(0, H - side)
        if not crop_overlaps(boxes, x0, y0, x0 + side, y0 + side, W, H):
            return img[y0 : y0 + side, x0 : x0 + side]
    return None


def crop_patch(img: np.ndarray, box: tuple, pad_ratio: float, rng: random.Random):
    """Crop the defect region with a margin, returning the patch and its foreground mask."""
    _, cx, cy, w, h = box
    H, W = img.shape[:2]
    # Add padding around the tight box so the patch has fabric context.
    pad = pad_ratio * max(w * W, h * H)
    x0 = int((cx - w / 2) * W - pad)
    y0 = int((cy - h / 2) * H - pad)
    x1 = int((cx + w / 2) * W + pad)
    y1 = int((cy + h / 2) * H + pad)
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(W, x1), min(H, y1)
    patch = img[y0:y1, x0:x1]
    mask = defect_mask(patch)
    return patch, mask


def defect_mask(patch: np.ndarray) -> np.ndarray:
    """Foreground mask. broken_spandex is too low-contrast for reliable Otsu
    segmentation (a 55x29 px thin line on dark fabric), so we paste the whole
    tight box — the original Simple Copy-Paste style. The box is mostly clean
    fabric plus a thin line, so the paste reads as a small, realistic defect."""
    return np.full(patch.shape[:2], 255, np.uint8)


def augment(patch: np.ndarray, mask: np.ndarray, rng: random.Random):
    """Random small-rotation/flip/scale, keeping the mask aligned with the patch.

    Thin defects (broken_spandex) are rotated only ±20° so the pasted box stays
    thin. The rotation uses an enlarged canvas so the patch is never clipped.
    """
    h, w = patch.shape[:2]
    ang = rng.uniform(-20, 20)
    rad = np.deg2rad(ang)
    nw = int(np.ceil(w * abs(np.cos(rad)) + h * abs(np.sin(rad))))
    nh = int(np.ceil(w * abs(np.sin(rad)) + h * abs(np.cos(rad))))
    M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0)
    M[0, 2] += (nw - w) / 2
    M[1, 2] += (nh - h) / 2
    patch = cv2.warpAffine(patch, M, (nw, nh), flags=cv2.INTER_LINEAR, borderValue=0)
    mask = cv2.warpAffine(mask, M, (nw, nh), flags=cv2.INTER_NEAREST, borderValue=0)
    if rng.random() < 0.5:
        patch, mask = cv2.flip(patch, 1), cv2.flip(mask, 1)
    s = rng.uniform(0.8, 1.4)
    nh, nw = max(2, int(round(nh * s))), max(2, int(round(nw * s)))
    patch = cv2.resize(patch, (nw, nh))
    mask = cv2.resize(mask, (nw, nh), interpolation=cv2.INTER_NEAREST)
    return patch, mask


def paste(bg: np.ndarray, patch: np.ndarray, mask: np.ndarray, rng: random.Random):
    """Paste the patch into ``bg`` via seamlessClone; return image + tight bbox (x,y,w,h)."""
    ph, pw = patch.shape[:2]
    bh, bw = bg.shape[:2]
    if pw >= bw - 4 or ph >= bh - 4:  # shrink to fit
        scale = min((bw - 8) / pw, (bh - 8) / ph)
        nw, nh = int(pw * scale), int(ph * scale)
        patch = cv2.resize(patch, (nw, nh))
        mask = cv2.resize(mask, (nw, nh), interpolation=cv2.INTER_NEAREST)
        ph, pw = patch.shape[:2]
    x = rng.randint(0, bw - pw - 1)
    y = rng.randint(0, bh - ph - 1)
    center = (x + pw // 2, y + ph // 2)
    try:
        out = cv2.seamlessClone(patch, bg, mask, center, cv2.NORMAL_CLONE)
    except cv2.error:
        out = bg.copy()
        roi = out[y : y + ph, x : x + pw]
        alpha = (mask > 0).astype(np.float32)[..., None]
        roi[:] = (patch * alpha + roi * (1 - alpha)).astype(np.uint8)
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        bx, by, bw_, bh_ = x, y, pw, ph
    else:
        bx, by = x + int(xs.min()), y + int(ys.min())
        bw_, bh_ = int(xs.max()) - int(xs.min()) + 1, int(ys.max()) - int(ys.min()) + 1
    return out, (bx, by, bw_, bh_)


def prep_background(p: Path, side: int, rng: random.Random) -> np.ndarray:
    """Resize clean fabric so its short side == ``side``, then random-crop a square."""
    img = cv2.imread(str(p))
    if img is None:
        return None
    if img.ndim == 2:  # AITEX negatives are 8-bit grayscale PNG
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    h, w = img.shape[:2]
    scale = side / min(h, w)
    img = cv2.resize(img, (int(w * scale), int(h * scale)))
    h, w = img.shape[:2]
    if h > side:  # crop height
        y = rng.randint(0, h - side)
        img = img[y : y + side]
        h = side
    if w > side:  # crop width
        x = rng.randint(0, w - side)
        img = img[:, x : x + side]
        w = side
    return img


def montage(images: list, out_path: Path, cols: int = 4, cell: int = 256) -> None:
    """Stack images into a grid thumbnail and write it."""
    rows = (len(images) + cols - 1) // cols
    tiles = []
    for img in images:
        t = cv2.resize(img, (cell, cell))
        tiles.append(
            cv2.copyMakeBorder(
                t, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=(255, 255, 255)
            )
        )
    canvas = np.full((rows * (cell + 4), cols * (cell + 4), 3), 255, np.uint8)
    for i, t in enumerate(tiles):
        r, c = divmod(i, cols)
        canvas[
            r * (cell + 4) : r * (cell + 4) + cell + 4,
            c * (cell + 4) : c * (cell + 4) + cell + 4,
        ] = t
    out_path.write_bytes(cv2.imencode(".png", canvas)[1].tobytes())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--class-id", type=int, default=15, help="18-class defect id to synthesize"
    )
    parser.add_argument(
        "--num", type=int, default=16, help="number of synthetic images"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--side", type=int, default=512, help="background square side (px)"
    )
    parser.add_argument("--defects-per-img", type=int, default=2)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    cname = CLASS_NAMES[args.class_id]

    sources = load_sources(args.class_id, limit=64)
    train = load_all_train(limit=256)
    aitex = load_backgrounds()
    print(
        f"[synth] target class {args.class_id} ({cname}): {len(sources)} source images, "
        f"{len(train)} train images for clean crops, {len(aitex)} AITEX negatives"
    )
    if not sources or not train:
        print("[synth] abort: need >=1 source image and >=1 train image")
        return 1
    if args.dry_run:
        return 0

    out_img = OUT / "images"
    out_lbl = OUT / "labels"
    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)

    gallery = []
    for i in range(args.num):
        src_img, boxes = rng.choice(sources)
        src = cv2.imread(str(src_img))
        if src is None:
            continue
        # Texture-matched Tianchi clean crop first, AITEX as a last resort.
        bg = tianchi_clean_crop(train, rng, args.side)
        if bg is None and aitex:
            bg = prep_background(rng.choice(aitex), args.side, rng)
        if bg is None:
            continue
        H, W = bg.shape[:2]
        label_lines = []
        for _ in range(args.defects_per_img):
            patch, mask = crop_patch(src, rng.choice(boxes), pad_ratio=0.15, rng=rng)
            patch, mask = augment(patch, mask, rng)
            bg, (bx, by, bw_, bh_) = paste(bg, patch, mask, rng)
            cx = (bx + bw_ / 2) / W
            cy = (by + bh_ / 2) / H
            nw, nh = bw_ / W, bh_ / H
            label_lines.append(f"{args.class_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

        stem = f"synth_{cname}_{i:03d}"
        cv2.imwrite(str(out_img / f"{stem}.jpg"), bg, [cv2.IMWRITE_JPEG_QUALITY, 95])
        (out_lbl / f"{stem}.txt").write_text("\n".join(label_lines) + "\n")
        gallery.append(bg)

    montage_path = OUT / "montage.png"
    montage(gallery, montage_path)
    print(f"[synth] wrote {len(gallery)} images + labels -> {OUT}")
    print(f"[synth] montage -> {montage_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
