"""
GT-box classification ceiling — decouple classification from localization.

Measures how well an 18-class classifier does *given a perfect box* (the GT
box), i.e. the classification ceiling of the current labels. This is the
diagnostic that decides whether the remaining error is (a) label noise /
irrecoverable visual confusion → re-label, or (b) feature discriminability →
stronger backbone / P2 / direction-sensitive heads.

A frozen ResNet-18 backbone + linear head is used (mirroring the objectness
probe), so this measures the *feature* ceiling, not how much a full fine-tune
can overfit. Boxes are balanced per class so the number is not dominated by the
long tail.

Usage:
    python scripts/eval_classification_ceiling.py                 # subset (fast)
    python scripts/eval_classification_ceiling.py --max-per-class 0  # all boxes
"""

from __future__ import annotations

import argparse
from pathlib import Path
import random
import sys

import cv2
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ROOT = Path("data/tianchi")
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
N_CLASSES = len(CLASS_NAMES)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def parse_yolo(path: Path) -> list:
    """Read a YOLO label into ``[(cls, cx, cy, w, h)]``."""
    out = []
    for ln in path.read_text().splitlines():
        if not ln.strip():
            continue
        p = ln.split()
        out.append(
            (int(float(p[0])), float(p[1]), float(p[2]), float(p[3]), float(p[4]))
        )
    return out


def crop_patch(
    img: np.ndarray, box: tuple, size: int, pad_ratio: float = 0.2
) -> np.ndarray:
    """Crop a padded box region, resize preserving aspect, pad to ``size`` (RGB).

    Warp-direction defects (断经/稀密档/松经) are long and thin; squashing them
    to a square destroys the direction cue. Preserve aspect so the classifier
    sees the real geometry, padding with the patch's mean color.
    """
    _, cx, cy, w, h = box
    H, W = img.shape[:2]
    pad = pad_ratio * max(w * W, h * H)
    x0 = int((cx - w / 2) * W - pad)
    y0 = int((cy - h / 2) * H - pad)
    x1 = int((cx + w / 2) * W + pad)
    y1 = int((cy + h / 2) * H + pad)
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(W, x1), min(H, y1)
    if x1 <= x0 or y1 <= y0:  # degenerate box; fall back to full image
        x0, y0, x1, y1 = 0, 0, W, H
    patch = img[y0:y1, x0:x1]
    ph, pw = patch.shape[:2]
    scale = size / max(ph, pw)
    patch = cv2.resize(
        patch, (max(1, int(round(pw * scale))), max(1, int(round(ph * scale))))
    )
    nh, nw = patch.shape[:2]
    canvas = np.full(
        (size, size, 3),
        patch.reshape(-1, 3).mean(axis=0).astype(np.uint8),
        dtype=np.uint8,
    )
    y = (size - nh) // 2
    x = (size - nw) // 2
    canvas[y : y + nh, x : x + nw] = patch
    return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)


def collect(split: str, max_per_class: int, size: int, rng: random.Random):
    """Collect balanced ``(patch_rgb_uint8, cls)`` pairs for a split."""
    labels_dir = ROOT / split / "labels"
    images_dir = ROOT / split / "images"
    by_class = {c: [] for c in range(N_CLASSES)}
    for lbl in sorted(labels_dir.glob("*.txt")):
        img_path = images_dir / (lbl.stem + ".jpg")
        if not img_path.exists():
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        for box in parse_yolo(lbl):
            cls = box[0]
            if max_per_class and len(by_class[cls]) >= max_per_class:
                continue
            by_class[cls].append((crop_patch(img, box, size), cls))
    patches, labels = [], []
    for cls in range(N_CLASSES):
        n = len(by_class[cls])
        take = n if not max_per_class else min(n, max_per_class)
        for patch, _ in by_class[cls][:take]:
            patches.append(patch)
            labels.append(cls)
    print(
        f"[ceiling] {split}: {len(patches)} boxes "
        f"(per-class {min(len(v) for v in by_class.values())}..{max(len(v) for v in by_class.values())})"
    )
    return patches, labels


def to_tensor(patches, labels, device):
    """Normalize uint8 RGB patches to an ImageNet-normalized float tensor."""
    arr = np.stack(patches).astype(np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    x = torch.from_numpy(arr).permute(0, 3, 1, 2).to(device)
    y = torch.tensor(labels, dtype=torch.long, device=device)
    return x, y


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-per-class", type=int, default=120)
    parser.add_argument("--size", type=int, default=224)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(
        f"[ceiling] device={device}, size={args.size}, max-per-class={args.max_per_class}"
    )

    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)

    # Frozen ImageNet ResNet-18 backbone.
    try:
        import torchvision

        backbone = torchvision.models.resnet18(
            weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1
        )
    except Exception as e:  # no cached weights / no network -> train from scratch
        import torchvision

        print(f"[ceiling] pretrained weights unavailable ({e}); using random init")
        backbone = torchvision.models.resnet18(weights=None)
    backbone.fc = nn.Identity()
    for p in backbone.parameters():
        p.requires_grad = False
    backbone.to(device).eval()

    head = nn.Linear(512, N_CLASSES).to(device)
    opt = torch.optim.Adam(head.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    train_patches, train_labels = collect("train", args.max_per_class, args.size, rng)
    val_patches, val_labels = collect("val", args.max_per_class, args.size, rng)

    xtr, ytr = to_tensor(train_patches, train_labels, device)
    xva, yva = to_tensor(val_patches, val_labels, device)

    # Precompute frozen features once (cheap, deterministic).
    def features(x):
        feats = []
        for i in range(0, x.shape[0], args.batch):
            with torch.no_grad():
                feats.append(backbone(x[i : i + args.batch]))
        return torch.cat(feats)

    ftr, fva = features(xtr), features(xva)

    for epoch in range(args.epochs):
        head.train()
        idx = torch.randperm(ftr.shape[0], device=device)
        total, correct = 0, 0
        for i in range(0, ftr.shape[0], args.batch):
            bi = idx[i : i + args.batch]
            logits = head(ftr[bi])
            loss = loss_fn(logits, ytr[bi])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += bi.shape[0]
            correct += (logits.argmax(1) == ytr[bi]).sum().item()
        acc = correct / total
        print(f"[ceiling] epoch {epoch + 1}/{args.epochs} train-top1 {acc:.3f}")

    # Evaluate on val.
    head.eval()
    with torch.no_grad():
        logits = head(fva)
        pred = logits.argmax(1).cpu().numpy()
    gt = yva.cpu().numpy()
    top1 = float((pred == gt).mean())

    # Per-class recall and confusion matrix.
    per_class = {}
    for c in range(N_CLASSES):
        m = gt == c
        if m.sum():
            per_class[c] = float((pred[m] == c).mean())
    conf = np.zeros((N_CLASSES, N_CLASSES), dtype=int)
    for g, p in zip(gt, pred, strict=False):
        conf[g, p] += 1

    # Most-confused off-diagonal pairs (each pair counted once).
    pairs = []
    for a in range(N_CLASSES):
        for b in range(a + 1, N_CLASSES):
            pairs.append((conf[a, b] + conf[b, a], a, b))
    pairs.sort(reverse=True)

    print("\n[ceiling] val 18-class Top-1 (perfect box) = {:.3f}".format(top1))
    print("[ceiling] per-class recall:")
    for c in range(N_CLASSES):
        print(f"  {c:2d} {CLASS_NAMES[c]:<16} {per_class.get(c, 0.0):.3f}")
    print("[ceiling] top confused pairs (sum of both directions):")
    for n, a, b in pairs[:8]:
        print(f"  {CLASS_NAMES[a]:<16} <-> {CLASS_NAMES[b]:<16} : {n}")

    out = ROOT / "synth" / "classification_ceiling_confusion.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    header = ",".join(CLASS_NAMES)
    np.savetxt(out, conf, delimiter=",", header=header, fmt="%d", comments="")
    print(f"[ceiling] confusion matrix -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
