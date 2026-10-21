"""Extract frozen-encoder features for all 18-class GT boxes (train or val split).

For each GT box in data/tianchi/<split>/labels/*.txt:
  - crop the defect region from the source image (padding = 15% of box size, clipped to image)
  - resize to 224x224 (stretch; fabric defects are texture patches, aspect kept as meta feature)
  - ResNet50 (ImageNet, frozen) 2048-d penultimate feature, L2-normalized
Also records per-box meta: class id, box area ratio, aspect ratio, absolute px size.

Output: runs/family_clustering/box_features[_<split>].npz
  feats (N,2048) float32 | cls (N,) int64 | meta (N,4) float32
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision.models import ResNet50_Weights, resnet50
import torchvision.transforms as T


def load_boxes(label_dir: Path, img_dir: Path):
    """Return list of (img_path, cls, cx, cy, w, h) in normalized YOLO coords."""
    items = []
    for lp in sorted(label_dir.glob("*.txt")):
        stem = lp.stem
        img_path = None
        for ext in (".jpg", ".jpeg", ".png", ".bmp"):
            cand = img_dir / (stem + ext)
            if cand.exists():
                img_path = cand
                break
        if img_path is None:
            continue
        for line in lp.read_text().splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            items.append((img_path, int(parts[0]), *[float(x) for x in parts[1:5]]))
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/tianchi")
    ap.add_argument("--split", default="train", choices=["train", "val"])
    ap.add_argument(
        "--out",
        default=None,
        help="default: runs/family_clustering/box_features[_<split>].npz",
    )
    ap.add_argument(
        "--pad", type=float, default=0.15, help="crop padding ratio of box size"
    )
    ap.add_argument("--batch", type=int, default=256)
    args = ap.parse_args()

    root = Path(args.root)
    out = (
        args.out
        or f"runs/family_clustering/box_features{'' if args.split == 'train' else '_' + args.split}.npz"
    )
    items = load_boxes(root / args.split / "labels", root / args.split / "images")
    print(f"split={args.split}, total GT boxes: {len(items)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    weights = ResNet50_Weights.IMAGENET1K_V2
    model = resnet50(weights=weights).to(device).eval()
    tf = T.Compose(
        [T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
    )

    feats, cls_ids, metas = [], [], []
    # group items by image to avoid re-reading
    from collections import defaultdict

    by_img = defaultdict(list)
    for it in items:
        by_img[it[0]].append(it)

    done = 0
    with torch.no_grad():
        for img_path, boxes in by_img.items():
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"  [warn] cannot read {img_path}, skip {len(boxes)} boxes")
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            H, W = img.shape[:2]
            crops, cids, cmeta = [], [], []
            for _, cls, cx, cy, bw, bh in boxes:
                x1 = max(0.0, cx - bw / 2 - args.pad * bw)
                y1 = max(0.0, cy - bh / 2 - args.pad * bh)
                x2 = min(1.0, cx + bw / 2 + args.pad * bw)
                y2 = min(1.0, cy + bh / 2 + args.pad * bh)
                if x2 - x1 < 1e-4 or y2 - y1 < 1e-4:
                    continue
                px1, py1, px2, py2 = (
                    int(x1 * W),
                    int(y1 * H),
                    int(np.ceil(x2 * W)),
                    int(np.ceil(y2 * H)),
                )
                crop = img[py1:py2, px1:px2]
                if crop.size == 0:
                    continue
                crop = cv2.resize(crop, (224, 224), interpolation=cv2.INTER_LINEAR)
                crops.append(tf(crop))
                cids.append(cls)
                # meta: area ratio, aspect (w/h), abs px w, abs px h
                cmeta.append([bw * bh, bw / max(bh, 1e-6), bw * W, bh * H])
            if not crops:
                continue
            for i in range(0, len(crops), args.batch):
                batch = torch.stack(crops[i : i + args.batch]).to(device)
                f = model.conv1(batch)
                f = model.bn1(f)
                f = model.relu(f)
                f = model.maxpool(f)
                f = model.layer1(f)
                f = model.layer2(f)
                f = model.layer3(f)
                f = model.layer4(f)
                f = model.avgpool(f).flatten(1)
                f = torch.nn.functional.normalize(f, dim=1)
                feats.append(f.cpu().numpy())
            cls_ids.extend(cids)
            metas.extend(cmeta)
            done += len(cids)
            if done % 2000 < args.batch:
                print(f"  {done}/{len(items)} boxes encoded")

    feats = np.concatenate(feats).astype(np.float32)
    cls_ids = np.array(cls_ids, dtype=np.int64)
    metas = np.array(metas, dtype=np.float32)
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, feats=feats, cls=cls_ids, meta=metas)
    print(
        f"saved {out_path}: feats {feats.shape}, classes {len(set(cls_ids.tolist()))}"
    )
    # per-class box count sanity
    for c in sorted(set(cls_ids.tolist())):
        print(f"  class {c:2d}: {(cls_ids == c).sum()} boxes")


if __name__ == "__main__":
    main()
