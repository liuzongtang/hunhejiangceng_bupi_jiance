"""0.3b — Fine-tune a 3-family coarse classifier on GT box crops.

Answers the plan's 0.3 question with real numbers: "can the coarse families reach 90%+?"

Setup (crop protocol identical to 0.4 for comparability):
  - crops: pad 15% of box size, 224x224 stretch, ImageNet normalize
  - train: 8570 GT boxes (data/tianchi/train), val: 953 GT boxes (data/tianchi/val)
  - labels: 18-class -> 3 families via runs/family_clustering/families.yaml (class_to_family3)
  - model: ResNet50 (ImageNet V2) NON-frozen fine-tune, 3-way head
  - loss: class-weighted CE (inverse frequency); aug: hflip + vflip only
    (90-degree rotations would swap warp/weft semantics — deliberately excluded)
  - baseline: logistic regression on the SAME frozen 2048-d features (0.4 pipeline)

Outputs (runs/family_clustering/):
  family_cls_best.pt          best val-acc weights
  family_cls_results.json     metrics: acc, per-family P/R/F1, confusion, bootstrap CI, baseline
  family_cls_curves.png       train/val acc + loss curves
"""

import json
from pathlib import Path
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import ResNet50_Weights, resnet50
import yaml

ROOT = Path("data/tianchi")
OUT = Path("runs/family_clustering")
FAMILIES = ["A_linear", "B_blotch", "C_point"]
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


def family_map():
    cfg = yaml.safe_load((OUT / "families.yaml").read_text(encoding="utf-8"))
    c2f = cfg["class_to_family3"]
    return {c: FAMILIES.index(f) for c, f in c2f.items()}


def load_boxes(split, c2f):
    items = []
    for lp in sorted((ROOT / split / "labels").glob("*.txt")):
        img = None
        for ext in (".jpg", ".jpeg", ".png", ".bmp"):
            cand = ROOT / split / "images" / (lp.stem + ext)
            if cand.exists():
                img = cand
                break
        if img is None:
            continue
        for line in lp.read_text().splitlines():
            p = line.split()
            if len(p) < 5:
                continue
            items.append(
                (
                    img,
                    c2f[CLASSES[int(p[0])]],
                    float(p[1]),
                    float(p[2]),
                    float(p[3]),
                    float(p[4]),
                )
            )
    return items


class BoxCropDataset(Dataset):
    def __init__(self, items, train=False, pad=0.15):
        self.items = items
        self.train = train
        self.pad = pad
        self.geom = transforms.RandomHorizontalFlip(p=0.5) if train else None
        self.geom2 = transforms.RandomVerticalFlip(p=0.5) if train else None
        self.norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        img_path, fam, cx, cy, bw, bh = self.items[i]
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        H, W = img.shape[:2]
        x1 = max(0.0, cx - bw / 2 - self.pad * bw)
        y1 = max(0.0, cy - bh / 2 - self.pad * bh)
        x2 = min(1.0, cx + bw / 2 + self.pad * bw)
        y2 = min(1.0, cy + bh / 2 + self.pad * bh)
        crop = img[
            int(y1 * H) : int(np.ceil(y2 * H)), int(x1 * W) : int(np.ceil(x2 * W))
        ]
        if crop.size == 0:
            crop = np.zeros((224, 224, 3), dtype=np.uint8)
        crop = cv2.resize(crop, (224, 224), interpolation=cv2.INTER_LINEAR)
        t = torch.from_numpy(crop).permute(2, 0, 1).float() / 255.0
        if self.geom is not None:
            t = self.geom2(self.geom(t))
        return self.norm(t), fam


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    preds, gts = [], []
    for x, y in loader:
        p = model(x.to(device)).argmax(1).cpu().numpy()
        preds.append(p)
        gts.append(y.numpy())
    return np.concatenate(preds), np.concatenate(gts)


def metrics(pred, gt):
    acc = float((pred == gt).mean())
    cm = np.zeros((3, 3), dtype=int)
    for g, p in zip(gt, pred, strict=False):
        cm[g, p] += 1
    per = {}
    for f, name in enumerate(FAMILIES):
        tp = int(cm[f, f])
        fp = int(cm[:, f].sum() - tp)
        fn = int(cm[f, :].sum() - tp)
        per[name] = {
            "recall": round(tp / max(tp + fn, 1), 4),
            "precision": round(tp / max(tp + fp, 1), 4),
            "f1": round(2 * tp / max(2 * tp + fp + fn, 1), 4),
            "support": int(cm[f, :].sum()),
        }
    # bootstrap 95% CI on accuracy (resample indices once, apply to both pred & gt)
    rng = np.random.default_rng(0)
    n = len(gt)
    accs = []
    for _ in range(1000):
        idx = rng.integers(0, n, n)
        accs.append((pred[idx] == gt[idx]).mean())
    return (
        acc,
        per,
        cm.tolist(),
        [
            round(float(np.percentile(accs, 2.5)), 4),
            round(float(np.percentile(accs, 97.5)), 4),
        ],
    )


def main():
    c2f = family_map()
    train_items = load_boxes("train", c2f)
    val_items = load_boxes("val", c2f)
    print(f"train boxes: {len(train_items)}, val boxes: {len(val_items)}")
    counts = np.bincount([it[1] for it in train_items], minlength=3)
    print(f"train family counts: {dict(zip(FAMILIES, counts.tolist(), strict=False))}")

    device = "cuda"
    train_ds = BoxCropDataset(train_items, train=True)
    val_ds = BoxCropDataset(val_items, train=False)
    train_dl = DataLoader(
        train_ds,
        batch_size=128,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )
    val_dl = DataLoader(
        val_ds, batch_size=256, shuffle=False, num_workers=4, pin_memory=True
    )

    weights = ResNet50_Weights.IMAGENET1K_V2
    model = resnet50(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, 3)
    model = model.to(device)

    w = counts.sum() / (3 * counts.astype(float))
    crit = nn.CrossEntropyLoss(
        weight=torch.tensor(w, dtype=torch.float32, device=device)
    )
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=1e-4, total_steps=25 * len(train_dl), pct_start=0.08
    )
    scaler = torch.amp.GradScaler("cuda")

    hist, best_acc, best_state = [], 0.0, None
    t0 = time.time()
    for ep in range(25):
        model.train()
        tot, nb = 0.0, 0
        for x, y in train_dl:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast("cuda"):
                loss = crit(model(x), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += loss.item()
            nb += 1
        pred, gt = evaluate(model, val_dl, device)
        acc = float((pred == gt).mean())
        hist.append(
            {"epoch": ep + 1, "loss": round(tot / nb, 4), "val_acc": round(acc, 4)}
        )
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        print(
            f"epoch {ep + 1:2d}/25  loss {tot / nb:.4f}  val_acc {acc:.4f}  best {best_acc:.4f}  ({time.time() - t0:.0f}s)"
        )

    model.load_state_dict(best_state)
    torch.save(model.state_dict(), OUT / "family_cls_best.pt")
    pred, gt = evaluate(model, val_dl, device)
    acc, per, cm, ci = metrics(pred, gt)
    print(f"\nFINE-TUNED (best val): top-1 acc = {acc:.4f}  95% CI {ci}")
    for f, m in per.items():
        print(
            f"  {f:9s} recall {m['recall']:.3f}  precision {m['precision']:.3f}  support {m['support']}"
        )
    print("  confusion (rows=GT):")
    for i, row in enumerate(cm):
        print(f"    {FAMILIES[i]:9s} {row}")

    # ---- baseline: logistic regression on frozen features ----
    from sklearn.linear_model import LogisticRegression

    dtr = np.load(OUT / "box_features.npz")
    dva = np.load(OUT / "box_features_val.npz")
    Xtr, ytr = dtr["feats"], np.array([c2f[CLASSES[c]] for c in dtr["cls"]])
    Xva, yva = dva["feats"], np.array([c2f[CLASSES[c]] for c in dva["cls"]])
    lr = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced").fit(Xtr, ytr)
    pva = lr.predict(Xva)
    acc_lr, per_lr, cm_lr, ci_lr = metrics(pva, yva)
    print(f"\nFROZEN linear-probe baseline: top-1 acc = {acc_lr:.4f}  95% CI {ci_lr}")
    for f, m in per_lr.items():
        print(f"  {f:9s} recall {m['recall']:.3f}  precision {m['precision']:.3f}")

    results = {
        "date": "2026-10-15",
        "crop_protocol": "pad 15%, 224x224 stretch, ImageNet normalize (identical to 0.4)",
        "n_train": len(train_items),
        "n_val": len(val_items),
        "train_family_counts": dict(zip(FAMILIES, counts.tolist(), strict=False)),
        "finetuned": {
            "top1_acc": round(acc, 4),
            "ci95": ci,
            "per_family": per,
            "confusion": cm,
        },
        "frozen_linear_probe": {
            "top1_acc": round(acc_lr, 4),
            "ci95": ci_lr,
            "per_family": per_lr,
            "confusion": cm_lr,
        },
        "history": hist,
        "notes": "aug = hflip+vflip only (90deg rotation swaps warp/weft semantics); class-weighted CE; 25 epochs AdamW 1e-4 + OneCycle",
    }
    (OUT / "family_cls_results.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )

    # curves
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4), dpi=150)
    a1.plot([h["loss"] for h in hist], "o-", ms=3)
    a1.set_title("train loss")
    a1.set_xlabel("epoch")
    a2.plot([h["val_acc"] for h in hist], "o-", ms=3, label="fine-tuned")
    a2.axhline(
        acc_lr, color="gray", ls="--", lw=1, label=f"frozen linear probe ({acc_lr:.3f})"
    )
    a2.set_title("val top-1 accuracy")
    a2.set_xlabel("epoch")
    a2.legend()
    fig.tight_layout()
    fig.savefig(OUT / "family_cls_curves.png")
    print(
        f"\nsaved {OUT / 'family_cls_results.json'}, family_cls_best.pt, family_cls_curves.png"
    )


if __name__ == "__main__":
    main()
