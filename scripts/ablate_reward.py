"""
A/B ablation: does the 7-dimension hybrid reward improve critical-defect recall?

Trains the same single-box defect detector twice on identical data and RNG
seed, differing only in the reward coefficients:

    A (baseline): beta=0.0, gamma=0.0   -> pure detection loss (CE + box reg)
    B (reward):   beta=0.1, gamma=0.05  -> detection + scalar reward + consistency

Both runs are evaluated on the val split for:
  - overall top-1 accuracy
  - per-class recall
  - group recall on critical {broken_warp=9, broken_spandex=16} and
    major skip {star_skip=15, weave_defect=19}

If the reward mechanism genuinely biases the model toward critical/major
classes, run B must beat run A on the broken/skip group recalls.

A gradient diagnostic is run before training to check whether the scalar
reward loss actually produces gradients with respect to the *model* (not just
the learnable dimension weights).

Usage:
    python scripts/ablate_reward.py --data data/tianchi --epochs 5 --seed 0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from backend.training.dataset import (
    TIANCHI_BROKEN_CLASS_IDS,
    TIANCHI_SKIP_CLASS_IDS,
    FabricDataset,
    auto_discover_dataset,
    defect_collate_fn,
)
from backend.training.hybrid_reward_trainer import HybridRewardTrainer
from backend.training.loss_functions import total_loss
from backend.training.model import SimpleDefectDetector

# Augmentation is disabled for a clean, reproducible A/B comparison.
_NO_AUGMENTATION = {
    "random_flip": 0.0,
    "random_rotate": 0,
    "color_jitter": [0.0, 0.0, 0.0],
    "mosaic": 0.0,
    "mixup": 0.0,
}


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and PyTorch (CPU + CUDA) for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_datasets(config) -> Tuple[FabricDataset, FabricDataset]:
    """Build train/val datasets from a DatasetConfig."""
    train_dataset = FabricDataset(
        image_dir=config.train_images,
        label_dir=config.train_labels,
        image_size=config.image_size,
        format=config.format,
        class_names=config.class_names,
        augment=False,
    )
    val_dataset = FabricDataset(
        image_dir=config.val_images,
        label_dir=config.val_labels,
        image_size=config.image_size,
        format=config.format,
        class_names=config.class_names,
        augment=False,
    )
    return train_dataset, val_dataset


def diagnose_reward_gradient(
    trainer: HybridRewardTrainer,
    images: torch.Tensor,
    targets: List[Dict],
) -> Tuple[int, int]:
    """
    Count how many model parameters receive a gradient from the scalar reward
    loss. Returns (params_with_reward_grad, total_trainable_params).
    """
    images = images.to(trainer.device)
    trainer.model.train()
    preds = trainer.model(images)
    preds = trainer._normalize_predictions(preds)
    dim_rewards = trainer.reward_computer.compute_all(preds, targets)
    losses = total_loss(
        preds,
        targets,
        dim_rewards,
        trainer.dim_weights,
        trainer.beta,
        trainer.gamma,
        trainer.alpha,
    )
    params = [p for p in trainer.model.parameters() if p.requires_grad]
    grads = torch.autograd.grad(
        losses["scalar"], params, allow_unused=True, retain_graph=True
    )
    n_nonzero = sum(1 for g in grads if g is not None and g.abs().sum().item() > 0.0)
    return n_nonzero, len(params)


def run_training(
    train_dataset: FabricDataset,
    val_dataset: FabricDataset,
    config,
    *,
    beta: float,
    gamma: float,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
    device: str,
    num_classes: int,
) -> Tuple[SimpleDefectDetector, HybridRewardTrainer]:
    """Train one configuration (baseline or reward) from scratch."""
    set_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        collate_fn=defect_collate_fn,
        generator=generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=defect_collate_fn,
    )

    model = SimpleDefectDetector(num_classes=num_classes, image_size=config.image_size)
    trainer = HybridRewardTrainer(
        model,
        train_loader,
        val_loader,
        learning_rate=lr,
        beta=beta,
        gamma=gamma,
        num_classes=num_classes,
        device=device,
        augmentation_config=_NO_AUGMENTATION,
    )

    # Gradient diagnostic on the first non-empty batch.
    images, targets = next(iter(train_loader))
    if images.numel() == 0:
        images, targets = next(iter(train_loader))
    n_grad, n_total = diagnose_reward_gradient(trainer, images, targets)
    print(
        f"[diagnostic] scalar-reward loss reaches {n_grad}/{n_total} "
        f"model parameters (beta={beta}, gamma={gamma})"
    )

    trainer.fit(epochs)
    return model, trainer


@torch.no_grad()
def evaluate(
    model: SimpleDefectDetector,
    val_loader: DataLoader,
    device: str,
    num_classes: int,
) -> Dict[str, object]:
    """Evaluate top-1 accuracy and per-class / group recalls on the val set."""
    model.eval()
    y_true: List[int] = []
    y_pred: List[int] = []

    for images, targets in val_loader:
        if images.numel() == 0:
            continue
        out = model(images.to(device))
        preds = out["classes"].tolist()
        for t, p in zip(targets, preds, strict=True):
            labels = t.get("labels", [])
            if not labels:
                continue
            y_true.append(int(labels[0]))
            y_pred.append(int(p))

    n = len(y_true)
    if n == 0:
        return {}

    accuracy = sum(1 for t, p in zip(y_true, y_pred, strict=True) if t == p) / n

    per_class: Dict[int, Optional[float]] = {}
    for c in range(num_classes):
        idx = [i for i, t in enumerate(y_true) if t == c]
        if idx:
            correct = sum(1 for i in idx if y_pred[i] == c)
            per_class[c] = correct / len(idx)
        else:
            per_class[c] = None

    def group_recall(group: set) -> float:
        idx = [i for i, t in enumerate(y_true) if t in group]
        if not idx:
            return float("nan")
        correct = sum(1 for i in idx if y_pred[i] in group)
        return correct / len(idx)

    recall_broken = group_recall(TIANCHI_BROKEN_CLASS_IDS)
    recall_skip = group_recall(TIANCHI_SKIP_CLASS_IDS)
    other_ids = (
        set(range(num_classes)) - TIANCHI_BROKEN_CLASS_IDS - TIANCHI_SKIP_CLASS_IDS
    )
    recall_other = group_recall(other_ids)

    valid = [r for r in per_class.values() if r is not None]
    macro_recall = float(np.mean(valid)) if valid else float("nan")

    return {
        "n_samples": n,
        "accuracy": accuracy,
        "macro_recall": macro_recall,
        "recall_broken": recall_broken,
        "recall_skip": recall_skip,
        "recall_other": recall_other,
        "per_class_recall": per_class,
    }


def _fmt(x: Optional[float], digits: int = 4) -> str:
    if x is None:
        return "  -   "
    if isinstance(x, float) and (np.isnan(x) or np.isinf(x)):
        return "  -   "
    return f"{x:.{digits}f}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/tianchi", help="Converted dataset root")
    parser.add_argument("--epochs", type=int, default=5, help="Epochs per run")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument(
        "--seed", type=int, default=0, help="RNG seed (shared by both runs)"
    )
    parser.add_argument("--device", default=None, help="cuda / cpu (auto if omitted)")
    parser.add_argument(
        "--out", default="runs/ablation_reward.json", help="JSON report path"
    )
    parser.add_argument(
        "--max-train",
        type=int,
        default=0,
        help="Subsample train set to N images (0 = all)",
    )
    parser.add_argument(
        "--max-val", type=int, default=0, help="Subsample val set to N images (0 = all)"
    )
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    config = auto_discover_dataset(args.data)
    num_classes = config.num_classes
    train_dataset, val_dataset = build_datasets(config)
    if args.max_train:
        train_dataset = Subset(train_dataset, range(args.max_train))
    if args.max_val:
        val_dataset = Subset(val_dataset, range(args.max_val))

    print("=" * 66)
    print("  A/B Ablation: hybrid reward vs pure detection loss")
    print("=" * 66)
    print(
        f"  Data:   {config.train_images} ({len(train_dataset)} train, "
        f"{len(val_dataset)} val)"
    )
    print(f"  Epochs: {args.epochs}   Batch: {args.batch_size}   Seed: {args.seed}")
    print(f"  Device: {device}")
    print("=" * 66)

    common = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "seed": args.seed,
        "device": device,
        "num_classes": num_classes,
    }

    print("\n>>> Run A (baseline): beta=0.0, gamma=0.0")
    model_a, _ = run_training(
        train_dataset, val_dataset, config, beta=0.0, gamma=0.0, **common
    )

    print("\n>>> Run B (reward): beta=0.1, gamma=0.05")
    model_b, _ = run_training(
        train_dataset, val_dataset, config, beta=0.1, gamma=0.05, **common
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=defect_collate_fn,
    )
    ev_a = evaluate(model_a, val_loader, device, num_classes)
    ev_b = evaluate(model_b, val_loader, device, num_classes)

    print("\n" + "=" * 66)
    print("  Results")
    print("=" * 66)
    print(f"{'metric':<18}{'A (baseline)':>14}{'B (reward)':>14}{'delta (B-A)':>14}")
    print("-" * 66)
    for key, label in [
        ("accuracy", "top-1 accuracy"),
        ("macro_recall", "macro recall"),
        ("recall_broken", "recall_broken"),
        ("recall_skip", "recall_skip"),
        ("recall_other", "recall_other"),
    ]:
        a, b = ev_a[key], ev_b[key]
        delta = b - a
        print(f"{label:<18}{_fmt(a):>14}{_fmt(b):>14}{_fmt(delta, 5):>14}")

    print("\nPer-class recall (exact):")
    print(f"{'class':<18}{'A':>10}{'B':>10}")
    for c in range(num_classes):
        name = config.class_names[c] if c < len(config.class_names) else f"cls_{c}"
        print(
            f"{name:<18}{_fmt(ev_a['per_class_recall'][c]):>10}"
            f"{_fmt(ev_b['per_class_recall'][c]):>10}"
        )

    # Verdict
    d_broken = ev_b["recall_broken"] - ev_a["recall_broken"]
    d_skip = ev_b["recall_skip"] - ev_a["recall_skip"]
    print("\n" + "=" * 66)
    print("  Verdict")
    print("=" * 66)
    improved = (d_broken > 0.01) or (d_skip > 0.01)
    if improved:
        print("  The reward run improved critical/major recall.")
        print("  => Claim is substantiated for these classes.")
    else:
        print("  No meaningful improvement in critical/major recall.")
        print("  => Claim is NOT substantiated by this experiment.")
    print(f"  delta recall_broken = {d_broken:+.4f}")
    print(f"  delta recall_skip   = {d_skip:+.4f}")
    print("=" * 66)

    report = {
        "config": {
            "data": args.data,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "seed": args.seed,
            "device": device,
        },
        "baseline": ev_a,
        "reward": ev_b,
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nReport saved to {args.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
