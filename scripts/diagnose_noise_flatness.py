"""
Diagnose P1: is the 7-dim reward locally flat w.r.t. query noise?

Sweeps the exploration scale ``sigma`` and, for each, samples N rollouts on a fixed
batch containing critical-class GTs (broken_warp 9 / star_skip 15 / broken_spandex 16
/ weave_defect 19), reporting the reward's mean and std. This answers whether
query-noise exploration has any effective window before deciding to run GRPO:

- flat-then-cliff  -> no window; GRPO query-noise is dead
- smooth monotonic -> there is a (degrading) signal, but needs tight sigma scheduling
- hump (up then down) -> an optimal sigma exists; GRPO can learn it

Usage:
    python scripts/diagnose_noise_flatness.py --num-samples 40 --sigmas 0,1e-4,3e-4,1e-3,3e-3,1e-2,3e-2,1e-1,3e-1,1
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.data import build_dataloader
from ultralytics.data.utils import check_det_dataset
from ultralytics.models.rtdetr.val import RTDETRDataset
from ultralytics.models.utils.ops import HungarianMatcher

from backend.training.dimension_rewards import DIMENSION_NAMES, DimensionRewardComputer
from backend.training.grpo_trainer import (
    RTDETRGRPOPolicy,
    align_rtdetr_predictions,
    prepare_rtdetr_batch,
)

CRITICAL = {9, 15, 16, 19}  # broken_warp / star_skip / broken_spandex / weave_defect


def _resolve_device(dev: str) -> torch.device:
    if dev == "cpu":
        return torch.device("cpu")
    if dev.isdigit():
        return torch.device(f"cuda:{dev}" if torch.cuda.is_available() else "cpu")
    return torch.device(dev)


def _score(boxes, scores, targets, align_fn, reward_computer) -> float:
    """Score one rollout with the 7-dim reward computer (scalar mean over dims)."""
    aligned = align_fn(boxes, scores, targets)
    if aligned is None:
        return 0.0
    predictions, aligned_targets = aligned
    rewards = reward_computer.compute_all(predictions, aligned_targets)
    return float(torch.stack([rewards[n] for n in DIMENSION_NAMES]).mean().item())


def _flat_cls(cls) -> list[int]:
    """Flatten a collated ``cls`` field (tensor or list-of-tensors) to a list of ints."""
    if isinstance(cls, (list, tuple)):
        out: list[int] = []
        for t in cls:
            out.extend(int(x) for x in torch.as_tensor(t).reshape(-1).tolist())
        return out
    return [int(x) for x in torch.as_tensor(cls).reshape(-1).tolist()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", default="runs/rtdetr/tianchi20/weights/best.pt")
    parser.add_argument("--data", default="data/tianchi/data.yaml")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--num-samples", type=int, default=40)
    parser.add_argument(
        "--sigmas",
        default="0,1e-4,3e-4,1e-3,3e-3,1e-2,3e-2,1e-1,3e-1,1",
    )
    parser.add_argument("--device", default="0")
    parser.add_argument("--project", default="runs/rtdetr-grpo")
    parser.add_argument("--name", default="noise_diag")
    args = parser.parse_args()
    sigmas = [float(s) for s in args.sigmas.split(",")]

    device = _resolve_device(args.device)
    args.project = os.path.abspath(args.project)
    save_dir = Path(args.project) / args.name
    save_dir.mkdir(parents=True, exist_ok=True)

    # ---- Model + policy (same setup as the GRPO training loop) -----------------
    yolo = YOLO(args.weights)
    rtdetr = yolo.model
    rtdetr.to(device)
    rtdetr.train()
    policy = RTDETRGRPOPolicy(
        rtdetr,
        num_groups=1,
        log_std_init=-2.3,
        ref_log_std=-4.6,
        detach_detections=True,
    ).to(device)

    reward_computer = DimensionRewardComputer(
        lambda_miss=1.5, lambda_fp=1.0, head_type="sigmoid"
    )
    matcher = HungarianMatcher(cost_gain={"class": 2, "bbox": 5, "giou": 2})
    align_fn = lambda boxes, scores, targets: align_rtdetr_predictions(  # noqa: E731
        boxes, scores, targets, matcher
    )

    # ---- Data: find a batch containing critical-class GTs ----------------------
    cfg = get_cfg(
        overrides={
            "data": args.data,
            "imgsz": args.imgsz,
            "batch": args.batch,
            "workers": 0,
            "rect": False,
            "mosaic": 0.0,
            "cache": False,
            "single_cls": False,
        }
    )
    data_dict = check_det_dataset(args.data)
    dataset = RTDETRDataset(
        img_path=data_dict["train"],
        imgsz=args.imgsz,
        batch_size=args.batch,
        augment=True,
        hyp=cfg,
        rect=False,
        cache=cfg.cache or None,
        single_cls=False,
        prefix="train: ",
        data=data_dict,
        fraction=1.0,
    )
    loader = build_dataloader(dataset, args.batch, workers=0, shuffle=True)

    batch = None
    for i, b in enumerate(loader):
        if set(_flat_cls(b["cls"])) & CRITICAL:
            batch = b
            break
        if i >= 100:
            break
    if batch is None:
        raise SystemExit("no batch with critical-class GTs found in first 100 batches")
    img, targets = prepare_rtdetr_batch(batch, device)
    cls_in_batch = sorted(set(_flat_cls(batch["cls"])))
    print(
        f"[diag] batch contains GT classes {cls_in_batch}; "
        f"critical present={sorted(set(cls_in_batch) & CRITICAL)}"
    )

    # ---- Sweep sigma -----------------------------------------------------------
    rows = []
    for sigma in sigmas:
        if sigma == 0.0:
            policy.noise_enabled = False
            rollouts = policy.sample_group(img, targets)
            rs = [
                _score(r.boxes, r.scores, targets, align_fn, reward_computer)
                for r in rollouts
            ]
        else:
            policy.noise_enabled = True
            with torch.no_grad():
                policy.latent.log_std.copy_(
                    torch.tensor(math.log(sigma), dtype=torch.float32)
                )
            rs = []
            for _ in range(args.num_samples):
                rollouts = policy.sample_group(img, targets)
                rs.append(
                    _score(
                        rollouts[0].boxes,
                        rollouts[0].scores,
                        targets,
                        align_fn,
                        reward_computer,
                    )
                )
        mean = float(sum(rs) / len(rs))
        var = float(sum((r - mean) ** 2 for r in rs) / len(rs))
        std = math.sqrt(var)
        rows.append((sigma, mean, std, min(rs), max(rs)))
        print(
            f"  sigma={sigma:>7.4f}  mean_reward={mean:+.5f}  std_reward={std:.5f}  "
            f"[{min(rs):+.5f}, {max(rs):+.5f}]"
        )

    # ---- Save CSV + plot -------------------------------------------------------
    csv_path = save_dir / "noise_flatness.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sigma", "mean_reward", "std_reward", "min_reward", "max_reward"])
        w.writerows(rows)
    print(f"[diag] wrote {csv_path}")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        s = [r[0] for r in rows]
        means = [r[1] for r in rows]
        stds = [r[2] for r in rows]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
        xs = [v if v > 0 else 1e-5 for v in s]
        ax1.plot(xs, means, "o-")
        ax1.set_xscale("log")
        ax1.set_xlabel("sigma (noise std)")
        ax1.set_ylabel("mean reward")
        ax1.set_title("reward vs noise scale")
        ax1.grid(True, which="both", alpha=0.3)
        ax2.plot(xs, stds, "o-", color="tab:red")
        ax2.set_xscale("log")
        ax2.set_xlabel("sigma (noise std)")
        ax2.set_ylabel("std reward (exploration signal)")
        ax2.set_title("reward variance vs noise scale")
        ax2.grid(True, which="both", alpha=0.3)
        fig.tight_layout()
        png = save_dir / "noise_flatness.png"
        fig.savefig(png, dpi=110)
        print(f"[diag] wrote {png}")
    except ImportError:
        print("[diag] matplotlib unavailable, skipped plot")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
