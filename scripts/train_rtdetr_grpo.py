"""
GRPO (Layer 3) fine-tune of the trained RT-DETR, using query-noise exploration.

This is a *minimal custom loop* (not a full Ultralytics ``model.train``): it loads
``runs/rtdetr/tianchi20/weights/best.pt``, wraps the decoder with a latent
Gaussian exploration policy (:class:`~backend.training.grpo_trainer.RTDETRGRPOPolicy`),
samples ``G`` candidate detection sets per image, scores each with the 7-dim
``DimensionRewardComputer``, forms a group-relative advantage, and steps with
REINFORCE (+ optional KL to a frozen reference, + optional differentiable reward
gradient to the decoder).

By default only the exploration temperature ``log_std`` is trained (the model is
frozen), which demonstrates the GRPO machinery without risking the detector. Use
``--beta-reward > 0 --train-decoder`` to also steer the decoder weights via the
differentiable reward, and ``--freeze-backbone`` (default) to keep the backbone
fixed.

Usage:
    # Pure GRPO: tune the query-noise temperature only (small, safe run)
    python scripts/train_rtdetr_grpo.py --epochs 1 --num-groups 4 --name grpo_skel

    # Hybrid: GRPO + differentiable reward on the decoder (heavier)
    python scripts/train_rtdetr_grpo.py --beta-reward 0.5 --train-decoder --epochs 3
"""

from __future__ import annotations

import argparse
import csv
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

from backend.logging_config import get_training_log
from backend.training.dimension_rewards import DimensionRewardComputer
from backend.training.grpo_trainer import (
    GRPOTrainer,
    RTDETRGRPOPolicy,
    align_rtdetr_predictions,
    prepare_rtdetr_batch,
)


def _resolve_device(dev: str) -> torch.device:
    """Map Ultralytics device strings to a ``torch.device``."""
    if dev == "cpu":
        return torch.device("cpu")
    if dev.isdigit():
        return torch.device(f"cuda:{dev}" if torch.cuda.is_available() else "cpu")
    return torch.device(dev)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", default="runs/rtdetr/tianchi20/weights/best.pt")
    parser.add_argument("--data", default="data/tianchi/data.yaml")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--lr-log-std",
        type=float,
        default=0.1,
        help="Separate LR for the log_std exploration scalar (single param; "
        "REINFORCE signal is weak over ~B*300*256 dims, so it wants a larger LR)",
    )
    parser.add_argument(
        "--num-groups", type=int, default=4, help="G rollouts per image"
    )
    parser.add_argument(
        "--log-std-init", type=float, default=-2.3, help="init noise ~0.1"
    )
    parser.add_argument(
        "--ref-log-std", type=float, default=-4.6, help="reference noise ~0.01"
    )
    parser.add_argument("--beta-kl", type=float, default=0.01)
    parser.add_argument(
        "--beta-reward",
        type=float,
        default=0.0,
        help="Differentiable reward gradient to the decoder (0 = pure GRPO)",
    )
    parser.add_argument(
        "--advantage-eps",
        type=float,
        default=1e-2,
        help="Floor on the group-advantage std denominator (prevents explosion when "
        "the group's rewards are nearly identical)",
    )
    parser.add_argument(
        "--advantage-clip",
        type=float,
        default=5.0,
        help="Clamp group advantages to [-clip, clip]",
    )
    parser.add_argument("--lambda-miss", type=float, default=1.5)
    parser.add_argument(
        "--train-decoder",
        action="store_true",
        help="Unfreeze the decoder heads (only meaningful with --beta-reward > 0)",
    )
    parser.add_argument(
        "--freeze-backbone",
        action="store_true",
        default=True,
        help="Freeze backbone+neck (default); the decoder stays frozen unless --train-decoder",
    )
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument(
        "--max-batches",
        type=int,
        default=0,
        help="Stop after this many total steps (0 = all epochs); useful for smoke tests",
    )
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--project", default="runs/rtdetr-grpo")
    parser.add_argument("--name", default="grpo_skel")
    args = parser.parse_args()

    args.project = os.path.abspath(args.project)
    save_dir = Path(args.project) / args.name
    save_dir.mkdir(parents=True, exist_ok=True)
    device = _resolve_device(args.device)
    log = get_training_log()

    # ---- Load model + wrap with the exploration policy ---------------------
    yolo = YOLO(args.weights)
    rtdetr = yolo.model  # RTDETRDetectionModel
    rtdetr.to(device)
    rtdetr.train()

    # Freeze everything by default; unfreeze the decoder when requested.
    for p in rtdetr.parameters():
        p.requires_grad = False
    if args.train_decoder:
        decoder = rtdetr.model[-1]
        for p in decoder.parameters():
            p.requires_grad = True
        log.info("decoder_unfrozen")

    policy = RTDETRGRPOPolicy(
        rtdetr,
        num_groups=args.num_groups,
        log_std_init=args.log_std_init,
        ref_log_std=args.ref_log_std,
        detach_detections=(args.beta_reward <= 0),
    ).to(device)

    reward_computer = DimensionRewardComputer(
        lambda_miss=args.lambda_miss,
        lambda_fp=1.0,
        head_type="sigmoid",  # RT-DETR per-class focal/vfl head
    )
    matcher = HungarianMatcher(cost_gain={"class": 2, "bbox": 5, "giou": 2})
    align_fn = lambda boxes, scores, targets: align_rtdetr_predictions(  # noqa: E731
        boxes, scores, targets, matcher
    )

    # Separate parameter groups: log_std (single exploration scalar) gets its own
    # (larger) LR and no weight decay; everything else (decoder, if unfrozen) uses --lr.
    log_std_params: list[torch.nn.Parameter] = []
    other_params: list[torch.nn.Parameter] = []
    for name, p in policy.named_parameters():
        if not p.requires_grad:
            continue
        (log_std_params if "log_std" in name else other_params).append(p)
    param_groups = []
    if other_params:
        param_groups.append(
            {"params": other_params, "lr": args.lr, "weight_decay": 1e-4}
        )
    if log_std_params:
        param_groups.append(
            {"params": log_std_params, "lr": args.lr_log_std, "weight_decay": 0.0}
        )
    optimizer = torch.optim.AdamW(param_groups)
    trainable = len(other_params) + len(log_std_params)
    trainer = GRPOTrainer(
        policy,
        reward_computer,
        optimizer,
        num_groups=args.num_groups,
        beta_kl=args.beta_kl,
        beta_reward=args.beta_reward,
        ref_log_std=args.ref_log_std,
        advantage_eps=args.advantage_eps,
        advantage_clip=args.advantage_clip,
        align_fn=align_fn,
        device=device,
    )

    # ---- Data --------------------------------------------------------------
    cfg = get_cfg(
        overrides={
            "data": args.data,
            "imgsz": args.imgsz,
            "batch": args.batch,
            "workers": args.workers,
            "rect": False,
            "mosaic": 0.0,
            "cache": False,
            "single_cls": False,
        }
    )
    data_dict = check_det_dataset(args.data)
    # Use RTDETRDataset (stretch-to-square, rect_mode=False) — the same class
    # ``model.train()`` uses — not ``build_yolo_dataset`` (aspect-preserving
    # letterbox), so the reward sees the model's real input distribution.
    dataset = RTDETRDataset(
        img_path=data_dict["train"],
        imgsz=args.imgsz,
        batch_size=args.batch,
        augment=True,  # mode == "train"
        hyp=cfg,
        rect=False,
        cache=cfg.cache or None,
        single_cls=False,
        prefix="train: ",
        data=data_dict,
        fraction=1.0,
    )
    loader = build_dataloader(dataset, args.batch, workers=args.workers, shuffle=True)
    nb = len(loader)

    print(
        f"[grpo] device={device} groups={args.num_groups} beta_kl={args.beta_kl} "
        f"beta_reward={args.beta_reward} lr_log_std={args.lr_log_std} "
        f"trainable={trainable} batches/epoch={nb}"
    )

    # ---- Loop --------------------------------------------------------------
    csv_path = save_dir / "grpo_history.csv"
    csv_file = csv_path.open("w", newline="")
    writer = csv.writer(csv_file)
    writer.writerow(
        ["epoch", "batch", "loss", "loss_policy", "mean_reward", "std_reward"]
    )

    total_steps = 0
    for epoch in range(1, args.epochs + 1):
        for i, batch in enumerate(loader):
            if i >= nb:
                break
            img, targets = prepare_rtdetr_batch(batch, device)
            metrics = trainer.train_step(img, targets)
            writer.writerow(
                [
                    epoch,
                    i,
                    round(metrics["loss"], 6),
                    round(metrics["loss_policy"], 6),
                    round(metrics["mean_reward"], 6),
                    round(metrics["std_reward"], 6),
                ]
            )
            total_steps += 1
            if i % args.log_every == 0:
                log.info(
                    "grpo_batch",
                    data={
                        "epoch": epoch,
                        "batch": i,
                        "log_std": round(policy.latent.log_std.item(), 5),
                        **metrics,
                    },
                )
            if args.max_batches and total_steps >= args.max_batches:
                break
        log.info(
            "grpo_epoch",
            data={"epoch": epoch, "log_std": round(policy.latent.log_std.item(), 5)},
        )
        if args.max_batches and total_steps >= args.max_batches:
            break

    csv_file.close()
    log.info(
        "grpo_complete",
        data={"save_dir": str(save_dir), "log_std": policy.latent.log_std.item()},
    )
    print(
        f"[grpo] done — log_std={policy.latent.log_std.item():.5f}; history -> {csv_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
