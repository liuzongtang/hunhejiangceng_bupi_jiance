"""
GRPO (Layer 3) fine-tune of the trained RT-DETR, using query-noise exploration.

This is a *minimal custom loop* (not a full Ultralytics ``model.train``): it loads
``runs/rtdetr/tianchi18_full/weights/best.pt`` (the 18-class detector), wraps the
decoder with a latent Gaussian exploration policy
(:class:`~backend.training.grpo_trainer.RTDETRGRPOPolicy`), samples ``G`` candidate
detection sets per image, scores each with the 7-dim ``DimensionRewardComputer``,
forms a group-relative advantage, and steps with REINFORCE (+ optional KL to a
frozen reference, + optional differentiable reward gradient to the decoder).

The loop closes the GRPO feedback loop end-to-end: after each epoch it runs a
noise-free RT-DETR validation on the val split and saves ``best.pt`` on critical
class recall (hole / broken_warp / broken_spandex, paper A 口径) improvement, and
writes ``final.pt`` plus an initial-vs-final comparison at the end.

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

    # Smoke test: verify gradient / NaN / OOM quickly (skip validation)
    python scripts/train_rtdetr_grpo.py --beta-reward 0.5 --train-decoder --batch 2 --max-batches 2 --no-val
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
from ultralytics.models.rtdetr.val import RTDETRDataset, RTDETRValidator
from ultralytics.models.utils.ops import HungarianMatcher
from ultralytics.nn.modules.head import RTDETRDecoder

from backend.logging_config import get_training_log
from backend.training.dimension_rewards import DimensionRewardComputer
from backend.training.grpo_trainer import (
    GRPOTrainer,
    RTDETRGRPOPolicy,
    align_rtdetr_predictions,
    prepare_rtdetr_batch,
)

# Paper A 口径 critical classes (18-class ids): hole, broken_warp, broken_spandex.
CRITICAL_CLASS_IDS = [0, 9, 15]
CRITICAL_CLASS_NAMES = ["hole", "broken_warp", "broken_spandex"]
# Secondary confusable/critical classes worth tracking: star_skip, weave_defect.
SECONDARY_CLASS_IDS = [14, 17]


def _resolve_device(dev: str) -> torch.device:
    """Map Ultralytics device strings to a ``torch.device``."""
    if dev == "cpu":
        return torch.device("cpu")
    if dev.isdigit():
        return torch.device(f"cuda:{dev}" if torch.cuda.is_available() else "cpu")
    return torch.device(dev)


def _decoder_grad_norm(rtdetr: torch.nn.Module) -> float:
    """Return the max |grad| over the decoder's parameters (0 if none present)."""
    decoder = rtdetr.model[-1]
    grads = [
        p.grad.detach().abs().max().item()
        for p in decoder.parameters()
        if p.grad is not None
    ]
    return max(grads) if grads else 0.0


def validate(
    rtdetr: torch.nn.Module,
    args: argparse.Namespace,
    device_str: str,
    save_dir: Path,
) -> dict:
    """
    Run noise-free RT-DETR validation on the val split and return metrics.

    Reuses Ultralytics' ``RTDETRValidator`` (stretch preprocessing, no rect) so
    the numbers are directly comparable to ``model.val()``. Overall mAP50 /
    mAP50-95 / precision / recall come from ``DetMetrics.results_dict``; per-class
    recall (at the max-F1 threshold) and AP50 come from ``DetMetrics.box``, whose
    ``r``/``ap50`` arrays are indexed by position in ``ap_class_index`` (the class
    ids that actually have data), not by class id — so we remap id -> value.

    Args:
        rtdetr: The (noise-disabled, eval-mode) RT-DETR model to validate.
        args: Parsed CLI args (``data`` / ``imgsz`` / ``batch`` are reused).
        device_str: Ultralytics device string (e.g. ``"0"``).
        save_dir: Directory for the validator's output (mostly empty, plots off).

    Returns:
        Dict with ``map50`` / ``map50_95`` / ``precision`` / ``recall`` /
        ``per_class`` (name -> recall) / ``critical_recall`` (mean over critical).
    """
    val_args = {
        "data": args.data,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": device_str,
        "workers": 0,
        "plots": False,
        "rect": True,
        "mode": "val",
        "verbose": False,
        "project": str(save_dir),
        "name": "val",
        "exist_ok": True,
    }
    validator = RTDETRValidator(args=val_args, save_dir=save_dir / "val")
    validator(model=rtdetr)

    results = validator.metrics.results_dict
    box = validator.metrics.box
    # box.r is ordered by position in ap_class_index (classes that have GT data),
    # not by class id — remap to id -> recall.
    ap_class_index = list(box.ap_class_index)
    r = list(box.r)
    per_class: dict = {}
    for c in range(validator.nc):
        name = validator.names[c]
        if c in ap_class_index:
            per_class[name] = float(r[ap_class_index.index(c)])
        else:
            per_class[name] = 0.0
    crit = [per_class[validator.names[c]] for c in CRITICAL_CLASS_IDS]
    return {
        "map50": float(results["metrics/mAP50(B)"]),
        "map50_95": float(results["metrics/mAP50-95(B)"]),
        "precision": float(results["metrics/precision(B)"]),
        "recall": float(results["metrics/recall(B)"]),
        "per_class": per_class,
        "critical_recall": sum(crit) / len(crit),
    }


def save_checkpoint(
    rtdetr: torch.nn.Module,
    yolo: "YOLO",
    policy: RTDETRGRPOPolicy,
    path: Path,
) -> None:
    """
    Save the (monkey-patched) model to a reloadable Ultralytics checkpoint.

    The exploration policy replaces ``decoder._get_decoder_input`` with a closure,
    which ``torch.save`` cannot pickle. Restore the original bound method first,
    save via ``YOLO.save`` (which emits the full ckpt metadata), then re-install
    the noise patch for continued training.
    """
    decoder = rtdetr.model[-1]
    decoder._get_decoder_input = RTDETRDecoder._get_decoder_input.__get__(
        decoder, RTDETRDecoder
    )
    yolo.save(path)
    policy._install_noise()  # re-patch the noise injection


def _print_metrics(tag: str, m: dict) -> None:
    """Print a compact validation summary (overall + critical-class recall)."""
    crit = "  ".join(
        f"{name}={m['per_class'][name]:.3f}" for name in CRITICAL_CLASS_NAMES
    )
    print(
        f"[grpo] {tag}: mAP50={m['map50']:.4f} mAP50-95={m['map50_95']:.4f} "
        f"P={m['precision']:.4f} R={m['recall']:.4f} crit_recall={m['critical_recall']:.4f}\n"
        f"        {crit}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--weights", default="runs/rtdetr/tianchi18_full/weights/best.pt"
    )
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
    parser.add_argument(
        "--no-val",
        action="store_true",
        help="Skip validation and best.pt saving (smoke tests / pure GRPO tuning)",
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

    def align_fn(boxes, scores, targets):
        return align_rtdetr_predictions(boxes, scores, targets, matcher)

    # Separate parameter groups: log_std (single exploration scalar) gets its own
    # (larger) LR and no weight decay; everything else (decoder, if unfrozen) uses --lr.
    log_std_params: list = []
    other_params: list = []
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

    # ---- Initial validation (noise-free baseline) --------------------------
    baseline = None
    if not args.no_val:
        policy.noise_enabled = False
        rtdetr.eval()
        baseline = validate(rtdetr, args, args.device, save_dir)
        rtdetr.train()
        policy.noise_enabled = True
        _print_metrics("initial", baseline)

    best_recall = baseline["critical_recall"] if baseline else -1.0

    # ---- Loop --------------------------------------------------------------
    csv_path = save_dir / "grpo_history.csv"
    csv_file = csv_path.open("w", newline="")
    writer = csv.writer(csv_file)
    writer.writerow(
        ["epoch", "batch", "loss", "loss_policy", "mean_reward", "std_reward"]
    )

    final_metrics = baseline
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
            if not math.isfinite(metrics["loss"]):
                log.error("nan_loss", data={"epoch": epoch, "batch": i})
                print("[grpo] NaN loss — aborting run")
                csv_file.close()
                return 1
            total_steps += 1
            if i % args.log_every == 0:
                log.info(
                    "grpo_batch",
                    data={
                        "epoch": epoch,
                        "batch": i,
                        "log_std": round(policy.latent.log_std.item(), 5),
                        "decoder_grad": (
                            round(_decoder_grad_norm(rtdetr), 6)
                            if args.train_decoder
                            else 0.0
                        ),
                        **metrics,
                    },
                )
            if args.max_batches and total_steps >= args.max_batches:
                break

        # ---- Per-epoch validation + best.pt save ---------------------------
        if not args.no_val:
            policy.noise_enabled = False
            rtdetr.eval()
            em = validate(rtdetr, args, args.device, save_dir)
            rtdetr.train()
            policy.noise_enabled = True
            final_metrics = em
            _print_metrics(f"epoch{epoch}", em)
            log.info(
                "grpo_epoch_val",
                data={
                    "epoch": epoch,
                    "map50": round(em["map50"], 5),
                    "critical_recall": round(em["critical_recall"], 5),
                    "log_std": round(policy.latent.log_std.item(), 5),
                },
            )
            if em["critical_recall"] > best_recall:
                best_recall = em["critical_recall"]
                save_checkpoint(rtdetr, yolo, policy, save_dir / "best.pt")
                log.info(
                    "grpo_save_best",
                    data={"epoch": epoch, "critical_recall": round(best_recall, 5)},
                )
        else:
            log.info(
                "grpo_epoch",
                data={
                    "epoch": epoch,
                    "log_std": round(policy.latent.log_std.item(), 5),
                },
            )

        if args.max_batches and total_steps >= args.max_batches:
            break

    csv_file.close()

    # ---- Final save + initial-vs-final comparison --------------------------
    save_checkpoint(rtdetr, yolo, policy, save_dir / "final.pt")
    log.info(
        "grpo_complete",
        data={"save_dir": str(save_dir), "log_std": policy.latent.log_std.item()},
    )

    if baseline is not None and final_metrics is not None:
        print("\n[grpo] initial vs final (conf=0.25 recall):")
        print(f"{'class':<16}{'initial':>10}{'final':>10}{'delta':>10}")
        tracked = [*CRITICAL_CLASS_NAMES, "star_skip", "weave_defect"]
        for name in tracked:
            a = baseline["per_class"].get(name, 0.0)
            b = final_metrics["per_class"].get(name, 0.0)
            print(f"{name:<16}{a:>10.3f}{b:>10.3f}{b - a:>+10.3f}")
        print(
            f"{'mAP50':<16}{baseline['map50']:>10.4f}{final_metrics['map50']:>10.4f}"
            f"{final_metrics['map50'] - baseline['map50']:>+10.4f}"
        )
        print(
            f"{'crit_recall':<16}{baseline['critical_recall']:>10.4f}"
            f"{final_metrics['critical_recall']:>10.4f}"
            f"{final_metrics['critical_recall'] - baseline['critical_recall']:>+10.4f}"
        )

    print(
        f"[grpo] done — log_std={policy.latent.log_std.item():.5f}; "
        f"best.pt/final.pt -> {save_dir}; history -> {csv_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
