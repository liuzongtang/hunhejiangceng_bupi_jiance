"""
Fine-tune the trained RT-DETR with the 7-dimension reward injected.

Loads ``runs/rtdetr/tianchi20/weights/best.pt`` (mAP50=0.594 baseline) and runs
Ultralytics' normal RT-DETR training, but with a ``loss_reward`` term added to
the criterion (see ``backend/training/reward_rtdetr.py``). The supervised part
(Hungarian matching + focal/vfl class loss + L1 + GIoU + denoising) is fully
preserved; the reward adds a critical-class recall signal on the matched
predictions.

This is the "Layer 1" baseline: prove the reward actually moves critical-defect
recall on a real RT-DETR, before any GRPO (Layer 3).

Usage:
    # Baseline (no reward) — same as plain RT-DETR fine-tune
    python scripts/train_rtdetr_reward.py --beta 0 --epochs 5 --name reward_off

    # Reward on
    python scripts/train_rtdetr_reward.py --beta 0.5 --epochs 5 --name reward_on
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import sys

# Allow `python scripts/train_rtdetr_reward.py` to import the `backend` package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ultralytics import YOLO

from backend.logging_config import get_training_log
from backend.training.dimension_rewards import DimensionRewardComputer
from backend.training.reward_rtdetr import attach_reward_criterion


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", default="runs/rtdetr/tianchi20/weights/best.pt")
    parser.add_argument("--data", default="data/tianchi/data.yaml")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--lr0", type=float, default=0.0001)
    parser.add_argument(
        "--beta", type=float, default=0.5, help="Reward weight (0 = off)"
    )
    parser.add_argument(
        "--include-misses",
        action="store_true",
        help="Feed missed critical GTs into the reward (spare-query recall signal)",
    )
    parser.add_argument(
        "--miss-threshold",
        type=float,
        default=0.5,
        help="Critical-set prob below which a GT counts as missed",
    )
    parser.add_argument(
        "--lambda-miss", type=float, default=1.5, help="D04 miss penalty"
    )
    parser.add_argument(
        "--log-every", type=int, default=50, help="Log reward every N steps (0=off)"
    )
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--project", default="runs/rtdetr-reward")
    parser.add_argument("--name", default="reward_on")
    args = parser.parse_args()

    args.project = os.path.abspath(args.project)

    model = YOLO(args.weights)

    reward_computer = None
    if args.beta > 0:
        reward_computer = DimensionRewardComputer(
            lambda_miss=args.lambda_miss,
            lambda_fp=1.0,
            head_type="sigmoid",  # RT-DETR per-class focal/vfl head
        )
        print(
            f"[reward] will attach RewardRTDETRDetectionLoss on train start "
            f"(beta={args.beta}, lambda_miss={args.lambda_miss})"
        )
    else:
        print("[reward] beta=0 — running supervised-only baseline")

    log = get_training_log()

    def _scalar(value):
        if isinstance(value, (int, float)):
            return round(float(value), 4) if isinstance(value, float) else value
        if hasattr(value, "item"):
            return round(float(value.item()), 4)
        return str(value)

    def _criterion(trainer):
        model = getattr(trainer, "model", None)
        return getattr(model, "criterion", None) if model is not None else None

    def _on_train_start(trainer):
        # The trainer rebuilds its own model inside _setup_train, so the criterion
        # must be attached here (after setup) — attaching to `model.model` before
        # `model.train()` is silently discarded.
        if reward_computer is None:
            return
        target = getattr(trainer.model, "module", trainer.model)
        attach_reward_criterion(
            target,
            reward_computer,
            beta=args.beta,
            include_misses=args.include_misses,
            miss_threshold=args.miss_threshold,
        )
        log.info(
            "reward_attached",
            data={
                "beta": args.beta,
                "include_misses": args.include_misses,
                "miss_threshold": args.miss_threshold,
            },
        )

    def _on_train_batch_end(trainer):
        """Throttled per-step reward logging (loss_reward, mean reward, match count)."""
        if args.log_every <= 0:
            return
        crit = _criterion(trainer)
        if crit is None or not getattr(crit, "reward_history", None):
            return
        step = len(crit.reward_history)
        if step % args.log_every != 0:
            return
        giou = cls = l1 = None
        if trainer.loss_items is not None:
            vals = trainer.loss_items.detach().flatten().tolist()
            giou, cls, l1 = (vals + [None] * 3)[:3]
        log.info(
            "batch_reward",
            data={
                "step": step,
                "loss_reward": crit.loss_reward_history[-1],
                "mean_reward": crit.reward_history[-1],
                "n_matched": crit.match_count_history[-1],
                "n_missed": (
                    crit.miss_count_history[-1]
                    if getattr(crit, "miss_count_history", None)
                    else 0
                ),
                "giou": _scalar(giou),
                "cls": _scalar(cls),
                "l1": _scalar(l1),
            },
        )

    def _on_fit_epoch_end(trainer):
        metrics = trainer.metrics or {}
        data = {
            "epoch": int(getattr(trainer, "epoch", 0)) + 1,
            "beta": args.beta,
            "metrics": {str(k): _scalar(v) for k, v in metrics.items()},
        }
        crit = _criterion(trainer)
        hist = getattr(crit, "reward_history", None)
        if hist:
            n = len(hist)
            data["reward_mean_step"] = round(sum(hist) / n, 5)
            data["loss_reward_mean_step"] = round(sum(crit.loss_reward_history) / n, 5)
            data["n_reward_steps"] = n
            data["n_matched_total"] = sum(crit.match_count_history)
            data["n_missed_total"] = sum(getattr(crit, "miss_count_history", []) or [])
        log.info("epoch_metrics", data=data)

    def _write_reward_csv(save_dir, crit) -> None:
        """Persist the per-step reward history next to ``results.csv``."""
        path = Path(save_dir) / "reward_history.csv"
        misses = getattr(crit, "miss_count_history", []) or []
        with path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["step", "loss_reward", "mean_reward", "n_matched", "n_missed"])
            for i in range(len(crit.reward_history)):
                w.writerow(
                    [
                        i + 1,
                        round(crit.loss_reward_history[i], 6),
                        round(crit.reward_history[i], 6),
                        crit.match_count_history[i],
                        misses[i] if i < len(misses) else 0,
                    ]
                )
        log.info(
            "reward_csv_written",
            data={"path": str(path), "rows": len(crit.reward_history)},
        )

    def _on_train_end(trainer):
        save_dir = str(getattr(trainer, "save_dir", ""))
        crit = _criterion(trainer)
        if getattr(crit, "reward_history", None):
            _write_reward_csv(save_dir, crit)
        png = None
        try:
            import reward_viz  # sibling module in scripts/ (matplotlib, headless)

            out = reward_viz.plot_reward_training(save_dir)
            png = str(out) if out else None
        except Exception as exc:  # plotting is best-effort; never fail training
            log.warning("plot_failed", data={"error": str(exc)})
        log.info(
            "training_complete",
            data={
                "epoch": int(getattr(trainer, "epoch", 0)) + 1,
                "beta": args.beta,
                "save_dir": save_dir,
                "plot": png,
            },
        )

    model.add_callback("on_train_start", _on_train_start)
    model.add_callback("on_train_batch_end", _on_train_batch_end)
    model.add_callback("on_fit_epoch_end", _on_fit_epoch_end)
    model.add_callback("on_train_end", _on_train_end)

    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        lr0=args.lr0,
        mosaic=0.0,
        close_mosaic=0,
        patience=20,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=args.name,
        exist_ok=True,
        seed=42,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
