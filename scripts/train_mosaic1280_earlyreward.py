#!/usr/bin/env python
"""train_mosaic1280_earlyreward.py -- C1: Layer-1 reward only during early epochs.

Stage C of ``docs/plan_stage2_reward_relaunch_handoff.md`` (section 7).

Hypothesis under test: on a converged detector the 7-dimension reward can only
slide the PR curve (four independent interfaces, all negative); applied while the
curve is still forming -- the first ``--reward-epochs`` epochs of a from-scratch
run -- it might move the curve itself. This is the direct test of "wrong
interface" vs "wrong timing".

Everything except the reward term is a line-for-line copy of
``scripts/train_mosaic1280.py`` (the recipe that produced the 0.6349 baseline):
COCO init ``rtdetr-l.pt``, rect, bf16 patch (autocast + check_amp), MuSGD(auto),
lr0 5e-4, lrf 0.01, mosaic 1.0, 60 epochs, patience 20, batch 4, workers 0.
Therefore ``runs/rtdetr/tianchi18_mosaic1280`` is the *same-seed, same-recipe,
beta=0-for-the-whole-run* control: the screen costs no extra control run.

Design decisions that must go into ``PREREG.md`` before launch:

* ``--reward-beta`` (default 0.5) is the constant used in A/B Layer-1 runs;
* after ``--reward-epochs`` (default 10) the criterion's beta is set to exactly
  0.0, so the remaining epochs are pure supervised training;
* ``--reward-ramp`` switches to a linear 0 -> beta ramp over the reward window
  (the C2 variant); the default is constant beta;
* the criterion is attached in ``on_train_start`` because the trainer rebuilds
  its model inside ``_setup_train`` (attaching to ``model.model`` before
  ``model.train()`` is silently discarded -- see ``train_rtdetr_reward.py``).

The BN watchdog is copied from ``train_mosaic1280.py``: fp16 AMP zeroes
``running_var`` on ~39 layers after epoch 9, which a 1-2 epoch smoke test cannot
see. bf16 is what the recipe uses; the watchdog stays as a tripwire.

Usage::

    # smoke (pipeline only, ~minutes on a fraction of the set):
    python scripts/train_mosaic1280_earlyreward.py --epochs 1 --fraction 0.02 \
        --name smoke_c1 --reward-epochs 1

    # screen arm (single seed; ~15 h):
    python scripts/train_mosaic1280_earlyreward.py --name c1_earlyreward_s42

    # resume after an interruption:
    python scripts/train_mosaic1280_earlyreward.py --resume --name c1_earlyreward_s42
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import torch
import ultralytics.engine.trainer as TR
import ultralytics.utils.torch_utils as TU

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.training.dimension_rewards import DimensionRewardComputer
from backend.training.reward_rtdetr import RewardRTDETRDetectionLoss, attach_reward_criterion

ROOT = Path(__file__).resolve().parent.parent


def _bf16_autocast(enabled=True, device="cuda"):
    """bf16 autocast patch (identical to scripts/train_mosaic1280.py).

    The three-line bf16 patch is load-bearing: without ``check_amp=True``
    ultralytics silently falls back to fp32 and the run is not the baseline
    recipe; with fp16 instead, BN ``running_var`` is zeroed after epoch 9.
    """
    return torch.amp.autocast(device, dtype=torch.bfloat16, enabled=bool(enabled))


TU.autocast = _bf16_autocast
TR.autocast = _bf16_autocast
TR.check_amp = lambda model: True


def make_bn_watchdog(health: Path, marker: Path):
    """Return the per-epoch BN tripwire used by the baseline recipe."""

    def bn_watchdog(trainer) -> None:
        zeroed_all: list[str] = []
        for mdl in (trainer.model, getattr(trainer.ema, "ema", None)):
            if mdl is None:
                continue
            for key, val in mdl.state_dict().items():
                if "running_var" in key and val.abs().max().item() == 0:
                    zeroed_all.append(key)
        with open(health, "a", encoding="utf-8") as fh:
            fh.write(f"epoch {trainer.epoch}: running_var归零层数={len(zeroed_all)}\n")
        if zeroed_all:
            marker.write_text(
                f"epoch {trainer.epoch}: {len(zeroed_all)}层归零 {zeroed_all[:5]}\n",
                encoding="utf-8",
            )
            raise RuntimeError(
                f"BN污染: {len(zeroed_all)}层 running_var 归零, 中止训练 "
                f"(see {marker})"
            )

    return bn_watchdog


def epoch_beta(epoch: int, beta: float, reward_epochs: int, ramp: bool) -> float:
    """Beta for a 0-based ``epoch`` index.

    Args:
        epoch: 0-based epoch index.
        beta: Full reward weight.
        reward_epochs: Number of epochs in the reward window.
        ramp: Linear 0->beta ramp across the window instead of constant beta.

    Returns:
        The beta to apply for this epoch (exactly 0.0 outside the window).
    """
    if reward_epochs <= 0 or epoch >= reward_epochs:
        return 0.0
    if ramp:
        return beta * (epoch + 1) / reward_epochs
    return beta


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    # --- recipe (identical to train_mosaic1280.py) ---
    ap.add_argument("--init", default=str(ROOT / "rtdetr-l.pt"), help="COCO init checkpoint")
    ap.add_argument("--data", default=str(ROOT / "data" / "tianchi" / "data.yaml"))
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--device", default="0")
    ap.add_argument("--lr0", type=float, default=5e-4)
    ap.add_argument("--lrf", type=float, default=0.01)
    ap.add_argument("--mosaic", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--project", default=str(ROOT / "runs" / "rtdetr"))
    ap.add_argument("--name", default="c1_earlyreward_s42")
    ap.add_argument("--fraction", type=float, default=1.0, help="dataset fraction (smoke only)")
    ap.add_argument("--resume", action="store_true")
    # --- reward (the only difference) ---
    ap.add_argument("--reward-beta", type=float, default=0.5)
    ap.add_argument("--reward-epochs", type=int, default=10, help="epochs with the reward on")
    ap.add_argument("--reward-ramp", action="store_true", help="linear 0->beta ramp (C2 variant)")
    ap.add_argument("--include-misses", action="store_true")
    ap.add_argument("--miss-threshold", type=float, default=0.5)
    ap.add_argument("--lambda-miss", type=float, default=1.5)
    ap.add_argument("--log-every", type=int, default=200)
    args = ap.parse_args()

    if not Path(args.data).exists():
        raise SystemExit(f"ERROR: {args.data} not found")
    if args.reward_beta > 0 and args.reward_epochs <= 0:
        raise SystemExit("ERROR: --reward-epochs must be > 0 when --reward-beta > 0")

    project = str(Path(args.project).resolve())
    run_dir = Path(project) / args.name
    health = ROOT / "logs" / "bn_health_c1.log"
    marker = ROOT / "logs" / "BN_POISONED.marker"
    health.parent.mkdir(parents=True, exist_ok=True)

    # Record the pre-registered design next to the run so the verdict cannot be
    # moved after the fact (handoff iron law 2).
    run_dir.mkdir(parents=True, exist_ok=True)
    design = {
        "arm": "C1 early-representation reward",
        "name": args.name,
        "seed": args.seed,
        "recipe": {
            "init": args.init,
            "data": args.data,
            "epochs": args.epochs,
            "patience": args.patience,
            "imgsz": args.imgsz,
            "batch": args.batch,
            "lr0": args.lr0,
            "lrf": args.lrf,
            "mosaic": args.mosaic,
            "optimizer": "auto (MuSGD)",
            "amp": "bf16 patch",
            "rect": True,
            "workers": args.workers,
            "fraction": args.fraction,
        },
        "reward": {
            "beta": args.reward_beta,
            "epochs": args.reward_epochs,
            "ramp": args.reward_ramp,
            "include_misses": args.include_misses,
            "miss_threshold": args.miss_threshold,
            "lambda_miss": args.lambda_miss,
        },
        "control": "runs/rtdetr/tianchi18_mosaic1280 (same seed, same recipe, beta=0 throughout)",
        "prereg_criterion": "screen delta_mean >= +0.005 vs mosaic1280 -> confirm; else NEGATIVE",
    }
    (run_dir / "design.json").write_text(
        json.dumps(design, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    from ultralytics import RTDETR

    model = RTDETR(args.init)
    state: dict = {"crit": None}

    reward_computer = None
    if args.reward_beta > 0:
        reward_computer = DimensionRewardComputer(
            lambda_miss=args.lambda_miss, lambda_fp=1.0, head_type="sigmoid"
        )

    def _on_train_start(trainer):
        if reward_computer is None:
            # --reward-beta 0 is the paired control run: pure supervised recipe,
            # no criterion swap at all. Attaching a criterion here would crash on
            # a None reward computer.
            print("[c1] beta=0 -> supervised-only run (paired control)")
            return
        crit = attach_reward_criterion(
            trainer.model,
            reward_computer,
            beta=epoch_beta(0, args.reward_beta, args.reward_epochs, args.reward_ramp),
            include_misses=args.include_misses,
            miss_threshold=args.miss_threshold,
        )
        state["crit"] = crit
        print(
            f"[c1] attached reward criterion: beta(epoch 0)={crit.beta:.4f}, "
            f"window={args.reward_epochs}ep, ramp={args.reward_ramp}"
        )

    def _on_train_epoch_start(trainer):
        crit = state.get("crit")
        if crit is None:
            return
        epoch = int(getattr(trainer, "epoch", 0))
        crit.beta = epoch_beta(epoch, args.reward_beta, args.reward_epochs, args.reward_ramp)
        print(f"[c1] epoch {epoch + 1}/{args.epochs}: reward beta = {crit.beta:.4f}")

    def _on_train_batch_end(trainer):
        crit = state.get("crit")
        if args.log_every <= 0 or crit is None or not getattr(crit, "reward_history", None):
            return
        step = len(crit.reward_history)
        if step % args.log_every != 0:
            return
        print(
            f"[c1] step {step}: loss_reward={crit.loss_reward_history[-1]:+.5f} "
            f"mean_reward={crit.reward_history[-1]:+.5f} "
            f"matched={crit.match_count_history[-1]}"
        )

    def _on_train_end(trainer):
        crit = state.get("crit")
        save_dir = Path(getattr(trainer, "save_dir", run_dir))
        if crit is None or not getattr(crit, "reward_history", None):
            print("[c1] no reward steps recorded (reward window may be 0)")
            return
        path = save_dir / "reward_history.csv"
        misses = getattr(crit, "miss_count_history", []) or []
        with path.open("w", newline="") as fh:
            w = csv.writer(fh)
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
        print(f"[c1] reward history -> {path} ({len(crit.reward_history)} steps)")

    model.add_callback("on_train_start", _on_train_start)
    model.add_callback("on_train_epoch_start", _on_train_epoch_start)
    model.add_callback("on_train_batch_end", _on_train_batch_end)
    model.add_callback("on_train_epoch_end", make_bn_watchdog(health, marker))
    model.add_callback("on_train_end", _on_train_end)

    print("[c1] recipe:", json.dumps(design["recipe"], ensure_ascii=False))
    model.train(
        data=args.data,
        project=project,
        name=args.name,
        imgsz=args.imgsz,
        epochs=args.epochs,
        patience=args.patience,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        rect=True,
        mosaic=args.mosaic,
        optimizer="auto",
        amp=True,  # patched to bf16 above
        seed=args.seed,
        lr0=args.lr0,
        lrf=args.lrf,
        fraction=args.fraction,
        resume=args.resume,
        exist_ok=True,
    )
    print("[c1] training complete")
    print(
        "[c1] next: python scripts/eval_multIou.py "
        f"--weights {run_dir / 'weights' / 'best.pt'} --data {args.data} "
        f"--out runs/multi_iou/{args.name}_official.json"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
