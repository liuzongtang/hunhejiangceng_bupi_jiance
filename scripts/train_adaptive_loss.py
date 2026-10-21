#!/usr/bin/env python
"""
train_adaptive_loss.py -- fine-tune RT-DETR with learnable cls/reg loss weights.

Stage 2A of docs/plan_stage2_adaptive.md: replaces the failed manual
reward/penalty interventions with automatic loss balancing.

  --method uw       Uncertainty Weighting (Kendall et al., CVPR 2018):
                    two learnable log-variances, per-step analytic update,
                    init 1:1 == uniform control at step 0, no extra backward.
  --method gradnorm GradNorm (Chen et al., ICML 2018): per-step autograd.grad
                    on the shared encoder trunk, ~2x backward cost.

Protocol is item-for-item identical to the strong-baseline control
(strong_r2_uniform, official mean 0.6158): mosaic1280 best.pt start,
5 epochs, batch 4, imgsz 1280 (square), lr0 1e-4, amp OFF (fp16 poisons BN
running_var at >=9 epochs), mosaic 0.0, deterministic, workers 0. The ONLY
variable is the loss-weighting mechanism.

Usage:
    # smoke (machinery check, ~2 min):
    python scripts/train_adaptive_loss.py --method uw --epochs 1 --fraction 0.05 --name smoke_uw
    # real arm:
    python scripts/train_adaptive_loss.py --method uw --seed 42 --name uw_seed42
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ultralytics import YOLO

from backend.training.adaptive_loss import AdaptiveRTDETRDetectionLoss


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--weights", default="runs/rtdetr/tianchi18_mosaic1280/weights/best.pt"
    )
    ap.add_argument("--data", default="data/tianchi/data.yaml")
    ap.add_argument("--method", choices=["uw", "gradnorm"], default="uw")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--fraction", type=float, default=1.0)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr0", type=float, default=0.0001)
    ap.add_argument(
        "--uw-lr",
        type=float,
        default=1e-3,
        help="per-step SGD rate for the UW log-variances",
    )
    ap.add_argument("--gradnorm-alpha", type=float, default=1.0)
    ap.add_argument("--gradnorm-lr", type=float, default=1e-3)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument(
        "--amp",
        action="store_true",
        help="fp16 AMP (default OFF -- AMP-BN pollution finding)",
    )
    ap.add_argument("--device", default="0")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--project", default="runs/ablation_adaptive")
    ap.add_argument("--name", default="uw_seed42")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    args.project = os.path.abspath(args.project)
    model = YOLO(args.weights)
    state: dict = {}

    def _trunk_params(target):
        """Shared trunk for GradNorm: the HybridEncoder (last shared layer
        approximation). RTDETRDetectionModel.model is a ModuleList
        [backbone, encoder, decoder]."""
        graph = getattr(target, "model", target)
        mods = list(graph) if hasattr(graph, "__iter__") else []
        enc = mods[-2] if len(mods) >= 2 else None
        if enc is None:
            print(
                "[adaptive] WARNING: encoder not found; using all graph params as trunk"
            )
            return [p for p in graph.parameters() if p.requires_grad]
        trunk = [p for p in enc.parameters() if p.requires_grad]
        n = sum(p.numel() for p in trunk)
        print(
            f"[adaptive] GradNorm trunk = {type(enc).__name__}: "
            f"{len(trunk)} tensors / {n / 1e6:.1f}M params"
        )
        return trunk

    def _on_train_start(trainer):
        # The trainer rebuilds its own model inside _setup_train, so the
        # criterion must be replaced here (after setup) -- same pattern as
        # train_rtdetr_reward.py.
        target = getattr(trainer.model, "module", trainer.model)
        trunk = _trunk_params(target) if args.method == "gradnorm" else None
        crit = AdaptiveRTDETRDetectionLoss(
            nc=target.nc,
            method=args.method,
            uw_lr=args.uw_lr,
            gradnorm_alpha=args.gradnorm_alpha,
            gradnorm_lr=args.gradnorm_lr,
            trunk_params=trunk,
            log_every=args.log_every,
        )
        target.criterion = crit
        state["crit"] = crit
        print(
            f"[adaptive] attached {args.method} criterion (nc={target.nc}); "
            f"init w_cls={crit.w['cls']} w_reg={crit.w['reg']}"
        )

    def _on_fit_epoch_end(trainer):
        crit = state.get("crit")
        if crit is not None:
            print(
                f"[adaptive] epoch {int(getattr(trainer, 'epoch', 0)) + 1}: "
                f"w_cls={crit.w['cls']:.4f} w_reg={crit.w['reg']:.4f} "
                f"(steps={crit.step})"
            )

    def _on_train_end(trainer):
        crit = state.get("crit")
        if crit is None:
            return
        save_dir = str(getattr(trainer, "save_dir", args.project))
        path = crit.dump_weight_log(save_dir)
        print(f"[adaptive] weight trajectory -> {path}")
        print(
            f"[adaptive] final: w_cls={crit.w['cls']:.4f} w_reg={crit.w['reg']:.4f} "
            f"lv={ {k: round(v, 4) for k, v in crit.lv.items()} }"
        )

    model.add_callback("on_train_start", _on_train_start)
    model.add_callback("on_fit_epoch_end", _on_fit_epoch_end)
    model.add_callback("on_train_end", _on_train_end)

    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        lr0=args.lr0,
        amp=args.amp,
        mosaic=0.0,
        close_mosaic=0,
        fraction=args.fraction,
        patience=20,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=args.name,
        exist_ok=True,
        seed=args.seed,
        deterministic=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
