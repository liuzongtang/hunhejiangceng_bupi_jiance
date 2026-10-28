"""
D1: fine-tune RT-DETR with an additive low-IoU localisation reward.

Hypothesis (handoff section 8)
    The competition metric is mean(mAP@IoU0.1, mAP@IoU0.3, mAP@IoU0.5). The
    AP@0.1 -> AP@0.5 drop of the locked best detector (mean 0.074, up to 0.26 on
    thin/linear classes) is mainly a localisation gap, so a reward that acts only
    on the box side and is computed at the competition's IoU scales (0.1 / 0.3)
    may recover part of it without touching the classification arm - and hence
    without the precision/recall zero-sum that killed the re-weighting arms.

Pre-registered criterion (handoff section 8)
    Mean gain >= +0.005, with the gain coming mainly from AP@0.3/AP@0.5 on the
    thin-frame classes (loose_warp / size_stain / broken_warp / hanging_warp);
    crit_recall must not drop. Judged on 3 seeds (42 / 123 / 2024).

Failure branch (handoff section 8)
    R3-like signature (mean down, crit_recall up) -> localisation cannot be moved
    outwards on this representation; record the negative. Double drop -> handoff
    section 11 F2 (halve weight/LR once, then UW gradient balancing, then abandon).

Difference from the failed R3/NWD arm (required wording)
    NWD *replaced* the metric (zero-sum, mean -0.015); D1 *adds* a localisation
    reward aligned to IoU 0.1/0.3 and leaves the supervised loss untouched.

Protocol (handoff R4, identical to ``train_rtdetr_boxweighted.py``)
    mosaic1280 ``best.pt`` start | 5 epochs | batch 4 | imgsz 1280 (square
    letterbox) | lr0 1e-4 | amp OFF | mosaic 0 | workers 0 | seed 42.

    ``--beta 0`` is an exact uniform control: the criterion still computes the
    term (so the soft-IoU trace is logged), but ``loss_loc == -0.0`` and its
    gradient is identically zero, leaving optimisation bit-identical to a plain
    fine-tune.

Artifacts written to ``<project>/<name>/``
    - ``design.json``: hypothesis, pre-registered criterion, failure branch, full
      protocol, beta/IoU scales, and (after training) baseline vs final metrics.
    - ``loc_history.csv``: per-step ``loss_loc``, mean soft-IoU, mean plain IoU
      and matched-pair count, all detached.

Usage:
    # Arm (beta 0.1 on the 0.1/0.3 saturating soft-IoU)
    python scripts/train_rtdetr_locreward.py --beta 0.1 --epochs 5 --name d1_loc_s42

    # Exact uniform control
    python scripts/train_rtdetr_locreward.py --beta 0 --epochs 5 --name d1_loc_beta0_s42

    # Then the official metric (the only numbers that count, handoff R3):
    python scripts/eval_multIou.py \
        --weights runs/rtdetr-loc/d1_loc_s42/weights/best.pt \
        --data data/tianchi/data.yaml \
        --out runs/multi_iou/d1_loc_s42_official.json
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
import os
from pathlib import Path
import sys
from typing import Optional

# Allow `python scripts/train_rtdetr_locreward.py` to import the `backend` package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from ultralytics import YOLO
from ultralytics.models.rtdetr.val import RTDETRValidator

from backend.training.locreward import DEFAULT_IOU_SCALES, attach_loc_criterion

# Critical / secondary class ids under the 18-class taxonomy (see data.yaml).
CRITICAL_CLASS_IDS = [0, 9, 15]  # hole, broken_warp, broken_spandex
CRITICAL_CLASS_NAMES = ["hole", "broken_warp", "broken_spandex"]
SECONDARY_CLASS_IDS = [14, 17]  # star_skip, weave_defect
TRACKED_NAMES = [*CRITICAL_CLASS_NAMES, "star_skip", "weave_defect"]

# Handoff R4 / section 8 references. These are NOT measured by this script; they
# are printed as context for the pre-registered comparison.
B_STAR = {"mean": 0.6349, "map50": 0.5917}  # mosaic1280 best.pt, locked baseline
HIST_UNIFORM = {"mean": 0.6158, "critical_recall": 0.6260}  # strong_r2_uniform

#: Default start checkpoint: the locked mosaic1280 baseline (handoff R4).
DEFAULT_WEIGHTS = "runs/rtdetr/tianchi18_mosaic1280/weights/best.pt"

FORMULA = (
    "iou_i = differentiable_iou(pred_i, gt_i); "
    "soft_iou_i = 0.5 * (min(1, iou_i / 0.1) + min(1, iou_i / 0.3)); "
    "loss_loc = -beta * mean_i soft_iou_i"
)


def validate(
    rtdetr: torch.nn.Module, args: argparse.Namespace, device_str: str
) -> dict:
    """
    Run noise-free RT-DETR validation and return overall + per-class recall.

    Args:
        rtdetr: The RT-DETR ``DetectionModel`` to evaluate.
        args: Parsed CLI args (uses ``data``, ``imgsz``, ``batch``, ``project``,
            ``name``).
        device_str: Ultralytics device string (e.g. ``"0"`` or ``"cpu"``).

    Returns:
        Dict with ``map50``, ``map50_95``, ``precision``, ``recall``,
        ``per_class`` (name -> recall @ max-F1) and ``critical_recall``.
    """
    val_args = {
        "data": args.data,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": device_str,
        "workers": 0,
        "plots": False,
        # RT-DETR validation forces square letterbox (rect=False) internally; the
        # flag is kept identical to train_rtdetr_boxweighted.py for comparability.
        "rect": True,
        "mode": "val",
        "verbose": False,
    }
    validator = RTDETRValidator(
        args=val_args, save_dir=Path(args.project) / args.name / "val"
    )
    validator(model=rtdetr)

    results = validator.metrics.results_dict
    box = validator.metrics.box
    ap_class_index = list(box.ap_class_index)
    r = list(box.r)
    per_class: dict = {}
    for c in range(validator.nc):
        name = validator.names[c]
        per_class[name] = (
            float(r[ap_class_index.index(c)]) if c in ap_class_index else 0.0
        )
    crit = [per_class[validator.names[c]] for c in CRITICAL_CLASS_IDS]
    return {
        "map50": float(results["metrics/mAP50(B)"]),
        "map50_95": float(results["metrics/mAP50-95(B)"]),
        "precision": float(results["metrics/precision(B)"]),
        "recall": float(results["metrics/recall(B)"]),
        "per_class": per_class,
        "critical_recall": sum(crit) / len(crit),
    }


def _print_metrics(tag: str, m: dict) -> None:
    """Print a compact validation summary (overall + tracked-class recall)."""
    tracked = "  ".join(f"{n}={m['per_class'][n]:.3f}" for n in TRACKED_NAMES)
    print(
        f"[loc] {tag}: mAP50={m['map50']:.4f} mAP50-95={m['map50_95']:.4f} "
        f"P={m['precision']:.4f} R={m['recall']:.4f} crit_recall={m['critical_recall']:.4f}\n"
        f"        {tracked}"
    )


def _load_baseline(args: argparse.Namespace) -> Optional[dict]:
    """
    Resolve the per-class baseline used in the final comparison table.

    ``--baseline auto`` validates the start checkpoint before fine-tuning, so the
    table is a same-machine, same-pipeline "start vs arm" comparison (no
    fabricated numbers). ``--baseline-json`` reuses an earlier result.
    """
    if args.baseline_json:
        path = Path(args.baseline_json)
        with path.open(encoding="utf-8") as fh:
            data = json.load(fh)
        data.setdefault("source", str(path))
        print(f"[loc] baseline <- {path}")
        _print_metrics("baseline", data)
        return data
    if args.baseline == "skip":
        print("[loc] baseline skipped (--baseline skip): table will show deltas as n/a")
        return None
    print(f"[loc] measuring baseline on the start checkpoint: {args.weights}")
    start = YOLO(args.weights)
    rtdetr = start.model
    rtdetr.eval()
    base = validate(rtdetr, args, args.device)
    base["source"] = f"start-checkpoint:{args.weights}"
    _print_metrics("baseline", base)
    return base


def _design_payload(args: argparse.Namespace) -> dict:
    """Build the pre-registration payload written to ``design.json``."""
    return {
        "arm": "D1",
        "module": "backend.training.locreward",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "hypothesis": (
            "The AP@0.1 -> AP@0.5 drop of the locked detector (mean 0.074, up to "
            "0.26 on thin classes) is a localisation gap; an additive reward on "
            "matched boxes at the competition's IoU scales (0.1/0.3) can recover "
            "part of it without touching the classification arm, hence without "
            "the R2/B/NWD precision-recall zero-sum."
        ),
        "prereg_criterion": (
            "转正 = 官方 mean > 同协议对照 +0.005，且增益主要来自 AP@0.3/@0.5 的"
            "薄框类（loose_warp / size_stain / broken_warp / hanging_warp）；"
            "crit_recall 不降；三种子 42/123/2024。"
        ),
        "failure_branch": (
            "若签名同 R3（mean 降、crit_recall 升）→ 定位侧在该特征下也不可外移，"
            "记阴性；若双降 → 按 §11 F2 处理（权重/lr 减半重试一次，再 UW 梯度平衡，"
            "再失败则换轴）。"
        ),
        "difference_from_r3_nwd": (
            "NWD *replaced* the metric (zero-sum, mean -0.015); D1 *adds* a "
            "localisation reward aligned to IoU 0.1/0.3 and leaves the supervised "
            "loss untouched."
        ),
        "formula": FORMULA,
        "protocol": {
            "weights": args.weights,
            "data": args.data,
            "epochs": args.epochs,
            "imgsz": args.imgsz,
            "batch": args.batch,
            "lr0": args.lr0,
            "amp": args.amp,
            "mosaic": 0.0,
            "close_mosaic": 0,
            "workers": args.workers,
            "seed": args.seed,
            "device": args.device,
            "beta": args.beta,
            "iou_scales": list(args.iou_scales),
        },
        "control": {
            "beta": 0.0,
            "note": (
                "exact uniform control: the criterion still logs soft-IoU, but "
                "loss_loc is -0.0 and its gradient is identically zero"
            ),
        },
        "handoff_references": {
            "B_star": B_STAR,
            "historical_uniform_control": HIST_UNIFORM,
            "official_eval": "scripts/eval_multIou.py (mean of mAP@0.1/0.3/0.5)",
        },
        "status": "started",
    }


def _write_design(save_dir: Path, payload: dict) -> None:
    """Write ``design.json`` (pre-registration at start, results merged at end)."""
    path = Path(save_dir) / "design.json"
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
        fh.write("\n")


def _write_history(save_dir: Path, criterion) -> None:
    """Write ``loc_history.csv`` from the criterion's detached per-step history."""
    path = Path(save_dir) / "loc_history.csv"
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["step", "loss_loc", "soft_iou", "raw_iou", "n_matched"])
        for i, loss_loc in enumerate(criterion.loss_loc_history):
            writer.writerow(
                [
                    i + 1,
                    round(loss_loc, 6),
                    round(criterion.soft_iou_history[i], 6),
                    round(criterion.raw_iou_history[i], 6),
                    criterion.match_count_history[i],
                ]
            )
    print(f"[loc] wrote {path} ({len(criterion.loss_loc_history)} steps)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--weights",
        default=DEFAULT_WEIGHTS,
        help="Start checkpoint (handoff R4: the locked mosaic1280 best.pt)",
    )
    parser.add_argument("--data", default="data/tianchi/data.yaml")
    parser.add_argument("--epochs", type=int, default=5, help="Fine-tune epochs")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument(
        "--lr0",
        type=float,
        default=0.0001,
        help="Fine-tune LR (handoff R4: 1e-4)",
    )
    parser.add_argument(
        "--amp",
        type=lambda s: str(s).lower() in ("true", "1", "yes", "on"),
        default=False,
        help="fp16 AMP; OFF by default (handoff section 14 pitfall 3: fp16 poisons BN)",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.1,
        help="Localisation-reward weight (0 = exact uniform control)",
    )
    parser.add_argument(
        "--iou-scales",
        type=float,
        nargs="+",
        default=list(DEFAULT_IOU_SCALES),
        help="Saturating IoU thresholds for the soft-IoU reward",
    )
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--project", default="runs/rtdetr-loc", help="Output project dir")
    parser.add_argument("--name", default="d1_loc_s42", help="Run name")
    parser.add_argument(
        "--log-every", type=int, default=50, help="Print per-step soft-IoU every N steps"
    )
    parser.add_argument(
        "--baseline",
        default="auto",
        choices=["auto", "skip"],
        help="auto = validate the start checkpoint before training; skip = no baseline",
    )
    parser.add_argument(
        "--baseline-json",
        default=None,
        help="Reuse a previously dumped metrics JSON instead of re-validating",
    )
    parser.add_argument(
        "--no-val", action="store_true", help="Skip the final per-class recall report"
    )
    args = parser.parse_args()

    # Resolve project to an absolute path so Ultralytics does not insert the task
    # name (runs/detect/<project>) for relative paths.
    args.project = os.path.abspath(args.project)
    if any(t <= 0 for t in args.iou_scales):
        parser.error("--iou-scales must all be positive")
    if args.beta < 0:
        parser.error("--beta must be >= 0")

    print(
        f"[loc] arm=D1 beta={args.beta} iou_scales={list(args.iou_scales)} "
        f"weights={args.weights}"
    )
    print(f"[loc] formula: {FORMULA}")
    if args.beta == 0.0:
        print(
            "[loc] beta=0 -> uniform control: loss_loc is identically zero and "
            "contributes no gradient"
        )

    baseline = None if args.no_val else _load_baseline(args)

    model = YOLO(args.weights)
    run_dir = Path(args.project) / args.name
    design = _design_payload(args)
    criterion_ref: dict = {"criterion": None}

    def _on_train_start(trainer) -> None:
        # The trainer rebuilds its model inside _setup_train, so the criterion must
        # be attached here - attaching before model.train() is silently discarded.
        target = getattr(trainer.model, "module", trainer.model)
        criterion = attach_loc_criterion(
            target,
            beta=args.beta,
            iou_scales=tuple(args.iou_scales),
        )
        criterion_ref["criterion"] = criterion
        save_dir = Path(getattr(trainer, "save_dir", run_dir))
        _write_design(save_dir, design)
        nc = int(getattr(target, "nc", target.model[-1].nc))
        print(
            f"[loc] attached LocRewardRTDETRDetectionLoss (beta={args.beta}, "
            f"nc={nc}); pre-registration -> {save_dir / 'design.json'}"
        )

    def _on_train_batch_end(trainer) -> None:
        criterion = criterion_ref["criterion"]
        if criterion is None or args.log_every <= 0:
            return
        step = len(criterion.loss_loc_history)
        if step == 0 or step % args.log_every != 0:
            return
        print(
            f"[loc] step {step}: soft_iou={criterion.soft_iou_history[-1]:.4f} "
            f"raw_iou={criterion.raw_iou_history[-1]:.4f} "
            f"n_matched={criterion.match_count_history[-1]} "
            f"loss_loc={criterion.loss_loc_history[-1]:+.5f}"
        )

    def _on_train_end(trainer) -> None:
        criterion = criterion_ref["criterion"]
        save_dir = Path(getattr(trainer, "save_dir", run_dir))
        if criterion is not None and criterion.loss_loc_history:
            _write_history(save_dir, criterion)
            design["reward_trace"] = {
                "steps": len(criterion.loss_loc_history),
                "mean_soft_iou": round(
                    sum(criterion.soft_iou_history) / len(criterion.soft_iou_history), 6
                ),
                "mean_raw_iou": round(
                    sum(criterion.raw_iou_history) / len(criterion.raw_iou_history), 6
                ),
                "mean_loss_loc": round(
                    sum(criterion.loss_loc_history) / len(criterion.loss_loc_history), 6
                ),
            }

    model.add_callback("on_train_start", _on_train_start)
    model.add_callback("on_train_batch_end", _on_train_batch_end)
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
        patience=20,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=args.name,
        exist_ok=True,
        seed=args.seed,
    )

    best = run_dir / "weights" / "best.pt"
    design["status"] = "trained"
    design["best_checkpoint"] = str(best)

    if args.no_val:
        _write_design(run_dir, design)
        return 0

    # Final per-class recall on the best checkpoint, vs the start checkpoint.
    final_model = YOLO(str(best))
    rtdetr = final_model.model
    rtdetr.eval()
    final = validate(rtdetr, args, args.device)
    _print_metrics("best", final)
    design["status"] = "validated"
    design["results"] = {"baseline": baseline, "final": final}
    _write_design(run_dir, design)

    print("\n[loc] baseline vs locreward (per-class recall @ max-F1):")
    print("class            baseline   final     delta")
    for name in TRACKED_NAMES:
        base = baseline["per_class"][name] if baseline else float("nan")
        arm = final["per_class"][name]
        delta = f"{arm - base:+.3f}" if baseline else "n/a"
        print(f"{name:<16}   {base:.3f}    {arm:.3f}   {delta}")
    if baseline:
        print(
            f"{'mAP50':<16}   {baseline['map50']:.4f}   {final['map50']:.4f}   "
            f"{(final['map50'] - baseline['map50']):+.4f}"
        )
        print(
            f"crit_recall      {baseline['critical_recall']:.4f}   "
            f"{final['critical_recall']:.4f}   "
            f"{(final['critical_recall'] - baseline['critical_recall']):+.4f}"
        )
    else:
        print(f"{'mAP50':<16}   n/a       {final['map50']:.4f}   n/a")
        print(f"crit_recall      n/a       {final['critical_recall']:.4f}   n/a")
    print(
        f"\n[loc] handoff references (not re-measured): B* mean={B_STAR['mean']:.4f} "
        f"mAP50={B_STAR['map50']:.4f}; historical uniform control "
        f"mean={HIST_UNIFORM['mean']:.4f} crit_recall={HIST_UNIFORM['critical_recall']:.4f}"
    )
    print(
        "[loc] next: run the official metric (handoff R3) and compare against the "
        "uniform control:\n"
        f"    python scripts/eval_multIou.py --weights {best} "
        f"--data {args.data} --out runs/multi_iou/{args.name}_official.json"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
