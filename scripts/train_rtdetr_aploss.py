"""
M1: fine-tune RT-DETR with a Smooth-AP proxy loss on matched-query scores.

Hypothesis (handoff section 10)
    The diagnosed bottleneck is ranking, not detection capability: at the
    confidence floor 89%..100% of critical-class GTs are still present, but the
    TP/FP separability (AUC 0.76..0.82) decides the PR curve. Every historical
    intervention re-weighted or replaced a loss; M1 instead adds a differentiable
    surrogate of the ranking metric itself (Smooth-AP, Rodriguez et al., 2020) on
    the Hungarian-matched queries, while GIoU/L1/classification stay untouched.

Pre-registered criterion (handoff section 10)
    ``mean > B* + 0.005`` with ``B* = 0.6349`` (locked mosaic1280 baseline), judged
    on 3 seeds (42 / 123 / 2024).

Failure branch (handoff section 10)
    M1 negative -> "detection mAP on this data/architecture is near its ceiling"
    becomes a strong conclusion; value moves to the decision layer (A2) plus the
    diagnostic negative-result chain. No further loss-side re-weighting arm is
    opened. If M1 is positive, re-run the then-strongest reward arm on the M1
    checkpoint (the reward headroom may reappear on a better PR curve).

Protocol (handoff R4, identical to ``train_rtdetr_boxweighted.py`` except lr0)
    mosaic1280 ``best.pt`` start | 5 epochs | batch 4 | imgsz 1280 (square
    letterbox) | lr0 5e-5 (handoff M1: "lr0 5e-5 起") | amp OFF | mosaic 0 |
    workers 0 | seed 42.

    ``--beta 0`` is an exact uniform control: the criterion still computes the
    proxy (so the AP trace is logged), but ``loss_ap == -0.0`` and its gradient is
    identically zero, so optimisation is bit-identical to a plain fine-tune.

Artifacts written to ``<project>/<name>/``
    - ``design.json``: hypothesis, pre-registered criterion, failure branch, full
      protocol, beta/temperature, and (after training) baseline vs final metrics.
    - ``ap_history.csv``: per-step ``loss_ap``, mean Smooth-AP proxy and the
      matched / unmatched query counts, all detached.

Usage:
    # Arm (beta 0.1, temperature 0.5)
    python scripts/train_rtdetr_aploss.py --epochs 5 --name m1_aploss_s42

    # Exact uniform control
    python scripts/train_rtdetr_aploss.py --beta 0 --epochs 5 --name m1_aploss_beta0_s42

    # Then the official metric (the only numbers that count, handoff R3):
    python scripts/eval_multIou.py \
        --weights runs/rtdetr-ap/m1_aploss_s42/weights/best.pt \
        --data data/tianchi/data.yaml \
        --out runs/multi_iou/m1_aploss_s42_official.json
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

# Allow `python scripts/train_rtdetr_aploss.py` to import the `backend` package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from ultralytics import YOLO
from ultralytics.models.rtdetr.val import RTDETRValidator

from backend.training.aploss import attach_ap_criterion

# Critical / secondary class ids under the 18-class taxonomy (see data.yaml).
CRITICAL_CLASS_IDS = [0, 9, 15]  # hole, broken_warp, broken_spandex
CRITICAL_CLASS_NAMES = ["hole", "broken_warp", "broken_spandex"]
SECONDARY_CLASS_IDS = [14, 17]  # star_skip, weave_defect
TRACKED_NAMES = [*CRITICAL_CLASS_NAMES, "star_skip", "weave_defect"]

# Handoff R4 / section 10 references. These are NOT measured by this script; they
# are printed as context for the pre-registered comparison.
B_STAR = {"mean": 0.6349, "map50": 0.5917}  # mosaic1280 best.pt, locked baseline
HIST_UNIFORM = {"mean": 0.6158, "critical_recall": 0.6260}  # strong_r2_uniform

#: Default start checkpoint: the locked mosaic1280 baseline (handoff R4).
DEFAULT_WEIGHTS = "runs/rtdetr/tianchi18_mosaic1280/weights/best.pt"

FORMULA = (
    "p_i = z[i, y_i] (GT-class logit) for matched i, p_k = max_c z[k, c] "
    "(top-class logit) for unmatched k; sigma(x) = 1 / (1 + exp(-x / T)); "
    "num_i = 1 + sum_{j in P, j != i} sigma(p_j - p_i); "
    "den_i = num_i + sum_{k in N} sigma(p_k - p_i); "
    "AP_b = mean_{i in P} num_i / den_i; loss_ap = -beta * mean_b AP_b"
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
        f"[ap] {tag}: mAP50={m['map50']:.4f} mAP50-95={m['map50_95']:.4f} "
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
        print(f"[ap] baseline <- {path}")
        _print_metrics("baseline", data)
        return data
    if args.baseline == "skip":
        print("[ap] baseline skipped (--baseline skip): table will show deltas as n/a")
        return None
    print(f"[ap] measuring baseline on the start checkpoint: {args.weights}")
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
        "arm": "M1",
        "module": "backend.training.aploss",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "hypothesis": (
            "The bottleneck is score ranking (TP/FP AUC 0.76..0.82; 89%..100% of "
            "critical GTs reachable at the confidence floor), so directly "
            "optimising a differentiable AP surrogate on the matched-query score "
            "order can move the PR curve where loss re-weighting only slid along "
            "it."
        ),
        "prereg_criterion": "转正 = 官方 mean > B* + 0.005（B* = 0.6349）；三种子 42/123/2024。",
        "failure_branch": (
            "M1 阴性 → “检测 mAP 在该数据/架构上已近天花板”成为强结论，项目价值主张"
            "转向决策层（A2）+ 诊断层（论文负结果链）+ 产线工作点；不再开新的"
            "损失侧重加权臂。"
        ),
        "difference_from_reweighting_arms": (
            "R2/B/NWD changed the weight or form of an existing term (zero-sum: "
            "mean -0.015 .. -0.029 for crit_recall +0.10 .. +0.18); M1 adds a "
            "ranking-metric surrogate and leaves every supervised term/gain intact."
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
            "temp": args.temp,
        },
        "control": {
            "beta": 0.0,
            "note": (
                "exact uniform control: the criterion still logs the AP proxy, "
                "but loss_ap is -0.0 and its gradient is identically zero"
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
    """Write ``ap_history.csv`` from the criterion's detached per-step history."""
    path = Path(save_dir) / "ap_history.csv"
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["step", "loss_ap", "ap_proxy", "n_pos", "n_neg"])
        for i, loss_ap in enumerate(criterion.loss_ap_history):
            writer.writerow(
                [
                    i + 1,
                    round(loss_ap, 6),
                    round(criterion.ap_history[i], 6),
                    criterion.n_pos_history[i],
                    criterion.n_neg_history[i],
                ]
            )
    print(f"[ap] wrote {path} ({len(criterion.loss_ap_history)} steps)")


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
        default=0.00005,
        help="Fine-tune LR (handoff M1: start at 5e-5, half the R4 default)",
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
        help="AP-proxy weight (0 = exact uniform control)",
    )
    parser.add_argument(
        "--temp",
        type=float,
        default=0.5,
        help="Smoothing temperature of the pairwise logistic ranking",
    )
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--project", default="runs/rtdetr-ap", help="Output project dir")
    parser.add_argument("--name", default="m1_aploss_s42", help="Run name")
    parser.add_argument(
        "--log-every", type=int, default=50, help="Print per-step AP proxy every N steps"
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
    if args.temp <= 0:
        parser.error("--temp must be positive")
    if args.beta < 0:
        parser.error("--beta must be >= 0")

    print(f"[ap] arm=M1 beta={args.beta} temp={args.temp} weights={args.weights}")
    print(f"[ap] formula: {FORMULA}")
    if args.beta == 0.0:
        print(
            "[ap] beta=0 -> uniform control: loss_ap is identically zero and "
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
        criterion = attach_ap_criterion(
            target,
            beta=args.beta,
            temp=args.temp,
        )
        criterion_ref["criterion"] = criterion
        save_dir = Path(getattr(trainer, "save_dir", run_dir))
        _write_design(save_dir, design)
        nc = int(getattr(target, "nc", target.model[-1].nc))
        print(
            f"[ap] attached SmoothAPRTDETRDetectionLoss (beta={args.beta}, "
            f"temp={args.temp}, nc={nc}); pre-registration -> "
            f"{save_dir / 'design.json'}"
        )

    def _on_train_batch_end(trainer) -> None:
        criterion = criterion_ref["criterion"]
        if criterion is None or args.log_every <= 0:
            return
        step = len(criterion.loss_ap_history)
        if step == 0 or step % args.log_every != 0:
            return
        print(
            f"[ap] step {step}: ap_proxy={criterion.ap_history[-1]:.4f} "
            f"n_pos={criterion.n_pos_history[-1]} n_neg={criterion.n_neg_history[-1]} "
            f"loss_ap={criterion.loss_ap_history[-1]:+.5f}"
        )

    def _on_train_end(trainer) -> None:
        criterion = criterion_ref["criterion"]
        save_dir = Path(getattr(trainer, "save_dir", run_dir))
        if criterion is not None and criterion.loss_ap_history:
            _write_history(save_dir, criterion)
            design["proxy_trace"] = {
                "steps": len(criterion.loss_ap_history),
                "mean_ap_proxy": round(
                    sum(criterion.ap_history) / len(criterion.ap_history), 6
                ),
                "mean_loss_ap": round(
                    sum(criterion.loss_ap_history) / len(criterion.loss_ap_history), 6
                ),
                "n_pos_total": sum(criterion.n_pos_history),
                "n_neg_total": sum(criterion.n_neg_history),
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

    print("\n[ap] baseline vs aploss (per-class recall @ max-F1):")
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
        f"\n[ap] handoff references (not re-measured): B* mean={B_STAR['mean']:.4f} "
        f"mAP50={B_STAR['map50']:.4f}; historical uniform control "
        f"mean={HIST_UNIFORM['mean']:.4f} crit_recall={HIST_UNIFORM['critical_recall']:.4f}"
    )
    print(
        "[ap] next: run the official metric (handoff R3) and compare against the "
        "uniform control:\n"
        f"    python scripts/eval_multIou.py --weights {best} "
        f"--data {args.data} --out runs/multi_iou/{args.name}_official.json"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
