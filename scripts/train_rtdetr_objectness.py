"""
Train RT-DETR with a cost-sensitive objectness head — Direction 3 (calibration).

The two-level diagnosis (``critical-class-miss-not-confusion``) showed the
critical thin defects are *missed below threshold*, and the B-口径 experiment
(``eval_binary_detection.py``) showed the standard "max class confidence" gate
cannot separate defect from clean fabric (cross-domain FPR ~0.99). Both stem
from DETR having no calibrated background score: RT-DETR's varifocal
``gt_score`` == predicted IoU, so thin defects get low confidence *targets*,
and there is no signal for "this query is background".

This script attaches a dedicated objectness head (``Linear(hd, 1)`` on the
decoder's final hidden state) and trains it with a **cost-sensitive** BCE so
that queries matched to a critical class are weighted up and background queries
are weighted down. It supports a two-stage schedule:

* Stage 1 ``--probe-only`` (default): freeze every existing parameter, train
  only the objectness head. This is near-guaranteed stable — it cannot damage
  the detector and directly answers whether a calibrated objectness gate beats
  max-class-confidence at the B-口径 level.
* Stage 2 (omit ``--probe-only``): joint fine-tune at low LR (riskier, run
  after Stage 1 confirms the objectness signal is useful).

At the end it reports (a) the mAP50 / critical-recall regression check and
(b) the B-口径 comparison — recall/FPR of the objectness gate vs the
max-class-confidence gate over the Tianchi val (957 positives) and AITEX
``NODefect_images`` (cross-domain negatives).

Usage:
    # Stage 1 — linear probe (safe, recommended first)
    python scripts/train_rtdetr_objectness.py --probe-only --epochs 3

    # Stage 2 — joint fine-tune
    python scripts/train_rtdetr_objectness.py --epochs 3 --lr0 1e-4 --name obj_joint
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.data import build_dataloader
from ultralytics.data.utils import check_det_dataset
from ultralytics.models.rtdetr.val import RTDETRDataset, RTDETRValidator
import ultralytics.models.utils.loss as _detr_loss

from backend.training.objectness import (
    ObjectnessRTDETRDetectionLoss,
    ObjectnessRTDETRTrainer,
    load_objectness_model,
    set_default_objectness,
    set_probe_only,
)

# Critical / secondary class ids under the 18-class taxonomy (see data.yaml).
CRITICAL_CLASS_IDS = [0, 9, 15]  # hole, broken_warp, broken_spandex
CRITICAL_CLASS_NAMES = ["hole", "broken_warp", "broken_spandex"]
TRACKED_NAMES = [*CRITICAL_CLASS_NAMES, "star_skip", "weave_defect"]

# Reference baseline (18-class tianchi18_full best.pt), measured with the same
# RTDETRValidator + per-class recall extraction used here.
BASELINE = {
    "map50": 0.541,
    "critical_recall": 0.5743,
    "per_class": {
        "hole": 0.718,
        "broken_warp": 0.521,
        "broken_spandex": 0.484,
        "star_skip": 0.686,
        "weave_defect": 0.487,
    },
}


def validate(
    rtdetr: torch.nn.Module, args: argparse.Namespace, device_str: str
) -> dict:
    """Run noise-free RT-DETR validation and return overall + per-class recall."""
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
        f"[obj] {tag}: mAP50={m['map50']:.4f} mAP50-95={m['map50_95']:.4f} "
        f"P={m['precision']:.4f} R={m['recall']:.4f} crit_recall={m['critical_recall']:.4f}\n"
        f"      {tracked}"
    )


@torch.no_grad()
def raw_scores(rtdetr: torch.nn.Module, img: torch.Tensor, device: torch.device):
    """Return per-image (max class conf, max objectness) arrays, each (B,)."""
    out = rtdetr(img.to(device))
    y = out[0] if isinstance(out, (tuple, list)) else out  # (B, 300, 6)
    max_conf = y[..., 4].max(dim=1).values.cpu().numpy()

    x = out[1] if isinstance(out, (tuple, list)) and len(out) > 1 else None
    dec_obj = x[2] if isinstance(x, (tuple, list)) and len(x) >= 3 else None
    if dec_obj is None:
        max_obj = np.full(max_conf.shape, np.nan)
    else:
        max_obj = torch.sigmoid(dec_obj).max(dim=1).values.squeeze(-1).cpu().numpy()
    return max_conf, max_obj


def collect_scores(rtdetr, data_dict, args, device):
    """Run the model over val + AITEX and return per-image (class_conf, obj)."""
    cfg = get_cfg(overrides={"imgsz": args.imgsz, "rect": False, "cache": False})
    dataset = RTDETRDataset(
        img_path=data_dict["val"],
        imgsz=args.imgsz,
        batch_size=args.batch,
        augment=False,
        hyp=cfg,
        rect=False,
        cache=None,
        prefix="val: ",
        data=data_dict,
    )
    loader = build_dataloader(dataset, args.batch, workers=0, shuffle=False)
    val_conf: list = []
    val_obj: list = []
    for i, batch in enumerate(loader):
        img = batch["img"].float() / 255.0
        c, o = raw_scores(rtdetr, img, device)
        val_conf.append(c)
        val_obj.append(o)
        if args.val_limit and (i + 1) * args.batch >= args.val_limit:
            break

    ait_conf: list = []
    ait_obj: list = []
    neg_dir = Path(args.ait_ex_dir)
    for sub in sorted(p for p in neg_dir.iterdir() if p.is_dir()):
        pngs = (
            sorted(sub.glob("*.png"))
            + sorted(sub.glob("*.jpg"))
            + sorted(sub.glob("*.jpeg"))
        )
        for j, p in enumerate(pngs):
            if args.ait_limit and j >= args.ait_limit:
                break
            c, o = raw_scores(rtdetr, load_stretch(p, args.imgsz), device)
            ait_conf.append(float(c[0]))
            ait_obj.append(float(o[0]))

    return (
        np.concatenate(val_conf)[: args.val_limit or None],
        np.concatenate(val_obj)[: args.val_limit or None],
        np.asarray(ait_conf),
        np.asarray(ait_obj),
    )


def load_stretch(img_path: Path, imgsz: int) -> torch.Tensor:
    """Load one image and stretch it to ``imgsz x imgsz`` (BGR, float/255)."""
    import cv2

    im = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if im is None:
        raise FileNotFoundError(f"cannot read {img_path}")
    im = cv2.resize(im, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR)
    t = torch.from_numpy(im).permute(2, 0, 1).contiguous().unsqueeze(0)
    return t.float() / 255.0


def report_gate(
    tag: str, val_scores: np.ndarray, ait_scores: np.ndarray, target_recall: float
) -> None:
    """Report the recall/FPR trade-off for one gate over a tau sweep."""
    val_scores = np.asarray(val_scores, dtype=float)
    ait_scores = np.asarray(ait_scores, dtype=float)
    n_pos, n_neg = int(val_scores.size), int(ait_scores.size)
    taus = np.arange(0.05, 0.91, 0.05)

    print(f"\n[obj] {tag} gate (recall on {n_pos} val, FPR on {n_neg} AITEX):")
    print(f"{'tau':>6} {'recall':>8} {'fpr':>8}")
    achieved = None
    for tau in taus:
        recall = float((val_scores > tau).mean()) if n_pos else 0.0
        fpr = float((ait_scores > tau).mean()) if n_neg else 0.0
        mark = ""
        if achieved is None and recall >= target_recall:
            achieved = float(tau)
            mark = "  <- target"
        print(f"{tau:>6.2f} {recall:>8.4f} {fpr:>8.4f}{mark}")

    if achieved is None:
        print(f"      target recall {target_recall:.2f} NOT reached")
        return
    recall = float((val_scores > achieved).mean())
    fpr = float((ait_scores > achieved).mean()) if n_neg else 0.0
    print(
        f"      min tau for recall>={target_recall:.2f}: {achieved:.2f} "
        f"-> recall={recall:.4f} FPR={fpr:.4f}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--weights", default="runs/rtdetr/tianchi18_full/weights/best.pt"
    )
    parser.add_argument("--data", default="data/tianchi/data.yaml")
    parser.add_argument("--epochs", type=int, default=3, help="Train epochs")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument(
        "--lr0", type=float, default=0.001, help="Probe LR (higher than fine-tune)"
    )
    parser.add_argument(
        "--amp",
        type=lambda s: str(s).lower() in ("true", "1", "yes", "on"),
        default=False,
    )
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--project", default="runs/rtdetr-objectness")
    parser.add_argument("--name", default="obj_probe")
    parser.add_argument(
        "--probe-only",
        action="store_true",
        help="Freeze all params except the objectness head (Stage 1 linear probe)",
    )
    parser.add_argument("--objectness-gain", type=float, default=1.0)
    parser.add_argument("--w-crit", type=float, default=3.0)
    parser.add_argument("--w-bg", type=float, default=0.1)
    parser.add_argument(
        "--no-val", action="store_true", help="Skip the final per-class report"
    )
    parser.add_argument(
        "--no-gate", action="store_true", help="Skip the B-口径 gate eval"
    )
    parser.add_argument("--val-limit", type=int, default=0, help="0 = all val images")
    parser.add_argument(
        "--ait-limit", type=int, default=0, help="0 = all AITEX negatives"
    )
    parser.add_argument("--ait-ex-dir", default="shujuji_AITEX/NODefect_images")
    parser.add_argument("--target-recall", type=float, default=0.90)
    args = parser.parse_args()

    args.project = os.path.abspath(args.project)

    set_default_objectness(
        gain=args.objectness_gain, w_crit=args.w_crit, w_bg=args.w_bg
    )
    set_probe_only(args.probe_only)
    _detr_loss.RTDETRDetectionLoss = ObjectnessRTDETRDetectionLoss

    print(
        f"[obj] mode={'probe-only' if args.probe_only else 'joint'} "
        f"gain={args.objectness_gain} w_crit={args.w_crit} w_bg={args.w_bg}"
    )

    model = YOLO(args.weights)
    model.train(
        trainer=ObjectnessRTDETRTrainer,
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
        seed=42,
    )

    best = Path(args.project) / args.name / "weights" / "best.pt"
    last = Path(args.project) / args.name / "weights" / "last.pt"

    # ---- regression check: mAP50 / critical recall on the frozen detector ----
    if not args.no_val:
        final_model = YOLO(str(best))
        rtdetr = final_model.model
        rtdetr.eval()
        m = validate(rtdetr, args, args.device)
        _print_metrics("best", m)
        print(
            f"[obj] regression: crit_recall {BASELINE['critical_recall']:.4f} -> "
            f"{m['critical_recall']:.4f} (probe-only should be ~unchanged)"
        )

    # ---- B-口径 gate comparison ----
    # In probe-only the detector is frozen, so detection fitness (mAP) is flat
    # and best.pt holds an *early* head; last.pt holds the final head. Joint mode
    # should use best.pt (detection fitness varies and selects the best head too).
    if not args.no_gate:
        device = torch.device("cpu")
        if args.device.isdigit() and torch.cuda.is_available():
            device = torch.device(f"cuda:{args.device}")
        gate_ckpt = best if not args.probe_only else last
        obj_model, _ = load_objectness_model(str(gate_ckpt), args.device)
        rtdetr = obj_model.model
        rtdetr.to(device)
        rtdetr.eval()

        data_dict = check_det_dataset(args.data)
        val_conf, val_obj, ait_conf, ait_obj = collect_scores(
            rtdetr, data_dict, args, device
        )

        report_gate("max-class-confidence", val_conf, ait_conf, args.target_recall)
        report_gate("objectness         ", val_obj, ait_obj, args.target_recall)

    print(
        "\n[obj] next: compare objectness FPR vs class-conf FPR at the target recall; "
        "if objectness is meaningfully lower, promote it to the deployment gate "
        "(B 口径) and optionally run Stage 2 joint fine-tune."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
