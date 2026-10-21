#!/usr/bin/env python
"""
eval_objectness_replay.py -- same-protocol image-level binary replay for the
objectness head (Stage 2E, optional completion of 4.9).

4.9's objectness conclusion so far rests on the B-gate (detection level):
matching recall, the objectness head cuts cross-domain FPR 2-6x. The
max-class-confidence gate, by contrast, was validated at the image level on
the exact replay protocol of 4.12 (957 val images = 583 defective + 374 clean,
141 AITEX cross-domain negatives). This script puts the objectness head on
that SAME protocol so the two scores are compared on identical data, identical
square letterbox pipeline and identical tau grid.

One model pass yields both scores per image:
  * max class confidence  = max over 300 queries of y[..., 4]
  * max objectness        = max over 300 queries of sigmoid(dec_obj)
so the comparison is same-model / same-queries, isolating only the score.

Usage:
    python scripts/eval_objectness_replay.py \
        --weights runs/rtdetr-objectness/obj_probe/weights/best.pt \
        --out runs/calibration/objectness_replay.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from ultralytics.cfg import get_cfg
from ultralytics.data import build_dataloader
from ultralytics.data.utils import check_det_dataset
from ultralytics.models.rtdetr.val import RTDETRDataset

from backend.training.objectness import load_objectness_model


def has_box(label_path: Path) -> bool:
    try:
        with open(label_path, "r", encoding="utf-8") as f:
            return any(line.strip() for line in f)
    except FileNotFoundError:
        return False


def img_to_label(im_file: str) -> Path:
    p = Path(im_file)
    parts = list(p.parts)
    for i, seg in enumerate(parts):
        if seg == "images":
            parts[i] = "labels"
            break
    return Path(*parts).with_suffix(".txt")


@torch.no_grad()
def raw_dual_scores(rtdetr: torch.nn.Module, img: torch.Tensor, device: torch.device):
    """Per-image (max class conf, max objectness) arrays, each (B,)."""
    out = rtdetr(img.to(device))
    y = out[0] if isinstance(out, (tuple, list)) else out  # (B, 300, 6)
    max_conf = y[..., 4].max(dim=1).values.float().cpu().numpy()

    x = out[1] if isinstance(out, (tuple, list)) and len(out) > 1 else None
    dec_obj = x[2] if isinstance(x, (tuple, list)) and len(x) >= 3 else None
    if dec_obj is None:
        max_obj = np.full(max_conf.shape, np.nan)
    else:
        max_obj = (
            torch.sigmoid(dec_obj).max(dim=1).values.squeeze(-1).float().cpu().numpy()
        )
    return max_conf, max_obj


@torch.no_grad()
def collect_val(rtdetr, data_dict, args, device):
    """Return (conf, obj, is_defective) aligned per val image, dataset order."""
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
    confs, objs, flags = [], [], []
    for batch in loader:
        img = batch["img"].float() / 255.0
        c, o = raw_dual_scores(rtdetr, img, device)
        confs.append(c)
        objs.append(o)
        flags.extend(has_box(img_to_label(f)) for f in batch["im_file"])
    return (np.concatenate(confs), np.concatenate(objs), np.asarray(flags, dtype=bool))


def load_stretch(img_path: Path, imgsz: int) -> torch.Tensor:
    import cv2

    im = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if im is None:
        raise FileNotFoundError(f"cannot read {img_path}")
    im = cv2.resize(im, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR)
    t = torch.from_numpy(im).permute(2, 0, 1).contiguous().unsqueeze(0)
    return t.float() / 255.0


@torch.no_grad()
def collect_ait_ex(rtdetr, neg_dir: Path, args, device):
    """AITEX no-defect images -> sample-name -> (conf, obj) arrays."""
    if not neg_dir.exists():
        return {}
    samples: dict = {}
    for sub in sorted(p for p in neg_dir.iterdir() if p.is_dir()):
        pngs = (
            sorted(sub.glob("*.png"))
            + sorted(sub.glob("*.jpg"))
            + sorted(sub.glob("*.jpeg"))
        )
        cs, os_ = [], []
        for p in pngs:
            c, o = raw_dual_scores(rtdetr, load_stretch(p, args.imgsz), device)
            cs.append(float(c[0]))
            os_.append(float(o[0]))
        if cs:
            samples[sub.name] = (np.asarray(cs), np.asarray(os_))
    return samples


def auc_rank(scores: np.ndarray, labels: np.ndarray) -> float:
    """Threshold-free AUC via rank sum (ties handled by average ranks)."""
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=float)
    sorted_s = scores[order]
    i = 0
    while i < len(scores):
        j = i
        while j + 1 < len(scores) and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        ranks[order[i : j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    pos = labels.sum()
    neg = len(labels) - pos
    if pos == 0 or neg == 0:
        return float("nan")
    return float((ranks[labels].sum() - pos * (pos + 1) / 2.0) / (pos * neg))


def confusion(conf: np.ndarray, gt: np.ndarray, tau: float) -> dict:
    pred = conf > tau
    tp = int((pred & gt).sum())
    fp = int((pred & ~gt).sum())
    tn = int((~pred & ~gt).sum())
    fn = int((~pred & gt).sum())
    p, n = int(gt.sum()), int((~gt).sum())
    return {
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn,
        "TPR_recall": round(tp / p, 4) if p else None,
        "FPR": round(fp / n, 4) if n else None,
        "precision": round(tp / (tp + fp), 4) if tp + fp else None,
        "accuracy": round((tp + tn) / (p + n), 4),
    }


def sweep(scores: np.ndarray, gt: np.ndarray, ait_scores: np.ndarray, args) -> dict:
    taus = np.round(np.arange(args.tau_min, args.tau_max + 1e-9, args.tau_step), 4)
    rows = []
    for tau in taus:
        c = confusion(scores, gt, float(tau))
        c["tau"] = float(tau)
        c["ait_fpr"] = (
            round(float((ait_scores > tau).mean()), 4) if ait_scores.size else None
        )
        rows.append(c)
    vals = [(r["accuracy"], r["tau"]) for r in rows if r.get("accuracy") is not None]
    best_acc_tau = max(vals)[1] if vals else None
    best_row = next((r for r in rows if r["tau"] == best_acc_tau), None)
    tau90 = next((r["tau"] for r in rows if (r["TPR_recall"] or 0) >= 0.90), None)
    return {
        "AUC": round(auc_rank(scores, gt), 4),
        "best_accuracy": best_row["accuracy"] if best_row else None,
        "best_tau": best_acc_tau,
        "recall_at_best_tau": best_row["TPR_recall"] if best_row else None,
        "fpr_at_best_tau": best_row["FPR"] if best_row else None,
        "tau90": tau90,
        "ait_fpr_at_best_tau": best_row["ait_fpr"] if best_row else None,
        "mean_defective": round(float(scores[gt].mean()), 4),
        "mean_no_defect": round(float(scores[~gt].mean()), 4),
        "rows": rows,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--weights",
        required=True,
        help="objectness-head checkpoint (best.pt of the probe run)",
    )
    ap.add_argument("--data", default="data/tianchi/data.yaml")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="0")
    ap.add_argument("--ait-ex-dir", default="shujuji_AITEX/NODefect_images")
    ap.add_argument("--out", default="runs/calibration/objectness_replay.json")
    ap.add_argument("--tau-min", type=float, default=0.05)
    ap.add_argument("--tau-max", type=float, default=0.95)
    ap.add_argument("--tau-step", type=float, default=0.05)
    args = ap.parse_args()

    device = torch.device("cpu")
    if args.device.isdigit() and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.device}")

    model, _head = load_objectness_model(args.weights, args.device)
    rtdetr = model.model.to(device).eval()

    data_dict = check_det_dataset(args.data)
    val_conf, val_obj, val_gt = collect_val(rtdetr, data_dict, args, device)
    n_pos, n_neg = int(val_gt.sum()), int((~val_gt).sum())
    if not np.isfinite(val_obj).all():
        n_nan = int((~np.isfinite(val_obj)).sum())
        print(
            f"[WARN] objectness score missing on {n_nan}/{val_obj.size} val images "
            f"(dec_obj aux output not returned)"
        )
    print(f"[obj-replay] val={val_conf.size} defective={n_pos} no-defect={n_neg}")

    ait = collect_ait_ex(rtdetr, Path(args.ait_ex_dir), args, device)
    ait_conf = np.concatenate([v[0] for v in ait.values()]) if ait else np.zeros(0)
    ait_obj = np.concatenate([v[1] for v in ait.values()]) if ait else np.zeros(0)
    if ait_conf.size:
        print(f"[obj-replay] AITEX cross-domain negatives={ait_conf.size}")

    rep = {
        "weights": str(Path(args.weights).resolve()),
        "protocol": "identical to scripts/eval_binary_replay.py (square letterbox, "
        "300 queries, tau grid 0.05-0.95 step 0.05)",
        "n_val": int(val_conf.size),
        "n_defective": n_pos,
        "n_no_defect": n_neg,
        "n_ait_ex": int(ait_conf.size),
        "max_class_conf": sweep(val_conf, val_gt, ait_conf, args),
        "objectness": sweep(val_obj, val_gt, ait_obj, args),
    }
    for key in ("max_class_conf", "objectness"):
        s = rep[key]
        print(
            f"[obj-replay] {key}: AUC={s['AUC']} best_acc={s['best_accuracy']} "
            f"@tau={s['best_tau']} (recall={s['recall_at_best_tau']}, "
            f"fpr={s['fpr_at_best_tau']}, ait_fpr={s['ait_fpr_at_best_tau']})"
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"report written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
