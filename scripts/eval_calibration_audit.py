#!/usr/bin/env python
"""
eval_calibration_audit.py -- confidence calibration audit for RT-DETR on fabric defects.

Answers four questions the paper's alarm framework (4.12) currently assumes:

  Q1  Is the model's max-class confidence calibrated at all?
      -> ECE (10 equal-width bins), NLL, Brier on the raw val predictions.
  Q2  Does temperature scaling help? Global / per-family / per-size-bin,
      all K-fold cross-fitted (T fitted on K-1 folds, ECE measured on the
      held-out fold) so the reported ECE is never in-sample.
  Q3  Does calibration change TP-vs-FP separability? It must not (any
      monotone transform preserves ranking) -- reported as AUC(score) before
      and after, which *quantifies* the paper's "true and false detections are
      inseparable by confidence" claim instead of asserting it.
  Q4  Where is the miscalibration concentrated? Per-class ECE and per-class
      TP/FP confidence separation for the critical classes.

Protocol: identical collection to eval_confusion_decomp.py (square letterbox,
official RT-DETR val protocol, 300 queries, conf floor 0.001). TP labels follow
the r_curve convention (utils/metrics.py match_predictions): GT-major greedy,
class-aware, IoU >= 0.5, each prediction usable once. Labels are computed over
all 300 queries independent of conf (exactly like the PR-curve matching), so the
confidence sweep is a pure post-hoc re-labelling.

Usage:
    python scripts/eval_calibration_audit.py --weights runs/rtdetr/tianchi18_full/weights/best.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.data import build_dataloader
from ultralytics.data.utils import check_det_dataset
from ultralytics.models.rtdetr.val import RTDETRDataset
import yaml

CRIT = ["hole", "broken_warp", "broken_spandex"]


# ---------------------------------------------------------------- geometry
def iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)))
    area_a = (boxes_a[:, 2] - boxes_a[:, 0]).clip(0) * (
        boxes_a[:, 3] - boxes_a[:, 1]
    ).clip(0)
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]).clip(0) * (
        boxes_b[:, 3] - boxes_b[:, 1]
    ).clip(0)
    lt = np.maximum(boxes_a[:, None, :2], boxes_b[None, :, :2])
    rb = np.minimum(boxes_a[:, None, 2:], boxes_b[None, :, 2:])
    wh = (rb - lt).clip(0)
    inter = wh[..., 0] * wh[..., 1]
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def cxcywh_to_xyxy(b: np.ndarray) -> np.ndarray:
    out = np.empty_like(b)
    out[:, 0] = b[:, 0] - b[:, 2] / 2
    out[:, 1] = b[:, 1] - b[:, 3] / 2
    out[:, 2] = b[:, 0] + b[:, 2] / 2
    out[:, 3] = b[:, 1] + b[:, 3] / 2
    return out


def match_tp_fp(pcls, gcls, ious, iou_thr=0.5):
    """r_curve convention: GT-major greedy, class-aware, each prediction once.

    Returns a boolean array (P,) marking predictions that are TPs.
    """
    correct = np.zeros(len(pcls), dtype=bool)
    if len(gcls) == 0:
        return correct
    iou = ious * (pcls[:, None] == gcls[None, :])  # class-aware mask
    for gi in range(len(gcls)):
        cand = np.where(iou[:, gi] >= iou_thr)[0]
        if len(cand) == 0:
            continue
        best = cand[int(np.argmax(iou[cand, gi]))]
        correct[best] = True
        iou[best, :] = 0.0  # prediction consumed
    return correct


# ---------------------------------------------------------------- metrics
def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def ece_equal_width(
    conf: np.ndarray, tp: np.ndarray, n_bins: int = 10, conf_lo: float = 0.0
) -> tuple:
    """Expected calibration error, equal-width bins over [conf_lo, 1]."""
    sel = conf >= conf_lo
    c, t = conf[sel], tp[sel]
    if len(c) == 0:
        return float("nan"), []
    edges = np.linspace(conf_lo, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(c, edges[1:-1], right=False), 0, n_bins - 1)
    ece, rows = 0.0, []
    for b in range(n_bins):
        m = idx == b
        n = int(m.sum())
        if n == 0:
            continue
        acc = float(t[m].mean())
        mconf = float(c[m].mean())
        ece += (n / len(c)) * abs(acc - mconf)
        rows.append(
            {
                "bin": b,
                "lo": float(edges[b]),
                "hi": float(edges[b + 1]),
                "n": n,
                "conf": mconf,
                "acc": acc,
            }
        )
    return float(ece), rows


def nll(conf: np.ndarray, tp: np.ndarray) -> float:
    p = np.clip(conf, 1e-6, 1 - 1e-6)
    return float(-np.mean(tp * np.log(p) + (1 - tp) * np.log(1 - p)))


def brier(conf: np.ndarray, tp: np.ndarray) -> float:
    return float(np.mean((conf - tp) ** 2))


def auc_score(scores: np.ndarray, labels: np.ndarray) -> float:
    """Rank AUC (P(score_pos > score_neg), ties 0.5)."""
    pos = labels == 1
    neg = ~pos
    if pos.sum() == 0 or neg.sum() == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    s = scores[order]
    ranks = np.empty(len(s), dtype=float)
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        ranks[i : j + 1] = 0.5 * (i + j) + 1.0  # average rank, 1-based
        i = j + 1
    r = np.empty(len(s), dtype=float)
    r[order] = ranks
    return float(
        (r[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * neg.sum())
    )


def fit_temperature(
    conf: np.ndarray, tp: np.ndarray, grid: np.ndarray | None = None
) -> float:
    """Sigmoid temperature scaling: minimise NLL over a 1-D grid (NLL(T) is
    smooth and unimodal for TS)."""
    if grid is None:
        grid = np.linspace(0.25, 5.0, 96)
    z = logit(conf)
    best_t, best_n = 1.0, np.inf
    for t in grid:
        p = sigmoid(z / t)
        v = nll(p, tp)
        if v < best_n:
            best_n, best_t = v, float(t)
    return best_t


def fit_isotonic(conf: np.ndarray, tp: np.ndarray) -> tuple:
    """Pool-adjacent-violators monotone regression on (conf, tp).

    Returns (x, y) step-function knots; use with np.interp. Unlike TS this is
    nonparametric and CAN reshape an arbitrary monotone reliability curve --
    the natural test of whether the operating region's miscalibration is a
    temperature offset (fixable) or a curve-shape problem (not fixable by any
    1-parameter monotone map)."""
    order = np.argsort(conf, kind="mergesort")
    x = conf[order]
    y = tp[order].astype(float)
    w = np.ones(len(y)).tolist()
    v = y.tolist()
    groups = [[i] for i in range(len(y))]
    i = 0
    while i < len(v) - 1:
        if v[i] <= v[i + 1] + 1e-12:
            i += 1
            continue
        nw = w[i] + w[i + 1]
        nv = (w[i] * v[i] + w[i + 1] * v[i + 1]) / nw
        w[i], v[i] = nw, nv
        groups[i] = groups[i] + groups[i + 1]
        del w[i + 1], v[i + 1], groups[i + 1]
        if i > 0:
            i -= 1
    out_x = np.array([x[g[0]] for g in groups])
    out_y = np.array([v[k] for k in range(len(groups))])
    return out_x, out_y


def apply_t(conf: np.ndarray, t: float) -> np.ndarray:
    return sigmoid(logit(conf) / t)


# ---------------------------------------------------------------- collection
@torch.no_grad()
def collect(args):
    """Run val once; return flat prediction records + GT instance stats."""
    model = YOLO(args.weights)
    device = torch.device(
        args.device
        if args.device.startswith("cuda") or args.device == "cpu"
        else f"cuda:{args.device}"
    )
    rtdetr = model.model.to(device).eval()
    names = model.names

    data_dict = check_det_dataset(args.data)
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

    fam = {}
    fam_path = Path("runs/family_clustering/families.yaml")
    if fam_path.exists():
        fy = yaml.safe_load(fam_path.read_text(encoding="utf-8"))
        for fam_name, members in fy["families3"].items():
            for m in members:
                fam[m] = fam_name
    cls_family = {i: fam.get(n, "?") for i, n in names.items()}

    recs = []  # per-prediction records
    gt_stats = {c: dict(total=0, area_sum=0.0) for c in names.values()}
    img_seen = 0
    for batch in loader:
        img = batch["img"].to(device).float() / 255.0
        out = rtdetr(img)
        preds = (
            (out[0] if isinstance(out, (tuple, list)) else out).cpu().numpy()
        )  # (B,300,6)
        B = preds.shape[0]
        H, W = img.shape[2], img.shape[3]
        scale = np.array([W, H, W, H], dtype=float)
        for j in range(B):
            img_idx = img_seen + j
            p = preds[j]  # (300,6): cx,cy,w,h,conf,cls (normalized cxcywh)
            pboxes = cxcywh_to_xyxy(p[:, :4]) * scale
            pconf = p[:, 4]
            pcls = p[:, 5].astype(int)
            sel = batch["batch_idx"].cpu().numpy() == j
            gcls = batch["cls"].cpu().numpy().reshape(-1)[sel].astype(int)
            gboxes_n = batch["bboxes"].cpu().numpy().reshape(-1, 4)[sel]
            gxyxy = cxcywh_to_xyxy(gboxes_n) * scale if len(gcls) else np.zeros((0, 4))

            ious = iou_matrix(pboxes, gxyxy)
            tp = match_tp_fp(pcls, gcls, ious, args.iou_thr)

            keep = pconf >= args.conf_floor
            for k in np.where(keep)[0]:
                c = int(pcls[k])
                area = float(
                    max(pboxes[k, 2] - pboxes[k, 0], 0)
                    * max(pboxes[k, 3] - pboxes[k, 1], 0)
                )
                recs.append(
                    {
                        "img": img_idx,
                        "conf": float(pconf[k]),
                        "cls": c,
                        "cname": names[c],
                        "tp": bool(tp[k]),
                        "area": area,
                        "family": cls_family[c],
                    }
                )
            for gc, gb in zip(gcls, gxyxy, strict=False):
                cn = names[int(gc)]
                gt_stats[cn]["total"] += 1
                gt_stats[cn]["area_sum"] += float(
                    max(gb[2] - gb[0], 0) * max(gb[3] - gb[1], 0)
                )
        img_seen += B
    return recs, gt_stats, names, img_seen


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", default="data/tianchi/data.yaml")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--iou-thr", type=float, default=0.5)
    ap.add_argument("--conf-floor", type=float, default=0.001)
    ap.add_argument("--device", default="0")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--out", default="runs/calibration/audit_tianchi18.json")
    args = ap.parse_args()

    recs, gt_stats, names, n_imgs = collect(args)
    conf = np.array([r["conf"] for r in recs])
    tp = np.array([r["tp"] for r in recs], dtype=bool)
    cls = np.array([r["cls"] for r in recs])
    area = np.array([r["area"] for r in recs])
    family = np.array([r["family"] for r in recs])
    img = np.array([r["img"] for r in recs])
    n = len(recs)
    print(f"\n=== calibration audit: {args.weights} ===")
    print(
        f"predictions collected: {n} from {n_imgs} images "
        f"(TP={int(tp.sum())}, FP={int((~tp).sum())})"
    )

    # fold assignment by image (deterministic round-robin over image order)
    fold_of_img = np.arange(n_imgs) % args.folds
    fold = fold_of_img[img]

    # ---------------- Q1 raw calibration
    ece_raw, bins = ece_equal_width(conf, tp)
    ece_ops, _ = ece_equal_width(conf, tp, conf_lo=0.05)
    nll_raw, brier_raw = nll(conf, tp), brier(conf, tp)
    auc_raw = auc_score(conf, tp)
    print(
        f"\n[Q1] raw:  ECE={ece_raw:.4f}  ECE(conf>=0.05)={ece_ops:.4f}  "
        f"NLL={nll_raw:.4f}  Brier={brier_raw:.4f}  AUC={auc_raw:.4f}"
    )

    # ---------------- Q2 cross-fitted temperature scaling
    def crossfit(keys: np.ndarray | None = None, ops_only: bool = False):
        """Mean held-out ECE over K folds. keys=None -> single global T.

        ops_only=True restricts BOTH the temperature fit and the ECE
        evaluation to the operational region (conf >= 0.05). Returns
        (full-region ECE, ops-region ECE, mean fitted T)."""
        eces, eces_ops, ts = [], [], []
        for f in range(args.folds):
            te = fold == f
            tr = ~te
            if ops_only:
                tr = tr & (conf >= 0.05)
                te_ops = te & (conf >= 0.05)
            else:
                te_ops = te
            if keys is None:
                t = fit_temperature(conf[tr], tp[tr])
                p = apply_t(conf, t)
            else:
                p = conf.copy()  # identity fallback for groups without enough data
                for g in np.unique(keys[tr]):
                    gte = te & (keys == g)
                    gtr = tr & (keys == g)
                    if gtr.sum() < 50 or gte.sum() == 0:
                        continue
                    t = fit_temperature(conf[gtr], tp[gtr])
                    p[gte] = apply_t(conf[gte], t)
            e, _ = ece_equal_width(p[te], tp[te])
            eces.append(e)
            # ops-region ECE: bin by the CALIBRATED score, floor 0.05 on it
            if te_ops.sum():
                eo, _ = ece_equal_width(p[te_ops], tp[te_ops], conf_lo=0.05)
                eces_ops.append(eo)
            ts.append(t)
        return float(np.mean(eces)), float(np.mean(eces_ops)), float(np.mean(ts))

    ece_ts, ece_ts_ops, t_global = crossfit(None)
    print(
        f"[Q2] global TS (cross-fitted):       held-out ECE={ece_ts:.4f} "
        f"(ops-region {ece_ts_ops:.4f}, T={t_global:.2f})"
    )

    # per-family TS
    fam_key = np.array([r["family"] for r in recs])
    ece_fam, ece_fam_ops, _ = crossfit(fam_key)
    print(
        f"[Q2] per-family TS (cross-fitted):   held-out ECE={ece_fam:.4f} "
        f"(ops-region {ece_fam_ops:.4f})"
    )

    # per-size-bin TS (tertiles of prediction box area, computed on all records)
    q1, q2 = np.quantile(area, [1 / 3, 2 / 3])
    size_key = np.where(area <= q1, "small", np.where(area <= q2, "mid", "large"))
    ece_size, ece_size_ops, _ = crossfit(size_key)
    print(
        f"[Q2] per-size-bin TS (cross-fitted): held-out ECE={ece_size:.4f} "
        f"(ops-region {ece_size_ops:.4f})"
    )

    # ops-region-only fit: can ANY single temperature fix the operating region?
    ece_opsfit, ece_opsfit_ops, t_ops = crossfit(None, ops_only=True)
    print(
        f"[Q2] ops-region-only TS fit:         held-out ops ECE={ece_opsfit_ops:.4f} "
        f"(T={t_ops:.2f}; raw ops ECE={ece_ops:.4f})"
    )

    # nonparametric monotone recalibration (isotonic), cross-fitted.
    # The decisive test: if the ops-region miscalibration is a curve-shape
    # problem, isotonic (which CAN reshape any monotone reliability curve)
    # fixes it where TS cannot; if even isotonic fails, the miscalibration
    # does not generalise across folds (noise), not a fixable offset.
    eces_iso, eces_iso_ops = [], []
    for f in range(args.folds):
        te = fold == f
        tr = ~te
        ix, iy = fit_isotonic(conf[tr], tp[tr])
        p = np.interp(conf, ix, iy)
        e, _ = ece_equal_width(p[te], tp[te])
        eces_iso.append(e)
        te_ops = te & (conf >= 0.05)
        if te_ops.sum():
            eo, _ = ece_equal_width(p[te_ops], tp[te_ops], conf_lo=0.05)
            eces_iso_ops.append(eo)
    ece_iso, ece_iso_ops = float(np.mean(eces_iso)), float(np.mean(eces_iso_ops))
    print(
        f"[Q2] isotonic (cross-fitted):        held-out ECE={ece_iso:.4f} "
        f"(ops-region {ece_iso_ops:.4f}; raw ops {ece_ops:.4f}, TS ops {ece_ts_ops:.4f})"
    )

    # separability after global TS (must equal AUC before: monotone transform)
    t_full = fit_temperature(conf, tp)
    auc_ts = auc_score(apply_t(conf, t_full), tp)
    print(
        f"[Q3] AUC raw={auc_raw:.4f} -> after global TS={auc_ts:.4f} "
        f"(must be identical: monotone)"
    )

    # ---------------- Q4 per-class diagnostics
    print(
        "\n[Q4] per-class (conf>=0.05 region): ECE / TP-conf median / FP-conf median / AUC"
    )
    per_class = {}
    for ci, cn in names.items():
        m = (cls == ci) & (conf >= 0.05)
        if m.sum() < 20:
            continue
        e, _ = ece_equal_width(conf[m], tp[m])
        tpm = m & tp
        fpm = m & ~tp
        per_class[cn] = {
            "n": int(m.sum()),
            "ece": e,
            "tp_conf_med": float(np.median(conf[tpm])) if tpm.any() else None,
            "fp_conf_med": float(np.median(conf[fpm])) if fpm.any() else None,
            "auc": auc_score(conf[m], tp[m]),
            "n_tp": int(tpm.sum()),
            "n_fp": int(fpm.sum()),
        }
    for cn in CRIT:
        if cn in per_class:
            d = per_class[cn]
            print(
                f"  {cn:<16} ECE={d['ece']:.3f}  TP={d['tp_conf_med']:.3f}  "
                f"FP={d['fp_conf_med']:.3f}  AUC={d['auc']:.3f}  (n={d['n']})"
            )

    # critical-class pooled separability
    mcrit = np.isin(cls, [i for i, c in names.items() if c in CRIT])
    auc_crit = auc_score(conf[mcrit], tp[mcrit])
    print(f"  critical pooled: AUC={auc_crit:.4f} (n={int(mcrit.sum())})")

    rep = {
        "weights": str(Path(args.weights).resolve()),
        "protocol": {
            "imgsz": args.imgsz,
            "rect": False,
            "iou_thr": args.iou_thr,
            "conf_floor": args.conf_floor,
            "folds": args.folds,
        },
        "n_pred": n,
        "n_tp": int(tp.sum()),
        "n_fp": int((~tp).sum()),
        "n_images": n_imgs,
        "raw": {
            "ece": ece_raw,
            "ece_ops": ece_ops,
            "nll": nll_raw,
            "brier": brier_raw,
            "auc": auc_raw,
        },
        "reliability": bins,
        "ts": {
            "global_heldout_ece": ece_ts,
            "global_ops_ece": ece_ts_ops,
            "T_global_crossfit": t_global,
            "per_family_heldout_ece": ece_fam,
            "per_family_ops_ece": ece_fam_ops,
            "per_size_heldout_ece": ece_size,
            "per_size_ops_ece": ece_size_ops,
            "ops_only_T": t_ops,
            "ops_only_heldout_ece": ece_opsfit_ops,
            "isotonic_heldout_ece": ece_iso,
            "isotonic_ops_ece": ece_iso_ops,
            "T_full": t_full,
            "auc_after_ts": auc_ts,
        },
        "per_class": per_class,
        "critical_pooled_auc": auc_crit,
    }
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nreport written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
