#!/usr/bin/env python
"""
eval_ensemble_uncertainty.py -- deep-ensemble uncertainty as an alarm gate.

Stage 2B-2 of docs/plan_stage2_adaptive.md: the Bayesian route to robust alarm
decisions. MC Dropout is unavailable (RT-DETR decoder has no dropout), so the
deep-ensemble approximation (Lakshminarayanan et al., NIPS 2017) is the only
practical epistemic-uncertainty source: K independently fine-tuned models from
the same start, disagreement as the review-gate signal.

Method:
  1. Run every member on val (official square protocol, 300 queries).
  2. Pool all members' predictions per image; cluster by class + IoU>=0.5
     (NMS-style greedy by confidence). Per cluster: n_members fired,
     mean_conf, std_conf (= epistemic disagreement at that location).
  3. Match clusters to GT with the r_curve convention (class-aware, GT-major,
     IoU>=0.5) -> TP/FP labels.
  4. Metrics:
     (a) ensemble recall vs each single member (the classic ensemble lift);
     (b) AUC of mean_conf as TP/FP discriminator vs single-model AUC;
     (c) disagreement as an FP flag: AUC of std_conf among FP clusters, and
         FPR at fixed TP recall with a "score AND low-disagreement" gate
         vs score-only -- the operational question for the review tier;
     (d) per-GT member-count histogram (how many defects only one model sees).

Usage:
    python scripts/eval_ensemble_uncertainty.py \
        --weights runs/ablation_strong/strong_r2_uniform/weights/best.pt \
                  runs/ablation_strong/l1_seed123_ctl/weights/best.pt \
                  runs/ablation_strong/l1_seed2024_ctl/weights/best.pt \
        --out runs/calibration/ensemble_ctl3.json
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


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    aa = (a[:, 2] - a[:, 0]).clip(0) * (a[:, 3] - a[:, 1]).clip(0)
    ab = (b[:, 2] - b[:, 0]).clip(0) * (b[:, 3] - b[:, 1]).clip(0)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = (rb - lt).clip(0)
    inter = wh[..., 0] * wh[..., 1]
    return inter / (aa[:, None] + ab[None, :] - inter + 1e-9)


def cxcywh_to_xyxy(b: np.ndarray) -> np.ndarray:
    out = np.empty_like(b)
    out[:, 0] = b[:, 0] - b[:, 2] / 2
    out[:, 1] = b[:, 1] - b[:, 3] / 2
    out[:, 2] = b[:, 0] + b[:, 2] / 2
    out[:, 3] = b[:, 1] + b[:, 3] / 2
    return out


def match_tp_fp(pcls, gcls, ious, iou_thr=0.5):
    """r_curve convention: GT-major greedy, class-aware, each prediction once."""
    correct = np.zeros(len(pcls), dtype=bool)
    if len(gcls) == 0:
        return correct
    iou = ious * (pcls[:, None] == gcls[None, :])
    for gi in range(len(gcls)):
        cand = np.where(iou[:, gi] >= iou_thr)[0]
        if len(cand) == 0:
            continue
        best = cand[int(np.argmax(iou[cand, gi]))]
        correct[best] = True
        iou[best, :] = 0.0
    return correct


def auc_score(scores: np.ndarray, labels: np.ndarray) -> float:
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
        ranks[i : j + 1] = 0.5 * (i + j) + 1.0
        i = j + 1
    r = np.empty(len(s), dtype=float)
    r[order] = ranks
    return float(
        (r[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * neg.sum())
    )


@torch.no_grad()
def member_predictions(weights: str, args):
    """Run one member on val; return per-image list of (conf, cls, box)."""
    model = YOLO(weights)
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
    per_image = []
    for batch in loader:
        img = batch["img"].to(device).float() / 255.0
        out = rtdetr(img)
        preds = (out[0] if isinstance(out, (tuple, list)) else out).cpu().numpy()
        H, W = img.shape[2], img.shape[3]
        scale = np.array([W, H, W, H], dtype=float)
        for j in range(preds.shape[0]):
            p = preds[j]
            boxes = cxcywh_to_xyxy(p[:, :4]) * scale
            per_image.append(
                {
                    "conf": p[:, 4].astype(float),
                    "cls": p[:, 5].astype(int),
                    "box": boxes,
                    "gcls": batch["cls"]
                    .cpu()
                    .numpy()
                    .reshape(-1)[batch["batch_idx"].cpu().numpy() == j]
                    .astype(int),
                    "gbox": cxcywh_to_xyxy(
                        batch["bboxes"]
                        .cpu()
                        .numpy()
                        .reshape(-1, 4)[batch["batch_idx"].cpu().numpy() == j]
                    )
                    * scale,
                }
            )
    return per_image, names


def cluster_predictions(members_preds, iou_thr=0.5):
    """Pool all members' predictions per image; cluster same-class boxes with
    IoU>=iou_thr (greedy by confidence). Returns per-image cluster records:
    n_members, mean_conf, std_conf, box, cls."""
    n_img = len(members_preds[0])
    K = len(members_preds)
    out = []
    for i in range(n_img):
        # pool: (conf, cls, box, member)
        allc = np.concatenate([m[i]["conf"] for m in members_preds])
        allk = np.concatenate([m[i]["cls"] for m in members_preds])
        allb = np.concatenate([m[i]["box"] for m in members_preds], axis=0)
        allm = np.concatenate(
            [np.full(len(m[i]["conf"]), k) for k, m in enumerate(members_preds)]
        )
        order = np.argsort(-allc, kind="mergesort")
        allc, allk, allb, allm = allc[order], allk[order], allb[order], allm[order]
        clusters = []  # list of dict(idx list)
        for t in range(len(allc)):
            placed = False
            for c in clusters:
                if (
                    c["cls"] == allk[t]
                    and iou_matrix(allb[t : t + 1], c["box"][None, :])[0, 0] >= iou_thr
                ):
                    c["idx"].append(t)
                    placed = True
                    break
            if not placed:
                clusters.append({"cls": int(allk[t]), "box": allb[t], "idx": [t]})
        recs = []
        for c in clusters:
            idx = np.array(c["idx"])
            confs = allc[idx]
            mems = set(allm[idx].tolist())
            recs.append(
                {
                    "cls": c["cls"],
                    "box": c["box"],
                    "n_members": len(mems),
                    "mean_conf": float(confs.mean()),
                    "std_conf": float(confs.std()) if len(confs) > 1 else 0.0,
                    "max_conf": float(confs.max()),
                }
            )
        out.append(recs)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", nargs="+", required=True)
    ap.add_argument("--data", default="data/tianchi/data.yaml")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--iou-thr", type=float, default=0.5)
    ap.add_argument("--device", default="0")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    print(f"loading {len(args.weights)} ensemble members...")
    members, names = [], None
    for w in args.weights:
        preds, names = member_predictions(w, args)
        members.append(preds)
        print(f"  {w}: {len(preds)} images")
    K = len(members)

    clusters = cluster_predictions(members, args.iou_thr)

    # ---- match clusters to GT (r_curve convention), collect labels
    rec_conf, rec_tp, rec_std, rec_nm = [], [], [], []
    gt_member_count = []  # per GT: how many members detected it (conf>=0.001)
    single_recall = [0] * K
    n_gt_total = 0
    for i in range(len(clusters)):
        gt = members[0][i]["gcls"]
        gtbox = members[0][i]["gbox"]
        n_gt_total += len(gt)
        recs = clusters[i]
        if recs:
            cb = np.array([r["box"] for r in recs])
            cc = np.array([r["cls"] for r in recs])
            tp = match_tp_fp(cc, gt, iou_matrix(cb, gtbox), args.iou_thr)
        else:
            tp = np.zeros(0, dtype=bool)
        for r, t in zip(recs, tp, strict=False):
            rec_conf.append(r["mean_conf"])
            rec_tp.append(bool(t))
            rec_std.append(r["std_conf"])
            rec_nm.append(r["n_members"])
        # per-GT member counts (single-member detection at conf floor 0.001)
        for gi in range(len(gt)):
            cnt = 0
            for m in members:
                ious = (
                    iou_matrix(m[i]["box"], gtbox[gi : gi + 1])[:, 0]
                    if len(gtbox)
                    else np.zeros(0)
                )
                same = (
                    (m[i]["cls"] == gt[gi])
                    & (ious >= args.iou_thr)
                    & (m[i]["conf"] >= 0.001)
                )
                if same.any():
                    cnt += 1
            gt_member_count.append(cnt)
        for k, m in enumerate(members):
            if len(gt):
                ious = iou_matrix(m[i]["box"], gtbox)
                same = (
                    (m[i]["cls"][:, None] == gt[None, :])
                    & (ious >= args.iou_thr)
                    & (m[i]["conf"][:, None] >= 0.001)
                )
                single_recall[k] += int(same.any(axis=0).sum())

    rec_conf = np.array(rec_conf)
    rec_tp = np.array(rec_tp, dtype=bool)
    rec_std = np.array(rec_std)
    rec_nm = np.array(rec_nm)
    gt_member_count = np.array(gt_member_count)

    ens_recall = float(rec_tp.sum()) / max(n_gt_total, 1)
    single = [float(r) / max(n_gt_total, 1) for r in single_recall]
    auc_ens = auc_score(rec_conf, rec_tp)
    # disagreement as a TP/FP discriminator: AUC of std_conf over ALL clusters
    # (the operational question is whether epistemic disagreement separates
    # true from false detections -- an all-negative label set has no AUC)
    auc_dis = auc_score(rec_std, rec_tp)
    fp = ~rec_tp

    # operational gate: keep TP with mean_conf>=t AND std_conf<=d; measure FPR@recall
    def fpr_at_recall(t_list, d_list):
        rows = []
        for t in t_list:
            for d in d_list:
                keep_tp = rec_tp & (rec_conf >= t) & (rec_std <= d)
                keep_all = (rec_conf >= t) & (rec_std <= d)
                r = keep_tp.sum() / max(rec_tp.sum(), 1)
                fpr = (~rec_tp & keep_all).sum() / max((~rec_tp).sum(), 1)
                rows.append({"t": t, "d": d, "recall": float(r), "fpr": float(fpr)})
        return rows

    print(f"\n=== ensemble uncertainty audit: K={K} members ===")
    print(f"GT instances: {n_gt_total}")
    print(f"recall: ensemble={ens_recall:.4f}  members={[round(s, 4) for s in single]}")
    print(
        f"AUC(mean_conf) = {auc_ens:.4f}   (single-model references: 0.763 weak / 0.823 strong)"
    )
    print(f"AUC(std_conf as TP/FP discriminator) = {auc_dis:.4f}")
    print(
        "per-GT member-count histogram: "
        + str({k: int((gt_member_count == k).sum()) for k in range(K + 1)})
    )
    only_one = int((gt_member_count == 1).sum())
    print(
        f"GT seen by exactly one member (epistemic-uncertainty candidates): {only_one}"
    )
    if fp.sum():
        print(
            f"disagreement among FP clusters: mean std={rec_std[fp].mean():.4f} "
            f"(TP: {rec_std[rec_tp].mean():.4f})"
        )

    rep = {
        "weights": [str(Path(w).resolve()) for w in args.weights],
        "K": K,
        "n_gt": n_gt_total,
        "recall_ensemble": ens_recall,
        "recall_members": single,
        "auc_mean_conf": auc_ens,
        "auc_std_conf": auc_dis,
        "gt_member_count_hist": {
            str(k): int((gt_member_count == k).sum()) for k in range(K + 1)
        },
        "fp_std_mean": float(rec_std[fp].mean()) if fp.sum() else None,
        "tp_std_mean": float(rec_std[rec_tp].mean()),
        "gate_grid": fpr_at_recall([0.25, 0.35, 0.45], [0.05, 0.1, 0.2, 1.0]),
    }
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"report written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
