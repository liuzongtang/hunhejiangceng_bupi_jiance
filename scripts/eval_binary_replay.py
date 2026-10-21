"""
In-domain binary replay: image-level 有瑕疵/无瑕疵 judgment on the Tianchi val split.

Unlike ``eval_binary_detection.py`` (which treats every val image as positive and
measures cross-domain FPR on AITEX), this script uses the *in-domain* negatives:
the val split contains 957 images = 583 defective (>=1 GT box) + 374 no-defect
(empty label). That makes a true binary classification accuracy computable.

Pipeline matches the official mAP eval (``eval_multIou.py``): the unmodified
ultralytics RTDETRDataset val dataloader (rect=False square letterbox,
augment=False) feeds the raw model; the image defect score is the maximum
confidence over the 300 final queries (RT-DETR is NMS-free, the 300 outputs are
already the final detection set).

Metrics per tau: TPR / FPR / precision / accuracy / F1 / Youden J, plus
threshold-free AUC (rank-based, no sklearn). Also reports the cross-domain AITEX
FPR at the same taus as a secondary robustness number.

Usage:
    python scripts/eval_binary_replay.py --weights runs/rtdetr/tianchi18_mosaic1280/weights/best.pt --tag mosaic1280
"""

from __future__ import annotations

import argparse
import csv
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


def has_box(label_path: Path) -> bool:
    """GT defective iff the YOLO label file exists and has >=1 non-empty line."""
    try:
        with open(label_path, "r", encoding="utf-8") as f:
            return any(line.strip() for line in f)
    except FileNotFoundError:
        return False


def img_to_label(im_file: str) -> Path:
    """Map .../val/images/xxx.jpg -> .../val/labels/xxx.txt (either separator)."""
    p = Path(im_file)
    parts = list(p.parts)
    for i, seg in enumerate(parts):
        if seg == "images":
            parts[i] = "labels"
            break
    return Path(*parts).with_suffix(".txt")


@torch.no_grad()
def raw_max_conf(
    rtdetr: torch.nn.Module, img: torch.Tensor, device: torch.device
) -> np.ndarray:
    """Per-image max confidence over the 300 queries -> (B,)."""
    out = rtdetr(img.to(device))
    y = out[0] if isinstance(out, (tuple, list)) else out  # (B, 300, 6)
    return y[..., 4].max(dim=1).values.float().cpu().numpy()


@torch.no_grad()
def collect_val(rtdetr, data_dict, args, device):
    """Return (max_conf, is_defective) aligned per val image, in dataset order."""
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
    confs: list = []
    flags: list = []
    for batch in loader:
        img = batch["img"].float() / 255.0
        confs.append(raw_max_conf(rtdetr, img, device))
        flags.extend(has_box(img_to_label(f)) for f in batch["im_file"])
    return np.concatenate(confs), np.asarray(flags, dtype=bool)


def load_stretch(img_path: Path, imgsz: int) -> torch.Tensor:
    import cv2

    im = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if im is None:
        raise FileNotFoundError(f"cannot read {img_path}")
    im = cv2.resize(im, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR)
    t = torch.from_numpy(im).permute(2, 0, 1).contiguous().unsqueeze(0)
    return t.float() / 255.0


@torch.no_grad()
def collect_ait_ex(rtdetr, neg_dir: Path, args, device) -> dict:
    """AITEX no-defect images (cross-domain negatives) -> sample-name -> conf array."""
    if not neg_dir.exists():
        return {}
    samples: dict = {}
    for sub in sorted(p for p in neg_dir.iterdir() if p.is_dir()):
        pngs = (
            sorted(sub.glob("*.png"))
            + sorted(sub.glob("*.jpg"))
            + sorted(sub.glob("*.jpeg"))
        )
        confs = [
            float(raw_max_conf(rtdetr, load_stretch(p, args.imgsz), device)[0])
            for p in pngs
        ]
        if confs:
            samples[sub.name] = np.asarray(confs)
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
        ranks[order[i : j + 1]] = 0.5 * (i + j) + 1.0  # 1-based average rank
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
        "TNR_specificity": round(tn / n, 4) if n else None,
        "precision": round(tp / (tp + fp), 4) if tp + fp else None,
        "accuracy": round((tp + tn) / (p + n), 4),
        "F1": round(2 * tp / (2 * tp + fp + fn), 4) if 2 * tp + fp + fn else None,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", required=True)
    ap.add_argument(
        "--tag", default=None, help="label for outputs (default: run dir name)"
    )
    ap.add_argument("--data", default="data/tianchi/data.yaml")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="0")
    ap.add_argument("--ait-ex-dir", default="shujuji_AITEX/NODefect_images")
    ap.add_argument("--out", default="runs/binary_replay")
    ap.add_argument("--tau-min", type=float, default=0.05)
    ap.add_argument("--tau-max", type=float, default=0.95)
    ap.add_argument("--tau-step", type=float, default=0.05)
    args = ap.parse_args()

    tag = args.tag or Path(args.weights).parent.parent.name
    device = torch.device("cpu")
    if args.device.isdigit() and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.device}")

    yolo = YOLO(args.weights)
    rtdetr = yolo.model
    rtdetr.to(device)
    rtdetr.eval()

    data_dict = check_det_dataset(args.data)
    val_conf, val_gt = collect_val(rtdetr, data_dict, args, device)
    n_pos, n_neg = int(val_gt.sum()), int((~val_gt).sum())
    print(f"[replay:{tag}] val={val_conf.size} defective={n_pos} no-defect={n_neg}")

    ait_samples = collect_ait_ex(rtdetr, Path(args.ait_ex_dir), args, device)
    ait_conf = (
        np.concatenate(list(ait_samples.values())) if ait_samples else np.zeros(0)
    )
    if ait_conf.size:
        print(f"[replay:{tag}] AITEX cross-domain negatives={ait_conf.size}")

    # ---- tau sweep ----
    taus = np.round(np.arange(args.tau_min, args.tau_max + 1e-9, args.tau_step), 4)
    rows = []
    for tau in taus:
        c = confusion(val_conf, val_gt, float(tau))
        c["tau"] = float(tau)
        c["ait_fpr"] = (
            round(float((ait_conf > tau).mean()), 4) if ait_conf.size else None
        )
        rows.append(c)

    # Youden J per row (needs TPR/FPR)
    for r in rows:
        if r["TPR_recall"] is not None and r["FPR"] is not None:
            r["YoudenJ"] = round(r["TPR_recall"] - r["FPR"], 4)

    def pick(key, mode):
        vals = [(r[key], r["tau"]) for r in rows if r.get(key) is not None]
        if not vals:
            return None
        return max(vals)[1] if mode == "max" else min(vals)[1]

    best_acc_tau = pick("accuracy", "max")
    tau90 = next((r["tau"] for r in rows if (r["TPR_recall"] or 0) >= 0.90), None)
    best_j_tau = pick("YoudenJ", "max")

    summary = {
        "tag": tag,
        "weights": str(args.weights),
        "n_val": int(val_conf.size),
        "n_defective": n_pos,
        "n_no_defect": n_neg,
        "n_ait_ex": int(ait_conf.size),
        "AUC": round(auc_rank(val_conf, val_gt), 4),
        "mean_conf_defective": round(float(val_conf[val_gt].mean()), 4),
        "mean_conf_no_defect": round(float(val_conf[~val_gt].mean()), 4),
        "tau_0.25": confusion(val_conf, val_gt, 0.25),
        "tau_0.50": confusion(val_conf, val_gt, 0.50),
        "best_accuracy": {
            "tau": best_acc_tau,
            **confusion(val_conf, val_gt, best_acc_tau),
        }
        if best_acc_tau is not None
        else None,
        "tau_at_recall_0.90": {"tau": tau90, **confusion(val_conf, val_gt, tau90)}
        if tau90 is not None
        else None,
        "best_youdenJ": {"tau": best_j_tau, **confusion(val_conf, val_gt, best_j_tau)}
        if best_j_tau is not None
        else None,
    }

    # ---- console report ----
    print(f"\n[replay:{tag}] ===== binary defect judgment (in-domain) =====")
    print(
        f"AUC={summary['AUC']}  mean conf: defective={summary['mean_conf_defective']} vs no-defect={summary['mean_conf_no_defect']}"
    )
    fixed = (("tau_0.25", 0.25), ("tau_0.50", 0.50))
    picked = ("best_accuracy", "tau_at_recall_0.90", "best_youdenJ")
    for name, tau in fixed:
        b = summary[name]
        print(
            f"{name:20s} tau={tau:.2f}  acc={b['accuracy']:.4f}  "
            f"TPR={b['TPR_recall']:.4f}  FPR={b['FPR']:.4f}  P={b['precision']}  F1={b['F1']}  "
            f"[TP={b['TP']} FP={b['FP']} TN={b['TN']} FN={b['FN']}]"
        )
    for name in picked:
        b = summary[name]
        if not b:
            print(f"{name:20s} — (not reached)")
            continue
        print(
            f"{name:20s} tau={b['tau']:.2f}  acc={b['accuracy']:.4f}  "
            f"TPR={b['TPR_recall']:.4f}  FPR={b['FPR']:.4f}  P={b['precision']}  F1={b['F1']}  "
            f"[TP={b['TP']} FP={b['FP']} TN={b['TN']} FN={b['FN']}]"
        )

    # ---- outputs ----
    out_dir = Path(args.out) / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    fields = [
        "tau",
        "TP",
        "FP",
        "TN",
        "FN",
        "TPR_recall",
        "FPR",
        "TNR_specificity",
        "precision",
        "accuracy",
        "F1",
        "YoudenJ",
        "ait_fpr",
    ]
    with (out_dir / "sweep.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    np.save(out_dir / "val_max_conf.npy", val_conf)
    np.save(out_dir / "val_gt_defective.npy", val_gt)
    if ait_conf.size:
        np.save(out_dir / "aitex_max_conf.npy", ait_conf)
    print(f"[replay:{tag}] wrote {out_dir}/summary.json, sweep.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
