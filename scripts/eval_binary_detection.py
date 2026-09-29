"""
B 口径 experiment: image-level binary defect detection (有瑕疵 / 无瑕疵).

The Tianchi scheme has no "normal" class, so image-level binary detection is
defined by thresholding the detector's raw confidence: an image is "defective"
iff the model's maximum class confidence over its 300 queries exceeds ``tau``.

Metrics:
  * recall — fraction of Tianchi val images (957, all defective) flagged defective.
  * FPR     — fraction of AITEX ``NODefect_images`` (cross-domain negatives)
              flagged defective. AITEX is a different fabric source, so this is a
              *cross-domain* false-positive rate, not an in-domain precision.

The script sweeps ``tau``, reports the recall-FPR trade-off, finds the lowest
``tau`` achieving the target recall (default 0.90), and writes a CSV + JSON
summary under ``runs/binary_eval/``.

Preprocessing matches the RT-DETR train/val convention: stretch the raw image to
a square ``imgsz x imgsz`` (no aspect-preserving letterbox), BGR, float/255.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.data import build_dataloader
from ultralytics.data.utils import check_det_dataset
from ultralytics.models.rtdetr.val import RTDETRDataset


def load_stretch(img_path: Path, imgsz: int) -> torch.Tensor:
    """Load one image and stretch it to ``imgsz x imgsz`` (BGR, uint8 -> float/255)."""
    im = cv2.imread(str(img_path), cv2.IMREAD_COLOR)  # BGR, matching RT-DETR val
    if im is None:
        raise FileNotFoundError(f"cannot read {img_path}")
    im = cv2.resize(im, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR)
    t = torch.from_numpy(im).permute(2, 0, 1).contiguous().unsqueeze(0)  # (1,3,H,W)
    return t.float() / 255.0


@torch.no_grad()
def raw_max_conf(
    rtdetr: torch.nn.Module, img: torch.Tensor, device: torch.device
) -> np.ndarray:
    """Return the per-image max confidence over the 300 queries (B,)."""
    out = rtdetr(img.to(device))
    y = out[0] if isinstance(out, (tuple, list)) else out  # (B, 300, 6)
    return y[..., 4].max(dim=1).values.cpu().numpy()


def collect_val_max_conf(rtdetr, data_dict, args, device) -> np.ndarray:
    """Run the model over the val split and return one max-conf per image."""
    cfg = get_cfg(overrides={"imgsz": args.imgsz, "rect": False, "cache": False})
    dataset = RTDETRDataset(
        img_path=data_dict["val"],
        imgsz=args.imgsz,
        batch_size=args.batch,
        augment=False,  # val mode: stretch to square, no augmentation
        hyp=cfg,
        rect=False,
        cache=None,
        prefix="val: ",
        data=data_dict,
    )
    loader = build_dataloader(dataset, args.batch, workers=0, shuffle=False)
    confs: list = []
    for i, batch in enumerate(loader):
        img = batch["img"].float() / 255.0
        confs.append(raw_max_conf(rtdetr, img, device))
        if args.val_limit and (i + 1) * args.batch >= args.val_limit:
            break
    return np.concatenate(confs)[: args.val_limit or None]


def collect_ait_ex_max_conf(rtdetr, neg_dir: Path, args, device) -> dict:
    """Run the model over AITEX NODefect images; return sample-name -> conf array."""
    samples: dict = {}
    subdirs = sorted(p for p in neg_dir.iterdir() if p.is_dir())
    for sub in subdirs:
        pngs = (
            sorted(sub.glob("*.png"))
            + sorted(sub.glob("*.jpg"))
            + sorted(sub.glob("*.jpeg"))
        )
        confs = []
        for j, p in enumerate(pngs):
            if args.ait_limit and j >= args.ait_limit:
                break
            confs.append(
                float(raw_max_conf(rtdetr, load_stretch(p, args.imgsz), device)[0])
            )
        if confs:
            samples[sub.name] = np.asarray(confs)
    return samples


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--weights", default="runs/rtdetr/tianchi18_full/weights/best.pt"
    )
    parser.add_argument("--data", default="data/tianchi/data.yaml")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument(
        "--ait-ex-dir",
        default="shujuji_AITEX/NODefect_images",
        help="Directory of AITEX no-defect samples (cross-domain negatives)",
    )
    parser.add_argument("--out", default="runs/binary_eval")
    parser.add_argument("--tau-min", type=float, default=0.05)
    parser.add_argument("--tau-max", type=float, default=0.90)
    parser.add_argument("--tau-step", type=float, default=0.05)
    parser.add_argument("--target-recall", type=float, default=0.90)
    parser.add_argument("--val-limit", type=int, default=0, help="0 = all val images")
    parser.add_argument(
        "--ait-limit", type=int, default=0, help="0 = all AITEX negatives"
    )
    args = parser.parse_args()

    device = torch.device("cpu")
    if args.device.isdigit() and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.device}")

    yolo = YOLO(args.weights)
    rtdetr = yolo.model
    rtdetr.to(device)
    rtdetr.eval()

    data_dict = check_det_dataset(args.data)
    val_conf = collect_val_max_conf(rtdetr, data_dict, args, device)
    ait_samples = collect_ait_ex_max_conf(rtdetr, Path(args.ait_ex_dir), args, device)
    ait_conf = (
        np.concatenate(list(ait_samples.values())) if ait_samples else np.zeros(0)
    )

    n_pos = int(val_conf.size)
    n_neg = int(ait_conf.size)
    print(
        f"[binary] val positives={n_pos}  AITEX negatives={n_neg} "
        f"(samples={len(ait_samples)})"
    )

    taus = np.arange(args.tau_min, args.tau_max + 1e-9, args.tau_step)
    rows = []
    tau_achieved = None
    for tau in taus:
        recall = float((val_conf > tau).mean()) if n_pos else 0.0
        fpr = float((ait_conf > tau).mean()) if n_neg else 0.0
        rows.append(
            {
                "tau": round(float(tau), 3),
                "recall": round(recall, 4),
                "fpr": round(fpr, 4),
            }
        )
        if tau_achieved is None and recall >= args.target_recall:
            tau_achieved = float(tau)

    # ---- Summary at the lowest tau reaching the target recall ----
    summary: dict = {"n_pos": n_pos, "n_neg": n_neg}
    if tau_achieved is None:
        print(
            f"[binary] target recall {args.target_recall:.2f} NOT reached "
            f"(max recall {max(r['recall'] for r in rows):.4f} at tau={taus[0]:.2f})"
        )
        summary["tau_achieved"] = None
    else:
        recall = float((val_conf > tau_achieved).mean())
        fpr = float((ait_conf > tau_achieved).mean()) if n_neg else 0.0
        tp = recall * n_pos
        fp = fpr * n_neg
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        accuracy = (tp + (n_neg - fp)) / (n_pos + n_neg)
        summary = {
            "n_pos": n_pos,
            "n_neg": n_neg,
            "tau_achieved": round(tau_achieved, 3),
            "recall": round(recall, 4),
            "fpr": round(fpr, 4),
            "precision": round(precision, 4),
            "accuracy": round(accuracy, 4),
            "per_sample_fpr": {
                name: round(float((c > tau_achieved).mean()), 4)
                for name, c in ait_samples.items()
            },
        }
        print(
            f"[binary] min tau for recall>={args.target_recall:.2f}: {tau_achieved:.2f}"
        )
        print(
            f"         recall={recall:.4f}  FPR={fpr:.4f}  precision={precision:.4f} "
            f"accuracy={accuracy:.4f} (cross-domain caveat)"
        )
        for name, c in ait_samples.items():
            print(f"           {name}: {int((c > tau_achieved).sum())}/{c.size} FP")

    # ---- Write outputs ----
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "sweep.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["tau", "recall", "fpr"])
        w.writeheader()
        w.writerows(rows)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    # Also save the raw per-image confidences for reuse (e.g. plotting).
    np.save(out_dir / "val_max_conf.npy", val_conf)
    np.save(out_dir / "aitex_max_conf.npy", ait_conf)

    print(f"[binary] wrote {csv_path} and {out_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
