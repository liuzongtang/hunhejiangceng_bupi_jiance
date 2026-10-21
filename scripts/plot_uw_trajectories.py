#!/usr/bin/env python
"""Plot UW learned-weight trajectories across seeds (stage 2 diagnostic).

Reads runs/ablation_adaptive/uw_seed*/adaptive_weights.json and overlays
w_cls / w_reg (and the ratio) for the three seeds on one figure.

Usage:
    python scripts/plot_uw_trajectories.py
    python scripts/plot_uw_trajectories.py --out runs/ablation_adaptive/uw_trajectories.png
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SEEDS = [42, 123, 2024, 7]
COLORS = {42: "#d62728", 123: "#1f77b4", 2024: "#2ca02c", 7: "#9467bd"}


def load(seed: int, root: Path):
    f = root / f"uw_seed{seed}" / "adaptive_weights.json"
    if not f.exists():
        return None
    d = json.load(open(f, encoding="utf-8"))
    tr = d["trajectory"]
    steps = np.array([r["step"] for r in tr], dtype=float)
    w_cls = np.array([r["w_cls"] for r in tr], dtype=float)
    w_reg = np.array([r["w_reg"] for r in tr], dtype=float)
    return steps, w_cls, w_reg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="runs/ablation_adaptive")
    ap.add_argument("--out", default="runs/ablation_adaptive/uw_trajectories.png")
    args = ap.parse_args()
    root = Path(args.root)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for s in SEEDS:
        r = load(s, root)
        if r is None:
            print(f"[skip] seed {s}: no adaptive_weights.json")
            continue
        steps, w_cls, w_reg = r
        ratio = w_cls / np.maximum(w_reg, 1e-9)
        axes[0].plot(steps, w_cls, color=COLORS[s], lw=1.4, label=f"seed {s} w_cls")
        axes[0].plot(
            steps, w_reg, color=COLORS[s], lw=1.2, ls="--", label=f"seed {s} w_reg"
        )
        axes[1].plot(steps, ratio, color=COLORS[s], lw=1.4, label=f"seed {s}")
        print(
            f"[ok] seed {s}: {len(steps)} points, final w_cls={w_cls[-1]:.3f} "
            f"w_reg={w_reg[-1]:.3f} ratio={ratio[-1]:.2f}"
        )

    axes[0].axhline(1.0, color="gray", lw=0.8, ls=":")
    axes[0].set_xlabel("optimizer step")
    axes[0].set_ylabel("learned weight")
    axes[0].set_title("UW learned weights (solid=cls, dashed=reg)")
    axes[0].legend(fontsize=8, ncol=2)
    axes[0].grid(alpha=0.3)

    axes[1].axhline(1.0, color="gray", lw=0.8, ls=":", label="uniform 1:1")
    axes[1].set_xlabel("optimizer step")
    axes[1].set_ylabel("w_cls / w_reg")
    axes[1].set_title("cls:reg ratio (stationary ≈ L_reg : L_cls)")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)

    fig.suptitle(
        "Stage 2 UW adaptive loss weighting — per-seed trajectories "
        "(mosaic1280 init, 5 epochs, official protocol)",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(args.out, dpi=160)
    print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
