"""
Plot the A/B comparison of the reward runs: matched-only vs include-misses.

Reads the per-epoch ``results.csv`` from the two Ultralytics run directories and
the final-epoch per-class recall (from a clean same-code re-validation) and renders
a 2x2 figure: overall recall / mAP50 / precision vs epoch, plus a grouped bar chart
of the four critical classes' recall.

Usage:
    python scripts/plot_ab_recall.py
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")  # headless backend
import matplotlib.pyplot as plt
import numpy as np

# Chinese glyphs on Windows; harmless fallback to DejaVu elsewhere.
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

BASE = Path("runs/rtdetr-reward")
RUNS = {
    "matched-only (β0.5)": BASE / "beta0.5_e5",
    "include-misses (β0.5)": BASE / "beta0.5_miss",
}
COLORS = {"matched-only (β0.5)": "tab:blue", "include-misses (β0.5)": "tab:red"}

# Critical class ids -> (en, zh).
CRITICAL = {
    9: ("broken_warp", "断经"),
    15: ("star_skip", "星跳"),
    16: ("broken_spandex", "断氨纶"),
    19: ("weave_defect", "死皱"),
}
# Final-epoch per-class recall from a clean same-code re-validation (2026-09-20).
CRIT_RECALL: Dict[str, Dict[int, float]] = {
    "matched-only (β0.5)": {9: 0.4194, 15: 0.5714, 16: 0.6556, 19: 0.5128},
    "include-misses (β0.5)": {9: 0.4516, 15: 0.6286, 16: 0.5556, 19: 0.6410},
}


def _read_results(run_dir: Path) -> Dict[str, List[float]]:
    """Return per-epoch ``epoch``/``precision``/``recall``/``mAP50`` columns."""
    path = run_dir / "results.csv"
    with path.open(newline="") as f:
        rows = [{k.strip(): v for k, v in r.items()} for r in csv.DictReader(f)]
    return {
        "epoch": [int(r["epoch"]) for r in rows],
        "precision": [float(r["metrics/precision(B)"]) for r in rows],
        "recall": [float(r["metrics/recall(B)"]) for r in rows],
        "mAP50": [float(r["metrics/mAP50(B)"]) for r in rows],
    }


def main() -> int:
    """Render the A/B recall figure into ``runs/rtdetr-reward/``."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    ax_r, ax_m, ax_p, ax_b = axes[0][0], axes[0][1], axes[1][0], axes[1][1]

    for label, run in RUNS.items():
        d = _read_results(run)
        ax_r.plot(d["epoch"], d["recall"], marker="o", color=COLORS[label], label=label)
        ax_m.plot(d["epoch"], d["mAP50"], marker="o", color=COLORS[label], label=label)
        ax_p.plot(
            d["epoch"], d["precision"], marker="o", color=COLORS[label], label=label
        )

    for ax, name in ((ax_r, "Recall"), (ax_m, "mAP50"), (ax_p, "Precision")):
        ax.set_xlabel("epoch")
        ax.set_ylabel(name)
        ax.set_title(f"Overall {name} vs epoch")
        ax.legend()
        ax.grid(alpha=0.3)

    ids = sorted(CRITICAL)
    x = np.arange(len(ids))
    width = 0.36
    for j, (label, rec) in enumerate(CRIT_RECALL.items()):
        ax_b.bar(
            x + (j - 0.5) * width,
            [rec[i] for i in ids],
            width,
            label=label,
            color=COLORS[label],
        )
    ax_b.set_xticks(x)
    ax_b.set_xticklabels(
        [f"{CRITICAL[i][1]}\n{CRITICAL[i][0]}" for i in ids], fontsize=9
    )
    ax_b.set_ylabel("Recall")
    ax_b.set_title("Critical-class recall (final epoch)")
    ax_b.set_ylim(0, 0.75)
    ax_b.legend()
    ax_b.grid(alpha=0.3, axis="y")

    out = BASE / "ab_recall_comparison.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"[plot] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
