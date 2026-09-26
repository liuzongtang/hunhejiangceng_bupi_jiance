# -*- coding: utf-8 -*-
"""Generate the three paper figures for docs/paper/paper_skeleton.md.

图1 技术路线图  fig_pipeline.png
图2 训练曲线    fig_training_curves.png   (tianchi18_full, 60 epochs)
图3 奖励A/B     fig_reward_ab.png         (matched-only vs include-misses)

Reads real results from runs/; writes into docs/paper/figures/.
Reuses the validated light palette + Chinese-font handling from
scripts/gen_research_figures.py.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# ---- Chinese font (Windows) ----
for name in ("Microsoft YaHei", "SimHei", "SimSun", "Noto Sans CJK SC"):
    if any(f.name == name for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.sans-serif"] = [name, "Arial"]
        break
plt.rcParams["axes.unicode_minus"] = False

# ---- Palette ----
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
RED = "#e34948"
GREEN = "#008300"
INK = "#0b0b0b"
SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASE = "#c3c2b7"
SURFACE = "#fcfcfb"

plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": SURFACE,
        "axes.edgecolor": BASE,
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "axes.axisbelow": True,
        "font.size": 11,
    }
)


def _style_axes(ax, xlabel=None, ylabel=None):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(length=0)
    if xlabel:
        ax.set_xlabel(xlabel, color=SECONDARY)
    if ylabel:
        ax.set_ylabel(ylabel, color=SECONDARY)


# ============================================================
# 图 1 — 技术路线图（五阶段 pipeline）
# ============================================================
def fig_pipeline():
    stages = [
        ("数据层", "Tianchi 布匹疵点\n20 类 · 1280 拉伸方图"),
        ("检测基线", "RT-DETR-L\n端到端 · 免 NMS"),
        ("七维奖惩", "D01–D07 可微奖励\nL = L_det − β·R"),
        ("标签噪声诊断", "混淆矩阵 → 双向判定\n→ 裁剪复核"),
        ("合并与评估", "20→18 类\n关键类召回 + 全局 mAP50"),
    ]
    n = len(stages)
    fig, ax = plt.subplots(figsize=(11.4, 3.2), dpi=150)
    ax.set_xlim(0, n)
    ax.set_ylim(0, 2.6)
    ax.axis("off")

    w = 0.86
    for i, (title, desc) in enumerate(stages):
        x = i + 0.07
        ax.add_patch(
            FancyBboxPatch(
                (x, 1.2),
                w,
                1.0,
                boxstyle="round,pad=0.05,rounding_size=0.12",
                facecolor="#eaf2fc",
                edgecolor=BLUE,
                lw=1.6,
            )
        )
        ax.text(
            x + w / 2,
            1.92,
            title,
            ha="center",
            va="center",
            fontsize=13,
            fontweight="bold",
            color=BLUE,
        )
        ax.text(
            x + w / 2,
            1.42,
            desc,
            ha="center",
            va="center",
            fontsize=9,
            color=SECONDARY,
            linespacing=1.5,
        )
        if i < n - 1:
            ax.annotate(
                "",
                xy=(x + w + 0.03, 1.7),
                xytext=(x + w + 0.17, 1.7),
                arrowprops={"arrowstyle": "-|>", "color": MUTED, "lw": 1.8},
            )

    # GRPO branch (next layer)
    ax.annotate(
        "",
        xy=(2.5, 1.13),
        xytext=(2.5, 0.5),
        arrowprops={"arrowstyle": "-|>", "color": ORANGE, "lw": 1.8, "ls": "--"},
    )
    ax.add_patch(
        FancyBboxPatch(
            (1.9, 0.05),
            1.2,
            0.45,
            boxstyle="round,pad=0.03,rounding_size=0.08",
            facecolor="#fdeee6",
            edgecolor=ORANGE,
            lw=1.4,
        )
    )
    ax.text(
        2.5,
        0.275,
        "GRPO（下一层）",
        ha="center",
        va="center",
        fontsize=10,
        fontweight="bold",
        color=ORANGE,
    )

    fig.tight_layout()
    fig.savefig(OUT / "fig_pipeline.png", bbox_inches="tight")
    plt.close(fig)


# ============================================================
# 图 2 — 训练曲线（tianchi18_full，60 epochs，0 NaN）
# ============================================================
def fig_training_curves():
    import pandas as pd

    df = pd.read_csv(ROOT / "runs" / "rtdetr" / "tianchi18_full" / "results.csv")
    ep = df["epoch"].astype(int).values
    m50 = df["metrics/mAP50(B)"].values
    m5095 = df["metrics/mAP50-95(B)"].values
    prec = df["metrics/precision(B)"].values
    rec = df["metrics/recall(B)"].values

    ibest = int(m50.argmax())

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.2, 6.2), dpi=150, sharex=True)
    ax1.plot(ep, m50, color=BLUE, lw=2, label="mAP50")
    ax1.plot(ep, m5095, color=ORANGE, lw=2, label="mAP50-95")
    ax1.scatter([ep[ibest]], [m50[ibest]], color=RED, s=48, zorder=5)
    ax1.annotate(
        f"最优 {m50[ibest]:.3f} @ e{ep[ibest]}",
        (ep[ibest], m50[ibest]),
        textcoords="offset points",
        xytext=(8, 8),
        color=RED,
        fontweight="bold",
        fontsize=10,
    )
    ax1.set_ylabel("mAP")
    ax1.set_ylim(0.10, 0.65)
    ax1.legend(frameon=False, loc="lower right", fontsize=10)
    _style_axes(ax1)

    ax2.plot(ep, prec, color=AQUA, lw=2, label="Precision")
    ax2.plot(ep, rec, color=ORANGE, lw=2, label="Recall")
    ax2.set_ylabel("Precision / Recall")
    ax2.set_ylim(0.10, 0.70)
    ax2.set_xlabel("训练轮次 epoch")
    ax2.legend(frameon=False, loc="lower right", fontsize=10)
    _style_axes(ax2, xlabel="训练轮次 epoch")

    fig.suptitle(
        "RT-DETR-L · 18 类（合并后）· 60 轮 · 全程 0 NaN（fp32）",
        fontsize=12,
        fontweight="bold",
        color=INK,
        y=0.99,
    )
    fig.tight_layout()
    fig.savefig(OUT / "fig_training_curves.png", bbox_inches="tight")
    plt.close(fig)


# ============================================================
# 图 3 — 奖励注入 A/B（匹配奖励 vs 漏检注入）
# ============================================================
def fig_reward_ab():
    import numpy as np

    # final-epoch (5) metrics from runs/rtdetr-reward/{beta0.5_e5,beta0.5_miss}/results.csv
    metrics = ["Recall\n召回率", "Precision\n精确率", "mAP50"]
    arm_matched = [0.52313, 0.59378, 0.57288]
    arm_miss = [0.55310, 0.57163, 0.57643]
    deltas = [a - b for a, b in zip(arm_miss, arm_matched, strict=False)]

    # critical-class recall delta (20-class ids), from clean same-code re-validation
    crit_names = [
        "断经\nbroken_warp",
        "星跳\nstar_skip",
        "死皱\nweave_defect",
        "断氨纶\nbroken_spandex",
    ]
    crit_delta = [0.0322, 0.0572, 0.1282, -0.1000]

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(11.4, 4.2), dpi=150, gridspec_kw={"width_ratios": [1.0, 1.1]}
    )

    x = np.arange(len(metrics))
    w = 0.36
    b1 = ax1.bar(
        x - w / 2, arm_matched, w, color=BLUE, label="奖励注入 β=0.5", zorder=3
    )
    b2 = ax1.bar(
        x + w / 2,
        arm_miss,
        w,
        color=ORANGE,
        label="+ 漏检注入 (include-misses)",
        zorder=3,
    )
    for bars in (b1, b2):
        for r in bars:
            ax1.text(
                r.get_x() + r.get_width() / 2,
                r.get_height() + 0.012,
                f"{r.get_height():.3f}",
                ha="center",
                color=SECONDARY,
                fontsize=9,
            )
    for xi, d in zip(x, deltas, strict=False):
        col = GREEN if d >= 0 else RED
        ax1.text(
            xi,
            max(arm_matched[xi], arm_miss[xi]) + 0.052,
            f"Δ {d:+.3f}".replace("+", "+").replace("-", "−"),
            ha="center",
            color=col,
            fontweight="bold",
            fontsize=10,
        )
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics, fontsize=9.5)
    ax1.set_ylim(0, 0.78)
    ax1.set_ylabel("数值")
    ax1.set_title("整体指标 A/B", fontsize=12, fontweight="bold", color=INK)
    ax1.legend(frameon=False, loc="lower left", fontsize=9)
    _style_axes(ax1)

    y = np.arange(len(crit_names))[::-1]
    colors = [GREEN if d >= 0 else RED for d in crit_delta]
    ax2.barh(y, crit_delta, color=colors, height=0.55, zorder=3)
    ax2.axvline(0, color=BASE, lw=1.2, zorder=2)
    for yi, d in zip(y, crit_delta, strict=False):
        off = 0.006 if d >= 0 else -0.006
        ax2.text(
            d + off,
            yi,
            f"{d:+.3f}".replace("+", "+").replace("-", "−"),
            va="center",
            ha="left" if d >= 0 else "right",
            color=SECONDARY,
            fontsize=9.5,
        )
    ax2.set_yticks(y)
    ax2.set_yticklabels(crit_names, fontsize=9.5)
    ax2.set_xlim(-0.16, 0.20)
    ax2.set_xlabel("关键类召回变化 Δ Recall")
    ax2.set_title(
        "关键类召回变化（漏检注入 − 匹配奖励）",
        fontsize=12,
        fontweight="bold",
        color=INK,
    )
    _style_axes(ax2, xlabel="关键类召回变化 Δ Recall")

    fig.tight_layout()
    fig.savefig(OUT / "fig_reward_ab.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fig_pipeline()
    fig_training_curves()
    fig_reward_ab()
    print("wrote figures to", OUT)
