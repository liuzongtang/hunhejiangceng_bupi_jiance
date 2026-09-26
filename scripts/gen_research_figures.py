# -*- coding: utf-8 -*-
"""Generate figures for docs/研究状况与选题价值.md (研究状况与选题价值).

Reads real training results from runs/ and writes 4 figures into docs/figures/.
Uses the project's validated light palette (see docs source-of-truth in README).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib import font_manager
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# ---- Chinese font (Windows) ----
for name in ("Microsoft YaHei", "SimHei", "SimSun", "Noto Sans CJK SC"):
    if any(f.name == name for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.sans-serif"] = [name, "Arial"]
        break
plt.rcParams["axes.unicode_minus"] = False

# ---- Palette (light surface) ----
BLUE = "#2a78d6"  # series 1
ORANGE = "#eb6834"  # series 2
AQUA = "#1baf7a"  # series 3
RED = "#e34948"  # series 8
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
# Fig 1 — 训练收敛曲线 (RT-DETR-L, 50 epochs)
# ============================================================
def fig1_training_curves():
    import pandas as pd

    df = pd.read_csv(ROOT / "runs" / "rtdetr" / "tianchi20" / "results.csv").dropna()
    ep = df["epoch"].astype(int).values
    m50 = df["metrics/mAP50(B)"].values
    m5095 = df["metrics/mAP50-95(B)"].values

    fig, ax = plt.subplots(figsize=(7.2, 4.1), dpi=150)
    ax.plot(ep, m50, color=BLUE, lw=2, label="mAP50")
    ax.plot(ep, m5095, color=ORANGE, lw=2, label="mAP50-95")

    # direct labels at final point (selective)
    ax.annotate(
        f"{m50[-1]:.3f}",
        (ep[-1], m50[-1]),
        textcoords="offset points",
        xytext=(-6, 6),
        color=BLUE,
        fontweight="bold",
        ha="right",
    )
    ax.annotate(
        f"{m5095[-1]:.3f}",
        (ep[-1], m5095[-1]),
        textcoords="offset points",
        xytext=(-6, -12),
        color=ORANGE,
        fontweight="bold",
        ha="right",
    )

    ax.set_xlabel("训练轮次 epoch")
    ax.set_ylabel("精度")
    ax.set_xlim(0, ep[-1] + 1)
    ax.set_ylim(0.15, 0.70)
    ax.legend(frameon=False, loc="lower right")
    _style_axes(ax)
    fig.tight_layout()
    fig.savefig(OUT / "fig1_training_curves.png")
    plt.close(fig)


# ============================================================
# Fig 2 — 逐类 mAP50（20 类，长尾分布）
# ============================================================
ZH = {
    "hole": "破洞",
    "stain": "污渍",
    "three_silk": "三丝",
    "knot": "结头",
    "flower_board": "花板跳",
    "hundred_feet": "百脚",
    "hair_particle": "毛粒",
    "coarse_warp": "粗经",
    "loose_warp": "松经",
    "broken_warp": "断经",
    "hanging_warp": "吊经",
    "coarse_weft": "粗纬",
    "weft_shrink": "纬缩",
    "size_stain": "浆斑",
    "warping_knot": "整经结",
    "star_skip": "星跳",
    "broken_spandex": "断氨纶",
    "dense_section": "稀密档",
    "surface_mark": "磨痕",
    "weave_defect": "死皱",
}


def fig2_per_class_map():
    meta = json.loads(
        (
            ROOT / "runs" / "rtdetr" / "tianchi20" / "analysis" / "metrics_summary.json"
        ).read_text(encoding="utf-8")
    )
    ap = meta["per_class_ap50"]
    overall = meta["map50"]
    items = sorted(ap.items(), key=lambda kv: kv[1])  # ascending

    names = [ZH.get(k, k) for k, _ in items]
    vals = [v for _, v in items]
    colors = [RED if k == "broken_warp" else BLUE for k, _ in items]

    fig, ax = plt.subplots(figsize=(7.2, 5.4), dpi=150)
    ax.barh(names, vals, color=colors, height=0.62, zorder=3)

    ax.axvline(overall, color=MUTED, lw=1.4, ls="--", zorder=2)
    ax.text(
        overall,
        len(names) - 0.35,
        f" 平均 mAP50 {overall:.3f}",
        color=SECONDARY,
        va="center",
        fontsize=10,
    )

    # selective direct labels on the lowest three + the critical class
    for i, (k, v) in enumerate(items):
        if k in ("hair_particle", "weft_shrink", "broken_warp"):
            ax.text(v + 0.006, i, f"{v:.3f}", va="center", color=SECONDARY, fontsize=9)

    # legend via proxy for the critical class
    from matplotlib.patches import Patch

    ax.legend(
        handles=[
            Patch(color=BLUE, label="普通类"),
            Patch(color=RED, label="断经（关键类，严重缺陷）"),
        ],
        frameon=False,
        loc="lower right",
        fontsize=10,
    )

    ax.set_xlabel("逐类 mAP50")
    ax.set_xlim(0, 1.0)
    ax.tick_params(axis="y", labelsize=10)
    _style_axes(ax)
    fig.tight_layout()
    fig.savefig(OUT / "fig2_per_class_map.png")
    plt.close(fig)


# ============================================================
# Fig 3 — 奖励注入 A/B（召回 vs 精确 vs mAP50）
# ============================================================
def fig3_reward_ab():
    import numpy as np

    # final-epoch (5) metrics from runs/rtdetr-reward/*/results.csv
    metrics = ["召回率 Recall", "精确率 Precision", "mAP50"]
    arm_reward = [0.52313, 0.59378, 0.57288]  # beta0.5_e5
    arm_miss = [0.55310, 0.57163, 0.57643]  # beta0.5_miss (include-misses)
    deltas = [a - b for a, b in zip(arm_miss, arm_reward, strict=False)]

    x = np.arange(len(metrics))
    w = 0.36
    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=150)
    b1 = ax.bar(x - w / 2, arm_reward, w, color=BLUE, label="奖励注入 β=0.5", zorder=3)
    b2 = ax.bar(x + w / 2, arm_miss, w, color=ORANGE, label="奖励 + 漏检注入", zorder=3)

    # direct labels for each bar
    for bars in (b1, b2):
        for r in bars:
            ax.text(
                r.get_x() + r.get_width() / 2,
                r.get_height() + 0.008,
                f"{r.get_height():.3f}",
                ha="center",
                color=SECONDARY,
                fontsize=9,
            )
    # delta annotations
    for xi, d in zip(x, deltas, strict=False):
        sign = "+" if d >= 0 else "−"
        ax.text(
            xi,
            max(arm_reward[xi], arm_miss[xi]) + 0.055,
            f"Δ {sign}{abs(d):.3f}",
            ha="center",
            color=GREEN if d >= 0 else RED,
            fontweight="bold",
            fontsize=10,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 0.72)
    ax.set_ylabel("数值")
    ax.legend(frameon=False, loc="lower left", fontsize=10)
    _style_axes(ax)
    fig.tight_layout()
    fig.savefig(OUT / "fig3_reward_ab.png")
    plt.close(fig)


# ============================================================
# Fig 4 — 混淆类合并后的召回恢复（标签噪声诊断）
# ============================================================
def fig4_merge_recovery():
    import numpy as np

    groups = ["纬缩 ∪ 粗纬", "断经 ∪ 磨痕"]
    a = [0.40, 0.52]  # 纬缩 / 断经
    b = [0.79, 0.83]  # 粗纬 / 磨痕
    merged = [0.88, 0.79]

    x = np.arange(len(groups))
    w = 0.25
    fig, ax = plt.subplots(figsize=(7.2, 4.0), dpi=150)
    ax.bar(x - w, a, w, color=BLUE, label="易混淆类 A", zorder=3)
    ax.bar(x, b, w, color="#86b6ef", label="易混淆类 B", zorder=3)
    ax.bar(x + w, merged, w, color=AQUA, label="合并后（同一类）", zorder=3)

    for xi, (ai, bi, mi) in enumerate(zip(a, b, merged, strict=False)):
        for off, v in ((-w, ai), (0, bi), (w, mi)):
            ax.text(
                xi + off,
                v + 0.012,
                f"{v:.2f}",
                ha="center",
                color=SECONDARY,
                fontsize=10,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(groups, fontsize=11)
    ax.set_ylabel("召回率 Recall")
    ax.set_ylim(0, 1.0)
    ax.legend(frameon=False, loc="upper left", fontsize=10)
    _style_axes(ax)
    fig.tight_layout()
    fig.savefig(OUT / "fig4_merge_recovery.png")
    plt.close(fig)


# ============================================================
# Fig 5 — 系统研究框架（六层）
# ============================================================
def fig5_framework():
    layers = [
        ("数据层", "Tianchi 20 类数据集 · 1280 拉伸方图"),
        ("模型层", "RT-DETR-L 检测器（PyTorch / ONNX 双后端）"),
        (
            "奖励层",
            "七维混合奖惩 D01~D07（可微）· L_total = L_det + β·L_scalar + γ·L_consistency",
        ),
        ("诊断层", "逐类 mAP + 混淆矩阵 + 双向判定 + 裁剪复核"),
        ("告警层", "effective_alert_severity() 易混淆类保守升级"),
        ("应用层", "FastAPI + WebSocket + 中文看板 + 边缘部署"),
    ]
    n = len(layers)
    fig, ax = plt.subplots(figsize=(8.2, 6.6), dpi=150)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, n * 1.6 + 0.4)
    ax.axis("off")
    for i, (title, desc) in enumerate(layers):
        y = (n - 1 - i) * 1.6 + 0.4
        ax.add_patch(
            plt.Rectangle(
                (0.8, y),
                8.4,
                1.25,
                facecolor="#eaf2fc",
                edgecolor=BLUE,
                lw=1.6,
                zorder=2,
            )
        )
        ax.text(
            1.05,
            y + 0.82,
            title,
            fontsize=13,
            fontweight="bold",
            color=BLUE,
            va="center",
            zorder=3,
        )
        ax.text(
            1.05, y + 0.34, desc, fontsize=9.5, color=SECONDARY, va="center", zorder=3
        )
        if i < n - 1:
            ax.annotate(
                "",
                xy=(5.0, y + 1.25),
                xytext=(5.0, y + 1.72),
                arrowprops={"arrowstyle": "-|>", "color": MUTED, "lw": 1.8},
            )
    fig.tight_layout()
    fig.savefig(OUT / "fig5_framework.png")
    plt.close(fig)


if __name__ == "__main__":
    fig1_training_curves()
    fig2_per_class_map()
    fig3_reward_ab()
    fig4_merge_recovery()
    fig5_framework()
    print("wrote figures to", OUT)
