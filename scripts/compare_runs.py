#!/usr/bin/env python
"""compare_runs.py -- compare training runs (stage 1.1 verdict tool).

Reads each run's results.csv (per-epoch mAP50 etc.) and, when available, the
official-pipeline multi-IoU JSON reports from scripts/eval_multIou.py, then
writes a markdown comparison: headline table, per-class AP@0.5 delta, critical
classes, and epoch-level curve summary.

Usage:
    python scripts/compare_runs.py \
        --run baseline=runs/rtdetr/tianchi18_full \
        --run hi1536=runs/rtdetr/tianchi18_hi1536 \
        --json baseline=runs/multi_iou/tianchi18_best.json \
        --json hi1536=runs/multi_iou/tianchi18_hi1536.json \
        --out runs/compare/hi1536_vs_baseline.md

--run/--json are repeatable (name=path). Missing files are skipped with a note,
so the script is safe to run while a run is still training.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

CRITICAL = ["hole", "broken_warp", "broken_spandex"]


def load_run(name: str, path: Path) -> dict | None:
    csv = path / "results.csv"
    if not csv.exists():
        print(f"[skip] {name}: no results.csv in {path}")
        return None
    df = pd.read_csv(csv)
    df.columns = [c.strip() for c in df.columns]
    ep_col = "epoch" if "epoch" in df.columns else df.columns[0]
    map50_col = next((c for c in df.columns if "mAP50(B)" in c), None)
    map5095_col = next((c for c in df.columns if "mAP50-95(B)" in c), None)
    p_col = next((c for c in df.columns if "precision(B)" in c), None)
    r_col = next((c for c in df.columns if "recall(B)" in c), None)
    if map50_col is None:
        print(f"[skip] {name}: mAP50 column not found in {csv}")
        return None
    best_i = df[map50_col].idxmax()
    args_yaml = path / "args.yaml"
    imgsz = epochs = None
    if args_yaml.exists():
        for line in args_yaml.read_text(encoding="utf-8").splitlines():
            if line.startswith("imgsz:"):
                imgsz = line.split(":", 1)[1].strip()
            elif line.startswith("epochs:"):
                epochs = line.split(":", 1)[1].strip()
    return {
        "name": name,
        "path": str(path),
        "imgsz": imgsz,
        "epochs_cfg": epochs,
        "df": df,
        "ep_col": ep_col,
        "best": {
            "epoch": int(df.loc[best_i, ep_col]),
            "mAP50": float(df.loc[best_i, map50_col]),
            "mAP50-95": float(df.loc[best_i, map5095_col]) if map5095_col else None,
            "P": float(df.loc[best_i, p_col]) if p_col else None,
            "R": float(df.loc[best_i, r_col]) if r_col else None,
        },
        "final": {
            "epoch": int(df.iloc[-1][ep_col]),
            "mAP50": float(df.iloc[-1][map50_col]),
            "mAP50-95": float(df.iloc[-1][map5095_col]) if map5095_col else None,
        },
        "map50_col": map50_col,
    }


def load_json(name: str, path: Path) -> dict | None:
    if not path.exists():
        print(f"[skip] {name}: no JSON at {path}")
        return None
    d = json.loads(path.read_text(encoding="utf-8"))
    d["_name"] = name
    return d


def fmt(v, nd=4):
    return "—" if v is None else f"{v:.{nd}f}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", default=[], help="name=path (repeatable)")
    ap.add_argument(
        "--json", action="append", default=[], help="name=path (repeatable)"
    )
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    runs = []
    for spec in args.run:
        name, _, p = spec.partition("=")
        r = load_run(name, Path(p))
        if r:
            runs.append(r)
    jsons = {}
    for spec in args.json:
        name, _, p = spec.partition("=")
        j = load_json(name, Path(p))
        if j:
            jsons[name] = j

    if not runs:
        print("no runs loaded — nothing to compare")
        return 1

    lines = []
    lines.append("# 训练运行对比（compare_runs.py 自动生成）\n")
    lines.append("## 1. 总览（results.csv，best = 最高 mAP50 的 epoch）\n")
    lines.append(
        "| 运行 | imgsz | epochs | best mAP50 | best epoch | final mAP50 | best P | best R | best mAP50-95 |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for r in runs:
        b, f = r["best"], r["final"]
        lines.append(
            f"| {r['name']} | {r['imgsz'] or '—'} | {r['epochs_cfg'] or '—'} "
            f"| **{b['mAP50']:.4f}** | {b['epoch']} | {f['mAP50']:.4f} "
            f"| {fmt(b['P'])} | {fmt(b['R'])} | {fmt(b['mAP50-95'])} |"
        )

    if len(runs) >= 2:
        base, new = runs[0], runs[-1]
        d = new["best"]["mAP50"] - base["best"]["mAP50"]
        lines.append("")
        lines.append(
            f"**best mAP50 变化：{base['name']} {base['best']['mAP50']:.4f} → "
            f"{new['name']} {new['best']['mAP50']:.4f}（{d:+.4f}）**"
        )

    lines.append("\n## 2. 官方口径三阈值（eval_multIou.py JSON）\n")
    if not jsons:
        lines.append("_未提供 JSON（训练完成后运行 eval_multIou.py 再传入）。_")
    else:
        lines.append("| 运行 | mAP@0.1 | mAP@0.3 | mAP@0.5 | 官方口径 mean |")
        lines.append("|---|---|---|---|---|")
        for name, j in jsons.items():
            m = j["mAP"]
            off = (m["mAP@0.1"] + m["mAP@0.3"] + m["mAP@0.5"]) / 3
            lines.append(
                f"| {name} | {m['mAP@0.1']:.4f} | {m['mAP@0.3']:.4f} | {m['mAP@0.5']:.4f} | **{off:.4f}** |"
            )
        # per-class delta between the first two jsons (by name)
        names = list(jsons)
        if len(names) >= 2:
            a, b = jsons[names[0]], jsons[names[1]]
            pa = {r["name"]: r for r in a["per_class"]}
            pb = {r["name"]: r for r in b["per_class"]}
            common = sorted(set(pa) & set(pb))
            lines.append(f"\n### 逐类 AP@0.5 变化（{names[0]} → {names[1]}）\n")
            lines.append("| 类 | n_inst | AP@0.5 base | AP@0.5 new | Δ |")
            lines.append("|---|---|---|---|---|")
            rows = []
            for c in common:
                delta = pb[c]["AP@0.5"] - pa[c]["AP@0.5"]
                rows.append(
                    (delta, c, pa[c]["n_inst"], pa[c]["AP@0.5"], pb[c]["AP@0.5"])
                )
            rows.sort(reverse=True)
            for delta, c, n, va, vb in rows:
                mark = " 🔴" if c in CRITICAL else ""
                lines.append(
                    f"| {c}{mark} | {n} | {va:.4f} | {vb:.4f} | {delta:+.4f} |"
                )
            lines.append("\n### 关键类（hole / broken_warp / broken_spandex）\n")
            lines.append(
                "| 类 | AP@0.1 base | AP@0.1 new | AP@0.5 base | AP@0.5 new | R@F1 base | R@F1 new |"
            )
            lines.append("|---|---|---|---|---|---|---|")
            for c in CRITICAL:
                if c in pa and c in pb:
                    lines.append(
                        f"| {c} | {pa[c]['AP@0.1']:.3f} | {pb[c]['AP@0.1']:.3f} "
                        f"| {pa[c]['AP@0.5']:.3f} | {pb[c]['AP@0.5']:.3f} "
                        f"| {pa[c]['R@maxF1']:.3f} | {pb[c]['R@maxF1']:.3f} |"
                    )

    lines.append("\n## 3. mAP50 曲线（每 1/6 处取样）\n")
    for r in runs:
        df, col, ep = r["df"], r["map50_col"], r["ep_col"]
        n = len(df)
        picks = sorted(set(int(x) for x in [n * i / 6 for i in range(7)]))
        picks = [min(p, n - 1) for p in picks]
        sample = ", ".join(
            f"e{int(df.iloc[i][ep])}={df.iloc[i][col]:.3f}" for i in picks
        )
        lines.append(f"- **{r['name']}**：{sample}")

    text = "\n".join(lines) + "\n"
    print(text)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        print(f"\nwritten to {out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
