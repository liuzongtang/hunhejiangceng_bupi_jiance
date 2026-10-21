#!/usr/bin/env python
"""summarize_uw_seeds.py -- stage 2 UW verdict against the pre-committed gate.

Gate (pre-registered in docs/plan_stage2_adaptive.md, BEFORE results):
  per-seed positive  <=> mean_official > 0.6208 (uniform control 0.6158 + 0.005)
                         AND crit_R not below control
  zero-sum clause    crit_R +>0.02 AND crit_P <-0.02 -> NOT positive
  verdict            positive fraction >= 2/3 of evaluated seeds -> UW POSITIVE
                     else -> UW NEGATIVE / unstable (directional only)

Note (D3 branch): seed 7 was added because the 3-seed verdict came out
UNSTABLE (1/3). With N=4 the same pre-registered 2/3 rule requires >=3
positive seeds; the 4th seed can only downgrade (to NEGATIVE) or leave
UNSTABLE in place -- it cannot by itself promote to POSITIVE.

Reads:
  runs/multi_iou/uw_seed*.json          (official-pipeline mean + per-class)
  runs/ablation_adaptive/uw_seed*/adaptive_weights.json (trajectory diagnostics)

Usage:
    python scripts/summarize_uw_seeds.py
    python scripts/summarize_uw_seeds.py --control runs/multi_iou/strong_r2_uniform.json
"""

import argparse
import json
import math
from pathlib import Path

SEEDS = [42, 123, 2024, 7]
CRIT = ("hole", "broken_warp", "broken_spandex")
GATE_MARGIN = 0.005  # pre-registered: mean > control + 0.005
CONTROL_DEFAULT = "runs/multi_iou/strong_r2_uniform.json"
# Pre-registered zero-sum clause (plan 1.4): a seed whose crit_R rises >0.02
# while crit_P falls >0.02 reproduces the recall-precision zero-sum and does
# NOT count as positive, even if its mean passes the gate.
ZS_R, ZS_P = 0.02, -0.02


def load_json(p: Path):
    if not p.exists():
        return None
    return json.load(open(p, encoding="utf-8"))


def official_mean(d: dict) -> float:
    m = d["mAP"]
    return (m["mAP@0.1"] + m["mAP@0.3"] + m["mAP@0.5"]) / 3


def crit_recall(d: dict) -> float:
    per = [p for p in d["per_class"] if p["name"] in CRIT]
    return sum(p["R@maxF1"] for p in per) / max(len(per), 1)


def crit_precision(d: dict) -> float:
    per = [p for p in d["per_class"] if p["name"] in CRIT]
    return sum(p["P@maxF1"] for p in per) / max(len(per), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--control", default=CONTROL_DEFAULT)
    ap.add_argument("--multi-root", default="runs/multi_iou")
    ap.add_argument("--adaptive-root", default="runs/ablation_adaptive")
    args = ap.parse_args()

    ctl = load_json(Path(args.control))
    if ctl is None:
        print(f"[ERR] control archive missing: {args.control}")
        return
    ctl_mean, ctl_crit, ctl_cp = (
        official_mean(ctl),
        crit_recall(ctl),
        crit_precision(ctl),
    )
    gate = ctl_mean + GATE_MARGIN
    print(
        f"control: mean={ctl_mean:.4f} crit_R={ctl_crit:.4f} crit_P={ctl_cp:.4f}  ->  gate mean>{gate:.4f}\n"
    )
    print(
        f"zero-sum clause (pre-registered): crit_R +>{ZS_R} AND crit_P <{ZS_P} -> NOT positive\n"
    )

    rows = []
    for s in SEEDS:
        d = load_json(Path(args.multi_root) / f"uw_seed{s}.json")
        w = load_json(
            Path(args.adaptive_root) / f"uw_seed{s}" / "adaptive_weights.json"
        )
        if d is None:
            print(
                f"seed {s:>4}: [MISSING] runs/multi_iou/uw_seed{s}.json not yet evaluated"
            )
            rows.append(None)
            continue
        mean, cr, cp = official_mean(d), crit_recall(d), crit_precision(d)
        dR, dP = cr - ctl_crit, cp - ctl_cp
        zs = (dR > ZS_R) and (dP < ZS_P)
        pos = (mean > gate) and (cr >= ctl_crit - 1e-9) and not zs
        wf = None
        if w is not None and w.get("trajectory"):
            last = w["trajectory"][-1]
            wf = (last["w_cls"], last["w_reg"])
        rows.append((s, mean, cr, cp, pos, wf, zs))
        tag = "POSITIVE" if pos else ("zero-sum(not positive)" if zs else "negative")
        print(
            f"seed {s:>4}: mean={mean:.4f} (Δ={mean - ctl_mean:+.4f})  crit_R={cr:.4f} "
            f"(Δ={dR:+.4f})  crit_P={cp:.4f} (Δ={dP:+.4f})  final_w={wf}  ->  {tag}"
        )

    ok = [r for r in rows if r is not None]
    if len(ok) < len(SEEDS):
        print(f"\n[PENDING] {len(ok)}/{len(SEEDS)} seeds evaluated")
        return
    n = len(ok)
    n_pos = sum(r[4] for r in ok)
    mean_avg = sum(r[1] for r in ok) / n
    crit_avg = sum(r[2] for r in ok) / n
    cp_avg = sum(r[3] for r in ok) / n
    spread = max(r[1] for r in ok) - min(r[1] for r in ok)
    need = math.ceil(2.0 / 3.0 * n)  # pre-registered 2/3 rule, N seeds
    print(
        f"\n=== VERDICT ({n_pos}/{n} positive after zero-sum clause; "
        f"pre-registered threshold >= {need}/{n}) ==="
    )
    print(
        f"seed-avg mean={mean_avg:.4f} (Δ={mean_avg - ctl_mean:+.4f})  "
        f"seed-avg crit_R={crit_avg:.4f} (Δ={crit_avg - ctl_crit:+.4f})  "
        f"seed-avg crit_P={cp_avg:.4f}  seed spread={spread:.4f} "
        f"({spread / GATE_MARGIN:.1f}x gate width)"
    )
    if n_pos >= need and mean_avg > gate and crit_avg >= ctl_crit - 1e-9:
        print("UW POSITIVE -- adaptive weighting breaks the zero-sum (>= 2/3 seeds)")
    elif mean_avg <= gate or crit_avg < ctl_crit - 1e-9:
        # pre-registered decision tree (verdict template section 2, branch 4):
        # seed-average mean below the gate OR crit_R down -> zero-sum, NEGATIVE
        print(
            "UW NEGATIVE -- seed-average mean below gate or crit_R down; "
            "adaptive weighting reproduces the zero-sum (recorded as a negative result)"
        )
    elif n_pos >= 1:
        print(f"UW UNSTABLE -- {n_pos}/{n} positive (< 2/3); directional signal only")
        if n > 3:
            print(
                "  4th-seed branch resolved: effect remains below the "
                "pre-registered 2/3 bar -- NOT claimed as a working intervention"
            )
    else:
        print("UW NEGATIVE -- adaptive weighting also fails to break the zero-sum")

    # trajectory diagnostic: do the seeds converge to the same ratio?
    ratios = [r[5][0] / r[5][1] for r in ok if r[5]]
    if ratios:
        print(
            f"\nfinal cls:reg ratios: {[round(r, 2) for r in ratios]} "
            f"(spread {max(ratios) / min(ratios):.2f}x)"
        )
        print(
            "consistent ratio >> 1 => default cls:reg gain is measurably sub-optimal "
            "for the official metric (diagnostic value regardless of verdict)"
        )


if __name__ == "__main__":
    main()
