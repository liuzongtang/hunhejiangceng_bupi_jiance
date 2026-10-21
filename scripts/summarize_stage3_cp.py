#!/usr/bin/env python
"""summarize_stage3_cp.py -- Copy-Paste arm verdict against the pre-committed gate.

Gate (pre-registered in docs/plan_stage3_breakthrough.md section 5, BEFORE training):
  positive  <=> mean_official > 0.6399 (mosaic1280 control 0.6349 + 0.005)
                 AND crit_R not below control
  zero-sum  <=> mean <= 0.6399 but crit_R +0.02 above control while crit_P drops
                 more than 0.02 below (data-layer zero-sum: stronger mechanism
                 evidence, same signature as stage 2 layers)
  negative  <=> everything else

Control: runs/multi_iou/mosaic1280_official.json (identical protocol, the only
difference is copy_paste=0.5 on the treatment arm).

Usage:
    python scripts/summarize_stage3_cp.py
"""

import json
from pathlib import Path

CRIT = ("hole", "broken_warp", "broken_spandex")
GATE_MARGIN = 0.005
CONTROL = Path("runs/multi_iou/mosaic1280_official.json")
TREAT = Path("runs/multi_iou/tianchi18_cp1280.json")


def official_mean(d):
    m = d["mAP"]
    return (m["mAP@0.1"] + m["mAP@0.3"] + m["mAP@0.5"]) / 3


def crit(d, key):
    per = [p for p in d["per_class"] if p["name"] in CRIT]
    return sum(p[key] for p in per) / len(per)


def main():
    if not TREAT.exists():
        print(f"[PENDING] {TREAT} not found -- training/eval chain still running")
        return
    ctl = json.load(open(CONTROL, encoding="utf-8"))
    trt = json.load(open(TREAT, encoding="utf-8"))
    cm, cR, cP = official_mean(ctl), crit(ctl, "R@maxF1"), crit(ctl, "P@maxF1")
    m, R, P = official_mean(trt), crit(trt, "R@maxF1"), crit(trt, "P@maxF1")
    gate = cm + GATE_MARGIN
    zero_sum = (R - cR > 0.02) and (P - cP < -0.02)
    positive = (m > gate) and (cR <= R) and not zero_sum
    print(
        f"control (mosaic1280): mean={cm:.4f} mAP50={ctl['mAP']['mAP@0.5']:.4f} "
        f"crit_R={cR:.4f} crit_P={cP:.4f}"
    )
    print(
        f"copy-paste arm      : mean={m:.4f} (d={m - cm:+.4f}) "
        f"mAP50={trt['mAP']['mAP@0.5']:.4f} crit_R={R:.4f} (d={R - cR:+.4f}) "
        f"crit_P={P:.4f} (d={P - cP:+.4f})"
    )
    print(
        f"gate: mean > {gate:.4f} and crit_R >= {cR:.4f}; "
        f"zero-sum clause: {'TRIGGERED' if zero_sum else 'not triggered'}"
    )
    if positive:
        print("=== VERDICT: POSITIVE -- data-layer copy-paste breaks the ceiling ===")
    elif zero_sum:
        print(
            "=== VERDICT: ZERO-SUM replicates at the data layer (negative result, "
            "stronger mechanism evidence) ==="
        )
    else:
        print("=== VERDICT: NEGATIVE ===")


if __name__ == "__main__":
    main()
