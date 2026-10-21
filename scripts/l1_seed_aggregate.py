"""Layer-1 奖励臂多种子聚合 (阶段1.3 收尾: seed 42/123/2024 重复实验)。

对每个种子计算 A/B delta (干预 vs 同种子均匀对照), 再跨种子聚合:
  - 逐种子 delta 表
  - delta 均值 ± 样本标准差, 正负号一致性
判定 (与协议同风格, 预先固定):
  CONFIRMED_POSITIVE : mean/crit 两个 delta 的均值均 > 0 且三种子同号
  LIKELY_POSITIVE    : 两个 delta 均值均 > 0 且 >= 2/3 种子同号
  NOISE              : 两个 delta 均值均接近 0 (|mean| < 0.002) 或符号不一致
  CONFIRMED_NEGATIVE : mean delta 均值 < -0.002
用法: python scripts/l1_seed_aggregate.py [--root runs/multi_iou] [seed...]
"""

import json
import statistics as st
import sys

CRIT = ["hole", "broken_warp", "broken_spandex"]
# seed -> (对照JSON名, 干预JSON名); seed 42 用臂4原始归档名
DEFAULT = {
    42: ("strong_l1_ctl.json", "strong_l1_reward.json"),
    123: ("l1_seed123_ctl.json", "l1_seed123_reward.json"),
    2024: ("l1_seed2024_ctl.json", "l1_seed2024_reward.json"),
}


def load(path: str):
    d = json.load(open(path, encoding="utf-8"))
    m = d["mAP"]
    mean = sum(m.values()) / len(m)
    crit = {e["name"]: e["R@maxF1"] for e in d["per_class"] if e["name"] in CRIT}
    missing = [c for c in CRIT if c not in crit]
    if missing:
        raise SystemExit(f"{path}: missing critical classes {missing}")
    return mean, sum(crit.values()) / len(crit)


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    root = "runs/multi_iou"
    if "--root" in sys.argv:
        root = sys.argv[sys.argv.index("--root") + 1]
    seeds = [int(s) for s in args] or sorted(DEFAULT)

    rows, missing = [], []
    for s in seeds:
        cname, aname = DEFAULT.get(
            s, (f"l1_seed{s}_ctl.json", f"l1_seed{s}_reward.json")
        )
        try:
            cmean, ccrit = load(f"{root}/{cname}")
            amean, acrit = load(f"{root}/{aname}")
        except (FileNotFoundError, json.JSONDecodeError) as e:
            missing.append(f"seed {s}: {e}")
            continue
        rows.append(
            dict(
                seed=s,
                cmean=cmean,
                amean=amean,
                d_mean=amean - cmean,
                ccrit=ccrit,
                acrit=acrit,
                d_crit=acrit - ccrit,
            )
        )

    print("=== per-seed A/B delta (Layer-1 reward vs uniform control) ===")
    print(
        f"{'seed':>6} {'ctl_mean':>9} {'arm_mean':>9} {'d_mean':>8} "
        f"{'ctl_crit':>9} {'arm_crit':>9} {'d_crit':>8}"
    )
    for r in rows:
        print(
            f"{r['seed']:>6} {r['cmean']:>9.4f} {r['amean']:>9.4f} {r['d_mean']:>+8.4f} "
            f"{r['ccrit']:>9.4f} {r['acrit']:>9.4f} {r['d_crit']:>+8.4f}"
        )
    if missing:
        print("MISSING:", *missing, sep="\n  ")
    if len(rows) < 2:
        print("VERDICT=INSUFFICIENT (need >= 2 seeds)")
        return 3

    dm = [r["d_mean"] for r in rows]
    dc = [r["d_crit"] for r in rows]
    dm_mean, dc_mean = st.mean(dm), st.mean(dc)
    dm_sd = st.stdev(dm) if len(dm) > 1 else 0.0
    dc_sd = st.stdev(dc) if len(dc) > 1 else 0.0
    pos_m = sum(1 for x in dm if x > 0)
    pos_c = sum(1 for x in dc if x > 0)
    print(
        f"\nn_seeds={len(rows)}  d_mean={dm_mean:+.4f}±{dm_sd:.4f} (min {min(dm):+.4f}, max {max(dm):+.4f}, {pos_m}/{len(dm)} positive)"
    )
    print(
        f"                d_crit={dc_mean:+.4f}±{dc_sd:.4f} (min {min(dc):+.4f}, max {max(dc):+.4f}, {pos_c}/{len(dc)} positive)"
    )

    n = len(rows)
    if dm_mean > 0 and dc_mean > 0 and pos_m == n and pos_c == n:
        verdict = "CONFIRMED_POSITIVE"
    elif (
        dm_mean > 0 and dc_mean > 0 and pos_m >= (n + 1) // 2 and pos_c >= (n + 1) // 2
    ):
        verdict = "LIKELY_POSITIVE"
    elif abs(dm_mean) < 0.002 and abs(dc_mean) < 0.002:
        verdict = "NOISE"
    elif dm_mean <= -0.002:
        verdict = "CONFIRMED_NEGATIVE"
    else:
        verdict = "NOISE"
    print(f"VERDICT={verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
