"""判定单个干预臂 vs 均匀对照 (阶段1.3 协议, 判定规则预先承诺)。

用法: python scripts/ablation_verdict.py <对照JSON> <干预JSON>
退出码: 0=转正(POSITIVE), 1=仍负(negative), 2=部分转正(mean升但crit_recall降)

转正 = 干预官方口径 mean mAP > 对照 + 0.005 且 crit_recall 不降。
crit_recall = hole/broken_warp/broken_spandex 三类 R@maxF1 均值。
"""

import json
import sys

CRIT = ["hole", "broken_warp", "broken_spandex"]


def load(path: str):
    d = json.load(open(path, encoding="utf-8"))
    m = d["mAP"]
    mean = sum(m.values()) / len(m)
    crit = {e["name"]: e["R@maxF1"] for e in d["per_class"] if e["name"] in CRIT}
    missing = [c for c in CRIT if c not in crit]
    if missing:
        raise SystemExit(f"{path}: missing critical classes {missing}")
    return mean, sum(crit.values()) / len(crit), crit


def main() -> int:
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    cmean, ccrit, cdetail = load(sys.argv[1])
    amean, acrit, adetail = load(sys.argv[2])
    d_mean = amean - cmean
    d_crit = acrit - ccrit
    if d_mean > 0.005 and d_crit >= 0.0:
        verdict, code = "POSITIVE", 0
    elif d_mean > 0.005 and d_crit < 0.0:
        verdict, code = "PARTIAL", 2
    else:
        verdict, code = "negative", 1
    print(f"control : mean={cmean:.4f} crit_recall={ccrit:.4f} {cdetail}")
    print(f"arm     : mean={amean:.4f} crit_recall={acrit:.4f} {adetail}")
    print(f"delta   : mean={d_mean:+.4f} crit_recall={d_crit:+.4f}")
    print(f"VERDICT={verdict}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
