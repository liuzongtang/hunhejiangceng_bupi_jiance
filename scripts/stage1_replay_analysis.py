"""
阶段1 CPU 回放分析 (计划文档 §四-阶段1, 预承诺门已登记, 事后不移动)
  1.2 (C3) 固定复核预算选择性拦截: 门 = 关键缺陷拦截率显著高于 max-conf
  1.3 (C7) 工业成本模型: 门 = 三档决策总成本 < 单阈值
  1.4 (C8) 七维可调性曲线: 门 = 连续可调 (TPR-FPR 平滑单调)
  1.5 (C4前置) objectness/分数域偏移分析: 域内 vs AITEX 跨域分布分离度

数据 (全部复用, 不碰 GPU):
  runs/alarm_replay/detections_mosaic1280.jsonl  (val 957 张, 逐查询 18 类分数)
  runs/alarm_replay/detections_aitex.jsonl        (AITEX 141 张跨域负样本)
  runs/alarm_replay/mosaic1280/{R_global,max_conf}.npy  (七维图像分 + max-conf 基线)
判定用标签: gt_defective + gt_boxes (关键类 = hole/broken_warp/broken_spandex = id 0/9/15)
"""

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
REPLAY = ROOT / "runs" / "alarm_replay"
OUT = REPLAY / "stage1"
OUT.mkdir(parents=True, exist_ok=True)

CRITICAL = {0, 9, 15}  # hole, broken_warp, broken_spandex


def load_jsonl(p: Path):
    rows = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def det_array(dets):
    """detections -> (n,24) ndarray: [cx,cy,w,h,conf,cls,s0..s17]"""
    if not dets:
        return np.zeros((0, 24), dtype=np.float64)
    return np.asarray(dets, dtype=np.float64)


def main():
    val = load_jsonl(REPLAY / "detections_mosaic1280.jsonl")
    aitex = load_jsonl(REPLAY / "detections_aitex.jsonl")
    R_global = np.load(REPLAY / "mosaic1280" / "R_global.npy")
    max_conf_npy = np.load(REPLAY / "mosaic1280" / "max_conf.npy")
    assert len(val) == len(R_global) == len(max_conf_npy) == 957, "数据长度不符"

    # ---- 每图标签与分数 ----
    y = np.array([bool(r["gt_defective"]) for r in val])  # 583 有瑕 / 374 干净
    crit_img = np.array(
        [any(int(g[0]) in CRITICAL for g in r["gt_boxes"]) for r in val]
    )
    n_crit_boxes = np.array(
        [sum(1 for g in r["gt_boxes"] if int(g[0]) in CRITICAL) for r in val]
    )
    max_conf = np.array(
        [
            det_array(r["detections"])[:, 4].max() if r["detections"] else 0.0
            for r in val
        ]
    )
    # 顺序对齐校验: 重算 max-conf 与存档 npy 必须一致
    align_ok = bool(np.allclose(max_conf, max_conf_npy, atol=1e-6))
    print(f"[对齐校验] 重算 max_conf vs npy allclose = {align_ok}")
    assert align_ok, "jsonl 与 npy 顺序不一致, 中止"

    n_crit_img = int(crit_img.sum())
    print(
        f"[数据] val 957: 有瑕 {int(y.sum())} / 干净 {int((~y).sum())}; "
        f"含关键类图 {n_crit_img}, 关键类 GT 框 {int(n_crit_boxes.sum())}"
    )

    scores = {"max_conf": max_conf, "seven_dim_R": R_global}

    # ================= 1.2 固定复核预算选择性拦截 (C3) =================
    # 注意: 既有 REPORT.md 称"580 张含关键类 GT"与原始 GT 不符 (关键类 {0,9,15} 实为 101 张/166 框,
    # 疑将有瑕图 583 张误计)。本分析以 jsonl 原始 gt_boxes 重算为准。
    # 公平性对照: R_global 含 D1 严重度先验 (关键类门控), class-agnostic 的 max-conf 没有;
    # 故补 sev_weighted (严重度×conf 的类感知单分) 作对照臂, 分离"严重度先验"与"七维机制"的贡献。
    SEV = {0: 1.0, 9: 1.0, 15: 1.0, 14: 0.6, 17: 0.6}  # 关键类 1.0, 次级 0.6, 其余 0.3
    sev_conf = np.array(
        [
            max((SEV.get(int(d[5]), 0.3) * d[4] for d in r["detections"]), default=0.0)
            for r in val
        ]
    )
    print("\n===== 1.2 固定复核预算选择性拦截 =====")
    budgets = [100, 200, 300, 400, 500]
    lines = ["budget,score,crit_img_intercept,crit_box_intercept,any_defect_intercept"]
    for name, s in [*scores.items(), ("sev_weighted", sev_conf)]:
        order = np.argsort(-s)
        for B in budgets:
            top = order[:B]
            ci = crit_img[top].sum() / n_crit_img
            cb = n_crit_boxes[top].sum() / n_crit_boxes.sum()
            ad = y[top].sum() / y.sum()
            lines.append(f"{B},{name},{ci:.4f},{cb:.4f},{ad:.4f}")
            print(
                f"  B={B:3d} {name:12s} 关键类图拦截 {ci:.4f}  关键框拦截 {cb:.4f}  有瑕图拦截 {ad:.4f}"
            )
    (OUT / "fixed_review_budget.csv").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )

    # ================= 1.3 工业成本模型 (C7) =================
    print("\n===== 1.3 工业成本模型 =====")
    # 成本参数 (归一化 C_false=1):
    #   C_miss = r (漏检一张有瑕图), C_false = 1 (干净图误停机), C_rev = 0.1 (人工复核一张)
    #   p_catch = 0.9 (人工复核发现缺陷的概率)
    C_rev, p_catch = 0.1, 0.9
    tau_stop, tau_review = 0.14, 0.02  # 论文 4.13 三档既定阈值 (R_global)

    def cost_single(s, tau, r):
        stop = s >= tau
        fp = int(((~y) & stop).sum())  # 干净图误停机
        fn = int((y & ~stop).sum())  # 有瑕图漏检
        return r * fn + 1.0 * fp, fn, fp

    def cost_three_tier(s, r, t_stop, t_rev):
        stop = s >= t_stop
        review = (s >= t_rev) & (~stop)
        fp = int(((~y) & stop).sum())
        fn_pass = int((y & (s < t_rev)).sum())  # 直接放行的有瑕图
        # 复核档: 有瑕图以 (1-p_catch) 概率漏检; 干净图付出复核成本
        fn_rev = (y & review).sum() * (1 - p_catch)
        n_rev = int(review.sum())
        return (
            r * (fn_pass + fn_rev) + 1.0 * fp + C_rev * n_rev,
            fn_pass + fn_rev,
            fp,
            n_rev,
        )

    tau_grid = np.round(np.arange(0.02, 0.66, 0.02), 4)
    cost_rows = [
        "policy,score,r,C_miss,C_false,C_rev,p_catch,total_cost,FN,FP_stop,n_review"
    ]
    print(
        f"  参数: C_rev={C_rev}, p_catch={p_catch}, 三档 tau_stop={tau_stop}/tau_review={tau_review}"
    )
    for r in [1, 2, 5, 10, 20, 50]:
        # P1 单阈值 max-conf (τ 优化)
        best = min((cost_single(max_conf, t, r)[0], t) for t in tau_grid)
        c, fn, fp = cost_single(max_conf, best[1], r)
        cost_rows.append(
            f"single_maxconf,max_conf,{r},{r},1,{C_rev},{p_catch},{c:.1f},{fn},{fp},0"
        )
        # P3 单分数三档 max-conf (双阈优化)
        best3 = min(
            (cost_three_tier(max_conf, r, ts, tr)[0], ts, tr)
            for ts in tau_grid
            for tr in tau_grid
            if tr < ts
        )
        c, fn, fp, nr = cost_three_tier(max_conf, r, best3[1], best3[2])
        cost_rows.append(
            f"three_tier_maxconf,max_conf,{r},{r},1,{C_rev},{p_catch},{c:.1f},{fn:.1f},{fp},{nr}"
        )
        # P2 论文三档 R_global (固定阈值)
        c, fn, fp, nr = cost_three_tier(R_global, r, tau_stop, tau_review)
        cost_rows.append(
            f"three_tier_Rglobal,seven_dim_R,{r},{r},1,{C_rev},{p_catch},{c:.1f},{fn:.1f},{fp},{nr}"
        )
        # P2' R_global 三档 (双阈优化, 公平对照)
        best3r = min(
            (cost_three_tier(R_global, r, ts, tr)[0], ts, tr)
            for ts in tau_grid
            for tr in tau_grid
            if tr < ts
        )
        c, fn, fp, nr = cost_three_tier(R_global, r, best3r[1], best3r[2])
        cost_rows.append(
            f"three_tier_Rglobal_opt,seven_dim_R,{r},{r},1,{C_rev},{p_catch},{c:.1f},{fn:.1f},{fp},{nr}"
        )
        print(
            f"  r={r:2d}: 单阈max-conf {cost_single(max_conf, best[1], r)[0]:7.1f} (τ={best[1]}) | "
            f"三档R_global固定 {cost_three_tier(R_global, r, tau_stop, tau_review)[0]:7.1f} | "
            f"三档R_global优化 {best3r[0]:7.1f} | 三档max-conf优化 {best3[0]:7.1f}"
        )
    (OUT / "cost_model.csv").write_text("\n".join(cost_rows) + "\n", encoding="utf-8")

    # ================= 1.4 七维可调性曲线 (C8) =================
    print("\n===== 1.4 可调性曲线 =====")
    tpr_fpr_rows = ["score,tau,TPR,FPR,precision,F1"]
    for name, s in scores.items():
        prev_fpr = 1.1
        mono = True
        for t in np.arange(0.001, 0.801, 0.002):
            pred = s >= t
            tp = int((y & pred).sum())
            fp = int(((~y) & pred).sum())
            tpr = tp / y.sum()
            fpr = fp / (~y).sum()
            prec = tp / (tp + fp) if tp + fp else 0.0
            f1 = 2 * prec * tpr / (prec + tpr) if prec + tpr else 0.0
            tpr_fpr_rows.append(
                f"{name},{t:.3f},{tpr:.4f},{fpr:.4f},{prec:.4f},{f1:.4f}"
            )
            if fpr > prev_fpr + 1e-9:
                mono = False
            prev_fpr = fpr
        # 工作报告点 (FPR = FP/(~y).sum())
        n_clean = int((~y).sum())
        for target in [0.90, 0.95, 0.98]:
            cand = []
            for t in np.arange(0.001, 0.801, 0.002):
                tp = int((y & (s >= t)).sum())
                fp = int(((~y) & (s >= t)).sum())
                cand.append((abs(tp / y.sum() - target), t, fp / n_clean))
            cand.sort()
            _, t, fpr = cand[0]
            print(f"  {name:12s} TPR≈{target:.2f}: τ={t:.3f} FPR={fpr:.4f}")
        print(f"  {name:12s} FPR 随 TPR 单调不增: {mono}")
    (OUT / "tunable_curve.csv").write_text(
        "\n".join(tpr_fpr_rows) + "\n", encoding="utf-8"
    )

    # ================= 1.5 域偏移分析 (C4 前置) =================
    print("\n===== 1.5 域偏移分析 (val-clean vs AITEX) =====")

    def det_stats(rows):
        confs, ents, gaps = [], [], []
        for r in rows:
            for d in det_array(r["detections"]):
                sc = d[6:24]
                confs.append(d[4])
                p = np.exp(sc - sc.max())
                p = p / p.sum()
                ents.append(-(p * np.log(p + 1e-12)).sum() / np.log(18))
                srt = np.sort(sc)[::-1]
                gaps.append(srt[0] - srt[1])
        return np.array(confs), np.array(ents), np.array(gaps)

    val_clean = [val[i] for i in range(len(val)) if not y[i]]
    c_in, e_in, g_in = det_stats(val_clean)
    c_ae, e_ae, g_ae = det_stats(aitex)

    def q(a):
        return {
            k: float(np.quantile(a, v))
            for k, v in [("p10", 0.1), ("p50", 0.5), ("p90", 0.9), ("mean", None)]
            if v is not None
        } | {"mean": float(a.mean())}

    img_max_in = np.array(
        [
            det_array(r["detections"])[:, 4].max() if r["detections"] else 0.0
            for r in val_clean
        ]
    )
    img_max_ae = np.array(
        [
            det_array(r["detections"])[:, 4].max() if r["detections"] else 0.0
            for r in aitex
        ]
    )
    n_det_in = np.array([len(r["detections"]) for r in val_clean], dtype=float)
    n_det_ae = np.array([len(r["detections"]) for r in aitex], dtype=float)

    # 域可分性: 用图像级统计量区分 val-clean vs AITEX (AUC)
    def auc(pos, neg):
        pos = np.sort(np.asarray(pos, float))
        neg = np.sort(np.asarray(neg, float))
        if len(pos) == 0 or len(neg) == 0:
            return float("nan")
        # Mann-Whitney U (含并列修正的近似: 用平均秩)
        allv = np.concatenate([pos, neg])
        order = np.argsort(allv, kind="mergesort")
        ranks = np.empty(len(allv))
        i = 0
        while i < len(allv):
            j = i
            while j + 1 < len(allv) and allv[order[j + 1]] == allv[order[i]]:
                j += 1
            rk = (i + j) / 2 + 1
            for k in range(i, j + 1):
                ranks[order[k]] = rk
            i = j + 1
        r_pos = ranks[: len(pos)].sum()
        u = r_pos - len(pos) * (len(pos) + 1) / 2
        return u / (len(pos) * len(neg))

    domain = {
        "n_val_clean": len(val_clean),
        "n_aitex": len(aitex),
        "per_detection": {
            "conf": {"val_clean": q(c_in), "aitex": q(c_ae)},
            "entropy_norm": {"val_clean": q(e_in), "aitex": q(e_ae)},
            "top1_top2_gap": {"val_clean": q(g_in), "aitex": q(g_ae)},
        },
        "per_image": {
            "max_conf": {"val_clean": q(img_max_in), "aitex": q(img_max_ae)},
            "n_detections": {"val_clean": q(n_det_in), "aitex": q(n_det_ae)},
        },
        "domain_auc": {
            "img_max_conf": float(auc(img_max_ae, img_max_in)),
            "img_n_det": float(auc(n_det_ae, n_det_in)),
        },
    }
    (OUT / "domain_shift.json").write_text(
        json.dumps(domain, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(
        f"  逐检测 conf: val-clean mean={c_in.mean():.4f} p90={np.quantile(c_in, 0.9):.4f} | "
        f"AITEX mean={c_ae.mean():.4f} p90={np.quantile(c_ae, 0.9):.4f}"
    )
    print(f"  逐检测归一化熵: val-clean {e_in.mean():.4f} | AITEX {e_ae.mean():.4f}")
    print(
        f"  图像级 max-conf: val-clean mean={img_max_in.mean():.4f} | AITEX mean={img_max_ae.mean():.4f}"
    )
    print(
        f"  域可分 AUC: img_max_conf={domain['domain_auc']['img_max_conf']:.4f}, "
        f"img_n_det={domain['domain_auc']['img_n_det']:.4f}"
    )

    print("\n[完成] 输出目录:", OUT)


if __name__ == "__main__":
    main()
