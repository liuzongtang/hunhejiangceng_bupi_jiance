"""
阶段1.1 (B2) 分布排序七维评分 — Dr. Seg (arXiv:2603.00152) Distribution-Ranked Reward 思想迁移
预承诺门 (计划文档 §六, 实验前登记, 事后不移动):
  正: 域内最佳正确率 >= 0.9373 (A10 口径) 且不劣于 max-conf 基线 0.9383 超过 -0.005
  中性: 0.925 ~ 0.9373; 负: < 0.925

设计 (概念迁移边界: Dr. Seg 面向 VLLM 推理分割的滚动历史队列分位数奖励;
我们是封闭集 RT-DETR 的离线回放, 用类条件经验 CDF 近似):
  q_conf  = F_c(conf)   同类检测置信度的经验分位数 (类内尺度不变)
  q_margin= F_gap(top1-top2) 同类检测 top1-top2 间隔的经验分位数
  R = D1(严重度门控) x Q, Q 为 q_* 与 D7(空间聚集) 的加权均值
  noisy-or 聚合 raw conf >= 0.25 的证据检测 (与主变体同规则)
公平性: 分位数变换用 2 折交叉拟合 (变换折的 CDF 只在另一半上估计), 避免同源乐观;
同时报告 in-sample 版本作参照。
"""

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
REPLAY = ROOT / "runs" / "alarm_replay"
OUT = REPLAY / "stage1" / "rank"
OUT.mkdir(parents=True, exist_ok=True)

from backend.inference.alarm_scorer import AlarmScorer

GATE_POS, GATE_NEUTRAL_HI, GATE_NEUTRAL_LO = 0.9373, 0.9373, 0.925
BASELINE_MAXCONF_ACC = 0.9383  # REPORT.md 主结果 (max-conf 域内最佳正确率)


def load_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")]


def det_array(dets):
    if not dets:
        return np.zeros((0, 24))
    return np.asarray(dets, dtype=np.float64)


def empirical_cdf(values, query):
    """values: 估计用样本; query: 待变换值 -> [0,1] 分位数 (平均秩/中点法)"""
    v = np.sort(np.asarray(values, dtype=float))
    q = np.asarray(query, dtype=float)
    if len(v) == 0:
        return np.full_like(q, 0.5)
    lo = np.searchsorted(v, q, side="left")
    hi = np.searchsorted(v, q, side="right")
    return ((lo + hi) / 2.0) / len(v)


def main():
    val = load_jsonl(REPLAY / "detections_mosaic1280.jsonl")
    scorer = AlarmScorer()
    n = len(val)
    y = np.array([bool(r["gt_defective"]) for r in val])

    # ---- 每检测原始量 ----
    dets = [det_array(r["detections"]) for r in val]
    conf = np.array([d[:, 4] if len(d) else np.zeros(0) for d in dets], dtype=object)
    cls = np.array(
        [d[:, 5].astype(int) if len(d) else np.zeros(0, int) for d in dets],
        dtype=object,
    )
    scores = np.array(
        [d[:, 6:24] if len(d) else np.zeros((0, 18)) for d in dets], dtype=object
    )
    margin = np.array(
        [
            (np.sort(s, axis=1)[:, ::-1][:, 0] - np.sort(s, axis=1)[:, ::-1][:, 1])
            if len(s)
            else np.zeros(0)
            for s in scores
        ],
        dtype=object,
    )

    # D1 严重度 + D7 聚集 (与 alarm_scorer 主变体同实现)
    d1 = np.array(
        [
            [scorer.d1_severity(int(c)) for c in cs] if len(cs) else np.zeros(0)
            for cs in cls
        ],
        dtype=object,
    )
    d7 = np.array(
        [
            [
                scorer.d7_clustering(r["detections"], i)
                for i in range(len(r["detections"]))
            ]
            for r in val
        ],
        dtype=object,
    )

    # ---- 扁平化 + 2 折交叉拟合的分位数变换 ----
    rng = np.random.RandomState(42)
    fold = rng.permutation(n) % 2
    keep_img = [i for i in range(n) if len(dets[i])]  # 有检测的图
    cls_flat = np.concatenate([dets[i][:, 5].astype(int) for i in keep_img])
    conf_flat = np.concatenate([dets[i][:, 4] for i in keep_img])
    margin_flat = np.concatenate(
        [
            (
                np.sort(dets[i][:, 6:24], axis=1)[:, ::-1][:, 0]
                - np.sort(dets[i][:, 6:24], axis=1)[:, ::-1][:, 1]
            )
            for i in keep_img
        ]
    )
    fold_flat = np.concatenate([np.full(len(dets[i]), fold[i]) for i in keep_img])
    assert len(cls_flat) == len(conf_flat) == len(margin_flat) == len(fold_flat)
    print(f"[扁平检测数] {len(cls_flat)} (来自 {len(keep_img)}/{n} 张图)")

    def cdf_transform(values, crossfit=True):
        """类条件经验分位数; crossfit=True 时每折的 CDF 只在另一折上估计"""
        out = np.full(len(values), 0.5)
        if crossfit:
            masks = [fold_flat == f for f in (0, 1)]
        else:
            masks = [np.ones(len(values), bool), None]  # in-sample: 全量估计
        for m_q in masks:
            if m_q is None:
                continue
            m_e = ~m_q
            for c in np.unique(cls_flat[m_q]):
                mq = m_q & (cls_flat == c)
                me = m_e & (cls_flat == c)
                if me.sum() == 0:
                    continue
                out[mq] = empirical_cdf(values[me], values[mq])
        if not crossfit:
            for c in np.unique(cls_flat):
                m = cls_flat == c
                out[m] = empirical_cdf(values[m], values[m])
        return out

    q_conf_xf = cdf_transform(conf_flat, crossfit=True)
    q_margin_xf = cdf_transform(margin_flat, crossfit=True)
    q_conf_is = cdf_transform(conf_flat, crossfit=False)
    q_margin_is = cdf_transform(margin_flat, crossfit=False)

    # ---- 拆回每图结构并聚合 ----
    def unflatten(flat):
        out, pos = [], 0
        for d in dets:
            k = len(d)
            out.append(flat[pos : pos + k])
            pos += k
        return out

    def noisy_or(vals):
        v = np.asarray(vals, dtype=float)
        if len(v) == 0:
            return 0.0
        return float(1.0 - np.prod(1.0 - v))

    variants = {
        "RANK-A": lambda qc, qm, d1v, d7v: d1v * qc,
        "RANK-B": lambda qc, qm, d1v, d7v: d1v * (0.5 * qc + 0.5 * qm),
        "RANK-C": lambda qc, qm, d1v, d7v: d1v * (0.5 * qc + 0.25 * qm + 0.25 * d7v),
    }

    def build_R(qc_flat, qm_flat):
        qc, qm = unflatten(qc_flat), unflatten(qm_flat)
        out = {}
        for name, fn in variants.items():
            per_det, per_img = [], []
            for i in range(n):
                keep = conf[i] >= 0.25  # 证据检测规则 (与主变体一致)
                if keep.sum() == 0:
                    per_img.append(0.0)
                    continue
                vals = fn(qc[i][keep], qm[i][keep], d1[i][keep], d7[i][keep])
                per_det.append(vals)
                per_img.append(noisy_or(vals))
            out[name] = np.array(per_img)
        return out

    def evaluate(R, tag):
        rows = [
            f"variant,{tag}_tau,{tag}_TP,{tag}_FP,{tag}_TN,{tag}_FN,{tag}_acc,{tag}_crit_recall"
        ]
        res = {}
        for name, r in R.items():
            best = None
            for t in np.arange(0.001, 0.801, 0.001):
                pred = r >= t
                tp = int((y & pred).sum())
                fp = int(((~y) & pred).sum())
                tn = int((~y & ~pred).sum())
                fn = int((y & ~pred).sum())
                acc = (tp + tn) / n
                if best is None or acc > best[0]:
                    best = (acc, t, tp, fp, tn, fn)
            acc, t, tp, fp, tn, fn = best
            # 关键类召回 (图级, 关键类 GT 且被预测为有瑕)
            crit = np.array(
                [any(int(g[0]) in (0, 9, 15) for g in r_["gt_boxes"]) for r_ in val]
            )
            cr = float(((y & crit) & (r >= t)).sum() / max(crit.sum(), 1))
            rows.append(f"{name},{t:.3f},{tp},{fp},{tn},{fn},{acc:.4f},{cr:.4f}")
            res[name] = {"tau": float(t), "acc": float(acc), "crit_recall": cr}
            print(
                f"  [{tag}] {name}: 最佳正确率 {acc:.4f} (τ={t:.3f}) 关键类召回 {cr:.4f}"
            )
        (OUT / f"rank_sweep_{tag}.csv").write_text(
            "\n".join(rows) + "\n", encoding="utf-8"
        )
        return res

    print("===== 1.1 (B2) 分布排序七维评分 =====")
    print(f"门: 正 >= {GATE_POS} (A10 口径) | max-conf 基线 {BASELINE_MAXCONF_ACC}")
    R_xf = build_R(q_conf_xf, q_margin_xf)
    R_is = build_R(q_conf_is, q_margin_is)
    res_xf = evaluate(R_xf, "crossfit")
    res_is = evaluate(R_is, "insample")

    # ---- 判定 ----
    verdict = {}
    for name in variants:
        a = res_xf[name]["acc"]
        if a >= GATE_POS:
            v = "POS"
        elif a >= GATE_NEUTRAL_LO:
            v = "NEUTRAL"
        else:
            v = "NEG"
        verdict[name] = {
            "crossfit_acc": a,
            "insample_acc": res_is[name]["acc"],
            "verdict": v,
        }
        print(
            f"  判定 {name}: {v} (crossfit {a:.4f} / insample {res_is[name]['acc']:.4f})"
        )
    summary = {
        "gate": {
            "pos": GATE_POS,
            "neutral": [GATE_NEUTRAL_LO, GATE_NEUTRAL_HI],
            "baseline_maxconf_acc": BASELINE_MAXCONF_ACC,
        },
        "concept_source": "Dr. Seg arXiv:2603.00152 Distribution-Ranked Reward (VLLM 推理分割; 概念迁移)",
        "design": "R = D1 x Q; q_conf/q_margin = 类条件经验分位数 (2折交叉拟合); noisy-or 聚合 conf>=0.25",
        "variants": verdict,
    }
    (OUT / "rank_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("[完成]", OUT)


if __name__ == "__main__":
    main()
