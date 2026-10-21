"""
Offline replay of the inference-side seven-dimension alarm scorer.

Reads the per-query detection dump (``scripts/alarm_replay_dump.py`` output),
scores every image with ``backend.inference.alarm_scorer``, and evaluates the
image-level defect decision against the max-confidence baseline:

  * tau sweep on R_global: accuracy / TPR / FPR / precision / F1 / AUC
  * critical-class recall (images whose GT contains hole/broken_warp/broken_spandex)
  * 3-tier decision (stop / review / pass) with trigger rates
  * 5-fold CV tau selection (report mean +- std; guards against val overfit)
  * ablations: leave-one-dimension-out + weight variants

Baseline = max query confidence over the same dumped detections (identical to
``scripts/eval_binary_replay.py`` up to dump rounding).

Usage:
    python scripts/alarm_replay_eval.py \
        --detections runs/alarm_replay/detections_mosaic1280.jsonl --tag mosaic1280
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from backend.inference.alarm_scorer import NUM_CLASSES, AlarmConfig, AlarmScorer

CRITICAL_CLASS_IDS = {0, 9, 15}


def load_dump(path: Path) -> list[dict]:
    recs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            recs.append(json.loads(line))
    return recs


def score_all(
    recs: list[dict], scorer: AlarmScorer
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (R_global, max_conf, is_critical_image) arrays."""
    r_glob, max_conf, is_crit = [], [], []
    for rec in recs:
        out = scorer.score_image(rec["detections"])
        r_glob.append(out["R_global"])
        max_conf.append(max((d[4] for d in rec["detections"]), default=0.0))
        is_crit.append(any(int(b[4]) in CRITICAL_CLASS_IDS for b in rec["gt_boxes"]))
    return np.asarray(r_glob), np.asarray(max_conf), np.asarray(is_crit, dtype=bool)


def auc_rank(scores: np.ndarray, labels: np.ndarray) -> float:
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=float)
    s_sorted = scores[order]
    i = 0
    while i < len(scores):
        j = i
        while j + 1 < len(scores) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        ranks[order[i : j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    pos, neg = int(labels.sum()), int((~labels).sum())
    if pos == 0 or neg == 0:
        return float("nan")
    return float((ranks[labels].sum() - pos * (pos + 1) / 2.0) / (pos * neg))


def confusion_at(scores: np.ndarray, gt: np.ndarray, tau: float) -> dict:
    pred = scores > tau
    tp = int((pred & gt).sum())
    fp = int((pred & ~gt).sum())
    tn = int((~pred & ~gt).sum())
    fn = int((~pred & gt).sum())
    p, n = int(gt.sum()), int((~gt).sum())
    return {
        "tau": round(float(tau), 4),
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn,
        "TPR": round(tp / p, 4) if p else None,
        "FPR": round(fp / n, 4) if n else None,
        "precision": round(tp / (tp + fp), 4) if tp + fp else None,
        "accuracy": round((tp + tn) / (p + n), 4),
        "F1": round(2 * tp / (2 * tp + fp + fn), 4) if 2 * tp + fp + fn else None,
    }


def sweep(scores: np.ndarray, gt: np.ndarray, taus: np.ndarray) -> list[dict]:
    return [confusion_at(scores, gt, float(t)) for t in taus]


def cv_best_tau(
    scores: np.ndarray, gt: np.ndarray, taus: np.ndarray, k: int = 5, seed: int = 42
) -> dict:
    """5-fold CV: pick best-accuracy tau on the train folds, evaluate on the held-out fold."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(scores))
    folds = np.array_split(idx, k)
    accs, taus_picked = [], []
    for i in range(k):
        test = folds[i]
        train = np.concatenate([folds[j] for j in range(k) if j != i])
        best = max(taus, key=lambda t: ((scores[train] > t) == gt[train]).mean())
        accs.append(float(((scores[test] > best) == gt[test]).mean()))
        taus_picked.append(float(best))
    return {
        "cv_accuracy_mean": round(float(np.mean(accs)), 4),
        "cv_accuracy_std": round(float(np.std(accs)), 4),
        "cv_taus": [round(t, 4) for t in taus_picked],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--detections", required=True, help="val dump (JSONL)")
    ap.add_argument(
        "--xdomain", default=None, help="cross-domain negatives dump (JSONL)"
    )
    ap.add_argument("--tag", default="mosaic1280")
    ap.add_argument("--tau-min", type=float, default=0.02)
    ap.add_argument("--tau-max", type=float, default=0.95)
    ap.add_argument("--tau-step", type=float, default=0.02)
    ap.add_argument("--out", default="runs/alarm_replay")
    ap.add_argument("--target-fpr", type=float, default=0.06)
    ap.add_argument("--target-tpr", type=float, default=0.95)
    args = ap.parse_args()

    recs = load_dump(Path(args.detections))
    xrecs = load_dump(Path(args.xdomain)) if args.xdomain else []
    gt = np.asarray([r["gt_defective"] for r in recs], dtype=bool)
    taus = np.round(np.arange(args.tau_min, args.tau_max + 1e-9, args.tau_step), 4)
    out_dir = Path(args.out) / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------- scoring
    scorer = AlarmScorer()
    r7, max_conf, is_crit = score_all(recs, scorer)
    n_pos, n_neg = int(gt.sum()), int((~gt).sum())
    n_crit = int(is_crit.sum())
    print(
        f"[alarm:{args.tag}] images={len(recs)} defective={n_pos} clean={n_neg} "
        f"critical_images={n_crit} xdomain={len(xrecs)}"
    )

    # cross-domain negatives: R_global + max-conf on the same dumped detections
    if xrecs:
        r7_x, mc_x, _ = score_all(xrecs, scorer)
    else:
        r7_x = mc_x = np.zeros(0)

    # ------------------------------------------------------- sweeps (both)
    sweep7 = sweep(r7, gt, taus)
    sweep_mc = sweep(max_conf, gt, taus)

    def best_row(rows, key="accuracy"):
        return max(rows, key=lambda r: r[key] if r[key] is not None else -1)

    b7, bmc = best_row(sweep7), best_row(sweep_mc)

    # -------------------------------------------- matched-recall comparison
    def at_recall(scores, target):
        """Highest (most selective) tau reaching TPR >= target; (tau, TPR, FPR)."""
        for t in taus[::-1]:
            tpr = float((scores[gt] > t).mean())
            if tpr >= target:
                return (
                    round(float(t), 4),
                    round(tpr, 4),
                    round(float((scores[~gt] > t).mean()), 4),
                )
        return None, None, None

    matched = {}
    for target in (0.95, 0.90):
        t7, tpr7, fpr7 = at_recall(r7, target)
        tmc, tprmc, fprmc = at_recall(max_conf, target)
        entry = {
            "seven_dim": {"tau": t7, "TPR": tpr7, "in_domain_FPR": fpr7},
            "max_conf": {"tau": tmc, "TPR": tprmc, "in_domain_FPR": fprmc},
        }
        if xrecs:
            entry["seven_dim"]["xdomain_FPR"] = (
                round(float((r7_x > t7).mean()), 4) if t7 is not None else None
            )
            entry["max_conf"]["xdomain_FPR"] = (
                round(float((mc_x > tmc).mean()), 4) if tmc is not None else None
            )
        matched[f"recall_{target}"] = entry

    # ------------------------------------------------- 3-tier decision rule
    # tau_stop: lowest tau on R_global whose in-domain FPR <= target_fpr
    tau_stop = next((t for t in taus if (r7 > t)[~gt].mean() <= args.target_fpr), None)
    if tau_stop is None:
        tau_stop = b7["tau"]
    # tau_review: lowest tau on R_global reaching target_tpr
    tau_review = next((t for t in taus if (r7 > t)[gt].mean() >= args.target_tpr), None)
    if tau_review is None:
        tau_review = float(taus[0])
    tier_stop = r7 > tau_stop
    tier_review = (r7 > tau_review) & ~tier_stop
    tier_pass = ~tier_review & ~tier_stop
    tier = {
        "tau_stop": round(float(tau_stop), 4),
        "tau_review": round(float(tau_review), 4),
        "stop_rate": round(float(tier_stop.mean()), 4),
        "review_rate": round(float(tier_review.mean()), 4),
        "pass_rate": round(float(tier_pass.mean()), 4),
        "critical_recall_at_tau_stop": round(float(tier_stop[is_crit].mean()), 4),
        "critical_missed_at_tau_stop": int((~tier_stop & is_crit).sum()),
        "critical_recall_at_stop_or_review": round(
            float((tier_stop | tier_review)[is_crit].mean()), 4
        ),
        "confusion_at_tau_stop": confusion_at(r7, gt, float(tau_stop)),
    }

    # ------------------------------------------------------------- 5-fold CV
    cv7 = cv_best_tau(r7, gt, taus)
    cv_mc = cv_best_tau(max_conf, gt, taus)

    # --------------------------------------------------- per-dim separation
    dim_means = {}
    sample_def, sample_clean = None, None
    for rec, g in zip(recs, gt, strict=False):
        if g and sample_def is None:
            sample_def = scorer.score_image(rec["detections"])["per_detection"]
        if not g and sample_clean is None:
            sample_clean = scorer.score_image(rec["detections"])["per_detection"]

    # aggregate over first 200 images per class for speed
    def agg(flag):
        vals = {d: [] for d in ("D1", "D2", "D3", "D4", "D5", "D7")}
        cnt = 0
        for rec, g in zip(recs, gt, strict=False):
            if g != flag or cnt >= 200:
                continue
            cnt += 1
            for p in scorer.score_image(rec["detections"])["per_detection"]:
                for d in vals:
                    vals[d].append(p[d])
        return {d: round(float(np.mean(v)), 4) for d, v in vals.items()}

    dim_means = {"defective": agg(True), "clean": agg(False)}

    # ------------------------------------------------------- ablations (1.4)
    ablations = []

    def eval_variant(name: str, cfg: AlarmConfig):
        s = AlarmScorer(cfg)
        r, _, _ = score_all(recs, s)
        rows = sweep(r, gt, taus)
        b = best_row(rows)
        cv = cv_best_tau(r, gt, taus)
        crit_r = float((r[is_crit] > b["tau"]).mean())
        entry = {
            "variant": name,
            "best_tau": b["tau"],
            "accuracy": b["accuracy"],
            "TPR": b["TPR"],
            "FPR": b["FPR"],
            "precision": b["precision"],
            "F1": b["F1"],
            "critical_recall": round(crit_r, 4),
            "cv_acc": cv["cv_accuracy_mean"],
            "cv_std": cv["cv_accuracy_std"],
            "AUC": round(auc_rank(r, gt), 4),
        }
        if xrecs:
            r_x, _, _ = score_all(xrecs, s)
            entry["xdomain_FPR_at_best_tau"] = round(float((r_x > b["tau"]).mean()), 4)
        return entry

    QW = ("D2", "D3", "D4", "D5", "D7")  # quality weights in product mode

    def w(**over):
        base = dict.fromkeys(("D1", "D2", "D3", "D4", "D5", "D6", "D7"), 1.0)
        base.update(over)
        return base

    ablations.append(eval_variant("equal (product, noisy-or)", AlarmConfig()))
    ablations.append(
        eval_variant("equal (product, max-agg)", AlarmConfig(aggregation="max"))
    )
    ablations.append(eval_variant("conf-heavy (D2x3)", AlarmConfig(weights=w(D2=3.0))))
    ablations.append(
        eval_variant("expert (D2x2 D5x0.5)", AlarmConfig(weights=w(D2=2.0, D5=0.5)))
    )
    for d in QW:
        ablations.append(eval_variant(f"drop {d}", AlarmConfig(weights=w(**{d: 0.0}))))
    ablations.append(
        eval_variant(
            "drop D1 (no severity gate)",
            AlarmConfig(severity=dict.fromkeys(range(NUM_CLASSES), 1.0)),
        )
    )
    ablations.append(
        eval_variant(
            "drop D3+D4 (conf+size+cluster)", AlarmConfig(weights=w(D3=0.0, D4=0.0))
        )
    )
    ablations.append(
        eval_variant("LINEAR (plan literal form)", AlarmConfig(combination="linear"))
    )
    # max-conf baseline row
    mc_best = bmc
    mc_row = {
        "variant": "BASELINE max-conf",
        "best_tau": mc_best["tau"],
        "accuracy": mc_best["accuracy"],
        "TPR": mc_best["TPR"],
        "FPR": mc_best["FPR"],
        "precision": mc_best["precision"],
        "F1": mc_best["F1"],
        "critical_recall": round(float((max_conf[is_crit] > mc_best["tau"]).mean()), 4),
        "cv_acc": cv_mc["cv_accuracy_mean"],
        "cv_std": cv_mc["cv_accuracy_std"],
        "AUC": round(auc_rank(max_conf, gt), 4),
    }
    if xrecs:
        mc_row["xdomain_FPR_at_best_tau"] = round(
            float((mc_x > mc_best["tau"]).mean()), 4
        )
    ablations.append(mc_row)

    # ---------------------------------------------------------------- report
    summary = {
        "tag": args.tag,
        "n_images": len(recs),
        "n_defective": n_pos,
        "n_clean": n_neg,
        "n_critical_images": n_crit,
        "seven_dim": {
            "AUC": round(auc_rank(r7, gt), 4),
            "best_accuracy": b7,
            "critical_recall_at_best_tau": round(
                float((r7[is_crit] > b7["tau"]).mean()), 4
            ),
            "cv": cv7,
        },
        "max_conf_baseline": {
            "AUC": round(auc_rank(max_conf, gt), 4),
            "best_accuracy": bmc,
            "critical_recall_at_best_tau": round(
                float((max_conf[is_crit] > bmc["tau"]).mean()), 4
            ),
            "cv": cv_mc,
        },
        "three_tier": tier,
        "matched_recall": matched,
        "dimension_means": dim_means,
        "ablations": ablations,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    for name, rows in (
        ("sweep_seven_dim.csv", sweep7),
        ("sweep_max_conf.csv", sweep_mc),
    ):
        with (out_dir / name).open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    np.save(out_dir / "R_global.npy", r7)
    np.save(out_dir / "max_conf.npy", max_conf)

    # console summary
    print(f"\n[alarm:{args.tag}] ===== 7-dim alarm score vs max-conf baseline =====")
    print(
        f"seven-dim : AUC={summary['seven_dim']['AUC']}  best acc={b7['accuracy']} @tau={b7['tau']} "
        f"(TPR={b7['TPR']} FPR={b7['FPR']} crit_recall={summary['seven_dim']['critical_recall_at_best_tau']}) "
        f"CV={cv7['cv_accuracy_mean']}±{cv7['cv_accuracy_std']}"
    )
    print(
        f"max-conf  : AUC={summary['max_conf_baseline']['AUC']}  best acc={bmc['accuracy']} @tau={bmc['tau']} "
        f"(TPR={bmc['TPR']} FPR={bmc['FPR']} crit_recall={summary['max_conf_baseline']['critical_recall_at_best_tau']}) "
        f"CV={cv_mc['cv_accuracy_mean']}±{cv_mc['cv_accuracy_std']}"
    )
    print(
        f"3-tier    : tau_stop={tier['tau_stop']} tau_review={tier['tau_review']} "
        f"stop={tier['stop_rate']} review={tier['review_rate']} pass={tier['pass_rate']} "
        f"crit_recall@stop={tier['critical_recall_at_tau_stop']} (missed {tier['critical_missed_at_tau_stop']})"
    )
    for key, m in matched.items():
        s7, smc = m["seven_dim"], m["max_conf"]
        line = (
            f"{key:14s} seven-dim: TPR={s7['TPR']} FPR={s7['in_domain_FPR']} @tau={s7['tau']}"
            f"  |  max-conf: TPR={smc['TPR']} FPR={smc['in_domain_FPR']} @tau={smc['tau']}"
        )
        if "xdomain_FPR" in s7:
            line += f"  ||  xdomain FPR: 7dim={s7['xdomain_FPR']} vs max-conf={smc['xdomain_FPR']}"
        print(f"matched    : {line}")
    print(f"\n[alarm:{args.tag}] ablations:")
    for a in ablations:
        xd = (
            f" xFPR={a['xdomain_FPR_at_best_tau']}"
            if "xdomain_FPR_at_best_tau" in a
            else ""
        )
        print(
            f"  {a['variant']:34s} acc={a['accuracy']} TPR={a['TPR']} FPR={a['FPR']} "
            f"crit={a['critical_recall']} CV={a['cv_acc']}±{a['cv_std']}{xd}"
        )
    print(f"[alarm:{args.tag}] wrote {out_dir}/summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
