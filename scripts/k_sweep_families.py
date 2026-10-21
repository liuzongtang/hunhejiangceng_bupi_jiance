"""0.4b' — Determine the natural number of families: k sweep + Ward dendrogram.

For k in 2..8 on class centroids (cosine):
  - KMeans seed agreement (20 seeds)
  - bootstrap consensus stability (100 rounds, 80% box resampling)
  - silhouette (cosine, on seed-0 partition)
Plus Ward dendrogram (average linkage, cosine) saved as PNG.
Output: runs/family_clustering/k_sweep.json + dendrogram.png
"""

import json
from pathlib import Path

import matplotlib
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage

CLASSES = [
    "hole",
    "stain",
    "three_silk",
    "knot",
    "flower_board",
    "hundred_feet",
    "hair_particle",
    "coarse_warp",
    "loose_warp",
    "broken_warp",
    "hanging_warp",
    "weft_shrink",
    "size_stain",
    "warping_knot",
    "star_skip",
    "broken_spandex",
    "dense_section",
    "weave_defect",
]
OUT = Path("runs/family_clustering")


def centroids():
    d = np.load(OUT / "box_features.npz")
    feats, cls = d["feats"], d["cls"]
    cents = np.stack([feats[cls == c].mean(0) for c in range(18)])
    return cents / np.linalg.norm(cents, axis=1, keepdims=True)


def stability_for(cents, k, n_boot=100, seed0=0):
    labels = [
        KMeans(n_clusters=k, n_init=200, random_state=s).fit_predict(cents)
        for s in range(20)
    ]
    base = labels[0]
    seed_agree = float(np.mean([np.mean(l == base) for l in labels]))
    rng = np.random.default_rng(0)
    d = np.load(OUT / "box_features.npz")
    feats, cls = d["feats"], d["cls"]
    co = np.zeros((18, 18), dtype=np.int32)
    for b in range(n_boot):
        sub = []
        for c in range(18):
            idx = np.where(cls == c)[0]
            samp = rng.choice(idx, size=max(1, int(0.8 * len(idx))), replace=False)
            sub.append(feats[samp].mean(0))
        sub = np.stack(sub)
        sub = sub / np.linalg.norm(sub, axis=1, keepdims=True)
        lab = KMeans(n_clusters=k, n_init=50, random_state=b).fit_predict(sub)
        for i in range(18):
            for j in range(i + 1, 18):
                if lab[i] == lab[j]:
                    co[i, j] += 1
    cons = KMeans(n_clusters=k, n_init=200, random_state=0).fit_predict(1 - co / n_boot)
    within = [
        co[i, j] / n_boot
        for i in range(18)
        for j in range(i + 1, 18)
        if cons[i] == cons[j]
    ]
    sil = float(silhouette_score(cents, base, metric="cosine"))
    return {
        "k": k,
        "seed_agreement": round(seed_agree, 4),
        "consensus_stability": round(float(np.mean(within)) if within else 0.0, 4),
        "silhouette": round(sil, 4),
        "n_singletons": int(sum(1 for c in range(k) if (cons == c).sum() == 1)),
    }


def main():
    cents = centroids()
    rows = [stability_for(cents, k) for k in range(2, 9)]
    for r in rows:
        print(r)
    (OUT / "k_sweep.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")

    # Ward dendrogram (average linkage, cosine distance)
    Z = linkage(cents, method="average", metric="cosine")
    fig, ax = plt.subplots(figsize=(12, 6), dpi=150)
    dendrogram(Z, labels=CLASSES, ax=ax, leaf_rotation=45, color_threshold=0.55)
    ax.set_ylabel("cosine distance")
    ax.set_title(
        "Ward-style dendrogram of 18 class centroids (average linkage, cosine)"
    )
    fig.tight_layout()
    fig.savefig(OUT / "dendrogram.png")
    print(f"saved {OUT / 'dendrogram.png'}")


if __name__ == "__main__":
    main()
