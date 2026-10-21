"""0.4b — Cluster 18 defect classes into 6 data-driven families.

Pipeline:
  1. per-class centroid = mean of L2-normalized box features (re-normalized)
  2. KMeans k=6 (n_init=200, 20 seeds) + Ward hierarchical (cosine) as cross-check
  3. bootstrap stability: 200 rounds of 80% box resampling -> recluster centroids,
     co-assignment matrix -> consensus partition + stability score
  4. sensitivity: same clustering on features concatenated with standardized
     box meta (area ratio, aspect, abs size) — does the partition change?
  5. silhouette on class centroids; t-SNE scatter of centroids colored by cluster

Output: runs/family_clustering/families.json + families.yaml + tsne.png + report.md (stub filled by 0.4c)
"""

import json
from pathlib import Path

import matplotlib
import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
K = 6
OUT = Path("runs/family_clustering")


def class_centroids(feats, cls):
    cents, sizes = [], []
    for c in range(18):
        f = feats[cls == c]
        cents.append(f.mean(0))
        sizes.append(len(f))
    cents = np.stack(cents)
    return cents / np.linalg.norm(cents, axis=1, keepdims=True), np.array(sizes)


def kmeans_labels(cents, k=K, seed=0):
    return KMeans(n_clusters=k, n_init=200, random_state=seed).fit_predict(cents)


def main():
    data = np.load(OUT / "box_features.npz")
    feats, cls, meta = data["feats"], data["cls"], data["meta"]
    cents, sizes = class_centroids(feats, cls)

    # ---- KMeans across seeds: check seed stability ----
    seed_labels = [kmeans_labels(cents, seed=s) for s in range(20)]
    base = seed_labels[0]
    seed_agree = np.mean([np.mean(s == base) for s in seed_labels])
    print(f"KMeans seed agreement: {seed_agree:.3f}")

    # ---- Ward hierarchical (cosine) ----
    ward = AgglomerativeClustering(
        n_clusters=K, metric="cosine", linkage="average"
    ).fit_predict(cents)

    # ---- bootstrap stability ----
    rng = np.random.default_rng(0)
    n_boot = 200
    coassign = np.zeros((18, 18), dtype=np.int32)
    boot_partitions = []
    for b in range(n_boot):
        sub_cents = []
        for c in range(18):
            idx = np.where(cls == c)[0]
            samp = rng.choice(idx, size=max(1, int(0.8 * len(idx))), replace=False)
            sub_cents.append(feats[samp].mean(0))
        sub_cents = np.stack(sub_cents)
        sub_cents = sub_cents / np.linalg.norm(sub_cents, axis=1, keepdims=True)
        lab = kmeans_labels(sub_cents, seed=b)
        boot_partitions.append(lab)
        for i in range(18):
            for j in range(i + 1, 18):
                if lab[i] == lab[j]:
                    coassign[i, j] += 1
    coassign_rate = coassign / n_boot
    # consensus: average pairwise co-assignment, then final KMeans on the co-assignment profile
    consensus = KMeans(n_clusters=K, n_init=200, random_state=0).fit_predict(
        1 - coassign_rate
    )
    # stability score: mean co-assignment rate within consensus clusters
    stab = []
    for i in range(18):
        for j in range(i + 1, 18):
            if consensus[i] == consensus[j]:
                stab.append(coassign_rate[i, j])
    stability = float(np.mean(stab)) if stab else 0.0
    print(
        f"consensus partition stability (within-cluster co-assignment): {stability:.3f}"
    )

    # ---- silhouette ----
    sil_km = silhouette_score(cents, base, metric="cosine")
    sil_cons = silhouette_score(cents, consensus, metric="cosine")
    print(
        f"silhouette (KMeans seed0): {sil_km:.3f} | silhouette (consensus): {sil_cons:.3f}"
    )

    # ---- sensitivity: + standardized meta features ----
    from sklearn.preprocessing import StandardScaler

    meta_z = StandardScaler().fit_transform(meta)
    cents_meta = []
    for c in range(18):
        cents_meta.append(meta_z[cls == c].mean(0))
    cents_meta = np.array(cents_meta)
    combined = np.concatenate([cents, cents_meta], axis=1)
    km_comb = kmeans_labels(combined)
    agree_comb = np.mean(
        [
            adjusted_rand := __import__(
                "sklearn.metrics", fromlist=["adjusted_rand_score"]
            ).adjusted_rand_score(base, km_comb)
        ]
    )
    print(f"ARI (visual-only vs visual+meta): {agree_comb:.3f}")

    # ---- t-SNE scatter ----
    emb = TSNE(
        n_components=2, perplexity=5, random_state=0, metric="cosine"
    ).fit_transform(cents)
    fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
    scatter = ax.scatter(emb[:, 0], emb[:, 1], c=consensus, cmap="tab10", s=120)
    for i, name in enumerate(CLASSES):
        ax.annotate(
            f"{i}:{name}",
            (emb[i, 0], emb[i, 1]),
            fontsize=8,
            xytext=(4, 4),
            textcoords="offset points",
        )
    ax.set_title(
        "Class centroids (ResNet50 GT-box features), colored by consensus family"
    )
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(OUT / "tsne_families.png")
    print(f"saved {OUT / 'tsne_families.png'}")

    # ---- dump ----
    result = {
        "k": K,
        "encoder": "resnet50 ImageNet V2, 224px stretch crop, pad 15%",
        "n_boxes": len(cls),
        "box_counts": {CLASSES[c]: int(sizes[c]) for c in range(18)},
        "seed_agreement": round(float(seed_agree), 4),
        "stability": round(stability, 4),
        "silhouette_kmeans": round(float(sil_km), 4),
        "silhouette_consensus": round(float(sil_cons), 4),
        "ari_visual_vs_meta": round(float(agree_comb), 4),
        "kmeans_seed0": base.tolist(),
        "ward": ward.tolist(),
        "consensus": consensus.tolist(),
        "coassign_rate": coassign_rate.tolist(),
    }
    (OUT / "families.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    print("\n=== consensus families (visual-only) ===")
    for f in range(K):
        members = [f"{i}:{CLASSES[i]}" for i in range(18) if consensus[i] == f]
        print(f"family {f}: {', '.join(members)}")
    print("\n=== KMeans seed0 (for reference) ===")
    for f in range(K):
        members = [f"{i}:{CLASSES[i]}" for i in range(18) if base[i] == f]
        print(f"cluster {f}: {', '.join(members)}")
    print("\n=== Ward ===")
    for f in range(K):
        members = [f"{i}:{CLASSES[i]}" for i in range(18) if ward[i] == f]
        print(f"cluster {f}: {', '.join(members)}")


if __name__ == "__main__":
    main()
