"""
Inference-side seven-dimension alarm scorer (B-gate decision layer).

This is the *inference-side* counterpart of the training-side seven-dimension
reward/punishment framework (``backend/training/dimension_rewards.py``). The two
share the methodology (asymmetric industrial costs decomposed into interpretable
dimensions) but NOT the dimension definitions: the training side needed
gradients (soft IoU / softmax mass on GT), the inference side needs calibrated
decision scores from the detector's own outputs.

Dimensions (all in [0, 1], higher = more alarming):

  D1 severity      — class severity weight (critical / secondary / minor)
  D2 confidence    — within-similar-group softmax mass on the top-1 class
  D3 uncertainty   — 1 - (q1 - q2) / q1 over the global softmax (margin)
  D4 group conflict— strongest runner-up among the top-1 class' confusion
                    partners, relative to the top-1 (within-group softmax)
  D5 target size   — bbox area / image area, linear ramp saturating at size_sat
  D6 frame stability— cross-frame recurrence (single-image scenarios: disabled,
                    weight redistributed to the other six)
  D7 clustering    — neighbouring detections (conf >= cluster_min_conf) within
                    cluster_radius, saturating at cluster_sat

Image score (noisy-or over detections):  R_global = 1 - prod(1 - R_i).

Provenance of the similar-group table (``SIMILAR_PARTNERS``): undirected
confusion pairs with count >= 5 from the 18-class baseline val confusion matrix
(conf>0.25, IoU>0.45; ``runs/family_clustering/confusion_top_pairs.json``).
Severity tiers follow the training-side taxonomy: critical = hole /
broken_warp / broken_spandex, secondary = star_skip / weave_defect.

Pure numpy — no torch — so it can score dumped replay data offline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, FrozenSet, Iterable, Sequence, Tuple

import numpy as np

# 18-class ids (backend.schemas.defect.TIANCHI_CLASS_NAMES):
# 0 hole, 1 stain, 2 three_silk, 3 knot, 4 flower_board, 5 hundred_feet,
# 6 hair_particle, 7 coarse_warp, 8 loose_warp, 9 broken_warp, 10 hanging_warp,
# 11 weft_shrink, 12 size_stain, 13 warping_knot, 14 star_skip,
# 15 broken_spandex, 16 dense_section, 17 weave_defect

NUM_CLASSES = 18

CRITICAL_CLASS_IDS = (0, 9, 15)  # hole / broken_warp / broken_spandex
SECONDARY_CLASS_IDS = (14, 17)  # star_skip / weave_defect

# Empirical confusion partners (see module docstring for provenance).
_SIMILAR_PAIRS: Tuple[Tuple[int, int], ...] = (
    (2, 3),  # three_silk <-> knot        (66)
    (3, 13),  # knot <-> warping_knot      (26)
    (8, 9),  # loose_warp <-> broken_warp (19)
    (14, 17),  # star_skip <-> weave_defect (19)
    (9, 17),  # broken_warp <-> weave_defect (15)
    (9, 10),  # broken_warp <-> hanging_warp (13)
    (3, 11),  # knot <-> weft_shrink       (11)
    (7, 17),  # coarse_warp <-> weave_defect (5)
)

SIMILAR_PARTNERS: Dict[int, FrozenSet[int]] = {
    c: frozenset() for c in range(NUM_CLASSES)
}
for _a, _b in _SIMILAR_PAIRS:
    SIMILAR_PARTNERS[_a] = SIMILAR_PARTNERS[_a] | {_b}
    SIMILAR_PARTNERS[_b] = SIMILAR_PARTNERS[_b] | {_a}

DIMENSION_NAMES = ("D1", "D2", "D3", "D4", "D5", "D6", "D7")


@dataclass
class AlarmConfig:
    """Weights and thresholds for the alarm scorer.

    Two combination rules (``combination``):

    * ``"linear"``  — the plan's literal form R = sum(w_d * D_d). Empirically
      broken: every detection floors at ~0.2 (D1 >= 0.3, D3/D4 >= 0), so the
      noisy-or over the evidence detections saturates and R_global degenerates
      to "has evidence or not" — weights have no effect (AUC 0.93, best
      accuracy identical across all weight variants).
    * ``"product"`` (default) — R = D1 * Q, where Q is the weighted mean of the
      confidence-quality dimensions {D2, 1-D3, 1-D4, D5, D7}. Severity gates a
      detection-quality score, which restores resolution (R spans ~0.05-0.8).

    ``weights`` need not be normalized; the scorer normalizes over the active
    members of the chosen rule (sum to 1). ``D6`` is disabled by default
    (single-image replay has no cross-frame information).
    """

    severity: Dict[int, float] = field(
        default_factory=lambda: {
            **dict.fromkeys(CRITICAL_CLASS_IDS, 1.0),
            **dict.fromkeys(SECONDARY_CLASS_IDS, 0.6),
            **{
                c: 0.3
                for c in range(NUM_CLASSES)
                if c not in CRITICAL_CLASS_IDS + SECONDARY_CLASS_IDS
            },
        }
    )
    weights: Dict[str, float] = field(
        default_factory=lambda: dict.fromkeys(DIMENSION_NAMES, 1.0)
    )
    combination: str = "product"  # "product" | "linear"
    aggregation: str = "noisy_or"  # "noisy_or" | "max" (over evidence detections)
    disabled_dims: FrozenSet[str] = frozenset({"D6"})
    size_sat: float = 0.05  # bbox area fraction at which D5 saturates
    cluster_radius: float = 0.15  # normalized distance for D7 neighbourhood
    cluster_min_conf: float = 0.25  # only confident detections seed neighbourhoods
    cluster_sat: int = 5  # neighbours at which D7 saturates
    evidence_conf: float = 0.25  # detections below this conf carry no alarm
    # evidence and are excluded from the noisy-or
    # (same evidence semantics as a max-conf gate)

    def normalized_weights(self) -> Dict[str, float]:
        if self.combination == "linear":
            active = [d for d in DIMENSION_NAMES if d not in self.disabled_dims]
        else:  # product: quality members are D2, D3, D4, D5, D7 (D1 is the gate)
            active = [
                d for d in ("D2", "D3", "D4", "D5", "D7") if d not in self.disabled_dims
            ]
        total = sum(self.weights.get(d, 0.0) for d in active)
        if total <= 0:
            n = max(len(active), 1)
            return {d: (1.0 / n if d in active else 0.0) for d in DIMENSION_NAMES}
        return {
            d: (self.weights.get(d, 0.0) / total if d in active else 0.0)
            for d in DIMENSION_NAMES
        }


class AlarmScorer:
    """Score dumped detections (or live model outputs) with the 7-D alarm score."""

    def __init__(self, config: AlarmConfig | None = None):
        self.cfg = config or AlarmConfig()
        if self.cfg.combination not in ("product", "linear"):
            raise ValueError(f"unknown combination {self.cfg.combination!r}")
        self.w = self.cfg.normalized_weights()

    # ------------------------------------------------------------------ D1
    def d1_severity(self, cls: int) -> float:
        return float(self.cfg.severity.get(int(cls), 0.3))

    # ------------------------------------------------------------- D2 (conf)
    def d2_confidence(self, det: Sequence[float]) -> float:
        """Top-1 class score (the plan's first D2 option: ``top1_score``).

        Within-group softmax normalization is reserved for D4 (group
        arbitration); the raw score is the standard confidence signal here.
        """
        return float(min(1.0, max(0.0, det[4])))

    # ------------------------------------------------------------------ D3
    @staticmethod
    def d3_uncertainty(scores: np.ndarray) -> float:
        z = scores - scores.max()
        e = np.exp(z)
        q = e / e.sum()
        top2 = np.sort(q)[-2:]
        q1, q2 = float(top2[1]), float(top2[0])
        if q1 <= 0:
            return 1.0
        return float(1.0 - (q1 - q2) / q1)

    # ------------------------------------------------------------- D4 (group)
    @staticmethod
    def _group_softmax(scores: np.ndarray, members: Sequence[int]) -> np.ndarray:
        """Softmax restricted to ``members`` (indices into ``scores``)."""
        z = scores[list(members)]
        z = z - z.max()
        e = np.exp(z)
        return e / e.sum()

    def d4_group_conflict(self, scores: np.ndarray, cls: int) -> float:
        partners = SIMILAR_PARTNERS.get(int(cls), frozenset())
        if not partners:
            return 0.0
        members = [int(cls), *sorted(partners)]
        q = self._group_softmax(scores, members)
        return float(q[1:].max() / q[0]) if q[0] > 0 else 0.0

    # ------------------------------------------------------------- D5 / D7
    def d5_size(self, w: float, h: float) -> float:
        area = max(0.0, float(w)) * max(0.0, float(h))
        return float(min(1.0, area / self.cfg.size_sat))

    def d7_clustering(self, detections: Sequence[Sequence[float]], idx: int) -> float:
        cx, cy = float(detections[idx][0]), float(detections[idx][1])
        r = self.cfg.cluster_radius
        n = 0
        for j, d in enumerate(detections):
            if j == idx or float(d[4]) < self.cfg.cluster_min_conf:
                continue
            if (float(d[0]) - cx) ** 2 + (float(d[1]) - cy) ** 2 <= r * r:
                n += 1
        return float(min(1.0, n / self.cfg.cluster_sat))

    # ----------------------------------------------------------- full score
    def score_detection(
        self, detections: Sequence[Sequence[float]], idx: int
    ) -> Dict[str, float]:
        """Per-detection dimension scores + combined R for one detection.

        ``detections[idx]`` layout: [cx, cy, w, h, conf, cls, s0, ..., s17].
        """
        det = detections[idx]
        scores = np.asarray(det[6:], dtype=float)
        if scores.size != NUM_CLASSES:
            raise ValueError(f"expected {NUM_CLASSES} class scores, got {scores.size}")
        cls = int(det[5])
        dims = {
            "D1": self.d1_severity(cls),
            "D2": self.d2_confidence(det),
            "D3": self.d3_uncertainty(scores),
            "D4": self.d4_group_conflict(scores, cls),
            "D5": self.d5_size(det[2], det[3]),
            "D6": 0.0,  # disabled in single-image replay
            "D7": self.d7_clustering(detections, idx),
        }
        if self.cfg.combination == "linear":
            r = sum(self.w[d] * dims[d] for d in DIMENSION_NAMES)
        else:  # product: severity gates the confidence-quality mean
            quality_terms = {
                "D2": dims["D2"],
                "D3": 1.0 - dims["D3"],  # certainty, not uncertainty
                "D4": 1.0 - dims["D4"],  # group unambiguity
                "D5": dims["D5"],
                "D7": dims["D7"],
            }
            q = sum(self.w[d] * quality_terms[d] for d in quality_terms)
            r = dims["D1"] * q
            dims["Q"] = float(q)
        dims["R"] = float(r)
        return dims

    def score_image(self, detections: Sequence[Sequence[float]]) -> Dict[str, object]:
        """Score all detections of one image; noisy-or into the image score.

        Only detections with conf >= ``evidence_conf`` contribute evidence:
        without this gate the noisy-or saturates (junk queries have flat,
        high-uncertainty score vectors that still score mid-range on every
        dimension, so 30 queries would drive R_global to ~1 on every image).
        """
        dets = [d for d in detections if float(d[4]) >= self.cfg.evidence_conf]
        if not dets:
            return {"R_global": 0.0, "per_detection": [], "n_scored": 0}
        per = [self.score_detection(dets, i) for i in range(len(dets))]
        if self.cfg.aggregation == "max":
            r_global = max(p["R"] for p in per)
        else:  # noisy_or
            survive = 1.0
            for p in per:
                survive *= 1.0 - p["R"]
            r_global = 1.0 - survive
        return {
            "R_global": float(r_global),
            "per_detection": per,
            "n_scored": len(per),
        }


def parse_detection(
    row: Iterable[float],
) -> Tuple[float, float, float, float, float, int, np.ndarray]:
    """Unpack a JSONL detection row into (cx, cy, w, h, conf, cls, scores)."""
    vals = list(row)
    cx, cy, w, h, conf, cls = vals[:6]
    return cx, cy, w, h, conf, int(cls), np.asarray(vals[6:], dtype=float)
