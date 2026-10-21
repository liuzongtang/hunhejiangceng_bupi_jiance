"""Unit tests for the inference-side alarm scorer (backend.inference.alarm_scorer)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from backend.inference.alarm_scorer import (
    NUM_CLASSES,
    SIMILAR_PARTNERS,
    AlarmConfig,
    AlarmScorer,
)


def make_scores(top: int, top_val: float = 2.0, others: float = 0.0) -> np.ndarray:
    s = np.full(NUM_CLASSES, others, dtype=float)
    s[top] = top_val
    return s


def make_det(cx, cy, w, h, conf, cls, scores) -> list:
    return [cx, cy, w, h, conf, cls, *[float(v) for v in scores]]


# ---------------------------------------------------------------- fixtures
@pytest.fixture
def scorer() -> AlarmScorer:
    return AlarmScorer()


# ------------------------------------------------------------------- tests
def test_severity_tiers(scorer):
    for c in (0, 9, 15):  # hole / broken_warp / broken_spandex
        assert scorer.d1_severity(c) == 1.0
    for c in (14, 17):  # star_skip / weave_defect
        assert scorer.d1_severity(c) == 0.6
    for c in (1, 4, 8, 16):  # minor classes
        assert scorer.d1_severity(c) == 0.3


def test_default_weights_normalized_d6_disabled(scorer):
    # product mode: weights live on the 5 quality dims; D1 is the multiplicative gate
    total = sum(scorer.w.values())
    assert math.isclose(total, 1.0, abs_tol=1e-9)
    assert scorer.w["D6"] == 0.0
    for d in ("D2", "D3", "D4", "D5", "D7"):
        assert math.isclose(scorer.w[d], 1 / 5, abs_tol=1e-9)


def test_custom_weights_normalized():
    cfg = AlarmConfig(
        weights={
            "D1": 3.0,
            "D2": 3.0,
            "D3": 1.0,
            "D4": 1.0,
            "D5": 1.0,
            "D7": 1.0,
            "D6": 0.0,
        }
    )
    s = AlarmScorer(cfg)
    assert math.isclose(sum(s.w.values()), 1.0, abs_tol=1e-9)
    assert math.isclose(
        s.w["D2"], 3 / 7, abs_tol=1e-9
    )  # D1 ignored (gate, not quality)


def test_d3_uncertainty_bounds(scorer):
    dominant = make_scores(3, top_val=8.0)
    assert scorer.d3_uncertainty(dominant) < 0.01
    tied = np.zeros(NUM_CLASSES)
    tied[3] = tied[9] = 2.0
    assert scorer.d3_uncertainty(tied) == pytest.approx(1.0, abs=1e-6)
    mid = make_scores(3, top_val=2.0)
    d3 = scorer.d3_uncertainty(mid)
    assert 0.0 <= d3 <= 1.0


def test_d4_zero_without_partners(scorer):
    assert SIMILAR_PARTNERS[1] == frozenset()  # stain: no confusion partner
    assert scorer.d4_group_conflict(make_scores(1), 1) == 0.0


def test_d4_partner_conflict_levels(scorer):
    # knot (3) partners: three_silk(2), warping_knot(13), weft_shrink(11)
    strong = make_scores(3, 2.0)
    strong[2] = 1.98  # runner-up essentially tied
    weak = make_scores(3, 2.0)
    for p in (2, 11, 13):
        weak[p] = -4.0  # ALL partners far below the top-1
    assert scorer.d4_group_conflict(strong, 3) > 0.9
    assert scorer.d4_group_conflict(weak, 3) < 0.05


def test_d2_is_top1_conf(scorer):
    # D2 = top1_score (the plan's first D2 option)
    det_strong = make_det(0.5, 0.5, 0.02, 0.02, 0.91, 3, make_scores(3))
    det_weak = make_det(0.5, 0.5, 0.02, 0.02, 0.26, 3, make_scores(3))
    assert scorer.d2_confidence(det_strong) == pytest.approx(0.91)
    assert scorer.d2_confidence(det_weak) == pytest.approx(0.26)


def test_group_softmax_restricted_to_members(scorer):
    s = make_scores(3, 2.0)
    s[2] = 0.0
    q = scorer._group_softmax(s, [3, 2, 13, 11])
    assert q.sum() == pytest.approx(1.0, abs=1e-9)
    assert q.size == 4
    # dominant top-1 in-group -> D2 (group mass) close to 1
    s2 = make_scores(3, 8.0)
    q2 = scorer._group_softmax(s2, [3, 2, 13, 11])
    assert q2[0] > 0.99


def test_d5_size_ramp_and_saturation(scorer):
    assert scorer.d5_size(0.0, 0.0) == 0.0
    assert scorer.d5_size(math.sqrt(0.05), math.sqrt(0.05)) == pytest.approx(1.0)
    half = scorer.d5_size(0.05, 0.5)  # area 0.025 = size_sat/2
    assert half == pytest.approx(0.5, abs=1e-6)


def test_d7_clustering_radius_and_saturation(scorer):
    centre = make_det(0.5, 0.5, 0.01, 0.01, 0.9, 3, make_scores(3))
    near = make_det(0.55, 0.52, 0.01, 0.01, 0.8, 3, make_scores(3))  # within r=0.15
    far = make_det(0.95, 0.95, 0.01, 0.01, 0.8, 3, make_scores(3))  # outside
    low_conf = make_det(
        0.52, 0.52, 0.01, 0.01, 0.1, 3, make_scores(3)
    )  # below min conf
    dets = [centre, near, far, low_conf]
    assert scorer.d7_clustering(dets, 0) == pytest.approx(1 / 5)
    # saturation at cluster_sat neighbours
    many = [centre] + [
        make_det(0.5 + 0.01 * i, 0.5, 0.01, 0.01, 0.9, 3, make_scores(3))
        for i in range(1, 8)
    ]
    assert scorer.d7_clustering(many, 0) == 1.0


def test_noisy_or_composition(scorer):
    det = make_det(0.5, 0.5, 0.02, 0.02, 0.9, 9, make_scores(9))
    single = scorer.score_image([det])
    r1 = single["per_detection"][0]["R"]
    assert single["R_global"] == pytest.approx(r1, abs=1e-9)
    det2 = make_det(0.2, 0.2, 0.02, 0.02, 0.9, 9, make_scores(9))
    double = scorer.score_image([det, det2])
    assert double["R_global"] == pytest.approx(1 - (1 - r1) ** 2, abs=1e-9)


def test_empty_and_evidence_gate(scorer):
    assert scorer.score_image([])["R_global"] == 0.0
    cfg = AlarmConfig(evidence_conf=0.5)
    s = AlarmScorer(cfg)
    low = make_det(0.5, 0.5, 0.02, 0.02, 0.1, 9, make_scores(9))
    assert s.score_image([low])["R_global"] == 0.0
    high = make_det(0.5, 0.5, 0.02, 0.02, 0.9, 9, make_scores(9))
    assert s.score_image([low, high])["R_global"] > 0.0


def test_evidence_gate_prevents_saturation(scorer):
    """Junk queries (flat uncertain scores) must not drive R_global to 1."""
    flat = np.zeros(NUM_CLASSES)  # maximally uncertain score vector
    junk = [make_det(0.1 * i, 0.5, 0.01, 0.01, 0.2, 1, flat) for i in range(30)]
    out = scorer.score_image(junk)
    assert out["R_global"] == 0.0  # all below evidence_conf
    borderline = [make_det(0.1 * i, 0.5, 0.01, 0.01, 0.26, 1, flat) for i in range(5)]
    out2 = scorer.score_image(borderline)
    assert 0.0 < out2["R_global"] < 1.0


def test_score_detection_shape_and_range(scorer):
    det = make_det(0.4, 0.4, 0.2, 0.25, 0.85, 9, make_scores(9))
    dims = scorer.score_detection([det], 0)
    for d in ("D1", "D2", "D3", "D4", "D5", "D6", "D7", "Q", "R"):
        assert d in dims
        assert 0.0 <= dims[d] <= 1.0
    assert dims["D6"] == 0.0
    # product form: critical severity (1.0) x quality (~0.72) -> high R
    assert 0.6 < dims["R"] < 0.85
    assert dims["R"] == pytest.approx(dims["D1"] * dims["Q"], abs=1e-9)
    # small fabric-scale box drags D5/Q down -> lower R (realistic case)
    small = make_det(0.4, 0.4, 0.03, 0.02, 0.85, 9, make_scores(9))
    assert scorer.score_detection([small], 0)["R"] < dims["R"]


def test_minor_class_severity_gates_score(scorer):
    """Same box/scores, minor vs critical class: R scales with severity.

    Uses hole (critical, no confusion partners) vs stain (minor, no partners)
    so D4 is identical and only the D1 gate differs.
    """
    scores = make_scores(1)  # top-1 = stain either way (same score vector)
    det_minor = make_det(0.4, 0.4, 0.2, 0.25, 0.85, 1, scores)
    det_crit = make_det(0.4, 0.4, 0.2, 0.25, 0.85, 0, scores)
    assert (
        scorer.d4_group_conflict(scores, 1)
        == scorer.d4_group_conflict(scores, 0)
        == 0.0
    )
    r_minor = scorer.score_detection([det_minor], 0)["R"]
    r_crit = scorer.score_detection([det_crit], 0)["R"]
    assert r_crit / r_minor == pytest.approx(1.0 / 0.3, rel=1e-6)


def test_linear_combination_available():
    """The plan's literal linear form is kept as a comparison arm."""
    cfg = AlarmConfig(combination="linear")
    s = AlarmScorer(cfg)
    det = make_det(0.4, 0.4, 0.2, 0.25, 0.85, 9, make_scores(9))
    dims = s.score_detection([det], 0)
    expected = sum(s.w[d] * dims[d] for d in ("D1", "D2", "D3", "D4", "D5", "D7"))
    assert dims["R"] == pytest.approx(expected, abs=1e-9)
    with pytest.raises(ValueError):
        AlarmScorer(AlarmConfig(combination="bogus"))


def test_wrong_score_vector_raises(scorer):
    bad = [0.5, 0.5, 0.02, 0.02, 0.9, 3, 0.1, 0.2]  # only 2 class scores
    with pytest.raises(ValueError):
        scorer.score_detection([bad], 0)


def test_real_replay_record_smoke():
    """Score a real dumped record end-to-end; check invariants."""
    path = Path("runs/alarm_replay/detections_mosaic1280.jsonl")
    if not path.exists():
        pytest.skip("replay dump not present")
    with open(path, encoding="utf-8") as f:
        rec = json.loads(f.readline())
    scorer = AlarmScorer()
    out = scorer.score_image(rec["detections"])
    assert out["n_scored"] <= len(rec["detections"])
    assert 0.0 <= out["R_global"] <= 1.0
    # image max-conf equals the dumped top detection's conf (topk order)
    top_conf = rec["detections"][0][4]
    assert top_conf == max(d[4] for d in rec["detections"])
    # per-detection R is a weighted mean of [0,1] dims -> in [0,1]
    for p in out["per_detection"]:
        assert 0.0 <= p["R"] <= 1.0
