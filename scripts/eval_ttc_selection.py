#!/usr/bin/env python
"""eval_ttc_selection.py -- test-time *selection* over candidate detection sets (S0).

Spec: handoff appendix C / section 3 ("S0 测试时选择实验", S0 test-time selection). The
experiment changes
only two things at inference time -- the candidate detection sets and the rule that
picks among them -- and never touches a weight. Because no training signal moves, the
precision/recall zero-sum that every weight-level intervention reproduced on this
project (handoff section 11 F1, paper section 5.5) cannot occur by construction. A
positive result therefore means "inference-time reward exists"; a negative result is
strong evidence that the decision/selection layer is already optimal and that the
reward can only act during early training.

What is pre-registered (fixed before looking at any number):

* IoU vector ``[0.1, 0.3, 0.5]`` and the official mean, i.e. the mean of mAP@0.1,
  mAP@0.3 and mAP@0.5 -- the same "official contest metric" every other script uses.
* The verdict rule: **any rule mean > B* + 0.005**. ``B*`` is the baseline mean of
  *this machine's* current checkpoint and must be supplied by the operator via
  ``--baseline-mean``. It is deliberately NOT hard-coded: the historical 0.6349
  belonged to ``runs/rtdetr/tianchi18_mosaic1280/weights/best.pt``, that checkpoint no
  longer exists here, and re-using its number as a gate would fake a comparison
  against a baseline that was never run. Without ``--baseline-mean`` the rule table is
  still printed, but no PASS/FAIL line is emitted.
* The IoU/conf/max_det protocol: conf=0.001, max_det=300, ultralytics
  ``match_predictions`` + ``ap_per_class``, so numbers are directly comparable with
  ``scripts/eval_multIou.py``.

Candidate sets (per image):

* ``C1`` -- the official pipeline output: the unmodified ultralytics validator run at
  ``--imgsz``, whose per-image predictions are captured after the official
  post-processing (so conf/max_det filtering is exactly the official one). The
  validator's own summary is kept as an independent check on the re-implemented
  metric math (see "Sanity" below).
* ``C2_k`` -- C1 truncated to the top-k scoring prefixes, k in {100, 200, 300}. The
  k=300 prefix is identical to C1 by construction (official ``max_det`` is 300), so it
  is reported but not duplicated as a separate candidate: a duplicate candidate makes
  ``argmax`` tie-breaking meaningless.
* ``C3`` -- multi-scale TTA fusion over imgsz {1280, 1536} x {identity, hflip} using a
  weighted box fusion (WBF) implemented in this file (no extra dependency). Fusion
  weight protocol is described below and is deliberately *not* part of the selection
  rules.

Selection rules:

* ``R1`` -- fixed baseline: every image uses C1. This is the control; by construction
  its mean equals the candidate-table C1 entry.
* ``R2`` -- manual scoring of the candidate set on seven dimensions. See
  :func:`score_candidate_r2` for the dimension-by-dimension substitutes and
  :data:`R2_DIM_WEIGHTS`.
* ``R3`` -- learned combiner: ridge regression on GT-free candidate statistics, label =
  each candidate's per-image mean-AP contribution against GT, 5-fold cross-fitting
  over the val images, reporting the crossfit mean. This is the only rule whose
  verdict may be trusted, because it is the only one whose choice function never sees
  the image it is scoring.

Sanity (what must hold before any rule number may be believed):
    The re-implemented metric path is checked against the validator's own summary on
    the identical C1 prediction set: ``compute_map(C1)['mAP@0.5']`` must equal the
    validator's ``metrics.box.map50`` (the default IoU vector's first entry is exactly
    0.5). A delta above ``--sanity-tol`` (default 0.004, the handoff section 11 F6
    pipeline-divergence tolerance) is printed as an error and recorded in the JSON;
    the run still completes so the operator can inspect it.

APIs verified against the installed ultralytics 8.4.91
------------------------------------------------------
* ``ap_per_class(tp, conf, pred_cls, target_cls, plot=...)`` in
  ``ultralytics/utils/metrics.py`` returns 12 values and an ``ap`` array shaped
  ``(n_present_classes, n_iou)`` in ``iouv`` order, which is the same layout
  ``Metric.all_ap`` exposes.
* ``match_predictions`` is **not** a module-level function in this release: it is
  ``BaseValidator.match_predictions`` (``engine/validator.py``), driven by ``self.iouv``,
  taking ``(pred_classes (N,), true_classes (M,), iou (M, N))`` and returning
  ``(N, n_iou)``. :class:`_IouvShim` supplies only ``iouv``, so the upstream matcher is
  reused verbatim rather than re-implemented.
* Metric plumbing matches what ``eval_multIou.py`` relies on: ``DetMetrics.process()``
  fills ``metrics.box`` during ``get_stats()``, ``on_val_end`` fires afterwards, so a
  capture hook sees populated ``all_ap`` / ``ap_class_index`` / ``nt_per_class``.
* ``LetterBox`` records ``ratio_pad = ((w_ratio, h_ratio), (left, top))`` -- a two-axis
  ratio, which is why :func:`_letterbox_to_original` inverts per axis rather than using
  ``ops.scale_boxes``' single-gain assumption.
* AP uses ultralytics' 101-point interpolation, under which a single perfectly matched
  box scores 0.995 rather than 1.0. The C1-vs-validator check is unaffected because both
  sides go through ``ap_per_class``.

WBF weight protocol (leakage note):
    The four TTA arms get reliability weights -- CLI-visible as ``g1280``, ``g1536`` and
    ``gflip`` -- searched over a small grid by maximising C3's own fused-set mean on val. Those
    weights are a *hyper-parameter of the candidate generator*, fixed once, then held
    constant for every rule -- no selection rule may touch them, so the rules cannot
    tune the candidate family to themselves. Two honest caveats are reported in the
    output: (a) the grid is scored on val, so C3's own candidate-table row is mildly
    optimistic; (b) ``--wbf-search-limit`` bounds that search to a seeded subset of the
    val images to keep the CPU cost sane. Neither caveat applies to R3's crossfit
    number, which is the pre-registered decision quantity.

Approximation, stated up front: this script has no GT at decision time. Anything that
needs GT is either replaced by a GT-free surrogate (documented per dimension) or
dropped. The verdict is a statement about *selection*, not about how good the
surrogates are.

Iron law 1: the project forbids GPU concurrency with training. ``--device`` therefore
defaults to ``cpu`` (957 images at RT-DETR-L/1280 is a deliberate CPU job, roughly
10-30 min per the handoff) and passing a CUDA device prints a warning.

Usage:
    python scripts/eval_ttc_selection.py \
        --weights runs/rtdetr/<run>/weights/best.pt \
        --data data/tianchi/data.yaml \
        --baseline-mean <this checkpoint's official mean> \
        --out runs/ttc/REPORT.json
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Official contest IoU vector (must match scripts/eval_multIou.py).
IOUS: Tuple[float, ...] = (0.1, 0.3, 0.5)

#: Handoff appendix C / section 3 S0 pre-registered effect threshold.
EFFECT_MARGIN: float = 0.005

#: Official val operating points (must match scripts/eval_multIou.py).
DEFAULT_CONF: float = 0.001
DEFAULT_MAX_DET: int = 300

#: Mirror of ``TIANCHI_BROKEN_CLASS_IDS`` / ``TIANCHI_SKIP_CLASS_IDS`` in
#: ``backend/training/dataset.py`` (18-class Tianchi scheme). Duplicated instead of
#: imported so ``--help`` works without the backend package installed; main() cross
#: checks the mirror against the backend when it is importable and warns on drift.
CRITICAL_BROKEN_IDS: Tuple[int, ...] = (9, 15)  # broken_warp, broken_spandex
CRITICAL_SKIP_IDS: Tuple[int, ...] = (14, 17)  # star_skip, weave_defect

#: Paper table 1 has no numeric weights; the training default in
#: ``backend/training/hybrid_reward_trainer.py`` is ``ones(7) / 7`` (softmaxed again in
#: ``loss_functions.scalar_reward_loss``), so uniform scoring is the faithful default.
R2_DIM_WEIGHTS: Dict[str, float] = {
    "loc": 1.0 / 7.0,
    "cls": 1.0 / 7.0,
    "cal": 1.0 / 7.0,
    "miss": 1.0 / 7.0,
    "fp": 1.0 / 7.0,
    "broken": 1.0 / 7.0,
    "skip": 1.0 / 7.0,
}

#: Penalty slopes copied from ``DimensionRewardComputer``'s constructor defaults
#: (``lambda_miss=1.5``, ``lambda_fp=1.0``) so D04/D05 keep their training scale.
R2_LAMBDA_MISS: float = 1.5
R2_LAMBDA_FP: float = 1.0

#: WBF reliability grid (3 x 4 x 3 = 36 combos), keyed by arm group. A size weight is
#: the arm's base reliability and the flip weight multiplies it. Deliberately small so
#: the search stays a hyper-parameter choice rather than a second learning problem.
WBF_WEIGHT_GRID: Dict[str, Tuple[float, ...]] = {
    "g1280": (0.6, 0.8, 1.0),
    "g1536": (0.4, 0.6, 0.8, 1.0),
    "gflip": (0.5, 0.75, 1.0),
}

#: Feature names consumed by R3's ridge combiner. Every entry is computable from the
#: candidate detection set alone (no GT, no image content beyond the box geometry).
R3_FEATURES: Tuple[str, ...] = (
    "log_n_det",
    "mean_conf",
    "max_conf",
    "std_conf",
    "conf_mass",
    "mean_area_frac",
    "max_area_frac",
    "std_area_frac",
    "mean_aspect",
    "std_aspect",
    "class_frac",
    "class_entropy",
    "redundancy",
    "top1_top5_gap",
    "is_c1",
    "is_c2_100",
    "is_c2_200",
    "is_c3",
)


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class DetSet:
    """A detection set for one image, in original-image pixel coordinates.

    Attributes:
        boxes: ``(n, 4)`` float64 array of xyxy boxes in original image pixels.
        scores: ``(n,)`` float64 confidence scores.
        classes: ``(n,)`` int64 class indices.
    """

    boxes: np.ndarray
    scores: np.ndarray
    classes: np.ndarray

    def __len__(self) -> int:
        """Return the number of detections."""
        return int(self.scores.shape[0])

    def top_k(self, k: int) -> "DetSet":
        """Return the k highest-scoring detections, keeping score order.

        Args:
            k: Number of detections to keep.

        Returns:
            A new :class:`DetSet` with at most ``k`` detections.
        """
        if len(self) <= k:
            return self
        order = np.argsort(-self.scores, kind="stable")[:k]
        return DetSet(self.boxes[order], self.scores[order], self.classes[order])


@dataclass
class GtSet:
    """Ground-truth boxes for one image, in original-image pixel coordinates.

    Attributes:
        boxes: ``(m, 4)`` float64 array of xyxy boxes.
        classes: ``(m,)`` int64 class indices.
    """

    boxes: np.ndarray
    classes: np.ndarray

    def __len__(self) -> int:
        """Return the number of ground-truth boxes."""
        return int(self.classes.shape[0])


@dataclass
class ImageRecord:
    """One val image: file, geometry, ground truth and official-pipeline output.

    Attributes:
        im_file: Path to the source image.
        ori_hw: Original ``(height, width)`` in pixels.
        gt: Ground truth for the image.
        c1: Official-pipeline detections (the C1 candidate).
    """

    im_file: Path
    ori_hw: Tuple[int, int]
    gt: GtSet
    c1: DetSet

    @property
    def area(self) -> float:
        """Return the original image area in pixels."""
        return float(self.ori_hw[0] * self.ori_hw[1])


#: Per-batch capture scratch space filled by the patched ``update_metrics``.
_CAPTURE: Dict[str, list] = {"records": []}


def _empty_detset() -> DetSet:
    """Return an empty detection set with correctly typed arrays."""
    return DetSet(
        boxes=np.zeros((0, 4), dtype=np.float64),
        scores=np.zeros((0,), dtype=np.float64),
        classes=np.zeros((0,), dtype=np.int64),
    )


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Compute the pairwise IoU matrix between two xyxy box arrays.

    Args:
        boxes_a: ``(n, 4)`` xyxy boxes.
        boxes_b: ``(m, 4)`` xyxy boxes.

    Returns:
        ``(n, m)`` float64 IoU matrix; empty ``(n, 0)`` / ``(0, m)`` when an input is
        empty.
    """
    if boxes_a.size == 0 or boxes_b.size == 0:
        return np.zeros((boxes_a.shape[0], boxes_b.shape[0]), dtype=np.float64)
    ax1, ay1, ax2, ay2 = (boxes_a[:, i][:, None] for i in range(4))
    bx1, by1, bx2, by2 = (boxes_b[:, i][None, :] for i in range(4))
    inter_w = np.clip(np.minimum(ax2, bx2) - np.maximum(ax1, bx1), 0.0, None)
    inter_h = np.clip(np.minimum(ay2, by2) - np.maximum(ay1, by1), 0.0, None)
    inter = inter_w * inter_h
    area_a = np.clip(ax2 - ax1, 0.0, None) * np.clip(ay2 - ay1, 0.0, None)
    area_b = np.clip(bx2 - bx1, 0.0, None) * np.clip(by2 - by1, 0.0, None)
    union = area_a + area_b - inter
    return np.where(union > 0.0, inter / np.maximum(union, 1e-12), 0.0)


class _IouvShim:
    """Minimal stand-in carrying the IoU vector for ``BaseValidator.match_predictions``.

    In ultralytics 8.4.91 the matcher is a validator method that reads ``self.iouv``
    and nothing else, so this shim reuses the exact upstream implementation without
    constructing (and running) a validator.

    Args:
        ious: IoU thresholds the matcher should produce columns for.
    """

    def __init__(self, ious: Sequence[float] = IOUS) -> None:
        import torch

        self.iouv = torch.tensor(tuple(ious), dtype=torch.float32)


def _match_to_correct(
    pred_cls: np.ndarray,
    gt_cls: np.ndarray,
    iou_gt_by_pred: np.ndarray,
    n_iou: int,
) -> np.ndarray:
    """Greedy class-aware matching via ultralytics' own ``match_predictions``.

    Verified against ultralytics 8.4.91: ``match_predictions`` is **not** a function in
    ``ultralytics.utils.metrics`` -- it is ``BaseValidator.match_predictions``, a method
    driven by ``self.iouv`` (``engine/validator.py``). It takes ``pred_classes (N,)``,
    ``true_classes (M,)`` and an ``(M, N)`` gt-major IoU matrix, and returns an
    ``(N, n_iou)`` bool tensor, which is exactly the layout ``ap_per_class`` expects.
    An older release exposed a module-level function instead, so that path is kept as a
    fallback.

    Args:
        pred_cls: ``(n_pred,)`` predicted class indices.
        gt_cls: ``(n_gt,)`` ground-truth class indices.
        iou_gt_by_pred: ``(n_gt, n_pred)`` IoU matrix.
        n_iou: Number of IoU thresholds in :data:`IOUS`.

    Returns:
        ``(n_pred, n_iou)`` bool matrix flagging true positives.

    Raises:
        RuntimeError: If the installed matcher returns an unexpected layout (guards
            against an ultralytics signature change).
    """
    import torch

    expected = (int(pred_cls.shape[0]), int(n_iou))
    pred_t = torch.as_tensor(np.asarray(pred_cls), dtype=torch.float32)
    gt_t = torch.as_tensor(np.asarray(gt_cls), dtype=torch.float32)
    iou_t = torch.as_tensor(np.asarray(iou_gt_by_pred), dtype=torch.float32)

    correct = None
    try:
        from ultralytics.engine.validator import BaseValidator
    except ImportError:  # pragma: no cover - only on very old releases
        BaseValidator = None  # type: ignore[assignment]
    if BaseValidator is not None and hasattr(BaseValidator, "match_predictions"):
        correct = BaseValidator.match_predictions(_IouvShim(), pred_t, gt_t, iou_t)
    else:  # pragma: no cover - only on very old releases
        from ultralytics.utils.metrics import match_predictions  # type: ignore[attr-defined]

        correct = match_predictions(pred_t, gt_t, iou_t)

    if torch.is_tensor(correct):
        correct = correct.detach().cpu().numpy()
    correct = np.asarray(correct, dtype=bool)
    if correct.shape == expected:
        return correct
    if correct.shape == (expected[1], expected[0]):
        return correct.T
    raise RuntimeError(
        f"match_predictions returned {correct.shape}, expected {expected}; "
        "the installed ultralytics layout changed -- re-verify before trusting any number"
    )


def compute_map(
    det_sets: Sequence[DetSet],
    gts: Sequence[GtSet],
    ious: Sequence[float] = IOUS,
) -> Dict[str, object]:
    """Compute official-protocol mAP at each IoU in ``ious``.

    Mirrors the ultralytics detection validator: per image build the
    ``(n_pred, n_iou)`` correctness matrix with ``match_predictions`` (class-aware,
    IoU-descending greedy), pool all images, then call ``ap_per_class`` once. Classes
    are averaged over the ones present in the targets, matching ``Metric.all_ap`` whose
    rows are ``ap_class_index``.

    Args:
        det_sets: Per-image detection sets (original-image pixel coordinates).
        gts: Per-image ground truth, same order and coordinate space.
        ious: IoU thresholds, defaults to :data:`IOUS`.

    Returns:
        Dict with ``mAP@<iou>`` floats, ``mean`` (official mean over the IoU vector),
        ``per_class`` (class id -> AP list) and ``n_classes_present``.
    """
    from ultralytics.utils.metrics import ap_per_class

    n_iou = len(ious)
    tp_chunks: List[np.ndarray] = []
    conf_chunks: List[np.ndarray] = []
    cls_chunks: List[np.ndarray] = []
    tgt_chunks: List[np.ndarray] = []

    for det, gt in zip(det_sets, gts):
        n_pred = len(det)
        n_gt = len(gt)
        if n_pred == 0:
            # Same contract as DetectionValidator.update_metrics: an image with GT but
            # no predictions still contributes its targets to the class counts.
            if n_gt:
                tp_chunks.append(np.zeros((0, n_iou), dtype=bool))
                conf_chunks.append(np.zeros((0,), dtype=np.float64))
                cls_chunks.append(np.zeros((0,), dtype=np.float64))
                tgt_chunks.append(np.asarray(gt.classes, dtype=np.float64))
            continue
        if n_gt:
            iou = iou_matrix(np.asarray(gt.boxes, dtype=np.float64), det.boxes)
            correct = _match_to_correct(det.classes, gt.classes, iou, n_iou)
        else:
            correct = np.zeros((n_pred, n_iou), dtype=bool)
        tp_chunks.append(correct)
        conf_chunks.append(np.asarray(det.scores, dtype=np.float64))
        cls_chunks.append(np.asarray(det.classes, dtype=np.float64))
        tgt_chunks.append(np.asarray(gt.classes, dtype=np.float64))

    if not tp_chunks or not tgt_chunks:
        zeros = {f"mAP@{t}": 0.0 for t in ious}
        return {
            **zeros,
            "mean": 0.0,
            "per_class": {},
            "n_classes_present": 0,
        }

    tp = np.concatenate(tp_chunks, axis=0).astype(np.float32)
    conf = np.concatenate(conf_chunks, axis=0)
    pred_cls = np.concatenate(cls_chunks, axis=0)
    target_cls = np.concatenate(tgt_chunks, axis=0)

    if tp.shape[0] == 0:
        tp, conf, pred_cls = (
            np.zeros((0, n_iou), dtype=np.float32),
            np.zeros((0,), dtype=np.float64),
            np.zeros((0,), dtype=np.float64),
        )

    # Degenerate sets (an image with no GT, or a class with no predictions) make
    # ap_per_class take the mean of an empty slice; numpy's warnings there are noise,
    # not signal, and R3 computes per-image AP thousands of times.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        (
            _tp,
            _fp,
            _p,
            _r,
            _f1,
            ap,
            unique_classes,
            *_rest,
        ) = ap_per_class(tp, conf, pred_cls, target_cls, plot=False)

    ap = np.asarray(ap, dtype=np.float64)
    if ap.ndim != 2 or ap.shape[0] == 0:
        map_at = {f"mAP@{t}": 0.0 for t in ious}
        return {
            **map_at,
            "mean": 0.0,
            "per_class": {},
            "n_classes_present": 0,
        }

    ap = ap[:, :n_iou]
    map_at = {f"mAP@{t}": float(ap[:, i].mean()) for i, t in enumerate(ious)}
    classes = [int(c) for c in np.asarray(unique_classes).ravel()]
    per_class = {
        str(c): [float(v) for v in ap[row]] for row, c in enumerate(classes)
    }
    mean = float(np.mean([map_at[k] for k in map_at]))
    return {
        **map_at,
        "mean": mean,
        "per_class": per_class,
        "n_classes_present": len(classes),
    }


# ---------------------------------------------------------------------------
# Weighted box fusion (implemented here: no extra dependency)
# ---------------------------------------------------------------------------


def _fuse_one_class(
    boxes: np.ndarray,
    scores: np.ndarray,
    iou_thr: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Fuse one class's boxes from several arms with weighted box fusion.

    Standard WBF (Solovyev et al., 2021): visit boxes in descending score order, attach
    each to the existing cluster whose current fused box has the highest IoU (>= iou_thr)
    or start a new cluster, then average coordinates weighted by score. Scores arriving
    here are already arm-weight-scaled.

    Args:
        boxes: ``(n, 4)`` xyxy boxes from all arms.
        scores: ``(n,)`` arm-weighted scores.
        iou_thr: Cluster attachment threshold.

    Returns:
        Tuple ``(fused_boxes, fused_scores)`` where fused score is the mean
        arm-weighted score divided by the number of arms that contributed to the
        cluster (``min(T, n) / T`` rescaling of the standard formulation).
    """
    order = np.argsort(-scores, kind="stable")
    clusters: List[List[int]] = []
    fused: List[np.ndarray] = []

    for idx in order:
        box = boxes[idx]
        if not clusters:
            clusters.append([int(idx)])
            fused.append(box.copy())
            continue
        ious = iou_matrix(np.vstack(fused), box[None, :]).ravel()
        j = int(np.argmax(ious))
        if ious[j] >= iou_thr:
            clusters[j].append(int(idx))
            members = np.asarray(clusters[j], dtype=np.int64)
            w = scores[members]
            wsum = float(w.sum())
            if wsum > 0.0:
                fused[j] = (boxes[members] * w[:, None]).sum(axis=0) / wsum
        else:
            clusters.append([int(idx)])
            fused.append(box.copy())

    out_boxes = np.zeros((len(clusters), 4), dtype=np.float64)
    out_scores = np.zeros((len(clusters),), dtype=np.float64)
    for j, members in enumerate(clusters):
        idxs = np.asarray(members, dtype=np.int64)
        out_boxes[j] = fused[j]
        out_scores[j] = float(scores[idxs].sum())
    return out_boxes, out_scores


def wbf_fuse(
    arm_dets: Sequence[DetSet],
    arm_weights: Sequence[float],
    iou_thr: float = 0.55,
) -> DetSet:
    """Weighted box fusion across TTA arms for a single image.

    Args:
        arm_dets: One :class:`DetSet` per TTA arm, original-image pixel coordinates.
        arm_weights: Reliability weight per arm, same length as ``arm_dets``.
        iou_thr: WBF cluster-attachment IoU.

    Returns:
        Fused :class:`DetSet`. The confidence of a fused box is
        ``sum_i(w_i * s_i) / sum_i(w_i) * min(T, n) / T`` where ``T`` is the number of
        arms and ``n`` how many arms contributed: this is the standard WBF score
        rescaling, which lets multi-arm consensus outrank single-arm detections.
    """
    n_arms = len(arm_dets)
    if n_arms == 0:
        return _empty_detset()
    weight_sum = float(np.sum(arm_weights))
    weight_sum = weight_sum if weight_sum > 0.0 else 1.0

    all_boxes: List[np.ndarray] = []
    all_scores: List[np.ndarray] = []
    all_classes: List[np.ndarray] = []

    present = sorted({int(c) for det in arm_dets for c in det.classes.tolist()})
    for cls in present:
        per_arm_boxes: List[np.ndarray] = []
        per_arm_scores: List[np.ndarray] = []
        for det, weight in zip(arm_dets, arm_weights):
            mask = det.classes == cls
            if not np.any(mask):
                continue
            per_arm_boxes.append(det.boxes[mask])
            per_arm_scores.append(det.scores[mask] * float(weight))
        if not per_arm_boxes:
            continue
        fused_boxes, fused_scores = _fuse_one_class(
            np.concatenate(per_arm_boxes, axis=0),
            np.concatenate(per_arm_scores, axis=0),
            iou_thr,
        )
        # Recover per-cluster arm counts by re-running the assignment cheaply: the
        # score sum already encodes the weights, so count contributing arms by matching
        # fused boxes back to at most one box per arm.
        counts = np.zeros((len(fused_boxes),), dtype=np.float64)
        for det in arm_dets:
            mask = det.classes == cls
            if not np.any(mask):
                continue
            ious = iou_matrix(fused_boxes, det.boxes[mask])
            counts += (ious.max(axis=1) >= iou_thr).astype(np.float64)
        counts = np.clip(counts, 1.0, float(n_arms))

        scores = (fused_scores / weight_sum) * (counts / float(n_arms))
        all_boxes.append(fused_boxes)
        all_scores.append(scores)
        all_classes.append(np.full((len(fused_boxes),), cls, dtype=np.int64))

    if not all_boxes:
        return _empty_detset()
    boxes = np.concatenate(all_boxes, axis=0)
    scores = np.concatenate(all_scores, axis=0)
    classes = np.concatenate(all_classes, axis=0)
    order = np.argsort(-scores, kind="stable")
    return DetSet(boxes[order], scores[order], classes[order])


# ---------------------------------------------------------------------------
# R2: seven-dimension manual scoring with GT-free substitutes
# ---------------------------------------------------------------------------


def score_candidate_r2(
    det: DetSet,
    ref: DetSet,
    n_reference: int,
    area: float,
) -> Dict[str, float]:
    """Score one candidate set on the seven reward dimensions, GT-free.

    ``backend/training/dimension_rewards.py`` defines the canonical seven dimensions
    against ground truth. At test time there is no GT, so each dimension that needs it
    is replaced by an explicitly documented surrogate:

    D01 ``loc`` (needs GT boxes):
        ``R = 2 * agreement - 1``, where ``agreement`` is the mean of each detection's
        best IoU against boxes of the *same class* from the other source family
        (C1-family candidates are scored against the C3 arms and vice versa). This is
        cross-view localization consistency instead of GT localization accuracy.
    D02 ``cls`` (needs the GT class):
        ``R = 2 * mean_conf - 1``. NOT computable in the true sense: the smooth
        accuracy term needs the target class, so mean confidence is the only available
        surrogate. Consequence, stated honestly: D02 collapses into a confidence term
        and therefore double-counts the same quantity that D03 measures the calibration
        *of*. R2 is consequently close to a confidence/agreement heuristic.
    D03 ``cal`` (needs p_true):
        ``R = -|mean_conf - agreement|`` -- agreement stands in for the soft accuracy,
        making this a confidence-versus-consensus gap rather than a confidence-versus-
        correctness gap. Still bounded in [-1, 0] like the original.
    D04 ``miss`` (needs GT to know a target was missed):
        ``R = -lambda_miss * clip(1 - n / n_reference, 0, 1)``, i.e. relative detection
        retention against the reference set. Monotone in detection count, so it rewards
        larger candidate sets; that is the honest price of having no recall oracle.
    D05 ``fp`` (the only dimension with a genuine GT-free analogue):
        ``R = -lambda_fp * mean(conf_i * (1 - agreement_i))``. An uncorroborated
        confident box is the standard signature of a false positive, which is exactly
        what the original ``soft_fp`` encodes -- the original still needs GT to decide
        whether the box was right, the surrogate uses cross-view agreement instead.
    D06 ``broken`` / D07 ``skip`` (need the GT label to condition on "this image should
        contain a critical defect"):
        ``R = 3 * max_conf_on_class_set - 2``, the unconditional form of the training
        reward (which only fires on critical-GT samples). Every candidate for the same
        image shares the same missing per-image offset, so the substitution is exact
        for candidate *ranking* even though the absolute score scale differs.

    Args:
        det: Candidate detection set for one image.
        ref: Cross-family reference detections used as the agreement target.
        n_reference: Detection count of the reference set (for the D04 proxy).
        area: Original image area in pixels (for scale-free area terms).

    Returns:
        Dict mapping the seven dimension names to reward floats in the training scale.
    """
    del area  # reserved: area normalisation is applied by R3 features, not here
    n = len(det)
    worst = {
        "loc": -1.0,
        "cls": -1.0,
        "cal": -1.0,
        "miss": -R2_LAMBDA_MISS,
        "fp": -R2_LAMBDA_FP,
        "broken": -2.0,
        "skip": -2.0,
    }
    if n == 0:
        return worst

    agreement = np.zeros((n,), dtype=np.float64)
    if len(ref) > 0:
        ious = iou_matrix(det.boxes, ref.boxes)  # (n, n_ref)
        same = det.classes[:, None] == ref.classes[None, :]
        masked = np.where(same, ious, 0.0)
        agreement = masked.max(axis=1)
    mean_conf = float(np.mean(det.scores))
    mean_agreement = float(np.mean(agreement))

    crit_broken = np.isin(det.classes, np.asarray(CRITICAL_BROKEN_IDS, dtype=np.int64))
    crit_skip = np.isin(det.classes, np.asarray(CRITICAL_SKIP_IDS, dtype=np.int64))
    max_conf_broken = float(det.scores[crit_broken].max()) if np.any(crit_broken) else 0.0
    max_conf_skip = float(det.scores[crit_skip].max()) if np.any(crit_skip) else 0.0

    retention = 1.0 - (n / max(float(n_reference), 1.0))
    retention = float(np.clip(retention, 0.0, 1.0))

    return {
        "loc": 2.0 * mean_agreement - 1.0,
        "cls": 2.0 * mean_conf - 1.0,
        "cal": -abs(mean_conf - mean_agreement),
        "miss": -R2_LAMBDA_MISS * retention,
        "fp": -R2_LAMBDA_FP * float(np.mean(det.scores * (1.0 - agreement))),
        "broken": 3.0 * max_conf_broken - 2.0,
        "skip": 3.0 * max_conf_skip - 2.0,
    }


def r2_aggregate(dims: Dict[str, float]) -> float:
    """Aggregate the seven dimension rewards with the uniform default weights.

    Mirrors ``loss_functions.scalar_reward_loss``, which computes
    ``softmax(dim_weights)`` over the seven dimensions and sums ``w_i * R_i``. With the
    training default ``ones(7) / 7`` the weights are already normalised, so the raw
    reward scale is preserved (no per-image re-normalisation, which would change what
    the dimensions mean).

    Args:
        dims: Mapping of the seven dimension names to rewards.

    Returns:
        Scalar candidate score; higher is better.
    """
    total = 0.0
    weight_sum = 0.0
    for name, weight in R2_DIM_WEIGHTS.items():
        total += weight * float(dims.get(name, 0.0))
        weight_sum += weight
    return total / weight_sum if weight_sum > 0.0 else 0.0


# ---------------------------------------------------------------------------
# R3: learned combiner features and ridge cross-fitting
# ---------------------------------------------------------------------------


def r3_features(det: DetSet, source: str, area: float) -> np.ndarray:
    """Build the GT-free feature vector the R3 combiner scores candidates with.

    Args:
        det: Candidate detection set.
        source: Candidate name (``C1`` / ``C2_100`` / ``C2_200`` / ``C3``); encoded as
            one-hot source indicators so the combiner may learn a source prior.
        area: Original image area in pixels.

    Returns:
        ``(len(R3_FEATURES),)`` float64 feature vector.
    """
    n = len(det)
    if n == 0:
        return np.zeros((len(R3_FEATURES),), dtype=np.float64)

    scores = det.scores
    boxes = det.boxes
    widths = np.clip(boxes[:, 2] - boxes[:, 0], 1e-9, None)
    heights = np.clip(boxes[:, 3] - boxes[:, 1], 1e-9, None)
    areas_frac = (widths * heights) / max(area, 1.0)
    aspects = widths / heights

    classes, counts = np.unique(det.classes, return_counts=True)
    probs = counts.astype(np.float64) / float(n)
    entropy = float(-np.sum(probs * np.log(np.maximum(probs, 1e-12))))
    entropy = entropy / math.log(max(len(classes), 2))

    k = min(20, n)
    top = boxes[:k]
    if k > 1:
        ious = iou_matrix(top, top)
        iu = np.triu_indices(k, k=1)
        redundancy = float(np.mean(ious[iu])) if iu[0].size else 0.0
    else:
        redundancy = 0.0

    top1_top5_gap = float(scores[0] - scores[min(4, n - 1)])

    return np.asarray(
        [
            math.log1p(n),
            float(np.mean(scores)),
            float(np.max(scores)),
            float(np.std(scores)),
            float(np.mean(scores[: min(20, n)])),
            float(np.mean(areas_frac)),
            float(np.max(areas_frac)),
            float(np.std(areas_frac)),
            float(np.mean(aspects)),
            float(np.std(aspects)),
            len(classes) / 18.0,
            entropy,
            redundancy,
            top1_top5_gap,
            1.0 if source == "C1" else 0.0,
            1.0 if source == "C2_100" else 0.0,
            1.0 if source == "C2_200" else 0.0,
            1.0 if source == "C3" else 0.0,
        ],
        dtype=np.float64,
    )


def ridge_fit(x: np.ndarray, y: np.ndarray, lam: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit a ridge regressor in closed form on standardised features.

    Args:
        x: ``(n, d)`` design matrix.
        y: ``(n,)`` targets.
        lam: L2 penalty (the intercept is not penalised).

    Returns:
        Tuple ``(coef, mean, std)`` such that predictions are
        ``((x - mean) / std) @ coef[:-1] + coef[-1]``.
    """
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std = np.where(std > 0.0, std, 1.0)
    xs = (x - mean) / std
    design = np.hstack([xs, np.ones((xs.shape[0], 1), dtype=np.float64)])
    penalty = np.eye(design.shape[1], dtype=np.float64) * lam
    penalty[-1, -1] = 0.0
    gram = design.T @ design + penalty
    coef = np.linalg.solve(gram, design.T @ y)
    return coef, mean, std


def ridge_predict(x: np.ndarray, coef: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Predict with a ridge model produced by :func:`ridge_fit`.

    Args:
        x: ``(n, d)`` design matrix.
        coef: Coefficients from :func:`ridge_fit`.
        mean: Feature means from :func:`ridge_fit`.
        std: Feature standard deviations from :func:`ridge_fit`.

    Returns:
        ``(n,)`` predictions.
    """
    xs = (x - mean) / np.where(std > 0.0, std, 1.0)
    design = np.hstack([xs, np.ones((xs.shape[0], 1), dtype=np.float64)])
    return design @ coef


# ---------------------------------------------------------------------------
# Official-pipeline capture
# ---------------------------------------------------------------------------


def _letterbox_to_original(
    boxes,
    lb_hw: Tuple[int, int],
    ori_hw: Tuple[int, int],
    ratio_pad=None,
) -> np.ndarray:
    """Invert the letterbox transform back to original-image pixels.

    Verified against ultralytics 8.4.91: ``LetterBox`` records
    ``ratio_pad = ((w_ratio, h_ratio), (left, top))`` -- a **two-element** ratio, not
    the single gain ``ops.scale_boxes`` assumes -- so the inversion is per-axis. This is
    also why the helper does not call ``scale_boxes``: that function (a) mutates the
    tensor it is given in place, which here would corrupt the validator's own
    ``pred["bboxes"]`` before the official metrics run, and (b) clips to the image,
    while the validator matches unclipped predictions. This reproduces the same
    arithmetic without mutating or clipping.

    Args:
        boxes: ``(n, 4)`` xyxy letterbox-space boxes (torch tensor or numpy array).
        lb_hw: Letterboxed ``(height, width)``.
        ori_hw: Original ``(height, width)``.
        ratio_pad: Dataset-recorded ``((w_ratio, h_ratio), (left, top))`` when
            available; otherwise derived from the two shapes assuming an
            aspect-preserving letterbox (the only case that can be re-derived).

    Returns:
        ``(n, 4)`` float64 array in original-image pixels.
    """
    import torch

    if torch.is_tensor(boxes):
        arr = boxes.detach().cpu().numpy().astype(np.float64, copy=True)
    else:
        arr = np.array(boxes, dtype=np.float64, copy=True)
    if arr.size == 0:
        return arr.reshape(0, 4)

    if ratio_pad is not None:
        ratio = ratio_pad[0]
        if isinstance(ratio, (tuple, list)) and len(ratio) == 2:
            ratio_w, ratio_h = float(ratio[0]), float(ratio[1])
        else:  # scalar gain from an older release
            ratio_w = ratio_h = float(ratio)
        pad_x, pad_y = float(ratio_pad[1][0]), float(ratio_pad[1][1])
    else:
        ratio_w = ratio_h = min(lb_hw[0] / ori_hw[0], lb_hw[1] / ori_hw[1])
        pad_x = round((lb_hw[1] - round(ori_hw[1] * ratio_w)) / 2 - 0.1)
        pad_y = round((lb_hw[0] - round(ori_hw[0] * ratio_h)) / 2 - 0.1)
    if ratio_w <= 0.0:
        ratio_w = 1.0
    if ratio_h <= 0.0:
        ratio_h = 1.0

    arr[:, [0, 2]] = (arr[:, [0, 2]] - pad_x) / ratio_w
    arr[:, [1, 3]] = (arr[:, [1, 3]] - pad_y) / ratio_h
    return arr


def install_capture(validator_cls) -> None:
    """Patch ``validator_cls.update_metrics`` to record per-image predictions and GT.

    Mirrors the monkeypatch style ``scripts/eval_multIou.py`` already uses, so the
    official pipeline (dataloader, letterbox policy, conf/max_det filtering,
    post-processing) is untouched; the original method still runs and still produces
    the official metrics.

    Args:
        validator_cls: ``RTDETRValidator`` or ``DetectionValidator`` class to patch.
    """
    import torch
    from ultralytics.utils.ops import xywh2xyxy

    if getattr(validator_cls, "_ttc_capture_installed", False):
        return
    validator_cls._ttc_capture_installed = True  # type: ignore[attr-defined]
    original = validator_cls.update_metrics

    def patched(self, preds, batch):
        try:
            images = batch["img"]
            ori_shapes = batch.get("ori_shape")
            ratio_pads = batch.get("ratio_pad")
            n = len(preds)
            for si in range(n):
                lb_hw = (int(images.shape[2]), int(images.shape[3]))
                shape = (
                    tuple(int(v) for v in ori_shapes[si])
                    if ori_shapes is not None
                    else (int(images.shape[2]), int(images.shape[3]))
                )
                # batch["ratio_pad"][si] is the letterbox (gain, (padw, padh)) the
                # dataset actually used, so the inverse transform is exact instead of
                # re-derived from shapes.
                ratio_pad = ratio_pads[si] if ratio_pads is not None else None
                idx = batch["batch_idx"] == si
                cls = batch["cls"][idx].reshape(-1)
                bbox = batch["bboxes"][idx]

                # GT arrives as letterbox-normalised xywh (DetectionValidator
                # ._prepare_batch multiplies by the letterbox HW). It is inverted to
                # original-image pixels so every candidate set shares one space.
                if cls.numel():
                    wh = torch.tensor(
                        (lb_hw[1], lb_hw[0], lb_hw[1], lb_hw[0]),
                        device=bbox.device,
                        dtype=bbox.dtype,
                    )
                    gt_lb = xywh2xyxy(bbox) * wh
                    gt = GtSet(
                        boxes=_letterbox_to_original(gt_lb, lb_hw, shape, ratio_pad),
                        classes=cls.detach().cpu().numpy().astype(np.int64),
                    )
                else:
                    gt = GtSet(
                        boxes=np.zeros((0, 4), dtype=np.float64),
                        classes=np.zeros((0,), dtype=np.int64),
                    )

                pred = preds[si]
                p_boxes = pred["bboxes"]
                if len(p_boxes):
                    # RT-DETR's postprocess already returns pixel xyxy in letterbox
                    # space (normalised xywh * imgsz) and YOLO's NMS output is in the
                    # same space, so the same inverse applies to both.
                    scaled = _letterbox_to_original(p_boxes, lb_hw, shape, ratio_pad)
                    p_scores = np.asarray(
                        pred["conf"].detach().cpu().numpy(), dtype=np.float64
                    ).reshape(-1)
                    p_classes = np.asarray(
                        pred["cls"].detach().cpu().numpy(), dtype=np.int64
                    ).reshape(-1)
                    # Force score order once here: C2 prefix truncation and the R3
                    # top-k features both assume descending confidence.
                    order = np.argsort(-p_scores, kind="stable")
                    det = DetSet(scaled[order], p_scores[order], p_classes[order])
                else:
                    det = _empty_detset()

                im_file = batch.get("im_file")
                path = Path(im_file[si]) if im_file is not None else Path(f"image_{si}")
                _CAPTURE["records"].append(ImageRecord(path, shape, gt, det))
        except Exception as exc:  # noqa: BLE001 - capture must never break the val run
            print(f"WARNING: per-image capture failed ({exc!r}); continuing", file=sys.stderr)
            _CAPTURE["records"].clear()

        return original(self, preds, batch)

    validator_cls.update_metrics = patched


#: TTA arms captured per image: (imgsz, flipped) -> list of DetSet indexed by image.
TtaArms = Dict[str, List[DetSet]]


def run_tta_arms(
    model,
    records: Sequence[ImageRecord],
    sizes: Sequence[int],
    flips: Sequence[bool],
    conf: float,
    max_det: int,
    device: str,
    chunk: int = 16,
) -> TtaArms:
    """Run every TTA arm over all val images with ``model.predict``.

    Documented caveat: ``predict`` uses ultralytics' automatic letterbox (``auto=True``
    when every image in the call shares a shape, i.e. rectangular; otherwise
    aspect-preserving padded), whereas the val pipeline used for C1 letterboxes every
    image to a square ``--imgsz``. C3 therefore differs from C1 in *two* factors --
    scale/flip ensemble and letterbox policy. The identity arm at the baseline imgsz
    doubles as the control that isolates the letterbox factor, and it is reported
    separately as ``predict_control`` in the diagnostics; it is not a selection
    candidate.

    Args:
        model: Loaded ultralytics model.
        records: Val image records (provides the official image order).
        sizes: imgsz values to run.
        flips: Boolean horizontal-flip flags to run.
        conf: Confidence floor (official value from ``--conf``).
        max_det: Detection cap (official value from ``--max-det``).
        device: ultralytics device string.
        chunk: Images per ``predict`` call, bounding host memory.

    Returns:
        Dict ``arm_name -> list[DetSet]`` in ``records`` order, boxes in original-image
        pixel coordinates (flip arms un-flipped back).
    """
    import cv2

    arms: TtaArms = {}
    total = len(records)
    for size in sizes:
        for flip in flips:
            name = f"sz{size}_{'flip' if flip else 'orig'}"
            arm: List[DetSet] = []
            print(f"[tta] arm {name}: {total} images", flush=True)
            for start in range(0, total, chunk):
                batch_records = records[start : start + chunk]
                images = []
                for rec in batch_records:
                    img = cv2.imread(str(rec.im_file))
                    if img is None:
                        raise FileNotFoundError(
                            f"cannot read val image {rec.im_file}; "
                            "check that --data points at an existing dataset"
                        )
                    if flip:
                        img = np.ascontiguousarray(img[:, ::-1])
                    images.append(img)
                results = model.predict(
                    images,
                    imgsz=int(size),
                    conf=float(conf),
                    max_det=int(max_det),
                    device=device,
                    verbose=False,
                    save=False,
                )
                for rec, res in zip(batch_records, results):
                    det = _result_to_detset(res, rec.ori_hw, flipped=flip)
                    arm.append(det)
                del images, results
            arms[name] = arm
    return arms


def _result_to_detset(result, ori_hw: Tuple[int, int], flipped: bool) -> DetSet:
    """Convert one ultralytics ``Results`` object into a :class:`DetSet`.

    Args:
        result: A ``Results`` instance from ``model.predict``.
        ori_hw: Original ``(height, width)`` of the source image.
        flipped: Whether the image was horizontally flipped before inference; when
            true the boxes are mirrored back into unflipped coordinates.

    Returns:
        Detection set in original-image pixel coordinates.
    """
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return _empty_detset()
    xyxy = np.asarray(boxes.xyxy.detach().cpu().numpy(), dtype=np.float64)
    scores = np.asarray(boxes.conf.detach().cpu().numpy(), dtype=np.float64).reshape(-1)
    classes = np.asarray(boxes.cls.detach().cpu().numpy(), dtype=np.int64).reshape(-1)
    if flipped:
        width = float(ori_hw[1])
        x1 = xyxy[:, 0].copy()
        x2 = xyxy[:, 2].copy()
        xyxy[:, 0] = width - x2
        xyxy[:, 2] = width - x1
    order = np.argsort(-scores, kind="stable")
    return DetSet(xyxy[order], scores[order], classes[order])


def search_wbf_weights(
    arms: TtaArms,
    records: Sequence[ImageRecord],
    iou_thr: float,
    limit: int,
    seed: int,
) -> Tuple[Dict[str, float], List[Dict[str, object]]]:
    """Grid-search the WBF arm reliability weights on val.

    See the module docstring "WBF weight protocol": the weights belong to the candidate
    generator, are fixed once, and are never touched by a selection rule.

    Args:
        arms: Raw TTA arm detections keyed by arm name.
        records: Val records (for GT and image order).
        iou_thr: WBF cluster attachment IoU.
        limit: Cap on the number of val images used for the search (0 = all).
        seed: Seed for the deterministic image subset.

    Returns:
        Tuple ``(best_weights, grid_table)`` where ``best_weights`` maps
        ``g1280``/``g1536``/``gflip`` to the chosen values and ``grid_table`` is the
        scored grid in descending order.
    """
    n = len(records)
    if limit and limit < n:
        rng = np.random.default_rng(seed)
        subset = np.sort(rng.choice(n, size=limit, replace=False))
    else:
        subset = np.arange(n)
    sub_records = [records[i] for i in subset]
    sub_arms = {
        name: [dets[i] for i in subset] for name, dets in arms.items()
    }

    table: List[Dict[str, object]] = []
    best: Optional[Dict[str, object]] = None
    for w1280 in WBF_WEIGHT_GRID["g1280"]:
        for w1536 in WBF_WEIGHT_GRID["g1536"]:
            for wflip in WBF_WEIGHT_GRID["gflip"]:
                weights = {
                    "g1280": w1280,
                    "g1536": w1536,
                    "gflip": wflip,
                }
                fused = _fuse_all(sub_arms, sub_records, weights, iou_thr)
                score = compute_map(fused, [r.gt for r in sub_records])
                row = {
                    "weights": weights,
                    "mean": round(float(score["mean"]), 4),
                    "mAP@0.5": round(float(score["mAP@0.5"]), 4),
                }
                table.append(row)
                if best is None or row["mean"] > best["mean"]:
                    best = row
    table.sort(key=lambda r: r["mean"], reverse=True)
    if best is None:
        return {"g1280": 1.0, "g1536": 1.0, "gflip": 1.0}, table
    return dict(best["weights"]), table


def _fuse_all(
    arms: TtaArms,
    records: Sequence[ImageRecord],
    weights: Dict[str, float],
    iou_thr: float,
) -> List[DetSet]:
    """Fuse the per-image arm detections with one weight setting.

    Args:
        arms: Arm detections keyed by arm name.
        records: Val records (for image count / order).
        weights: Arm-group weights ``g1280``, ``g1536``, ``gflip``. The size weight is
            the arm's base reliability and the flip weight multiplies it, so the four
            arms get ``g1280``, ``g1280*gflip``, ``g1536``, ``g1536*gflip``.
        iou_thr: WBF cluster attachment IoU.

    Returns:
        One fused :class:`DetSet` per image in ``records`` order.
    """
    names = sorted(arms.keys())
    per_arm_weights = []
    for name in names:
        base = weights["g1536"] if "1536" in name else weights["g1280"]
        per_arm_weights.append(float(base) * (weights["gflip"] if "flip" in name else 1.0))
    return [
        wbf_fuse([arms[name][i] for name in names], per_arm_weights, iou_thr)
        for i in range(len(records))
    ]


# ---------------------------------------------------------------------------
# Selection rules
# ---------------------------------------------------------------------------


def rule_r1(candidates: Dict[str, List[DetSet]], n_images: int) -> List[str]:
    """R1: always use the C1 baseline candidate.

    Args:
        candidates: Candidate name -> per-image detection sets.
        n_images: Number of images.

    Returns:
        One candidate name per image.
    """
    del candidates
    return ["C1"] * n_images


def rule_r2(
    candidates: Dict[str, List[DetSet]],
    records: Sequence[ImageRecord],
) -> Tuple[List[str], List[Dict[str, float]]]:
    """R2: pick the candidate with the best seven-dimension manual score.

    Each candidate is scored against the *other* source family (C1-family candidates
    against the fused C3 set, C3 against C1) so a candidate never corroborates itself.

    Args:
        candidates: Candidate name -> per-image detection sets.
        records: Val records.

    Returns:
        Tuple ``(choice_per_image, dims_per_image)`` where dims records the winning
        candidate's dimension rewards for inspection.
    """
    names = list(candidates.keys())
    c3 = candidates.get("C3")
    largest_c2 = candidates.get("C2_200") or candidates.get("C2_100")
    choices: List[str] = []
    chosen_dims: List[Dict[str, float]] = []
    for i, rec in enumerate(records):
        best_name = names[0]
        best_score = -np.inf
        best_dims: Dict[str, float] = {}
        for name in names:
            det = candidates[name][i]
            if c3 is not None:
                # Cross-family reference: C1-family candidates are checked against the
                # fused TTA set and vice versa, so nothing corroborates itself.
                ref = c3[i] if name != "C3" else candidates["C1"][i]
            elif name != "C1":
                # --skip-tta: no cross-family reference exists. C2 is a subset of C1, so
                # agreement saturates at 1.0 and D01/D03/D05 stop discriminating; only
                # the D02/D04 terms separate candidates. Documented in main()'s warning.
                ref = candidates["C1"][i]
            else:
                ref = largest_c2[i] if largest_c2 is not None else _empty_detset()
            dims = score_candidate_r2(det, ref, max(len(candidates["C1"][i]), 1), rec.area)
            score = r2_aggregate(dims)
            if score > best_score:
                best_score = score
                best_name = name
                best_dims = dims
        choices.append(best_name)
        chosen_dims.append(best_dims)
    return choices, chosen_dims


def _assemble(
    candidates: Dict[str, List[DetSet]],
    choices: Sequence[str],
) -> List[DetSet]:
    """Build the selected detection set per image from per-image choices.

    Args:
        candidates: Candidate name -> per-image detection sets.
        choices: Chosen candidate name per image.

    Returns:
        Selected detection sets in image order.
    """
    return [candidates[name][i] for i, name in enumerate(choices)]


def rule_r3(
    candidates: Dict[str, List[DetSet]],
    records: Sequence[ImageRecord],
    folds: int,
    lam: float,
    seed: int,
) -> Dict[str, object]:
    """R3: ridge combiner over GT-free candidate statistics, 5-fold cross-fitted.

    Labels are each candidate's per-image mean AP against GT (the "mean-AP
    contribution"). That label is only ever used on the training folds; the held-out
    fold is scored from features alone, so the reported number is an out-of-fold
    estimate. An in-sample fit is also reported, labelled NOT TRUSTWORTHY, purely as a
    leakage reference.

    Args:
        candidates: Candidate name -> per-image detection sets.
        records: Val records.
        folds: Number of cross-fitting folds.
        lam: Ridge penalty.
        seed: Shuffle seed for fold assignment.

    Returns:
        Dict with ``choices`` (per-image candidate name), ``crossfit`` (metric dict),
        ``per_fold`` (list of fold metric dicts), ``in_sample`` (dict, leakage
        reference), ``chosen_counts`` and ``label_means``.
    """
    names = list(candidates.keys())
    n_images = len(records)

    labels = np.zeros((n_images, len(names)), dtype=np.float64)
    feats = np.zeros((n_images, len(names), len(R3_FEATURES)), dtype=np.float64)
    for i, rec in enumerate(records):
        for j, name in enumerate(names):
            det = candidates[name][i]
            labels[i, j] = float(compute_map([det], [rec.gt])["mean"])
            feats[i, j, :] = r3_features(det, name, rec.area)

    x_all = feats.reshape(n_images * len(names), -1)
    y_all = labels.reshape(-1)

    rng = np.random.default_rng(seed)
    order = rng.permutation(n_images)
    fold_of = np.zeros((n_images,), dtype=np.int64)
    for rank, img_i in enumerate(order):
        fold_of[img_i] = rank % folds

    choices: List[str] = [names[0]] * n_images
    per_fold: List[Dict[str, object]] = []

    for fold in range(folds):
        test_imgs = np.where(fold_of == fold)[0]
        train_imgs = np.where(fold_of != fold)[0]
        if test_imgs.size == 0 or train_imgs.size == 0:
            continue
        train_rows = (train_imgs[:, None] * len(names) + np.arange(len(names))[None, :]).reshape(-1)
        test_rows = (test_imgs[:, None] * len(names) + np.arange(len(names))[None, :]).reshape(-1)
        coef, mean, std = ridge_fit(x_all[train_rows], y_all[train_rows], lam)
        pred = ridge_predict(x_all[test_rows], coef, mean, std).reshape(test_imgs.size, len(names))
        best = pred.argmax(axis=1)
        for local, img_i in enumerate(test_imgs):
            choices[int(img_i)] = names[int(best[local])]
        fold_dets = [
            candidates[names[int(best[k])]][int(img_i)]
            for k, img_i in enumerate(test_imgs)
        ]
        fold_map = compute_map(fold_dets, [records[int(i)].gt for i in test_imgs])
        per_fold.append(
            {
                "fold": fold,
                "n_images": int(test_imgs.size),
                "mean": round(float(fold_map["mean"]), 4),
                "mAP@0.5": round(float(fold_map["mAP@0.5"]), 4),
            }
        )

    crossfit = compute_map(_assemble(candidates, choices), [r.gt for r in records])

    coef, mean, std = ridge_fit(x_all, y_all, lam)
    in_sample_pred = ridge_predict(x_all, coef, mean, std).reshape(n_images, len(names))
    in_sample_choices = [names[int(j)] for j in in_sample_pred.argmax(axis=1)]
    in_sample = compute_map(
        _assemble(candidates, in_sample_choices), [r.gt for r in records]
    )

    counts = {name: choices.count(name) for name in names}
    return {
        "choices": choices,
        "crossfit": {
            **{k: float(crossfit[k]) for k in crossfit if k.startswith("mAP@")},
            "mean": float(crossfit["mean"]),
            "per_class": crossfit["per_class"],
            "n_classes_present": crossfit["n_classes_present"],
        },
        "per_fold": per_fold,
        "in_sample": {
            "mean": float(in_sample["mean"]),
            "mAP@0.5": float(in_sample["mAP@0.5"]),
            "note": "in-sample fit; reporting only, NOT TRUSTWORTHY, overfits by construction",
        },
        "chosen_counts": counts,
        "chosen_fraction": {k: v / max(n_images, 1) for k, v in counts.items()},
        "label_means": {
            name: float(labels[:, j].mean()) for j, name in enumerate(names)
        },
        "coefficients": {
            feat: float(coef[k]) for k, feat in enumerate(R3_FEATURES)
        },
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _fmt_table(rows: Sequence[Tuple[str, Dict[str, object]]]) -> str:
    """Render the per-rule metric table as aligned plain text.

    Args:
        rows: ``(label, metrics_dict)`` pairs.

    Returns:
        Multi-line table string.
    """
    header = f"  {'rule':<26s}  {'mean':>7s}  {'mAP@0.1':>7s}  {'mAP@0.3':>7s}  {'mAP@0.5':>7s}"
    lines = [header, "  " + "-" * (len(header) - 2)]
    for label, m in rows:
        lines.append(
            f"  {label:<26s}  {float(m['mean']):>7.4f}  "
            f"{float(m['mAP@0.1']):>7.4f}  {float(m['mAP@0.3']):>7.4f}  "
            f"{float(m['mAP@0.5']):>7.4f}"
        )
    return "\n".join(lines)


def write_report_md(
    path: Path,
    report: Dict[str, object],
    verdict_lines: Sequence[str],
) -> None:
    """Write the human-readable REPORT.md for the run.

    Args:
        path: Destination markdown path.
        report: The JSON-serialisable report dict.
        verdict_lines: Pre-rendered verdict lines (already gated on ``--baseline-mean``).
    """
    lines: List[str] = []
    lines.append("# S0 test-time selection report")
    lines.append("")
    lines.append(f"- weights: `{report['weights']}`")
    lines.append(f"- data: `{report['data']}`  (arch={report['arch']}, imgsz={report['imgsz']})")
    lines.append(f"- device: `{report['device']}`")
    lines.append(
        f"- val images: {report['n_images']}  |  GT boxes: {report['n_gt']}  "
        f"|  conf={report['conf']}, max_det={report['max_det']}"
    )
    lines.append(f"- official mean = mean(mAP@0.1, mAP@0.3, mAP@0.5); IoU vector {report['ious']}")
    lines.append("")
    lines.append("## Candidate sets")
    lines.append("")
    lines.append("| candidate | mean | mAP@0.1 | mAP@0.3 | mAP@0.5 | dets/image |")
    lines.append("|---|---|---|---|---|---|")
    for name, m in report["candidates"].items():
        lines.append(
            f"| {name} | {m['mean']:.4f} | {m['mAP@0.1']:.4f} | {m['mAP@0.3']:.4f} | "
            f"{m['mAP@0.5']:.4f} | {m['dets_per_image']:.1f} |"
        )
    lines.append("")
    lines.append("## Selection rules")
    lines.append("")
    lines.append("| rule | mean | mAP@0.1 | mAP@0.3 | mAP@0.5 | chosen candidate mix |")
    lines.append("|---|---|---|---|---|---|")
    for name, m in report["rules"].items():
        mix = ", ".join(f"{k}:{v:.2f}" for k, v in m.get("chosen_fraction", {}).items())
        lines.append(
            f"| {name} | {m['mean']:.4f} | {m['mAP@0.1']:.4f} | {m['mAP@0.3']:.4f} | "
            f"{m['mAP@0.5']:.4f} | {mix or '-'} |"
        )
    lines.append("")
    lines.append("## Verdict")
    lines.append("")
    lines.append(f"Pre-registered rule: `{report['verdict']['pre_registered']}`")
    lines.append("")
    lines.extend(verdict_lines)
    lines.append("")
    lines.append("## Sanity")
    lines.append("")
    sanity = report["sanity"]
    lines.append(
        f"- recomputed C1 mAP@0.5 `{sanity['recomputed_map50']:.4f}` vs validator "
        f"`{sanity['validator_map50']:.4f}` -> delta `{sanity['delta']:+.6f}` "
        f"(tolerance {sanity['tolerance']})"
    )
    lines.append(f"- status: {sanity['status']}")
    if report.get("diag"):
        lines.append("")
        lines.append("## Diagnostics")
        lines.append("")
        for key, value in report["diag"].items():
            lines.append(f"- {key}: {value}")
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _verify_class_id_mirror() -> None:
    """Cross-check the critical class-id mirrors against the backend package.

    Warns (never fails) when ``backend.training.dataset`` is importable but disagrees,
    which would mean this script's D06/D07 substitutes target the wrong classes.
    """
    try:
        from backend.training.dataset import (  # type: ignore[import-not-found]
            TIANCHI_BROKEN_CLASS_IDS,
            TIANCHI_SKIP_CLASS_IDS,
        )
    except Exception:  # noqa: BLE001 - backend is optional for this script
        return
    if set(TIANCHI_BROKEN_CLASS_IDS) != set(CRITICAL_BROKEN_IDS):
        print(
            f"WARNING: broken class ids drifted from backend: {TIANCHI_BROKEN_CLASS_IDS} "
            f"vs {CRITICAL_BROKEN_IDS}",
            file=sys.stderr,
        )
    if set(TIANCHI_SKIP_CLASS_IDS) != set(CRITICAL_SKIP_IDS):
        print(
            f"WARNING: skip class ids drifted from backend: {TIANCHI_SKIP_CLASS_IDS} "
            f"vs {CRITICAL_SKIP_IDS}",
            file=sys.stderr,
        )


def _parse_int_list(text: str) -> List[int]:
    """Parse a comma-separated integer list.

    Args:
        text: e.g. ``"1280,1536"``.

    Returns:
        List of integers.

    Raises:
        argparse.ArgumentTypeError: If a token is not an integer.
    """
    try:
        return [int(tok.strip()) for tok in text.split(",") if tok.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected comma-separated ints, got {text!r}") from exc


def _parse_bool_list(text: str) -> List[bool]:
    """Parse a comma-separated boolean list (0/1/true/false).

    Args:
        text: e.g. ``"0,1"``.

    Returns:
        List of booleans.

    Raises:
        argparse.ArgumentTypeError: If a token is not a recognised boolean.
    """
    out: List[bool] = []
    for tok in text.split(","):
        tok = tok.strip().lower()
        if not tok:
            continue
        if tok in {"1", "true", "yes", "y"}:
            out.append(True)
        elif tok in {"0", "false", "no", "n"}:
            out.append(False)
        else:
            raise argparse.ArgumentTypeError(f"expected booleans, got {text!r}")
    return out


def main() -> int:
    """Run the S0 test-time selection experiment.

    Returns:
        Process exit code: 0 on success, 2 when the official val pass captured nothing.
    """
    ap = argparse.ArgumentParser(
        description="S0 test-time selection over candidate detection sets (CPU by design)"
    )
    ap.add_argument("--weights", required=True, help="path to .pt checkpoint")
    ap.add_argument(
        "--data", default="data/tianchi/data.yaml", help="dataset yaml (val split)"
    )
    ap.add_argument(
        "--arch",
        default="rtdetr",
        choices=("rtdetr", "yolo"),
        help="family of --weights; rtdetr keeps the official pipeline's forced-square val",
    )
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument(
        "--device",
        default="cpu",
        help="ultralytics device; defaults to cpu (iron law 1: no GPU alongside training)",
    )
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--split", default="val")
    ap.add_argument("--conf", type=float, default=DEFAULT_CONF)
    ap.add_argument("--max-det", type=int, default=DEFAULT_MAX_DET)
    ap.add_argument(
        "--baseline-mean",
        type=float,
        default=None,
        help=(
            "B*: this checkpoint's official mean on this machine. Required for a "
            "PASS/FAIL verdict; omit it and the rule table is printed with no verdict. "
            "Never hard-code the historical 0.6349 here."
        ),
    )
    ap.add_argument("--limit", type=int, default=0, help="cap val images (0 = all)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--folds", type=int, default=5, help="R3 cross-fitting folds")
    ap.add_argument("--ridge-lambda", type=float, default=1.0)
    ap.add_argument("--tta-sizes", type=_parse_int_list, default=[1280, 1536])
    ap.add_argument("--tta-flips", type=_parse_bool_list, default=[False, True])
    ap.add_argument("--wbf-iou", type=float, default=0.55)
    ap.add_argument(
        "--wbf-search-limit",
        type=int,
        default=200,
        help="val images used for the WBF weight grid (0 = all); bounds CPU cost",
    )
    ap.add_argument(
        "--skip-tta",
        action="store_true",
        help="skip C3/TTA entirely (R1/R2/R3 then choose among C1/C2 only)",
    )
    ap.add_argument(
        "--sanity-tol",
        type=float,
        default=0.004,
        help="tolerance for the recomputed-vs-validator mAP@0.5 check (F6)",
    )
    ap.add_argument(
        "--out",
        default="runs/ttc/REPORT.json",
        help="where to write the JSON report (runs/... by convention)",
    )
    ap.add_argument(
        "--report",
        default="runs/ttc/REPORT.md",
        help="where to write the markdown report",
    )
    args = ap.parse_args()

    _verify_class_id_mirror()

    if not str(args.device).lower().startswith("cpu"):
        print(
            "WARNING: --device is a CUDA device. Project iron law 1 forbids GPU work "
            "concurrent with training; this experiment is CPU by design. Proceeding as "
            "asked, but the run may contend with a training job.",
            file=sys.stderr,
        )

    import ultralytics

    if args.arch == "yolo":
        from ultralytics import YOLO as _Model
        from ultralytics.models.yolo.detect.val import DetectionValidator as _Validator
    else:
        from ultralytics import RTDETR as _Model
        from ultralytics.models.rtdetr.val import RTDETRValidator as _Validator

    weights = Path(args.weights)
    if not weights.is_file():
        print(
            f"ERROR: weights not found: {weights}\n"
            "This machine currently has no trained checkpoint; supply --weights for a real run.",
            file=sys.stderr,
        )
        return 2

    print(f"[ttc] ultralytics {ultralytics.__version__} | arch={args.arch} | device={args.device}")
    model = _Model(str(weights))

    _CAPTURE["records"].clear()
    install_capture(_Validator)
    metrics = model.val(
        data=args.data,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        conf=args.conf,
        max_det=args.max_det,
        split=args.split,
        plots=False,
        verbose=False,
        save=False,
        name=f"ttc_{weights.parent.parent.name}",
    )

    records: List[ImageRecord] = list(_CAPTURE["records"])
    if not records:
        print("ERROR: per-image capture failed; cannot build candidate sets", file=sys.stderr)
        return 2
    if args.limit and args.limit < len(records):
        print(f"[ttc] --limit {args.limit} of {len(records)} captured images")
        records = records[: args.limit]

    n_gt = int(sum(len(r.gt) for r in records))
    print(f"[ttc] {len(records)} images, {n_gt} GT boxes, {len(records[0].c1)} C1 dets on image 0")

    candidates: Dict[str, List[DetSet]] = {
        "C1": [r.c1 for r in records],
        "C2_100": [r.c1.top_k(100) for r in records],
        "C2_200": [r.c1.top_k(200) for r in records],
    }

    diag: Dict[str, object] = {
        "note": "C2_300 is identical to C1 by construction and is not duplicated as a candidate"
    }
    wbf_info: Dict[str, object] = {"enabled": False}
    if args.skip_tta:
        print(
            "WARNING: --skip-tta disables C3. R2 then has no cross-family agreement "
            "reference (C2 is a subset of C1, so agreement saturates at 1.0 and only "
            "the D02/D04 terms still separate candidates).",
            file=sys.stderr,
        )
    if not args.skip_tta:
        arms = run_tta_arms(
            model,
            records,
            args.tta_sizes,
            args.tta_flips,
            args.conf,
            args.max_det,
            args.device,
        )
        best_weights, grid_table = search_wbf_weights(
            arms, records, args.wbf_iou, args.wbf_search_limit, args.seed
        )
        print(
            f"[wbf] best weights {best_weights} "
            f"(grid top mean {grid_table[0]['mean'] if grid_table else 'n/a'})"
        )
        candidates["C3"] = _fuse_all(arms, records, best_weights, args.wbf_iou)

        # Diagnostic-only reference sets: each raw arm, plus the predict-pipeline control
        # at the baseline imgsz (isolates the letterbox-policy change from the TTA gain).
        diag["tta_arms"] = {}
        for name, dets in arms.items():
            arm_map = compute_map(dets, [r.gt for r in records])
            diag["tta_arms"][name] = {
                "mean": round(float(arm_map["mean"]), 4),
                "mAP@0.5": round(float(arm_map["mAP@0.5"]), 4),
            }
        control_name = f"sz{args.imgsz}_orig"
        if control_name in arms:
            ctrl = compute_map(arms[control_name], [r.gt for r in records])
            diag["predict_control"] = {
                "candidate": control_name,
                "mean": round(float(ctrl["mean"]), 4),
                "mAP@0.5": round(float(ctrl["mAP@0.5"]), 4),
                "why": "predict-pipeline control: same imgsz/identity as C1 but auto letterbox",
            }
        wbf_info = {
            "enabled": True,
            "iou_thr": args.wbf_iou,
            "weights": best_weights,
            "search_images": min(args.wbf_search_limit, len(records))
            if args.wbf_search_limit
            else len(records),
            "grid_top": grid_table[:5],
            "protocol": (
                "weights are a candidate-generator hyper-parameter fixed once on val and "
                "never touched by a selection rule; C3's own row is therefore mildly "
                "optimistic, R3's crossfit number is the pre-registered decision quantity"
            ),
        }

    # ---- candidate table -------------------------------------------------
    cand_table: Dict[str, Dict[str, object]] = {}
    for name, dets in candidates.items():
        m = compute_map(dets, [r.gt for r in records])
        cand_table[name] = {
            "mean": round(float(m["mean"]), 4),
            "mAP@0.1": round(float(m["mAP@0.1"]), 4),
            "mAP@0.3": round(float(m["mAP@0.3"]), 4),
            "mAP@0.5": round(float(m["mAP@0.5"]), 4),
            "per_class": m["per_class"],
            "n_classes_present": m["n_classes_present"],
            "dets_per_image": float(np.mean([len(d) for d in dets])) if dets else 0.0,
        }

    # ---- sanity: recomputed C1 vs the validator's own summary ------------
    validator_map50 = float(metrics.box.map50)
    recomputed = float(cand_table["C1"]["mAP@0.5"])
    delta = recomputed - validator_map50
    sanity_ok = bool(abs(delta) <= args.sanity_tol)
    sanity_status = (
        "OK"
        if sanity_ok
        else "DIVERGED (see handoff section 11 F6: fix the pipeline before trusting AP@0.1/@0.3)"
    )
    sanity = {
        "recomputed_map50": recomputed,
        "validator_map50": validator_map50,
        "delta": delta,
        "tolerance": args.sanity_tol,
        "status": sanity_status,
    }
    if abs(delta) > args.sanity_tol:
        print(
            f"ERROR: sanity failed: recomputed mAP@0.5 {recomputed:.4f} vs validator "
            f"{validator_map50:.4f} (delta {delta:+.6f} > {args.sanity_tol})",
            file=sys.stderr,
        )
    # ---- rules -----------------------------------------------------------
    rule_metrics: Dict[str, Dict[str, object]] = {}
    choices_map: Dict[str, List[str]] = {}

    r1_choices = rule_r1(candidates, len(records))
    choices_map["R1"] = r1_choices
    rule_metrics["R1"] = compute_map(_assemble(candidates, r1_choices), [r.gt for r in records])

    r2_choices, r2_dims = rule_r2(candidates, records)
    choices_map["R2"] = r2_choices
    rule_metrics["R2"] = compute_map(_assemble(candidates, r2_choices), [r.gt for r in records])

    r3 = rule_r3(candidates, records, args.folds, args.ridge_lambda, args.seed)
    choices_map["R3"] = r3["choices"]  # type: ignore[assignment]
    rule_metrics["R3"] = r3["crossfit"]  # type: ignore[assignment]

    rules_table: Dict[str, Dict[str, object]] = {}
    for name, m in rule_metrics.items():
        chosen = choices_map[name]
        counts = {c: chosen.count(c) / max(len(chosen), 1) for c in sorted(set(chosen))}
        rules_table[name] = {
            "mean": round(float(m["mean"]), 4),
            "mAP@0.1": round(float(m["mAP@0.1"]), 4),
            "mAP@0.3": round(float(m["mAP@0.3"]), 4),
            "mAP@0.5": round(float(m["mAP@0.5"]), 4),
            "per_class": m.get("per_class", {}),
            "chosen_fraction": counts,
        }
    rules_table["R3"]["per_fold"] = r3["per_fold"]
    rules_table["R3"]["in_sample"] = r3["in_sample"]
    rules_table["R3"]["label_means"] = r3["label_means"]
    rules_table["R3"]["coefficients"] = r3["coefficients"]
    rules_table["R2"]["dimension_weights"] = R2_DIM_WEIGHTS
    rules_table["R2"]["lambda_miss"] = R2_LAMBDA_MISS
    rules_table["R2"]["lambda_fp"] = R2_LAMBDA_FP
    rules_table["R2"]["winning_dims_mean"] = {
        k: round(float(np.mean([d.get(k, 0.0) for d in r2_dims])), 4)
        for k in R2_DIM_WEIGHTS
    }

    # ---- verdict (gated on --baseline-mean) ------------------------------
    pre_registered = "any rule mean > B* + 0.005"
    verdict: Dict[str, object] = {
        "pre_registered": pre_registered,
        "baseline_mean": args.baseline_mean,
        "gate": None,
        "printed": False,
        "outcomes": {},
    }
    verdict_lines: List[str] = []
    withheld_reasons: List[str] = []
    if args.baseline_mean is None:
        withheld_reasons.append("--baseline-mean was not supplied, so B* is unknown")
    if not sanity_ok:
        withheld_reasons.append(
            f"the metric path diverged from the validator by {delta:+.6f} > "
            f"{args.sanity_tol} (handoff section 11 F6)"
        )
    if withheld_reasons:
        verdict["withheld_reasons"] = withheld_reasons
        verdict_lines.append("VERDICT: WITHHELD -- no PASS/FAIL is printed because " + "; ".join(withheld_reasons) + ".")
        if args.baseline_mean is None:
            verdict_lines.append(
                "Supply the current checkpoint's official mean (measured on this machine, "
                "not the historical 0.6349) to obtain the pre-registered verdict."
            )
        print(f"\nVERDICT: WITHHELD ({'; '.join(withheld_reasons)}; no PASS/FAIL printed)")
    else:
        gate = float(args.baseline_mean) + EFFECT_MARGIN
        verdict["gate"] = gate
        verdict["printed"] = True
        for name in ("R1", "R2", "R3"):
            mean = float(rules_table[name]["mean"])
            passed = mean > gate
            verdict["outcomes"][name] = {
                "mean": mean,
                "delta_vs_baseline": mean - float(args.baseline_mean),
                "pass": passed,
            }
            verdict_lines.append(
                f"- {name}: mean {mean:.4f} vs gate {gate:.4f} -> "
                f"{'PASS' if passed else 'FAIL'} (delta {mean - float(args.baseline_mean):+.4f})"
            )
        any_pass = any(bool(v["pass"]) for v in verdict["outcomes"].values())
        verdict["any_pass"] = any_pass
        verdict_lines.append("")
        verdict_lines.append(
            "PRE-REGISTERED VERDICT: "
            + (
                "POSITIVE -- inference-time selection beats B* + 0.005; S0 opens S2"
                if any_pass
                else "NEGATIVE -- no rule beats B* + 0.005; the decision/selection layer is saturated"
            )
        )
        print(
            f"\nPRE-REGISTERED VERDICT ({pre_registered}, B*={args.baseline_mean:.4f}, "
            f"gate={gate:.4f}): {'POSITIVE' if any_pass else 'NEGATIVE'}"
        )
        verdict_lines.append(
            "Reminder: only R3's crossfit mean is a trustworthy decision quantity; "
            "R1 is the control and R2 is a documented heuristic."
        )

    # ---- console output --------------------------------------------------
    print("\n================ S0 test-time selection ================")
    print(f"weights : {weights.resolve()}")
    print(
        f"data    : {Path(args.data).resolve()}  (arch={args.arch}, imgsz={args.imgsz}, "
        f"conf={args.conf}, max_det={args.max_det})"
    )
    print(f"images  : {len(records)}  |  GT boxes: {n_gt}")
    print("\n  candidate sets")
    print(_fmt_table([(name, m) for name, m in cand_table.items()]))
    print("\n  selection rules")
    print(_fmt_table([(name, m) for name, m in rules_table.items()]))
    print(
        f"\n  sanity  : recomputed C1 mAP@0.5 {recomputed:.4f} vs validator "
        f"{validator_map50:.4f} (delta {delta:+.6f}, tol {args.sanity_tol}) -> {sanity_status}"
    )
    if wbf_info.get("enabled"):
        print(f"  wbf     : weights {wbf_info['weights']}, iou {wbf_info['iou_thr']}")
    for line in verdict_lines:
        if line and not line.startswith("-"):
            print(f"  {line}")
    print("=======================================================\n")

    # ---- persist ---------------------------------------------------------
    report: Dict[str, object] = {
        "spec": "handoff appendix C / section 3 S0 test-time selection",
        "weights": str(weights.resolve()),
        "data": str(Path(args.data).resolve()),
        "arch": args.arch,
        "imgsz": args.imgsz,
        "device": args.device,
        "conf": args.conf,
        "max_det": args.max_det,
        "ious": list(IOUS),
        "n_images": len(records),
        "n_gt": n_gt,
        "candidates": cand_table,
        "rules": rules_table,
        "wbf": wbf_info,
        "verdict": verdict,
        "sanity": sanity,
        "diag": diag,
        "approximations": {
            "R2": (
                "D01 replaced by cross-source geometric agreement; D02 by mean confidence; "
                "D03 by |confidence - agreement|; D04 by detection retention vs C1; "
                "D05 uses agreement as the FP oracle substitute; D06/D07 use the "
                "unconditional 3*max_conf-2 form (the per-image conditional offset cancels "
                "in the argmax). See score_candidate_r2 for each substitution."
            ),
            "R3_label": (
                "per-image mean AP over IoU 0.1/0.3/0.5, a surrogate for dataset-level "
                "mean-AP contribution (true contributions are pooled per class across "
                "images, which a single-image label cannot express)"
            ),
            "C3": (
                "TTA arms run through model.predict (auto letterbox) while C1 runs through "
                "the val pipeline; the predict_control diagnostic isolates that factor"
            ),
        },
        "env": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "ultralytics": ultralytics.__version__,
            "torch": _torch_version(),
        },
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"report written to {out.resolve()}")

    report_md = Path(args.report)
    write_report_md(report_md, report, verdict_lines)
    print(f"markdown report written to {report_md.resolve()}")

    return 0


def _torch_version() -> str:
    """Return the installed torch version string, or ``"absent"``.

    Returns:
        Version string for the environment block of the report.
    """
    try:
        import torch
    except Exception:  # noqa: BLE001 - env probe only
        return "absent"
    return str(torch.__version__)


if __name__ == "__main__":
    raise SystemExit(main())
