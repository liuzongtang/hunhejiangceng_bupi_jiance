#!/usr/bin/env python
"""oracle_probe.py -- oracle headroom probe for RT-DETR (stage-2 P0.2 / P0.4).

Spec: ``docs/plan_stage2_reward_relaunch_handoff.md`` appendix A (precise version of
P0.2) and section 3 P0.4. The script runs the *official* ultralytics RT-DETR val
pipeline once per invocation and, inside ``RTDETRValidator.update_metrics``, either
leaves predictions untouched (``none``), applies an oracle transform (``box``/``cls``)
or partitions false positives (``fp``). The official matching / ``ap_per_class`` code
then runs unchanged, so the reported mAP shares the metric math of
``scripts/eval_multIou.py``.

Modes
-----
``none``
    Untouched pipeline, mAP at IoU 0.1 / 0.3 / 0.5. This is the pipeline self-check.
``box``
    Every prediction matched to a GT (one-to-one, *class-agnostic*, IoU >= 0.5) gets
    the matched GT box; class and confidence are kept. Recompute mAP.
``cls``
    Every matched prediction gets the matched GT class; box and confidence are kept.
    Recompute mAP.
``fp``
    At ``--tau`` (default 0.25) partition predictions that are not true positives into
    (a) background FP on clean images (no GT), (b) duplicate/neighbour FP on defect
    images (a same-class box on an already-matched GT, or a box overlapping a matched
    prediction box), (c) wrong-class FP (IoU >= 0.5 with a GT of a different class).
    No mAP is reported in this mode.

Official protocol (identical to ``eval_multIou.py`` / ``eval_confusion_decomp.py``)
---------------------------------------------------------------------------------
Square letterbox (``RTDETRDataset(rect=False)``, hardcoded by
``RTDETRValidator.build_dataset``), ``imgsz=1280``, ``conf=0.001``, ``max_det=300``,
``batch=8``, ``workers=0``, and ``iouv=[0.1, 0.3, 0.5]`` (``niou`` kept in sync).
``Metric.map75`` indexes ``all_ap[:, 5]`` and is guarded for ``niou < 6`` exactly as in
``eval_multIou.py``.

Acceptance contract
-------------------
- ``mode=none`` must reproduce ``scripts/eval_multIou.py`` on the same weights:
  ``tianchi18_mosaic1280`` best.pt -> mean 0.6349 / mAP@0.5 0.5917, and
  ``tianchi18_full`` best.pt -> mAP@0.5 0.541. When
  ``runs/multi_iou/handoff_baseline.json`` (or ``--baseline-json``) exists the
  comparison is measured and recorded in the ``sanity`` block.
- ``mode=box`` and ``mode=cls`` mAP must be >= ``mode=none`` (monotonicity). A single
  invocation cannot see the other modes, so the block records the contract and, when a
  baseline JSON exists, the per-IoU delta against it.

Known deviation (not used for any reported number)
--------------------------------------------------
``Metric.update_image_metrics`` reads column 0 as IoU 0.5. With ``iouv=[0.1, 0.3, 0.5]``
column 0 is IoU 0.1, so the internal per-image P/R/F1 dict describes IoU 0.1. All mAP
numbers come from ``all_ap`` and are unaffected; ``eval_multIou.py`` has the same
property.

Note on box-mode headroom
-------------------------
With the spec's match threshold (IoU >= 0.5) the box swap only rewrites predictions that
are already true positives at all three reported IoUs (a 0.5 match implies a 0.1/0.3
match), so ``mode=box`` reproduces ``mode=none`` by construction and can only re-rank
duplicate predictions. The measurable localization headroom is the *near-miss* oracle:
run ``mode=box`` with a lower ``--iou-thr`` (0.3 or 0.1) to snap boxes that miss the GT
at IoU 0.5 onto it; its delta then tests how much of the AP@0.1 - AP@0.5 gap (0.074 on
mosaic1280 per the P0.2 quick version) is recoverable by better boxes.

Usage:
    python scripts/oracle_probe.py --weights runs/rtdetr/tianchi18_mosaic1280/weights/best.pt \
        --mode none --out runs/headroom/mosaic1280_none.json
    python scripts/oracle_probe.py --weights ... --mode box --out runs/headroom/mosaic1280_box.json
    python scripts/oracle_probe.py --weights ... --mode cls --out runs/headroom/mosaic1280_cls.json
    python scripts/oracle_probe.py --weights ... --mode fp --tau 0.25 \
        --out runs/headroom/mosaic1280_fp.json
"""

from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
import platform
import sys

IOUS = (0.1, 0.3, 0.5)
DEFAULT_VAL_CONF = 0.001  # official RT-DETR val confidence floor
DEFAULT_MAX_DET = 300  # RT-DETR final queries
DEFAULT_MATCH_IOU = 0.5  # GT-level matching threshold of the oracle transform
DEFAULT_TAU = 0.25  # working confidence for the FP partition (P0.4)
SANITY_TOL = 1e-4  # mode=none vs eval_multIou tolerance on 4-decimal mAP values
DEFAULT_BASELINE_JSON = Path("runs/multi_iou/handoff_baseline.json")
DEFAULT_OUT_DIR = Path("runs/headroom")


def patch_metric_map75(metric_cls) -> None:
    """Guard ``Metric.map75`` so ``summary()`` cannot crash when ``niou < 6``.

    ``Metric.map75`` reads ``all_ap[:, 5]``; with ``iouv=[0.1, 0.3, 0.5]`` there is no
    column 5. Same guard as ``scripts/eval_multIou.py``; nothing reads map75 afterwards.

    Args:
        metric_cls: The ``ultralytics.utils.metrics.Metric`` class to patch in place.
    """
    import numpy as np

    def _map75(self) -> float:
        all_ap = getattr(self, "all_ap", None)
        if all_ap is None or len(all_ap) == 0:
            return 0.0
        array = np.asarray(all_ap)
        if array.ndim != 2 or array.shape[1] <= 5:
            return float("nan")
        return float(array[:, 5].mean())

    metric_cls.map75 = property(_map75)


def iou_matrix(boxes_a, boxes_b):
    """IoU matrix between two xyxy box tensors, as a numpy array.

    Args:
        boxes_a: ``torch.Tensor`` of shape ``(A, 4)`` in xyxy coordinates.
        boxes_b: ``torch.Tensor`` of shape ``(B, 4)`` in xyxy coordinates.

    Returns:
        numpy.ndarray: ``(A, B)`` IoU matrix (all zeros when either side is empty).
    """
    import numpy as np
    from ultralytics.utils.metrics import box_iou

    if boxes_a.numel() == 0 or boxes_b.numel() == 0:
        return np.zeros((boxes_a.shape[0], boxes_b.shape[0]))
    return box_iou(boxes_a, boxes_b).detach().cpu().numpy()


def greedy_pairs(iou, allowed=None, iou_thr: float = DEFAULT_MATCH_IOU) -> list:
    """One-to-one greedy assignment, candidate pairs taken in descending IoU order.

    Mirrors the greedy rule of ``ultralytics.engine.validator.match_predictions`` (sort
    candidate pairs by IoU, keep the first pair whose GT and prediction are both free),
    so the oracle transform and the FP partition share one assignment convention.

    Args:
        iou: ``(n_gt, n_pred)`` IoU matrix.
        allowed: Optional ``(n_gt, n_pred)`` boolean mask (e.g. same class, above tau).
        iou_thr: Minimum IoU for a candidate pair.

    Returns:
        list: ``(gt_index, pred_index)`` pairs; every GT and prediction is used once.
    """
    import numpy as np

    n_gt, n_pred = iou.shape
    if n_gt == 0 or n_pred == 0:
        return []
    candidate = iou >= float(iou_thr)
    if allowed is not None:
        candidate = candidate & allowed
    gi, pi = np.nonzero(candidate)
    if gi.size == 0:
        return []
    order = np.argsort(-iou[gi, pi], kind="mergesort")
    gt_used = np.zeros(n_gt, dtype=bool)
    pred_used = np.zeros(n_pred, dtype=bool)
    pairs = []
    for k in order:
        g, p = int(gi[k]), int(pi[k])
        if gt_used[g] or pred_used[p]:
            continue
        gt_used[g] = True
        pred_used[p] = True
        pairs.append((g, p))
    return pairs


def classify_false_positives(
    iou_gt_fp,
    gt_cls,
    pred_cls_fp,
    matched_gt,
    iou_fp_vs_tp,
    iou_thr: float = DEFAULT_MATCH_IOU,
) -> tuple:
    """Assign above-tau false positives to the P0.4 buckets.

    The four masks partition every false positive exactly once under this precedence
    (also recorded in the JSON output): wrong class, then a same-class box on an
    already-matched GT, then a box overlapping an already-matched prediction, then the
    residual spurious box on a defect image.

    Args:
        iou_gt_fp: ``(n_gt, n_fp)`` IoU between GT boxes and the FP predictions.
        gt_cls: ``(n_gt,)`` GT class ids.
        pred_cls_fp: ``(n_fp,)`` predicted class ids of the FP predictions.
        matched_gt: ``(n_gt,)`` bool mask of GT already matched by a true positive.
        iou_fp_vs_tp: ``(n_fp, n_tp)`` IoU between FP and TP prediction boxes.
        iou_thr: Matching IoU threshold.

    Returns:
        tuple: Four boolean masks of length ``n_fp``: ``wrong_class``,
        ``duplicate_same_class_matched_gt``, ``overlaps_matched_pred_box``,
        ``residual_defect_image``.
    """
    import numpy as np

    n_fp = pred_cls_fp.shape[0]
    if n_fp == 0:
        return tuple(np.zeros(0, dtype=bool) for _ in range(4))

    hit = iou_gt_fp >= float(iou_thr)
    wrong_class = (hit & (gt_cls[:, None] != pred_cls_fp[None, :])).any(axis=0)
    duplicate_gt = (
        hit & (gt_cls[:, None] == pred_cls_fp[None, :]) & matched_gt[:, None]
    ).any(axis=0) & ~wrong_class
    if iou_fp_vs_tp.shape[1]:
        overlaps_matched = (iou_fp_vs_tp >= float(iou_thr)).any(axis=1)
    else:
        overlaps_matched = np.zeros(n_fp, dtype=bool)
    duplicate_pred = overlaps_matched & ~wrong_class & ~duplicate_gt
    residual = ~(wrong_class | duplicate_gt | duplicate_pred)
    return wrong_class, duplicate_gt, duplicate_pred, residual


def build_probe_validator(mode: str, tau: float, iou_thr: float):
    """Build the ``RTDETRValidator`` subclass used for one probe run.

    The subclass only overrides ``__init__`` (fixing ``iouv``/``niou`` and registering
    the instance) and ``update_metrics`` (oracle transform / FP accumulation) and then
    calls ``super().update_metrics`` so the official matching and AP math run unchanged.

    Args:
        mode: One of ``none``, ``box``, ``cls``, ``fp``.
        tau: Working confidence threshold for the ``fp`` partition.
        iou_thr: Matching IoU threshold for the oracle transform and the FP partition.

    Returns:
        type: A subclass of ``RTDETRValidator``. After a run its ``last_instance`` class
        attribute holds that run's validator (used to read metrics and FP counters).
    """
    import torch
    from ultralytics.models.rtdetr.val import RTDETRValidator

    class OracleProbeValidator(RTDETRValidator):
        """RT-DETR validator that applies an oracle transform before official matching."""

        probe_mode = mode
        probe_tau = float(tau)
        probe_iou_thr = float(iou_thr)
        last_instance = None

        def __init__(self, *args, **kwargs):
            """Initialise the validator and fix the IoU vector to the three official IoUs.

            Args:
                *args: Positional arguments forwarded to ``RTDETRValidator``.
                **kwargs: Keyword arguments forwarded to ``RTDETRValidator``.
            """
            super().__init__(*args, **kwargs)
            self.iouv = torch.tensor(IOUS, dtype=torch.float32)
            self.niou = int(self.iouv.numel())  # keep niou in sync (empty-pred fast path)
            type(self).last_instance = self
            self.fp_stats = {
                "n_images": 0,
                "n_images_clean": 0,
                "n_images_defect": 0,
                "n_preds_above_tau": 0,
                "n_tp": 0,
                "background": 0,
                "duplicate": 0,
                "wrong_class": 0,
                "dup_matched_gt": 0,
                "dup_matched_pred": 0,
                "dup_residual": 0,
            }

        def update_metrics(self, preds, batch) -> None:
            """Apply the probe transform, then run the official metric update.

            Args:
                preds: List of post-processed prediction dicts (``bboxes`` xyxy in image
                    scale, ``conf``, ``cls``), one per image in ``batch``.
                batch: Batch dict with the letterboxed images and the GT targets.
            """
            if self.probe_mode == "fp":
                self._accumulate_fp(preds, batch)
            elif self.probe_mode in {"box", "cls"}:
                preds = [
                    self._apply_oracle(pred, si, batch)
                    for si, pred in enumerate(preds)
                ]
            super().update_metrics(preds, batch)

        def _apply_oracle(self, pred: dict, si: int, batch: dict) -> dict:
            """Swap matched boxes (``box``) or matched classes (``cls``) to the GT values.

            Args:
                pred: Prediction dict for image ``si``.
                si: Image index inside the batch.
                batch: Batch dict holding the GT targets.

            Returns:
                dict: Prediction dict with ``bboxes`` or ``cls`` replaced; unchanged when
                nothing matches or the image has no prediction/GT.
            """
            pbatch = self._prepare_batch(si, batch)  # same GT convention as the matcher
            gt_boxes = pbatch["bboxes"]
            gt_cls = pbatch["cls"]
            if gt_boxes.numel() == 0 or pred["cls"].shape[0] == 0:
                return pred

            pairs = greedy_pairs(
                iou_matrix(gt_boxes, pred["bboxes"]), iou_thr=self.probe_iou_thr
            )
            if not pairs:
                return pred

            out = dict(pred)
            if self.probe_mode == "box":
                boxes = pred["bboxes"].clone()
                for g, p in pairs:
                    boxes[p] = gt_boxes[g].to(boxes.dtype)
                out["bboxes"] = boxes
            else:  # cls
                cls = pred["cls"].clone()
                for g, p in pairs:
                    cls[p] = gt_cls[g].to(cls.dtype)
                out["cls"] = cls
            return out

        def _accumulate_fp(self, preds, batch) -> None:
            """Split above-``tau`` non-TP predictions into the three P0.4 FP buckets.

            Precedence (documented in the JSON output): clean-image background first, then
            wrong-class, then a same-class box on an already-matched GT, then a box
            overlapping an already-matched prediction, then the residual spurious box on a
            defect image (reported as a sub-reason of (b) so the three buckets stay
            exhaustive).

            Args:
                preds: List of post-processed prediction dicts, one per image.
                batch: Batch dict holding the GT targets.
            """
            import numpy as np

            st = self.fp_stats
            thr = float(self.probe_iou_thr)
            for si, pred in enumerate(preds):
                pbatch = self._prepare_batch(si, batch)
                st["n_images"] += 1

                conf = pred["conf"].detach().cpu().numpy()
                pcls = pred["cls"].detach().cpu().numpy().astype(int).reshape(-1)
                above = conf >= float(self.probe_tau)
                st["n_preds_above_tau"] += int(above.sum())

                gcls = pbatch["cls"].detach().cpu().numpy().astype(int).reshape(-1)
                if gcls.size == 0:
                    st["n_images_clean"] += 1
                    st["background"] += int(above.sum())
                    continue
                st["n_images_defect"] += 1

                gbox, pbox = pbatch["bboxes"], pred["bboxes"]
                iou = iou_matrix(gbox, pbox)  # (G, P)
                allowed = (gcls[:, None] == pcls[None, :]) & above[None, :]
                pairs = greedy_pairs(iou, allowed=allowed, iou_thr=thr)

                tp = np.zeros(pcls.shape[0], dtype=bool)
                matched_gt = np.zeros(gcls.shape[0], dtype=bool)
                for g, p in pairs:
                    tp[p] = True
                    matched_gt[g] = True
                st["n_tp"] += int(tp.sum())

                fp_index = np.nonzero(above & ~tp)[0]
                if fp_index.size == 0:
                    continue

                tp_index = np.nonzero(tp)[0]
                if tp_index.size:
                    fp_mask = torch.as_tensor(fp_index, device=pbox.device)
                    tp_mask = torch.as_tensor(tp_index, device=pbox.device)
                    iou_fp_vs_tp = iou_matrix(pbox[fp_mask], pbox[tp_mask])  # (F, T)
                else:
                    iou_fp_vs_tp = np.zeros((fp_index.size, 0))

                wrong, dup_gt, dup_pred, residual = classify_false_positives(
                    iou[:, fp_index],
                    gcls,
                    pcls[fp_index],
                    matched_gt,
                    iou_fp_vs_tp,
                    iou_thr=thr,
                )
                st["wrong_class"] += int(wrong.sum())
                st["dup_matched_gt"] += int(dup_gt.sum())
                st["dup_matched_pred"] += int(dup_pred.sum())
                st["dup_residual"] += int(residual.sum())
                st["duplicate"] += int(dup_gt.sum() + dup_pred.sum() + residual.sum())

        def fp_report(self) -> dict:
            """Build the ``fp`` mode report from the accumulated counters.

            Returns:
                dict: Counts and shares for the three FP categories plus sub-reasons.

            Raises:
                RuntimeError: If the partition does not sum to the FP total, which would
                    mean the accounting logic is inconsistent.
            """
            st = self.fp_stats
            n_fp = st["background"] + st["duplicate"] + st["wrong_class"]
            n_preds = st["n_preds_above_tau"]
            implied_fp = n_preds - st["n_tp"]
            sub_sum = st["dup_matched_gt"] + st["dup_matched_pred"] + st["dup_residual"]
            if sub_sum != st["duplicate"]:
                raise RuntimeError(
                    f"FP accounting error: duplicate={st['duplicate']} != sub-reasons={sub_sum}"
                )
            if implied_fp != n_fp:
                raise RuntimeError(
                    f"FP accounting error: above_tau - tp={implied_fp} != category sum={n_fp}"
                )

            def share(count: int) -> float | None:
                return round(count / n_fp, 4) if n_fp else None

            def share_of_preds(count: int) -> float | None:
                return round(count / n_preds, 4) if n_preds else None

            def bucket(count: int, description: str) -> dict:
                return {
                    "count": int(count),
                    "share_of_fp": share(count),
                    "share_of_preds_above_tau": share_of_preds(count),
                    "description": description,
                }

            return {
                "tau": float(self.probe_tau),
                "iou_thr": float(self.probe_iou_thr),
                "n_images": st["n_images"],
                "n_images_clean": st["n_images_clean"],
                "n_images_defect": st["n_images_defect"],
                "n_preds_above_tau": n_preds,
                "n_tp_above_tau": st["n_tp"],
                "n_fp": n_fp,
                "categories": {
                    "background_clean_image": bucket(
                        st["background"], "FP on an image with no GT (clean image)"
                    ),
                    "duplicate_or_neighbour": {
                        **bucket(
                            st["duplicate"],
                            "FP on a defect image explained by an already-matched object "
                            "(or a residual spurious box, see sub_reasons)",
                        ),
                        "sub_reasons": {
                            "duplicate_same_class_matched_gt": bucket(
                                st["dup_matched_gt"],
                                "IoU >= iou_thr with an already-matched GT of the predicted class",
                            ),
                            "overlaps_matched_pred_box": bucket(
                                st["dup_matched_pred"],
                                "IoU >= iou_thr with an already-matched (TP) prediction box",
                            ),
                            "residual_defect_image": bucket(
                                st["dup_residual"],
                                "residual FP on a defect image: no >= iou_thr GT overlap and no "
                                "matched-box overlap (kept in (b) so the partition is exhaustive)",
                            ),
                        },
                    },
                    "wrong_class": bucket(
                        st["wrong_class"],
                        "IoU >= iou_thr with a GT of a different class",
                    ),
                },
                "precedence": [
                    "background_clean_image",
                    "wrong_class",
                    "duplicate_same_class_matched_gt",
                    "overlaps_matched_pred_box",
                    "residual_defect_image",
                ],
                "partition_check": {
                    "sum_of_categories": int(n_fp),
                    "n_preds_above_tau_minus_tp": int(implied_fp),
                    "equals_n_fp": True,
                },
            }

    return OracleProbeValidator


def parse_args(argv=None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Optional argument list (defaults to ``sys.argv[1:]``).

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Oracle headroom probe for RT-DETR (appendix A of "
            "docs/plan_stage2_reward_relaunch_handoff.md): mAP at IoU 0.1/0.3/0.5 with an "
            "untouched, box-oracle or class-oracle prediction set, or the P0.4 FP split."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--weights", required=True, help="path to the .pt checkpoint")
    parser.add_argument(
        "--mode",
        required=True,
        choices=["none", "box", "cls", "fp"],
        help="none=untouched self-check, box/cls=oracle transform, fp=false-positive split",
    )
    parser.add_argument(
        "--data", default="data/tianchi/data.yaml", help="dataset yaml (val split)"
    )
    parser.add_argument("--imgsz", type=int, default=1280, help="square letterbox size")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0", help="ultralytics device string")
    parser.add_argument(
        "--workers", type=int, default=0, help="dataloader workers (0 = official repro setting)"
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=DEFAULT_TAU,
        help="fp mode: working confidence threshold for the FP partition",
    )
    parser.add_argument(
        "--iou-thr",
        type=float,
        default=DEFAULT_MATCH_IOU,
        help="matching IoU threshold for the oracle transform and the FP partition",
    )
    parser.add_argument(
        "--val-conf",
        type=float,
        default=DEFAULT_VAL_CONF,
        help="val confidence floor; keep 0.001 for parity with eval_multIou.py",
    )
    parser.add_argument(
        "--max-det",
        type=int,
        default=DEFAULT_MAX_DET,
        help="max detections per image; keep 300 for RT-DETR parity",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="JSON report path (default: runs/headroom/<run_tag>_<mode>.json)",
    )
    parser.add_argument(
        "--project",
        default="runs/headroom/_val",
        help="ultralytics save_dir project (empty dirs only: plots/save are disabled)",
    )
    parser.add_argument(
        "--baseline-json",
        default=None,
        help=(
            "eval_multIou.py JSON used for the sanity contract / oracle delta "
            "(default: auto-detect runs/multi_iou/handoff_baseline.json)"
        ),
    )
    parser.add_argument(
        "--sanity-tol",
        type=float,
        default=SANITY_TOL,
        help="tolerance on 4-decimal mAP when comparing mode=none to the baseline",
    )
    return parser.parse_args(argv)


def run_tag(weights: Path) -> str:
    """Derive the short run tag used in the default output path."""
    tag = weights.parent.parent.name or weights.parent.name or weights.stem
    return tag or "run"


def default_out(weights: Path, mode: str) -> Path:
    """Default JSON report path: ``runs/headroom/<tag>_<mode>.json``."""
    return DEFAULT_OUT_DIR / f"{run_tag(weights)}_{mode}.json"


def load_baseline(path: Path | None) -> tuple[dict | None, Path]:
    """Load the eval_multIou baseline JSON when it exists.

    Args:
        path: Explicit path from ``--baseline-json``; ``None`` auto-detects the handoff baseline.

    Returns:
        tuple: ``(payload_or_None, resolved_path)``.
    """
    resolved = Path(path) if path else DEFAULT_BASELINE_JSON
    if not resolved.is_file():
        return None, resolved
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None, resolved
    return payload, resolved


def mAP_deltas(mAP: dict, baseline: dict) -> dict:
    """Per-IoU delta between this run and a baseline ``mAP`` dict.

    Args:
        mAP: This run's rounded ``mAP`` dict.
        baseline: Baseline JSON payload containing a ``mAP`` dict.

    Returns:
        dict: ``{iou_key: delta}`` for the keys present in both, plus ``mean_delta``.
    """
    deltas: dict = {}
    base_map = baseline.get("mAP") or {}
    for key in (f"mAP@{t}" for t in IOUS):
        if key in base_map and mAP.get(key) is not None:
            deltas[key] = round(float(mAP[key]) - float(base_map[key]), 4)
    base_mean = base_map.get("mean")
    if base_mean is None:
        values = [base_map[k] for k in (f"mAP@{t}" for t in IOUS) if k in base_map]
        base_mean = sum(values) / len(values) if values else None
    if base_mean is not None and mAP.get("mean") is not None:
        deltas["mean_delta"] = round(float(mAP["mean"]) - float(base_mean), 4)
    return deltas


def build_sanity(
    mode: str, mAP: dict | None, weights: Path, baseline: dict | None,
    baseline_path: Path, tolerance: float,
) -> dict:
    """Build the self-check block without claiming anything that was not measured.

    Args:
        mode: Probe mode.
        mAP: This run's rounded ``mAP`` dict (``None`` in fp mode).
        weights: Weights path of this run.
        baseline: Loaded baseline payload or ``None``.
        baseline_path: Path that was checked for the baseline.
        tolerance: Accepted absolute delta on rounded mAP values.

    Returns:
        dict: Sanity block for the JSON report.
    """
    contract = (
        "mode=none must reproduce scripts/eval_multIou.py on the same weights "
        "(tianchi18_mosaic1280 best.pt: mean 0.6349, mAP@0.5 0.5917; tianchi18_full "
        "best.pt: mAP@0.5 0.541); mode=box and mode=cls mAP must be >= mode=none"
    )
    sanity = {
        "contract": contract,
        "mode": mode,
        "eval_multIou_reproduced": None,
        "baseline_json": str(baseline_path),
        "baseline_loaded": baseline is not None,
        "tolerance": tolerance,
        "per_iou_delta": None,
        "max_abs_delta": None,
        "note": "",
    }

    if mode != "none":
        sanity["note"] = (
            "only mode=none self-checks against eval_multIou; run mode=none with the same "
            "weights and compare to verify the contract and the box/cls monotonicity"
        )
        return sanity

    if baseline is None:
        sanity["note"] = (
            "no eval_multIou baseline JSON on disk; the contract was not measured by this run"
        )
        return sanity

    if mAP is None:
        sanity["note"] = "no mAP available from this run; contract not measured"
        return sanity

    deltas = mAP_deltas(mAP, baseline)
    per_iou = {k: v for k, v in deltas.items() if k != "mean_delta"}
    sanity["per_iou_delta"] = per_iou
    sanity["max_abs_delta"] = max((abs(v) for v in per_iou.values()), default=None)
    sanity["eval_multIou_reproduced"] = (
        bool(sanity["max_abs_delta"] <= tolerance) if sanity["max_abs_delta"] is not None else None
    )
    same_weights = str(Path(baseline.get("weights", "")).resolve()) == str(weights.resolve())
    sanity["baseline_weights"] = baseline.get("weights")
    sanity["same_weights"] = same_weights
    sanity["note"] = (
        "measured against the stored baseline"
        if sanity["eval_multIou_reproduced"]
        else "measured against the stored baseline; MAX DELTA EXCEEDS TOLERANCE -- do not trust "
        "the oracle numbers until the pipeline mismatch is understood"
    )
    if not same_weights:
        sanity["note"] += " (baseline weights differ from --weights)"
    return sanity


def env_block(device: str) -> dict:
    """Collect versions and device information for the report."""
    import torch
    import ultralytics

    try:
        cuda = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception:  # device name is informational only
        cuda = None
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "ultralytics": ultralytics.__version__,
        "cuda": cuda,
        "device_arg": device,
    }


def build_report(
    args: argparse.Namespace,
    model,
    validator,
    baseline: dict | None,
    baseline_path: Path,
) -> dict:
    """Assemble the JSON report from the finished validator.

    Args:
        args: Parsed CLI arguments.
        model: The loaded ``ultralytics.RTDETR`` wrapper (for class names).
        validator: The probe validator instance after ``__call__``.
        baseline: Loaded baseline payload or ``None``.
        baseline_path: Path that was checked for the baseline.

    Returns:
        dict: JSON-serialisable report.
    """
    import numpy as np

    weights = Path(args.weights)
    metrics = validator.metrics
    box = metrics.box
    all_ap = np.asarray(box.all_ap, dtype=float)
    if all_ap.size:
        all_ap = all_ap.reshape(-1, len(IOUS))
    class_idx = [int(c) for c in np.asarray(box.ap_class_index).reshape(-1)]
    nt = np.asarray(metrics.nt_per_class).reshape(-1)
    names = dict(validator.names or getattr(model, "names", {}))

    mAP: dict = {}
    if all_ap.size:
        for i, iou in enumerate(IOUS):
            mAP[f"mAP@{iou}"] = round(float(all_ap[:, i].mean()), 4)
        mAP["mean"] = round(float(all_ap.mean()), 4)

    per_class = []
    for row, c in enumerate(class_idx):
        per_class.append(
            {
                "id": c,
                "name": names.get(c, str(c)),
                "n_inst": int(nt[c]) if c < nt.size else 0,
                "AP@0.1": round(float(all_ap[row, 0]), 4),
                "AP@0.3": round(float(all_ap[row, 1]), 4),
                "AP@0.5": round(float(all_ap[row, 2]), 4),
                "P@maxF1": round(float(box.p[row]), 4),
                "R@maxF1": round(float(box.r[row]), 4),
            }
        )

    sanity = build_sanity(
        args.mode, mAP if args.mode != "fp" else None, weights, baseline,
        baseline_path, args.sanity_tol,
    )
    delta = None
    if baseline is not None and mAP and args.mode != "fp":
        delta = {
            "baseline_json": str(baseline_path),
            "baseline_weights": baseline.get("weights"),
            "same_weights": str(Path(baseline.get("weights", "")).resolve())
            == str(weights.resolve()),
            **mAP_deltas(mAP, baseline),
            "note": "positive = this mode scores higher than the stored baseline "
            "(P0.2: box-oracle mean gain >= +0.01 triggers D1)",
        }

    report = {
        "weights": str(weights.resolve()),
        "data": str(Path(args.data).resolve()),
        "mode": args.mode,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "ious": list(IOUS),
        "val_conf": args.val_conf,
        "max_det": args.max_det,
        "n_classes_present": len(class_idx),
        "n_instances_total": int(nt.sum()),
        "mAP": None if args.mode == "fp" else mAP,
        "per_class": [] if args.mode == "fp" else per_class,
        "sanity": sanity,
        "delta_vs_baseline": delta,
        "matching": {
            "protocol": "one-to-one greedy assignment, candidate pairs taken in descending "
            "IoU order (same rule as ultralytics match_predictions)",
            "iou_thr": args.iou_thr,
            "box_cls_class_aware": False,
            "fp_tau": args.tau,
            "fp_class_aware_tp_iou_0.5": True,
            "box_oracle_note": (
                "iou_thr >= 0.5 rewrites only predictions that are already TP at every "
                "reported IoU, so mode=box equals mode=none by construction; use "
                "--iou-thr 0.3/0.1 for the near-miss localization oracle"
                if args.mode == "box" and args.iou_thr >= DEFAULT_MATCH_IOU
                else None
            ),
        },
        "env": env_block(args.device),
    }
    if args.mode == "fp":
        report["mAP_note"] = "not reported in fp mode (spec: fp reports the FP partition only)"
        report["fp"] = validator.fp_report()
    return report


def print_report(report: dict) -> None:
    """Print a compact human-readable summary of the report.

    Args:
        report: Report dict returned by :func:`build_report`.
    """
    mode = report["mode"]
    print("\n================ oracle probe ================")
    print(f"weights : {report['weights']}")
    print(
        f"data    : {report['data']}  (imgsz={report['imgsz']}, conf={report['val_conf']}, "
        f"max_det={report['max_det']}, square letterbox)"
    )
    print(f"mode    : {mode}   ious={report['ious']}")
    print(f"instances: {report['n_instances_total']} across {report['n_classes_present']} classes")

    if mode != "fp":
        for key in (f"mAP@{t}" for t in IOUS):
            value = report["mAP"].get(key)
            print(f"  {key:10s} = {value:.4f}" if value is not None else f"  {key:10s} = n/a")
        mean = report["mAP"].get("mean")
        print(f"  {'mean':10s} = {mean:.4f}" if mean is not None else "  mean       = n/a")

        delta = report.get("delta_vs_baseline")
        if delta:
            print(
                f"\n  baseline: {delta['baseline_json']} "
                f"(same_weights={delta['same_weights']})"
            )
            for key in (f"mAP@{t}" for t in IOUS):
                if key in delta:
                    print(f"    {key:10s} delta = {delta[key]:+.4f}")
            if "mean_delta" in delta:
                print(f"    {'mean':10s} delta = {delta['mean_delta']:+.4f}  (P0.2 gate: >= +0.01)")

        if report["matching"].get("box_oracle_note"):
            print(f"\n  NOTE: {report['matching']['box_oracle_note']}")

        print("\n  class            n_inst   AP@0.1  AP@0.3  AP@0.5   P@F1   R@F1")
        for row in report["per_class"]:
            print(
                f"  {row['name']:<16s} {row['n_inst']:>6d}  {row['AP@0.1']:>6.3f} "
                f"{row['AP@0.3']:>6.3f} {row['AP@0.5']:>6.3f} {row['P@maxF1']:>6.3f} "
                f"{row['R@maxF1']:>6.3f}"
            )

    if mode == "fp":
        fp = report["fp"]
        cats = fp["categories"]
        print(f"\ntau={fp['tau']}  iou_thr={fp['iou_thr']}")
        print(
            f"images: {fp['n_images']} (defect {fp['n_images_defect']} / clean "
            f"{fp['n_images_clean']})"
        )
        print(
            f"predictions above tau: {fp['n_preds_above_tau']}   TP: {fp['n_tp_above_tau']}   "
            f"FP: {fp['n_fp']}"
        )
        print(f"\n  {'category':<34s} {'count':>7s}  {'share_of_fp':>11s}  {'share_of_preds':>14s}")
        for key in ("background_clean_image", "duplicate_or_neighbour", "wrong_class"):
            c = cats[key]
            print(
                f"  {key:<34s} {c['count']:>7d}  {_fmt_share(c['share_of_fp']):>11s}  "
                f"{_fmt_share(c['share_of_preds_above_tau']):>14s}"
            )
        subs = cats["duplicate_or_neighbour"]["sub_reasons"]
        for key, c in subs.items():
            print(
                f"    of which {key:<30s} {c['count']:>7d}  {_fmt_share(c['share_of_fp']):>11s}"
            )
        print(
            f"\n  partition check: {fp['partition_check']['sum_of_categories']} == "
            f"{fp['n_fp']} [{'OK' if fp['partition_check']['equals_n_fp'] else 'FAIL'}]"
        )
        print("  P0.4 gates: (a) background > 40% -> abstention axis has headroom; "
              "(b) high -> inference-level dedup; (c) high -> trust P0.1 instead")

    if mode == "none":
        print(
            "\nCONTRACT: mode=none must reproduce scripts/eval_multIou.py exactly on the same"
        )
        print(
            "  weights (mosaic1280 best.pt: mean 0.6349 / mAP@0.5 0.5917; tianchi18_full"
        )
        print("  best.pt: mAP@0.5 0.541). Any mismatch means this script diverged.")
        sanity = report["sanity"]
        if sanity.get("eval_multIou_reproduced") is None:
            print(f"  measured here: NOT MEASURED ({sanity['note']})")
        else:
            verdict = "PASS" if sanity["eval_multIou_reproduced"] else "FAIL"
            print(
                f"  measured here: {verdict}  max_abs_delta={sanity['max_abs_delta']} "
                f"(tol={sanity['tolerance']}) vs {sanity['baseline_json']}"
            )
    print("=============================================\n")


def _fmt_share(value) -> str:
    """Format a share/None as a short percentage string."""
    return "n/a" if value is None else f"{value * 100:.2f}%"


def main(argv=None) -> int:
    """Run one oracle probe and write its JSON report.

    Args:
        argv: Optional argument list (defaults to ``sys.argv[1:]``).

    Returns:
        int: Process exit code (0 = success, 2 = missing input / pipeline failure).
    """
    args = parse_args(argv)
    weights = Path(args.weights)
    out_path = Path(args.out) if args.out else default_out(weights, args.mode)

    if not weights.is_file():
        print(f"ERROR: weights not found: {weights}", file=sys.stderr)
        print('sanity: "not run: no weights"', file=sys.stderr)
        return 2

    import ultralytics  # noqa: F401 - imported for the env block / version check
    from ultralytics import RTDETR
    from ultralytics.utils.metrics import Metric

    patch_metric_map75(Metric)
    validator_cls = build_probe_validator(args.mode, args.tau, args.iou_thr)
    model = RTDETR(str(weights))

    val_kwargs = {
        "data": args.data,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "workers": args.workers,
        "conf": args.val_conf,
        "max_det": args.max_det,
        "plots": False,
        "save": False,
        "verbose": False,
        "project": args.project,
        "name": f"oracle_{args.mode}",
    }

    print(f"[oracle] mode={args.mode} weights={weights} data={args.data}")
    if "validator" in inspect.signature(model.val).parameters:
        # 8.4.91 Model.val(): the `validator` argument is a CLASS, instantiated as
        # validator(args=..., _callbacks=...). Passing an instance would break.
        model.val(validator=validator_cls, **val_kwargs)
    else:
        print("[oracle] Model.val() has no `validator=` parameter; instantiating directly")
        save_dir = Path(args.project) / f"oracle_{args.mode}"
        validator = validator_cls(
            args={**val_kwargs, "model": str(weights), "mode": "val"}, save_dir=save_dir
        )
        validator(model=model.model)
        model.metrics = validator.metrics

    validator = validator_cls.last_instance
    if validator is None or getattr(validator, "metrics", None) is None:
        print("ERROR: validator produced no metrics", file=sys.stderr)
        return 2

    baseline, baseline_path = load_baseline(args.baseline_json)
    report = build_report(args, model, validator, baseline, baseline_path)
    print_report(report)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"report written to {out_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
