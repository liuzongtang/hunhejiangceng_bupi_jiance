"""
Normalized Wasserstein Distance (NWD) box loss for RT-DETR — Direction B (R3).

The R2 box-loss *weighting* (``boxweighted.py``) was the sixth negative: it
up-weighted the same IoU-based box arm, but the diagnosis (``critical-class-
miss-not-confusion``) shows the critical thin defects (broken warp / spandex
yarn, small holes) are missed at *low confidence* because RT-DETR's varifocal
``gt_score`` == predicted IoU, and IoU is pathologically sensitive to tiny
localization errors on thin targets (< ~32 px in one dimension).

NWD (Wang et al., "A Normalized Gaussian Wasserstein Distance for Tiny Object
Detection") models a box as a 2D Gaussian N(mu, Sigma) with mu = (cx, cy) and
Sigma = diag(w^2/4, h^2/4), then uses the 2-Wasserstein distance between the two
Gaussians as a *smooth* substitute for IoU::

    W2^2(Na, Nb) = ||cx_a-cx_b||^2 + ||cy_a-cy_b||^2
                   + ||w_a/2 - w_b/2||^2 + ||h_a/2 - h_b/2||^2
    NWD(Na, Nb)  = exp(-sqrt(W2^2) / C)
    L_NWD        = 1 - NWD

For thin defects, a few-pixel centre shift collapses IoU to ~0 (harsh, vanishing
gradient) but keeps NWD smoothly high, so the model gets a usable gradient to
localize thin defects *and* a confidence target (``gt_score``) that stays high
under small offsets. This module therefore replaces BOTH IoU uses in the box arm:

1. the GIoU regression term -> ``1 - NWD`` (kept under the ``loss_giou`` key,
   which ultralytics' RT-DETR trainer hard-codes in ``loss_items``), and
2. the varifocal ``gt_score`` confidence target -> NWD similarity.

Only the box arm changes; the Hungarian matcher, the classification (VFL) arm
and the denoising term are inherited unchanged (same injection pattern as
``costsensitive.py`` / ``boxweighted.py``).

Usage::

    import ultralytics.models.utils.loss as _detr_loss
    from backend.training.nwd import (
        BoxNWDRTDETRDetectionLoss,
        set_default_nwd,
    )

    set_default_nwd(c=0.05, use_gt_score=True)
    _detr_loss.RTDETRDetectionLoss = BoxNWDRTDETRDetectionLoss
"""

from __future__ import annotations

import glob
import math
import os
from typing import Optional

import torch
import torch.nn.functional as F
from ultralytics.models.utils.loss import RTDETRDetectionLoss
from ultralytics.utils.metrics import bbox_iou

# Module-level defaults so ``BoxNWDRTDETRDetectionLoss`` acts as a drop-in
# replacement: ``init_criterion`` constructs it with only ``nc`` / ``use_vfl``,
# so C and the gt_score switch are resolved from here.
_DEFAULT_NWD_C: Optional[float] = None
_DEFAULT_USE_NWD_GT_SCORE: bool = True

#: Fallback C when neither an explicit value nor a label-based estimate exists.
_DEFAULT_C_FALLBACK = 0.05


def set_default_nwd(c: Optional[float] = None, use_gt_score: bool = True) -> None:
    """Set the module-level NWD constant and gt_score switch used when unset."""
    global _DEFAULT_NWD_C, _DEFAULT_USE_NWD_GT_SCORE
    _DEFAULT_NWD_C = c
    _DEFAULT_USE_NWD_GT_SCORE = use_gt_score


def nwd_similarity(
    pred_boxes: torch.Tensor, gt_boxes: torch.Tensor, c: float = 0.05
) -> torch.Tensor:
    """Return NWD similarity in [0, 1] between xywh (normalized) boxes.

    Args:
        pred_boxes: Predicted boxes of shape (..., 4) in xywh normalized to [0, 1].
        gt_boxes: Ground-truth boxes of shape (..., 4), same convention.
        c: Dataset-specific constant (a typical object linear size, in the same
            normalized units). Smaller C makes NWD more sensitive to offsets.

    Returns:
        Tensor of shape ``pred_boxes.shape[:-1]`` with NWD in [0, 1] (1 == perfect).
    """
    p_cwh = torch.cat([pred_boxes[..., :2], pred_boxes[..., 2:] / 2.0], dim=-1)
    g_cwh = torch.cat([gt_boxes[..., :2], gt_boxes[..., 2:] / 2.0], dim=-1)
    w2 = ((p_cwh - g_cwh) ** 2).sum(-1)  # squared 2-Wasserstein distance
    return torch.exp(-torch.sqrt(w2) / c)


def nwd_loss(
    pred_boxes: torch.Tensor, gt_boxes: torch.Tensor, c: float = 0.05
) -> torch.Tensor:
    """Return the NWD box loss ``1 - NWD`` in [0, 1] between xywh boxes."""
    return 1.0 - nwd_similarity(pred_boxes, gt_boxes, c)


def estimate_nwd_c(labels_dir: str) -> float:
    """Estimate the NWD constant C as the mean ``sqrt(w*h)`` over all GT boxes.

    Uses the geometric-mean linear extent (in normalized units) of every ground
    truth box, which is the natural scale of the Gaussian model and keeps thin
    defects (small ``sqrt(w*h)``) on a more sensitive part of the NWD curve than
    large defects. Falls back to ``_DEFAULT_C_FALLBACK`` if no labels are found.
    """
    sizes: list[float] = []
    for path in glob.glob(os.path.join(labels_dir, "*.txt")):
        with open(path, "r", encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                parts = line.split()
                if len(parts) < 5:
                    continue
                w, h = float(parts[3]), float(parts[4])
                if w > 0 and h > 0:
                    sizes.append(math.sqrt(w * h))
    if not sizes:
        return _DEFAULT_C_FALLBACK
    return float(sum(sizes) / len(sizes))


class BoxNWDRTDETRDetectionLoss(RTDETRDetectionLoss):
    """
    RT-DETR detection loss whose box arm uses NWD instead of IoU/GIoU.

    Replaces the GIoU regression term with ``1 - NWD`` (reported under the
    ``loss_giou`` key the trainer hard-codes) and, optionally, the varifocal
    ``gt_score`` confidence target with NWD similarity. The L1 term, Hungarian
    matcher, VFL classification arm and denoising term are unchanged.
    """

    def __init__(
        self,
        nc: int = 80,
        nwd_c: Optional[float] = None,
        use_nwd_gt_score: Optional[bool] = None,
        **kwargs,
    ):
        """Build the upstream loss, then record the NWD constant and gt_score switch."""
        super().__init__(nc=nc, **kwargs)
        self.nwd_c = (
            nwd_c if nwd_c is not None else (_DEFAULT_NWD_C or _DEFAULT_C_FALLBACK)
        )
        self.use_nwd_gt_score = (
            use_nwd_gt_score
            if use_nwd_gt_score is not None
            else _DEFAULT_USE_NWD_GT_SCORE
        )

    def _get_loss(
        self,
        pred_bboxes: torch.Tensor,
        pred_scores: torch.Tensor,
        gt_bboxes: torch.Tensor,
        gt_cls: torch.Tensor,
        gt_groups: list,
        masks: Optional[torch.Tensor] = None,
        gt_mask: Optional[torch.Tensor] = None,
        postfix: str = "",
        match_indices: Optional[list] = None,
    ) -> dict:
        """Mirror upstream :meth:`_get_loss`, swapping IoU for NWD in the box arm."""
        if match_indices is None:
            match_indices = self.matcher(
                pred_bboxes,
                pred_scores,
                gt_bboxes,
                gt_cls,
                gt_groups,
                masks=masks,
                gt_mask=gt_mask,
            )

        idx, gt_idx = self._get_index(match_indices)
        pred_bboxes_m, gt_bboxes_m = pred_bboxes[idx], gt_bboxes[gt_idx]

        bs, nq = pred_scores.shape[:2]
        targets = torch.full(
            (bs, nq), self.nc, device=pred_scores.device, dtype=gt_cls.dtype
        )
        targets[idx] = gt_cls[gt_idx]

        gt_scores = torch.zeros([bs, nq], device=pred_scores.device)
        if len(gt_bboxes):
            if self.use_nwd_gt_score:
                gt_scores[idx] = nwd_similarity(
                    pred_bboxes_m.detach(), gt_bboxes_m, self.nwd_c
                )
            else:
                gt_scores[idx] = bbox_iou(
                    pred_bboxes_m.detach(), gt_bboxes_m, xywh=True
                ).squeeze(-1)

        class_loss = self._get_loss_class(
            pred_scores, targets, gt_scores, len(gt_bboxes), postfix
        )
        box_loss = self._nwd_loss_bbox(
            pred_bboxes_m, gt_bboxes_m, len(gt_bboxes), postfix
        )
        return {**class_loss, **box_loss}

    def _nwd_loss_bbox(
        self,
        pred_bboxes: torch.Tensor,
        gt_bboxes: torch.Tensor,
        num_gts: int,
        postfix: str = "",
    ) -> dict:
        """L1 + NWD box loss over the matched queries (NWD kept under ``loss_giou``)."""
        name_bbox = f"loss_bbox{postfix}"
        name_giou = f"loss_giou{postfix}"  # key fixed by RT-DETR trainer loss_items
        if num_gts == 0:
            return {
                name_bbox: torch.tensor(0.0, device=self.device),
                name_giou: torch.tensor(0.0, device=self.device),
            }

        loss_bbox = (
            self.loss_gain["bbox"]
            * F.l1_loss(pred_bboxes, gt_bboxes, reduction="sum")
            / num_gts
        )
        loss_nwd = nwd_loss(pred_bboxes, gt_bboxes, self.nwd_c)  # (M,)
        loss_giou = self.loss_gain["giou"] * loss_nwd.sum() / num_gts

        return {name_bbox: loss_bbox.squeeze(), name_giou: loss_giou.squeeze()}
