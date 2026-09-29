"""
Box-loss severity weighting for RT-DETR — Direction B refinement (R2).

The confusion-matrix + threshold-calibration diagnosis showed the critical-class
recall gap is a *localization* problem, not a classification one: thin defects
(broken_warp / broken_spandex / hole) are detected with systematically low
confidence because RT-DETR's varifocal ``gt_score`` is the predicted IoU — i.e.
"low confidence" == "the model self-assesses low localization quality". So
weighting the *classification* arm (``costsensitive.py``, Direction B) cannot
help: there is no classification error to weight.

This module instead up-weights the *box* arm (L1 + GIoU) of each matched query
by the severity of its assigned ground-truth class. Forcing better localization
of critical defects should raise their predicted IoU, and therefore their
confidence, which is the actual lever the diagnosis pointed at.

Monkey-patches ``ultralytics.models.utils.loss.RTDETRDetectionLoss`` so
``RTDETRDetectionModel.init_criterion`` constructs the weighted loss unchanged
(same injection pattern as ``costsensitive.py``).

Usage::

    import ultralytics.models.utils.loss as _detr_loss
    from backend.training.boxweighted import (
        BoxWeightedRTDETRDetectionLoss,
        set_default_box_weights,
    )

    set_default_box_weights(severity_weights)  # shape (nc,)
    _detr_loss.RTDETRDetectionLoss = BoxWeightedRTDETRDetectionLoss
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from ultralytics.models.utils.loss import RTDETRDetectionLoss
from ultralytics.utils.metrics import bbox_iou

# Module-level default so ``BoxWeightedRTDETRDetectionLoss`` can act as a drop-in
# replacement: ``init_criterion`` constructs it with only ``nc`` / ``use_vfl`` and
# no explicit weights, so the weights are resolved from here.
_DEFAULT_BOX_CLASS_WEIGHTS: Optional[torch.Tensor] = None


def set_default_box_weights(weights: "list[float] | torch.Tensor") -> None:
    """Set the module-level severity weights used when none are passed explicitly."""
    global _DEFAULT_BOX_CLASS_WEIGHTS
    _DEFAULT_BOX_CLASS_WEIGHTS = torch.as_tensor(weights, dtype=torch.float32)


class BoxWeightedRTDETRDetectionLoss(RTDETRDetectionLoss):
    """
    RT-DETR detection loss whose box arm (L1 + GIoU) is weighted by the severity
    of the matched ground-truth class.

    Only the box arm changes; the Hungarian matcher, the classification (VFL)
    arm and the (optional) denoising term are inherited unchanged. A matched
    query assigned to a critical class gets ``class_weights[c]`` on its L1/GIoU
    loss, so the model is pushed to localize those defects better and thereby
    predict a higher confidence.
    """

    def __init__(
        self, nc: int = 80, class_weights: Optional[torch.Tensor] = None, **kwargs
    ):
        """Build the upstream loss, then record the per-class box-severity weights."""
        super().__init__(nc=nc, **kwargs)
        weights = (
            class_weights if class_weights is not None else _DEFAULT_BOX_CLASS_WEIGHTS
        )
        self.box_class_weights = (
            torch.as_tensor(weights, dtype=torch.float32)
            if weights is not None
            else None
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
        """Mirror upstream :meth:`_get_loss`, replacing the box arm with a weighted one."""
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
            gt_scores[idx] = bbox_iou(
                pred_bboxes_m.detach(), gt_bboxes_m, xywh=True
            ).squeeze(-1)

        class_loss = self._get_loss_class(
            pred_scores, targets, gt_scores, len(gt_bboxes), postfix
        )
        box_loss = self._weighted_loss_bbox(
            pred_bboxes_m, gt_bboxes_m, gt_cls[gt_idx], len(gt_bboxes), postfix
        )
        return {**class_loss, **box_loss}

    def _weighted_loss_bbox(
        self,
        pred_bboxes: torch.Tensor,
        gt_bboxes: torch.Tensor,
        matched_cls: torch.Tensor,
        num_gts: int,
        postfix: str = "",
    ) -> dict:
        """Weighted L1 + GIoU box loss over the matched queries, scaled by class severity."""
        name_bbox = f"loss_bbox{postfix}"
        name_giou = f"loss_giou{postfix}"
        if num_gts == 0:
            return {
                name_bbox: torch.tensor(0.0, device=self.device),
                name_giou: torch.tensor(0.0, device=self.device),
            }

        if self.box_class_weights is None:
            weights = torch.ones_like(matched_cls, dtype=pred_bboxes.dtype)
        else:
            # Criterion stays on CPU while predictions live on CUDA: move lazily.
            weights = self.box_class_weights.to(pred_bboxes.device)[matched_cls].to(
                pred_bboxes.dtype
            )

        l1 = F.l1_loss(pred_bboxes, gt_bboxes, reduction="none").sum(-1)  # (M,)
        loss_bbox = self.loss_gain["bbox"] * (l1 * weights).sum() / num_gts

        giou = (1.0 - bbox_iou(pred_bboxes, gt_bboxes, xywh=True, GIoU=True)).squeeze(
            -1
        )  # (M,)
        loss_giou = self.loss_gain["giou"] * (giou * weights).sum() / num_gts

        return {name_bbox: loss_bbox.squeeze(), name_giou: loss_giou.squeeze()}
