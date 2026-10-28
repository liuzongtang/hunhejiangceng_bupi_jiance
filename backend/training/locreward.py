"""
D1: low-IoU localisation reward for RT-DETR fine-tuning (additive only).

Hypothesis (handoff section 8)
    The competition metric is mean(mAP@IoU0.1, mAP@IoU0.3, mAP@IoU0.5), and the
    AP@0.1 -> AP@0.5 drop of the current best detector (mean 0.074, up to 0.26 on
    thin/linear classes) is a localisation gap, not a classification one. A reward
    that acts *only* on the box arm, computed at the competition's IoU scales
    (0.1 / 0.3 instead of 0.5), may recover part of that drop without touching the
    classification arm - and therefore without triggering the precision/recall
    zero-sum that killed the reweighting arms.

Pre-registered criterion (handoff section 8)
    ``mean`` gain >= +0.005, with the gain coming mainly from AP@0.3/AP@0.5 on the
    thin-frame classes (loose_warp / size_stain / broken_warp / hanging_warp);
    ``crit_recall`` must not drop. Judged with 3 seeds (42 / 123 / 2024).

Failure branch (handoff section 8)
    If the signature matches R3 (mean down, crit_recall up) -> localisation cannot
    be pushed outwards on this representation either; record the negative. If both
    drop -> handoff section 11 F2 (optimisation instability: halve the
    intervention weight / LR once, then UW gradient balancing, then abandon).

Difference from the failed R3/NWD arm
    NWD *replaced* the metric: the whole box arm (and the varifocal ``gt_score``
    target) was swapped for a Gaussian-Wasserstein similarity, it was zero-sum and
    cost mean -0.015. D1 *adds* a localisation reward aligned to IoU 0.1/0.3 and
    leaves the supervised loss untouched: GIoU, L1, the VFL classification term,
    the Hungarian matcher and the denoising term all keep their upstream values
    and gains. The new term is

        L_total = L_giou + L_class + L_bbox + L_loc
        L_loc   = -beta * mean_i soft_iou_i

    so at ``beta == 0`` the criterion is an exact uniform control (the added key is
    identically zero, and the matcher consumes no RNG).

Exact soft-IoU formula used here
    For every Hungarian-matched (query, GT) pair ``i``::

        iou_i     = differentiable_iou(pred_i, gt_i)        # in [0, 1]
        soft_iou_i = (1 / K) * sum_k min(1, iou_i / t_k)    # t_k in {0.1, 0.3}
                   = 0.5 * (min(1, iou_i / 0.1) + min(1, iou_i / 0.3))

    and the loss is ``-beta * mean_i soft_iou_i``. Each saturating term contributes
    a gradient of ``1 / t_k`` while ``iou_i < t_k`` and zero above it, so the
    reward concentrates capacity on the low-IoU regime the competition scores,
    and stops pushing once a box is already good at that scale (no incentive to
    over-fit precise boundaries at IoU 0.5+).

    RT-DETR emits normalised ``cxcywh`` boxes; they are converted to the
    top-left ``xywh`` convention expected by
    :func:`backend.training.dimension_rewards.differentiable_iou` (so the reused
    helper computes a true IoU). The convex ``maximum`` / ``minimum`` /
    ``clamp(min=0)`` path makes the term differentiable w.r.t. the predicted
    boxes; as with any hard-clamped IoU the gradient is exactly zero for a pair
    with no overlap at all (matched pairs in a warmed-up detector overlap).

The term is added to the criterion's output dict under ``loss_loc``; Ultralytics'
``RTDETRDetectionModel.loss`` sums every entry of that dict, which folds it into
the supervised objective with no changes to the training loop. As with
``reward_rtdetr.py``, :func:`attach_loc_criterion` must be called from an
``on_train_start`` callback, because the trainer rebuilds its model inside
``_setup_train`` and an attachment made before ``model.train()`` is discarded.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from ultralytics.models.utils.loss import RTDETRDetectionLoss
from ultralytics.nn.tasks import RTDETRDetectionModel

from backend.training.dimension_rewards import differentiable_iou

#: Competition IoU scales this reward saturates at (handoff section 8).
DEFAULT_IOU_SCALES: Tuple[float, ...] = (0.1, 0.3)


def _cxcywh_to_xywh(boxes: torch.Tensor) -> torch.Tensor:
    """
    Convert normalised ``cxcywh`` boxes to top-left ``xywh`` boxes.

    Args:
        boxes: Tensor ``(..., 4)`` in ``(cx, cy, w, h)`` (RT-DETR convention).

    Returns:
        Tensor ``(..., 4)`` in ``(x1, y1, w, h)``; differentiable w.r.t. ``boxes``.
    """
    cx, cy, w, h = boxes.unbind(-1)
    x1 = cx - w / 2.0
    y1 = cy - h / 2.0
    return torch.stack((x1, y1, w, h), dim=-1)


def soft_iou_reward(
    pred_boxes: torch.Tensor,
    gt_boxes: torch.Tensor,
    iou_scales: Sequence[float] = DEFAULT_IOU_SCALES,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Differentiable, saturating soft-IoU at the competition's IoU scales.

    ``soft = mean_k min(1, iou / t_k)``. With the default scales this is
    ``0.5 * (min(1, iou / 0.1) + min(1, iou / 0.3))``.

    Args:
        pred_boxes: Predicted boxes ``(M, 4)`` in normalised ``cxcywh``.
        gt_boxes: Ground-truth boxes ``(M, 4)`` in normalised ``cxcywh``.
        iou_scales: Thresholds ``t_k`` at which each saturating term reaches 1.
        eps: Numerical-stability term forwarded to ``differentiable_iou``.

    Returns:
        Tuple ``(soft_iou, raw_iou)`` of shape ``(M,)``. ``soft_iou`` is
        differentiable w.r.t. ``pred_boxes``; ``raw_iou`` is the unsaturating
        base IoU and is only used for detached logging.
    """
    pred_tl = _cxcywh_to_xywh(pred_boxes)
    gt_tl = _cxcywh_to_xywh(gt_boxes)
    # ``differentiable_iou`` is 2-D only; flatten any batch/query leading dims.
    shape = pred_tl.shape[:-1]
    iou = differentiable_iou(
        pred_tl.reshape(-1, 4), gt_tl.reshape(-1, 4), eps=eps
    ).reshape(shape)
    iou = iou.clamp(0.0, 1.0)
    scales = [float(t) for t in iou_scales if float(t) > 0.0]
    if not scales:
        raise ValueError("iou_scales must contain at least one positive threshold")
    terms = [torch.clamp(iou / t, max=1.0) for t in scales]
    soft = torch.stack(terms, dim=-1).mean(dim=-1)
    return soft, iou


class LocRewardRTDETRDetectionLoss(RTDETRDetectionLoss):
    """
    RT-DETR criterion that appends ``-beta * mean(soft_iou)`` on matched queries.

    The supervised losses (GIoU, L1, VFL classification, denoising) are inherited
    unchanged; only one extra key (``loss_loc``) is added to the returned dict.

    Args:
        nc: Number of defect classes.
        beta: Weight of the localisation reward (0 = exact uniform control).
        iou_scales: Saturating IoU thresholds (default ``(0.1, 0.3)``).
        loss_gain: Optional per-term gains forwarded to the RT-DETR criterion.

    Attributes:
        loss_loc_history: Detached per-step ``loss_loc`` values (for CSV/plots).
        soft_iou_history: Detached per-step mean saturating soft-IoU.
        raw_iou_history: Detached per-step mean plain IoU.
        match_count_history: Number of matched pairs behind each step's mean.
    """

    def __init__(
        self,
        nc: int,
        beta: float = 0.1,
        iou_scales: Sequence[float] = DEFAULT_IOU_SCALES,
        loss_gain: Optional[Dict[str, float]] = None,
    ):
        super().__init__(nc=nc, use_vfl=True, loss_gain=loss_gain)
        self.beta = float(beta)
        self.iou_scales = tuple(float(t) for t in iou_scales)
        # Detached per-step history for logging; stats only, gradients never pass
        # through these lists.
        self.loss_loc_history: List[float] = []
        self.soft_iou_history: List[float] = []
        self.raw_iou_history: List[float] = []
        self.match_count_history: List[int] = []

    def forward(
        self,
        preds: tuple,
        batch: Dict[str, Any],
        dn_bboxes: Optional[torch.Tensor] = None,
        dn_scores: Optional[torch.Tensor] = None,
        dn_meta: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute the supervised RT-DETR loss plus the localisation reward.

        Args:
            preds: ``(dec_bboxes, dec_scores)`` of shape ``(L, B, N, 4)`` /
                ``(L, B, N, C)``, already split from denoising queries by the
                caller (as in ``RTDETRDetectionModel.loss``).
            batch: Target dict with flat ``cls`` / ``bboxes`` and ``gt_groups``.
            dn_bboxes/dn_scores/dn_meta: Denoising tensors, passed through.

        Returns:
            Loss dict including the extra ``loss_loc`` key.
        """
        loss = super().forward(
            preds,
            batch,
            dn_bboxes=dn_bboxes,
            dn_scores=dn_scores,
            dn_meta=dn_meta,
        )
        dec_bboxes, dec_scores = preds
        loss["loss_loc"] = self._loc_loss(dec_bboxes[-1], dec_scores[-1], batch)
        return loss

    def _loc_loss(
        self,
        pred_bboxes: torch.Tensor,
        pred_scores: torch.Tensor,
        batch: Dict[str, Any],
    ) -> torch.Tensor:
        """
        Compute ``-beta * mean(soft_iou)`` over the Hungarian-matched pairs.

        Args:
            pred_bboxes: ``(B, N, 4)`` final decoder-layer boxes (normalised cxcywh).
            pred_scores: ``(B, N, C)`` final decoder-layer class logits.
            batch: Target dict with flat ``bboxes`` / ``cls`` and ``gt_groups``.

        Returns:
            Differentiable scalar on ``pred_bboxes.device`` (identically zero when
            ``beta == 0``).
        """
        # Same matcher and inputs the supervised loss already used: the reward
        # must target exactly the queries supervision selected.
        match_indices = self.matcher(
            pred_bboxes, pred_scores, batch["bboxes"], batch["cls"], batch["gt_groups"]
        )

        pred_matched: List[torch.Tensor] = []
        gt_matched: List[torch.Tensor] = []
        for i, (src, dst) in enumerate(match_indices):
            if len(src) == 0:
                continue
            pred_matched.append(pred_bboxes[i][src])
            gt_matched.append(batch["bboxes"][dst])

        if not pred_matched:
            self._record_step(0.0, 0.0, 0.0, 0)
            return pred_bboxes.new_zeros(())

        pred = torch.cat(pred_matched, dim=0)
        gt = torch.cat(gt_matched, dim=0).to(device=pred.device, dtype=pred.dtype)
        soft_iou, raw_iou = soft_iou_reward(pred, gt, self.iou_scales)

        loss_loc = -self.beta * soft_iou.mean()
        # Detach only the logging statistics; the loss tensor keeps its graph.
        self._record_step(
            float(loss_loc.detach().item()),
            float(soft_iou.mean().detach().item()),
            float(raw_iou.mean().detach().item()),
            int(pred.shape[0]),
        )
        return loss_loc

    def _record_step(
        self,
        loss_loc: float,
        mean_soft_iou: float,
        mean_raw_iou: float,
        n_matched: int,
    ) -> None:
        """Append one training step's detached localisation statistics."""
        self.loss_loc_history.append(loss_loc)
        self.soft_iou_history.append(mean_soft_iou)
        self.raw_iou_history.append(mean_raw_iou)
        self.match_count_history.append(n_matched)


def attach_loc_criterion(
    model: RTDETRDetectionModel,
    beta: float = 0.1,
    iou_scales: Sequence[float] = DEFAULT_IOU_SCALES,
    loss_gain: Optional[Dict[str, float]] = None,
) -> LocRewardRTDETRDetectionLoss:
    """
    Install the low-IoU localisation criterion on an RT-DETR model.

    Must be called from an ``on_train_start`` callback: the trainer rebuilds its
    model inside ``_setup_train``, so a criterion attached before
    ``model.train()`` is silently discarded.

    Both ``model.criterion`` and ``model.init_criterion`` are patched so the
    criterion survives Ultralytics' setup path (which may lazily re-init it).

    Args:
        model: The trainer's ``RTDETRDetectionModel`` (unwrapped from DDP).
        beta: Localisation-reward weight (0 = exact uniform control).
        iou_scales: Saturating IoU thresholds.
        loss_gain: Optional per-term gains forwarded to the RT-DETR criterion.

    Returns:
        The installed criterion.
    """
    nc = getattr(model, "nc", None)
    if nc is None:
        nc = model.model[-1].nc
    criterion = LocRewardRTDETRDetectionLoss(
        nc=int(nc),
        beta=beta,
        iou_scales=iou_scales,
        loss_gain=loss_gain,
    )
    model.criterion = criterion
    model.init_criterion = lambda: criterion  # type: ignore[method-assign]
    return criterion
