"""
Reward-injected RT-DETR training (Layer 1 of the three-tier plan).

Hooks the 7-dimension reward into Ultralytics' RT-DETR training by subclassing
``RTDETRDetectionLoss`` and appending a differentiable ``loss_reward`` term
computed on the Hungarian-matched predictions of the final decoder layer.

Because ``RTDETRDetectionModel.loss()`` sums *every* entry of the criterion's
returned dict, injecting a single ``loss_reward`` key folds the reward into the
supervised objective with zero changes to Ultralytics' training loop:

    L_total = L_giou + L_class + L_bbox + beta * (-R_scalar)

The reward is computed on the *same* Hungarian matching the supervised loss
uses (same ``matcher``, same inputs), so it rewards exactly the detections that
supervision already selected — the reward's only job is to add the critical-
class recall signal that CE+GIoU+denoising cannot express.

RT-DETR's head emits per-class sigmoid logits (focal/vfl), so the reward
computer is constructed with ``head_type="sigmoid"``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set

import torch
from ultralytics.models.utils.loss import RTDETRDetectionLoss
from ultralytics.nn.tasks import RTDETRDetectionModel

from backend.training.dimension_rewards import (
    DIMENSION_NAMES,
    DimensionRewardComputer,
)


class RewardRTDETRDetectionLoss(RTDETRDetectionLoss):
    """
    RT-DETR criterion that appends a differentiable scalar-reward loss term.

    Args:
        nc: Number of defect classes.
        reward_computer: A :class:`DimensionRewardComputer` (``head_type="sigmoid"``).
        beta: Weight of the reward term relative to the supervised losses.
        include_misses: When True, missed critical-class GTs (matched query below
            ``miss_threshold`` on the critical set) are also fed into the reward
            via their best spare query, so D04/D06/D07 see the miss.
        miss_threshold: Critical-set probability below which a GT counts as missed.
        loss_gain: Optional per-term gains forwarded to the RT-DETR criterion.
    """

    def __init__(
        self,
        nc: int,
        reward_computer: DimensionRewardComputer,
        beta: float = 0.5,
        include_misses: bool = False,
        miss_threshold: float = 0.5,
        loss_gain: Optional[Dict[str, float]] = None,
    ):
        super().__init__(nc=nc, use_vfl=True, loss_gain=loss_gain)
        self.reward_computer = reward_computer
        self.beta = beta
        self.include_misses = include_misses
        self.miss_threshold = miss_threshold
        # Detached per-step history for logging/visualization (appended once per
        # training batch; empty when beta=0 or when no targets were matched).
        self.loss_reward_history: List[float] = []
        self.reward_history: List[float] = []
        self.match_count_history: List[int] = []
        self.miss_count_history: List[int] = []

    def forward(
        self,
        preds: tuple,
        batch: Dict[str, Any],
        dn_bboxes: Optional[torch.Tensor] = None,
        dn_scores: Optional[torch.Tensor] = None,
        dn_meta: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute the supervised RT-DETR loss plus a differentiable reward loss.

        Args:
            preds: ``(dec_bboxes, dec_scores)`` of shape ``(L, B, N, 4)`` /
                ``(L, B, N, C)`` (already split from any denoising queries by
                the caller, as in ``RTDETRDetectionModel.loss``).
            batch: Target dict with flat ``cls``/``bboxes`` and ``gt_groups``.
            dn_bboxes/dn_scores/dn_meta: Denoising tensors, passed through.

        Returns:
            Loss dict including the extra ``loss_reward`` key.
        """
        loss = super().forward(
            preds,
            batch,
            dn_bboxes=dn_bboxes,
            dn_scores=dn_scores,
            dn_meta=dn_meta,
        )
        dec_bboxes, dec_scores = preds
        loss["loss_reward"] = self._reward_loss(dec_bboxes[-1], dec_scores[-1], batch)
        return loss

    def _reward_loss(
        self,
        pred_bboxes: torch.Tensor,
        pred_scores: torch.Tensor,
        batch: Dict[str, Any],
    ) -> torch.Tensor:
        """
        Compute ``-beta * mean(R_k)`` on the final-layer predictions.

        By default only Hungarian-matched pairs are fed to the reward. With
        ``include_misses``, a critical-class GT whose matched query scores below
        ``miss_threshold`` on the critical set is additionally fed via its best
        *spare* query, so the miss drives D04/D06/D07 and a free query gets the
        recall gradient.

        Args:
            pred_bboxes: ``(B, N, 4)`` final decoder layer boxes.
            pred_scores: ``(B, N, C)`` final decoder layer class logits.
            batch: Target dict with flat ``cls``/``bboxes`` and ``gt_groups``.

        Returns:
            Differentiable scalar reward loss on ``pred_bboxes.device``.
        """
        # Same matcher (and inputs) the supervised loss already used. Note the
        # RT-DETR Hungarian matcher assigns every GT to a query (N >= num_gt), so
        # a "missed" defect is one whose matched query has low critical-class
        # confidence — not an unmatched GT.
        match_indices = self.matcher(
            pred_bboxes, pred_scores, batch["bboxes"], batch["cls"], batch["gt_groups"]
        )

        pred_logits: List[torch.Tensor] = []
        pred_boxes: List[torch.Tensor] = []
        targets: List[Dict] = []
        n_missed = 0
        gt_groups = batch["gt_groups"]
        gt_offset = 0
        for i, (src, dst) in enumerate(match_indices):
            # Matched pairs: the assigned query/GT pair, as before.
            for k in range(len(src)):
                pred_logits.append(pred_scores[i][src[k]])
                pred_boxes.append(pred_bboxes[i][src[k]])
                targets.append(
                    {
                        "boxes": [batch["bboxes"][dst[k]].tolist()],
                        "labels": [int(batch["cls"][dst[k]])],
                    }
                )
            # Missed critical GTs -> point a spare query at them.
            if self.include_misses:
                gt_to_query = {
                    int(g): int(q)
                    for g, q in zip(dst.tolist(), src.tolist(), strict=True)
                }
                matched = {int(q) for q in src.tolist()}
                for g in range(gt_offset, gt_offset + int(gt_groups[i])):
                    label = int(batch["cls"][g])
                    class_ids = self._critical_ids_for(label)
                    if class_ids is None:
                        continue
                    if (
                        self._p_crit(pred_scores[i][gt_to_query[g]], class_ids)
                        >= self.miss_threshold
                    ):
                        continue
                    spare = self._best_spare_query(i, class_ids, matched, pred_scores)
                    if spare is None:
                        continue
                    pred_logits.append(pred_scores[i][spare])
                    pred_boxes.append(pred_bboxes[i][spare])
                    targets.append(
                        {"boxes": [batch["bboxes"][g].tolist()], "labels": [label]}
                    )
                    n_missed += 1
            gt_offset += int(gt_groups[i])

        if not targets:
            self._record_step(0.0, 0.0, 0, 0)
            return torch.tensor(0.0, device=pred_bboxes.device)

        predictions = {
            "logits": torch.stack(pred_logits),
            "boxes": torch.stack(pred_boxes),
        }
        rewards = self.reward_computer.compute_all(predictions, targets)
        reward_tensor = torch.stack([rewards[name] for name in DIMENSION_NAMES])
        mean_reward = reward_tensor.mean()
        loss_reward = -self.beta * mean_reward
        self._record_step(
            float(loss_reward.detach().item()),
            float(mean_reward.detach().item()),
            len(targets),
            n_missed,
        )
        return loss_reward

    def _critical_ids_for(self, label: int) -> Optional[Set[int]]:
        """Return the critical class set containing ``label``, else None."""
        if label in self.reward_computer.broken_class_ids:
            return self.reward_computer.broken_class_ids
        if label in self.reward_computer.skip_class_ids:
            return self.reward_computer.skip_class_ids
        return None

    def _p_crit(self, logits: torch.Tensor, class_ids: Set[int]) -> float:
        """Detached max sigmoid over ``class_ids`` for a single query's logits."""
        ids = sorted(class_ids)
        probs = torch.sigmoid(logits.detach())
        return float(probs[ids].max().item())

    def _best_spare_query(
        self,
        img: int,
        class_ids: Set[int],
        matched: Set[int],
        pred_scores: torch.Tensor,
    ) -> Optional[int]:
        """
        Return the unmatched query with the highest critical-set probability.

        Args:
            img: Batch image index.
            class_ids: Critical class ids (broken or skip set).
            matched: Query indices already assigned to a GT in this image.
            pred_scores: ``(B, N, C)`` final-layer class logits.

        Returns:
            Query index in ``[0, N)``, or None if no query is spare.
        """
        ids = sorted(class_ids)
        probs = torch.sigmoid(pred_scores[img].detach())  # (N, C)
        p_crit = probs[:, ids].max(dim=1).values  # (N,)
        if len(matched) >= p_crit.numel():
            return None
        mask = torch.ones(p_crit.numel(), dtype=torch.bool, device=p_crit.device)
        for q in matched:
            mask[q] = False
        p_crit = p_crit.masked_fill(~mask, -1.0)
        return int(p_crit.argmax().item())

    def _record_step(
        self,
        loss_reward: float,
        mean_reward: float,
        n_matched: int,
        n_missed: int,
    ) -> None:
        """Record one training step's reward statistics (detached, for plots)."""
        self.loss_reward_history.append(loss_reward)
        self.reward_history.append(mean_reward)
        self.match_count_history.append(n_matched)
        self.miss_count_history.append(n_missed)


def attach_reward_criterion(
    model: RTDETRDetectionModel,
    reward_computer: DimensionRewardComputer,
    beta: float = 0.5,
    include_misses: bool = False,
    miss_threshold: float = 0.5,
) -> RewardRTDETRDetectionLoss:
    """
    Install the reward criterion on an RT-DETR model before ``model.train()``.

    Both ``model.criterion`` and ``model.init_criterion`` are patched so the
    criterion survives Ultralytics' setup path (which may lazily re-init it).

    Args:
        model: Loaded ``RTDETRDetectionModel`` (``YOLO(weights).model``).
        reward_computer: A ``head_type="sigmoid"`` reward computer.
        beta: Reward weight.
        include_misses: Feed missed critical GTs into the reward (see
            :class:`RewardRTDETRDetectionLoss`).
        miss_threshold: Critical-set probability below which a GT counts as missed.

    Returns:
        The installed criterion.
    """
    criterion = RewardRTDETRDetectionLoss(
        nc=model.nc,
        reward_computer=reward_computer,
        beta=beta,
        include_misses=include_misses,
        miss_threshold=miss_threshold,
    )
    model.criterion = criterion
    model.init_criterion = lambda: criterion  # type: ignore[method-assign]
    return criterion
