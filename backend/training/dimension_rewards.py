"""
7-Dimension Reward Computation for Hybrid Reward Trainer.

Section 5.2.1 of the project document:

  D01: Localization precision (soft IoU)   R_loc = 2*IoU - 1           [-1, 1]
  D02: Classification (smooth accuracy)    R_cls = 2*p_true - 1         [-1, 1]
  D03: Confidence calibration              R_cal = -|conf - p_true|      [-1, 0]
  D04: Miss detection penalty (soft)       R_miss = -lam * soft_miss
  D05: False positive penalty (soft)       R_fp = -lam * soft_fp
  D06: Broken yarn sensitivity (soft)      R_broken = 3*p_broken - 2
  D07: Skip/weave sensitivity (soft)       R_skip = 3*p_skip - 2

Every reward is differentiable with respect to the model's ``logits`` and
``boxes``, so the scalar/consistency terms in ``loss_functions.total_loss``
backpropagate into the model itself (not just the learnable ``dim_weights``).
Hard decisions (``argmax``, ``.item()``, set membership, hard IoU thresholds)
are replaced with softmax probabilities and sigmoid soft indicators:

  - D01 uses a vectorized, differentiable IoU on the predicted box.
  - D02 uses ``softmax(logits)`` mass on the ground-truth class (smooth acc).
  - D03 uses the gap between model confidence and soft accuracy.
  - D04/D05 use ``sigmoid((threshold - IoU) / tau)`` as a soft miss/fp indicator.
  - D06/D07 use the softmax mass on the critical class set instead of the
    predicted-class set.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Set

import torch

from backend.training.dataset import (
    TIANCHI_BROKEN_CLASS_IDS,
    TIANCHI_SKIP_CLASS_IDS,
)

# Dimension indices
DIM_LOC = 0
DIM_CLS = 1
DIM_CAL = 2
DIM_MISS = 3
DIM_FP = 4
DIM_BROKEN = 5
DIM_SKIP = 6

NUM_DIMENSIONS = 7

DIMENSION_NAMES = [
    "loc",  # D01
    "cls",  # D02
    "cal",  # D03
    "miss",  # D04
    "fp",  # D05
    "broken",  # D06
    "skip",  # D07
]


def differentiable_iou(
    pred_boxes: torch.Tensor, gt_boxes: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    """
    Vectorized, differentiable IoU for boxes in ``[x, y, w, h]`` format.

    Args:
        pred_boxes: Predicted boxes ``(N, 4)``.
        gt_boxes: Ground-truth boxes ``(N, 4)``.
        eps: Numerical stability term for the union.

    Returns:
        IoU tensor ``(N,)``, differentiable w.r.t. ``pred_boxes``.
    """
    px1, py1, pw, ph = pred_boxes.unbind(1)
    gx1, gy1, gw, gh = gt_boxes.unbind(1)
    px2, py2 = px1 + pw, py1 + ph
    gx2, gy2 = gx1 + gw, gy1 + gh

    ix1 = torch.maximum(px1, gx1)
    iy1 = torch.maximum(py1, gy1)
    ix2 = torch.minimum(px2, gx2)
    iy2 = torch.minimum(py2, gy2)

    inter = torch.clamp(ix2 - ix1, min=0.0) * torch.clamp(iy2 - iy1, min=0.0)
    union = pw * ph + gw * gh - inter
    return inter / (union + eps)


class DimensionRewardComputer:
    """
    Computes the 7-dimensional reward vector for a batch of predictions.

    Rewards are differentiable with respect to the model's ``logits`` and
    ``boxes`` so the scalar/consistency loss terms can steer the model. The
    single-box detector aligns prediction ``i`` with ground-truth ``i`` (the
    ``defect_collate_fn`` keeps one dominant box/label per image).
    """

    def __init__(
        self,
        lambda_miss: float = 1.5,
        lambda_fp: float = 1.0,
        iou_threshold: float = 0.5,
        tau: float = 0.1,
        head_type: str = "softmax",
        broken_class_ids: Optional[Set[int]] = None,
        skip_class_ids: Optional[Set[int]] = None,
    ):
        self.lambda_miss = lambda_miss
        self.lambda_fp = lambda_fp
        self.iou_threshold = iou_threshold
        self.tau = tau
        # Classification head normalization: "softmax" (single-label head, the
        # training `SimpleDefectDetector`) or "sigmoid" (RT-DETR's per-class
        # focal/vfl head).
        self.head_type = head_type
        # Critical class ids for the D06/D07 sensitivity dimensions.
        self.broken_class_ids = (
            broken_class_ids
            if broken_class_ids is not None
            else TIANCHI_BROKEN_CLASS_IDS
        )
        self.skip_class_ids = (
            skip_class_ids if skip_class_ids is not None else TIANCHI_SKIP_CLASS_IDS
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _device(self, predictions: Dict[str, torch.Tensor]) -> torch.device:
        """Infer the device of the predictions (CPU if none is a tensor)."""
        for key in ("logits", "boxes", "confidences", "classes"):
            val = predictions.get(key)
            if torch.is_tensor(val) and val.numel() > 0:
                return val.device
        return torch.device("cpu")

    @staticmethod
    def _as_float(x, device: torch.device) -> torch.Tensor:
        """Coerce a tensor / numpy array / list to a float tensor on ``device``."""
        if torch.is_tensor(x):
            return x.float()
        return torch.as_tensor(x, dtype=torch.float32, device=device)

    def _class_probs(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Class probabilities from head logits.

        ``softmax`` for a single-label head, ``sigmoid`` for RT-DETR's per-class
        focal/vfl head (where each class is an independent binary prediction).
        """
        if self.head_type == "sigmoid":
            return torch.sigmoid(logits)
        return torch.softmax(logits, dim=-1)

    def _aligned_boxes(
        self, predictions: Dict[str, torch.Tensor], targets: List[Dict]
    ) -> Optional[tuple]:
        """Return aligned ``(pred_boxes, gt_boxes)`` as ``(N, 4)`` tensors."""
        pred_boxes = predictions.get("boxes")
        if pred_boxes is None or len(pred_boxes) == 0:
            return None
        gt_list = [list(t["boxes"][0]) for t in targets if t.get("boxes")]
        if not gt_list:
            return None
        device = self._device(predictions)
        pred = self._as_float(pred_boxes, device)
        gt = torch.as_tensor(gt_list, dtype=torch.float32, device=device)
        n = min(pred.shape[0], gt.shape[0])
        return pred[:n], gt[:n]

    def _aligned_logits_labels(
        self, predictions: Dict[str, torch.Tensor], targets: List[Dict]
    ) -> Optional[tuple]:
        """Return aligned ``(logits, labels)`` as ``(N, C)`` / ``(N,)`` tensors."""
        logits = predictions.get("logits")
        if logits is None or logits.numel() == 0:
            return None
        labels_list = [int(t["labels"][0]) for t in targets if t.get("labels")]
        if not labels_list:
            return None
        device = self._device(predictions)
        logits = self._as_float(logits, device)
        labels = torch.as_tensor(labels_list, dtype=torch.long, device=device)
        n = min(logits.shape[0], labels.shape[0])
        return logits[:n], labels[:n]

    def _confidence(
        self, predictions: Dict[str, torch.Tensor], n: int
    ) -> Optional[torch.Tensor]:
        """Return the model confidence for the first ``n`` samples (or None)."""
        logits = predictions.get("logits")
        if logits is not None and logits.numel() > 0:
            device = self._device(predictions)
            probs = self._class_probs(self._as_float(logits, device)[:n])
            return probs.max(dim=-1).values
        conf = predictions.get("confidences")
        if conf is not None and len(conf) > 0:
            return self._as_float(conf, self._device(predictions))[:n]
        return None

    # ------------------------------------------------------------------
    # Full 7-dim vector
    # ------------------------------------------------------------------

    def compute_all(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: List[Dict],
    ) -> Dict[str, torch.Tensor]:
        """
        Compute all 7 dimension rewards.

        Args:
            predictions: Dict with 'boxes' (N,4), 'logits' (N,C), 'classes' (N,),
                'confidences' (N,).
            targets: List of target dicts with 'boxes' (M,4) and 'labels' (M,).

        Returns:
            Dict mapping dimension name -> scalar reward tensor (differentiable).
        """
        return {
            "loc": self.compute_loc_reward(predictions, targets),
            "cls": self.compute_cls_reward(predictions, targets),
            "cal": self.compute_cal_reward(predictions, targets),
            "miss": self.compute_miss_reward(predictions, targets),
            "fp": self.compute_fp_reward(predictions, targets),
            "broken": self.compute_broken_reward(predictions, targets),
            "skip": self.compute_skip_reward(predictions, targets),
        }

    # ------------------------------------------------------------------
    # D01: Localization (soft IoU)
    # ------------------------------------------------------------------

    def compute_loc_reward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: List[Dict],
    ) -> torch.Tensor:
        """
        R_loc = mean(2 * IoU_i - 1)

        Maps IoU in [0, 1] to reward in [-1, 1]. Differentiable via ``differentiable_iou``.
        """
        aligned = self._aligned_boxes(predictions, targets)
        if aligned is None:
            return torch.tensor(-1.0)
        pred, gt = aligned
        iou = differentiable_iou(pred, gt)
        return (2.0 * iou - 1.0).mean()

    # ------------------------------------------------------------------
    # D02: Classification (smooth accuracy)
    # ------------------------------------------------------------------

    def compute_cls_reward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: List[Dict],
    ) -> torch.Tensor:
        """
        R_cls = mean(2 * p_true_i - 1)

        ``p_true`` is the softmax probability assigned to the ground-truth class,
        a differentiable proxy for top-1 accuracy.
        """
        aligned = self._aligned_logits_labels(predictions, targets)
        if aligned is None:
            return torch.tensor(-1.0)
        logits, labels = aligned
        probs = self._class_probs(logits)
        p_true = probs.gather(1, labels.unsqueeze(1)).squeeze(1)
        return (2.0 * p_true - 1.0).mean()

    # ------------------------------------------------------------------
    # D03: Confidence calibration
    # ------------------------------------------------------------------

    def compute_cal_reward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: List[Dict],
    ) -> torch.Tensor:
        """
        R_cal = -mean(|conf_i - p_true_i|)

        Penalizes the gap between the model's confidence and its (soft) accuracy;
        a well-calibrated model has confidence == accuracy. Always <= 0.
        """
        aligned = self._aligned_logits_labels(predictions, targets)
        if aligned is None:
            return torch.tensor(0.0)
        logits, labels = aligned
        probs = self._class_probs(logits)
        p_true = probs.gather(1, labels.unsqueeze(1)).squeeze(1)
        conf = probs.max(dim=-1).values
        return -(conf - p_true).abs().mean()

    # ------------------------------------------------------------------
    # D04: Miss Detection Penalty (soft)
    # ------------------------------------------------------------------

    def compute_miss_reward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: List[Dict],
    ) -> torch.Tensor:
        """
        R_miss = -lambda_miss * mean(soft_miss_i)

        ``soft_miss = sigmoid((threshold - IoU) / tau)`` is a differentiable
        indicator that a ground-truth box was not localized.
        """
        aligned = self._aligned_boxes(predictions, targets)
        if aligned is None:
            return torch.tensor(0.0)
        pred, gt = aligned
        iou = differentiable_iou(pred, gt)
        miss_soft = torch.sigmoid((self.iou_threshold - iou) / self.tau)
        return -self.lambda_miss * miss_soft.mean()

    # ------------------------------------------------------------------
    # D05: False Positive Penalty (soft)
    # ------------------------------------------------------------------

    def compute_fp_reward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: List[Dict],
    ) -> torch.Tensor:
        """
        R_fp = -lambda_fp * mean(conf_i * soft_fp_i)

        ``soft_fp = 1 - sigmoid((IoU - threshold) / tau)`` flags predictions that
        are confident yet poorly localized.
        """
        aligned = self._aligned_boxes(predictions, targets)
        if aligned is None:
            return torch.tensor(0.0)
        pred, gt = aligned
        iou = differentiable_iou(pred, gt)
        matched = torch.sigmoid((iou - self.iou_threshold) / self.tau)
        fp_soft = 1.0 - matched
        conf = self._confidence(predictions, len(pred))
        if conf is not None:
            fp_soft = conf * fp_soft
        return -self.lambda_fp * fp_soft.mean()

    # ------------------------------------------------------------------
    # D06 / D07: Sensitivity (soft)
    # ------------------------------------------------------------------

    def compute_broken_reward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: List[Dict],
    ) -> torch.Tensor:
        """R_broken: +1 when broken-yarn fully detected, -2 when missed (soft)."""
        return self._sensitivity_reward(predictions, targets, self.broken_class_ids)

    def compute_skip_reward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: List[Dict],
    ) -> torch.Tensor:
        """R_skip: +1 when skip/weave fully detected, -2 when missed (soft)."""
        return self._sensitivity_reward(predictions, targets, self.skip_class_ids)

    def _sensitivity_reward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: List[Dict],
        class_ids: Set[int],
    ) -> torch.Tensor:
        """
        Soft sensitivity reward for a critical class set.

        For each sample whose ground-truth label is in ``class_ids``, reward is
        ``3 * p_crit - 2`` where ``p_crit`` is the softmax mass on ``class_ids``
        (1 -> +1 fully detected, 0 -> -2 missed). Non-critical samples contribute 0.
        """
        aligned = self._aligned_logits_labels(predictions, targets)
        if aligned is None:
            return torch.tensor(0.0)
        logits, labels = aligned
        ids = torch.as_tensor(sorted(class_ids), dtype=torch.long, device=logits.device)
        probs = self._class_probs(logits)
        if self.head_type == "sigmoid":
            # Per-class head: max confidence over the critical set (bounded [0,1]).
            p_crit = probs[:, ids].max(dim=1).values
        else:
            p_crit = probs[:, ids].sum(dim=1)
        is_crit = (labels.unsqueeze(1) == ids.unsqueeze(0)).any(dim=1)
        reward = torch.where(is_crit, 3.0 * p_crit - 2.0, torch.zeros_like(p_crit))
        return reward.mean()

    # ------------------------------------------------------------------
    # IoU Computation (non-differentiable helper for diagnostics/tests)
    # ------------------------------------------------------------------

    @staticmethod
    def _box_iou(box1: List[float], box2: List[float]) -> float:
        """
        Compute IoU between two bounding boxes in [x, y, w, h] format.

        Kept as a plain-Python helper for diagnostics and unit tests; training
        uses :func:`differentiable_iou` instead.
        """
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2

        # Convert to [x1, y1, x2, y2]
        b1_x2, b1_y2 = x1 + w1, y1 + h1
        b2_x2, b2_y2 = x2 + w2, y2 + h2

        # Intersection
        inter_x1 = max(x1, x2)
        inter_y1 = max(y1, y2)
        inter_x2 = min(b1_x2, b2_x2)
        inter_y2 = min(b1_y2, b2_y2)

        inter_w = max(0, inter_x2 - inter_x1)
        inter_h = max(0, inter_y2 - inter_y1)
        intersection = inter_w * inter_h

        # Union
        area1 = w1 * h1
        area2 = w2 * h2
        union = area1 + area2 - intersection

        if union <= 0:
            return 0.0
        return intersection / union


def compute_reward_tensor(rewards: Dict[str, torch.Tensor]) -> torch.Tensor:
    """
    Convert reward dict to tensor: [R_loc, R_cls, R_cal, R_miss, R_fp, R_broken, R_skip].
    """
    return torch.stack([rewards[name] for name in DIMENSION_NAMES])
