"""
7-Dimension Reward Computation for Hybrid Reward Trainer.

Matches Section 5.2.1 of the project document:

  D01: Localization precision (IoU)    R_loc = 2*IoU - 1           [-1, 1]
  D02: Classification accuracy         R_cls = +1 correct / -1 wrong  {-1, 1}
  D03: Confidence calibration (ECE)    R_cal = -ECE                 [-inf, 0]
  D04: Miss detection penalty          R_miss = -lam * miss_count   [-inf, 0]
  D05: False positive penalty          R_fp = -lam * fp_count       [-inf, 0]
  D06: Broken yarn sensitivity         R_broken = +1 detected / -2 missed
  D07: Skip/weave defect sensitivity   R_skip = +1 detected / -2 missed

Dimension reward vector: R_dim = [R_loc, R_cls, R_cal, R_miss, R_fp, R_broken, R_skip]
"""

from typing import Dict, List, Optional, Set

import numpy as np
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


class DimensionRewardComputer:
    """
    Computes the 7-dimensional reward vector for a batch of predictions.

    Each dimension is independently computed and returned as a dict for
    transparent logging and analysis.
    """

    def __init__(
        self,
        lambda_miss: float = 1.5,
        lambda_fp: float = 1.0,
        iou_threshold: float = 0.5,
        num_bins: int = 10,
        broken_class_ids: Optional[Set[int]] = None,
        skip_class_ids: Optional[Set[int]] = None,
    ):
        self.lambda_miss = lambda_miss
        self.lambda_fp = lambda_fp
        self.iou_threshold = iou_threshold
        self.num_bins = num_bins
        # Critical class ids for the D06/D07 sensitivity dimensions.
        self.broken_class_ids = (
            broken_class_ids
            if broken_class_ids is not None
            else TIANCHI_BROKEN_CLASS_IDS
        )
        self.skip_class_ids = (
            skip_class_ids if skip_class_ids is not None else TIANCHI_SKIP_CLASS_IDS
        )

    def compute_all(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: List[Dict],
    ) -> Dict[str, torch.Tensor]:
        """
        Compute all 7 dimension rewards.

        Args:
            predictions: Dict with 'boxes' (N,4), 'classes' (N,), 'confidences' (N,).
            targets: List of target dicts with 'boxes' (M,4) and 'labels' (M,).

        Returns:
            Dict mapping dimension name -> scalar reward tensor.
        """
        rewards = {}

        # D01: Localization precision (IoU)
        rewards["loc"] = self.compute_loc_reward(predictions, targets)

        # D02: Classification accuracy
        rewards["cls"] = self.compute_cls_reward(predictions, targets)

        # D03: Confidence calibration (ECE)
        rewards["cal"] = self.compute_cal_reward(predictions, targets)

        # D04: Miss detection penalty
        rewards["miss"] = self.compute_miss_reward(predictions, targets)

        # D05: False positive penalty
        rewards["fp"] = self.compute_fp_reward(predictions, targets)

        # D06: Broken yarn sensitivity
        rewards["broken"] = self.compute_broken_reward(predictions, targets)

        # D07: Skip/weave defect sensitivity
        rewards["skip"] = self.compute_skip_reward(predictions, targets)

        return rewards

    # ------------------------------------------------------------------
    # D01: Localization (IoU)
    # ------------------------------------------------------------------

    def compute_loc_reward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: List[Dict],
    ) -> torch.Tensor:
        """
        R_loc = 2 * IoU - 1

        Maps IoU ∈ [0, 1] to reward ∈ [-1, 1].
        IoU > 0.5 → positive reward.
        """
        pred_boxes = predictions.get("boxes")
        if pred_boxes is None or len(pred_boxes) == 0 or not targets:
            return torch.tensor(-1.0)

        ious = []
        for target in targets:
            t_boxes = target.get("boxes", [])
            if not t_boxes:
                ious.append(0.0)
                continue

            best_iou = 0.0
            for t_box in t_boxes:
                for p_box in pred_boxes:
                    iou = self._box_iou(
                        p_box.tolist() if isinstance(p_box, torch.Tensor) else p_box,
                        list(t_box)
                        if isinstance(t_box, (list, tuple))
                        else t_box.tolist(),
                    )
                    best_iou = max(best_iou, iou)
            ious.append(best_iou)

        mean_iou = torch.tensor(ious).mean()
        return 2.0 * mean_iou - 1.0

    # ------------------------------------------------------------------
    # D02: Classification
    # ------------------------------------------------------------------

    def compute_cls_reward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: List[Dict],
    ) -> torch.Tensor:
        """
        R_cls = +1 if correctly classified, -1 otherwise.
        """
        pred_classes = predictions.get("classes")
        if pred_classes is None or len(pred_classes) == 0:
            return torch.tensor(-1.0)

        correct = 0
        total = 0
        for i, target in enumerate(targets):
            t_labels = target.get("labels", [])
            if not t_labels:
                continue
            # Match prediction by index if available, else use best score
            if i < len(pred_classes):
                p_cls = (
                    pred_classes[i].item()
                    if isinstance(pred_classes[i], torch.Tensor)
                    else pred_classes[i]
                )
            elif len(pred_classes) > 0:
                # Use first prediction for unmatched targets
                p_cls = (
                    pred_classes[0].item()
                    if isinstance(pred_classes[0], torch.Tensor)
                    else pred_classes[0]
                )
            else:
                continue
            if p_cls in t_labels:
                correct += 1
            total += 1

        if total == 0:
            return torch.tensor(-1.0)
        # Map accuracy ∈ [0,1] → reward ∈ [-1, 1]
        acc = correct / total
        return torch.tensor(2.0 * acc - 1.0)

    # ------------------------------------------------------------------
    # D03: Confidence Calibration (ECE)
    # ------------------------------------------------------------------

    def compute_cal_reward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: List[Dict],
    ) -> torch.Tensor:
        """
        R_cal = -ECE (Expected Calibration Error).

        Lower ECE → better calibrated confidences → higher reward.
        """
        confidences = predictions.get("confidences")
        pred_classes = predictions.get("classes")

        if confidences is None or len(confidences) == 0:
            return torch.tensor(-1.0)

        # Convert to torch if numpy
        if isinstance(confidences, np.ndarray):
            confidences = torch.from_numpy(confidences).float()

        # Bin confidences
        bins = torch.linspace(0, 1, self.num_bins + 1)
        ece = 0.0

        for i in range(self.num_bins):
            bin_start, bin_end = bins[i], bins[i + 1]
            mask = (confidences >= bin_start) & (confidences < bin_end)
            n_bin = mask.sum().item()

            if n_bin == 0:
                continue

            # Average confidence in bin
            avg_conf = confidences[mask].mean().item()

            # Accuracy in bin (only for predictions with matching targets)
            correct = 0
            matched = 0
            for j in range(min(len(confidences), len(targets))):
                if mask[j] and j < len(pred_classes):
                    t_labels = targets[j].get("labels", [])
                    p_cls = (
                        pred_classes[j].item()
                        if isinstance(pred_classes[j], torch.Tensor)
                        else pred_classes[j]
                    )
                    if p_cls in t_labels:
                        correct += 1
                    matched += 1

            if matched > 0:
                acc_in_bin = correct / max(matched, 1)
                ece += (matched / len(confidences)) * abs(avg_conf - acc_in_bin)

        return torch.tensor(-ece)

    # ------------------------------------------------------------------
    # D04: Miss Detection Penalty
    # ------------------------------------------------------------------

    def compute_miss_reward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: List[Dict],
    ) -> torch.Tensor:
        """
        R_miss = -lambda_miss * miss_count

        Penalizes ground-truth defects not detected by the model.
        """
        pred_boxes = predictions.get("boxes", [])
        miss_count = 0

        for target in targets:
            t_boxes = target.get("boxes", [])
            for t_box in t_boxes:
                detected = False
                for p_box in pred_boxes:
                    iou = self._box_iou(
                        p_box.tolist() if isinstance(p_box, torch.Tensor) else p_box,
                        list(t_box)
                        if isinstance(t_box, (list, tuple))
                        else t_box.tolist(),
                    )
                    if iou >= self.iou_threshold:
                        detected = True
                        break
                if not detected:
                    miss_count += 1

        return torch.tensor(-self.lambda_miss * miss_count)

    # ------------------------------------------------------------------
    # D05: False Positive Penalty
    # ------------------------------------------------------------------

    def compute_fp_reward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: List[Dict],
    ) -> torch.Tensor:
        """
        R_fp = -lambda_fp * fp_count

        Penalizes predictions with no matching ground-truth.
        """
        pred_boxes = predictions.get("boxes", [])
        fp_count = 0

        for p_box in pred_boxes:
            matched = False
            for target in targets:
                for t_box in target.get("boxes", []):
                    iou = self._box_iou(
                        p_box.tolist() if isinstance(p_box, torch.Tensor) else p_box,
                        list(t_box)
                        if isinstance(t_box, (list, tuple))
                        else t_box.tolist(),
                    )
                    if iou >= self.iou_threshold:
                        matched = True
                        break
                if matched:
                    break
            if not matched:
                fp_count += 1

        return torch.tensor(-self.lambda_fp * fp_count)

    # ------------------------------------------------------------------
    # D06: Broken Yarn Sensitivity
    # ------------------------------------------------------------------

    def compute_broken_reward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: List[Dict],
    ) -> torch.Tensor:
        """
        R_broken = +1 if a broken-yarn defect detected, -2 if missed.

        Heavy penalty for missing critical broken defects (e.g. broken_warp,
        broken_spandex in the Tianchi 20-class scheme).
        """
        pred_classes = predictions.get("classes", [])
        p_classes_set = {
            p.item() if isinstance(p, torch.Tensor) else p for p in pred_classes
        }

        has_broken = any(
            self._labels_contain(target.get("labels", []), self.broken_class_ids)
            for target in targets
        )

        if not has_broken:
            return torch.tensor(0.0)

        detected = bool(p_classes_set & self.broken_class_ids)

        return torch.tensor(1.0 if detected else -2.0)

    # ------------------------------------------------------------------
    # D07: Skip / Weave Sensitivity
    # ------------------------------------------------------------------

    def compute_skip_reward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: List[Dict],
    ) -> torch.Tensor:
        """
        R_skip = +1 if a skip/weave defect detected, -2 if missed.

        Heavy penalty for missing critical skip defects (e.g. star_skip,
        weave_defect/跳纱 in the Tianchi 20-class scheme).
        """
        pred_classes = predictions.get("classes", [])
        p_classes_set = {
            p.item() if isinstance(p, torch.Tensor) else p for p in pred_classes
        }

        has_skip = any(
            self._labels_contain(target.get("labels", []), self.skip_class_ids)
            for target in targets
        )

        if not has_skip:
            return torch.tensor(0.0)

        detected = bool(p_classes_set & self.skip_class_ids)

        return torch.tensor(1.0 if detected else -2.0)

    # ------------------------------------------------------------------
    # Label membership helper
    # ------------------------------------------------------------------

    @staticmethod
    def _labels_contain(labels: List, class_ids: Set[int]) -> bool:
        """Return True if any label (int or string) matches the class-id set."""
        for lbl in labels:
            try:
                if int(lbl) in class_ids:
                    return True
            except (TypeError, ValueError):
                pass
            if isinstance(lbl, str) and str(lbl) in class_ids:
                return True
        return False

    # ------------------------------------------------------------------
    # IoU Computation (bounding box)
    # ------------------------------------------------------------------

    @staticmethod
    def _box_iou(box1: List[float], box2: List[float]) -> float:
        """
        Compute IoU between two bounding boxes in [x, y, w, h] format.
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
