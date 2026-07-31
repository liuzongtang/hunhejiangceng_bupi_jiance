"""
Loss functions for the hybrid reward training mechanism.

Section 5.2.3:
  L_total = L_detection + beta * L_scalar + gamma * L_consistency

Where:
  L_detection: traditional detection loss (BCE + regression)
  L_scalar: negative scalar reward (-R_scalar, maximizing reward)
  L_consistency: dimension variance penalty
"""

from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F


def detection_loss(
    predictions: Dict[str, torch.Tensor],
    targets: List[Dict],
) -> torch.Tensor:
    """
    Standard detection loss: classification BCE + bounding box regression.

    For each predicted box, computes:
    - Classification loss: BCE between predicted class and ground truth
    - Regression loss: Smooth L1 between predicted and target boxes

    Args:
        predictions: Dict with 'logits', 'boxes'.
        targets: List of target dicts with 'labels', 'boxes'.
    """
    logits = predictions.get("logits")
    pred_boxes = predictions.get("boxes")

    loss = torch.tensor(0.0, device=logits.device if logits is not None else "cpu")

    # Classification loss
    if logits is not None and targets:
        # Build target labels tensor
        target_labels = []
        for _i, t in enumerate(targets):
            labels = t.get("labels", [0])
            target_labels.append(labels[0] if labels else 0)
        if target_labels:
            target_tensor = torch.tensor(target_labels, device=logits.device)
            if logits.dim() == 1:
                logits = logits.unsqueeze(0)
            if logits.size(0) == len(target_labels):
                loss = loss + F.cross_entropy(logits, target_tensor)

    # Bounding box regression loss
    if pred_boxes is not None and len(pred_boxes) > 0 and targets:
        for i, target in enumerate(targets):
            t_boxes = target.get("boxes", [])
            if t_boxes and i < len(pred_boxes):
                t_box = torch.tensor(t_boxes[0], dtype=torch.float32)
                p_box = pred_boxes[i]
                if isinstance(p_box, np.ndarray):
                    p_box = torch.from_numpy(p_box).float()
                loss = loss + F.smooth_l1_loss(p_box, t_box)

    return loss


def scalar_reward_loss(
    dim_rewards: Dict[str, torch.Tensor],
    dim_weights: torch.Tensor,
    consistency_coeff: float = 0.3,
) -> torch.Tensor:
    """
    L_scalar = -R_scalar

    R_scalar = Σ(w_k * R_dim_k) + alpha * R_consistency

    We want to maximize scalar reward, so loss = -R_scalar.

    Args:
        dim_rewards: Dict of dimension_name -> reward tensor.
        dim_weights: Normalized dimension weights (7,).
        consistency_coeff: alpha coefficient for consistency reward.

    Returns:
        Scalar loss value (negated reward).
    """
    # Stack rewards into tensor [7]
    from backend.training.dimension_rewards import DIMENSION_NAMES

    reward_tensor = torch.stack([dim_rewards[name] for name in DIMENSION_NAMES])

    # Weighted sum
    weights = torch.softmax(dim_weights, dim=0)
    r_scalar = torch.sum(weights * reward_tensor)

    # Consistency reward: lower variance = more consistent = higher reward
    consistency = -torch.var(reward_tensor)
    r_scalar = r_scalar + consistency_coeff * consistency

    # Loss = negative reward (maximize reward)
    return -r_scalar


def consistency_loss(dim_rewards: Dict[str, torch.Tensor]) -> torch.Tensor:
    """
    L_consistency = Σ(R_k - mean(R))²

    Penalizes dimensions that diverge significantly from the mean,
    encouraging balanced optimization across all dimensions.

    Args:
        dim_rewards: Dict of dimension_name -> reward tensor.

    Returns:
        Scalar consistency penalty.
    """
    from backend.training.dimension_rewards import DIMENSION_NAMES

    reward_tensor = torch.stack([dim_rewards[name] for name in DIMENSION_NAMES])
    mean_reward = torch.mean(reward_tensor)
    return torch.sum((reward_tensor - mean_reward) ** 2)


def total_loss(
    predictions: Dict[str, torch.Tensor],
    targets: List[Dict],
    dim_rewards: Dict[str, torch.Tensor],
    dim_weights: torch.Tensor,
    beta: float = 0.1,
    gamma: float = 0.05,
    alpha: float = 0.3,
) -> Dict[str, torch.Tensor]:
    """
    Compute total training loss (Section 5.2.3).

    L_total = L_detection + beta * L_scalar + gamma * L_consistency

    Returns:
        Dict with individual loss components for logging.
    """
    l_det = detection_loss(predictions, targets)
    l_scalar = scalar_reward_loss(dim_rewards, dim_weights, alpha)
    l_consistency = consistency_loss(dim_rewards)

    total = l_det + beta * l_scalar + gamma * l_consistency

    return {
        "total": total,
        "detection": l_det,
        "scalar": l_scalar,
        "consistency": l_consistency,
    }
