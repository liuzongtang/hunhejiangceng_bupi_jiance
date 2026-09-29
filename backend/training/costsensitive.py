"""
Cost-sensitive (severity-weighted) RT-DETR classification loss — Direction B.

RT-DETR's classification head is trained with :class:`VarifocalLoss`
(``RTDETRDetectionModel.init_criterion`` returns ``RTDETRDetectionLoss(nc=...,
use_vfl=True)``). Varifocal loss down-weights easy negatives but treats every
class equally, so it cannot express the asymmetric industrial cost where missing
a *critical* defect (hole / broken_warp / broken_spandex) is far worse than
missing a *minor* one.

The varifocal weighting factor decomposes into a positive branch (``gt_score *
label``, the IoU-aware score of the assigned class) and a negative branch
(``alpha * sigmoid^gamma * (1 - label)``, which suppresses false positives on
background queries). Two weighting modes are supported:

* ``"positive"`` (default) — scale only the positive branch by ``class_weights``,
  i.e. up-weight *catching* a critical class when it is present, without touching
  the false-positive suppression. This is the cost-sensitive-recall formulation
  that avoids the over-suppression seen with full-term weighting.
* ``"full"`` — scale the whole per-element loss (positive and negative). This was
  found counterproductive (critical recall dropped vs. uniform fine-tune), kept
  here only for ablation.

Usage (drop-in via monkey-patching the loss class so ``init_criterion`` picks it
up without touching Ultralytics source)::

    import ultralytics.models.utils.loss as _detr_loss
    from backend.training.costsensitive import (
        CostSensitiveRTDETRDetectionLoss,
        set_default_class_weights,
        set_default_weight_mode,
    )

    set_default_class_weights(severity_weights)  # shape (nc,)
    set_default_weight_mode("positive")  # or "full"
    _detr_loss.RTDETRDetectionLoss = CostSensitiveRTDETRDetectionLoss
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from ultralytics.models.utils.loss import RTDETRDetectionLoss
from ultralytics.utils.loss import VarifocalLoss
from ultralytics.utils.torch_utils import autocast

# Module-level defaults so ``CostSensitiveRTDETRDetectionLoss`` can act as a
# drop-in replacement for ``RTDETRDetectionLoss``: ``init_criterion`` constructs
# it with only ``nc`` / ``use_vfl`` and no explicit weights, so the weights and
# mode are resolved from here.
_DEFAULT_CLASS_WEIGHTS: Optional[torch.Tensor] = None
_DEFAULT_WEIGHT_MODE: str = "positive"


def set_default_class_weights(weights: "list[float] | torch.Tensor") -> None:
    """Set the module-level severity weights used when none are passed explicitly."""
    global _DEFAULT_CLASS_WEIGHTS
    _DEFAULT_CLASS_WEIGHTS = torch.as_tensor(weights, dtype=torch.float32)


def set_default_weight_mode(mode: str) -> None:
    """Set the module-level weighting mode (``"positive"`` or ``"full"``)."""
    global _DEFAULT_WEIGHT_MODE
    if mode not in ("positive", "full"):
        raise ValueError(f"weight_mode must be 'positive' or 'full', got {mode!r}")
    _DEFAULT_WEIGHT_MODE = mode


class WeightedVarifocalLoss(VarifocalLoss):
    """
    :class:`VarifocalLoss` with an optional per-class severity weighting factor.

    ``mode="positive"`` scales only the positive (assigned-class) branch of the
    varifocal weight, up-weighting *catching* critical classes without amplifying
    the false-positive suppression. ``mode="full"`` scales the whole per-element
    loss (the earlier, counterproductive behaviour).
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: float = 0.75,
        class_weights: Optional[torch.Tensor] = None,
        mode: str = "positive",
    ):
        """Initialize with the same ``gamma``/``alpha`` as upstream VarifocalLoss."""
        super().__init__(gamma, alpha)
        if mode not in ("positive", "full"):
            raise ValueError(f"mode must be 'positive' or 'full', got {mode!r}")
        self.mode = mode
        self.class_weights = (
            torch.as_tensor(class_weights, dtype=torch.float32).view(1, 1, -1)
            if class_weights is not None
            else None
        )

    def forward(
        self, pred_score: torch.Tensor, gt_score: torch.Tensor, label: torch.Tensor
    ) -> torch.Tensor:
        """Compute varifocal loss, weighting critical classes by ``class_weights``."""
        neg_weight = self.alpha * pred_score.sigmoid().pow(self.gamma) * (1 - label)
        pos_weight = gt_score * label
        if self.class_weights is not None:
            # The criterion stays on CPU while pred_scores live on CUDA, so move
            # lazily (no-op once aligned).
            cw = self.class_weights.to(pred_score.device)
            if self.mode == "positive":
                # Up-weight only catching the assigned critical class.
                pos_weight = pos_weight * cw
            else:  # "full": scale the whole per-element loss (legacy ablation).
                with autocast(enabled=False):
                    loss = F.binary_cross_entropy_with_logits(
                        pred_score.float(), gt_score.float(), reduction="none"
                    ) * (neg_weight + pos_weight)
                return (loss * cw).mean(1).sum()

        weight = neg_weight + pos_weight
        with autocast(enabled=False):
            loss = (
                F.binary_cross_entropy_with_logits(
                    pred_score.float(), gt_score.float(), reduction="none"
                )
                * weight
            )
            return loss.mean(1).sum()


class CostSensitiveRTDETRDetectionLoss(RTDETRDetectionLoss):
    """
    RT-DETR detection loss whose classification term carries per-class severity weights.

    Only the classification arm changes; the Hungarian matcher, box L1/GIoU and
    (optional) denoising terms are inherited unchanged. ``class_weights`` and
    ``weight_mode`` may be passed explicitly or resolved from the module-level
    defaults set by :func:`set_default_class_weights` / :func:`set_default_weight_mode`.
    """

    def __init__(
        self,
        nc: int = 80,
        class_weights: Optional[torch.Tensor] = None,
        weight_mode: Optional[str] = None,
        **kwargs,
    ):
        """Build the upstream DETR loss, then swap in the weighted varifocal head."""
        super().__init__(nc=nc, **kwargs)
        if self.vfl is not None:
            weights = (
                class_weights if class_weights is not None else _DEFAULT_CLASS_WEIGHTS
            )
            mode = weight_mode if weight_mode is not None else _DEFAULT_WEIGHT_MODE
            self.vfl = WeightedVarifocalLoss(
                self.vfl.gamma, self.vfl.alpha, weights, mode
            )
