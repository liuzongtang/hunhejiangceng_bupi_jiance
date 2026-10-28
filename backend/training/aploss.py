"""
M1: Smooth-AP proxy loss for RT-DETR fine-tuning (score-ranking arm).

Hypothesis (handoff section 10)
    The bottleneck diagnosed for this detector is not detection capability but
    *ranking*: at the confidence floor, 89%..100% of critical-class GTs are still
    present, yet TP/FP separability (AUC 0.76..0.82) is what decides the PR curve.
    Every historical intervention re-weighted or replaced a *loss*; M1 instead
    optimises a differentiable surrogate of the ranking metric itself (Smooth-AP,
    Rodriguez et al., 2020) on the scores of the Hungarian-matched queries, in
    addition to the untouched GIoU/L1/classification losses.

Pre-registered criterion (handoff section 10)
    ``mean > B* + 0.005`` where ``B* = 0.6349`` is the locked mosaic1280 baseline,
    judged with 3 seeds (42 / 123 / 2024).

Failure branch (handoff section 10)
    M1 negative -> "detection mAP on this data/architecture is near its ceiling"
    becomes a strong conclusion; the project's value proposition moves to the
    decision layer (A2) plus the diagnostic/negative-result chain. No further
    loss-side re-weighting arm is opened.

Difference from the re-weighting arms (R2 / B / R3-NWD)
    Those arms *slid along* an already-converged PR curve by changing the weight
    or the form of an existing term (zero-sum: mean -0.015 .. -0.029 bought by
    crit_recall +0.10 .. +0.18). M1 leaves every supervised term and gain intact
    and *adds* a term whose gradient is the derivative of a ranking metric:

        L_total = L_giou + L_class + L_bbox + L_ap
        L_ap    = -beta * mean_b AP_b

Exact AP-proxy formula used here
    For image ``b`` let ``P_b`` be the set of matched query indices (one per GT,
    the same pairs the supervised Hungarian matcher produced) and ``N_b`` the set
    of all other queries. With the head's raw class logits ``z`` (RT-DETR's
    per-class sigmoid head) and temperature ``T``::

        p_i = z_{i, y_i}                  for i in P_b   (GT-class logit)
        p_k = max_c z_{k, c}              for k in N_b   (top-class logit)
        sigma(x) = 1 / (1 + exp(-x / T))

        num_i = 1 + sum_{j in P_b, j != i} sigma(p_j - p_i)
        den_i = num_i + sum_{k in N_b}      sigma(p_k - p_i)
        AP_b  = (1 / |P_b|) * sum_{i in P_b} num_i / den_i
        L_ap  = -beta * mean_{b: |P_b| > 0} AP_b

    This is the Smooth-AP ratio with the standard diagonal exclusion; each matched
    positive is ranked against the other positives and against every background
    query. Logits are used rather than probabilities because sigmoid is monotone
    (the ranking, hence AP, is identical) while probability differences saturate,
    which would cap the soft ratio well below 1 even for a perfect ranking. ``T``
    (``--temp``, default 0.5) controls how sharply a logit gap is penalised:
    smaller ``T`` approximates the hard AP ranking more closely. Images with no GT
    hold no positives and are skipped (AP is undefined there); images where every
    query is matched contribute AP = 1.

    The term is a function of the class logits only, so its gradient is non-zero
    w.r.t. ``dec_scores`` and structurally zero w.r.t. ``dec_bboxes`` - it leaves
    the box arm to the supervised GIoU/L1 terms (and, in a combination arm, to
    D1's ``loss_loc``). Nothing is detached: the Smooth-AP graph flows straight
    into the decoder's class logits. At ``beta == 0`` the criterion is an exact
    uniform control (the added key is identically zero and the matcher consumes
    no RNG).

The term is added under the ``loss_ap`` key; Ultralytics' RT-DETR loss sums every
entry of the criterion dict, folding it into the training objective.
:func:`attach_ap_criterion` must be called from an ``on_train_start`` callback
because the trainer rebuilds its model inside ``_setup_train``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from ultralytics.models.utils.loss import RTDETRDetectionLoss
from ultralytics.nn.tasks import RTDETRDetectionModel


def smooth_ap(
    pred_scores: torch.Tensor,
    match_indices: Sequence[Tuple[torch.Tensor, torch.Tensor]],
    gt_cls: torch.Tensor,
    temp: float = 0.5,
) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
    """
    Smooth-AP proxy over the matched-query scores of one batch.

    Scores are the raw head logits (the sigmoid applied at inference is monotone,
    so the ranking - and hence AP - is unchanged); working in logit space keeps
    the logistic ranking informative across the full score range, whereas
    probability differences saturate by construction.

    Args:
        pred_scores: ``(B, N, C)`` class logits from the final decoder layer.
        match_indices: Per-image ``(src, dst)`` query/GT index pairs, exactly as
            returned by the RT-DETR Hungarian matcher.
        gt_cls: Flat ``(num_gt,)`` ground-truth class ids across the batch.
        temp: Smoothing temperature ``T`` (smaller = closer to hard AP).

    Returns:
        Tuple ``(loss_ap, mean_ap, n_pos, n_neg)`` where ``loss_ap`` is the
        differentiable scalar ``-mean_b AP_b`` (without ``beta``), ``mean_ap`` is
        the detached mean AP proxy, and the last two are matched / unmatched query
        counts backing it.
    """
    if temp <= 0.0:
        raise ValueError("temp must be positive")

    images: List[torch.Tensor] = []
    n_pos_total = 0
    n_neg_total = 0
    for i, (src, dst) in enumerate(match_indices):
        if len(src) == 0:
            continue
        src = src.to(device=pred_scores.device)
        dst = dst.to(device=pred_scores.device)
        labels = gt_cls[dst].to(device=pred_scores.device, dtype=torch.long)
        p_pos = pred_scores[i, src, labels]  # matched query score on its GT class

        matched_mask = torch.zeros(
            pred_scores.shape[1], dtype=torch.bool, device=pred_scores.device
        )
        matched_mask[src] = True
        p_neg = (
            pred_scores[i][~matched_mask].max(dim=-1).values
            if (~matched_mask).any()
            else None
        )

        pos_diff = (p_pos.unsqueeze(0) - p_pos.unsqueeze(1)) / temp  # [i, j]
        pos_rank = torch.sigmoid(pos_diff)
        # Exclude j == i from both sums (Smooth-AP diagonal exclusion). Masking,
        # rather than an in-place fill, keeps the sigmoid graph intact.
        off_diag = ~torch.eye(p_pos.numel(), dtype=torch.bool, device=p_pos.device)
        pos_sum = (pos_rank * off_diag).sum(dim=1)  # (P,)

        if p_neg is not None and p_neg.numel() > 0:
            neg_rank = torch.sigmoid((p_neg.unsqueeze(0) - p_pos.unsqueeze(1)) / temp)
            num = 1.0 + pos_sum
            den = 1.0 + pos_sum + neg_rank.sum(dim=1)
            n_neg_total += int(p_neg.numel())
        else:
            # No unmatched queries: nothing to outrank, AP_b == 1 (zero gradient).
            num = torch.ones_like(pos_sum)
            den = torch.ones_like(pos_sum)

        images.append((num / den).mean())
        n_pos_total += int(src.numel())

    if not images:
        device = pred_scores.device
        return pred_scores.new_zeros(()), pred_scores.new_zeros(()), 0, 0

    mean_ap = torch.stack(images).mean()
    return -mean_ap, mean_ap.detach(), n_pos_total, n_neg_total


class SmoothAPRTDETRDetectionLoss(RTDETRDetectionLoss):
    """
    RT-DETR criterion that appends a Smooth-AP ranking loss on matched queries.

    The supervised losses (GIoU, L1, VFL classification, denoising) are inherited
    unchanged; only one extra key (``loss_ap``) is added to the returned dict.

    Args:
        nc: Number of defect classes.
        beta: Weight of the AP-proxy term (0 = exact uniform control).
        temp: Smoothing temperature of the pairwise logistic ranking.
        loss_gain: Optional per-term gains forwarded to the RT-DETR criterion.

    Attributes:
        loss_ap_history: Detached per-step ``loss_ap`` values (for CSV/plots).
        ap_history: Detached per-step mean Smooth-AP proxy (higher is better).
        n_pos_history: Matched-query count behind each step.
        n_neg_history: Unmatched-query count behind each step.
    """

    def __init__(
        self,
        nc: int,
        beta: float = 0.1,
        temp: float = 0.5,
        loss_gain: Optional[Dict[str, float]] = None,
    ):
        super().__init__(nc=nc, use_vfl=True, loss_gain=loss_gain)
        self.beta = float(beta)
        self.temp = float(temp)
        self.loss_ap_history: List[float] = []
        self.ap_history: List[float] = []
        self.n_pos_history: List[int] = []
        self.n_neg_history: List[int] = []

    def forward(
        self,
        preds: tuple,
        batch: Dict[str, Any],
        dn_bboxes: Optional[torch.Tensor] = None,
        dn_scores: Optional[torch.Tensor] = None,
        dn_meta: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute the supervised RT-DETR loss plus the Smooth-AP proxy.

        Args:
            preds: ``(dec_bboxes, dec_scores)`` of shape ``(L, B, N, 4)`` /
                ``(L, B, N, C)``, already split from denoising queries by the
                caller (as in ``RTDETRDetectionModel.loss``).
            batch: Target dict with flat ``cls`` / ``bboxes`` and ``gt_groups``.
            dn_bboxes/dn_scores/dn_meta: Denoising tensors, passed through.

        Returns:
            Loss dict including the extra ``loss_ap`` key.
        """
        loss = super().forward(
            preds,
            batch,
            dn_bboxes=dn_bboxes,
            dn_scores=dn_scores,
            dn_meta=dn_meta,
        )
        dec_bboxes, dec_scores = preds
        loss["loss_ap"] = self._ap_loss(dec_bboxes[-1], dec_scores[-1], batch)
        return loss

    def _ap_loss(
        self,
        pred_bboxes: torch.Tensor,
        pred_scores: torch.Tensor,
        batch: Dict[str, Any],
    ) -> torch.Tensor:
        """
        Compute ``-beta * mean_b AP_b`` on the Hungarian-matched queries.

        Args:
            pred_bboxes: ``(B, N, 4)`` final decoder-layer boxes (matcher input).
            pred_scores: ``(B, N, C)`` final decoder-layer class logits.
            batch: Target dict with flat ``bboxes`` / ``cls`` and ``gt_groups``.

        Returns:
            Differentiable scalar on ``pred_scores.device`` (identically zero when
            ``beta == 0``).
        """
        # Same matcher and inputs the supervised loss already used.
        match_indices = self.matcher(
            pred_bboxes, pred_scores, batch["bboxes"], batch["cls"], batch["gt_groups"]
        )
        loss_ap, mean_ap, n_pos, n_neg = smooth_ap(
            pred_scores, match_indices, batch["cls"], temp=self.temp
        )
        loss_ap = self.beta * loss_ap
        # Detach only the logging statistics; the loss tensor keeps its graph.
        self.loss_ap_history.append(float(loss_ap.detach().item()))
        self.ap_history.append(float(mean_ap.item()))
        self.n_pos_history.append(n_pos)
        self.n_neg_history.append(n_neg)
        return loss_ap


def attach_ap_criterion(
    model: RTDETRDetectionModel,
    beta: float = 0.1,
    temp: float = 0.5,
    loss_gain: Optional[Dict[str, float]] = None,
) -> SmoothAPRTDETRDetectionLoss:
    """
    Install the Smooth-AP proxy criterion on an RT-DETR model.

    Must be called from an ``on_train_start`` callback: the trainer rebuilds its
    model inside ``_setup_train``, so a criterion attached before
    ``model.train()`` is silently discarded.

    Both ``model.criterion`` and ``model.init_criterion`` are patched so the
    criterion survives Ultralytics' setup path.

    Args:
        model: The trainer's ``RTDETRDetectionModel`` (unwrapped from DDP).
        beta: AP-proxy weight (0 = exact uniform control).
        temp: Smoothing temperature of the pairwise logistic ranking.
        loss_gain: Optional per-term gains forwarded to the RT-DETR criterion.

    Returns:
        The installed criterion.
    """
    nc = getattr(model, "nc", None)
    if nc is None:
        nc = model.model[-1].nc
    criterion = SmoothAPRTDETRDetectionLoss(
        nc=int(nc),
        beta=beta,
        temp=temp,
        loss_gain=loss_gain,
    )
    model.criterion = criterion
    model.init_criterion = lambda: criterion  # type: ignore[method-assign]
    return criterion
