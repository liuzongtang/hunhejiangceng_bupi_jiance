"""
Group-relative reward optimization (GRPO) trainer for detection (Layer 3).

Wires the model-agnostic GRPO primitives in :mod:`backend.training.grpo` to the
7-dimension ``DimensionRewardComputer``: sample a group of ``G`` candidate
detection sets per image (via query-noise exploration), score each with the
reward computer, form a group-relative advantage, and step the policy with
REINFORCE (+ optional KL to a frozen reference).

The practical recipe combines two signals:

1. ``L_grpo = -mean(A_g * log_prob_g)`` tunes the exploration temperature using
   the group baseline — the part that cannot "reward hack", because the baseline
   is the group's own mean/std.
2. ``L_reward = -mean(R_g)`` (optional, ``beta_reward > 0``) is the
   *differentiable* reward gradient that steers the model weights themselves
   (the Layer-1 signal), now regularized by the group baseline.

With the likelihood-ratio score, ``log_prob`` carries a non-zero gradient w.r.t.
both ``log_std`` and the (non-detached) latent mean. But RT-DETR detaches its
content queries in training, so the mean gradient does not reach the model:
``L_grpo`` in practice tunes only the exploration temperature, and ``L_reward``
(the differentiable path) is what moves the detector's mean. Both are exposed so
the trade-off can be ablated.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional

import torch
import torch.nn as nn
from ultralytics.models.utils.ops import HungarianMatcher
from ultralytics.nn.tasks import RTDETRDetectionModel

from backend.training.dimension_rewards import (
    DIMENSION_NAMES,
    DimensionRewardComputer,
)
from backend.training.grpo import (
    LatentGaussianPolicy,
    SampledRollout,
    group_advantages,
    grpo_policy_loss,
)


class GRPOPolicy(nn.Module):
    """Stochastic policy producing ``G`` rollouts per batch (interface)."""

    def sample_group(
        self,
        images: torch.Tensor,
        targets: Dict,
    ) -> List[SampledRollout]:
        """Sample ``G`` candidate detection sets for the batch."""
        raise NotImplementedError

    def kl_penalty(self) -> torch.Tensor:
        """KL penalty toward the reference policy (scalar, differentiable)."""
        raise NotImplementedError


def prepare_rtdetr_batch(
    batch: Dict, device: torch.device
) -> tuple[torch.Tensor, Dict]:
    """
    Convert an Ultralytics collated batch into the model-batch dict RT-DETR expects.

    Mirrors ``RTDETRDetectionModel.loss``: the image is scaled to float/255 and the
    flat ``cls``/``bboxes`` are paired with a per-image ``gt_groups`` count.

    Args:
        batch: Collated dataloader batch with ``img``/``cls``/``bboxes``/``batch_idx``.
        device: Target device.

    Returns:
        ``(img, targets)`` where ``img`` is ``(B, C, H, W)`` float and ``targets``
        has ``cls``/``bboxes``/``batch_idx``/``gt_groups``.
    """
    img = batch["img"].to(device).float() / 255.0
    bs = img.shape[0]
    batch_idx = batch["batch_idx"].to(device, dtype=torch.long)
    gt_groups = [(batch_idx == i).sum().item() for i in range(bs)]
    targets = {
        "cls": batch["cls"].to(device, dtype=torch.long).view(-1),
        "bboxes": batch["bboxes"].to(device),
        "batch_idx": batch_idx.view(-1),
        "gt_groups": gt_groups,
    }
    return img, targets


def align_rtdetr_predictions(
    pred_boxes: torch.Tensor,
    pred_scores: torch.Tensor,
    targets: Dict,
    matcher: HungarianMatcher,
) -> Optional[tuple]:
    """
    Hungarian-align RT-DETR detections to GT, returning ``(predictions, targets)``.

    Uses the same matcher (and cost gains) as the supervised RT-DETR loss, so the
    reward sees exactly the query/GT pairs supervision selected.

    Args:
        pred_boxes: ``(B, N, 4)`` predicted boxes (xywh).
        pred_scores: ``(B, N, C)`` predicted class logits.
        targets: Model-batch dict with flat ``cls``/``bboxes`` and ``gt_groups``.
        matcher: A :class:`~ultralytics.models.utils.ops.HungarianMatcher`.

    Returns:
        ``(predictions, aligned_targets)`` where ``predictions`` has ``logits``/
        ``boxes`` stacked over all matched pairs, or None if nothing matched.
    """
    match_indices = matcher(
        pred_boxes, pred_scores, targets["bboxes"], targets["cls"], targets["gt_groups"]
    )
    pred_logits: List[torch.Tensor] = []
    pred_boxes_aligned: List[torch.Tensor] = []
    aligned_targets: List[Dict] = []
    for i, (src, dst) in enumerate(match_indices):
        for k in range(len(src)):
            pred_logits.append(pred_scores[i][src[k]])
            pred_boxes_aligned.append(pred_boxes[i][src[k]])
            aligned_targets.append(
                {
                    "boxes": [targets["bboxes"][dst[k]].tolist()],
                    "labels": [int(targets["cls"][dst[k]])],
                }
            )
    if not aligned_targets:
        return None
    return (
        {"logits": torch.stack(pred_logits), "boxes": torch.stack(pred_boxes_aligned)},
        aligned_targets,
    )


class GRPOTrainer:
    """
    Model-agnostic GRPO training loop (CPU-testable with a fake policy).

    Args:
        policy: A :class:`GRPOPolicy` producing ``num_groups`` rollouts per batch.
        reward_computer: A ``head_type="sigmoid"`` :class:`DimensionRewardComputer`.
        optimizer: Optimizer over the policy's trainable parameters.
        num_groups: ``G``, number of rollouts per group.
        beta_kl: Weight of the KL penalty to the reference policy.
        beta_reward: Weight of the differentiable reward-gradient term (0 disables).
        ref_log_std: Reference policy's fixed log noise scale.
        advantage_eps: Floor on the group-advantage std denominator (see
            :func:`group_advantages`). A non-trivial floor keeps the advantage from
            exploding when the group's rewards are nearly identical.
        advantage_clip: Clamp group advantages to ``[-clip, clip]`` (safety net).
        align_fn: ``(boxes, scores, targets) -> (predictions, aligned_targets)``.
            Defaults to index-alignment (prediction ``i`` ↔ target ``i``), which
            matches ``DimensionRewardComputer``'s own index alignment.
        device: Compute device.
    """

    def __init__(
        self,
        policy: GRPOPolicy,
        reward_computer: DimensionRewardComputer,
        optimizer: torch.optim.Optimizer,
        num_groups: int = 4,
        beta_kl: float = 0.01,
        beta_reward: float = 0.0,
        ref_log_std: float = -4.6,  # reference noise ~0.01
        advantage_eps: float = 1e-2,
        advantage_clip: float = 5.0,
        align_fn: Optional[Callable] = None,
        device: Optional[str] = None,
    ):
        self.policy = policy
        self.reward_computer = reward_computer
        self.optimizer = optimizer
        self.num_groups = num_groups
        self.beta_kl = beta_kl
        self.beta_reward = beta_reward
        self.ref_log_std = ref_log_std
        self.advantage_eps = advantage_eps
        self.advantage_clip = advantage_clip
        self.align_fn = align_fn or _index_align
        if device is not None:
            self.device = device
        else:
            params = list(policy.parameters())
            self.device = params[0].device if params else torch.device("cpu")

    def train_step(self, images: torch.Tensor, targets: Dict) -> Dict[str, float]:
        """
        One GRPO step: sample, score, advantage, REINFORCE, step.

        Args:
            images: ``(B, C, H, W)`` float image tensor.
            targets: Model-batch dict with ``cls``/``bboxes``/``gt_groups``.

        Returns:
            Dict of scalar diagnostics (loss, policy loss, reward mean/std,
            per-rollout advantages).
        """
        rollouts = self.policy.sample_group(images, targets)
        rewards: List[float] = []
        reward_tensors: List[torch.Tensor] = []
        for rollout in rollouts:
            scored = self._score(rollout, targets)
            rewards.append(scored["reward"])
            reward_tensors.append(scored["reward_tensor"])

        rewards_t = torch.tensor(rewards, device=self.device)
        advantages = group_advantages(
            rewards_t, eps=self.advantage_eps, clip=self.advantage_clip
        )
        log_probs = torch.stack([r.log_prob for r in rollouts])

        loss_policy = grpo_policy_loss(log_probs, advantages)
        loss = loss_policy
        if self.beta_kl > 0:
            loss = loss + self.beta_kl * self.policy.kl_penalty()
        if self.beta_reward > 0:
            loss = loss - self.beta_reward * torch.stack(reward_tensors).mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
        self.optimizer.step()

        return {
            "loss": float(loss.detach().item()),
            "loss_policy": float(loss_policy.detach().item()),
            "mean_reward": float(rewards_t.mean().item()),
            "std_reward": float(rewards_t.std().item()),
            "advantages": [round(float(a), 4) for a in advantages.detach().tolist()],
        }

    def _score(self, rollout: SampledRollout, targets: Dict) -> Dict:
        """
        Score one rollout with the 7-dim reward computer.

        Returns:
            ``{"reward": detached scalar float, "reward_tensor": differentiable scalar}``.
        """
        aligned = self.align_fn(rollout.boxes, rollout.scores, targets)
        if aligned is None:
            zero = torch.tensor(0.0, device=self.device)
            return {"reward": 0.0, "reward_tensor": zero}
        predictions, aligned_targets = aligned
        rewards = self.reward_computer.compute_all(predictions, aligned_targets)
        reward_tensor = torch.stack([rewards[name] for name in DIMENSION_NAMES]).mean()
        return {
            "reward": float(reward_tensor.detach().item()),
            "reward_tensor": reward_tensor,
        }


def _index_align(
    pred_boxes: torch.Tensor,
    pred_scores: torch.Tensor,
    targets: Dict,
) -> Optional[tuple]:
    """Default align: pair prediction ``i`` with target ``i`` (no matching)."""
    boxes = pred_boxes[0]
    scores = pred_scores[0]
    n = min(boxes.shape[0], len(targets["cls"]))
    if n == 0:
        return None
    aligned_targets = [
        {"boxes": [targets["bboxes"][i].tolist()], "labels": [int(targets["cls"][i])]}
        for i in range(n)
    ]
    return {"logits": scores[:n], "boxes": boxes[:n]}, aligned_targets


class RTDETRGRPOPolicy(GRPOPolicy):
    """
    Wraps an RT-DETR model with Gaussian noise on the decoder's content queries.

    The decoder's ``_get_decoder_input`` is monkey-patched to add ``std * eps``
    to the content-query embeddings (the last ``num_queries`` rows of ``embed``,
    after any denoising prefix) and to record the sampled latent's log-prob. Each
    ``sample_group`` call runs ``G`` forward passes with fresh noise.

    Only ``log_std`` (and anything the caller keeps trainable on the model) is a
    policy parameter; the model weights are frozen unless the caller unfreezes
    them and enables ``beta_reward`` on the trainer.
    """

    def __init__(
        self,
        model: RTDETRDetectionModel,
        num_groups: int = 4,
        log_std_init: float = -2.3,
        ref_log_std: float = -4.6,
        detach_detections: bool = True,
    ):
        super().__init__()
        self.model = model
        self.decoder = model.model[-1]
        self.num_groups = num_groups
        self.latent = LatentGaussianPolicy(log_std_init)
        self.ref_log_std = ref_log_std
        self.noise_enabled = True
        # Detach the rollout boxes/scores so the decoder's graph is freed each
        # rollout (pure-GRPO mode). Set False only when ``beta_reward > 0`` needs
        # the differentiable reward gradient to flow into the decoder.
        self.detach_detections = detach_detections
        self._last_log_prob: Optional[torch.Tensor] = None
        self._install_noise()

    def _install_noise(self) -> None:
        """Patch ``_get_decoder_input`` to inject query noise and record log-prob."""
        decoder = self.decoder
        latent = self.latent
        orig = decoder._get_decoder_input

        def patched(feats, shapes, dn_embed=None, dn_bbox=None):
            embed, refer_bbox, enc_bboxes, enc_scores = orig(
                feats, shapes, dn_embed=dn_embed, dn_bbox=dn_bbox
            )
            if self.noise_enabled:
                nq = decoder.num_queries
                content = embed[:, -nq:] if embed.shape[1] > nq else embed
                z, log_prob = latent.sample(content)
                self._last_log_prob = log_prob
                if embed.shape[1] > nq:
                    embed = torch.cat([embed[:, :-nq], z], dim=1)
                else:
                    embed = z
            return embed, refer_bbox, enc_bboxes, enc_scores

        decoder._get_decoder_input = patched

    def sample_group(self, images: torch.Tensor, targets: Dict) -> List[SampledRollout]:
        """Run ``G`` noisy forward passes and return their final-layer detections."""
        rollouts: List[SampledRollout] = []
        for _ in range(self.num_groups):
            preds = self.model.predict(images, batch=targets)
            dec_bboxes, dec_scores, _enc_bboxes, _enc_scores, dn_meta = preds
            if dn_meta is not None:
                _, dec_bboxes = torch.split(dec_bboxes, dn_meta["dn_num_split"], dim=2)
                _, dec_scores = torch.split(dec_scores, dn_meta["dn_num_split"], dim=2)
            # The denoising split returns non-contiguous views; the Hungarian
            # matcher needs a contiguous tensor for its `.view(-1, nc)`.
            boxes = dec_bboxes[-1].contiguous()
            scores = dec_scores[-1].contiguous()
            if self.detach_detections:
                boxes = boxes.detach()
                scores = scores.detach()
            rollouts.append(
                SampledRollout(
                    log_prob=self._last_log_prob,
                    boxes=boxes,
                    scores=scores,
                )
            )
        return rollouts

    def kl_penalty(self) -> torch.Tensor:
        """KL penalty toward the reference policy (same mean, fixed noise scale)."""
        return self.latent.kl_penalty(self.ref_log_std)
