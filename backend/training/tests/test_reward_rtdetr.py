"""
Tests for the reward-injected RT-DETR criterion (Layer 1 integration).

These exercise the criterion in isolation (no model load) to keep the suite
fast: the reward term must (a) appear in the loss dict, (b) be positive on a
critical-class hit, and (c) backpropagate to both the class logits and the box
regression outputs.
"""

from __future__ import annotations

import torch

from backend.training.dimension_rewards import DimensionRewardComputer
from backend.training.reward_rtdetr import (
    RewardRTDETRDetectionLoss,
    attach_reward_criterion,
)


class TestRewardRTDETRDetectionLoss:
    """Unit tests for the reward criterion."""

    @staticmethod
    def _criterion(beta: float = 1.0) -> RewardRTDETRDetectionLoss:
        comp = DimensionRewardComputer(
            head_type="sigmoid",
            broken_class_ids={9, 15},
            skip_class_ids={14, 17},
        )
        return RewardRTDETRDetectionLoss(nc=20, reward_computer=comp, beta=beta)

    def test_injects_loss_reward(self):
        """The criterion must append a `loss_reward` key."""
        crit = self._criterion()
        # One confident broken_warp (class 9) query against a matching GT.
        pred_bboxes = torch.tensor([[[0.5, 0.5, 0.2, 0.2]] * 4])
        pred_scores = torch.zeros(1, 4, 20)
        pred_scores[0, 0, 9] = 4.0
        batch = {
            "cls": torch.tensor([9]),
            "bboxes": torch.tensor([[0.5, 0.5, 0.2, 0.2]]),
            "gt_groups": [1],
        }
        loss = crit((pred_bboxes.unsqueeze(0), pred_scores.unsqueeze(0)), batch)
        assert "loss_reward" in loss
        # Critical class hit -> positive scalar reward -> negative loss term.
        assert loss["loss_reward"].item() < 0.0

    def test_loss_reward_backprops_to_scores_and_boxes(self):
        """The reward must be differentiable w.r.t. both heads."""
        crit = self._criterion()
        pred_bboxes = torch.tensor(
            [[[0.5, 0.5, 0.2, 0.2]] * 2], dtype=torch.float32, requires_grad=True
        )
        boost = torch.tensor(
            [[[1.0 if c == 9 else 0.0 for c in range(20)] for _ in range(2)]]
        )
        pred_scores = torch.zeros(1, 2, 20, requires_grad=True) + boost
        batch = {
            "cls": torch.tensor([9]),
            "bboxes": torch.tensor([[0.5, 0.5, 0.2, 0.2]]),
            "gt_groups": [1],
        }
        loss = crit((pred_bboxes.unsqueeze(0), pred_scores.unsqueeze(0)), batch)
        grads = torch.autograd.grad(
            loss["loss_reward"], (pred_scores, pred_bboxes), allow_unused=True
        )
        assert grads[0] is not None and grads[0].abs().sum() > 0  # class head
        assert grads[1] is not None and grads[1].abs().sum() > 0  # box head

    def test_include_misses_feeds_missed_critical_gt(self):
        """A missed critical GT gets a spare-query entry when include_misses=True."""
        comp = DimensionRewardComputer(
            head_type="sigmoid", broken_class_ids={9, 15}, skip_class_ids={14, 17}
        )
        crit = RewardRTDETRDetectionLoss(
            nc=20,
            reward_computer=comp,
            beta=1.0,
            include_misses=True,
            miss_threshold=0.5,
        )
        pred_bboxes = torch.tensor([[[0.5, 0.5, 0.2, 0.2]] * 4])
        # All queries are unconfident -> the matched GT counts as "missed".
        pred_scores = torch.full((1, 4, 20), -2.0)
        batch = {
            "cls": torch.tensor([9]),
            "bboxes": torch.tensor([[0.5, 0.5, 0.2, 0.2]]),
            "gt_groups": [1],
        }
        loss = crit((pred_bboxes.unsqueeze(0), pred_scores.unsqueeze(0)), batch)
        assert "loss_reward" in loss
        # 1 matched GT + 1 spare-query miss entry.
        assert crit.match_count_history[-1] == 2
        assert crit.miss_count_history[-1] == 1
        assert torch.isfinite(loss["loss_reward"])

    def test_include_misses_off_skips_spare_query(self):
        """Default (matched-only) never adds a spare-query miss entry."""
        comp = DimensionRewardComputer(
            head_type="sigmoid", broken_class_ids={9, 15}, skip_class_ids={14, 17}
        )
        crit = RewardRTDETRDetectionLoss(
            nc=20, reward_computer=comp, beta=1.0, include_misses=False
        )
        pred_bboxes = torch.tensor([[[0.5, 0.5, 0.2, 0.2]] * 4])
        pred_scores = torch.full((1, 4, 20), -2.0)
        batch = {
            "cls": torch.tensor([9]),
            "bboxes": torch.tensor([[0.5, 0.5, 0.2, 0.2]]),
            "gt_groups": [1],
        }
        crit((pred_bboxes.unsqueeze(0), pred_scores.unsqueeze(0)), batch)
        assert crit.match_count_history[-1] == 1
        assert crit.miss_count_history[-1] == 0


class _FakeRTDETRModel:
    """Minimal stand-in for ``RTDETRDetectionModel`` (just the attrs used)."""

    nc = 20
    criterion = None


class TestAttachRewardCriterion:
    """Tests for the criterion-install helper."""

    def test_attach_patches_criterion_and_init(self):
        model = _FakeRTDETRModel()
        comp = DimensionRewardComputer(head_type="sigmoid")
        crit = attach_reward_criterion(model, comp, beta=0.5)
        assert model.criterion is crit
        assert model.init_criterion() is crit
