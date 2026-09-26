"""
Tests for 7-dimension reward computation (differentiable).
"""

import pytest
import torch

from backend.training.dimension_rewards import (
    DIMENSION_NAMES,
    NUM_DIMENSIONS,
    DimensionRewardComputer,
)


class TestDimensionRewards:
    """Tests for individual dimension reward computation."""

    @pytest.fixture
    def computer(self):
        # Use class ids 0/1 in isolation to test the D06/D07 dimension logic.
        return DimensionRewardComputer(
            lambda_miss=1.5,
            lambda_fp=1.0,
            broken_class_ids={0},
            skip_class_ids={1},
        )

    @pytest.fixture
    def sample_predictions(self):
        return {
            "boxes": torch.tensor(
                [[100, 200, 80, 60], [500, 300, 40, 40]], dtype=torch.float32
            ),
            # Sample 0 -> class 0 (broken), sample 1 -> class 1 (skip).
            "logits": torch.tensor([[2.0, -2.0], [-2.0, 2.0]], dtype=torch.float32),
            "classes": torch.tensor([0, 1]),
            "confidences": torch.tensor([0.95, 0.70]),
        }

    @pytest.fixture
    def sample_targets(self):
        return [
            {"boxes": [[100, 200, 80, 60]], "labels": [0]},  # broken
            {"boxes": [[500, 300, 40, 40]], "labels": [1]},  # skip
        ]

    # ----------------------------------------------------------------
    # D01: Localization (soft IoU)
    # ----------------------------------------------------------------

    def test_loc_reward_perfect_iou(self, computer, sample_predictions, sample_targets):
        """Perfect IoU should give reward near +1.0."""
        reward = computer.compute_loc_reward(sample_predictions, sample_targets)
        assert reward.item() > 0.9, f"Expected >0.9, got {reward.item():.4f}"

    def test_loc_reward_no_boxes(self, computer):
        """No boxes should give -1.0."""
        reward = computer.compute_loc_reward(
            {"boxes": torch.zeros(0, 4)},
            [{"boxes": [], "labels": []}],
        )
        assert reward.item() == -1.0

    # ----------------------------------------------------------------
    # D02: Classification (smooth accuracy)
    # ----------------------------------------------------------------

    def test_cls_reward_all_correct(self, computer, sample_predictions, sample_targets):
        """All correct classifications should give positive reward."""
        reward = computer.compute_cls_reward(sample_predictions, sample_targets)
        assert reward.item() > 0

    def test_cls_reward_no_predictions(self, computer):
        """No predictions should give -1.0."""
        reward = computer.compute_cls_reward(
            {"classes": torch.zeros(0, dtype=torch.long)},
            [{"labels": [0]}],
        )
        assert reward.item() == -1.0

    # ----------------------------------------------------------------
    # D03: Calibration
    # ----------------------------------------------------------------

    def test_cal_reward_valid_range(self, computer, sample_predictions, sample_targets):
        """Calibration reward should be <= 0."""
        reward = computer.compute_cal_reward(sample_predictions, sample_targets)
        assert reward.item() <= 0.0

    # ----------------------------------------------------------------
    # D04: Miss Detection (soft)
    # ----------------------------------------------------------------

    def test_miss_reward_no_misses(self, computer, sample_predictions, sample_targets):
        """When all targets are localized, reward should be ~0."""
        reward = computer.compute_miss_reward(sample_predictions, sample_targets)
        assert abs(reward.item()) < 0.05, f"Expected ~0, got {reward.item():.4f}"

    def test_miss_reward_with_misses(self, computer):
        """Missed targets should produce negative reward."""
        reward = computer.compute_miss_reward(
            {"boxes": torch.tensor([[0, 0, 10, 10]], dtype=torch.float32)},
            [{"boxes": [[100, 100, 50, 50]]}],  # Far away -> not detected
        )
        assert reward.item() < 0.0

    # ----------------------------------------------------------------
    # D05: False Positive (soft)
    # ----------------------------------------------------------------

    def test_fp_reward_no_fps(self, computer, sample_predictions, sample_targets):
        """When all predictions are well localized, reward should be ~0."""
        reward = computer.compute_fp_reward(sample_predictions, sample_targets)
        assert abs(reward.item()) < 0.05, f"Expected ~0, got {reward.item():.4f}"

    # ----------------------------------------------------------------
    # D06 & D07: Sensitivity (soft)
    # ----------------------------------------------------------------

    def test_broken_detected(self, computer):
        """High probability on the broken class should give reward near +1.0."""
        reward = computer.compute_broken_reward(
            {"logits": torch.tensor([[10.0, -10.0]])},
            [{"labels": [0]}],
        )
        assert abs(reward.item() - 1.0) < 0.05, (
            f"Expected ~1.0, got {reward.item():.4f}"
        )

    def test_broken_missed(self, computer):
        """Low probability on the broken class should give reward near -2.0."""
        reward = computer.compute_broken_reward(
            {"logits": torch.tensor([[-10.0, 10.0]])},
            [{"labels": [0]}],
        )
        assert abs(reward.item() - (-2.0)) < 0.05, (
            f"Expected ~-2.0, got {reward.item():.4f}"
        )

    def test_skip_detected(self, computer):
        """High probability on the skip class should give reward near +1.0."""
        reward = computer.compute_skip_reward(
            {"logits": torch.tensor([[-10.0, 10.0]])},
            [{"labels": [1]}],
        )
        assert abs(reward.item() - 1.0) < 0.05, (
            f"Expected ~1.0, got {reward.item():.4f}"
        )

    def test_skip_missed(self, computer):
        """Low probability on the skip class should give reward near -2.0."""
        reward = computer.compute_skip_reward(
            {"logits": torch.tensor([[10.0, -10.0]])},
            [{"labels": [1]}],
        )
        assert abs(reward.item() - (-2.0)) < 0.05, (
            f"Expected ~-2.0, got {reward.item():.4f}"
        )

    def test_default_class_ids(self):
        """Default D06/D07 ids should be the Tianchi broken/skip class sets."""
        comp = DimensionRewardComputer()
        assert comp.broken_class_ids == {9, 15}
        assert comp.skip_class_ids == {14, 17}

    # ----------------------------------------------------------------
    # Full 7-dim vector
    # ----------------------------------------------------------------

    def test_all_rewards_shape(self, computer, sample_predictions, sample_targets):
        """Should return 7 rewards."""
        rewards = computer.compute_all(sample_predictions, sample_targets)
        assert len(rewards) == NUM_DIMENSIONS
        for name in DIMENSION_NAMES:
            assert name in rewards
            assert isinstance(rewards[name], torch.Tensor)


class TestDifferentiability:
    """Regression: the reward tensors must backpropagate into the model."""

    def test_rewards_backprop_to_logits_and_boxes(self):
        comp = DimensionRewardComputer(broken_class_ids={0}, skip_class_ids={1})
        logits = torch.tensor([[2.0, -2.0], [-2.0, 2.0]], requires_grad=True)
        boxes = torch.tensor(
            [[100, 200, 80, 60], [500, 300, 40, 40]],
            dtype=torch.float32,
            requires_grad=True,
        )
        preds = {
            "logits": logits,
            "boxes": boxes,
            "classes": logits.argmax(dim=-1),
            "confidences": logits.softmax(dim=-1).max(dim=-1).values,
        }
        targets = [
            {"boxes": [[100, 200, 80, 60]], "labels": [0]},
            {"boxes": [[500, 300, 40, 40]], "labels": [1]},
        ]

        rewards = comp.compute_all(preds, targets)
        total = torch.stack([rewards[name] for name in DIMENSION_NAMES]).sum()
        grads = torch.autograd.grad(total, (logits, boxes), allow_unused=True)

        assert grads[0] is not None, "reward must produce gradients w.r.t. logits"
        assert grads[1] is not None, "reward must produce gradients w.r.t. boxes"
        assert grads[0].abs().sum() > 0
        assert grads[1].abs().sum() > 0


class TestBoxIoU:
    """Tests for the non-differentiable IoU helper."""

    def test_perfect_iou(self):
        """Identical boxes should have IoU = 1.0."""
        iou = DimensionRewardComputer._box_iou([0, 0, 100, 100], [0, 0, 100, 100])
        assert iou == 1.0

    def test_no_overlap(self):
        """Non-overlapping boxes should have IoU = 0.0."""
        iou = DimensionRewardComputer._box_iou([0, 0, 10, 10], [100, 100, 10, 10])
        assert iou == 0.0

    def test_half_overlap(self):
        """Half-overlapping boxes."""
        iou = DimensionRewardComputer._box_iou([0, 0, 10, 10], [5, 0, 10, 10])
        assert abs(iou - 1 / 3) < 0.01


class TestSigmoidHead:
    """RT-DETR's per-class focal/vfl head uses sigmoid (not softmax)."""

    def test_sigmoid_cls_reward_uses_per_class_confidence(self):
        comp = DimensionRewardComputer(head_type="sigmoid")
        logits = torch.tensor([[5.0, -5.0]])  # sigmoid(5) ~ 0.993
        reward = comp.compute_cls_reward({"logits": logits}, [{"labels": [0]}])
        # p_true = sigmoid(5) -> reward = 2 * 0.993 - 1 ~ 0.987
        assert reward.item() > 0.9

    def test_sigmoid_sensitivity_reward_bounded(self):
        comp = DimensionRewardComputer(
            head_type="sigmoid", broken_class_ids={0}, skip_class_ids={1}
        )
        # Both classes 0 and 1 highly confident: p_crit = max (bounded <= 1).
        logits = torch.tensor([[5.0, 5.0]])
        reward = comp.compute_broken_reward({"logits": logits}, [{"labels": [0]}])
        assert reward.item() <= 1.0

    def test_default_stays_softmax(self):
        """Default head_type must remain softmax (backward compatible)."""
        assert DimensionRewardComputer().head_type == "softmax"
