"""
Tests for 7-dimension reward computation.
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
            "classes": torch.tensor([0, 1]),  # 0=broken, 1=skip
            "confidences": torch.tensor([0.95, 0.70]),
        }

    @pytest.fixture
    def sample_targets(self):
        return [
            {"boxes": [[100, 200, 80, 60]], "labels": [0]},  # broken
            {"boxes": [[500, 300, 40, 40]], "labels": [1]},  # skip
        ]

    # ----------------------------------------------------------------
    # D01: Localization (IoU)
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
    # D02: Classification
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
    # D03: ECE
    # ----------------------------------------------------------------

    def test_cal_reward_valid_range(self, computer, sample_predictions, sample_targets):
        """ECE reward should be <= 0 (negative or zero ECE → negative reward)."""
        reward = computer.compute_cal_reward(sample_predictions, sample_targets)
        # ECE >= 0, so -ECE <= 0
        assert reward.item() <= 0.0

    # ----------------------------------------------------------------
    # D04: Miss Detection
    # ----------------------------------------------------------------

    def test_miss_reward_no_misses(self, computer, sample_predictions, sample_targets):
        """When all targets are detected, reward should be 0."""
        reward = computer.compute_miss_reward(sample_predictions, sample_targets)
        assert reward.item() == 0.0

    def test_miss_reward_with_misses(self, computer):
        """Missed targets should produce negative reward."""
        reward = computer.compute_miss_reward(
            {"boxes": torch.tensor([[0, 0, 10, 10]], dtype=torch.float32)},
            [{"boxes": [[100, 100, 50, 50]]}],  # Far away → not detected
        )
        assert reward.item() < 0.0

    # ----------------------------------------------------------------
    # D05: False Positive
    # ----------------------------------------------------------------

    def test_fp_reward_no_fps(self, computer, sample_predictions, sample_targets):
        """When all predictions match targets, reward should be 0."""
        reward = computer.compute_fp_reward(sample_predictions, sample_targets)
        assert reward.item() == 0.0

    # ----------------------------------------------------------------
    # D06 & D07: Sensitivity
    # ----------------------------------------------------------------

    def test_broken_detected(self, computer):
        """Detecting a broken defect should give +1.0."""
        reward = computer.compute_broken_reward(
            {"classes": torch.tensor([0])},
            [{"labels": [0]}],
        )
        assert reward.item() == 1.0

    def test_broken_missed(self, computer):
        """Missing a broken defect should give -2.0."""
        reward = computer.compute_broken_reward(
            {"classes": torch.tensor([1])},
            [{"labels": [0]}],
        )
        assert reward.item() == -2.0

    def test_skip_detected(self, computer):
        """Detecting a skip defect should give +1.0."""
        reward = computer.compute_skip_reward(
            {"classes": torch.tensor([1])},
            [{"labels": [1]}],
        )
        assert reward.item() == 1.0

    def test_skip_missed(self, computer):
        """Missing a skip defect should give -2.0."""
        reward = computer.compute_skip_reward(
            {"classes": torch.tensor([0])},
            [{"labels": [1]}],
        )
        assert reward.item() == -2.0

    def test_default_class_ids(self):
        """Default D06/D07 ids should be the Tianchi broken/skip class sets."""
        comp = DimensionRewardComputer()
        assert comp.broken_class_ids == {9, 16}
        assert comp.skip_class_ids == {15, 19}

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


class TestBoxIoU:
    """Tests for IoU computation."""

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
        # Intersection: [5,0,10,10] → 5*10 = 50
        # Union: 100 + 100 - 50 = 150
        # IoU = 50/150 = 0.333...
        assert abs(iou - 1 / 3) < 0.01
