"""
Unit tests for MultiDimensionalRewardManager.
"""

from unittest.mock import MagicMock

import pytest
import torch

from mixed_reward.reward_manager.multi_dimensional import (
    MultiDimensionalRewardManager,
    register_scorer,
)


class TestMultiDimensionalRewardManager:
    """Tests for the reward manager."""

    @pytest.fixture
    def manager(self):
        return MultiDimensionalRewardManager(
            dimensions=["accuracy", "safety", "completeness", "format"],
            weights=[1.0, 1.0, 0.5, 0.5],
        )

    @pytest.fixture
    def mock_scorers(self):
        """Create mock scorers returning known values."""
        return {
            "accuracy": lambda response,
            ground_truth=None,
            question=None,
            **kwargs: 0.8,
            "safety": lambda response, ground_truth=None, question=None, **kwargs: 0.9,
            "completeness": lambda response,
            ground_truth=None,
            question=None,
            **kwargs: 0.7,
            "format": lambda response, ground_truth=None, question=None, **kwargs: 0.6,
        }

    def test_initialization_default(self):
        manager = MultiDimensionalRewardManager()
        assert manager.dimensions == ["accuracy", "safety", "completeness", "format"]
        assert manager.weights == [1.0, 1.0, 1.0, 1.0]
        assert manager.num_dims == 4

    def test_initialization_custom(self):
        manager = MultiDimensionalRewardManager(
            dimensions=["accuracy", "safety"],
            weights=[1.0, 2.0],
        )
        assert manager.dimensions == ["accuracy", "safety"]
        assert manager.weights == [1.0, 2.0]

    def test_initialization_weight_padding(self):
        """Weights that are too short should be padded with 1.0."""
        manager = MultiDimensionalRewardManager(
            dimensions=["a", "b", "c"],
            weights=[0.5],
        )
        assert manager.weights == [0.5, 1.0, 1.0]

    def test_initialization_weight_truncation(self):
        """Weights that are too long should be truncated."""
        manager = MultiDimensionalRewardManager(
            dimensions=["a", "b"],
            weights=[1.0, 2.0, 3.0, 4.0],
        )
        assert manager.weights == [1.0, 2.0]

    def test_weight_normalization(self):
        manager = MultiDimensionalRewardManager(
            dimensions=["a", "b"],
            weights=[1.0, 1.0],
            normalize_weights=True,
        )
        assert manager.weights == [0.5, 0.5]

    def test_compute_reward_vector_mocked(self, manager, mock_scorers):
        """With mocked scorers, should return known values."""
        manager.scorer_overrides = mock_scorers
        reward_dict = manager.compute_reward_vector(
            response="test response",
            ground_truth="test truth",
            question="test question",
        )
        assert reward_dict["accuracy"] == 0.8
        assert reward_dict["safety"] == 0.9
        assert reward_dict["completeness"] == 0.7
        assert reward_dict["format"] == 0.6

    def test_get_total_reward(self, manager):
        """Total should be weighted sum."""
        reward_dict = {
            "accuracy": 0.8,
            "safety": 0.9,
            "completeness": 0.7,
            "format": 0.6,
        }
        # 1.0*0.8 + 1.0*0.9 + 0.5*0.7 + 0.5*0.6 = 0.8 + 0.9 + 0.35 + 0.30 = 2.35
        total = manager.get_total_reward(reward_dict)
        assert total == pytest.approx(2.35)

    def test_get_reward_tensor(self, manager):
        """Should return tensor with correct shape and order."""
        reward_dict = {
            "accuracy": 0.8,
            "safety": 0.9,
            "completeness": 0.7,
            "format": 0.6,
        }
        tensor = manager.get_reward_tensor(reward_dict)
        assert tensor.shape == (4,)
        assert torch.allclose(tensor, torch.tensor([0.8, 0.9, 0.7, 0.6]))

    def test_compute_batch(self, manager, mock_scorers):
        """Batch computation should return correct number of results."""
        manager.scorer_overrides = mock_scorers
        results = manager.compute_batch(
            responses=["r1", "r2", "r3"],
            ground_truths=["g1", "g2", "g3"],
        )
        assert len(results) == 3
        for r in results:
            assert set(r.keys()) == {"accuracy", "safety", "completeness", "format"}

    def test_get_batch_tensors(self, manager, mock_scorers):
        """Batch tensor conversion should have correct shapes."""
        manager.scorer_overrides = mock_scorers
        results = manager.compute_batch(responses=["r1", "r2", "r3"])
        tensors = manager.get_batch_tensors(results)

        assert tensors["reward_vectors"].shape == (3, 4)
        assert tensors["total_rewards"].shape == (3,)

    def test_get_average_reward_vector(self, manager, mock_scorers):
        """Average reward vector should be mean across batch."""
        manager.scorer_overrides = mock_scorers
        results = manager.compute_batch(responses=["r1", "r2"])
        avg = manager.get_average_reward_vector(results)
        assert avg.shape == (4,)

    def test_custom_scorer_override(self):
        """Custom scorer override should take precedence."""
        custom_scorer = MagicMock(return_value=0.42)
        manager = MultiDimensionalRewardManager(
            dimensions=["accuracy", "custom_dim"],
            scorer_overrides={"custom_dim": custom_scorer},
        )
        reward_dict = manager.compute_reward_vector(response="test")
        assert reward_dict["custom_dim"] == 0.42
        custom_scorer.assert_called_once()

    def test_register_scorer(self):
        """Custom scorers registered via register_scorer should work."""

        def my_scorer(response, **kwargs):
            return 0.99

        register_scorer("my_dimension", my_scorer)
        manager = MultiDimensionalRewardManager(dimensions=["my_dimension"])
        reward_dict = manager.compute_reward_vector(response="test")
        assert reward_dict["my_dimension"] == 0.99

    def test_unknown_dimension_defaults_to_zero(self):
        """Dimensions without registered scorers should return 0.0."""
        manager = MultiDimensionalRewardManager(
            dimensions=["nonexistent_dim"],
        )
        reward_dict = manager.compute_reward_vector(response="test")
        assert reward_dict["nonexistent_dim"] == 0.0

    def test_scorer_error_handling(self, manager):
        """Scorer exceptions should be caught and default to 0.0."""

        # Register a scorer that always raises
        def failing_scorer(**kwargs):
            raise RuntimeError("Scorer failed")

        manager.scorer_overrides = {"accuracy": failing_scorer}
        reward_dict = manager.compute_reward_vector(response="test")
        assert reward_dict["accuracy"] == 0.0  # Should fallback
        # Other dimensions should still work
        assert "safety" in reward_dict

    def test_score_clamped_to_range(self, manager):
        """Scores should be clamped to [0, 1]."""

        def out_of_range_scorer(**kwargs):
            return 5.0  # Above 1.0

        manager.scorer_overrides = {"accuracy": out_of_range_scorer}
        reward_dict = manager.compute_reward_vector(response="test")
        assert 0.0 <= reward_dict["accuracy"] <= 1.0

        def negative_scorer(**kwargs):
            return -3.0

        manager.scorer_overrides = {"accuracy": negative_scorer}
        reward_dict = manager.compute_reward_vector(response="test")
        assert 0.0 <= reward_dict["accuracy"] <= 1.0

    def test_summary(self, manager):
        """Summary should return a string."""
        reward_dict = {
            "accuracy": 0.8,
            "safety": 0.9,
            "completeness": 0.7,
            "format": 0.6,
        }
        s = manager.summary(reward_dict)
        assert isinstance(s, str)
        assert "accuracy" in s

    def test_repr(self, manager):
        r = repr(manager)
        assert "MultiDimensionalRewardManager" in r
        assert "accuracy" in r
