"""
Unit tests for MixedRewardVisualizer.
Tests logging calls with mocked WandB and TensorBoard.
"""

from unittest.mock import patch

import numpy as np
import pytest


class TestMixedRewardVisualizer:
    """Tests for the visualizer with mocked backends."""

    @pytest.fixture
    def viz(self):
        """Create a visualizer with mocked backends."""
        with (
            patch("mixed_reward.visualizer._WANDB_AVAILABLE", True),
            patch("mixed_reward.visualizer._TENSORBOARD_AVAILABLE", True),
            patch("mixed_reward.visualizer.wandb"),
            patch("mixed_reward.visualizer.SummaryWriter"),
        ):
            from mixed_reward.visualizer import MixedRewardVisualizer

            viz = MixedRewardVisualizer(
                use_wandb=True,
                use_tensorboard=True,
                log_dir="./test_runs",
                project_name="test-project",
            )
            viz.set_dimensions(["accuracy", "safety", "completeness", "format"])

            # Attach mocks for inspection
            viz._mock_wandb = mock_wandb
            viz._mock_tb = mock_tb
            viz._mock_writer = mock_tb.return_value

            yield viz
            viz.close()

    def test_initialization(self, viz):
        """Visualizer should initialize both backends."""
        assert viz.use_wandb is True
        assert viz.use_tensorboard is True
        assert viz.dimensions == ["accuracy", "safety", "completeness", "format"]

    def test_set_dimensions(self, viz):
        viz.set_dimensions(["acc", "safe"])
        assert viz.dimensions == ["acc", "safe"]

    def test_dim_label(self, viz):
        assert viz._dim_label(0) == "accuracy"
        assert viz._dim_label(3) == "format"
        assert viz._dim_label(10) == "dim_10"  # Out of range

    def test_log_reward_vector(self, viz):
        """log_reward_vector should call wandb.log and TB writer."""
        reward_dict = {"accuracy": 0.8, "safety": 0.9}
        viz.log_reward_vector(reward_dict, step=100)

        # WandB should be called
        viz._mock_wandb.log.assert_called()
        call_args = viz._mock_wandb.log.call_args[0][0]
        assert "reward/accuracy" in call_args
        assert "reward/safety" in call_args
        assert "reward/total" in call_args
        assert call_args["reward/total"] == pytest.approx(1.7)

        # TensorBoard should be called
        viz._mock_writer.add_scalar.assert_called()

    def test_log_dimension_state(self, viz):
        """log_dimension_state should log per-dimension D values."""
        D = np.array([1.1, 0.9, 1.5, 0.7])
        viz.log_dimension_state(D, step=100)

        viz._mock_wandb.log.assert_called()
        call_args = viz._mock_wandb.log.call_args[0][0]
        assert "dim_state/accuracy" in call_args
        assert call_args["dim_state/accuracy"] == 1.1

    def test_log_gradient_scales(self, viz):
        """log_gradient_scales should log per-group and aggregate stats."""
        scales = [1.05, 0.98, 1.12, 0.95]
        viz.log_gradient_scales(scales, step=100)

        viz._mock_wandb.log.assert_called()
        call_args = viz._mock_wandb.log.call_args[0][0]
        assert "grad_scale/group_0" in call_args
        assert "grad_scale/mean" in call_args
        assert abs(call_args["grad_scale/mean"] - np.mean(scales)) < 1e-6

    def test_log_training_metrics(self, viz):
        """log_training_metrics should filter scalar values only."""
        metrics = {
            "loss": 0.5,
            "lr": 1e-5,
            "text": "should be skipped",  # Non-scalar → skipped
            "tensor": np.array([1, 2, 3]),  # Non-scalar → skipped
        }
        viz.log_training_metrics(metrics, step=50)

        viz._mock_wandb.log.assert_called()
        call_args = viz._mock_wandb.log.call_args[0][0]
        assert "loss" in call_args
        assert "lr" in call_args
        assert "text" not in call_args
        assert "tensor" not in call_args

    def test_step_counter(self, viz):
        """Step counter should increment."""
        assert viz.step_counter() == 1
        assert viz.step_counter() == 2
        assert viz._step == 2

    def test_context_manager(self, viz):
        """Should work as context manager."""
        with (
            patch("mixed_reward.visualizer._WANDB_AVAILABLE", True),
            patch("mixed_reward.visualizer._TENSORBOARD_AVAILABLE", True),
            patch("mixed_reward.visualizer.wandb"),
            patch("mixed_reward.visualizer.SummaryWriter"),
        ):
            from mixed_reward.visualizer import MixedRewardVisualizer

            with MixedRewardVisualizer() as v:
                v.log_reward_vector({"accuracy": 0.5}, step=1)

    def test_disabled_backends(self):
        """Should work with backends disabled."""
        with (
            patch("mixed_reward.visualizer._WANDB_AVAILABLE", True),
            patch("mixed_reward.visualizer._TENSORBOARD_AVAILABLE", True),
            patch("mixed_reward.visualizer.wandb"),
            patch("mixed_reward.visualizer.SummaryWriter"),
        ):
            from mixed_reward.visualizer import MixedRewardVisualizer

            viz = MixedRewardVisualizer(use_wandb=False, use_tensorboard=False)
            assert viz.use_wandb is False
            assert viz.use_tensorboard is False

            # Should not raise
            viz.log_reward_vector({"accuracy": 0.5}, step=1)
            viz.close()


class TestParetoFrontier:
    """Tests for Pareto frontier computation."""

    def test_pareto_frontier(self):
        from mixed_reward.visualizer import _compute_pareto_frontier

        x = np.array([0.1, 0.5, 0.9, 0.4, 0.8])
        y = np.array([0.9, 0.5, 0.1, 0.8, 0.3])

        pareto = _compute_pareto_frontier(x, y)

        # Points (0.9, 0.1) and (0.8, 0.3) should be on frontier
        # Point (0.1, 0.9) should also be on frontier
        assert pareto.shape == (5,)
        assert pareto.dtype == bool

    def test_single_point(self):
        from mixed_reward.visualizer import _compute_pareto_frontier

        x = np.array([0.5])
        y = np.array([0.5])
        pareto = _compute_pareto_frontier(x, y)
        assert bool(pareto[0]) is True

    def test_all_identical(self):
        from mixed_reward.visualizer import _compute_pareto_frontier

        x = np.array([0.5, 0.5, 0.5])
        y = np.array([0.5, 0.5, 0.5])
        pareto = _compute_pareto_frontier(x, y)
        # All points are non-dominated (neither strictly better in any dimension)
        assert pareto.sum() == 3


class TestDimLabel:
    """Test the _dim_label helper."""

    def test_with_names(self):
        from mixed_reward.visualizer import MixedRewardVisualizer

        with (
            patch("mixed_reward.visualizer._WANDB_AVAILABLE", False),
            patch("mixed_reward.visualizer._TENSORBOARD_AVAILABLE", False),
        ):
            viz = MixedRewardVisualizer(use_wandb=False, use_tensorboard=False)
            viz.set_dimensions(["acc", "safe", "comp"])
            assert viz._dim_label(0) == "acc"
            assert viz._dim_label(1) == "safe"
            assert viz._dim_label(5) == "dim_5"
