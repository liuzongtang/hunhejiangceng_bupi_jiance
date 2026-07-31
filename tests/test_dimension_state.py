"""
Unit tests for DimensionState.
Tests the mathematical correctness of D vector updates.
"""

import numpy as np
import pytest
import torch

from mixed_reward.dimension.state import DimensionState, RunningBaseline


class TestRunningBaseline:
    """Tests for the EMA baseline tracker."""

    def test_initialization(self):
        baseline = RunningBaseline(num_dimensions=4, momentum=0.99)
        assert baseline.value.shape == (4,)
        assert torch.allclose(baseline.value, torch.ones(4) * 0.5)

    def test_first_update(self):
        baseline = RunningBaseline(num_dimensions=4)
        reward = torch.tensor([0.8, 0.6, 0.9, 0.7])
        result = baseline.update(reward)
        # First update applies EMA: 0.99*0.5 + 0.01*reward
        expected = 0.99 * torch.ones(4) * 0.5 + 0.01 * reward
        assert torch.allclose(result, expected)
        assert baseline._initialized is True

    def test_ema_behavior(self):
        baseline = RunningBaseline(num_dimensions=1, momentum=0.9, init_value=0.5)
        baseline.update(torch.tensor([1.0]))
        # After update: 0.9*0.5 + 0.1*1.0 = 0.45 + 0.10 = 0.55
        assert torch.allclose(baseline.value, torch.tensor([0.55]), atol=1e-6)

    def test_reset(self):
        baseline = RunningBaseline(num_dimensions=3)
        baseline.update(torch.tensor([0.7, 0.8, 0.9]))
        baseline.reset()
        assert not baseline._initialized


class TestDimensionState:
    """Tests for the core DimensionState class."""

    @pytest.fixture
    def dim_state(self):
        return DimensionState(
            num_dimensions=4,
            init_value=1.0,
            d_min=0.5,
            d_max=2.0,
            forget_factor=0.999,
            learning_rate=0.01,
            dimension_names=["accuracy", "safety", "completeness", "format"],
        )

    def test_initialization(self, dim_state):
        """D should initialize to [1.0, 1.0, 1.0, 1.0]."""
        D = dim_state.get_state()
        assert D.shape == (4,)
        assert torch.allclose(D, torch.ones(4))

    def test_initialization_custom(self):
        ds = DimensionState(num_dimensions=3, init_value=1.5)
        assert torch.allclose(ds.get_state(), torch.ones(3) * 1.5)

    def test_update_shapes(self, dim_state):
        """Update should work with various input types."""
        # Tensor
        D1 = dim_state.update(torch.tensor([0.8, 0.6, 0.9, 0.7]))
        assert D1.shape == (4,)

        # List
        dim_state.reset()
        D2 = dim_state.update([0.7, 0.5, 0.8, 0.6])
        assert D2.shape == (4,)

        # Numpy
        dim_state.reset()
        D3 = dim_state.update(np.array([0.9, 0.4, 0.7, 0.8]))
        assert D3.shape == (4,)

    def test_update_type_error(self, dim_state):
        """Should raise TypeError for invalid input."""
        with pytest.raises(TypeError):
            dim_state.update("invalid")

    def test_update_dimension_mismatch(self, dim_state):
        """Should raise ValueError for wrong number of dimensions."""
        with pytest.raises(ValueError):
            dim_state.update(torch.tensor([0.5, 0.5, 0.5]))  # 3 dims, expected 4

    def test_clamping(self):
        """D should never exceed [d_min, d_max] regardless of input."""
        ds = DimensionState(num_dimensions=1, d_min=0.5, d_max=2.0, learning_rate=1.0)

        # Push D up with very high rewards (relative to baseline of 0.5)
        for _ in range(1000):
            ds.update(torch.tensor([1.0]), stage="train")

        D = ds.get_state()
        assert D[0].item() <= 2.01  # Allow small floating point error

        # Reset and push D down with very low rewards
        ds.reset()
        for _ in range(1000):
            ds.update(torch.tensor([0.0]), stage="train")

        D = ds.get_state()
        assert D[0].item() >= 0.49  # Allow small floating point error

    def test_forget_decay(self):
        """D should drift toward 1.0 when rewards are neutral (0.5)."""
        ds = DimensionState(
            num_dimensions=1,
            init_value=2.0,  # Start amplified
            forget_factor=0.9,  # Aggressive forget for testing
            learning_rate=0.0,  # No update influence
        )

        D_before = ds.get_state()[0].item()
        assert D_before == 2.0

        # With neutral rewards and no learning, D should move toward 1.0
        ds.update(torch.tensor([0.5]))
        D_after = ds.get_state()[0].item()

        # D = 2.0 * 0.9 + (1 - 0.9) * 1.0 = 1.8 + 0.1 = 1.9
        expected = 2.0 * 0.9 + 0.1 * 1.0
        assert abs(D_after - expected) < 1e-6

    def test_forget_converges_to_one(self):
        """After many forget steps with no signal, D should converge to 1.0."""
        ds = DimensionState(
            num_dimensions=1,
            init_value=2.0,
            forget_factor=0.9,
            learning_rate=0.0,
        )

        for _ in range(100):
            ds.update(torch.tensor([0.5]))

        D = ds.get_state()[0].item()
        # After many steps: D → 1.0 (geometric series)
        assert abs(D - 1.0) < 1e-3

    def test_history_tracking(self, dim_state):
        """History should record D after each update."""
        for i in range(10):
            dim_state.update(torch.tensor([0.5 + i * 0.05] * 4))

        history = dim_state.get_history()
        assert history.shape == (10, 4)
        assert dim_state.step_count == 10

    def test_gradient_scale_round_robin(self, dim_state):
        """get_gradient_scale should map param groups to dimensions via mod."""
        # Set D to known values
        dim_state.D = torch.tensor([1.1, 0.9, 1.5, 0.7])
        dim_state.num_dimensions = 4

        # param_group 0 → dim 0, pg 1 → dim 1, pg 4 → dim 0 (wrap)
        assert dim_state.get_gradient_scale(0) == pytest.approx(1.1)
        assert dim_state.get_gradient_scale(1) == pytest.approx(0.9)
        assert dim_state.get_gradient_scale(2) == pytest.approx(1.5)
        assert dim_state.get_gradient_scale(3) == pytest.approx(0.7)
        assert dim_state.get_gradient_scale(4) == pytest.approx(1.1)  # Wrap

    def test_state_dict_roundtrip(self, dim_state):
        """Checkpoint save/load should restore state."""
        dim_state.update(torch.tensor([0.8, 0.7, 0.6, 0.9]))
        dim_state.update(torch.tensor([0.9, 0.8, 0.7, 0.5]))

        # Save
        state = dim_state.get_state_dict()
        D_before = dim_state.get_state().clone()

        # Create new instance and load
        ds2 = DimensionState(num_dimensions=4)
        ds2.load_state_dict(state)

        assert torch.allclose(ds2.get_state(), D_before)
        assert ds2.step_count == dim_state.step_count

    def test_reset(self, dim_state):
        """Reset should restore initial state."""
        dim_state.update(torch.tensor([0.9, 0.8, 0.7, 0.6]))
        dim_state.reset()

        assert torch.allclose(dim_state.get_state(), torch.ones(4))
        assert len(dim_state.history) == 0
        assert dim_state.step_count == 0

    def test_summary(self, dim_state):
        """Summary should be a non-empty string."""
        dim_state.update(torch.tensor([0.8, 0.7, 0.6, 0.9]))
        s = dim_state.summary()
        assert isinstance(s, str)
        assert len(s) > 0
        assert "accuracy" in s

    def test_repr(self, dim_state):
        """repr should include dimension names and values."""
        r = repr(dim_state)
        assert "DimensionState" in r

    # ========================================================================
    # Behavioral tests: D vector dynamics
    # ========================================================================

    def test_high_reward_amplifies(self):
        """Persistently high rewards should push D above 1.0."""
        ds = DimensionState(num_dimensions=1, learning_rate=0.05, forget_factor=0.99)

        # Feed consistently high rewards
        for _ in range(50):
            ds.update(torch.tensor([0.9]))

        D = ds.get_state()[0].item()
        assert D > 1.0, f"Expected D > 1.0 for high rewards, got {D}"

    def test_low_reward_suppresses(self):
        """Persistently low rewards should push D below 1.0."""
        ds = DimensionState(num_dimensions=1, learning_rate=0.05, forget_factor=0.99)

        # Feed consistently low rewards
        for _ in range(50):
            ds.update(torch.tensor([0.1]))

        D = ds.get_state()[0].item()
        assert D < 1.0, f"Expected D < 1.0 for low rewards, got {D}"

    def test_neutral_reward_maintains(self):
        """Neutral rewards should keep D near 1.0."""
        ds = DimensionState(num_dimensions=1, learning_rate=0.01, forget_factor=0.99)

        for _ in range(50):
            ds.update(torch.tensor([0.5]))

        D = ds.get_state()[0].item()
        assert 0.95 <= D <= 1.05, f"Expected D ~= 1.0, got {D}"

    def test_no_baseline_mode(self):
        """Without baseline, relative performance = reward - 0.5."""
        ds = DimensionState(
            num_dimensions=1,
            use_baseline=False,
            learning_rate=0.1,
            forget_factor=1.0,  # No forget for this test
        )
        ds.update(torch.tensor([0.8]))  # Above 0.5 → should amplify

        # D = 1.0 * (1 + 0.1 * tanh(0.8 - 0.5))
        # D = 1.0 * (1 + 0.1 * tanh(0.3))
        # tanh(0.3) ≈ 0.2913
        # D ≈ 1.0 * 1.02913 ≈ 1.02913
        expected = 1.0 * (1.0 + 0.1 * np.tanh(0.3))
        D = ds.get_state()[0].item()
        assert abs(D - expected) < 1e-4

    def test_multi_dimension_independence(self):
        """Different dimensions should evolve independently."""
        ds = DimensionState(
            num_dimensions=2,
            init_value=1.0,
            learning_rate=0.05,
            forget_factor=0.99,
            use_baseline=False,
        )

        # Dim 0 gets high reward, Dim 1 gets low reward
        for _ in range(30):
            ds.update(torch.tensor([0.9, 0.1]))

        D = ds.get_state()
        assert D[0].item() > 1.0, f"Dim 0 should be amplified, got {D[0]}"
        assert D[1].item() < 1.0, f"Dim 1 should be suppressed, got {D[1]}"
