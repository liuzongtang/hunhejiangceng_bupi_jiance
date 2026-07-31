"""
Dimension State Manager.

Maintains and updates the dimension state vector D that modulates gradients
per parameter group. The D vector captures per-dimension training progress:
dimensions that are performing well get D > 1 (amplified gradients), while
struggling dimensions get D < 1 (suppressed gradients).

Core behaviors:
1. Forget: D decays toward 1.0 each step (prevents unbounded growth)
2. Update: D adjusts based on relative reward performance vs baseline
3. Clamp: D is clipped to [d_min, d_max] for stability
4. History: D trajectory is recorded for visualization

Formula:
    D_new = clamp(D_old * forget_factor + (1 - forget_factor) * 1.0
                   * (1 + lr * tanh(reward - baseline)),
                   d_min, d_max)
"""

from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch


class RunningBaseline:
    """Exponential moving average baseline for stable relative performance."""

    def __init__(
        self, num_dimensions: int, momentum: float = 0.99, init_value: float = 0.5
    ):
        self.num_dimensions = num_dimensions
        self.momentum = momentum
        self.value = torch.ones(num_dimensions) * init_value
        self._initialized = False

    def update(self, reward_vector: torch.Tensor) -> torch.Tensor:
        """Update baseline with new observation, return updated value."""
        if not self._initialized:
            # Apply EMA even on first update: blend init_value with first observation
            self.value = (
                self.momentum * self.value + (1 - self.momentum) * reward_vector
            )
            self._initialized = True
        else:
            self.value = (
                self.momentum * self.value + (1 - self.momentum) * reward_vector
            )
        return self.value

    def get(self) -> torch.Tensor:
        return self.value

    def reset(self):
        self._initialized = False


class DimensionState:
    """
    Dimension state manager for the hybrid scalar+dimension reward mechanism.

    Maintains a D vector ∈ [d_min, d_max]^N where N is the number of reward
    dimensions. Each entry D[i] modulates gradients for parameter groups mapped
    to dimension i.

    The D vector serves as a learned "gradient mask":
    - D[i] > 1.0 → amplify gradients for dimension i (rewarding good performance)
    - D[i] < 1.0 → suppress gradients for dimension i (penalizing poor performance)
    - D[i] ≈ 1.0 → neutral, no modulation
    """

    def __init__(
        self,
        num_dimensions: int,
        init_value: float = 1.0,
        d_min: float = 0.5,
        d_max: float = 2.0,
        forget_factor: float = 0.999,
        learning_rate: float = 0.01,
        use_baseline: bool = True,
        baseline_momentum: float = 0.99,
        dimension_names: Optional[List[str]] = None,
    ):
        """
        Args:
            num_dimensions: Number of reward dimensions.
            init_value: Initial value for all D entries (typically 1.0 = neutral).
            d_min: Minimum allowed D value (prevents gradient vanishing).
            d_max: Maximum allowed D value (prevents gradient explosion).
            forget_factor: Per-step decay factor toward 1.0.
                           At 0.999, a dimension at 2.0 decays to ~1.8 after 100
                           steps without reinforcement.
            learning_rate: Step size for D updates (controls responsiveness).
            use_baseline: If True, use running EMA baseline for relative scoring.
            baseline_momentum: EMA momentum for the baseline tracker.
            dimension_names: Optional human-readable names for logging.
        """
        if num_dimensions < 1:
            raise ValueError(f"num_dimensions must be >= 1, got {num_dimensions}")

        self.num_dimensions = num_dimensions
        self.init_value = init_value
        self.d_min = d_min
        self.d_max = d_max
        self.forget_factor = forget_factor
        self.learning_rate = learning_rate

        # Initialize D vector
        self.D = torch.ones(num_dimensions) * init_value

        # Running baseline for relative performance evaluation
        self.use_baseline = use_baseline
        self.baseline = (
            RunningBaseline(
                num_dimensions=num_dimensions,
                momentum=baseline_momentum,
            )
            if use_baseline
            else None
        )

        # Dimension names for logging
        self.dimension_names = dimension_names or [
            f"dim_{i}" for i in range(num_dimensions)
        ]

        # History tracking
        self.history: List[np.ndarray] = []
        self._step_count = 0

    def update(
        self,
        reward_vector: Union[torch.Tensor, np.ndarray, List[float]],
        stage: str = "train",
    ) -> torch.Tensor:
        """
        Update D vector based on current reward observations.

        Args:
            reward_vector: Per-dimension rewards [r1, r2, ..., rN], each in [0, 1].
            stage: Training stage label (for potential stage-aware logic).

        Returns:
            Updated D vector.
        """
        # Convert input to tensor
        if isinstance(reward_vector, np.ndarray):
            reward_vector = torch.from_numpy(reward_vector).float()
        elif isinstance(reward_vector, list):
            reward_vector = torch.tensor(reward_vector, dtype=torch.float32)
        elif not isinstance(reward_vector, torch.Tensor):
            raise TypeError(
                f"reward_vector must be Tensor, ndarray, or list, got {type(reward_vector)}"
            )

        if reward_vector.shape[-1] != self.num_dimensions:
            raise ValueError(
                f"reward_vector has {reward_vector.shape[-1]} dimensions, "
                f"expected {self.num_dimensions}"
            )

        # Ensure 1D
        reward_vector = reward_vector.reshape(-1)

        # Step 1: Forget — decay toward 1.0
        # D = D * forget_factor + (1 - forget_factor) * 1.0
        self.D = self.D * self.forget_factor + (1.0 - self.forget_factor) * 1.0

        # Step 2: Compute relative performance
        if self.use_baseline and self.baseline is not None:
            baseline = self.baseline.get()
            relative_performance = reward_vector - baseline
            # Update baseline AFTER computing relative performance
            self.baseline.update(reward_vector)
        else:
            # No baseline — compare against neutral 0.5
            relative_performance = reward_vector - 0.5

        # Step 3: Update D
        # Using tanh to bound the update magnitude regardless of reward scale
        # delta ∈ [-lr, +lr] for each dimension
        delta = self.learning_rate * torch.tanh(relative_performance)

        # Multiplicative update: D = D * (1 + delta)
        self.D = self.D * (1.0 + delta)

        # Step 4: Clamp to safe range
        self.D = torch.clamp(self.D, self.d_min, self.d_max)

        # Step 5: Record history
        self.history.append(self.D.clone().detach().numpy())
        self._step_count += 1

        return self.D

    def get_state(self) -> torch.Tensor:
        """Get current D vector."""
        return self.D

    def get_state_numpy(self) -> np.ndarray:
        """Get current D vector as numpy array."""
        return self.D.detach().numpy()

    def get_gradient_scale(self, param_group_idx: int) -> float:
        """
        Get the gradient scale factor for a parameter group.

        Maps param_group_idx → dimension_idx via round-robin assignment,
        then returns D[dim_idx].

        The gradient modulation formula is:
            scale = 1 + beta * (D[dim_idx] - 1)

        where beta is applied externally by the trainer.

        Args:
            param_group_idx: Index of the parameter group.

        Returns:
            Current D value for the corresponding dimension.
        """
        dim_idx = param_group_idx % self.num_dimensions
        return self.D[dim_idx].item()

    def get_dimension_scale(self, dim_idx: int) -> float:
        """Get D value for a specific dimension index."""
        return self.D[dim_idx].item()

    def get_history(self) -> np.ndarray:
        """
        Get full D vector history.

        Returns:
            Array of shape (steps, num_dimensions).
        """
        if not self.history:
            return np.array([])
        return np.stack(self.history)

    def get_state_dict(self) -> Dict[str, Any]:
        """Serialize state for checkpointing."""
        return {
            "D": self.D.clone(),
            "history": [h.copy() for h in self.history[-1000:]],  # Keep last 1000
            "step_count": self._step_count,
            "baseline": self.baseline.get().clone() if self.baseline else None,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Restore state from checkpoint."""
        self.D = state_dict["D"]
        self.history = state_dict.get("history", [])
        self._step_count = state_dict.get("step_count", 0)
        if state_dict.get("baseline") is not None and self.baseline is not None:
            self.baseline.value = state_dict["baseline"]
            self.baseline._initialized = True

    @property
    def step_count(self) -> int:
        """Number of update steps performed."""
        return self._step_count

    def reset(self):
        """Reset D vector and history to initial state."""
        self.D = torch.ones(self.num_dimensions) * self.init_value
        self.history = []
        self._step_count = 0
        if self.baseline is not None:
            self.baseline.reset()

    def summary(self) -> str:
        """Human-readable summary of current state."""
        lines = ["Dimension State Summary:", "-" * 40]
        for i, name in enumerate(self.dimension_names):
            val = self.D[i].item()
            status = "[UP]" if val > 1.05 else ("[DN]" if val < 0.95 else "  ~ ")
            lines.append(f"  {name:20s}: {val:.4f}  {status}")
        lines.append(f"  Steps: {self._step_count}")
        lines.append(f"  Range: [{self.D.min().item():.4f}, {self.D.max().item():.4f}]")
        return "\n".join(lines)

    def __repr__(self) -> str:
        dims = ", ".join(
            f"{name}={self.D[i].item():.3f}"
            for i, name in enumerate(self.dimension_names)
        )
        return f"DimensionState(step={self._step_count}, {dims})"
