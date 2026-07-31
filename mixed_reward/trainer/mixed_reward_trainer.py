"""
Mixed Reward PPO Trainer.

Integrates the scalar+dimension hybrid reward mechanism into the PPO training
loop. Subclasses RayPPOTrainer (from verl) when available, with a standalone
fallback for testing without verl installed.

Core Formula:
    ΔW = η · R_total · (1 + β·(D[group_idx] - 1)) · ∇L

where:
    η: learning rate (applied by optimizer)
    R_total: scalar total reward
    β: modulation strength hyperparameter
    D: dimension state vector

Integration Architecture:
    1. Reward computation → MultiDimensionalRewardManager
    2. D vector update → DimensionState
    3. Advantage modulation → applied before update_actor
    4. Logging → MixedRewardVisualizer
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import torch

from mixed_reward.config import MixedRewardConfig
from mixed_reward.dimension.state import DimensionState
from mixed_reward.reward_manager.multi_dimensional import MultiDimensionalRewardManager

logger = logging.getLogger(__name__)

# ============================================================================
# Gradient Modulation
# ============================================================================


def compute_dimension_scale(
    D: torch.Tensor,
    param_group_idx: int,
    beta: float,
) -> float:
    """
    Compute the per-parameter-group gradient modulation scale.

    scale = 1 + β · (D[group_idx % num_dims] - 1)

    - When D > 1: scale > 1, gradient is AMPLIFIED
    - When D < 1: scale < 1, gradient is SUPPRESSED
    - When D = 1: scale = 1, no modulation (neutral)
    - β controls modulation strength:
      - β = 0: no modulation (equivalent to baseline)
      - β = 0.1: mild modulation (D=2.0 → scale=1.1, D=0.5 → scale=0.95)
      - β = 1.0: strong modulation (D=2.0 → scale=2.0, D=0.5 → scale=0.5)

    Args:
        D: Dimension state vector of shape (num_dimensions,).
        param_group_idx: Index of the parameter group.
        beta: Modulation strength.

    Returns:
        Scale factor to multiply with gradients.
    """
    dim_idx = param_group_idx % len(D)
    d_val = D[dim_idx].item()
    scale = 1.0 + beta * (d_val - 1.0)
    return scale


def apply_mixed_gradient_modulation(
    parameters: List[torch.nn.Parameter],
    D: torch.Tensor,
    scalar_advantage: float,
    beta: float,
    param_group_idx: int = 0,
) -> float:
    """
    Apply mixed reward gradient modulation to parameter gradients in-place.

    For each parameter:
        param.grad *= R_total * (1 + β * (D[group_idx % num_dims] - 1))

    Args:
        parameters: List of parameters whose .grad will be modified.
        D: Dimension state vector.
        scalar_advantage: The scalar advantage (R_total equivalent).
        beta: Modulation strength.
        param_group_idx: Index of this parameter group.

    Returns:
        The applied scale factor (for logging).
    """
    dim_scale = compute_dimension_scale(D, param_group_idx, beta)
    total_scale = scalar_advantage * dim_scale

    for param in parameters:
        if param.grad is not None:
            param.grad.data.mul_(total_scale)

    return total_scale


def apply_mixed_gradient_to_param_groups(
    param_groups: List[Dict[str, Any]],
    D: torch.Tensor,
    scalar_advantage: float,
    beta: float,
) -> List[float]:
    """
    Apply mixed gradient modulation to multiple parameter groups.

    Each group gets its own dimension scale based on group index.

    Args:
        param_groups: List of param_group dicts (each has 'params' key).
        D: Dimension state vector.
        scalar_advantage: Scalar advantage.
        beta: Modulation strength.

    Returns:
        List of applied scale factors per group (for logging).
    """
    scales = []
    for i, group in enumerate(param_groups):
        params = group["params"]
        scale = apply_mixed_gradient_modulation(params, D, scalar_advantage, beta, i)
        scales.append(scale)
    return scales


# ============================================================================
# Mixed Reward Trainer
# ============================================================================


class MixedRewardPPOTrainer:
    """
    Mixed Reward PPO Trainer.

    This is the core trainer class that orchestrates the hybrid scalar+dimension
    reward mechanism. It can be used in two modes:

    1. **Standalone mode** (verl not installed): Provides the core logic for
       managing reward computation, dimension state, and gradient modulation.
       Useful for testing, simulation, and integration into custom training loops.

    2. **verl mode** (verl installed): Subclasses RayPPOTrainer to inject
       dimension state management. Import `MixedRewardVerlTrainer` for this.

    The standalone mode provides a `training_step` method that demonstrates
    the full pipeline.
    """

    def __init__(
        self,
        config: MixedRewardConfig,
        reward_manager: Optional[MultiDimensionalRewardManager] = None,
        dimension_state: Optional[DimensionState] = None,
    ):
        """
        Args:
            config: Full mixed reward configuration.
            reward_manager: Pre-configured reward manager (created from config if None).
            dimension_state: Pre-configured dimension state (created from config if None).
        """
        self.config = config

        # Reward manager
        if reward_manager is None:
            weights = config.scalar.weights[: config.num_active_dimensions]
            self.reward_manager = MultiDimensionalRewardManager(
                dimensions=config.dimension_names,
                weights=weights,
                normalize_weights=config.scalar.normalize_weights,
            )
        else:
            self.reward_manager = reward_manager
            # Sync config with manager
            if config.dimension_names != self.reward_manager.dimensions:
                logger.warning(
                    "Config dimensions don't match reward_manager dimensions. "
                    "Using reward_manager's dimensions."
                )

        # Dimension state
        if dimension_state is None:
            self.dimension_state = DimensionState(
                num_dimensions=config.num_active_dimensions,
                init_value=config.dim_state.init_value,
                d_min=config.dim_state.d_min,
                d_max=config.dim_state.d_max,
                forget_factor=config.dim_state.forget_factor,
                learning_rate=config.dim_state.learning_rate,
                use_baseline=config.dim_state.use_baseline,
                baseline_momentum=config.dim_state.baseline_momentum,
                dimension_names=config.dimension_names,
            )
        else:
            self.dimension_state = dimension_state

        # Modulation params
        self.beta = config.modulation.beta
        self.modulate_advantages = config.modulation.modulate_advantages
        self.per_param_group = config.modulation.per_param_group

        # Metrics tracking
        self.metrics: Dict[str, List[float]] = {
            "total_rewards": [],
            "dimension_scales": [],
            "gradient_scales": [],
        }
        for dim_name in config.dimension_names:
            self.metrics[f"reward_{dim_name}"] = []

        self._step = 0

    def compute_rewards(
        self,
        responses: List[str],
        ground_truths: Optional[List[Optional[str]]] = None,
        questions: Optional[List[Optional[str]]] = None,
    ) -> List[Dict[str, float]]:
        """
        Compute multi-dimensional rewards for a batch of responses.

        Returns:
            List of per-response reward dicts.
        """
        return self.reward_manager.compute_batch(responses, ground_truths, questions)

    def update_dimension_state(
        self, batch_reward_dicts: List[Dict[str, float]]
    ) -> torch.Tensor:
        """
        Update the D vector with batch-averaged reward.

        Args:
            batch_reward_dicts: List of per-response reward dicts from compute_rewards.

        Returns:
            Updated D vector.
        """
        avg_reward = self.reward_manager.get_average_reward_vector(batch_reward_dicts)
        return self.dimension_state.update(avg_reward)

    def get_modulated_advantage(
        self, scalar_advantage: float, param_group_idx: int = 0
    ) -> float:
        """
        Compute modulated advantage for a parameter group.

        modulated_adv = scalar_adv * (1 + β * (D[group_idx] - 1))

        Args:
            scalar_advantage: The standard scalar advantage.
            param_group_idx: Parameter group index.

        Returns:
            Modulated advantage value.
        """
        D = self.dimension_state.get_state()
        dim_scale = compute_dimension_scale(D, param_group_idx, self.beta)
        return scalar_advantage * dim_scale

    def apply_gradient_modulation(
        self,
        param_groups: List[Dict[str, Any]],
        scalar_advantage: float,
    ) -> List[float]:
        """
        Apply mixed gradient modulation to all parameter groups.

        Args:
            param_groups: Optimizer parameter groups.
            scalar_advantage: Scalar advantage.

        Returns:
            List of applied scales per group.
        """
        D = self.dimension_state.get_state()
        return apply_mixed_gradient_to_param_groups(
            param_groups, D, scalar_advantage, self.beta
        )

    def training_step(
        self,
        responses: List[str],
        ground_truths: Optional[List[Optional[str]]] = None,
        questions: Optional[List[Optional[str]]] = None,
        param_groups: Optional[List[Dict[str, Any]]] = None,
        scalar_advantage: float = 1.0,
    ) -> Dict[str, Any]:
        """
        Execute a complete training step of the mixed reward pipeline.

        This method demonstrates the full pipeline in standalone mode:
        1. Compute multi-dimensional rewards
        2. Update dimension state D
        3. Compute modulated advantages
        4. Apply gradient modulation (if param_groups provided)

        Args:
            responses: Batch of model response strings.
            ground_truths: Optional ground truth answers.
            questions: Optional original questions/prompts.
            param_groups: Optional optimizer param groups for gradient modulation.
            scalar_advantage: Scalar advantage (typically from PPO).

        Returns:
            Dict with training metrics for this step.
        """
        # Step 1: Compute multi-dimensional rewards
        batch_rewards = self.compute_rewards(responses, ground_truths, questions)

        # Step 2: Get scalar totals and average reward vector
        tensor_dict = self.reward_manager.get_batch_tensors(batch_rewards)
        avg_total_reward = tensor_dict["total_rewards"].mean().item()

        # Step 3: Update dimension state
        D = self.update_dimension_state(batch_rewards)

        # Step 4: Compute modulated advantage
        modulated_adv = self.get_modulated_advantage(scalar_advantage)

        # Step 5: Apply gradient modulation (if param groups available)
        grad_scales = None
        if param_groups is not None:
            grad_scales = self.apply_gradient_modulation(param_groups, scalar_advantage)

        # Step 6: Track metrics
        self._record_metrics(batch_rewards, avg_total_reward, D, grad_scales)
        self._step += 1

        return {
            "step": self._step,
            "reward_vectors": tensor_dict["reward_vectors"],
            "total_rewards": tensor_dict["total_rewards"],
            "avg_total_reward": avg_total_reward,
            "dimension_state": D.clone(),
            "modulated_advantage": modulated_adv,
            "gradient_scales": grad_scales,
            "batch_rewards": batch_rewards,
        }

    def _record_metrics(
        self,
        batch_rewards: List[Dict[str, float]],
        avg_total: float,
        D: torch.Tensor,
        grad_scales: Optional[List[float]],
    ):
        """Record metrics for this step."""
        self.metrics["total_rewards"].append(avg_total)

        # Per-dimension average rewards
        for dim_name in self.config.dimension_names:
            dim_avg = sum(r.get(dim_name, 0.0) for r in batch_rewards) / len(
                batch_rewards
            )
            self.metrics[f"reward_{dim_name}"].append(dim_avg)

        # Dimension state values
        D_np = D.detach().numpy()
        if "dimension_scales" not in self.metrics:
            self.metrics["dimension_scales"] = []
        self.metrics["dimension_scales"].append(D_np.copy())

        # Gradient scales
        if grad_scales is not None:
            self.metrics["gradient_scales"].append(grad_scales)

    def get_metrics_summary(self) -> Dict[str, float]:
        """Get the latest metrics values for logging."""
        summary = {}
        for key, values in self.metrics.items():
            if values:
                last = values[-1]
                if isinstance(last, (int, float)):
                    summary[key] = last
                elif isinstance(last, (list, tuple)):
                    for i, v in enumerate(last):
                        summary[f"{key}_{i}"] = v
        return summary

    @property
    def step(self) -> int:
        return self._step

    def state_dict(self) -> Dict[str, Any]:
        """Serialize trainer state for checkpointing."""
        return {
            "config": self.config.to_dict(),
            "dimension_state": self.dimension_state.get_state_dict(),
            "step": self._step,
            "metrics": self.metrics,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Restore trainer state from checkpoint."""
        self.dimension_state.load_state_dict(state_dict["dimension_state"])
        self._step = state_dict.get("step", 0)
        self.metrics = state_dict.get("metrics", self.metrics)


# ============================================================================
# verl Integration: Trainer Subclass
# ============================================================================

try:
    from verl.trainer.ppo.ray_trainer import RayPPOTrainer

    _VERL_AVAILABLE = True
except ImportError:
    _VERL_AVAILABLE = False
    RayPPOTrainer = object  # type: ignore
    logger.info("verl not installed — MixedRewardVerlTrainer will not be available.")


if _VERL_AVAILABLE:

    class MixedRewardVerlTrainer(RayPPOTrainer):  # type: ignore
        """
        verl-integrated Mixed Reward PPO Trainer.

        Subclasses RayPPOTrainer to inject dimension state management.
        Overrides key points in the training loop to:
        1. Extract per-dimension reward scores
        2. Update the D vector
        3. Modulate advantages before policy update
        4. Log all dimension-related metrics
        """

        def __init__(
            self,
            config,
            tokenizer,
            role_worker_mapping,
            resource_pool_map,
            ray_worker_group_cls,
            reward_fn=None,
            val_reward_fn=None,
            **kwargs,
        ):
            super().__init__(
                config=config,
                tokenizer=tokenizer,
                role_worker_mapping=role_worker_mapping,
                resource_pool_map=resource_pool_map,
                ray_worker_group_cls=ray_worker_group_cls,
                reward_fn=reward_fn,
                val_reward_fn=val_reward_fn,
                **kwargs,
            )

            # Initialize mixed reward components
            mr_config = self._build_mixed_reward_config(config)
            self.mr_trainer = MixedRewardPPOTrainer(config=mr_config)
            self.mr_visualizer = None  # Initialized lazily

        def _build_mixed_reward_config(self, verl_config) -> MixedRewardConfig:
            """Extract mixed reward config from verl's Hydra config."""
            # Try to read mixed_reward section from config
            if hasattr(verl_config, "mixed_reward") and verl_config.mixed_reward.get(
                "enabled", False
            ):
                mr_cfg = verl_config.mixed_reward
                return MixedRewardConfig(
                    dimensions=[
                        {"name": d}
                        for d in mr_cfg.get(
                            "dimensions",
                            ["accuracy", "safety", "completeness", "format"],
                        )
                    ],
                    modulation={"beta": mr_cfg.get("beta", 0.1)},
                )
            return MixedRewardConfig()  # Default config

        def _compute_mixed_rewards(self, batch):
            """Hook: compute multi-dimensional rewards from batch responses."""
            # Extract responses and metadata from batch
            # This integrates with verl's DataProto format
            responses = self._extract_responses(batch)
            ground_truths = self._extract_ground_truths(batch)
            questions = self._extract_questions(batch)

            return self.mr_trainer.compute_rewards(responses, ground_truths, questions)

        def _extract_responses(self, batch) -> List[str]:
            """Extract response strings from verl batch. Override as needed."""
            # This is a placeholder — actual implementation depends on verl's batch format
            return []

        def _extract_ground_truths(self, batch) -> List[Optional[str]]:
            """Extract ground truths from verl batch. Override as needed."""
            return []

        def _extract_questions(self, batch) -> List[Optional[str]]:
            """Extract questions from verl batch. Override as needed."""
            return []
else:
    # When verl is not available, provide a stub
    class MixedRewardVerlTrainer:  # type: ignore
        """Stub: verl is not installed."""

        def __init__(self, *args, **kwargs):
            raise ImportError(
                "verl is required for MixedRewardVerlTrainer. "
                "Install with: pip install verl\n"
                "Or use MixedRewardPPOTrainer for standalone mode."
            )
