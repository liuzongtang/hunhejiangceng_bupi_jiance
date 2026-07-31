"""
Multi-Dimensional Reward Manager.

Computes per-dimension reward scores for each model response and produces
both a reward vector (for dimension state tracking) and a scalar total
(for standard PPO advantage computation).

Integration with verl:
    Set in Hydra config:
        reward_model.reward_manager = "multi_dimensional"
        custom_reward_function.path = "mixed_reward.reward_manager.multi_dimensional"
        custom_reward_function.name = "compute_multi_dimensional_score"

Or use programmatically as a standalone reward function.
"""

import logging
from typing import Any, Callable, Dict, List, Optional

import torch

from mixed_reward.scoring.accuracy import compute_accuracy
from mixed_reward.scoring.completeness import compute_completeness
from mixed_reward.scoring.format import compute_format
from mixed_reward.scoring.safety import compute_safety

logger = logging.getLogger(__name__)


# Registry of available scorer functions
SCORER_REGISTRY: Dict[str, Callable] = {
    "accuracy": compute_accuracy,
    "safety": compute_safety,
    "completeness": compute_completeness,
    "format": compute_format,
}


def register_scorer(name: str, scorer_fn: Callable):
    """Register a custom scorer function."""
    SCORER_REGISTRY[name] = scorer_fn


class MultiDimensionalRewardManager:
    """
    Multi-dimensional reward manager.

    Computes independent reward scores across N configured dimensions,
    producing both:
    - A reward vector [r_dim1, r_dim2, ..., r_dimN] for dimension state tracking
    - A scalar total R_total = Σ(w_i * r_i) for standard PPO

    Each dimension uses a pluggable scoring function from SCORER_REGISTRY.
    Users can register custom scorers or override per-dimension via config.

    Usage:
        manager = MultiDimensionalRewardManager(
            dimensions=["accuracy", "safety", "completeness", "format"],
            weights=[1.0, 1.0, 0.5, 0.5]
        )

        # Single response
        r_dict = manager.compute_reward_vector(
            response="The answer is 42.",
            ground_truth="42",
            question="What is the meaning of life?"
        )
        total = manager.get_total_reward(r_dict)

        # Batch
        batch_results = manager.compute_batch(
            responses=["answer1", "answer2"],
            ground_truths=["gt1", "gt2"]
        )
    """

    def __init__(
        self,
        dimensions: Optional[List[str]] = None,
        weights: Optional[List[float]] = None,
        scorer_overrides: Optional[Dict[str, Callable]] = None,
        scorer_configs: Optional[Dict[str, Dict[str, Any]]] = None,
        normalize_weights: bool = False,
    ):
        """
        Args:
            dimensions: Names of reward dimensions.
                        Default: ["accuracy", "safety", "completeness", "format"]
            weights: Scalar weights for each dimension (used in get_total_reward).
                     Default: all 1.0
            scorer_overrides: Dict mapping dimension name → custom scorer function.
            scorer_configs: Dict mapping dimension name → kwargs for scorer function.
            normalize_weights: If True, normalize weights to sum to 1.
        """
        self.dimensions = dimensions or ["accuracy", "safety", "completeness", "format"]
        self.num_dims = len(self.dimensions)

        # Initialize weights
        self.weights = weights or [1.0] * self.num_dims
        if len(self.weights) != self.num_dims:
            logger.warning(
                f"weights length ({len(self.weights)}) != dimensions length "
                f"({self.num_dims}). Padding/extending with 1.0."
            )
            if len(self.weights) < self.num_dims:
                self.weights.extend([1.0] * (self.num_dims - len(self.weights)))
            else:
                self.weights = self.weights[: self.num_dims]

        if normalize_weights:
            total_w = sum(self.weights)
            if total_w > 0:
                self.weights = [w / total_w for w in self.weights]

        # Set up scorers
        self.scorer_overrides = scorer_overrides or {}
        self.scorer_configs = scorer_configs or {}

        # Validate all dimensions have available scorers
        for dim in self.dimensions:
            if dim not in SCORER_REGISTRY and dim not in self.scorer_overrides:
                logger.warning(
                    f"No scorer registered for dimension '{dim}'. "
                    f"Will return 0.0 for this dimension. "
                    f"Available scorers: {list(SCORER_REGISTRY.keys())}"
                )

    def _get_scorer(self, dim: str) -> Callable:
        """Get the scorer function for a dimension."""
        if dim in self.scorer_overrides:
            return self.scorer_overrides[dim]
        if dim in SCORER_REGISTRY:
            return SCORER_REGISTRY[dim]
        # Fallback: return a no-op scorer
        return lambda response, ground_truth=None, question=None, **kwargs: 0.0

    def _get_scorer_kwargs(self, dim: str) -> Dict[str, Any]:
        """Get kwargs for a dimension's scorer."""
        return self.scorer_configs.get(dim, {})

    def compute_reward_vector(
        self,
        response: str,
        ground_truth: Optional[str] = None,
        question: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, float]:
        """
        Compute per-dimension reward scores for a single response.

        Args:
            response: The model's generated response text.
            ground_truth: Reference/expected answer (used by accuracy scorer).
            question: The original prompt/question (used by completeness scorer).
            **kwargs: Additional context passed to scorers.

        Returns:
            Dict mapping dimension name → score in [0.0, 1.0].
        """
        reward_dict: Dict[str, float] = {}

        for dim in self.dimensions:
            try:
                scorer = self._get_scorer(dim)
                scorer_kwargs = self._get_scorer_kwargs(dim)

                # Build scorer-specific arguments
                scorer_args: Dict[str, Any] = {"response": response}
                if dim in ("accuracy",):
                    scorer_args["ground_truth"] = ground_truth
                if dim in ("completeness",):
                    scorer_args["question"] = question
                scorer_args.update(scorer_kwargs)
                scorer_args.update(kwargs)

                score = scorer(**scorer_args)
                # Ensure score is in [0, 1]
                score = max(0.0, min(1.0, float(score)))
                reward_dict[dim] = score

            except Exception as e:
                logger.error(f"Error computing '{dim}' score: {e}")
                reward_dict[dim] = 0.0

        return reward_dict

    def compute_batch(
        self,
        responses: List[str],
        ground_truths: Optional[List[Optional[str]]] = None,
        questions: Optional[List[Optional[str]]] = None,
        **kwargs,
    ) -> List[Dict[str, float]]:
        """
        Compute reward vectors for a batch of responses.

        Args:
            responses: List of model response strings.
            ground_truths: Optional list of ground truth answers (one per response).
            questions: Optional list of original prompts (one per response).
            **kwargs: Additional context passed to scorers.

        Returns:
            List of per-response reward dicts.
        """
        n = len(responses)

        # Normalize optional lists
        if ground_truths is None:
            ground_truths = [None] * n
        if questions is None:
            questions = [None] * n

        results = []
        for i in range(n):
            r_dict = self.compute_reward_vector(
                response=responses[i],
                ground_truth=ground_truths[i] if i < len(ground_truths) else None,
                question=questions[i] if i < len(questions) else None,
                **kwargs,
            )
            results.append(r_dict)

        return results

    def get_total_reward(self, reward_dict: Dict[str, float]) -> float:
        """
        Compute scalar total reward as weighted sum.

        R_total = Σ(w_i * r_i)

        Args:
            reward_dict: Output from compute_reward_vector.

        Returns:
            Scalar total reward.
        """
        total = 0.0
        for i, dim in enumerate(self.dimensions):
            total += self.weights[i] * reward_dict.get(dim, 0.0)
        return total

    def get_reward_tensor(self, reward_dict: Dict[str, float]) -> torch.Tensor:
        """
        Get reward vector as a PyTorch tensor.

        Args:
            reward_dict: Output from compute_reward_vector.

        Returns:
            Tensor of shape (num_dimensions,).
        """
        values = [reward_dict.get(dim, 0.0) for dim in self.dimensions]
        return torch.tensor(values, dtype=torch.float32)

    def get_batch_tensors(
        self, batch_results: List[Dict[str, float]]
    ) -> Dict[str, torch.Tensor]:
        """
        Convert batch results to tensors.

        Returns:
            Dict with:
                - "reward_vectors": (batch_size, num_dimensions)
                - "total_rewards": (batch_size,)
        """
        vectors = []
        totals = []
        for r_dict in batch_results:
            vectors.append([r_dict.get(dim, 0.0) for dim in self.dimensions])
            totals.append(self.get_total_reward(r_dict))

        return {
            "reward_vectors": torch.tensor(vectors, dtype=torch.float32),
            "total_rewards": torch.tensor(totals, dtype=torch.float32),
        }

    def get_average_reward_vector(
        self, batch_results: List[Dict[str, float]]
    ) -> torch.Tensor:
        """
        Compute the average reward vector across a batch.

        Useful as input to DimensionState.update().
        """
        tensors = self.get_batch_tensors(batch_results)
        return tensors["reward_vectors"].mean(dim=0)

    def summary(self, reward_dict: Dict[str, float]) -> str:
        """Human-readable reward summary."""
        lines = ["Reward Summary:", "-" * 30]
        for dim in self.dimensions:
            val = reward_dict.get(dim, 0.0)
            bar = "#" * int(val * 20) + "-" * (20 - int(val * 20))
            lines.append(f"  {dim:15s}: {val:.3f}  {bar}")
        total = self.get_total_reward(reward_dict)
        lines.append(f"  {'TOTAL':15s}: {total:.3f}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"MultiDimensionalRewardManager("
            f"dimensions={self.dimensions}, "
            f"weights={self.weights})"
        )


# ============================================================================
# verl Integration: Custom reward function
# ============================================================================


def compute_multi_dimensional_score(
    data_source: str,
    solution_str: str,
    ground_truth: Optional[str] = None,
    extra_info: Optional[Dict] = None,
    **kwargs,
) -> float:
    """
    verl-compatible reward function.

    This function signature matches verl's custom_reward_function protocol.
    It instantiates a MultiDimensionalRewardManager and returns the scalar
    total reward, while storing per-dimension scores in extra_info for
    later retrieval by the trainer.

    Config in Hydra YAML:
        custom_reward_function:
            path: "mixed_reward.reward_manager.multi_dimensional"
            name: "compute_multi_dimensional_score"
            reward_kwargs:
                dimensions: ["accuracy", "safety", "completeness", "format"]
                weights: [1.0, 1.0, 0.5, 0.5]

    Args:
        data_source: Dataset identifier (unused by default, can route scoring).
        solution_str: The model's generated response.
        ground_truth: The expected answer.
        extra_info: Additional context dict (may contain 'question').
        **kwargs: Additional args including reward_kwargs from config.

    Returns:
        Scalar total reward.
    """
    # Extract reward kwargs from config
    reward_kwargs = kwargs.get("reward_kwargs", {})

    # Build manager (cached at module level for efficiency)
    manager = _get_or_create_manager(reward_kwargs)

    # Extract question from extra_info if present
    question = None
    if extra_info is not None:
        question = extra_info.get("question", None)

    # Compute reward vector
    reward_dict = manager.compute_reward_vector(
        response=solution_str,
        ground_truth=ground_truth,
        question=question,
    )

    # Store per-dimension scores in extra_info for the trainer
    if extra_info is not None:
        extra_info["_reward_vector"] = reward_dict

    return manager.get_total_reward(reward_dict)


# Module-level manager cache
_manager_cache: Optional[MultiDimensionalRewardManager] = None
_manager_config_hash: str = ""


def _get_or_create_manager(config: Dict) -> MultiDimensionalRewardManager:
    """Get or create a cached reward manager instance."""
    global _manager_cache, _manager_config_hash
    config_hash = str(sorted(config.items()))
    if _manager_cache is None or config_hash != _manager_config_hash:
        _manager_cache = MultiDimensionalRewardManager(**config)
        _manager_config_hash = config_hash
    return _manager_cache
