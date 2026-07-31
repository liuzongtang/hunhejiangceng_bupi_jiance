"""
Mixed Reward RLHF — "标量+维度" Hybrid Reward/Punishment Mechanism.

Combines scalar rewards (global step-size control) with dimension state vectors
(local gradient modulation) for multi-objective RLHF alignment of LLMs.

Core formula:
    ΔW = η · R_total · (1 + β·(D-1)) · ∇L

where:
    - R_total = Σ(w_i * r_i) : weighted scalar reward
    - D : dimension state vector ∈ [d_min, d_max]
    - β : modulation strength hyperparameter
"""

__version__ = "1.0.0"
__author__ = "Mixed Reward Team"

from mixed_reward.config import MixedRewardConfig
from mixed_reward.dimension.state import DimensionState
from mixed_reward.reward_manager.multi_dimensional import MultiDimensionalRewardManager
from mixed_reward.visualizer import MixedRewardVisualizer

__all__ = [
    "DimensionState",
    "MixedRewardConfig",
    "MixedRewardVisualizer",
    "MultiDimensionalRewardManager",
]
