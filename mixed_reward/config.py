"""
Configuration system for the Mixed Reward mechanism.

Uses Python dataclasses for programmatic configuration, with optional
Hydra/OmegaConf integration for verl compatibility.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class DimensionConfig:
    """Configuration for a single reward dimension."""

    name: str
    weight: float = 1.0
    enabled: bool = True
    kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DimensionStateConfig:
    """Configuration for the DimensionState (D vector) manager."""

    # Number of dimensions (must match reward dimensions)
    num_dimensions: int = 4

    # Initial value for all D entries
    init_value: float = 1.0

    # Clamping range to prevent gradient vanish/explode
    d_min: float = 0.5
    d_max: float = 2.0

    # Forget factor: D decays toward 1.0 each step
    forget_factor: float = 0.999

    # Learning rate for D updates
    learning_rate: float = 0.01

    # Whether to use baseline (running average) for relative performance
    use_baseline: bool = True

    # Baseline momentum (if use_baseline=True)
    baseline_momentum: float = 0.99


@dataclass
class ScalarConfig:
    """Configuration for the scalar reward branch."""

    # Dimension weights for weighted sum R_total = Σ(w_i * r_i)
    weights: List[float] = field(default_factory=lambda: [1.0, 1.0, 1.0, 1.0])

    # Whether to normalize weights to sum to 1
    normalize_weights: bool = False


@dataclass
class ModulationConfig:
    """Configuration for gradient modulation."""

    # β: modulation strength — higher = more dimension influence
    beta: float = 0.1

    # Whether to modulate advantages or gradients directly
    modulate_advantages: bool = True

    # Per-parameter-group modulation (True) vs global modulation (False)
    per_param_group: bool = True


@dataclass
class VisualizerConfig:
    """Configuration for the visualization system."""

    # Enable WandB logging
    use_wandb: bool = True

    # Enable TensorBoard logging
    use_tensorboard: bool = True

    # Log directory for TensorBoard
    log_dir: str = "./runs/mixed_reward"

    # WandB project name
    project_name: str = "mixed-reward-rlhf"

    # WandB entity (team/user)
    wandb_entity: Optional[str] = None

    # Logging interval (steps)
    log_interval: int = 10

    # Heatmap generation interval (steps)
    heatmap_interval: int = 100

    # WandB run name
    run_name: Optional[str] = None


@dataclass
class MixedRewardConfig:
    """Master configuration for the Mixed Reward system."""

    # Reward dimensions
    dimensions: List[DimensionConfig] = field(
        default_factory=lambda: [
            DimensionConfig(name="accuracy", weight=1.0),
            DimensionConfig(name="safety", weight=1.0),
            DimensionConfig(name="completeness", weight=0.5),
            DimensionConfig(name="format", weight=0.5),
        ]
    )

    # Dimension state config
    dim_state: DimensionStateConfig = field(default_factory=DimensionStateConfig)

    # Scalar reward config
    scalar: ScalarConfig = field(default_factory=ScalarConfig)

    # Modulation config
    modulation: ModulationConfig = field(default_factory=ModulationConfig)

    # Visualizer config
    visualizer: VisualizerConfig = field(default_factory=VisualizerConfig)

    # Experiment metadata
    experiment_name: str = "mixed_reward_experiment"
    seed: int = 42

    def __post_init__(self):
        """Validate configuration consistency and convert dict inputs."""
        # Convert dict dimensions to DimensionConfig objects
        self.dimensions = [
            DimensionConfig(**d) if isinstance(d, dict) else d for d in self.dimensions
        ]

        # Convert nested config dicts to dataclass objects
        if isinstance(self.dim_state, dict):
            self.dim_state = DimensionStateConfig(**self.dim_state)
        if isinstance(self.scalar, dict):
            self.scalar = ScalarConfig(**self.scalar)
        if isinstance(self.modulation, dict):
            self.modulation = ModulationConfig(**self.modulation)
        if isinstance(self.visualizer, dict):
            self.visualizer = VisualizerConfig(**self.visualizer)

        # Sync num_dimensions with actual dimensions list
        enabled_dims = [d for d in self.dimensions if d.enabled]
        self.dim_state.num_dimensions = len(enabled_dims)

        # Sync scalar weights count
        if len(self.scalar.weights) != len(self.dimensions) and len(
            self.scalar.weights
        ) < len(self.dimensions):
            # Extend with 1.0
            self.scalar.weights.extend(
                [1.0] * (len(self.dimensions) - len(self.scalar.weights))
            )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for Hydra/OmegaConf compatibility."""
        from dataclasses import asdict

        return asdict(self)

    @property
    def dimension_names(self) -> List[str]:
        """Get list of enabled dimension names."""
        return [d.name for d in self.dimensions if d.enabled]

    @property
    def num_active_dimensions(self) -> int:
        """Count of enabled dimensions."""
        return len(self.dimension_names)
