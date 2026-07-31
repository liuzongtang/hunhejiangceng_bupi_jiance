#!/usr/bin/env python3
"""
Mixed Reward RLHF Training Launcher.

Entry point for training LLMs with the hybrid scalar+dimension reward mechanism.
Integrates with the verl framework for PPO/GRPO training.

Usage:
    # Standalone demo mode (no verl required):
    python scripts/train_mixed_reward.py --mode demo --steps 100

    # With verl (requires verl + GPU + Qwen2.5-1.5B):
    python scripts/train_mixed_reward.py \\
        --mode verl \\
        --config configs/mixed_reward_grpo.yaml \\
        model.path=Qwen/Qwen2.5-1.5B-Instruct \\
        mixed_reward.beta=0.1
"""

import argparse
import logging
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("train_mixed_reward")


def run_demo_mode(args):
    """Run a demo simulation without verl/GPU."""
    import numpy as np

    from mixed_reward import MixedRewardConfig
    from mixed_reward.dimension.state import DimensionState
    from mixed_reward.reward_manager.multi_dimensional import (
        MultiDimensionalRewardManager,
    )
    from mixed_reward.trainer.mixed_reward_trainer import MixedRewardPPOTrainer
    from mixed_reward.visualizer import MixedRewardVisualizer

    logger.info("Running in DEMO mode (no verl/GPU required)")

    # Build config
    config = MixedRewardConfig(
        dimensions=[
            {"name": "accuracy", "weight": 1.0},
            {"name": "safety", "weight": 1.0},
            {"name": "completeness", "weight": 0.5},
            {"name": "format", "weight": 0.5},
        ],
        modulation={"beta": args.beta},
    )

    # Initialize components
    reward_manager = MultiDimensionalRewardManager(
        dimensions=config.dimension_names,
        weights=config.scalar.weights,
    )

    dim_state = DimensionState(
        num_dimensions=config.num_active_dimensions,
        init_value=1.0,
        d_min=0.5,
        d_max=2.0,
        forget_factor=0.999,
        learning_rate=0.01,
        dimension_names=config.dimension_names,
    )

    trainer = MixedRewardPPOTrainer(
        config=config,
        reward_manager=reward_manager,
        dimension_state=dim_state,
    )

    viz_log_dir = args.log_dir or "./runs/mixed_reward_train"
    viz = MixedRewardVisualizer(
        use_wandb=args.wandb,
        use_tensorboard=args.tensorboard,
        log_dir=viz_log_dir,
        project_name="mixed-reward-rlhf",
    )
    viz.set_dimensions(config.dimension_names)

    logger.info(f"Running {args.steps} demo steps...")

    rng = np.random.RandomState(args.seed)
    for step in range(args.steps):
        # Simulate a batch of 8 responses
        batch_rewards = trainer.compute_rewards(
            responses=[f"Simulated response {step}_{i}" for i in range(8)],
        )

        # Override with synthetic scores for demonstration
        t = step / max(1, args.steps)
        for rd in batch_rewards:
            rd["accuracy"] = float(np.clip(0.3 + 0.6 * t + rng.normal(0, 0.05), 0, 1))
            rd["safety"] = float(np.clip(0.85 + rng.normal(0, 0.03), 0, 1))
            rd["completeness"] = float(
                np.clip(
                    0.6 + 0.2 * np.sin(2 * np.pi * t * 5) + rng.normal(0, 0.04), 0, 1
                )
            )
            rd["format"] = float(
                np.clip(0.2 + 0.7 * (1 - np.exp(-t / 0.1)) + rng.normal(0, 0.03), 0, 1)
            )

        # Update dimension state
        D = trainer.update_dimension_state(batch_rewards)

        # Log
        if step % args.log_interval == 0:
            avg_reward = reward_manager.get_average_reward_vector(batch_rewards)
            avg_dict = {
                dim: float(avg_reward[i].item())
                for i, dim in enumerate(config.dimension_names)
            }
            avg_dict["total"] = float(sum(avg_reward).item())

            viz.log_reward_vector(avg_dict, step=step)
            viz.log_dimension_state(D, step=step)

            # Simulate gradient scales
            scales = [1.0 + args.beta * (D[i % len(D)].item() - 1.0) for i in range(8)]
            viz.log_gradient_scales(scales, step=step)

            logger.info(
                f"Step {step:5d} | Total: {avg_dict['total']:.4f} | "
                f"D: [{D[0].item():.3f}, {D[1].item():.3f}, {D[2].item():.3f}, {D[3].item():.3f}]"
            )

        if step % 100 == 0 and step > 0:
            history = dim_state.get_history()
            viz.log_dimension_state_heatmap(D, step=step, history=history)

    # Final summary
    logger.info("Training complete!")
    logger.info(f"\n{dim_state.summary()}")

    viz.close()
    logger.info(f"TensorBoard logs saved to: {viz_log_dir}")


def run_verl_mode(args):
    """Run with the verl framework (requires verl + GPU)."""
    try:
        import mixed_reward  # noqa: F401 -- triggers registrations
        from mixed_reward.trainer.mixed_reward_trainer import (
            MixedRewardVerlTrainer,  # noqa: F401
        )
    except ImportError as e:
        logger.error(f"Cannot import verl components: {e}")
        logger.error("Install verl: pip install verl")
        logger.error("Or run in demo mode: --mode demo")
        sys.exit(1)

    logger.info("Running in VERL mode")
    logger.info(f"Config: {args.config}")

    # verl uses Hydra for config management
    try:
        import hydra
        from omegaconf import OmegaConf

        @hydra.main(
            config_path=os.path.join("..", "configs"),
            config_name=args.config or "mixed_reward_grpo",
        )
        def main_hydra(cfg):
            logger.info("Hydra config loaded")
            logger.info(OmegaConf.to_yaml(cfg))

            # TODO: Initialize MixedRewardVerlTrainer with Hydra config
            # trainer = MixedRewardVerlTrainer(
            #     config=cfg,
            #     tokenizer=...,
            #     ...
            # )
            # trainer.fit()

            logger.info("verl mode -- full integration requires GPU + model setup")

        main_hydra()

    except ImportError:
        logger.warning("Hydra not installed. Running with manual config.")
        logger.info("To use verl mode, install: pip install hydra-core omegaconf")

        # Fallback: manual config loading
        if args.config and os.path.exists(args.config):
            import yaml

            with open(args.config, "r") as f:
                config = yaml.safe_load(f)
            logger.info(
                f"Loaded config from {args.config}: {list(config.keys()) if config else 'empty'}"
            )
        else:
            logger.info("No config file found. Using defaults.")


def main():
    parser = argparse.ArgumentParser(description="Mixed Reward RLHF Training Launcher")

    # Mode
    parser.add_argument(
        "--mode",
        choices=["demo", "verl"],
        default="demo",
        help="Run mode: demo (CPU simulation) or verl (full training)",
    )

    # Demo mode args
    parser.add_argument(
        "--steps", type=int, default=200, help="Number of training steps (demo mode)"
    )
    parser.add_argument(
        "--beta", type=float, default=0.1, help="Modulation strength beta"
    )

    # verl mode args
    parser.add_argument(
        "--config", type=str, default=None, help="Path to Hydra YAML config (verl mode)"
    )

    # Logging
    parser.add_argument("--wandb", action="store_true", help="Enable WandB logging")
    parser.add_argument(
        "--tensorboard", action="store_true", help="Enable TensorBoard logging"
    )
    parser.add_argument(
        "--log-dir", type=str, default=None, help="Log directory for TensorBoard"
    )
    parser.add_argument(
        "--log-interval", type=int, default=10, help="Log every N steps"
    )

    # Misc
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    if args.mode == "demo":
        run_demo_mode(args)
    elif args.mode == "verl":
        run_verl_mode(args)


if __name__ == "__main__":
    main()
