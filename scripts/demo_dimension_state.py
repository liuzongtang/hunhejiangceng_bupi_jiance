#!/usr/bin/env python3
"""
Standalone demo of the mixed reward mechanism.

Simulates a training run with synthetic reward signals to demonstrate:
1. Multi-dimensional reward computation
2. Dimension state (D vector) evolution
3. Gradient modulation effects
4. Visualization output (console + optional WandB/TB)

No GPU or verl required -- runs entirely on CPU with simulated data.

Usage:
    python scripts/demo_dimension_state.py [--steps 200] [--wandb] [--tensorboard]
"""

import argparse
import os
import sys
import time

import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mixed_reward import (
    DimensionState,
    MixedRewardConfig,
    MixedRewardVisualizer,
    MultiDimensionalRewardManager,
)
from mixed_reward.trainer.mixed_reward_trainer import MixedRewardPPOTrainer


def simulate_reward_signals(
    num_steps: int,
    num_dims: int = 4,
    seed: int = 42,
) -> list:
    """
    Generate synthetic reward signals for demonstration.

    Creates interesting dynamics:
    - Accuracy: starts low, improves over time
    - Safety: consistently high with small fluctuations
    - Completeness: oscillating pattern
    - Format: sharp improvement then plateau

    Returns:
        List of (responses, ground_truths, questions) tuples.
    """
    rng = np.random.RandomState(seed)

    # Simulated reward curves
    t = np.arange(num_steps)

    # Accuracy: sigmoid improvement
    accuracy = 0.3 + 0.6 / (1 + np.exp(-(t - num_steps * 0.4) / (num_steps * 0.1)))
    accuracy += rng.normal(0, 0.05, num_steps)

    # Safety: consistently high
    safety = 0.85 + rng.normal(0, 0.03, num_steps)

    # Completeness: sine wave oscillation
    completeness = 0.6 + 0.2 * np.sin(2 * np.pi * t / (num_steps * 0.2))
    completeness += rng.normal(0, 0.04, num_steps)

    # Format: rapid improvement then plateau
    format_score = 0.2 + 0.7 * (1 - np.exp(-t / (num_steps * 0.05)))
    format_score += rng.normal(0, 0.03, num_steps)

    # Clamp to [0, 1]
    accuracy = np.clip(accuracy, 0, 1)
    safety = np.clip(safety, 0, 1)
    completeness = np.clip(completeness, 0, 1)
    format_score = np.clip(format_score, 0, 1)

    return accuracy, safety, completeness, format_score


def main():
    parser = argparse.ArgumentParser(
        description="Demo the mixed reward mechanism with simulated data"
    )
    parser.add_argument(
        "--steps", type=int, default=200, help="Number of simulation steps"
    )
    parser.add_argument("--beta", type=float, default=0.1, help="Modulation strength")
    parser.add_argument("--wandb", action="store_true", help="Enable WandB logging")
    parser.add_argument(
        "--tensorboard", action="store_true", help="Enable TensorBoard logging"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    print("=" * 60)
    print("  Mixed Reward Mechanism -- Simulation Demo")
    print("=" * 60)
    print(f"  Steps: {args.steps}")
    print(f"  Beta:  {args.beta}")
    print(f"  WandB: {args.wandb}")
    print(f"  TB:    {args.tensorboard}")
    print("=" * 60)
    print()

    # Configuration
    config = MixedRewardConfig(
        dimensions=[
            {"name": "accuracy", "weight": 1.0},
            {"name": "safety", "weight": 1.0},
            {"name": "completeness", "weight": 0.5},
            {"name": "format", "weight": 0.5},
        ],
        modulation={"beta": args.beta},
        visualizer={
            "use_wandb": args.wandb,
            "use_tensorboard": args.tensorboard,
            "log_dir": "./runs/demo_mixed_reward",
            "project_name": "mixed-reward-demo",
        },
    )

    # Initialize components
    print("[1/4] Initializing reward manager...")
    reward_manager = MultiDimensionalRewardManager(
        dimensions=config.dimension_names,
        weights=config.scalar.weights,
    )
    print(f"      Dimensions: {reward_manager.dimensions}")
    print(f"      Weights:    {reward_manager.weights}")

    print("[2/4] Initializing dimension state...")
    dim_state = DimensionState(
        num_dimensions=config.num_active_dimensions,
        init_value=config.dim_state.init_value,
        d_min=config.dim_state.d_min,
        d_max=config.dim_state.d_max,
        forget_factor=config.dim_state.forget_factor,
        learning_rate=config.dim_state.learning_rate,
        dimension_names=config.dimension_names,
    )
    print(f"      Init D:     {dim_state.get_state_numpy()}")
    print(f"      Clamp:      [{config.dim_state.d_min}, {config.dim_state.d_max}]")
    print(f"      Forget:     {config.dim_state.forget_factor}")

    print("[3/4] Initializing trainer...")
    MixedRewardPPOTrainer(
        config=config,
        reward_manager=reward_manager,
        dimension_state=dim_state,
    )

    print("[4/4] Initializing visualizer...")
    viz = MixedRewardVisualizer(
        use_wandb=args.wandb,
        use_tensorboard=args.tensorboard,
        log_dir="./runs/demo_mixed_reward",
        project_name="mixed-reward-demo",
        config=config.to_dict(),
    )
    viz.set_dimensions(config.dimension_names)

    # Generate synthetic reward signals
    print()
    print("Generating synthetic reward signals...")
    acc, saf, comp, fmt = simulate_reward_signals(args.steps, seed=args.seed)

    # Simulation loop
    print()
    print("Running simulation...")
    print("-" * 60)
    print(
        f"{'Step':<8} {'Total':<10} {'Accuracy':<10} {'Safety':<10} {'Comp':<10} {'Format':<10} {'D[0]':<8} {'D[1]':<8}"
    )
    print("-" * 60)

    start_time = time.time()

    for step in range(args.steps):
        # Simulate batch: use 4 samples per step
        batch_rewards = []
        for _ in range(4):
            batch_rewards.append(
                {
                    "accuracy": float(acc[step] + np.random.normal(0, 0.02)),
                    "safety": float(saf[step] + np.random.normal(0, 0.01)),
                    "completeness": float(comp[step] + np.random.normal(0, 0.02)),
                    "format": float(fmt[step] + np.random.normal(0, 0.01)),
                }
            )

        # Update dimension state with average reward
        avg_reward = reward_manager.get_average_reward_vector(batch_rewards)
        D = dim_state.update(avg_reward)

        # Compute metrics
        total = (
            sum(
                r.mean()
                for r in [
                    np.array([b["accuracy"] for b in batch_rewards]),
                    np.array([b["safety"] for b in batch_rewards]),
                    np.array([b["completeness"] for b in batch_rewards]),
                    np.array([b["format"] for b in batch_rewards]),
                ]
            )
            / 4.0
        )  # Simplified -- in real code, weighted sum

        # Logging
        if step % 10 == 0:
            avg_acc = np.mean([b["accuracy"] for b in batch_rewards])
            avg_saf = np.mean([b["safety"] for b in batch_rewards])
            avg_comp = np.mean([b["completeness"] for b in batch_rewards])
            avg_fmt = np.mean([b["format"] for b in batch_rewards])
            D_np = D.numpy()

            print(
                f"{step:<8} {total:<10.4f} {avg_acc:<10.4f} {avg_saf:<10.4f} "
                f"{avg_comp:<10.4f} {avg_fmt:<10.4f} {D_np[0]:<8.4f} {D_np[1]:<8.4f}"
            )

        # Visualizer logging
        if step % config.visualizer.log_interval == 0:
            avg_reward_dict = {
                "accuracy": float(np.mean([b["accuracy"] for b in batch_rewards])),
                "safety": float(np.mean([b["safety"] for b in batch_rewards])),
                "completeness": float(
                    np.mean([b["completeness"] for b in batch_rewards])
                ),
                "format": float(np.mean([b["format"] for b in batch_rewards])),
            }
            viz.log_reward_vector(avg_reward_dict, step=step)
            viz.log_dimension_state(D, step=step)

            # Gradient scale simulation
            scales = []
            for pg_idx in range(8):  # Simulate 8 param groups
                scale = 1.0 + args.beta * (D[pg_idx % 4].item() - 1.0)
                scales.append(scale)
            viz.log_gradient_scales(scales, step=step)

        # Heatmap every 100 steps
        if step % config.visualizer.heatmap_interval == 0 and step > 0:
            history = dim_state.get_history()
            if len(history) > 1:
                viz.log_dimension_state_heatmap(D, step=step, history=history)

    elapsed = time.time() - start_time
    print("-" * 60)
    print(f"Simulation complete in {elapsed:.2f}s ({args.steps / elapsed:.0f} steps/s)")
    print()

    # Final summary
    print("=" * 60)
    print("  Final Dimension State")
    print("=" * 60)
    print(dim_state.summary())
    print()

    # Show D vector trajectory
    history = dim_state.get_history()
    print(f"D vector trajectory shape: {history.shape}")
    print(f"  Initial: {history[0]}")
    print(f"  Final:   {history[-1]}")
    print(f"  Range:   [{history.min():.4f}, {history.max():.4f}]")

    # Per-dimension analysis
    print()
    print("Dimension Analysis:")
    for i, name in enumerate(config.dimension_names):
        dim_traj = history[:, i]
        trend = "UP" if dim_traj[-1] > dim_traj[0] else "DN"
        print(
            f"  {name:15s}: start={dim_traj[0]:.4f} end={dim_traj[-1]:.4f} "
            f"mean={dim_traj.mean():.4f} trend={trend}"
        )

    viz.close()
    print()
    print("Done! To view TensorBoard logs:")
    print("  tensorboard --logdir ./runs/demo_mixed_reward")
    print()


if __name__ == "__main__":
    main()
