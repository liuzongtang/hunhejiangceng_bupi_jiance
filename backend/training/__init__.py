"""
Hybrid Reward Model Training Module.

Implements the 7-dimension scalar+dimension hybrid reward mechanism
for fabric defect detection training (Section 5 of the project document).

Key components:
  - DimensionRewardComputer: D01-D07 reward computation
  - HybridRewardTrainer: Full training loop with adaptive weighting
  - DefectAugmentation: Data augmentation pipeline
  - TrainingMetrics: Monitoring (Section 5.5)

Usage:
    from backend.training import HybridRewardTrainer, DimensionRewardComputer

    trainer = HybridRewardTrainer(model, train_loader, val_loader)
    metrics = trainer.fit(epochs=200)
"""

from backend.training.augmentation import DefectAugmentation, build_augmentation
from backend.training.dataset import (
    DatasetConfig,
    FabricDataset,
    auto_discover_dataset,
    create_dataloaders,
)
from backend.training.dimension_rewards import (
    DIMENSION_NAMES,
    NUM_DIMENSIONS,
    DimensionRewardComputer,
    compute_reward_tensor,
)
from backend.training.hybrid_reward_trainer import HybridRewardTrainer
from backend.training.loss_functions import (
    consistency_loss,
    detection_loss,
    scalar_reward_loss,
    total_loss,
)
from backend.training.metrics import TrainingMetrics
from backend.training.model import SimpleDefectDetector
from backend.training.visualizer import TrainingVisualizer

__all__ = [
    "DIMENSION_NAMES",
    "NUM_DIMENSIONS",
    "DatasetConfig",
    "DefectAugmentation",
    "DimensionRewardComputer",
    "FabricDataset",
    "HybridRewardTrainer",
    "SimpleDefectDetector",
    "TrainingMetrics",
    "TrainingVisualizer",
    "auto_discover_dataset",
    "build_augmentation",
    "compute_reward_tensor",
    "consistency_loss",
    "create_dataloaders",
    "detection_loss",
    "scalar_reward_loss",
    "total_loss",
]
