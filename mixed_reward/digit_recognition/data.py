"""
MNIST data loading and visualization utilities.

Provides:
- DataLoader creation with standard MNIST normalization
- Prediction visualization for logging to WandB / TensorBoard
- Sample grid generation
"""

from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# MNIST standard normalization values
MNIST_MEAN = 0.1307
MNIST_STD = 0.3081


def get_transforms(train: bool = True) -> transforms.Compose:
    """
    Get standard MNIST transforms.

    Training: normalization only (no augmentation needed for MNIST baseline).
    Test: normalization only.
    """
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((MNIST_MEAN,), (MNIST_STD,)),
        ]
    )


def get_mnist_dataloaders(
    batch_size: int = 64,
    data_dir: str = "./data",
    num_workers: int = 0,
    val_split: float = 0.0,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create MNIST train and test DataLoaders.

    Downloads MNIST automatically on first run.

    Args:
        batch_size: Batch size for both loaders.
        data_dir: Directory to store/download MNIST data.
        num_workers: Number of data loading worker processes.
        val_split: Fraction of training data to use as validation (0 = no split).

    Returns:
        (train_loader, test_loader) tuple.
    """
    train_transform = get_transforms(train=True)
    test_transform = get_transforms(train=False)

    full_train = datasets.MNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=train_transform,
    )

    test_dataset = datasets.MNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=test_transform,
    )

    if val_split > 0:
        val_size = int(len(full_train) * val_split)
        train_size = len(full_train) - val_size
        train_dataset, _ = torch.utils.data.random_split(
            full_train, [train_size, val_size]
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )
    else:
        train_loader = DataLoader(
            full_train,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, test_loader


def denormalize(images: torch.Tensor) -> torch.Tensor:
    """Reverse MNIST normalization for visualization."""
    mean = torch.tensor(MNIST_MEAN).view(1, 1, 1)
    std = torch.tensor(MNIST_STD).view(1, 1, 1)
    return images * std + mean


def create_prediction_grid(
    images: torch.Tensor,
    labels: torch.Tensor,
    predictions: torch.Tensor,
    max_samples: int = 16,
) -> "plt.Figure":
    """
    Create a matplotlib figure showing images with true and predicted labels.

    Correct predictions shown in green, incorrect in red.

    Args:
        images: Batch of images (B, 1, 28, 28) -- normalized.
        labels: True digit labels (B,).
        predictions: Predicted digit labels (B,).
        max_samples: Maximum number of samples to display.

    Returns:
        matplotlib Figure.
    """
    import matplotlib.pyplot as plt

    n = min(len(images), max_samples)
    cols = min(8, n)
    rows = (n + cols - 1) // cols

    # Denormalize for display
    images_denorm = denormalize(images[:n])

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)

    for i in range(n):
        r, c = i // cols, i % cols
        ax = axes[r, c]
        img = images_denorm[i].squeeze().numpy()
        ax.imshow(img, cmap="gray")
        true_label = labels[i].item()
        pred_label = predictions[i].item()
        color = "green" if true_label == pred_label else "red"
        ax.set_title(f"T:{true_label} P:{pred_label}", color=color, fontsize=10)
        ax.axis("off")

    # Hide unused subplots
    for i in range(n, rows * cols):
        r, c = i // cols, i % cols
        axes[r, c].axis("off")

    plt.tight_layout()
    return fig


def create_sample_grid(
    dataloader: DataLoader,
    n_samples: int = 16,
) -> "plt.Figure":
    """
    Create a figure showing sample MNIST digits with their labels.

    Args:
        dataloader: DataLoader to sample from.
        n_samples: Number of digits to display.

    Returns:
        matplotlib Figure.
    """
    import matplotlib.pyplot as plt

    images, labels = next(iter(dataloader))
    n = min(len(images), n_samples)
    cols = min(8, n)
    rows = (n + cols - 1) // cols

    images_denorm = denormalize(images[:n])

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)

    for i in range(n):
        r, c = i // cols, i % cols
        ax = axes[r, c]
        ax.imshow(images_denorm[i].squeeze().numpy(), cmap="gray")
        ax.set_title(f"Label: {labels[i].item()}", fontsize=10)
        ax.axis("off")

    for i in range(n, rows * cols):
        r, c = i // cols, i % cols
        axes[r, c].axis("off")

    plt.tight_layout()
    return fig


def compute_per_class_accuracy(
    all_preds: np.ndarray,
    all_labels: np.ndarray,
    num_classes: int = 10,
) -> dict:
    """
    Compute per-class accuracy from accumulated predictions.

    Args:
        all_preds: Array of predicted class indices (N,).
        all_labels: Array of true class indices (N,).
        num_classes: Number of classes.

    Returns:
        Dict mapping class index -> accuracy.
    """
    per_class = {}
    for c in range(num_classes):
        mask = all_labels == c
        if mask.sum() > 0:
            per_class[c] = (all_preds[mask] == all_labels[mask]).mean()
        else:
            per_class[c] = 0.0
    return per_class
