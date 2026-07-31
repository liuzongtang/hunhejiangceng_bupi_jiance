"""
Handwritten digit recognition module.

Provides a CNN model, data loading, and training loop with live
visualization for MNIST digit classification.

Usage:
    from mixed_reward.digit_recognition import (
        DigitCNN, DigitTrainer, get_mnist_dataloaders, create_model
    )

    model = create_model()
    train_loader, test_loader = get_mnist_dataloaders(batch_size=64)
    trainer = DigitTrainer(model, train_loader, test_loader)
    history = trainer.fit(epochs=5)
"""

from mixed_reward.digit_recognition.data import (
    compute_per_class_accuracy,
    create_prediction_grid,
    create_sample_grid,
    get_mnist_dataloaders,
)
from mixed_reward.digit_recognition.model import (
    DigitCNN,
    count_parameters,
    create_model,
)
from mixed_reward.digit_recognition.trainer import DigitTrainer

__all__ = [
    "DigitCNN",
    "DigitTrainer",
    "compute_per_class_accuracy",
    "count_parameters",
    "create_model",
    "create_prediction_grid",
    "create_sample_grid",
    "get_mnist_dataloaders",
]
