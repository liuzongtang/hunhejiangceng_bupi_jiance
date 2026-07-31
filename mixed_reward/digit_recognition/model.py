"""
CNN model for handwritten digit recognition (MNIST).

Architecture:
    Conv1: 1 -> 32 channels, 3x3, BN, ReLU, MaxPool(2)
    Conv2: 32 -> 64 channels, 3x3, BN, ReLU, MaxPool(2)
    FC1:  64*7*7 -> 128, Dropout(0.5)
    FC2:  128 -> 10 (digits 0-9)

~50K parameters -- trains to 99%+ accuracy on MNIST in under 2 minutes on CPU.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DigitCNN(nn.Module):
    """2-layer CNN for 28x28 grayscale digit classification."""

    def __init__(self, dropout: float = 0.5):
        super().__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        # Pooling
        self.pool = nn.MaxPool2d(2, 2)

        # Fully connected layers
        # After two 2x2 pools: 28 -> 14 -> 7
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(128, 10)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Kaiming initialization for conv layers, Xavier for FC."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, 1, 28, 28).

        Returns:
            Logits tensor of shape (batch, 10).
        """
        # Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool(x)  # (batch, 32, 14, 14)

        # Block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool(x)  # (batch, 64, 7, 7)

        # Flatten + FC
        x = x.view(x.size(0), -1)  # (batch, 64*7*7)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)  # (batch, 10)

        return x

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Return class predictions (0-9)."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            return logits.argmax(dim=1)


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def create_model(dropout: float = 0.5) -> DigitCNN:
    """Factory function: create a DigitCNN with sensible defaults."""
    model = DigitCNN(dropout=dropout)
    return model
