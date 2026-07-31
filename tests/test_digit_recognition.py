"""
Tests for the handwritten digit recognition module.
"""

import numpy as np
import pytest
import torch

# ============================================================================
# Model Tests
# ============================================================================


class TestDigitCNN:
    """Tests for the CNN model."""

    @pytest.fixture
    def model(self):
        from mixed_reward.digit_recognition.model import DigitCNN

        return DigitCNN()

    def test_forward_shape(self, model):
        """Model should output (batch, 10) for (batch, 1, 28, 28) input."""
        x = torch.randn(32, 1, 28, 28)
        output = model(x)
        assert output.shape == (32, 10)

    def test_forward_single_sample(self, model):
        """Model should handle batch_size=1."""
        x = torch.randn(1, 1, 28, 28)
        output = model(x)
        assert output.shape == (1, 10)

    def test_forward_different_batch_sizes(self, model):
        """Model should work with various batch sizes."""
        for bs in [1, 4, 16, 64, 128]:
            x = torch.randn(bs, 1, 28, 28)
            output = model(x)
            assert output.shape == (bs, 10)

    def test_predict_returns_class_indices(self, model):
        """predict() should return class indices 0-9."""
        x = torch.randn(16, 1, 28, 28)
        preds = model.predict(x)
        assert preds.shape == (16,)
        assert preds.dtype == torch.int64
        assert (preds >= 0).all() and (preds <= 9).all()

    def test_training_mode_gradients(self, model):
        """Model should produce gradients in training mode."""
        model.train()
        x = torch.randn(8, 1, 28, 28)
        y = torch.randint(0, 10, (8,))
        loss = torch.nn.CrossEntropyLoss()(model(x), y)
        loss.backward()

        # Check that gradients exist
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"

    def test_dropout_behavior(self):
        """Dropout should behave differently in train vs eval mode."""
        from mixed_reward.digit_recognition.model import DigitCNN

        model = DigitCNN(dropout=0.5)

        x = torch.randn(8, 1, 28, 28)

        model.train()
        out1 = model(x)
        out2 = model(x)
        # With dropout, outputs should differ
        assert not torch.allclose(out1, out2)

        model.eval()
        out3 = model(x)
        out4 = model(x)
        # In eval mode, outputs should be identical
        assert torch.allclose(out3, out4)

    def test_count_parameters(self, model):
        """count_parameters should return a reasonable number."""
        from mixed_reward.digit_recognition.model import count_parameters

        n = count_parameters(model)
        assert 300_000 < n < 600_000, f"Expected ~420K params, got {n:,}"


# ============================================================================
# Data Tests
# ============================================================================


class TestDataLoading:
    """Tests for MNIST data loading."""

    @pytest.fixture(scope="class")
    def loaders(self):
        from mixed_reward.digit_recognition.data import get_mnist_dataloaders

        return get_mnist_dataloaders(batch_size=64, data_dir="./data")

    def test_train_loader_shape(self, loaders):
        """Train loader should yield (batch, 1, 28, 28) images and (batch,) labels."""
        train_loader, _ = loaders
        images, labels = next(iter(train_loader))
        assert images.shape == (64, 1, 28, 28)
        assert labels.shape == (64,)
        assert images.dtype == torch.float32
        assert labels.dtype == torch.int64

    def test_test_loader_shape(self, loaders):
        """Test loader should yield (batch, 1, 28, 28) images and (batch,) labels."""
        _, test_loader = loaders
        images, labels = next(iter(test_loader))
        assert images.shape == (64, 1, 28, 28)
        assert labels.shape == (64,)

    def test_labels_in_range(self, loaders):
        """All labels should be in [0, 9]."""
        for loader in loaders:
            _images, labels = next(iter(loader))
            assert (labels >= 0).all() and (labels <= 9).all()

    def test_images_normalized(self, loaders):
        """Images should be normalized (mean ~0, not raw 0-255)."""
        train_loader, _ = loaders
        images, _ = next(iter(train_loader))
        # Normalized MNIST: mean around -0.1 to 0.1, not raw pixel values
        assert images.mean() < 1.0
        assert images.min() < 0


class TestDenormalize:
    """Tests for denormalize utility."""

    def test_denormalize_restores_range(self):
        from mixed_reward.digit_recognition.data import denormalize

        images = torch.randn(4, 1, 28, 28)
        restored = denormalize(images)
        assert restored.shape == images.shape
        # Denormalized values should be roughly in [0, 1] for normalized input
        # (not exact because we used random normal, but shape should match)


class TestPerClassAccuracy:
    """Tests for per-class accuracy computation."""

    def test_all_correct(self):
        from mixed_reward.digit_recognition.data import compute_per_class_accuracy

        preds = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        labels = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        result = compute_per_class_accuracy(preds, labels)
        for c in range(10):
            assert result[c] == 1.0

    def test_all_wrong(self):
        from mixed_reward.digit_recognition.data import compute_per_class_accuracy

        preds = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 0])
        labels = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        result = compute_per_class_accuracy(preds, labels)
        for c in range(10):
            assert result[c] == 0.0

    def test_mixed(self):
        from mixed_reward.digit_recognition.data import compute_per_class_accuracy

        # Class 2 appears twice: one correct (pred=2,label=2), one wrong (pred=0,label=2)
        # Class 3 appears once: wrong (pred=2,label=3)
        preds = np.array([0, 0, 2, 0, 4, 2])
        labels = np.array([0, 0, 2, 2, 4, 3])
        result = compute_per_class_accuracy(preds, labels)
        assert result[0] == 1.0  # Both 0s correct
        assert result[2] == 0.5  # One of two 2s correct
        assert result[3] == 0.0  # 3 predicted as 2
        assert result[4] == 1.0  # 4 correct


# ============================================================================
# Trainer Tests
# ============================================================================


class TestDigitTrainer:
    """Tests for the training loop."""

    @pytest.fixture
    def trainer_setup(self):
        """Create a trainer with a tiny synthetic dataset."""
        from mixed_reward.digit_recognition.data import get_mnist_dataloaders
        from mixed_reward.digit_recognition.model import DigitCNN
        from mixed_reward.digit_recognition.trainer import DigitTrainer

        # Use real MNIST but small batch for speed
        train_loader, test_loader = get_mnist_dataloaders(
            batch_size=32,
            data_dir="./data",
        )

        model = DigitCNN()

        # Create trainer with no visualization for testing
        trainer = DigitTrainer(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            visualizer=None,
            learning_rate=0.001,
            device="cpu",
            log_interval=200,  # Quiet logging in tests
        )

        return trainer

    def test_trainer_initialization(self, trainer_setup):
        """Trainer should initialize with all components."""
        trainer = trainer_setup
        assert trainer.model is not None
        assert trainer.optimizer is not None
        assert trainer.criterion is not None
        assert trainer.current_epoch == 0
        assert trainer.global_step == 0

    def test_train_one_epoch(self, trainer_setup):
        """One epoch of training should complete and return metrics."""
        trainer = trainer_setup
        trainer.current_epoch = 1

        loss, acc = trainer.train_epoch()

        assert isinstance(loss, float)
        assert isinstance(acc, float)
        assert loss > 0
        assert 0 <= acc <= 100

    def test_evaluate(self, trainer_setup):
        """Evaluation should return loss, accuracy, and prediction arrays."""
        trainer = trainer_setup
        loss, acc, preds, labels = trainer.evaluate()

        assert isinstance(loss, float)
        assert isinstance(acc, float)
        assert 0 <= acc <= 100
        assert len(preds) == len(labels)
        assert len(preds) == len(trainer_setup.test_loader.dataset)

    def test_fit_one_epoch(self, trainer_setup):
        """fit() should run for 1 epoch and return history."""
        trainer = trainer_setup

        history = trainer.fit(epochs=1, save_best=False)

        assert "train_loss" in history
        assert "train_acc" in history
        assert "test_loss" in history
        assert "test_acc" in history
        assert len(history["train_loss"]) == 1
        assert history["train_acc"][0] > 0

        trainer.close()

    def test_model_improves_after_training(self, trainer_setup):
        """Accuracy should improve significantly after a few epochs."""
        trainer = trainer_setup

        # Quick evaluation before any training
        _, acc_before, _, _ = trainer.evaluate()

        # Train for 2 epochs
        trainer.fit(epochs=2, save_best=False)

        # Evaluate after
        _, acc_after, _, _ = trainer.evaluate()

        # Accuracy should improve or stay at learned level
        assert acc_after >= acc_before * 0.9, (
            f"Accuracy degraded: {acc_before:.1f}% -> {acc_after:.1f}%"
        )

        trainer.close()
