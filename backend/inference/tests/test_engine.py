"""Tests for inference engine backends."""

import numpy as np
import pytest

from backend.inference.engine import (
    DummyInferenceEngine,
    create_engine,
)


class TestDummyEngine:
    """Tests for DummyInferenceEngine."""

    @pytest.fixture
    def engine(self):
        return DummyInferenceEngine(input_size=(640, 640), seed=42)

    def test_create_dummy(self, engine):
        """Should create a dummy engine."""
        assert engine.model_path == "dummy"
        engine.load()

    def test_infer_returns_list(self, engine):
        """infer() should return a list of per-image dicts."""
        images = np.random.randn(2, 3, 640, 640).astype(np.float32)
        results = engine.infer(images)
        assert isinstance(results, list)
        assert len(results) == 2

    def test_infer_dict_keys(self, engine):
        """Each result should have expected keys."""
        images = np.random.randn(1, 3, 640, 640).astype(np.float32)
        results = engine.infer(images)
        for key in ["boxes", "scores", "classes", "class_names"]:
            assert key in results[0]

    def test_infer_empty_batch(self, engine):
        """Should handle empty batch."""
        images = np.zeros((0, 3, 640, 640), dtype=np.float32)
        results = engine.infer(images)
        assert len(results) == 0

    def test_warmup(self, engine):
        """warmup() should not crash."""
        engine.warmup(iterations=2)
        assert engine._warmed_up

    def test_benchmark(self, engine):
        """benchmark() should return timing metrics."""
        engine.warmup(2)
        metrics = engine.benchmark(batch_size=1, iterations=10)
        assert "avg_latency_ms" in metrics
        assert "fps" in metrics
        assert metrics["avg_latency_ms"] > 0

    def test_deterministic_seed(self):
        """Same seed should produce same results."""
        e1 = DummyInferenceEngine(seed=42)
        e2 = DummyInferenceEngine(seed=42)
        images = np.random.randn(1, 3, 640, 640).astype(np.float32)
        r1 = e1.infer(images)
        r2 = e2.infer(images)
        # Boxes and scores should be identical
        np.testing.assert_array_equal(r1[0]["boxes"], r2[0]["boxes"])


class TestCreateEngine:
    """Tests for the engine factory."""

    def test_create_dummy(self):
        engine = create_engine("dummy")
        assert isinstance(engine, DummyInferenceEngine)

    def test_create_unknown(self):
        with pytest.raises(ValueError, match="Unknown backend"):
            create_engine("nonexistent")
