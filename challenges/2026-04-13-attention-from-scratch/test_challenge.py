"""Tests for Scaled Dot-Product Attention."""
import numpy as np
import pytest
from challenge import softmax, attention


class TestSoftmax:
    def test_sums_to_one(self):
        x = np.array([1.0, 2.0, 3.0])
        result = softmax(x)
        np.testing.assert_allclose(result.sum(), 1.0, atol=1e-6)

    def test_all_positive(self):
        x = np.random.randn(10)
        result = softmax(x)
        assert np.all(result > 0)

    def test_numerical_stability(self):
        """Large values shouldn't cause overflow."""
        x = np.array([1000.0, 1001.0, 1002.0])
        result = softmax(x)
        assert np.all(np.isfinite(result))
        np.testing.assert_allclose(result.sum(), 1.0, atol=1e-6)

    def test_2d(self):
        x = np.random.randn(3, 4)
        result = softmax(x, axis=-1)
        np.testing.assert_allclose(result.sum(axis=-1), np.ones(3), atol=1e-6)

    def test_preserves_shape(self):
        x = np.random.randn(5, 7)
        assert softmax(x).shape == (5, 7)


class TestAttention:
    def test_output_shape(self):
        np.random.seed(42)
        Q = np.random.randn(10, 64)
        K = np.random.randn(10, 64)
        V = np.random.randn(10, 32)
        output, weights = attention(Q, K, V)
        assert output.shape == (10, 32)
        assert weights.shape == (10, 10)

    def test_weights_sum_to_one(self):
        np.random.seed(42)
        Q = np.random.randn(5, 16)
        K = np.random.randn(5, 16)
        V = np.random.randn(5, 8)
        _, weights = attention(Q, K, V)
        np.testing.assert_allclose(weights.sum(axis=-1), np.ones(5), atol=1e-6)

    def test_weights_nonnegative(self):
        np.random.seed(42)
        Q = np.random.randn(4, 8)
        K = np.random.randn(4, 8)
        V = np.random.randn(4, 6)
        _, weights = attention(Q, K, V)
        assert np.all(weights >= 0)

    def test_scaling_prevents_saturation(self):
        """With proper scaling, attention should not be one-hot even with large d_k."""
        np.random.seed(42)
        d_k = 128
        Q = np.random.randn(4, d_k) * 3
        K = np.random.randn(4, d_k) * 3
        V = np.random.randn(4, 8)
        _, weights = attention(Q, K, V)
        entropy = -np.sum(weights * np.log(weights + 1e-10), axis=-1)
        assert np.mean(entropy) > 0.1

    def test_masking(self):
        """Masked positions should get near-zero attention weight."""
        np.random.seed(42)
        Q = np.random.randn(3, 8)
        K = np.random.randn(3, 8)
        V = np.random.randn(3, 4)
        mask = np.triu(np.ones((3, 3), dtype=bool), k=1)
        _, weights = attention(Q, K, V, mask=mask)
        assert weights[0, 1] < 1e-6
        assert weights[0, 2] < 1e-6
        assert weights[1, 2] < 1e-6
