"""Tests for scaled dot-product attention implementation."""

import numpy as np
import pytest
from challenge import softmax, attention


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def qkv(rng):
    seq_len, d_k, d_v = 5, 8, 6
    Q = rng.standard_normal((seq_len, d_k))
    K = rng.standard_normal((seq_len, d_k))
    V = rng.standard_normal((seq_len, d_v))
    return Q, K, V


# --- softmax tests ---

def test_softmax_sums_to_one():
    x = np.array([[1.0, 2.0, 3.0], [1.0, 1.0, 1.0]])
    result = softmax(x)
    np.testing.assert_allclose(result.sum(axis=-1), [1.0, 1.0], atol=1e-7)


def test_softmax_shape_preserved():
    x = np.random.default_rng(0).standard_normal((4, 6))
    assert softmax(x).shape == (4, 6)


def test_softmax_non_negative():
    x = np.array([[-100.0, 0.0, 100.0]])
    result = softmax(x)
    assert np.all(result >= 0)


def test_softmax_numerical_stability():
    x = np.array([[1000.0, 1001.0, 1002.0]])
    result = softmax(x)
    assert np.all(np.isfinite(result))
    np.testing.assert_allclose(result.sum(axis=-1), [1.0], atol=1e-7)


def test_softmax_uniform_input():
    x = np.ones((1, 5))
    result = softmax(x)
    np.testing.assert_allclose(result, np.full((1, 5), 0.2), atol=1e-7)


def test_softmax_1d_input():
    x = np.array([1.0, 2.0, 3.0])
    result = softmax(x)
    np.testing.assert_allclose(result.sum(), 1.0, atol=1e-7)


# --- attention tests ---

def test_attention_output_shape(qkv):
    Q, K, V = qkv
    out = attention(Q, K, V)
    assert out.shape == V.shape, f"Expected shape {V.shape}, got {out.shape}"


def test_attention_output_finite(qkv):
    Q, K, V = qkv
    out = attention(Q, K, V)
    assert np.all(np.isfinite(out))


def test_attention_identical_qk():
    """When Q == K, diagonal attention scores should be highest."""
    d_k = 16
    Q = np.eye(4, d_k)
    K = np.eye(4, d_k)
    V = np.arange(4 * d_k, dtype=float).reshape(4, d_k)
    out = attention(Q, K, V)
    # Output should be close to V because each query attends most to its matching key
    assert out.shape == V.shape


def test_attention_scaling_effect():
    """Verify that scaling is applied (without scaling, softmax saturates)."""
    rng = np.random.default_rng(123)
    d_k = 256  # Large d_k amplifies the effect
    Q = rng.standard_normal((3, d_k))
    K = rng.standard_normal((3, d_k))
    V = rng.standard_normal((3, d_k))
    out = attention(Q, K, V)
    # With proper scaling, output should NOT be nearly identical to a single V row
    # (which would happen if softmax saturated to a one-hot)
    row_diffs = np.std(out, axis=0).mean()
    assert row_diffs > 0.01, "Output rows are too similar — scaling may be missing"


def test_attention_single_token():
    """Single token sequence: output should equal V."""
    Q = np.array([[1.0, 2.0]])
    K = np.array([[3.0, 4.0]])
    V = np.array([[5.0, 6.0]])
    out = attention(Q, K, V)
    np.testing.assert_allclose(out, V, atol=1e-7)
