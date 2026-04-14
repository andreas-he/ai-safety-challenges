"""Tests for KL divergence implementation."""

import numpy as np
import pytest
from challenge import kl_divergence, asymmetry_example


# --- kl_divergence tests ---

def test_kl_identical_distributions():
    """KL(P||P) = 0 for any distribution P."""
    p = np.array([0.25, 0.25, 0.25, 0.25])
    assert abs(kl_divergence(p, p)) < 1e-10


def test_kl_non_negative():
    """KL divergence is always >= 0 (Gibbs' inequality)."""
    p = np.array([0.1, 0.4, 0.5])
    q = np.array([0.3, 0.3, 0.4])
    assert kl_divergence(p, q) >= -1e-10


def test_kl_known_value():
    """Test against a hand-computed KL value."""
    p = np.array([0.5, 0.5])
    q = np.array([0.25, 0.75])
    # KL = 0.5*ln(0.5/0.25) + 0.5*ln(0.5/0.75)
    expected = 0.5 * np.log(2.0) + 0.5 * np.log(2.0 / 3.0)
    np.testing.assert_allclose(kl_divergence(p, q), expected, atol=1e-10)


def test_kl_uniform_vs_peaked():
    """KL from uniform to peaked should be positive."""
    p = np.array([0.5, 0.5])
    q = np.array([0.99, 0.01])
    result = kl_divergence(p, q)
    assert result > 0


def test_kl_zero_p_handled():
    """When p[i] = 0, that term contributes 0 to KL."""
    p = np.array([0.0, 1.0])
    q = np.array([0.5, 0.5])
    expected = 1.0 * np.log(1.0 / 0.5)
    np.testing.assert_allclose(kl_divergence(p, q), expected, atol=1e-10)


def test_kl_zero_q_where_p_positive():
    """When p[i] > 0 but q[i] = 0, KL should be inf."""
    p = np.array([0.5, 0.5])
    q = np.array([1.0, 0.0])
    assert kl_divergence(p, q) == float('inf')


def test_kl_larger_distribution():
    """Test with a larger distribution."""
    rng = np.random.default_rng(42)
    raw_p = rng.uniform(0.1, 1.0, size=10)
    raw_q = rng.uniform(0.1, 1.0, size=10)
    p = raw_p / raw_p.sum()
    q = raw_q / raw_q.sum()
    result = kl_divergence(p, q)
    assert result >= -1e-10
    assert np.isfinite(result)


# --- asymmetry_example tests ---

def test_asymmetry_returns_valid():
    """Check that asymmetry_example returns valid distributions and values."""
    P, Q, kl_pq, kl_qp = asymmetry_example()
    np.testing.assert_allclose(P.sum(), 1.0, atol=1e-7)
    np.testing.assert_allclose(Q.sum(), 1.0, atol=1e-7)
    assert np.all(P >= 0)
    assert np.all(Q >= 0)


def test_asymmetry_values_correct():
    """Check that returned KL values match recomputed ones."""
    P, Q, kl_pq, kl_qp = asymmetry_example()
    np.testing.assert_allclose(kl_pq, kl_divergence(P, Q), atol=1e-7)
    np.testing.assert_allclose(kl_qp, kl_divergence(Q, P), atol=1e-7)


def test_asymmetry_is_actually_asymmetric():
    """The example must actually demonstrate asymmetry."""
    P, Q, kl_pq, kl_qp = asymmetry_example()
    assert abs(kl_pq - kl_qp) > 1e-6, "KL(P||Q) and KL(Q||P) should differ meaningfully"
