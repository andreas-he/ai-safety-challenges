"""
Build a Linear Probe for Sentiment
===================================
Train a logistic regression probe on simulated GPT-2 residual stream
activations and interpret the result geometrically.

See README.md for full problem statement.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def generate_data(n=500, d=768, signal_strength=3.0, seed=42):
    """Generate simulated residual stream activations with a sentiment direction.

    Returns:
        activations: shape (n, d) — simulated layer-6 residual stream
        labels: shape (n,) — binary sentiment (1=positive, 0=negative)
        true_direction: shape (d,) — the true sentiment direction (unit vector)
    """
    np.random.seed(seed)
    s = np.random.randn(d)
    s /= np.linalg.norm(s)
    labels = np.array([1] * (n // 2) + [0] * (n // 2))
    activations = np.random.randn(n, d)
    activations += signal_strength * labels[:, None] * s[None, :]
    return activations, labels, s


def train_probe(X_train, y_train):
    """Train a logistic regression probe on activation data.

    Args:
        X_train: training activations, shape (n_train, d)
        y_train: training labels, shape (n_train,)

    Returns:
        probe: fitted LogisticRegression model
    """
    # TODO: Train logistic regression with minimal regularization (C=1e4)
    raise NotImplementedError


def cosine_similarity(a, b):
    """Compute cosine similarity between two vectors.

    Args:
        a, b: 1D arrays of the same length

    Returns:
        float: cosine similarity in [-1, 1]
    """
    # TODO: Implement cosine similarity: dot(a,b) / (||a|| * ||b||)
    raise NotImplementedError


def random_baseline_accuracy(X_test, y_test, seed=123):
    """Compute accuracy of a random-direction baseline.

    Project test activations onto a random unit vector, classify by median threshold.

    Args:
        X_test: test activations, shape (n_test, d)
        y_test: test labels, shape (n_test,)
        seed: random seed for reproducibility

    Returns:
        float: baseline accuracy
    """
    # TODO: Generate random unit vector, project data, threshold at median
    raise NotImplementedError
