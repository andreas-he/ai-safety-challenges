"""Reference solution: Linear Probe for Sentiment."""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def generate_data(n=500, d=768, signal_strength=3.0, seed=42):
    """Generate simulated residual stream activations with a sentiment direction."""
    np.random.seed(seed)
    s = np.random.randn(d)
    s /= np.linalg.norm(s)
    labels = np.array([1] * (n // 2) + [0] * (n // 2))
    activations = np.random.randn(n, d)
    activations += signal_strength * labels[:, None] * s[None, :]
    return activations, labels, s


def train_probe(X_train, y_train):
    """Train logistic regression with minimal regularization.

    The weight vector w = probe.coef_[0] IS the sentiment direction in
    activation space — the Linear Representation Hypothesis in action.
    """
    probe = LogisticRegression(C=1e4, max_iter=1000)
    probe.fit(X_train, y_train)
    return probe


def cosine_similarity(a, b):
    """Cosine similarity between two vectors."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def random_baseline_accuracy(X_test, y_test, seed=123):
    """Random direction baseline — should give ~50% accuracy."""
    np.random.seed(seed)
    d = X_test.shape[1]
    random_dir = np.random.randn(d)
    random_dir /= np.linalg.norm(random_dir)
    projections = X_test @ random_dir
    preds = (projections > np.median(projections)).astype(int)
    return float(accuracy_score(y_test, preds))
