"""Tests for Linear Probe for Sentiment."""
import numpy as np
import pytest
from challenge import generate_data, train_probe, cosine_similarity, random_baseline_accuracy
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


@pytest.fixture
def data():
    return generate_data(n=500, d=768, signal_strength=3.0, seed=42)


@pytest.fixture
def split_data(data):
    activations, labels, s = data
    X_train, X_test, y_train, y_test = train_test_split(
        activations, labels, test_size=0.2, random_state=0
    )
    return X_train, X_test, y_train, y_test, s


class TestTrainProbe:
    def test_returns_fitted_model(self, split_data):
        X_train, _, y_train, _, _ = split_data
        probe = train_probe(X_train, y_train)
        assert hasattr(probe, "predict")
        assert hasattr(probe, "coef_")

    def test_correct_weight_shape(self, split_data):
        X_train, _, y_train, _, _ = split_data
        probe = train_probe(X_train, y_train)
        assert probe.coef_.shape == (1, 768)

    def test_high_accuracy(self, split_data):
        X_train, X_test, y_train, y_test, _ = split_data
        probe = train_probe(X_train, y_train)
        acc = accuracy_score(y_test, probe.predict(X_test))
        assert acc > 0.90, f"Probe accuracy should be >90%, got {acc:.3f}"


class TestCosineSimilarity:
    def test_identical(self):
        v = np.array([1.0, 2.0, 3.0])
        np.testing.assert_allclose(cosine_similarity(v, v), 1.0, atol=1e-6)

    def test_opposite(self):
        v = np.array([1.0, 2.0, 3.0])
        np.testing.assert_allclose(cosine_similarity(v, -v), -1.0, atol=1e-6)

    def test_orthogonal(self):
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([0.0, 1.0, 0.0])
        np.testing.assert_allclose(cosine_similarity(a, b), 0.0, atol=1e-6)

    def test_probe_recovers_direction(self, split_data):
        X_train, _, y_train, _, s = split_data
        probe = train_probe(X_train, y_train)
        cos_sim = cosine_similarity(probe.coef_[0], s)
        assert abs(cos_sim) > 0.85, f"Probe should recover true direction, got {cos_sim:.4f}"


class TestRandomBaseline:
    def test_near_chance(self, split_data):
        _, X_test, _, y_test, _ = split_data
        acc = random_baseline_accuracy(X_test, y_test)
        assert isinstance(acc, float)
        assert 0.3 < acc < 0.7, f"Random baseline should be ~50%, got {acc:.3f}"
