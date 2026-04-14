"""Reference solution: Scaled Dot-Product Attention."""
import numpy as np


def softmax(x, axis=-1):
    """Numerically stable softmax."""
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def attention(Q, K, V, mask=None):
    """Scaled dot-product attention.

    Key insight: 1/sqrt(d_k) prevents softmax saturation in high dimensions.
    Without it, dot products grow with d_k (variance proportional to d_k),
    pushing softmax toward one-hot — vanishing gradients in training.
    """
    d_k = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(d_k)
    if mask is not None:
        scores[mask] = -1e9
    weights = softmax(scores, axis=-1)
    output = weights @ V
    return output, weights
