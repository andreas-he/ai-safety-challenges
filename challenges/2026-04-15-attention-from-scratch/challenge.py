"""
Implement Scaled Dot-Product Attention from scratch.

Given Q, K, V matrices, compute:
    Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V

Use only NumPy — no PyTorch or TensorFlow.
"""

import numpy as np


def softmax(x: np.ndarray) -> np.ndarray:
    """Compute numerically stable softmax along the last axis.

    Args:
        x: Input array of any shape. Softmax is applied along axis=-1.

    Returns:
        Array of same shape with softmax applied along last axis.
    """
    # TODO: Implement numerically stable softmax
    raise NotImplementedError


def attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray) -> np.ndarray:
    """Compute scaled dot-product attention.

    Args:
        Q: Query matrix of shape [seq_len, d_k]
        K: Key matrix of shape [seq_len, d_k]
        V: Value matrix of shape [seq_len, d_v]

    Returns:
        Output matrix of shape [seq_len, d_v]
    """
    # TODO: Implement scaled dot-product attention
    raise NotImplementedError
