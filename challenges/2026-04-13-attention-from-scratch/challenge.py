"""
Implement Scaled Dot-Product Attention
======================================
Implement attention(Q, K, V) from scratch using only NumPy.
Include the scaling factor and softmax.

See README.md for full problem statement.
"""
import numpy as np


def softmax(x, axis=-1):
    """Numerically stable softmax along the given axis.

    Args:
        x: input array
        axis: axis along which to compute softmax

    Returns:
        Array of same shape with softmax applied along axis
    """
    # TODO: Implement numerically stable softmax
    # Hint: subtract the max before exp to prevent overflow
    raise NotImplementedError


def attention(Q, K, V, mask=None):
    """Scaled dot-product attention.

    Args:
        Q: query matrix, shape (seq_len, d_k)
        K: key matrix, shape (seq_len, d_k)
        V: value matrix, shape (seq_len, d_v)
        mask: optional boolean mask — True where attention should be blocked

    Returns:
        output: shape (seq_len, d_v)
        weights: attention weights, shape (seq_len, seq_len)
    """
    # TODO: Implement scaled dot-product attention
    # 1. Compute scores = Q @ K^T
    # 2. Scale by 1/sqrt(d_k)
    # 3. Apply mask if provided (set masked positions to -1e9)
    # 4. Apply softmax row-wise
    # 5. Compute output = weights @ V
    raise NotImplementedError
