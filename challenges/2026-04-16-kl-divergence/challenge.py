"""
KL Divergence: Properties and Intuition

Implement KL divergence computation and demonstrate its key properties
relevant to model distillation and alignment.
"""

import numpy as np


def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Compute KL(P || Q) for discrete probability distributions.

    Args:
        p: Array of probabilities (must sum to 1).
        q: Array of probabilities (must sum to 1, same length as p).

    Returns:
        KL divergence value (non-negative float).

    Note:
        Handle cases where q[i] = 0 by convention: 0 * log(0/q) = 0.
        If p[i] > 0 and q[i] = 0, return float('inf').
    """
    # TODO: Implement KL divergence
    raise NotImplementedError


def asymmetry_example() -> tuple[np.ndarray, np.ndarray, float, float]:
    """Create a concrete example demonstrating KL asymmetry.

    Returns:
        Tuple of (P, Q, kl_pq, kl_qp) where:
        - P, Q are probability distributions (numpy arrays summing to 1)
        - kl_pq = KL(P||Q)
        - kl_qp = KL(Q||P)
        - kl_pq != kl_qp (the asymmetry)
    """
    # TODO: Define P, Q and compute both KL directions
    raise NotImplementedError
