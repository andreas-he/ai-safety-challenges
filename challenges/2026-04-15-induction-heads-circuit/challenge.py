"""
Challenge: Circuit Analysis — Induction Heads

Induction heads are a canonical two-head circuit discovered by Olsson et al. (2022) in
"In-context Learning and Induction Heads." They implement in-context pattern matching:
given a repeated sequence [A, B, ..., A, B, ...], an induction head at position i
attends back to the token *after* the previous occurrence of the current token,
enabling the model to predict that the same continuation will follow.

Example:
    Sequence: [A, B, C, D, A, B, C, D]   (positions 0-7, seq_half=4)
    At position 5 (second B): induction head attends to position 2
    (the position *after* the first B, where C appeared)

Your task: implement four functions to detect and rank induction heads
using their attention weight matrices.
"""

import numpy as np
from typing import List

# ── Data generation (fully implemented — do not modify) ──────────────────────


def make_repeated_sequence(vocab_size: int = 20, seq_half: int = 10, seed: int = 42) -> np.ndarray:
    """
    Generate a repeated token sequence of length 2 * seq_half.
    The second half is an exact copy of the first half.

    Returns:
        tokens: np.ndarray of shape (2*seq_half,), dtype int
    """
    rng = np.random.default_rng(seed)
    first_half = rng.integers(0, vocab_size, size=seq_half)
    return np.concatenate([first_half, first_half])


def make_attention_pattern(
    tokens: np.ndarray,
    head_type: str = "induction",
    noise: float = 0.1,
    seed: int = 0,
) -> np.ndarray:
    """
    Simulate an (L, L) causal attention weight matrix for a given head type.

    Rows are queries (destination), columns are keys (source).
    Entry [i, j] is the attention weight from position i to position j.
    Rows sum to 1; the upper triangle is causally masked.

    head_type:
        "induction"      — attends strongly to (i - seq_half + 1) in the second half
        "previous_token" — attends strongly to (i - 1) at every position
        "random"         — roughly uniform attention (no structured pattern)

    Args:
        tokens:    shape (L,) integer token sequence
        head_type: one of "induction", "previous_token", "random"
        noise:     std of Gaussian noise added before softmax
        seed:      random seed

    Returns:
        attn: (L, L) float array, rows sum to 1, causally masked
    """
    rng = np.random.default_rng(seed)
    L = len(tokens)
    logits = np.full((L, L), -1e9)

    # Causal mask: position i may attend to positions 0..i
    for i in range(L):
        logits[i, : i + 1] = 0.0

    if head_type == "induction":
        seq_half = L // 2
        for i in range(seq_half, L):
            target = i - seq_half + 1
            if 0 <= target <= i:
                logits[i, : i + 1] = -5.0   # suppress non-target positions
                logits[i, target] = 10.0     # strong attention on induction target

    elif head_type == "previous_token":
        for i in range(1, L):
            logits[i, : i + 1] = -5.0       # suppress non-target positions
            logits[i, i - 1] = 10.0          # strong attention on previous token

    # "random" keeps logits[i, :i+1] = 0 → near-uniform distribution after softmax

    # Add small Gaussian noise (causal: lower-triangular only)
    noise_matrix = rng.normal(0, noise, size=(L, L))
    logits += np.tril(noise_matrix)

    def _softmax(x: np.ndarray) -> np.ndarray:
        x = x - np.max(x)
        e = np.exp(x)
        return e / e.sum()

    return np.array([_softmax(logits[i]) for i in range(L)])


# ── Your implementations below ────────────────────────────────────────────────


def compute_induction_score(attn: np.ndarray) -> float:
    """
    Compute the induction score for an attention pattern on a repeated sequence.

    For a sequence of length L = 2 * seq_half, the induction score is the mean
    attention weight placed on the induction position — defined as (i - seq_half + 1)
    for each query position i in the second half of the sequence.

    Args:
        attn: (L, L) causal attention weight matrix (rows sum to 1)

    Returns:
        score: float in [0, 1] — higher means more induction-like

    Example:
        For L=20 (seq_half=10) and a perfect induction head:
            At i=10: induction_pos=1, attn[10, 1] ≈ 1.0
            At i=11: induction_pos=2, attn[11, 2] ≈ 1.0  ... and so on
            score ≈ 1.0
    """
    raise NotImplementedError("TODO: implement compute_induction_score")


def compute_previous_token_score(attn: np.ndarray) -> float:
    """
    Compute the previous-token score for an attention pattern.

    The score is the mean attention weight placed on position (i - 1),
    averaged over all query positions i = 1, 2, ..., L-1.

    Args:
        attn: (L, L) causal attention weight matrix (rows sum to 1)

    Returns:
        score: float in [0, 1] — higher means more previous-token-like
    """
    raise NotImplementedError("TODO: implement compute_previous_token_score")


def classify_head(attn: np.ndarray, threshold: float = 0.4) -> str:
    """
    Classify an attention head by its dominant pattern.

    Classification rules (applied in order):
        1. If induction_score >= threshold        → return "induction"
        2. If previous_token_score >= threshold   → return "previous_token"
        3. Otherwise                             → return "unclassified"

    Args:
        attn:      (L, L) causal attention weight matrix
        threshold: minimum score to assign a label (default 0.4)

    Returns:
        label: "induction", "previous_token", or "unclassified"
    """
    raise NotImplementedError("TODO: implement classify_head")


def rank_heads_by_induction(attention_patterns: List[np.ndarray]) -> List[int]:
    """
    Rank a list of attention heads by their induction score (descending).

    Args:
        attention_patterns: list of (L, L) causal attention matrices, one per head

    Returns:
        ranked_indices: list of head indices sorted by descending induction score

    Example:
        If scores are [0.1, 0.9, 0.5] for heads [0, 1, 2]:
            ranked_indices = [1, 2, 0]
    """
    raise NotImplementedError("TODO: implement rank_heads_by_induction")
