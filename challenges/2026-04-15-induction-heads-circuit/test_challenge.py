"""Tests for the Induction Heads Circuit Analysis challenge."""

import numpy as np
import pytest
from challenge import (
    make_repeated_sequence,
    make_attention_pattern,
    compute_induction_score,
    compute_previous_token_score,
    classify_head,
    rank_heads_by_induction,
)


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def repeated_tokens():
    return make_repeated_sequence(vocab_size=20, seq_half=10, seed=42)


@pytest.fixture
def induction_attn(repeated_tokens):
    return make_attention_pattern(repeated_tokens, head_type="induction", noise=0.05, seed=0)


@pytest.fixture
def prev_token_attn(repeated_tokens):
    return make_attention_pattern(repeated_tokens, head_type="previous_token", noise=0.05, seed=1)


@pytest.fixture
def random_attn(repeated_tokens):
    return make_attention_pattern(repeated_tokens, head_type="random", noise=0.1, seed=2)


# ── compute_induction_score ───────────────────────────────────────────────────


def test_induction_score_returns_scalar(induction_attn):
    """Score is a scalar (float or numpy floating)."""
    score = compute_induction_score(induction_attn)
    assert isinstance(score, (float, np.floating))


def test_induction_score_in_range(induction_attn):
    """Score is between 0 and 1 inclusive."""
    score = compute_induction_score(induction_attn)
    assert 0.0 <= score <= 1.0


def test_induction_score_high_for_induction_head(induction_attn):
    """A genuine induction head should score above 0.7."""
    score = compute_induction_score(induction_attn)
    assert score > 0.7, f"Expected > 0.7, got {score:.4f}"


def test_induction_score_low_for_previous_token_head(prev_token_attn):
    """A previous-token head is NOT an induction head; score should be below 0.3."""
    score = compute_induction_score(prev_token_attn)
    assert score < 0.3, f"Expected < 0.3, got {score:.4f}"


def test_induction_score_low_for_random_head(random_attn):
    """A random (uniform) head has no induction structure; score should be below 0.3."""
    score = compute_induction_score(random_attn)
    assert score < 0.3, f"Expected < 0.3, got {score:.4f}"


# ── compute_previous_token_score ─────────────────────────────────────────────


def test_prev_token_score_in_range(prev_token_attn):
    """Score is between 0 and 1 inclusive."""
    score = compute_previous_token_score(prev_token_attn)
    assert 0.0 <= score <= 1.0


def test_prev_token_score_high_for_prev_token_head(prev_token_attn):
    """A previous-token head should score above 0.7."""
    score = compute_previous_token_score(prev_token_attn)
    assert score > 0.7, f"Expected > 0.7, got {score:.4f}"


def test_prev_token_score_low_for_induction_head(induction_attn):
    """An induction head attends to a non-adjacent position in the second half; score < 0.6."""
    score = compute_previous_token_score(induction_attn)
    assert score < 0.6, f"Expected < 0.6, got {score:.4f}"


# ── classify_head ────────────────────────────────────────────────────────────


def test_classify_induction_head(induction_attn):
    assert classify_head(induction_attn) == "induction"


def test_classify_previous_token_head(prev_token_attn):
    assert classify_head(prev_token_attn) == "previous_token"


def test_classify_random_head(random_attn):
    assert classify_head(random_attn) == "unclassified"


# ── rank_heads_by_induction ──────────────────────────────────────────────────


def test_rank_returns_correct_length(repeated_tokens):
    """Output length matches number of input heads."""
    patterns = [
        make_attention_pattern(repeated_tokens, "induction", seed=i) for i in range(4)
    ]
    ranked = rank_heads_by_induction(patterns)
    assert len(ranked) == 4


def test_rank_contains_all_indices(repeated_tokens):
    """All original head indices appear exactly once in the ranking."""
    patterns = [
        make_attention_pattern(repeated_tokens, "induction", seed=i) for i in range(5)
    ]
    ranked = rank_heads_by_induction(patterns)
    assert sorted(ranked) == list(range(5))


def test_rank_induction_head_first(repeated_tokens):
    """The induction head (index 1) should rank highest among mixed head types."""
    patterns = [
        make_attention_pattern(repeated_tokens, "random", noise=0.1, seed=10),          # 0
        make_attention_pattern(repeated_tokens, "induction", noise=0.05, seed=11),       # 1
        make_attention_pattern(repeated_tokens, "previous_token", noise=0.05, seed=12),  # 2
    ]
    ranked = rank_heads_by_induction(patterns)
    assert ranked[0] == 1, f"Expected induction head (1) first, got order {ranked}"
