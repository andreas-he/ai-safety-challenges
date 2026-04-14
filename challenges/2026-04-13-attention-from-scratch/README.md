# Implement Scaled Dot-Product Attention

**Category:** Coding | **Difficulty:** Medium | **Date:** 2026-04-13

## Why this matters

Scaled dot-product attention is the core primitive of every transformer you'll
analyze in mech interp work. Understanding it from scratch — including the
scaling factor and its effect on softmax saturation — is foundational before
studying induction heads, attention patterns, or the IOI circuit.

## Problem

Implement `attention(Q, K, V)` from scratch using only NumPy.

Given:
- `Q` of shape `(seq_len, d_k)` — query matrix
- `K` of shape `(seq_len, d_k)` — key matrix
- `V` of shape `(seq_len, d_v)` — value matrix

Your function should:
1. Compute raw attention scores: `Q @ K.T`
2. Scale by `1/sqrt(d_k)` to prevent softmax saturation
3. Apply softmax row-wise to get attention weights
4. Compute the output: `weights @ V`

Also implement the `softmax` helper with numerical stability.

**Bonus:** Support causal masking via the `mask` parameter.

## Running tests

```bash
pytest test_challenge.py -v
```

## Hints

<details>
<summary>Hint 1</summary>
For numerically stable softmax: subtract the row maximum before exponentiating —
exp(x - max(x)) gives the same result but avoids overflow.
</details>

<details>
<summary>Hint 2</summary>
Causal masking: create an upper-triangular boolean mask with np.triu, then set
masked positions to -1e9 before softmax.
</details>
