# Implement Scaled Dot-Product Attention

**Category:** Coding | **Difficulty:** Medium | **Date:** 2026-04-15

## Why This Matters

Attention is the core mechanism in transformers. Understanding it from scratch — without framework abstractions — is essential for mechanistic interpretability work, where you need to reason about what attention heads are doing at the mathematical level.

## Problem

Implement the scaled dot-product attention function using only NumPy.

Given query (Q), key (K), and value (V) matrices:

1. Compute the attention scores: `scores = Q @ K^T / sqrt(d_k)`
2. Apply softmax to get attention weights
3. Multiply weights by values to get the output

Your implementation must:
- Handle arbitrary batch dimensions (2D matrices: `[seq_len, d_k]`)
- Include the scaling factor `sqrt(d_k)` to prevent softmax saturation
- Implement a numerically stable softmax

## Running Tests

```bash
pytest test_challenge.py -v
```

## Hints

<details>
<summary>Hint 1</summary>
For numerically stable softmax, subtract the max value from each row before exponentiating: `exp(x - max(x))`.
</details>

<details>
<summary>Hint 2</summary>
The scaling factor `d_k` is the last dimension of Q (or K). Without scaling, dot products grow with dimension, pushing softmax into regions with tiny gradients.
</details>
