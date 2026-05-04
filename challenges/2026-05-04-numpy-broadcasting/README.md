# Broadcasting Without Loops — The Shape Rules That Make NumPy Fast

**Day 16 | ML Engineering | Monday short-coding | 2026-05-04**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/andreas-he/ai-safety-challenges/blob/main/challenges/2026-05-04-numpy-broadcasting/challenge.ipynb)

## Relevance

NumPy broadcasting is the floor beneath every mech interp implementation you'll write — pairwise distances appear in probing, masked attention in circuit analysis, row normalization in feature geometry. LASR CodeSignal Module 3 (Python ML coding) expects fluent vectorized NumPy; loop-free implementations are a direct signal of ML engineering proficiency.

## Problem Summary

Implement three broadcast-only functions (no Python `for` loops):

1. **`pairwise_l2(X)`** — `(n, d)` → `(n, n)` distance matrix where `D[i,j] = ||X[i]-X[j]||`
2. **`standardize_rows(X)`** — `(n, d)` → `(n, d)`, each row zero-mean and unit-std
3. **`causal_mask_logits(logits)`** — `(T, T)` → `(T, T)`, upper triangle set to `-inf` via index broadcasting

## How to Solve

1. Open in Colab via the badge above
2. Fill in each `raise NotImplementedError` stub
3. Run the assert cells — all three must print `✓`
4. Check the collapsed Solution cell to compare your approach

## Key Concepts

- `X[:, None, :]` inserts a new axis — shape goes from `(n, d)` to `(n, 1, d)`
- `keepdims=True` preserves dimensions after reduction — essential for broadcasting against the original shape
- Index broadcasting (`arange[:, None]` vs `arange[None, :]`) lets you compare every (i, j) pair without a loop

## Difficulty

Standard (20–30 min)
