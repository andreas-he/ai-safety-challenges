# Day 15 — Einstein Summation: The Notation Language of ML

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/andreas-he/ai-safety-challenges/blob/main/challenges/2026-05-01-einsum/challenge.ipynb)

**Date:** 2026-05-01 | **Format:** short-coding | **Category:** ML Engineering | **Difficulty:** standard (~25 min)

## Problem Summary

`np.einsum` (and its PyTorch equivalent) is the notation that transformer code lives in. Instead of loops, reshapes, and transposes, einsum expresses tensor contractions in a single readable string. In this challenge you implement three canonical patterns — from the outer product up to full single-head attention — using nothing but `np.einsum` and `softmax`.

**Relevance:** einsum is the floor under every ARENA exercise, every circuit analysis script, and every model implementation you'll work with in SAIGE and LASR. The LASR CodeSignal Python ML module tests tensor ops fluency directly. Every circuit analysis tool built this month (logit lens, activation patching, DLA) is one einsum call at its core.

## How to Solve

1. Click **Open In Colab** above (or clone and run locally)
2. Read the problem description in each markdown cell
3. Replace `raise NotImplementedError` with your implementation (each task is 1–3 lines)
4. Run the assert cells — all passing = solved
5. Expand the `Solution` details block to compare your approach

## Tasks

| Task | Function | Einsum pattern | Description |
|------|----------|----------------|-------------|
| 1 | `einsum_outer` | `'i,j->ij'` | Outer product of two 1D vectors |
| 2 | `einsum_batch_matmul` | `'bik,bkj->bij'` | Batched matrix multiply |
| 3 | `einsum_attention` | `'id,jd->ij'` + `'ij,jd->id'` | Single-head scaled dot-product attention |

## Key Concepts

- **Subscript convention:** each letter is a dimension name. A letter that appears in inputs but not the output is contracted (summed over). A letter in all three is a batch dimension.
- **Outer product:** when no index is shared between inputs, einsum computes the Cartesian product — `'i,j->ij'` produces an (n, m) matrix from two 1D arrays.
- **Batch matmul:** `b` appears in all three (batch dimension preserved), `k` appears in both inputs but not the output (contracted = summed over), producing the matrix product for each batch element.
- **Attention decomposition:** logits are an outer product over positions with a dot product over head dimensions (`'id,jd->ij'`); value aggregation contracts over key positions (`'ij,jd->id'`).

## Further Reading

📚 **Attention Is All You Need**
*Vaswani, Shazeer, Parmar, Uszkoreit, Jones, Gomez, Kaiser, Polosukhin (2017)*
[arxiv.org/pdf/1706.03762](https://arxiv.org/pdf/1706.03762)

The three tasks in this challenge are a direct implementation of equations (1) and (2) in this paper. Reading Section 3.2.1 after completing the challenge makes the einsum subscripts map directly to the paper's notation.
