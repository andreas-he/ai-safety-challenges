# Feature Superposition — Geometry of Compressed Representations

**Day 5 · Thursday · Mech Interp · ~25 min · short-coding**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/andreas-he/ai-safety-challenges/blob/main/challenges/2026-04-17-superposition-metrics/challenge.ipynb)

## Why This Matters

Superposition is the central phenomenon in *Toy Models of Superposition* (Elhage et al. 2022) — one of the must-reads on the LASR list. When `n > d` (more features than hidden dimensions), a model *must* represent features at non-orthogonal angles, trading inference noise for capacity. Today's three metrics directly implement the paper's key equations and give you the vocabulary to discuss superposition quantitatively in SAIGE/LASR interviews.

## Problem

Three pure-numpy functions measuring the geometry of a feature weight matrix `W ∈ R^{d×n}`:

1. **`normalize_columns(W)`** — project each column to unit L2 norm (prerequisite for cos-sim arithmetic).
2. **`interference_matrix(W_normed)`** — `n×n` matrix of pairwise absolute cosine similarities, diagonal zeroed. `I[i,j]` = residual activation of feature `i`'s direction when feature `j` fires.
3. **`feature_dimensionality(W_normed)`** — effective dimensionality per feature: `D_i = 1 / Σ_j (w_i · w_j)²`. Equals 1.0 in an orthonormal basis; drops below 1.0 in superposition.

## How to Solve

1. Open the notebook in Colab (badge above)
2. Fill the three `raise NotImplementedError` stubs — each is ≤ 3 lines
3. Run the assert cells — all green = solved
4. Read the collapsed `Solution` cell for the key insight

## Key Concepts

- **Gram matrix** `G = W^T W`: the n×n matrix of all pairwise cosine similarities (since columns are unit-normed)
- **Monosemantic** (D_i = 1.0): feature has a dedicated, orthogonal dimension
- **Superposed** (D_i < 1.0): feature is compressed — shares dimensions with other features
- **Total effective rank** `Σ_i D_i ≤ d`: can't exceed the actual hidden-state dimension
