# Day 11 — SAE Reconstruction: Implement the Sparse Autoencoder Core

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/andreas-he/ai-safety-challenges/blob/main/challenges/2026-04-27-sae-reconstruction/challenge.ipynb)

**Date:** 2026-04-27 | **Format:** short-coding | **Category:** Mech Interp | **Difficulty:** standard (~25 min)

## Problem Summary

Sparse Autoencoders (SAEs) are the primary tool in modern mechanistic interpretability for decomposing transformer residual streams into interpretable features. In this challenge you implement the three core functions of the SAE training loop — from scratch, using only NumPy.

**Relevance:** Implementing the encode-decode-loss cycle is direct LASR Stage 3 prep, relevant to feature-level interventions in alignment projects, and foundational for tracking which features collapse as compute scales.

## How to Solve

1. Click **Open In Colab** above (or clone and run locally)
2. Read the problem description in each markdown cell
3. Replace `raise NotImplementedError` with your implementation
4. Run the assert cells — all passing = solved
5. Expand the `Solution` details block to compare your approach

## Tasks

| Task | Function | Core operation |
|------|----------|----------------|
| 1 | `sae_encode` | ReLU(W_enc @ (x − b_pre) + b_enc) |
| 2 | `sae_decode` | W_dec @ f + b_pre |
| 3 | `sae_loss`   | ‖x − x̂‖₂² + λ ‖f‖₁ |

## Key Concepts

- **Pre-encoder bias (b_pre):** subtracting before encoding and adding after decoding centres the reconstruction around the data mean — the SAE learns residuals, not absolute activations
- **Unit-norm decoder columns:** forces the decoder to encode *direction*, with magnitude living in the feature value; prevents magnitude-cheating that would collapse sparsity
- **ReLU sparsity:** unlike smooth activations, ReLU creates exact zeros — features are fully off or partially on, enabling circuit-level interpretability
- **L1 penalty:** creates genuine zero features (unlike L2 which only shrinks them), driving the sparse activation profile that makes features monosemantic

## Further Reading

📚 **Features as Rewards: Using Interpretability to Reduce Hallucinations (RLFR)**
*Prasad, Watts, Merullo, Gala, Lewis, McGrath, Lubana — Goodfire, 2026*
[goodfire.ai/research/rlfr](https://www.goodfire.ai/research/rlfr)

This paper extends the encode→decode pipeline you built here: SAE feature activations become *reward signals* in RLHF, turning mech interp from a diagnostic tool into an active alignment technique.
