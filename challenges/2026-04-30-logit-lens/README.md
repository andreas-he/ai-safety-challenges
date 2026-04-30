[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/andreas-he/ai-safety-challenges/blob/main/challenges/2026-04-30-logit-lens/challenge.ipynb)

# Logit Lens — Watching Predictions Form Layer by Layer

**Date:** 2026-04-30 &nbsp;|&nbsp; **Category:** Mech Interp &nbsp;|&nbsp; **Format:** short-coding &nbsp;|&nbsp; **Difficulty:** standard

## Why this matters

The logit lens reveals *when* a transformer commits to a prediction and *which layers* are responsible for building that commitment. For the SAIGE inoculation project and LASR Stage 3 capability evaluation, understanding layer-wise representational dynamics is essential: a model that sandbags or behaves differently in evaluation contexts may show anomalous early-layer confidence patterns detectable with exactly this tool.

## How to solve

1. **Open in Colab** — click the badge above
2. **Read the problem** — the first two markdown cells explain the setup
3. **Fill in the three stubs** — each `raise NotImplementedError` becomes your implementation
4. **Run the assert cells** — all three test cells must pass without error
5. Check the collapsed `<details>Solution</details>` cell only after you've solved it

## Problem summary

A transformer's residual stream at each layer encodes a "current best guess" about the next token. The **logit lens** makes this explicit: project each layer's residual stream through the unembedding matrix and take a softmax. You will implement three functions:

1. `logit_lens(residuals, W_U)` — project all layer residuals through the unembedding matrix  
2. `token_confidence_curve(logit_lens_out, target_token_idx)` — per-layer softmax probability of a target token  
3. `first_confident_layer(confidence_curve, threshold)` — earliest layer exceeding a confidence threshold

## Setup

```python
import numpy as np
np.random.seed(42)

N_LAYERS, SEQ_LEN, D_MODEL, D_VOCAB = 4, 3, 8, 10
residuals = np.random.randn(N_LAYERS, SEQ_LEN, D_MODEL)  # [layers, positions, d_model]
W_U = np.random.randn(D_MODEL, D_VOCAB)                  # [d_model, d_vocab]
```

Dependencies: `numpy` only — runs on Colab free tier.

## Further reading

📚 Neel Nanda et al. (Google DeepMind, 2025), *A Pragmatic Vision for Interpretability* —  
https://www.lesswrong.com/posts/StENzDcD3kpfGJssR/a-pragmatic-vision-for-interpretability

---

*Part of the [AI Safety Challenges](https://github.com/andreas-he/ai-safety-challenges) series.*
