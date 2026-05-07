# Attribution Patching — Fast Causal Analysis with One Backward Pass

**Day 19 | Mech Interp | short-coding | 30 min**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/andreas-he/ai-safety-challenges/blob/main/challenges/2026-05-07-attribution-patching/challenge.ipynb)

## Problem

Activation patching — swapping a component's activation between a clean and corrupted run — is the gold standard for causal analysis in transformers, but it costs O(n) forward passes. **Attribution patching** reduces this to one forward + one backward pass using a first-order Taylor approximation:

```
score ≈ ∇_a L · (a_patch − a_clean)
```

You'll implement this pipeline in pure numpy.

## Tasks

1. `attribution_score(grad_a, a_clean, a_patch)` — scalar score for one component
2. `batched_attribution(grads, a_clean, a_patch)` — vectorized scores for all components (no loops)
3. `rank_components(scores, top_k)` — indices of top-k components by |score|

## How to solve

1. Open the notebook in Colab (badge above)
2. Implement each function stub (replace `raise NotImplementedError`)
3. Run the assert cells — passing = solved
4. Check the collapsed **Solution** cell to compare approaches

## Relevance

Attribution patching is the scalable first pass in any circuit discovery workflow. When investigating components in the SAIGE inoculation/split-personality project or running ARENA mech interp exercises, you'll use this to narrow O(n) candidates down to the handful worth full activation patching.
