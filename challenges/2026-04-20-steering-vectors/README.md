# Steering Vectors — Editing Model Behavior via Residual Stream Directions

**Date:** 2026-04-20 | **Format:** short-coding | **Category:** Mech Interp | **Difficulty:** standard

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/andreas-he/ai-safety-challenges/blob/main/challenges/2026-04-20-steering-vectors/challenge.ipynb)

---

## Problem

Steering vectors are directions in activation space that reliably separate two semantic clusters — e.g. "honest" vs "deceptive", "safe" vs "harmful". To steer a model, you add a scaled version of this direction to the residual stream at inference time, shifting model behavior without any weight update.

Implement three pure-numpy utilities that together form a complete steering-vector pipeline:

1. **`compute_steering_vector(pos_acts, neg_acts)`** — given activations for positive (`n_pos × d`) and negative (`n_neg × d`) examples at some layer, compute the **mean-difference direction** and return it normalized to unit L2 norm.

2. **`projection_onto_vector(acts, v)`** — project each row of `acts` (`n × d`) onto unit vector `v` (`d`,). Returns shape `(n,)` of signed scalar projections. Positive = points toward positive class.

3. **`steer_activations(acts, v, alpha)`** — add `alpha` copies of `v` to every row of `acts`. Returns shape `(n, d)`. Preserves all components orthogonal to `v`.

All functions are ≤ 2 lines each. Dependencies: numpy only.

---

## Relevance

Steering vectors (Turner et al. 2023 "Activation Addition", Zou et al. 2023 "Representation Engineering") are one of the most practically important tools in AI safety: they let you test whether dangerous capabilities can be suppressed via activation-level edits, and whether concept representations are linear. Core vocabulary for LASR/SAIGE interviews and directly relevant to the inverse-scaling incoherence project.

---

## How to solve

1. Click **Open in Colab** above
2. Fill the three `raise NotImplementedError` stubs (each ≤ 2 lines)
3. Run each assert cell — green output = correct
4. Check the collapsed `Solution` cell only after you've tried

---

<details>
<summary><b>Solution</b></summary>

```python
def compute_steering_vector(pos_acts, neg_acts):
    v = pos_acts.mean(axis=0) - neg_acts.mean(axis=0)
    return v / np.linalg.norm(v)

def projection_onto_vector(acts, v):
    return acts @ v  # v is unit norm — dot product = projection

def steer_activations(acts, v, alpha):
    return acts + alpha * v  # broadcasts (n,d) + (d,) correctly
```

**Key insight.** All three functions are one line each. The power of steering vectors lies not in the arithmetic but in the *choice of contrastive pairs* — the dataset curation that determines whether `v` is a clean semantic direction or a noisy mixture of correlated concepts. In production (Zou et al., Turner et al.), `v` is averaged across layers and scaled to match the activation norm at the target layer. The numpy primitives here are identical to what runs inside those experiments.

</details>
