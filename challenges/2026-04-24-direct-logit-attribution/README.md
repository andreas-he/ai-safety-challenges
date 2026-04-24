# Direct Logit Attribution — Decomposing Transformer Predictions

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/andreas-he/ai-safety-challenges/blob/main/challenges/2026-04-24-direct-logit-attribution/challenge.ipynb)

**Format:** short-coding | **Category:** Mech Interp | **Difficulty:** standard | **~25 min**

## Problem

In a residual-stream transformer the final logit for any token `t` is approximately:

```
logit(t) ≈ W_U[t] @ x_final
```

Because `x_final = embed + Σ attn_heads + Σ mlps` is a **sum**, we can distribute:

```
logit(t) ≈ Σ_components (W_U[t] @ component_output_c)
```

This is **Direct Logit Attribution (DLA)**: each component gets a scalar credit toward predicting `t`.
On the IOI task ("Mary and John went to the store, John gave the bag to ___") the **logit difference**
(IO token − S token) tells us which heads are name-movers vs. S-inhibitors.

## How to solve

1. Open the notebook in Colab (badge above)
2. Implement `token_logit_contribution`, `compute_dla`, and `top_k_contributors`
3. Run all assertion cells — green = done
4. Check the collapsed Solution cell to compare your approach

## Relevance

DLA is the first diagnostic in every circuit paper (IOI, refusal direction, indirect object identification).
Implementing it from scratch makes the residual stream's linearity concrete — and shows exactly where
that linearity breaks down (which is one of the open problems in Sharkey et al. 2025).

## Further reading

📚 [Open Problems in Mechanistic Interpretability](https://arxiv.org/abs/2501.16496)
— Sharkey, Chughtai, Batson, Lindsey, Wu, Bushnaq, …Nanda, McGrath (2025).
Section on attribution methods surveys where DLA works, fails, and what stronger tools are needed.
