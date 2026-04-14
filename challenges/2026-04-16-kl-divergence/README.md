# KL Divergence: Properties and Intuition

**Category:** Math | **Difficulty:** Medium | **Date:** 2026-04-16

## Why This Matters

KL divergence is fundamental to model distillation, RLHF reward modeling, and understanding how language model distributions shift during fine-tuning. The asymmetry of KL has direct consequences for alignment: forward KL (mean-seeking) vs reverse KL (mode-seeking) produce qualitatively different student models.

## Problem

1. **Prove Gibbs' inequality**: Show that KL(P||Q) >= 0 for any two probability distributions P and Q, using Jensen's inequality.

2. **Implement KL divergence**: Write a function that computes KL(P||Q) for discrete distributions.

3. **Demonstrate asymmetry**: Create a concrete example where KL(P||Q) != KL(Q||P) and compute both values.

4. **Interpret for distillation**: Explain what the asymmetry means when training a student model to approximate a teacher. When would you minimize KL(teacher||student) vs KL(student||teacher)?

## Running Tests

```bash
pytest test_challenge.py -v
```

## Hints

<details>
<summary>Hint 1</summary>
Jensen's inequality: for a concave function f, E[f(X)] <= f(E[X]). The logarithm is concave.
</details>

<details>
<summary>Hint 2</summary>
For a bimodal teacher P and unimodal student Q: forward KL(P||Q) forces Q to cover all modes (mean-seeking), while reverse KL(Q||P) lets Q collapse to one mode (mode-seeking).
</details>
