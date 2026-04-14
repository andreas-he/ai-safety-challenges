# Build a Linear Probe for Sentiment

**Category:** Coding | **Difficulty:** Medium | **Date:** 2026-04-14

## Why this matters

Linear probing is the workhorse of mechanistic interpretability. If a direction
in the residual stream linearly separates a concept, the model represents that
concept as a direction — the Linear Representation Hypothesis. Building one from
scratch cements your intuition for what "a feature direction" actually means.

## Problem

You have simulated GPT-2 layer-6 residual stream activations (d=768) with a
known true sentiment direction embedded in Gaussian noise.

Implement:
1. `train_probe(X_train, y_train)` — train a logistic regression probe
2. `cosine_similarity(a, b)` — compute cosine similarity between two vectors
3. `random_baseline_accuracy(X_test, y_test)` — random direction baseline

The probe's weight vector should recover the true sentiment direction (high
cosine similarity) and achieve >90% test accuracy.

## Running tests

```bash
pytest test_challenge.py -v
```

## Hints

<details>
<summary>Hint 1</summary>
The probe weight vector w = probe.coef_[0] is a direction in R^768. Its cosine
similarity with the true direction should approach 1.0 when signal_strength is
high.
</details>

<details>
<summary>Hint 2</summary>
For the random baseline: project onto a random unit vector, threshold at the
median. This gives ~50% accuracy and shows the probe captures genuine structure.
</details>
