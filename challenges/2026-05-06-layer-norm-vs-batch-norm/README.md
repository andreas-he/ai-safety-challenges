# Why Transformers Use Layer Norm — The Batch Statistics Problem

**Date:** 2026-05-06 | **Format:** Explain | **Category:** Deep Learning | **Difficulty:** Standard | **Streak:** 18

**Relevance:** Layer normalization is inside every transformer block you'll study in ARENA (it gates the residual stream before every attention and MLP sub-layer). Understanding *why* it replaced batch norm — not just that it did — is the floor under every mech interp experiment: activation patching, logit lens, and residual stream analysis all implicitly rely on LayerNorm's per-token independence. It also underpins the transformer internals you'll need to explain cleanly in SAIGE technical sessions.

---

## Paper Excerpt

*From "Layer Normalization"* (Ba, Kiros & Hinton, 2016, [arXiv:1607.06450](https://arxiv.org/abs/1607.06450)), Introduction:

> Training state-of-the-art, deep neural networks is computationally expensive. One way to reduce the training time is to normalize the activities of the neurons. A recently introduced technique called batch normalization uses the distribution of the summed input to a neuron over a mini-batch of training cases to compute a mean and variance which are then used to normalize the summed input to that neuron on each training case. This significantly reduces training time in feed-forward neural networks. **However, the effect of batch normalization is dependent on the mini-batch size and it is not obvious how to apply it to recurrent neural networks.**

> Unlike batch normalization, layer normalization directly estimates the normalization statistics from the summed inputs to the neurons within a hidden layer so the normalization **does not introduce any new dependencies between training cases**. This means that layer normalization can be naturally applied to recurrent neural networks as the normalization terms only depend on the summed inputs to a layer at the current time-step.

---

## Prompt

Explain why batch normalization is "not obvious how to apply" to recurrent networks and transformer language models, while layer normalization handles this naturally.

Your answer should address all three of the following:

1. **What dimension each method normalizes over** — be specific about the tensor shape `(batch, seq_len, d_model)` and which axes each method pools statistics across.
2. **Two concrete failure modes** that arise when you naively apply batch normalization to variable-length text sequences — describe what actually breaks and why.
3. **Why this matters for mech interp** — in a transformer activation patching experiment, why would batch-norm-based normalization make residual stream analysis harder or less interpretable?

Target length: 5–8 sentences covering all three points.

---

## Hints

<details>
<summary>Hint 1 — Dimension analysis</summary>

Draw a tensor of shape `(batch, seq_len, d_model)`. Batch normalization averages across all dimensions *except* the feature dimension — meaning it pools across the batch axis and the seq_len axis simultaneously. Ask: what happens when two sequences in the batch have lengths 5 and 12? Which tokens are being averaged together?

</details>

<details>
<summary>Hint 2 — Failure scenarios</summary>

Think of two separate failure scenarios: (a) deployment — what does the batch variance become when you process a single sequence at inference time? (b) training with padding — if padding tokens are included in the BN statistics, how does this distort the mean and variance for real tokens? Then ask yourself: does LayerNorm have either of these problems?

</details>

---

## Solution

<details>
<summary><b>Solution</b></summary>

Batch normalization normalizes each feature (hidden unit dimension) across the *batch* and *sequence-position* axes — for a tensor `(B, T, D)`, BN computes mean and variance by pooling over the `(B, T)` dimensions for each feature in `D`.

**Failure mode 1 — variable-length padding corruption:** in a mini-batch of text sequences with different lengths, padding tokens are often included in the `(B, T)` pool. Padding values (typically zeros) artificially deflate the mean and compress the variance, so the normalization constants seen by real tokens are corrupted by non-informative positions. Even with masking, shorter sequences contribute fewer real-token statistics than longer ones, so normalization is inconsistent across positions.

**Failure mode 2 — batch-size-1 degeneration:** at inference time (or in online settings), batch size = 1 collapses the batch-axis variance to zero, making normalization either undefined or forcing a switch to a separately-maintained running-average statistic that was computed under different training conditions — a train/inference mismatch.

Layer normalization avoids both by computing mean and variance exclusively over the *feature* dimension `D` within a single `(b, t)` position: every token is independently normalized using only its own `d_model` activations, with no cross-example or cross-position pooling.

**For mech interp:** per-token independence is essential. In an activation patching experiment you want to swap the residual stream at a specific `(b, t)` position and ask what changes downstream. If normalization statistics depended on other tokens in the batch, patching position `t` would silently change the effective normalization at every *other* position — confounding the causal attribution. LayerNorm's locality is what makes "patch here, measure there" experiments interpretable.

</details>
