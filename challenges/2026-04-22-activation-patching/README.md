# Activation Patching — Why Corruption Design Is the Real Science

**Date:** 2026-04-22 | **Format:** Explain | **Category:** Mech Interp | **Difficulty:** Quick (15 min)

> **Relevance:** Activation patching is the workhorse causal technique in mech interp — central to the IOI circuit paper (Wang et al. 2022), ACDC (Conmy et al. 2023), and the Thought Anchors ablation framework. Being able to explain *why* patching is causal and *why* zero-ablation gives misleading answers is the kind of conceptual precision that distinguishes candidates in technical AI safety interviews.

---

## Problem

The following is adapted from **"Interpretability in the Wild: a Circuit for Indirect Object Identification in GPT-2 small"** (Wang et al., 2022, arXiv:2211.00593, Section 2):

> We measure the importance of each model component to the IOI task using **activation patching** (also called causal tracing). The procedure runs the model twice:
>
> 1. *Clean run:* The model processes an IOI sentence such as *"When Mary and John went to the store, John gave a book to"* and computes activations at every layer and head.
> 2. *Corrupted run:* The model processes a semantically plausible but task-relevant corrupted version — typically the same sentence with the subject names swapped (*"When John and Mary went to the store, Mary gave a book to"*). This causes the model to predict the wrong indirect object (IO token).
>
> To measure whether a component c is causally important for the IOI behavior, we run the corrupted input forward but **intervene at c**: we replace c's activation during the corrupted run with its counterpart from the clean run. We then measure the *logit difference* of the correct IO token vs. the subject (S) token in the model's final output.
>
> A high logit-difference restoration indicates that c carries information critical to the task.

Given this experimental setup, explain:

**(a)** Why does activation patching yield a genuinely **causal** — not just correlational — measure of a component's importance to the IOI task?

**(b)** Why would replacing c's activation with **zero** ("zero-ablation") give a *different* result, and why is that result harder to interpret?

---

## Hints

<details>
<summary>Hint 1</summary>

Think about what "causal" means in the sense of Pearl's do-calculus: a causal measurement requires *intervening* on a variable (setting it to a value), not just *observing* it. How does activation patching structure itself as an intervention rather than an observation? Could any confound explain a logit-difference restoration if *only* c was restored from the corrupted baseline?

</details>

<details>
<summary>Hint 2</summary>

For zero-ablation: a transformer was trained on realistic activations sampled from real data. Is `[0, 0, ..., 0]` ever a realistic residual stream value at a given position? If the zero vector is out-of-distribution, what is the model really responding to — the "absence" of a signal, or the *presence* of an unusual, never-seen input?

</details>

---

## Solution

<details>
<summary><b>Solution</b></summary>

### Part (a) — Causal, not correlational

Activation patching is causal because it directly *intervenes* on a component's activation value and measures the downstream effect — rather than observing correlations between that component and the output across many inputs. Correlational methods can be confounded: two attention heads might both correlate with correct IO predictions because both respond to a shared upstream signal (e.g. both fire on long-range token pairs), not because either one drives the IO token into the output. Patching isolates the counterfactual: we hold *all* other activations at the corrupted baseline and restore *only* c, so any output recovery must be attributable to c alone.

The logic mirrors Pearl's *do-calculus* — we are asking "what happens if we **set** c's activation to its clean value?" not "when c **happens** to have its clean value, what is the output?" No upstream confounder can explain a logit-difference restoration, because the only change between the two runs is the single intervened component.

### Part (b) — Why zero-ablation is different and harder to interpret

Zero-ablation sets c to the all-zeros vector, which is almost certainly out-of-distribution for a trained transformer: the residual stream was never presented with `[0, 0, ..., 0]` at that position during training, so downstream components will respond to an input they were never designed to handle. This means zero-ablation measures something closer to "how much does model behavior collapse when c is *destroyed*?" rather than "does c specifically carry the IOI-relevant signal?"

A component that plays a load-bearing structural role — contributing the bulk of residual stream magnitude that a later LayerNorm depends on, or feeding scale-sensitive attention keys — will appear highly important under zero-ablation even if it plays *no role* in retrieving the correct IO token. The corrupted-run baseline keeps the model in a realistic activation regime (just one where the task-relevant direction is absent), so restoration specifically isolates task contribution, not structural necessity. This is why ACDC and most circuit-discovery work use corrupted-run patching, not zero-ablation: the latter conflates "this component matters for the task" with "this component is architecturally load-bearing."

---

**Key insight:** The choice of corruption is not a detail — it defines what question you are actually asking. A good corrupted input is one that is *in-distribution* for the model but *off-distribution* for the specific task, so that restoration cleanly measures task contribution. Designing corruptions that satisfy this constraint is often the hardest methodological choice in a circuit investigation.

</details>

---

## Further Reading

📚 **[Open Problems in Mechanistic Interpretability](https://arxiv.org/abs/2501.16496)** — Sharkey, Chughtai, Batson, Lindsey, Wu, Bushnaq, ...Nanda, McGrath (2025). Section 4 surveys circuit discovery methods and their ablation choices in depth — a map of where the field is and where it's going.
