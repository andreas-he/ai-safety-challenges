# Causal Scrubbing — From Patching to Full Hypothesis Testing

**Date:** 2026-04-23  
**Format:** explain  
**Category:** Mech Interp  
**Difficulty:** standard  
**Streak:** Day 9

---

## Relevance

Causal scrubbing is the rigorous upgrade to activation patching — it tests entire subgraph hypotheses, not just individual node importance. The IOI circuit paper (Wang et al.) and ACDC (Conmy et al.) both depend on this logical foundation. With LASR CodeSignal due tomorrow, being able to articulate exactly *what additional claim* activation patching makes vs. causal scrubbing — and why the distinction matters for verifying a circuit — is the kind of depth that separates a strong LASR candidate from a surface-level one.

---

## Problem

The following is adapted from **"Causal Scrubbing: a method for rigorously testing interpretability hypotheses"** (Chan, Arber, Conmy, Lindsey, Gould, Kran, McDiarmid, Henighan, and Nanda; Redwood Research / Anthropic, 2022):

> Existing methods — most prominently activation patching — tell us that a component is *causally important* for a behaviour. But they cannot confirm whether the **pathway** through which it acts matches our mechanistic hypothesis. A more rigorous test asks: if we replace every activation in the network that our hypothesis says is *irrelevant* with an activation drawn from a reference input, does the network's output remain unchanged?
>
> We call this **causal scrubbing**. Formally, given a hypothesis H — a subgraph G_H of the full computational graph together with an interchangeability relation on inputs — the procedure works as follows. For every node v *not* in G_H, replace v's activation with that of a reference input x′ drawn from a distribution ρ. The replacement propagates forward normally: nodes downstream of v receive v's replaced activation. If the final output after all replacements is statistically indistinguishable from the original output, H is **consistent** with the computation.
>
> Activation patching, as used in Wang et al. (2022) and subsequent work, is a strict special case of causal scrubbing: G_H consists of a **single node**, and ρ is the activation distribution induced by one fixed corrupted input x̄ rather than a reference population.

Given this, explain:

**(a)** Suppose you've confirmed via activation patching that head 9.9 in GPT-2 small is causally important for the IOI task. Two competing mechanistic hypotheses remain: (i) head 9.9 directly copies the IO token identity into the final-position residual stream; (ii) head 9.9 encodes a structural feature of the sentence which MLP11 later decodes into the IO logit. **Why does activation patching fail to distinguish (i) from (ii)?** What additional information does causal scrubbing supply?

**(b)** The passage requires that out-of-circuit node replacements use activations drawn from a **reference distribution ρ**, not a single fixed corrupted input x̄. What goes wrong if you use x̄ for every out-of-circuit replacement — and how does this failure mode connect to the zero-ablation critique?

---

## Hints

<details>
<summary>Hint 1</summary>

For (a): think about what activation patching actually measures — it restores *one* node's activation and asks whether output improves. Now ask: if head 9.9 carries the right information in both hypotheses (just via different downstream paths), would patching behave differently between them? What aspect of the *downstream computation* is invisible to patching?

</details>

<details>
<summary>Hint 2</summary>

For (b): ask what property ρ guarantees that a single fixed x̄ does not. If x̄ is a corrupted input (names-swapped IOI sentence), the activations it produces at out-of-circuit nodes aren't neutral — they're systematically corrupted. How does that distort what you're actually measuring? Does this sound familiar from zero-ablation's failure mode?

</details>

---

## Solution

<details>
<summary><b>Solution</b></summary>

### Part (a) — Why patching fails to distinguish the two hypotheses

Activation patching restores head 9.9's activation in the corrupted run and measures the resulting change in output — but it cannot see *what happens downstream* of that node. Both hypotheses (i) and (ii) predict that patching head 9.9 will improve the logit difference: in (i) because the IO token identity is now written directly to the final residual stream; in (ii) because the structural feature that MLP11 needs is now present. From patching's vantage point, both produce the same outcome: output improves when head 9.9 is restored.

Causal scrubbing distinguishes them by testing the **full subgraph**. Under hypothesis (i), G_H includes head 9.9 → final residual stream → unembedding, but *excludes* MLP11. Under hypothesis (ii), MLP11 must also be included. You run scrubbing under each hypothesis separately: replace all out-of-G_H nodes with reference activations and check whether output is preserved. If it is only when MLP11 is included, hypothesis (ii) is supported; if it holds even when MLP11 is replaced, hypothesis (i) is supported. The key additional claim causal scrubbing tests is: **are the downstream paths consistent with G_H?** Patching tests one node in isolation; scrubbing tests whether an entire hypothesized circuit, and only that circuit, accounts for the behavior.

### Part (b) — Why a single x̄ corrupts the measurement

Using a single fixed corrupted input x̄ for every out-of-circuit replacement conflates two distinct effects: (1) "this node's activation doesn't carry task-relevant information" and (2) "this node now carries information *actively hostile* to the task." If x̄ is a names-swapped IOI sentence, the activations it induces at out-of-circuit nodes aren't neutral — they encode the *wrong* IO token direction. Replacing every irrelevant node with x̄'s activations simultaneously injects many corrupting signals, so any output failure may reflect interference from those spurious signals rather than evidence that the hypothesized subgraph is incomplete.

Reference distribution ρ solves this by averaging over many inputs: individual inputs' spurious signals cancel out, so the replacement activations are neither systematically helpful nor harmful — they are simply uninformative about the target behavior. This is exactly the zero-ablation failure mode: zero-ablation replaces activations with the all-zeros vector, which is out-of-distribution and actively disrupts downstream computations, making it impossible to cleanly attribute output failures to the ablated component vs. to the perturbation's distributional violence. A fixed x̄ corrupts for the same reason: the reference isn't neutral, it's adversarial.

</details>

---

*Part of the [AI Safety Challenges](https://github.com/andreas-he/ai-safety-challenges) daily practice series.*
