# 2026-04-16 — Logit Difference (the Workhorse Mech-Interp Metric)

**Format:** short-coding &middot; **Category:** Mech Interp &middot; **Est:** 25 min

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/andreas-he/ai-safety-challenges/blob/main/challenges/2026-04-16-logit-difference/challenge.ipynb)

## The problem

Implement three tiny utilities that together give you the full activation-patching metric stack:

1. `logit_diff(logits, correct, incorrect)` — scalar case
2. `logit_diff_batch(logits, correct_idx, incorrect_idx)` — batched, no Python loop
3. `patching_score(clean_ld, corrupted_ld, patched_ld)` — normalised 0→1 score

All pure numpy. See `challenge.ipynb` for the full statement, starter code, and inline asserts.

## How to solve

1. Click the Colab badge above (or clone locally and run in Jupyter)
2. Fill in the `raise NotImplementedError` stubs
3. Run each cell; the asserts validate your implementation
4. Stuck? Unfold the `<details>Solution</details>` cell at the bottom of the notebook

## Why this matters

Logit difference is the currency of mech-interp circuit analysis — IOI, activation patching, attribution patching, direct logit attribution all boil down to `logits[correct] - logits[incorrect]` and its normalised cousins. Being fluent with it cuts the time-to-first-result on any circuit-analysis project in half.
