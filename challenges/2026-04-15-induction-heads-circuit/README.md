# Circuit Analysis: Induction Heads

**Category:** Conceptual-coding | **Difficulty:** Medium | **Date:** 2026-04-15

---

## Why This Matters

Induction heads are one of the most important circuits discovered through mechanistic
interpretability. First described by Olsson et al. (2022) in *In-context Learning and
Induction Heads*, this two-head circuit implements **in-context pattern matching**:
given a repeated sequence `[A, B, ..., A, B, ...]`, the induction head predicts that
`B` follows `A` by attending back to the token *after* the previous occurrence of the
current token.

Understanding how to detect induction heads programmatically is a core skill for circuit
analysis — the foundation of mechanistic interpretability and a central topic in the
ARENA curriculum and Neel Nanda's 200 Concrete Open Problems.

---

## Background: The Two-Head Induction Circuit

The circuit is a **composition across two layers**:

1. **Previous-token head** (typically layer 0):
   - At each position `i`, attends strongly to position `i-1`
   - Writes token `t-1`'s information into the key at position `i`

2. **Induction head** (typically layer 1):
   - Via Q-K composition with the previous-token head, looks for positions where the
     *preceding* token matches the current token `t-1`
   - In a repeated sequence, this means attending to position `i - k + 1`
     (the slot *after* the first occurrence of the current token)

### The Induction Signature on Repeated Sequences

For a repeated sequence of length `2k`:

```
tokens:    [a,  b,  c,  d,  |  a,  b,  c,  d ]
positions:  0   1   2   3   |  4   5   6   7      (k = 4)
```

At position `5` (second `b`): the induction head should attend to position
`5 - 4 + 1 = 2` (the `c` that followed `b` the first time around).

This produces a characteristic **shifted diagonal** in the attention matrix:
for rows `k` to `2k-1`, high weight falls on column `i - k + 1`.

---

## Your Tasks

`challenge.py` gives you:
- **`make_repeated_sequence`** — generates a repeated token sequence
- **`make_attention_pattern`** — simulates realistic attention matrices for three head types

You must implement four functions:

| Function | What it computes |
|---|---|
| `compute_induction_score(attn)` | Mean attention on the induction position across second-half rows |
| `compute_previous_token_score(attn)` | Mean attention on position `i-1` across all rows |
| `classify_head(attn, threshold=0.4)` | Label a head: `"induction"`, `"previous_token"`, or `"unclassified"` |
| `rank_heads_by_induction(patterns)` | Sort a list of heads by descending induction score |

None of these require training a model — they are **pure numpy operations on attention matrices**.

---

## Running Tests

```bash
pip install numpy pytest
pytest test_challenge.py -v
```

All 15 tests should pass with a correct implementation.

---

## Hints

<details>
<summary>Hint 1: Induction score formula</summary>

For a sequence of length `L = 2 * seq_half`, iterate over positions in the second half:

```python
seq_half = L // 2
scores = []
for i in range(seq_half, L):
    induction_pos = i - seq_half + 1
    scores.append(attn[i, induction_pos])
return float(np.mean(scores))
```

This extracts the weight on the "one step after the previous occurrence" column at each second-half row and averages them.

</details>

<details>
<summary>Hint 2: Classifier ordering</summary>

Check the induction score *before* the previous-token score:

```python
if compute_induction_score(attn) >= threshold:
    return "induction"
elif compute_previous_token_score(attn) >= threshold:
    return "previous_token"
return "unclassified"
```

Reason: in the first half of the sequence, the induction head has no signal and
behaves roughly uniformly — giving it a modest previous-token score. Checking
induction first prevents mis-classifying it.

</details>
