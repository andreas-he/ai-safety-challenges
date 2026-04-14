# AI Safety Challenges

Daily coding and math challenges tailored to technical AI safety. Builds research muscle for mechanistic interpretability, ML math, and safety-relevant coding.

## How It Works

1. A new challenge appears each weekday morning in `challenges/`
2. Pull the latest, open the challenge directory in your IDE
3. Read `README.md` for the problem statement
4. Implement your solution in `challenge.py`
5. Run `pytest test_challenge.py -v` to validate
6. Commit and push — CI evaluates automatically
7. On pass, the reference solution is revealed from the `solutions` branch

## Quick Start

```bash
git clone https://github.com/andreas-he/ai-safety-challenges.git
cd ai-safety-challenges
pip install -r requirements.txt
```

## Daily Workflow

```bash
git pull
cd challenges/<today's-challenge>/
# Read README.md, edit challenge.py
pytest test_challenge.py -v
git commit -am "solve: <slug>" && git push
```

## Challenge Types

| Day | Category | Format |
|-----|----------|--------|
| Mon, Thu | Coding | Implement from scratch (attention, SAE, probes) |
| Tue, Fri | Math | Proofs, derivations, information theory |
| Wed | Conceptual-Coding | Circuit analysis, probe design, feature geometry |

## Structure

```
challenges/
  YYYY-MM-DD-slug/
    README.md           # Problem statement
    challenge.py        # Starter code with TODOs
    test_challenge.py   # Pytest suite (validates your solution)
    metadata.json       # Category, difficulty, hints
    solution.py         # Reference solution (solutions branch only)
```
