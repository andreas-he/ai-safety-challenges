#!/usr/bin/env bash
# Setup script for ai-safety-challenges repo
# Prerequisites:
#   1. Create the repo on GitHub (web UI):
#      https://github.com/new — name: ai-safety-challenges, public, no README
#   2. Run: cd /tmp/ai-safety-challenges && bash setup-repo.sh

set -e

REPO="andreas-he/ai-safety-challenges"

echo "=== Setting up ai-safety-challenges repo ==="

git config user.name "Life Brain"
git config user.email "brain@andreas.he"

# Step 1: Stash solution.py files aside (they go to solutions branch only)
echo "1/5 Separating solutions from main content..."
mkdir -p /tmp/solutions-stash
find challenges -name "solution.py" -type f | while IFS= read -r f; do
  dir="/tmp/solutions-stash/$(dirname "$f")"
  mkdir -p "$dir"
  cp "$f" "$dir/"
  rm "$f"
done

# Step 2: Commit main branch (everything except solution.py)
echo "2/5 Committing main branch..."
git add -A
git commit -m "Initial scaffold: CI workflow, seed challenges, stats tracking

[skip ci]"

# Step 3: Push main
echo "3/5 Pushing main branch..."
git remote add origin "https://github.com/${REPO}.git" 2>/dev/null || true
git push -u origin main

# Step 4: Create orphan solutions branch with only solution.py files
echo "4/5 Creating solutions branch..."
git checkout --orphan solutions
git rm -rf . > /dev/null 2>&1
# Restore solution files from stash
cp -r /tmp/solutions-stash/challenges/ challenges/ 2>/dev/null || true
find challenges -name "solution.py" -type f -exec git add {} \;
git commit -m "Reference solutions for seed challenges"

# Step 5: Push solutions branch
echo "5/5 Pushing solutions branch..."
git push -u origin solutions

# Return to main
git checkout main
rm -rf /tmp/solutions-stash

echo ""
echo "=== Done! ==="
echo "Repo: https://github.com/${REPO}"
echo "Main branch: challenges without solutions"
echo "Solutions branch: reference solutions only (revealed by CI after passing tests)"
