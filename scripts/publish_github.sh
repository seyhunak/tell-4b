#!/usr/bin/env bash
# Publish tell-4b to GitHub using the GitHub CLI (gh).
# Usage: ./scripts/publish_github.sh YOUR_GITHUB_USERNAME tell-4b
set -euo pipefail

if [ "$#" -ne 2 ]; then
  echo "Usage: $0 YOUR_GITHUB_USERNAME REPO_NAME" >&2
  exit 1
fi

USER="$1"
REPO="$2"

if ! command -v git >/dev/null 2>&1; then
  echo "ERROR: git is not installed." >&2
  exit 1
fi
if ! command -v gh >/dev/null 2>&1; then
  echo "ERROR: GitHub CLI (gh) is not installed. See https://cli.github.com/" >&2
  exit 1
fi

# 1. Initialize git if necessary.
if [ ! -d .git ]; then
  git init
  echo "Initialized git repository."
fi

git add -A
if git diff --cached --quiet; then
  echo "Nothing to commit."
else
  git commit -m "Tell-4B: specialist decision model (LoRA on Qwen 4B, MPS-ready)"
fi

git branch -M main || true

# 2. Create the GitHub repository if necessary.
if gh repo view "$USER/$REPO" >/dev/null 2>&1; then
  echo "Repository $USER/$REPO already exists."
else
  gh repo create "$USER/$REPO" --public --source=. --push
  echo "Created and pushed to $USER/$REPO."
  exit 0
fi

# 3-5. Add remote, commit (done above), push.
if git remote get-url origin >/dev/null 2>&1; then
  git remote set-url origin "https://github.com/$USER/$REPO.git"
else
  git remote add origin "https://github.com/$USER/$REPO.git"
fi

git push -u origin main
echo "Pushed to https://github.com/$USER/$REPO"
