#!/usr/bin/env bash
set -euo pipefail

echo "🔍 Running pre-commit checks..."

# Run the CI checks script
if ! ./scripts/ci_checks.sh; then
  echo "❌ Pre-commit checks failed. Please fix the issues before committing."
  exit 1
fi

# Run ruff check on staged files
echo "Running ruff on staged Python files..."
staged_files=$(git diff --cached --name-only --diff-filter=ACM | grep '\.py$' || true)

if [[ -n "$staged_files" ]]; then
  echo "Checking staged files: $staged_files"
  
  # Check if ruff is available
  if command -v ruff >/dev/null 2>&1; then
    echo "Running ruff check..."
    echo "$staged_files" | xargs ruff check --output-format=text
  else
    echo "⚠️  ruff not found. Install with: pip install ruff"
    echo "Skipping ruff check..."
  fi
else
  echo "No Python files staged for commit."
fi

echo "✅ Pre-commit checks passed!"
echo "�� Ready to commit!"
