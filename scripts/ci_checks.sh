#!/usr/bin/env bash
set -euo pipefail

echo "🔍 Running CI checks to prevent common bugs..."

# 1) forbid direct st.rerun or st.experimental_rerun (use safe_rerun)
echo "Checking for direct rerun calls..."
if find . -name "*.py" -not -path "./.venv/*" -not -path "./__pycache__/*" -not -path "./ui/utils/rerun.py" -exec grep -l "st\.\(experimental_\)\?rerun(" {} \; | grep -q .; then
  echo "❌ ERROR: use safe_rerun() instead of st.rerun/experimental_rerun"
  echo "Found these violations:"
  find . -name "*.py" -not -path "./.venv/*" -not -path "./__pycache__/*" -not -path "./ui/utils/rerun.py" -exec grep -H "st\.\(experimental_\)\?rerun(" {} \;
  echo ""
  echo "Fix by importing and using: from ui.utils.rerun import safe_rerun"
  exit 1
fi
echo "✅ No direct rerun calls found"

# 2) forbid shadowing state helpers as locals/params
echo "Checking for state helper shadowing in function parameters..."
if find . -name "*.py" -not -path "./.venv/*" -not -path "./__pycache__/*" -not -path "./utils/state.py" -exec grep -l "def .*(" {} \; | xargs grep -h "def .*(" | grep -E "(get_current_sheet|get_active_df|get_active_workbook|set_current_sheet|get_wb_nonce|set_active_workbook)" | grep -q .; then
  echo "❌ ERROR: do not use state helper names as parameters"
  echo "Found these violations:"
  find . -name "*.py" -not -path "./.venv/*" -not -path "./__pycache__/*" -not -path "./utils/state.py" -exec grep -H "def .*(" {} \; | grep -E "(get_current_sheet|get_active_df|get_active_workbook|set_current_sheet|get_wb_nonce|set_active_workbook)"
  echo ""
  echo "Fix by renaming parameters to avoid shadowing imported functions"
  exit 1
fi
echo "✅ No state helper shadowing in parameters found"

echo "Checking for state helper shadowing in assignments..."
if find . -name "*.py" -not -path "./.venv/*" -not -path "./__pycache__/*" -not -path "./utils/state.py" -exec grep -l "=" {} \; | xargs grep -h "=" | grep -E "\b(get_current_sheet|get_active_df|get_active_workbook|set_current_sheet|get_wb_nonce|set_active_workbook)\s*=" | grep -q .; then
  echo "❌ ERROR: do not assign to state helper names"
  echo "Found these violations:"
  find . -name "*.py" -not -path "./.venv/*" -not -path "./__pycache__/*" -not -path "./utils/state.py" -exec grep -H "=" {} \; | grep -E "\b(get_current_sheet|get_active_df|get_active_workbook|set_current_sheet|get_wb_nonce|set_active_workbook)\s*="
  echo ""
  echo "Fix by renaming local variables to avoid shadowing imported functions"
  exit 1
fi
echo "✅ No state helper shadowing in assignments found"

# 3) check for proper imports
echo "Checking for proper state helper imports..."
missing_imports=0
for file in ui/tabs/*.py; do
  if [[ -f "$file" ]]; then
    if ! grep -q "from ui.utils.rerun import safe_rerun" "$file"; then
      if grep -q "safe_rerun" "$file"; then
        echo "❌ ERROR: $file uses safe_rerun but doesn't import it"
        missing_imports=$((missing_imports + 1))
      fi
    fi
  fi
done

if [[ $missing_imports -gt 0 ]]; then
  echo "❌ Found $missing_imports files with missing safe_rerun imports"
  exit 1
fi
echo "✅ All files properly import safe_rerun"

echo ""
echo "🎉 All CI checks passed!"
echo "✅ No direct rerun calls"
echo "✅ No state helper shadowing"
echo "✅ All imports are correct"
