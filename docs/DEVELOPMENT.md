# Development Guide

## 🚀 Quick Start

### Prerequisites
- Python 3.10, 3.11, or 3.12
- pip or conda

### Setup
1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Install development tools: `pip install ruff`

## 🛠️ Development Tools

### Ruff Linting
We use [Ruff](https://docs.astral.sh/ruff/) for fast Python linting and formatting.

```bash
# Check all files
ruff check .

# Fix auto-fixable issues
ruff check . --fix

# Format code
ruff format .
```

### Custom CI Checks
We have custom checks to prevent common bugs:

```bash
# Run all CI checks
./scripts/ci_checks.sh

# Run pre-commit checks (includes CI checks + ruff on staged files)
./scripts/pre-commit-hook.sh
```

## 🚫 Prevented Bugs

### 1. Direct Rerun Calls
**❌ Don't do this:**
```python
st.rerun()                    # Deprecated
st.experimental_rerun()       # May be removed in future
```

**✅ Do this instead:**
```python
from ui.utils.rerun import safe_rerun
safe_rerun()  # Works across all Streamlit versions
```

### 2. State Helper Shadowing
**❌ Don't do this:**
```python
def my_function(get_current_sheet):  # Shadows imported function
    pass

get_active_df = "some value"  # Shadows imported function
```

**✅ Do this instead:**
```python
def my_function(current_sheet_name):  # Different name
    pass

current_df = "some value"  # Different name
```

### 3. Missing Imports
**❌ Don't do this:**
```python
safe_rerun()  # Used without importing
```

**✅ Do this instead:**
```python
from ui.utils.rerun import safe_rerun
safe_rerun()
```

## 🔧 Pre-commit Setup

### Option 1: Manual Pre-commit Hook
```bash
# Run before each commit
./scripts/pre-commit-hook.sh
```

### Option 2: Git Pre-commit Hook (Recommended)
```bash
# Install pre-commit
pip install pre-commit

# Install the git hook
pre-commit install

# Now it runs automatically on every commit
```

### Option 3: GitHub Actions
The CI checks run automatically on every push and pull request.

## 📋 Configuration Files

### pyproject.toml
- **Ruff settings**: Line length, target Python version, rule selection
- **Import sorting**: Known first-party modules
- **Formatting**: Quote style, indentation, line endings

### .github/workflows/ci.yml
- Runs on push to main/develop and pull requests
- Installs Python 3.12 and ruff
- Runs all CI checks
- Fails if any critical bugs are found

## 🐛 Common Issues and Fixes

### Ruff Errors
```bash
# Fix auto-fixable issues
ruff check . --fix

# Show detailed output
ruff check . --output-format=text
```

### CI Check Failures
1. **Direct rerun calls**: Replace with `safe_rerun()`
2. **State helper shadowing**: Rename variables/parameters
3. **Missing imports**: Add `from ui.utils.rerun import safe_rerun`

### Import Issues
```bash
# Sort imports
ruff check . --select I --fix

# Check import formatting
ruff check . --select I
```

## 🎯 Best Practices

### 1. Always Use safe_rerun()
- Import it at the top of your file
- Use it instead of any direct Streamlit rerun calls
- It handles version compatibility automatically

### 2. Avoid Shadowing
- Never use imported function names as local variables
- Never use imported function names as function parameters
- Use descriptive names that don't conflict

### 3. Run Checks Locally
- Run `./scripts/ci_checks.sh` before pushing
- Use the pre-commit hook for automatic checking
- Fix issues before they reach the CI pipeline

### 4. Keep Imports Clean
- Use absolute imports from project root
- Group imports logically (standard library, third-party, local)
- Let ruff handle import sorting

## 🔍 Troubleshooting

### Ruff Not Found
```bash
pip install ruff
# or
conda install -c conda-forge ruff
```

### Permission Denied
```bash
chmod +x scripts/*.sh
```

### CI Checks Failing
1. Run `./scripts/ci_checks.sh` locally
2. Fix any issues found
3. Commit and push again

### Import Errors
1. Check that all required imports are present
2. Verify import paths are correct
3. Run `ruff check . --select I` to check imports

## 📚 Additional Resources

- [Ruff Documentation](https://docs.astral.sh/ruff/)
- [Streamlit Best Practices](https://docs.streamlit.io/library/advanced-features/performance)
- [Python Type Hints](https://docs.python.org/3/library/typing.html)
- [Git Pre-commit Hooks](https://pre-commit.com/)
