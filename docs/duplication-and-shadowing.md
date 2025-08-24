# Duplication and Shadowing Audit

## Executive Summary

This document identifies exact duplicates, near-duplicates, multi-source imports, shadowing incidents, and repeated imports across the codebase that could lead to maintenance issues, conflicts, or runtime errors.

## A) Exact Duplicates

### Constants

#### CANON_HEADERS
- **Source of Truth:** `utils/constants.py:2`
- **Duplicates Found:**
  - `ui/tabs/symptoms.py:18` - Redefines as `["Vital Measurement", *NODE_COLS, "Diagnostic Triage", "Actions"]`
  - `ui/tabs/outcomes.py:127` - Redefines as `["Vital Measurement","Node 1","Node 2","Node 3","Node 4","Node 5","Diagnostic Triage","Actions"]`
  - `monolith.py:30` - Redefines as `["Vital Measurement", "Node 1", "Node 2", "Node 3", "Node 4", "Node 5", "Diagnostic Triage", "Actions"]`

#### LEVEL_COLS
- **Source of Truth:** `utils/constants.py:6`
- **Duplicates Found:**
  - `ui/tabs/symptoms.py:17` - Redefines as `["Node 1", "Node 2", "Node 3", "Node 4", "Node 5"]`

#### MAX_CHILDREN_PER_PARENT
- **Source of Truth:** `utils/constants.py:13`
- **Duplicates Found:**
  - `ui/tabs/symptoms.py:8` - Imported from `utils.constants`

### Functions

#### normalize_text
- **Source of Truth:** `utils/helpers.py:6`
- **Duplicates Found:**
  - `monolith.py:142` - Redefines with different signature: `def normalize_text(x: str) -> str:`

## B) Near-Duplicates (~80%+ Similarity)

### Constants

#### CANON_HEADERS Variations
- **utils/constants.py:** `["Vital Measurement", "Node 1", "Node 2", "Node 3", "Node 4", "Node 5", "Diagnostic Triage", "Actions"]`
- **ui/tabs/symptoms.py:** `["Vital Measurement", *NODE_COLS, "Diagnostic Triage", "Actions"]` (uses NODE_COLS)
- **ui/tabs/outcomes.py:** `["Vital Measurement","Node 1","Node 2","Node 3","Node 4","Node 5","Diagnostic Triage","Actions"]` (hardcoded)
- **monolith.py:** `["Vital Measurement", "Node 1", "Node 2", "Node 3", "Node 4", "Node 5", "Diagnostic Triage", "Actions"]` (hardcoded)

**Similarity:** 95% - Only difference is whether NODE_COLS is used or hardcoded

#### LEVEL_COLS Variations
- **utils/constants.py:** `["Node 1", "Node 2", "Node 3", "Node 4", "Node 5"]`
- **ui/tabs/symptoms.py:** `["Node 1", "Node 2", "Node 3", "Node 4", "Node 5"]` (redefined)

**Similarity:** 100% - Exact duplicate

### Functions

#### normalize_text Variations
- **utils/helpers.py:** `def normalize_text(x) -> str:` (handles None, NaN, converts to string)
- **monolith.py:** `def normalize_text(x: str) -> str:` (assumes string input)

**Similarity:** 80% - Same purpose, different input handling

## C) Multi-Source Imports

### Within Same File

#### utils/__init__.py
```python
# Multiple import sources for same symbols
from .constants import (APP_VERSION, CANON_HEADERS, LEVEL_COLS, MAX_LEVELS, TAB_ICONS)
from .helpers import (normalize_text, validate_headers, ...)
from .state import (set_active_workbook, get_active_workbook, ...)
```
**Status:** ✅ Clean - Each symbol imported from single source

#### streamlit_app_upload.py
```python
# Mixed import patterns
from utils import APP_VERSION, CANON_HEADERS, LEVEL_COLS, normalize_text, validate_headers
from utils.state import (set_active_workbook, get_active_workbook, ...)
```
**Status:** ⚠️ Mixed - Some from utils package, some from utils.state

#### ui/tabs/symptoms.py
```python
# Mixed import patterns
from utils import (CANON_HEADERS, LEVEL_COLS, normalize_text, validate_headers)
import utils.state as USTATE
```
**Status:** ⚠️ Mixed - Some from utils package, some as module import

### Across Multiple Files

#### normalize_text
- **Primary source:** `utils/helpers.py`
- **Secondary source:** `monolith.py` (redefined)
- **Import pattern:** `from utils import normalize_text` (most files)

#### CANON_HEADERS
- **Primary source:** `utils/constants.py`
- **Secondary sources:** `ui/tabs/symptoms.py`, `ui/tabs/outcomes.py`, `monolith.py`
- **Import pattern:** `from utils import CANON_HEADERS` (most files)

#### LEVEL_COLS
- **Primary source:** `utils/constants.py`
- **Secondary source:** `ui/tabs/symptoms.py`
- **Import pattern:** `from utils import LEVEL_COLS` (most files)

## D) Shadowing Incidents

### Local Redefinitions

#### ui/tabs/symptoms.py:18
```python
# Imports from utils
from utils import (CANON_HEADERS, LEVEL_COLS, normalize_text, validate_headers)

# Then redefines locally
CANON_HEADERS = ["Vital Measurement", *NODE_COLS, "Diagnostic Triage", "Actions"]
```
**Risk:** HIGH - Imported name is shadowed by local definition
**Impact:** Could cause UnboundLocalError if local variable is modified

#### ui/tabs/symptoms.py:17
```python
# Imports from utils
from utils import (CANON_HEADERS, LEVEL_COLS, normalize_text, validate_headers)

# Then redefines locally
NODE_COLS = ["Node 1", "Node 2", "Node 3", "Node 4", "Node 5"]
```
**Risk:** MEDIUM - Creates new local constant, doesn't shadow import
**Impact:** Potential confusion, but no runtime risk

#### ui/tabs/outcomes.py:127
```python
# Imports from utils
from utils import (CANON_HEADERS, LEVEL_COLS, normalize_text, validate_headers)

# Then redefines locally
CANON_HEADERS = ["Vital Measurement","Node 1","Node 2","Node 3","Node 4","Node 5","Diagnostic Triage","Actions"]
```
**Risk:** HIGH - Imported name is shadowed by local definition
**Impact:** Could cause UnboundLocalError if local variable is modified

### Function Parameter Shadowing

#### ui/tabs/symptoms.py:25
```python
def parent_cols(L: int) -> str:
    """Return parent columns for level L (empty list for L=1)."""
    return NODE_COLS[:clamp_level(L) - 1]
```
**Risk:** LOW - Parameter `L` doesn't shadow any imports
**Impact:** No runtime risk

#### ui/tabs/symptoms.py:30
```python
def sanitize_parent_tuple(L: int, p) -> Tuple[str, ...]:
    # ... function body ...
    L = clamp_level(L)  # Reassigns parameter
```
**Risk:** MEDIUM - Parameter `L` is reassigned
**Impact:** Could cause confusion, but no runtime risk

## E) Repeated Imports

### Within Same File

#### ui/tabs/source.py
```python
# Multiple imports of same function
from utils.state import coerce_workbook_to_dataframes  # Line 78
from utils.state import ensure_active_sheet            # Line 91
from utils.state import get_workbook_status           # Line 106
from utils.state import ensure_active_sheet            # Line 147
from utils.state import get_workbook_status           # Line 165
from utils.state import set_active_workbook, verify_active_workbook, coerce_workbook_to_dataframes  # Line 212
from utils.state import ensure_active_sheet            # Line 230
from utils.state import get_workbook_status           # Line 248
from utils.state import ensure_active_sheet            # Line 473
from utils.state import get_workbook_status           # Line 491
```
**Status:** ❌ Poor - Multiple imports of same functions
**Recommendation:** Consolidate to single import statement

#### ui/tabs/validation.py
```python
# Multiple imports of same function
from utils.state import get_wb_nonce  # Line 66
from utils.state import get_wb_nonce  # Line 117
```
**Status:** ❌ Poor - Duplicate import
**Recommendation:** Remove duplicate

### Across Multiple Files

#### get_active_workbook
- **Import pattern:** `from utils.state import get_active_workbook`
- **Files:** 8 tabs + main app
- **Status:** ✅ Good - Consistent import pattern

#### get_current_sheet
- **Import pattern:** `from utils.state import get_current_sheet`
- **Files:** 8 tabs + main app
- **Status:** ✅ Good - Consistent import pattern

#### get_active_df
- **Import pattern:** `from utils.state import get_active_df`
- **Files:** 8 tabs + main app
- **Status:** ✅ Good - Consistent import pattern

## F) Import Pattern Analysis

### Current Patterns

#### Pattern 1: Direct Function Import
```python
from utils.state import get_active_workbook, get_current_sheet, get_active_df
```
**Usage:** `get_active_workbook()`, `get_current_sheet()`, `get_active_df()`
**Risk:** HIGH - Could cause UnboundLocalError if local variables shadow names

#### Pattern 2: Module Import
```python
import utils.state as USTATE
```
**Usage:** `USTATE.get_active_workbook()`, `USTATE.get_current_sheet()`, `USTATE.get_active_df()`
**Risk:** LOW - No risk of shadowing

#### Pattern 3: Package Import
```python
from utils import CANON_HEADERS, LEVEL_COLS, normalize_text, validate_headers
```
**Usage:** `CANON_HEADERS`, `LEVEL_COLS`, `normalize_text()`, `validate_headers()`
**Risk:** MEDIUM - Could cause UnboundLocalError if local variables shadow names

### Recommended Patterns

#### For State Functions
```python
# RECOMMENDED: Module import
import utils.state as USTATE

# Usage
wb = USTATE.get_active_workbook()
sheet = USTATE.get_current_sheet()
df = USTATE.get_active_df()
```

#### For Constants
```python
# RECOMMENDED: Module import
import utils.constants as UC

# Usage
headers = UC.CANON_HEADERS
cols = UC.LEVEL_COLS
```

#### For Helper Functions
```python
# RECOMMENDED: Module import
import utils.helpers as UH

# Usage
text = UH.normalize_text(input_value)
valid = UH.validate_headers(dataframe)
```

## G) Risk Assessment

### High Risk (Immediate Action Required)

1. **CANON_HEADERS redefinition in symptoms.py:18**
   - **Risk:** UnboundLocalError if local variable is modified
   - **Impact:** Tab could crash or render blank
   - **Fix:** Remove local definition, use imported constant

2. **CANON_HEADERS redefinition in outcomes.py:127**
   - **Risk:** UnboundLocalError if local variable is modified
   - **Impact:** Tab could crash or render blank
   - **Fix:** Remove local definition, use imported constant

3. **normalize_text redefinition in monolith.py:142**
   - **Risk:** Function signature mismatch
   - **Impact:** Runtime errors if wrong function is called
   - **Fix:** Remove duplicate, use imported function

### Medium Risk (Next Sprint)

4. **LEVEL_COLS redefinition in symptoms.py:17**
   - **Risk:** Confusion, maintenance burden
   - **Impact:** No runtime risk, but could drift from source
   - **Fix:** Remove local definition, use imported constant

5. **Mixed import patterns**
   - **Risk:** Inconsistent code style, potential confusion
   - **Impact:** No runtime risk, but maintenance burden
   - **Fix:** Standardize on module import pattern

### Low Risk (Future Improvements)

6. **Repeated imports in source.py**
   - **Risk:** Code duplication, maintenance burden
   - **Impact:** No runtime risk
   - **Fix:** Consolidate to single import statements

7. **Duplicate imports in validation.py**
   - **Risk:** Code duplication
   - **Impact:** No runtime risk
   - **Fix:** Remove duplicate imports

## H) Action Plan

### Phase 1: Critical Fixes (Immediate)
1. Remove `CANON_HEADERS` redefinition from `ui/tabs/symptoms.py:18`
2. Remove `CANON_HEADERS` redefinition from `ui/tabs/outcomes.py:127`
3. Remove `normalize_text` redefinition from `monolith.py:142`
4. Remove `LEVEL_COLS` redefinition from `ui/tabs/symptoms.py:17`

### Phase 2: Import Standardization (Next Sprint)
1. Convert all `from utils.state import X` to `import utils.state as USTATE`
2. Convert all `from utils import X` to `import utils.constants as UC` and `import utils.helpers as UH`
3. Update all function calls to use module-qualified names

### Phase 3: Code Cleanup (Future)
1. Consolidate repeated imports in `ui/tabs/source.py`
2. Remove duplicate imports in `ui/tabs/validation.py`
3. Audit for any remaining duplications

## I) Patch Suggestions

### Critical Fix 1: Remove CANON_HEADERS redefinition in symptoms.py
```diff
# ui/tabs/symptoms.py
- # Constants for canonical mapping
- NODE_COLS = ["Node 1", "Node 2", "Node 3", "Node 4", "Node 5"]
- CANON_HEADERS = ["Vital Measurement", *NODE_COLS, "Diagnostic Triage", "Actions"]
+ # Use imported constants
+ from utils.constants import NODE_COLS, CANON_HEADERS
```

### Critical Fix 2: Remove CANON_HEADERS redefinition in outcomes.py
```diff
# ui/tabs/outcomes.py
- # Ensure columns exist before editing, using CANON_HEADERS order
- CANON_HEADERS = ["Vital Measurement","Node 1","Node 2","Node 3","Node 4","Node 5","Diagnostic Triage","Actions"]
+ # Use imported constants
+ from utils.constants import CANON_HEADERS
```

### Critical Fix 3: Remove normalize_text redefinition in monolith.py
```diff
# monolith.py
- def normalize_text(x: str) -> str:
-     """Normalize text input."""
-     return x.strip() if x else ""
+ # Use imported function
+ from utils.helpers import normalize_text
```

### Import Standardization: Convert to module imports
```diff
# In all tab files
- from utils.state import get_active_workbook, get_current_sheet, get_active_df
+ import utils.state as USTATE

# Update function calls
- wb = get_active_workbook()
+ wb = USTATE.get_active_workbook()
- sheet = get_current_sheet()
+ sheet = USTATE.get_current_sheet()
- df = get_active_df()
+ df = USTATE.get_active_df()
```

---

*Report generated from analysis of `ui/tabs/*.py`, `utils/*.py`, `monolith.py`, and other source files*
