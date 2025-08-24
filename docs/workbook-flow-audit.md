# Workbook Flow Audit

## Executive Summary

This document provides a comprehensive audit of the workbook flow from upload/ingestion through tab rendering, identifying potential blank-tab root causes and flow inconsistencies.

## A) Flow Mapping (Upload → Tabs)

### Upload/Ingest Paths

#### 1. File Upload Flow (`ui/tabs/source.py`)
```
File Upload → DataFrame Creation → Workbook Validation → set_active_workbook() → ensure_active_sheet() → safe_rerun()
```

**Key Session State Changes:**
- `st.session_state["workbook"] = clean_wb` (via `set_active_workbook`)
- `st.session_state["current_sheet"] = picked` (via `ensure_active_sheet`)
- `st.session_state["sheet_name"] = picked` (via `ensure_active_sheet`)
- `st.session_state["wb_nonce"] += "•"` (via `set_active_workbook`)

**Critical Functions:**
- `coerce_workbook_to_dataframes()` - Ensures all entries are DataFrames
- `set_active_workbook()` - Sets workbook and bumps nonce
- `ensure_active_sheet()` - Sets current_sheet and sheet_name

#### 2. Google Sheets Flow (`ui/tabs/source.py`)
```
Google Sheets Connection → DataFrame Reading → Workbook Update → set_active_workbook() → ensure_active_sheet() → safe_rerun()
```

**Key Session State Changes:**
- `st.session_state["workbook"] = updated_wb` (via `set_active_workbook`)
- `st.session_state["current_sheet"] = sheet_name` (via `ensure_active_sheet`)
- `st.session_state["sheet_name"] = sheet_name` (via `ensure_active_sheet`)

### Session State Key Management

#### Core Workbook Keys
- **`workbook`** - Primary workbook dict (Dict[str, pd.DataFrame])
- **`current_sheet`** - Currently active sheet name
- **`sheet_name`** - Legacy sheet name (kept in sync)
- **`wb_nonce`** - Cache invalidation token

#### Legacy Keys (Migration Support)
- **`upload_workbook`** - Legacy upload workbook
- **`gs_workbook`** - Legacy Google Sheets workbook
- **`work_context`** - Source context information

#### State Synchronization
```python
# In set_current_sheet()
st.session_state["current_sheet"] = name
st.session_state["sheet_name"] = name  # Keep in sync

# In ensure_active_sheet()
st.session_state["current_sheet"] = pick
st.session_state["sheet_name"] = pick  # Keep in sync
```

### Active DataFrame Derivation

#### Primary Path
```python
# utils/state.py:get_active_df_safe()
wb = _wb_dict()  # Prefer 'workbook', else 'gs_workbook'
sheet = st.session_state.get("current_sheet") or st.session_state.get("sheet_name")
if not sheet:
    sheet = ensure_active_sheet()  # Auto-repair
df = wb.get(sheet)
```

#### Fallback Path
```python
# utils/state.py:get_active_df()
wb = get_active_workbook()  # workbook or gs_workbook
name = get_current_sheet()  # current_sheet with fallback
return wb.get(name) if wb and name else None
```

#### Header Validation
```python
# utils/helpers.py:validate_headers()
return list(df.columns[:len(CANON_HEADERS)]) == CANON_HEADERS
```

### Tab Guard Conditions

#### Source Tab
- **No guards** - Always renders
- **Purpose:** Data loading and workbook management

#### Workspace Selection Tab
- **Guards:** `wb = get_active_workbook()`, `sheet = get_current_sheet()`
- **Early Return:** If `not wb or not sheet`
- **Purpose:** Sheet selection and preview

#### Validation Tab
- **Guards:** `wb = get_active_workbook()`, `sheet = get_current_sheet()`, `df = get_active_df()`
- **Early Return:** If `not wb or not sheet` or `df is None` or `not validate_headers(df)`
- **Purpose:** Data integrity validation

#### Conflicts Tab
- **Guards:** `wb = get_active_workbook()`, `sheet = get_current_sheet()`, `df = get_active_df()`
- **Early Return:** If `not wb or not sheet` or `df is None` or `not validate_headers(df)`
- **Purpose:** Conflict detection and resolution

#### Symptoms Tab
- **Guards:** `wb = USTATE.get_active_workbook()`, `sheet = USTATE.get_current_sheet()`, `df = USTATE.get_active_df_safe()`
- **Early Return:** If `not wb` or `not sheet` or `df is None` or `status != "ok"`
- **Purpose:** Symptom quality management

#### Outcomes Tab
- **Guards:** `wb = USTATE.get_active_workbook()`, `sheet = USTATE.get_current_sheet()`, `df = USTATE.get_active_df_safe()`
- **Early Return:** If `not wb` or `not sheet` or `df is None` or `status != "ok"`
- **Purpose:** Diagnostic triage and actions

#### Calculator Tab
- **Guards:** `wb = USTATE.get_active_workbook()`, `sheet = USTATE.get_current_sheet()`, `df = USTATE.get_active_df_safe()`
- **Early Return:** If `not wb` or `not sheet` or `df is None` or `status != "ok"`
- **Purpose:** Path navigation

#### Dictionary Tab
- **Guards:** `wb = get_active_workbook()`, `sheet = get_current_sheet()`, `df = get_active_df()`
- **Early Return:** If `not wb or not sheet` or `df is None` or `not validate_headers(df)`
- **Purpose:** Vocabulary management

#### Visualizer Tab
- **Guards:** `wb = get_active_workbook()`, `sheet = get_current_sheet()`, `df = get_active_df()`
- **Early Return:** If `not wb or not sheet` or `df is None` or `not validate_headers(df)`
- **Purpose:** Tree visualization

#### Push Log Tab
- **Guards:** `wb = get_active_workbook()`, `sheet = get_current_sheet()`, `df = get_active_df()`
- **Early Return:** If `not wb or not sheet` or `df is None` or `not validate_headers(df)`
- **Purpose:** Data push operations

### Editing Write-Back Paths

#### Branch Overrides
```python
# Common pattern across tabs
overrides_all = st.session_state.get("branch_overrides", {})
overrides_sheet = overrides_all.get(sheet, {})
# ... editing logic ...
st.session_state["branch_overrides"] = overrides_all
```

#### Workbook Updates
```python
# In conflicts tab
wb = get_active_workbook() or {}
wb[sheet_name] = new_df
set_active_workbook(wb, default_sheet=sheet_name, source="conflicts_single_parent")
```

#### Cache Invalidation
```python
# In set_active_workbook()
st.session_state["wb_nonce"] = (st.session_state.get("wb_nonce") or "") + "•"
```

## B) Blank-Tab Risk & Gating

### Early Return Paths by Tab

#### Source Tab
- **File Upload Failure:** `if not clean_wb: return` (line 78)
- **No guards for rendering** - Always shows content

#### Workspace Selection Tab
- **No Workbook:** `if not wb or not sheet: return` (line 129)
- **No DataFrame:** `if df is None: return` (line 135)

#### Validation Tab
- **No Workbook:** `if not wb or not sheet: return` (line 32)
- **No DataFrame:** `if df is None: return` (line 38)
- **Invalid Headers:** `if not validate_headers(df): return` (line 42)

#### Conflicts Tab
- **No Workbook:** `if not wb or not sheet: return` (line 67)
- **No DataFrame:** `if df is None: return` (line 71)
- **Invalid Headers:** `if not validate_headers(df): return` (line 75)

#### Symptoms Tab
- **No Workbook:** `if not wb: return` (line 172)
- **No Sheet:** `if not sheet: return` (line 175)
- **No DataFrame:** `if df is None: return` (line 189)
- **Invalid Status:** `if status != "ok": return` (line 177)

#### Outcomes Tab
- **No Workbook:** `if not wb: return` (line 62)
- **No Sheet:** `if not sheet: return` (line 65)
- **No DataFrame:** `if df is None: return` (line 79)
- **Invalid Status:** `if status != "ok": return` (line 69)

#### Calculator Tab
- **No Workbook:** `if not wb: return` (line 32)
- **No Sheet:** `if not sheet: return` (line 35)
- **No DataFrame:** `if df is None: return` (line 49)
- **Invalid Status:** `if status != "ok": return` (line 39)

#### Dictionary Tab
- **No Workbook:** `if not wb or not sheet: return` (line 32)
- **No DataFrame:** `if df is None: return` (line 38)
- **Invalid Headers:** `if not validate_headers(df): return` (line 42)

#### Visualizer Tab
- **No Workbook:** `if not wb or not sheet: return` (line 32)
- **No DataFrame:** `if df is None: return` (line 38)
- **Invalid Headers:** `if not validate_headers(df): return` (line 42)

#### Push Log Tab
- **No Workbook:** `if not wb or not sheet: return` (line 32)
- **No DataFrame:** `if df is None: return` (line 38)
- **Invalid Headers:** `if not validate_headers(df): return` (line 42)

### Error Handling Analysis

#### Try/Except Patterns
- **Source Tab:** Basic try/except around main render
- **Workspace Tab:** Basic try/except around main render
- **Validation Tab:** Basic try/except around main render
- **Conflicts Tab:** Basic try/except around main render
- **Symptoms Tab:** **IMPROVED** - Try/except around streamlined editor + debug guards
- **Outcomes Tab:** **IMPROVED** - Try/except around main render + debug guards
- **Calculator Tab:** **IMPROVED** - Try/except around main render + debug guards
- **Dictionary Tab:** Basic try/except around main render
- **Visualizer Tab:** Basic try/except around main render
- **Push Log Tab:** Basic try/except around main render

#### Silent Failures
- **Most tabs** return early without explaining why
- **Symptoms, Outcomes, Calculator** now show visible warnings
- **Source tab** has some error messages but could be more descriptive

### CSS/HTML Hiding Risks

#### Symptoms Tab
- **TEMPORARILY DISABLED** - CSS wrapper was commented out to debug blank tabs
- **Previously had:** Extensive CSS hiding rules for `.symp` class
- **Risk:** Could hide content if re-enabled

#### Other Tabs
- **No CSS hiding detected** - All other tabs use standard Streamlit components

### Session Key Mismatches

#### Potential Issues
- **`current_sheet` vs `sheet_name`** - Both are kept in sync but could drift
- **`workbook` vs `gs_workbook` vs `upload_workbook`** - Legacy keys could contain stale data
- **`wb_nonce`** - Cache invalidation depends on this being bumped

#### Conflict Resolution State
- **After conflicts resolved:** `st.session_state["current_tab"] = "symptoms"`
- **Potential issue:** This sets a navigation hint but doesn't guarantee tab content will render
- **Root cause:** The `current_tab` key is not used by the main tab router

## C) Caching & Invalidation

### Cached Functions

#### Main App (`streamlit_app_upload.py`)
- `compute_header_badge()` - TTL 600s, depends on `df` and `nonce`
- `get_cached_branch_options_for_ui()` - TTL 600s, depends on `df`, `sheet_name`, and `nonce`

#### Conflicts Tab (`ui/tabs/conflicts.py`)
- `get_conflict_summary()` - TTL 600s, depends on `df` and `nonce`
- `_get_cached_branch_options()` - TTL 600s, depends on `df` and `nonce`
- `_get_cached_validation_report()` - TTL 600s, depends on `df` and `nonce`

#### Symptoms Tab (`ui/tabs/symptoms.py`)
- `_get_cached_branch_options()` - TTL 600s, depends on `df` and `nonce`
- `_get_cached_validation_report()` - TTL 600s, depends on `df` and `nonce`

#### Calculator Tab (`ui/tabs/calculator.py`)
- `_get_cached_branch_options()` - TTL 600s, depends on `df` and `nonce`

#### Workspace Tab (`ui/tabs/workspace.py`)
- `_get_cached_branch_options()` - TTL 600s, depends on `df` and `nonce`

#### Analysis (`ui/analysis.py`)
- `get_conflict_summary_with_root()` - TTL 600s, depends on `df` and `nonce`

### Cache Key Dependencies

#### Workbook Nonce
- **All cached functions** depend on `wb_nonce` for invalidation
- **Nonce is bumped** when `set_active_workbook()` is called
- **Potential issue:** If nonce isn't bumped, caches could remain stale

#### DataFrame Shape
- **Some functions** depend on `df.shape` for cache keys
- **Good practice:** Ensures cache invalidation when data structure changes

#### Sheet Name
- **Most functions** depend on `sheet_name` for cache keys
- **Good practice:** Ensures cache invalidation when switching sheets

## D) What to Verify at Runtime

### Critical State Keys
1. **`workbook`** - Should be a dict with sheet names as keys
2. **`current_sheet`** - Should be a string that exists in workbook keys
3. **`sheet_name`** - Should match current_sheet
4. **`wb_nonce`** - Should change when workbook is updated

### Expected Values
- **`workbook`**: `{"Sheet1": DataFrame, "Sheet2": DataFrame}`
- **`current_sheet`**: `"Sheet1"` (or any key from workbook)
- **`sheet_name`**: Same as current_sheet
- **`wb_nonce`**: String that changes after workbook updates

### Debug Checks
1. **Check workbook keys:** `list(st.session_state.get("workbook", {}).keys())`
2. **Check current sheet:** `st.session_state.get("current_sheet")`
3. **Check sheet name:** `st.session_state.get("sheet_name")`
4. **Check wb_nonce:** `st.session_state.get("wb_nonce")`
5. **Check active df:** `get_active_df()` should return a DataFrame

### Common Failure Modes
1. **`workbook` is empty dict** - No data loaded
2. **`current_sheet` is None** - No sheet selected
3. **`current_sheet` not in workbook keys** - Sheet mismatch
4. **`wb_nonce` unchanged** - Cache invalidation failed
5. **`get_active_df()` returns None** - DataFrame access failed

## E) Recommendations

### Immediate Actions
1. **Fix conflict resolution flow** - Ensure `current_tab` navigation works properly
2. **Add DataFrame shape validation** - All tabs should check `df.shape` before proceeding
3. **Surface guard reasons** - Replace silent returns with visible warnings

### Medium Term
1. **Standardize error handling** - Extend debug pattern to all tabs
2. **Validate session state consistency** - Ensure workbook/sheet state is always consistent
3. **Review cache invalidation** - Ensure wb_nonce is always bumped

### Long Term
1. **Unify state management** - Remove legacy workbook keys
2. **Add comprehensive logging** - Track state changes and failures
3. **Implement state recovery** - Auto-heal inconsistent state

---

*Report generated from analysis of `ui/tabs/*.py`, `utils/state.py`, and `streamlit_app_upload.py`*
