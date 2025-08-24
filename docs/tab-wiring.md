# Tab Wiring & Guard Analysis

## Executive Summary

This document provides a comprehensive analysis of how tabs are wired in the decision tree application, including their registration, guard conditions, and potential failure modes.

## A) Tab Registration & Wiring

### Main App Tab Router (`streamlit_app_upload.py`)

#### Tab Registry Definition
```python
# Lines 570-580
TAB_REGISTRY = [
    ("📂 Source", source.render),
    ("🗂 Workspace Selection", workspace.render),
    ("🔎 Validation", validation.render),
    ("⚖️ Conflicts", conflicts.render),
    ("🧬 Symptoms", symptoms.render),
    ("📝 Outcomes", outcomes.render),
    ("📖 Dictionary", dictionary.render),
    ("🧮 Calculator", calculator.render),
    ("🌐 Visualizer", visualizer.render),
    ("📜 Push Log", push_log.render),
]
```

#### Tab Creation & Rendering
```python
# Lines 583-590
tab_names = [t[0] for t in TAB_REGISTRY]
tabs = st.tabs(tab_names)

# Render each tab with render_guard for crash protection
for i, (tab_name, fn) in enumerate(TAB_REGISTRY):
    with tabs[i]:
        # Extract the actual tab name without emoji
        clean_tab_name = tab_name.split(" ", 1)[1] if " " in tab_name else tab_name
        
        # Use render_guard for all tabs to prevent blank panes
        render_guard(clean_tab_name, fn)
```

#### Render Guard Implementation
```python
# ui/utils/debug.py
def render_guard(label: str, fn):
    """Run a tab render function with visible error reporting (no blank tabs)."""
    import traceback as _tb
    banner(f"DISPATCH {label}")
    try:
        return fn()
    except Exception as e:
        st.error(f"Exception in {label}.render(): {type(e).__name__}: {e}")
        st.code(_tb.format_exc())
        return None
```

### Conditional Tab Rendering

#### Workbook Check Gate
```python
# Lines 500-540
if wb_status != "ok":
    # No workbook loaded - show clear message
    st.info("📂 **No workbook loaded yet**")
    # ... instructions ...
    
    # Only show Source tab when no workbook
    st.info("🚦 DISPATCH Source/Workbook loader")
    source.render()
    return

# DEV bypass: if force_symptoms is checked, render Symptoms directly and stop
if st.session_state.get("DEV_FORCE_SYMPTOMS"):
    st.info("DEV: Forcing Symptoms.render()")
    from ui.tabs import symptoms as T_SYMPT
    T_SYMPT.render()
    st.stop()  # prevent the rest of the app from double-rendering

# Render all tabs normally (workbook is available)
_render_all_tabs()
```

#### Dev Mode Bypasses
```python
# Lines 280-290
# DEV bypass to prove the Symptoms render path
with st.sidebar:
    st.subheader("🔧 Dev")
    force_symptoms = st.checkbox("Force Symptoms render (dev)", key="DEV_FORCE_SYMPTOMS", help="Bypass nav and render Symptoms now")
    
    # Temporary bypass to restore all tabs
    force_all_tabs = st.checkbox("Force All Tabs (bypass workbook check)", key="DEV_FORCE_ALL_TABS", help="Temporarily bypass workbook check to restore all tabs")

# Lines 500-510
# Check if user wants to bypass workbook check
if st.session_state.get("DEV_FORCE_ALL_TABS"):
    st.info("🚀 **DEV MODE: Bypassing workbook check - All tabs available**")
    st.warning("⚠️ This bypasses normal workbook validation. Use only for debugging.")
    
    # Render all tabs normally
    _render_all_tabs()
    return
```

## B) Tab-by-Tab Analysis

### 📂 Source Tab

#### Module & Entry Point
- **Module:** `ui.tabs.source`
- **Render Entry:** `source.render`
- **Purpose:** Data loading and workbook management

#### Guard Conditions
- **No guards for rendering** - Always shows content
- **Early returns only on validation failures**

#### Early Return Paths
```python
# Line 78
if not clean_wb:
    st.error("Uploaded workbook did not contain any valid sheets (as DataFrames).")
    return
```

#### State Dependencies
- **Reads:** None (always renders)
- **Writes:** `workbook`, `current_sheet`, `sheet_name`, `wb_nonce`, `upload_filename`

#### Failure Modes
- **None** - Tab always renders content
- **Validation failures** show error messages but don't leave tab blank

---

### 🗂 Workspace Selection Tab

#### Module & Entry Point
- **Module:** `ui.tabs.workspace`
- **Render Entry:** `workspace.render`
- **Purpose:** Sheet selection and preview

#### Guard Conditions
```python
# Lines 129-135
wb = get_active_workbook()
sheet = get_current_sheet()
if not wb or not sheet:
    st.warning("No active workbook/sheet. Load a workbook in 📂 Source or select a sheet in 🗂 Workspace.")
    return

df = get_active_df()
if df is None:
    st.warning("No active sheet selected. Please load a workbook in the Source tab and select a sheet.")
    return
```

#### Early Return Paths
- **No Workbook:** `if not wb or not sheet: return` (line 129)
- **No DataFrame:** `if df is None: return` (line 135)

#### State Dependencies
- **Reads:** `sheet_id`, `sheet_name`, `_nav_hint`, `editing_parent`, `branch_overrides`
- **Writes:** `_nav_hint`, `editing_parent`, `branch_overrides`, `quick_append_child_input`

#### Failure Modes
- **Blank tab** if no workbook or sheet
- **No DataFrame validation** - could proceed with invalid data

---

### 🔎 Validation Tab

#### Module & Entry Point
- **Module:** `ui.tabs.validation`
- **Render Entry:** `validation.render`
- **Purpose:** Data integrity validation

#### Guard Conditions
```python
# Lines 32-42
wb = get_active_workbook()
sheet = get_current_sheet()
if not wb or not sheet:
    st.warning("No active workbook/sheet. Load a workbook in 📂 Source or select a sheet in 🗂 Workspace.")
    return

df = get_active_df()
if df is None:
    st.warning("No active sheet selected. Please load a workbook in the Source tab and select a sheet.")
    return

if not validate_headers(df):
    st.warning("Active sheet has invalid headers. Please ensure it has the required columns.")
    return
```

#### Early Return Paths
- **No Workbook:** `if not wb or not sheet: return` (line 32)
- **No DataFrame:** `if df is None: return` (line 38)
- **Invalid Headers:** `if not validate_headers(df): return` (line 42)

#### State Dependencies
- **Reads:** `branch_overrides`, `symptom_quality`
- **Writes:** None

#### Failure Modes
- **Blank tab** if any guard condition fails
- **No DataFrame shape validation**

---

### ⚖️ Conflicts Tab

#### Module & Entry Point
- **Module:** `ui.tabs.conflicts`
- **Render Entry:** `conflicts.render`
- **Purpose:** Conflict detection and resolution

#### Guard Conditions
```python
# Lines 67-75
wb = get_active_workbook()
sheet = get_current_sheet()
if not wb or not sheet:
    st.warning("No active workbook/sheet. Load a workbook in 📂 Source or select a sheet in 🗂 Workspace.")
    return

df = get_active_df()
if df is None:
    st.warning("No active sheet selected. Please load a workbook in the Source tab and select a sheet.")
    return

if not validate_headers(df):
    st.warning("Active sheet has invalid headers. Please ensure it has the required columns.")
    return
```

#### Early Return Paths
- **No Workbook:** `if not wb or not sheet: return` (line 67)
- **No DataFrame:** `if df is None: return` (line 71)
- **Invalid Headers:** `if not validate_headers(df): return` (line 75)

#### State Dependencies
- **Reads:** `branch_overrides`, `conflict_idx`, `current_tab`
- **Writes:** `conflict_idx`, `current_tab`, `branch_overrides`

#### Failure Modes
- **Blank tab** if any guard condition fails
- **No DataFrame shape validation**
- **Conflict resolution sets `current_tab = "symptoms"` but this key is not used by the main router**

---

### 🧬 Symptoms Tab

#### Module & Entry Point
- **Module:** `ui.tabs.symptoms`
- **Render Entry:** `symptoms.render`
- **Purpose:** Symptom quality management

#### Guard Conditions
```python
# Lines 172-189
wb = USTATE.get_active_workbook()
sheet = USTATE.get_current_sheet()
st.caption(f"🔎 [Symptoms] start — current_sheet={sheet!r}  sheet_name={st.session_state.get('sheet_name')!r}")

if not wb:
    st.warning("Symptoms: No active workbook in memory (wb is falsy). Load one in 📂 Source.")
    return
if not sheet:
    st.warning("Symptoms: No active sheet selected (current_sheet is falsy). Choose a sheet in 🗂 Workspace or Source.")
    return

df, status, detail = USTATE.get_active_df_safe()
if status != "ok":
    # ... handle various status codes ...
    return

if df is None:
    st.warning("Symptoms: Active DataFrame is None. (Did the upload complete and was a sheet selected?)")
    dump_state("Session (df is None)")
    return
else:
    st.caption(f"Symptoms: df shape = {getattr(df, 'shape', None)}")
```

#### Early Return Paths
- **No Workbook:** `if not wb: return` (line 172)
- **No Sheet:** `if not sheet: return` (line 175)
- **Invalid Status:** `if status != "ok": return` (line 177)
- **No DataFrame:** `if df is None: return` (line 189)

#### State Dependencies
- **Reads:** `current_sheet`, `sheet_name`, `__red_flags_map`, `branch_overrides`, `sym_simple_parent_index`
- **Writes:** `__red_flags_map`, `__symptom_prevalence`, `undo_stack`, `redo_stack`, `branch_overrides`

#### Failure Modes
- **Blank tab** if any guard condition fails
- **IMPROVED:** Now shows visible warnings and dumps state for debugging
- **CSS wrapper temporarily disabled** - could hide content if re-enabled

---

### 📝 Outcomes Tab

#### Module & Entry Point
- **Module:** `ui.tabs.outcomes`
- **Render Entry:** `outcomes.render`
- **Purpose:** Diagnostic triage and actions

#### Guard Conditions
```python
# Lines 62-79
wb = USTATE.get_active_workbook()
sheet = USTATE.get_current_sheet()
st.caption(f"🔎 [Outcomes] start — current_sheet={sheet!r}  sheet_name={st.session_state.get('sheet_name')!r}")

if not wb:
    st.warning("Outcomes: No active workbook in memory (wb is falsy). Load one in 📂 Source.")
    return
if not sheet:
    st.warning("Outcomes: No active sheet selected (current_sheet is falsy). Choose a sheet in 🗂 Workspace or Source.")
    return

df, status, detail = USTATE.get_active_df_safe()
if status != "ok":
    # ... handle various status codes ...
    return

if df is None:
    st.warning("Outcomes: Active DataFrame is None. (Did the upload complete and was a sheet selected?)")
    dump_state("Session (df is None)")
    return
else:
    st.caption(f"Outcomes: df shape = {getattr(df, 'shape', None)}")
```

#### Early Return Paths
- **No Workbook:** `if not wb: return` (line 62)
- **No Sheet:** `if not sheet: return` (line 65)
- **Invalid Status:** `if status != "ok": return` (line 69)
- **No DataFrame:** `if df is None: return` (line 79)

#### State Dependencies
- **Reads:** `work_context`, `gs_workbook`, `upload_workbook`, `sheet_name`, `term_dictionary`
- **Writes:** `upload_workbook`, `gs_workbook`, `out_cur_idx_*`, `term_dictionary`

#### Failure Modes
- **Blank tab** if any guard condition fails
- **IMPROVED:** Now shows visible warnings and dumps state for debugging
- **Legacy data source fallback logic** could be complex

---

### 🧮 Calculator Tab

#### Module & Entry Point
- **Module:** `ui.tabs.calculator`
- **Render Entry:** `calculator.render`
- **Purpose:** Path navigation

#### Guard Conditions
```python
# Lines 32-49
wb = USTATE.get_active_workbook()
sheet = USTATE.get_current_sheet()
st.caption(f"🔎 [Calculator] start — current_sheet={sheet!r}  sheet_name={st.session_state.get('sheet_name')!r}")

if not wb:
    st.warning("Calculator: No active workbook in memory (wb is falsy). Load one in 📂 Source.")
    return
if not sheet:
    st.warning("Calculator: No active sheet selected (current_sheet is falsy). Choose a sheet in 🗂 Workspace or Source.")
    return

df, status, detail = USTATE.get_active_df_safe()
if status != "ok":
    # ... handle various status codes ...
    return

if df is None:
    st.warning("Calculator: Active DataFrame is None. (Did the upload complete and was a sheet selected?)")
    dump_state("Session (df is None)")
    return
else:
    st.caption(f"Calculator: df shape = {getattr(df, 'shape', None)}")
```

#### Early Return Paths
- **No Workbook:** `if not wb: return` (line 32)
- **No Sheet:** `if not sheet: return` (line 35)
- **Invalid Status:** `if status != "ok": return` (line 39)
- **No DataFrame:** `if df is None: return` (line 49)

#### State Dependencies
- **Reads:** `sheet_name`, `calc_nav_*`, `calc_nav_sheet`, `branch_overrides`
- **Writes:** `calc_nav_*`, `calc_nav_sheet`

#### Failure Modes
- **Blank tab** if any guard condition fails
- **IMPROVED:** Now shows visible warnings and dumps state for debugging
- **Complex path navigation logic** could have edge cases

---

### 📖 Dictionary Tab

#### Module & Entry Point
- **Module:** `ui.tabs.dictionary`
- **Render Entry:** `dictionary.render`
- **Purpose:** Vocabulary management

#### Guard Conditions
```python
# Lines 32-42
wb = get_active_workbook()
sheet = get_current_sheet()
if not wb or not sheet:
    st.warning("No active workbook/sheet. Load a workbook in 📂 Source or select a sheet in 🗂 Workspace.")
    return

df = get_active_df()
if df is None:
    st.warning("No active sheet selected. Please load a workbook in the Source tab and select a sheet.")
    return

if not validate_headers(df):
    st.warning("Active sheet has invalid headers. Please ensure it has the required columns.")
    return
```

#### Early Return Paths
- **No Workbook:** `if not wb or not sheet: return` (line 32)
- **No DataFrame:** `if df is None: return` (line 38)
- **Invalid Headers:** `if not validate_headers(df): return` (line 42)

#### State Dependencies
- **Reads:** `term_dictionary`
- **Writes:** `term_dictionary`

#### Failure Modes
- **Blank tab** if any guard condition fails
- **No DataFrame shape validation**

---

### 🌐 Visualizer Tab

#### Module & Entry Point
- **Module:** `ui.tabs.visualizer`
- **Render Entry:** `visualizer.render`
- **Purpose:** Tree visualization

#### Guard Conditions
```python
# Lines 32-42
wb = get_active_workbook()
sheet = get_current_sheet()
if not wb or not sheet:
    st.warning("No active workbook/sheet. Load a workbook in 📂 Source or select a sheet in 🗂 Workspace.")
    return

df = get_active_df()
if df is None:
    st.warning("No active sheet selected. Please load a workbook in the Source tab and select a sheet.")
    return

if not validate_headers(df):
    st.warning("Active sheet has invalid headers. Please ensure it has the required columns.")
    return
```

#### Early Return Paths
- **No Workbook:** `if not wb or not sheet: return` (line 32)
- **No DataFrame:** `if df is None: return` (line 38)
- **Invalid Headers:** `if not validate_headers(df): return` (line 42)

#### State Dependencies
- **Reads:** None detected
- **Writes:** None detected

#### Failure Modes
- **Blank tab** if any guard condition fails
- **No DataFrame shape validation**

---

### 📜 Push Log Tab

#### Module & Entry Point
- **Module:** `ui/tabs.push_log`
- **Render Entry:** `push_log.render`
- **Purpose:** Data push operations

#### Guard Conditions
```python
# Lines 32-42
wb = get_active_workbook()
sheet = get_current_sheet()
if not wb or not sheet:
    st.warning("No active workbook/sheet. Load a workbook in 📂 Source or select a sheet in 🗂 Workspace.")
    return

df = get_active_df()
if df is None:
    st.warning("No active sheet selected. Please load a workbook in the Source tab and select a sheet.")
    return

if not validate_headers(df):
    st.warning("Active sheet has invalid headers. Please ensure it has the required columns.")
    return
```

#### Early Return Paths
- **No Workbook:** `if not wb or not sheet: return` (line 32)
- **No DataFrame:** `if df is None: return` (line 38)
- **Invalid Headers:** `if not validate_headers(df): return` (line 42)

#### State Dependencies
- **Reads:** `push_log`
- **Writes:** None detected

#### Failure Modes
- **Blank tab** if any guard condition fails
- **No DataFrame shape validation**

---

## C) Guard Pattern Analysis

### Common Guard Patterns

#### Pattern 1: Basic Workbook/Sheet Check
```python
wb = get_active_workbook()
sheet = get_current_sheet()
if not wb or not sheet:
    st.warning("No active workbook/sheet. Load a workbook in 📂 Source or select a sheet in 🗂 Workspace.")
    return
```

**Used by:** Workspace, Validation, Conflicts, Dictionary, Visualizer, Push Log
**Risk:** Medium - Silent return, no DataFrame validation

#### Pattern 2: Enhanced Guard with Status Check
```python
wb = USTATE.get_active_workbook()
sheet = USTATE.get_current_sheet()
if not wb:
    st.warning("Tab: No active workbook in memory (wb is falsy). Load one in 📂 Source.")
    return
if not sheet:
    st.warning("Tab: No active sheet selected (current_sheet is falsy). Choose a sheet in 🗂 Workspace or Source.")
    return

df, status, detail = USTATE.get_active_df_safe()
if status != "ok":
    # ... handle various status codes ...
    return

if df is None:
    st.warning("Tab: Active DataFrame is None. (Did the upload complete and was a sheet selected?)")
    dump_state("Session (df is None)")
    return
else:
    st.caption(f"Tab: df shape = {getattr(df, 'shape', None)}")
```

**Used by:** Symptoms, Outcomes, Calculator
**Risk:** Low - Visible warnings, state dumps, DataFrame validation

#### Pattern 3: Header Validation
```python
if not validate_headers(df):
    st.warning("Active sheet has invalid headers. Please ensure it has the required columns.")
    return
```

**Used by:** Validation, Conflicts, Dictionary, Visualizer, Push Log
**Risk:** Medium - No DataFrame shape validation

### Guard Coverage Analysis

#### Fully Protected Tabs
- **Symptoms** ✅ - Comprehensive guards with visible warnings
- **Outcomes** ✅ - Comprehensive guards with visible warnings  
- **Calculator** ✅ - Comprehensive guards with visible warnings

#### Partially Protected Tabs
- **Validation** ⚠️ - Basic guards, no DataFrame shape validation
- **Conflicts** ⚠️ - Basic guards, no DataFrame shape validation
- **Dictionary** ⚠️ - Basic guards, no DataFrame shape validation
- **Visualizer** ⚠️ - Basic guards, no DataFrame shape validation
- **Push Log** ⚠️ - Basic guards, no DataFrame shape validation

#### Unprotected Tabs
- **Source** ❌ - No guards (always renders)
- **Workspace** ❌ - Basic guards only

## D) Failure Mode Analysis

### Blank Tab Root Causes

#### 1. Missing Workbook
- **Condition:** `workbook` is empty dict or None
- **Affects:** All tabs except Source
- **Detection:** `get_active_workbook()` returns None/empty
- **Recovery:** Load workbook in Source tab

#### 2. Missing Current Sheet
- **Condition:** `current_sheet` is None or not in workbook keys
- **Affects:** All tabs except Source
- **Detection:** `get_current_sheet()` returns None
- **Recovery:** Select sheet in Workspace tab

#### 3. Missing DataFrame
- **Condition:** `get_active_df()` returns None
- **Affects:** All tabs except Source
- **Detection:** DataFrame access fails
- **Recovery:** Check workbook integrity, reload data

#### 4. Invalid Headers
- **Condition:** DataFrame columns don't match CANON_HEADERS
- **Affects:** Validation, Conflicts, Dictionary, Visualizer, Push Log
- **Detection:** `validate_headers(df)` returns False
- **Recovery:** Fix DataFrame structure, reload data

#### 5. Cache Invalidation Failure
- **Condition:** `wb_nonce` not bumped after workbook changes
- **Affects:** All tabs with cached functions
- **Detection:** Cached data appears stale
- **Recovery:** Manually bump nonce or reload workbook

### Conflict Resolution Flow Issues

#### Current Flow
```python
# In conflicts tab after resolution
st.session_state["current_tab"] = "symptoms"
st.rerun()
```

#### Problem
- **`current_tab` key is set** but not used by the main tab router
- **Main router** uses `TAB_REGISTRY` and `render_guard`
- **No navigation logic** to switch to Symptoms tab
- **Result:** User stays on Conflicts tab even after resolution

#### Root Cause
- **Missing navigation logic** in main app
- **`current_tab` key** is legacy/unused
- **Tab switching** not implemented

## E) Recommendations

### Immediate Actions
1. **Fix conflict resolution flow** - Implement proper tab navigation
2. **Extend enhanced guards** to all tabs (Symptoms pattern)
3. **Add DataFrame shape validation** to all tabs

### Medium Term
1. **Implement tab navigation** using `current_tab` key
2. **Standardize guard patterns** across all tabs
3. **Add comprehensive error reporting** to all tabs

### Long Term
1. **Remove unused state keys** like `current_tab`
2. **Implement state recovery** for inconsistent workbook state
3. **Add comprehensive logging** for debugging

---

*Report generated from analysis of `streamlit_app_upload.py`, `ui/tabs/*.py`, and `ui/utils/debug.py`*
