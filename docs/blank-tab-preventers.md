# Blank Tab Prevention Checklist

## Executive Summary

This document provides a prioritized, actionable checklist to prevent blank tabs in the decision tree application. Each item includes specific implementation details and risk mitigation strategies.

## Priority 1: Critical Fixes (Immediate Action Required)

### 1. Fix Conflict Resolution Flow
**Problem:** After conflicts are resolved, `st.session_state["current_tab"] = "symptoms"` is set but the main tab router doesn't use this key.

**Impact:** Users stay on Conflicts tab even after resolution, appearing as if the tab is blank or broken.

**Solution:** Implement proper tab navigation logic in the main app.
```python
# In streamlit_app_upload.py, after tab creation
if st.session_state.get("current_tab") and st.session_state.get("current_tab") != "source":
    # Find the target tab index
    target_tab = st.session_state["current_tab"]
    tab_names = [t[0].split(" ", 1)[1] if " " in t[0] else t[0] for t in TAB_REGISTRY]
    if target_tab in tab_names:
        target_index = tab_names.index(target_tab)
        # Switch to the target tab
        st.session_state["active_tab_index"] = target_index
        # Clear the navigation hint
        del st.session_state["current_tab"]
```

**Risk Level:** HIGH - Direct cause of blank tab appearance
**Effort:** Medium - Requires main app modification

---

### 2. Remove CANON_HEADERS Redefinitions
**Problem:** Multiple tabs redefine `CANON_HEADERS` locally, shadowing imported constants and risking UnboundLocalError.

**Impact:** If local variables are modified, tabs could crash or render blank.

**Solution:** Remove local definitions and use imported constants.
```diff
# ui/tabs/symptoms.py:18
- CANON_HEADERS = ["Vital Measurement", *NODE_COLS, "Diagnostic Triage", "Actions"]
+ # Use imported constant
+ from utils.constants import CANON_HEADERS

# ui/tabs/outcomes.py:127  
- CANON_HEADERS = ["Vital Measurement","Node 1","Node 2","Node 3","Node 4","Node 5","Diagnostic Triage","Actions"]
+ # Use imported constant
+ from utils.constants import CANON_HEADERS
```

**Risk Level:** HIGH - Could cause UnboundLocalError
**Effort:** Low - Simple text replacement

---

### 3. Add DataFrame Shape Validation
**Problem:** Most tabs don't validate DataFrame shape before proceeding, potentially causing crashes on malformed data.

**Impact:** Tabs could crash or render blank if DataFrame structure is unexpected.

**Solution:** Add shape validation to all tabs.
```python
# Add after DataFrame retrieval in all tabs
if df is not None:
    if not hasattr(df, 'shape') or len(df.shape) != 2:
        st.error(f"Invalid DataFrame structure: expected 2D array, got {type(df).__name__}")
        dump_state("Session (invalid df structure)")
        return
    if df.shape[0] == 0:
        st.warning("DataFrame is empty. Please check your data source.")
        return
    st.caption(f"Tab: df shape = {df.shape}")
```

**Risk Level:** HIGH - Missing validation could cause crashes
**Effort:** Medium - Add to 7 tabs

---

## Priority 2: High Impact Improvements (Next Sprint)

### 4. Extend Enhanced Guards to All Tabs
**Problem:** Only 3 tabs (Symptoms, Outcomes, Calculator) have comprehensive error handling and debugging.

**Impact:** Other tabs fail silently, making debugging difficult and potentially causing blank panes.

**Solution:** Apply the Symptoms tab pattern to all other tabs.
```python
# Pattern to implement in all tabs
import utils.state as USTATE
from ui.utils.debug import dump_state, banner

def render():
    banner("Tab RENDER ENTRY")
    dump_state("Session (pre-tab)")
    
    wb = USTATE.get_active_workbook()
    sheet = USTATE.get_current_sheet()
    
    if not wb:
        st.warning("Tab: No active workbook in memory. Load one in 📂 Source.")
        return
    if not sheet:
        st.warning("Tab: No active sheet selected. Choose a sheet in 🗂 Workspace or Source.")
        return
        
    df, status, detail = USTATE.get_active_df_safe()
    if status != "ok":
        st.warning(f"Tab: DataFrame error: {detail}")
        dump_state("Session (df error)")
        return
```

**Risk Level:** MEDIUM - Improves debugging and error handling
**Effort:** High - Modify 7 tabs

---

### 5. Standardize Import Patterns
**Problem:** Mixed import patterns (`from utils.state import X` vs `import utils.state as USTATE`) create inconsistency and UnboundLocalError risk.

**Impact:** Code maintenance burden and potential runtime errors.

**Solution:** Convert all imports to module-qualified pattern.
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

**Risk Level:** MEDIUM - Reduces UnboundLocalError risk
**Effort:** Medium - Update 8 tabs

---

### 6. Surface Guard Reasons Instead of Silent Returns
**Problem:** Most tabs return early without explaining why, leaving users confused about why tabs appear blank.

**Impact:** Poor user experience and difficult debugging.

**Solution:** Replace silent returns with visible warnings.
```python
# Instead of silent return
if not wb:
    return

# Show visible warning
if not wb:
    st.warning("Tab: No active workbook. Please load a workbook in the Source tab.")
    st.info("Current session state:")
    dump_state("Session (no workbook)")
    return
```

**Risk Level:** MEDIUM - Improves user experience
**Effort:** Medium - Update 7 tabs

---

## Priority 3: Medium Impact Improvements (Future Sprint)

### 7. Implement Cache Invalidation Strategy
**Problem:** Cached functions depend on `wb_nonce` but there's no guarantee it's always bumped after workbook changes.

**Impact:** Stale data could cause tabs to appear blank or show incorrect information.

**Solution:** Ensure `wb_nonce` is always bumped and add cache clearing.
```python
# In set_active_workbook()
def set_active_workbook(wb: dict, default_sheet: str | None = None, source: str = "unspecified"):
    st.session_state["workbook"] = wb
    # ... existing logic ...
    
    # Always bump nonce
    old_nonce = st.session_state.get("wb_nonce", "")
    st.session_state["wb_nonce"] = old_nonce + "•"
    
    # Clear relevant caches
    if hasattr(st, 'cache_data'):
        st.cache_data.clear()
```

**Risk Level:** MEDIUM - Prevents stale data issues
**Effort:** Low - Modify single function

---

### 8. Add Session State Consistency Validation
**Problem:** Multiple workbook keys (`workbook`, `gs_workbook`, `upload_workbook`) could become inconsistent.

**Impact:** Tabs might access stale or incorrect data, causing blank panes.

**Solution:** Implement state validation and auto-repair.
```python
# Add to main app initialization
def validate_session_state_consistency():
    workbook = st.session_state.get("workbook", {})
    gs_workbook = st.session_state.get("gs_workbook", {})
    upload_workbook = st.session_state.get("upload_workbook", {})
    
    # Ensure only one workbook key has data
    active_keys = [k for k, v in [("workbook", workbook), ("gs_workbook", gs_workbook), ("upload_workbook", upload_workbook)] if v]
    
    if len(active_keys) > 1:
        st.warning("Multiple workbook keys detected. Consolidating...")
        # Merge and consolidate
        # ... implementation ...
```

**Risk Level:** MEDIUM - Prevents data inconsistency
**Effort:** Medium - Add validation logic

---

### 9. Restore CSS Wrapper with Safety Checks
**Problem:** Symptoms tab CSS wrapper was disabled for debugging, but it's needed for proper styling.

**Impact:** Tab styling is broken, potentially affecting user experience.

**Solution:** Re-enable CSS wrapper with safety checks.
```python
# In _render_streamlined_symptoms_editor
def _render_streamlined_symptoms_editor(df, df_norm, sheet, summary, queue_a, queue_b):
    # Safety check before applying CSS
    if df is None or df.empty:
        st.warning("Cannot render editor: no data available")
        return
        
    # Re-enable CSS wrapper
    st.markdown("""
    <style>
    .symp { /* ... CSS rules ... */ }
    </style>
    """, unsafe_allow_html=True)
    
    # ... rest of editor logic ...
```

**Risk Level:** LOW - Restores proper styling
**Effort:** Low - Modify single function

---

## Priority 4: Long-term Improvements (Future Releases)

### 10. Implement Comprehensive State Recovery
**Problem:** No automatic recovery mechanism for corrupted or inconsistent session state.

**Impact:** Users might need to manually reset the app to recover from errors.

**Solution:** Add state recovery and auto-healing.
```python
# Add to main app
def auto_heal_session_state():
    """Attempt to recover from corrupted session state."""
    try:
        # Check workbook integrity
        wb = st.session_state.get("workbook")
        if wb and not isinstance(wb, dict):
            st.warning("Invalid workbook state detected. Attempting recovery...")
            st.session_state["workbook"] = {}
            
        # Check sheet consistency
        current_sheet = st.session_state.get("current_sheet")
        if current_sheet and wb and current_sheet not in wb:
            st.warning("Current sheet not in workbook. Attempting recovery...")
            # ... recovery logic ...
            
    except Exception as e:
        st.error(f"State recovery failed: {e}")
        # Fall back to reset
        if st.button("Reset Session State"):
            st.session_state.clear()
            st.rerun()
```

**Risk Level:** LOW - Improves user experience
**Effort:** High - Complex implementation

---

## Implementation Checklist

### Phase 1: Critical Fixes (Week 1)
- [ ] Fix conflict resolution flow in main app
- [ ] Remove CANON_HEADERS redefinitions from symptoms.py and outcomes.py
- [ ] Add DataFrame shape validation to all tabs

### Phase 2: High Impact (Week 2-3)
- [ ] Extend enhanced guards to all tabs
- [ ] Standardize import patterns across all tabs
- [ ] Surface guard reasons instead of silent returns

### Phase 3: Medium Impact (Week 4-5)
- [ ] Implement cache invalidation strategy
- [ ] Add session state consistency validation
- [ ] Restore CSS wrapper with safety checks

### Phase 4: Long-term (Future)
- [ ] Implement comprehensive state recovery
- [ ] Add comprehensive logging and monitoring
- [ ] Performance optimization and cleanup

## Risk Mitigation

### Testing Strategy
1. **Unit Tests:** Test each guard condition individually
2. **Integration Tests:** Test complete tab rendering flows
3. **Error Injection:** Intentionally corrupt state to test recovery
4. **User Acceptance:** Test with real data and workflows

### Rollback Plan
1. **Feature Flags:** Implement toggles for new guard logic
2. **Gradual Rollout:** Deploy to subset of users first
3. **Monitoring:** Watch for increased error rates
4. **Quick Revert:** Ability to disable new logic immediately

### Success Metrics
1. **Reduced Blank Tabs:** Measure frequency of blank tab reports
2. **Improved Error Visibility:** Track user understanding of failures
3. **Faster Debugging:** Measure time to identify root causes
4. **User Satisfaction:** Survey feedback on error handling

---

*Checklist generated from comprehensive audit of workbook flow, tab wiring, and duplication analysis*
