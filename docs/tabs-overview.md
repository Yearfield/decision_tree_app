# Decision Tree App - Tabs Overview & Risk Analysis

## Executive Summary

**Total Tabs Found:** 10  
**Report Location:** `docs/tabs-overview.md`  
**Top Risk Discovered:** Multiple tabs use `from utils.state import X` patterns that could lead to UnboundLocalError if local variables shadow imported names.

## 1) Tab Registry (High-Level)

| Index | Displayed Label | Module | Render Entry | Notes |
|-------|----------------|---------|--------------|-------|
| 0 | 📂 Source | `ui.tabs.source` | `source.render` | Data loading & workbook management |
| 1 | 🗂 Workspace Selection | `ui.tabs.workspace` | `workspace.render` | Sheet selection & preview |
| 2 | 🔎 Validation | `ui.tabs.validation` | `validation.render` | Data integrity checks |
| 3 | ⚖️ Conflicts | `ui.tabs.conflicts` | `conflicts.render` | Conflict detection & resolution |
| 4 | 🧬 Symptoms | `ui.tabs.symptoms` | `symptoms.render` | Symptom quality management |
| 5 | 📝 Outcomes | `ui/tabs.outcomes` | `outcomes.render` | Diagnostic triage & actions |
| 6 | 📖 Dictionary | `ui/tabs.dictionary` | `dictionary.render` | Vocabulary management |
| 7 | 🧮 Calculator | `ui.tabs.calculator` | `calculator.render` | Path navigation |
| 8 | 🌐 Visualizer | `ui.tabs.visualizer` | `visualizer.render` | Tree visualization |
| 9 | 📜 Push Log | `ui.tabs.push_log` | `push_log.render` | Data push operations |

## 2) Per-Tab Deep Dive

### 📂 Source — (`ui/tabs/source.py`)

**Purpose:** Primary data loading interface for workbooks (XLSX/CSV) and Google Sheets integration. Handles workbook validation and sheet selection.

**Entry Points:**
- `render()` - Main tab render function

**Functions:**
- `render()` - Main tab render with upload, Google Sheets, and VM builder sections
- `_render_upload_section()` - File upload handling for XLSX/CSV
- `_render_google_sheets_section()` - Google Sheets connection interface
- `_render_vm_builder_section()` - Vital Measurement builder wizard
- `_render_new_sheet_wizard_section()` - New sheet creation wizard

**Imports & External Calls:**
- `utils.state` (multiple functions), `logic.tree.build_raw_plus_v630`, `io_utils.sheets.read_google_sheet`

**Session State:**
- **Reads:** `sheet_name`, `current_sheet`, `workbook`, `gs_workbook`, `upload_workbook`, `sheet_id`, `vm_builder_queue`, `wiz_step`, `wiz_vms`, `wiz_vm_entry`, `wiz_n1_vals`, `wiz_sheet_name`, `branch_overrides`, `work_context`
- **Writes:** `upload_filename`, `workbook`, `gs_workbook`, `sheet_id`, `sheet_name`, `vm_builder_queue`, `vm_builder_new`, `wiz_step`, `wiz_vms`, `wiz_vm_entry`, `wiz_n1_vals`, `wiz_sheet_name`, `branch_overrides`, `work_context`

**UI Components:**
- File uploader, text inputs, buttons, expanders, progress indicators

**Early Returns & Guards:**
- Returns early if no file uploaded
- Returns early if workbook coercion fails

**Error Handling:**
- Try/except wrapper around main render
- Validation of uploaded data

**CSS / Hiding Risks:**
- None detected

**Known Pitfalls / Blank-Tab Risk:**
- **HIGH RISK:** Multiple `from utils.state import X` calls (lines 8, 78, 91, 106, 147, 165, 212, 230, 248, 473, 491) - could cause UnboundLocalError
- Silent returns on validation failures
- No DataFrame shape validation before proceeding

---

### 🗂 Workspace Selection — (`ui/tabs/workspace.py`)

**Purpose:** Sheet selection interface with preview capabilities, parent-child analysis, and branch editing tools.

**Entry Points:**
- `render()` - Main tab render function

**Functions:**
- `render()` - Main tab render with navigation hints and sheet preview
- `_compute_parents_vectorized()` - Vectorized parent analysis computation
- `count_full_paths()` - Count complete decision tree paths
- `_check_and_display_nav_hint()` - Navigation guidance display
- `_render_sheet_preview()` - Sheet content preview
- `_render_parent_editor()` - Parent editing interface
- `_render_branch_editor()` - Branch editing tools
- `_render_quick_append()` - Quick child addition

**Imports & External Calls:**
- `utils.state` (multiple functions), `logic.tree.infer_branch_options`, `utils.constants`, `ui.utils.rerun.safe_rerun`

**Session State:**
- **Reads:** `sheet_id`, `sheet_name`, `_nav_hint`, `editing_parent`, `branch_overrides`, `quick_append_child_input`, `branch_editor_level`, `branch_editor_parent`, `resolve_level`, `resolve_parent_label`
- **Writes:** `_nav_hint`, `editing_parent`, `branch_overrides`, `quick_append_child_input`, `branch_editor_level`, `branch_editor_parent`, `resolve_level`, `resolve_parent_label`

**UI Components:**
- Sheet selector, data preview, parent editor, branch editor, navigation controls

**Early Returns & Guards:**
- Returns early if no active workbook/sheet
- Returns early if DataFrame is None

**Error Handling:**
- Try/except wrapper around main render
- Basic error handling in sub-functions

**CSS / Hiding Risks:**
- None detected

**Known Pitfalls / Blank-Tab Risk:**
- **HIGH RISK:** `from utils.state import X` pattern (line 21) - UnboundLocalError risk
- Silent returns on missing data
- No DataFrame shape validation

---

### 🔎 Validation — (`ui/tabs/validation.py`)

**Purpose:** Data integrity validation including orphan detection, loop detection, and red flag analysis.

**Entry Points:**
- `render()` - Main tab render function

**Functions:**
- `render()` - Main validation interface
- `_run_validation_checks()` - Execute validation pipeline
- `_display_orphan_analysis()` - Show orphan node results
- `_display_loop_analysis()` - Show loop detection results
- `_display_missing_red_flags_analysis()` - Show red flag analysis

**Imports & External Calls:**
- `utils.state` (multiple functions), `streamlit_app_upload.get_cached_validation_summary_for_ui`

**Session State:**
- **Reads:** `branch_overrides`, `symptom_quality`
- **Writes:** None

**UI Components:**
- Validation options checkboxes, run button, metrics display, expandable results

**Early Returns & Guards:**
- Returns early if no active workbook/sheet
- Returns early if DataFrame is None
- Returns early if headers are invalid

**Error Handling:**
- Try/except wrapper around main render
- Basic error handling in validation functions

**CSS / Hiding Risks:**
- None detected

**Known Pitfalls / Blank-Tab Risk:**
- **HIGH RISK:** `from utils.state import X` pattern (line 8) - UnboundLocalError risk
- Silent returns on missing data
- No DataFrame shape validation

---

### ⚖️ Conflicts — (`ui/tabs/conflicts.py`)

**Purpose:** Conflict detection and resolution for decision tree inconsistencies, with simple and advanced modes.

**Entry Points:**
- `render()` - Main tab render function

**Functions:**
- `render()` - Main conflicts interface with mode toggle
- `get_conflict_summary()` - Cached conflict summary computation
- `_render_simple_conflicts_navigator()` - Simple conflict navigation
- `_render_advanced_conflicts()` - Advanced conflict analysis
- `_render_conflict_resolution()` - Conflict resolution interface

**Imports & External Calls:**
- `utils.state` (multiple functions), `logic.tree` (multiple functions), `logic.materialize` (multiple functions), `ui.analysis.get_conflict_summary_with_root`

**Session State:**
- **Reads:** `branch_overrides`, `conflict_idx`, `current_tab`
- **Writes:** `conflict_idx`, `current_tab`, `branch_overrides`

**UI Components:**
- Mode toggle, conflict navigator, resolution interface, metrics display

**Early Returns & Guards:**
- Returns early if no active workbook/sheet
- Returns early if DataFrame is None
- Returns early if headers are invalid

**Error Handling:**
- Try/except wrapper around main render
- Basic error handling in sub-functions

**CSS / Hiding Risks:**
- None detected

**Known Pitfalls / Blank-Tab Risk:**
- **HIGH RISK:** `from utils.state import X` pattern (line 21) - UnboundLocalError risk
- Silent returns on missing data
- No DataFrame shape validation

---

### 🧬 Symptoms — (`ui/tabs/symptoms.py`)

**Purpose:** Symptom quality management and branch building with streamlined editor and queue-based navigation.

**Entry Points:**
- `render()` - Main tab render function

**Functions:**
- `render()` - Main symptoms interface with debugging and guards
- `_render_streamlined_symptoms_editor()` - Main editor interface
- `_render_simple_symptoms_editor()` - Alternative simple editor
- `_render_symptom_quality_editor()` - Symptom quality management
- `_render_branch_editor()` - Branch editing tools
- `_render_red_flags_section()` - Red flag management
- `_render_undo_redo()` - Undo/redo functionality

**Imports & External Calls:**
- `utils.state as USTATE`, `ui.utils.debug`, `logic.tree` (multiple functions), `logic.materialize` (multiple functions)

**Session State:**
- **Reads:** `current_sheet`, `sheet_name`, `__red_flags_map`, `branch_overrides`, `sym_simple_parent_index`, `__symptom_prevalence`, `undo_stack`, `redo_stack`, `sym_show_debug`, `sym_pos_A_*`, `sym_pos_B_*`, `sym_active_queue_*`, `sym_next_queue_b`, `sym_queue_b_pos`, `sym_last_target_key_*`
- **Writes:** `__red_flags_map`, `__symptom_prevalence`, `undo_stack`, `redo_stack`, `sym_simple_parent_index`, `sym_pos_A_*`, `sym_pos_B_*`, `sym_active_queue_*`, `sym_next_queue_b`, `sym_queue_b_pos`, `sym_last_target_key_*`, `branch_overrides`

**UI Components:**
- Streamlined editor, queue navigation, symptom quality editor, branch editor, red flags

**Early Returns & Guards:**
- **IMPROVED:** Now has comprehensive guards with visible warnings
- Returns early if no active workbook/sheet
- Returns early if DataFrame is None
- Returns early if headers are invalid

**Error Handling:**
- **IMPROVED:** Try/except wrapper around streamlined editor
- Debug banners and state dumps
- Visible error reporting

**CSS / Hiding Risks:**
- **TEMPORARILY DISABLED:** CSS wrapper was commented out to debug blank tabs
- Previously had extensive CSS hiding rules for `.symp` class

**Known Pitfalls / Blank-Tab Risk:**
- **MEDIUM RISK:** Some `from utils.state import X` calls in sub-functions (lines 1366, 1727, 1800)
- CSS wrapper temporarily disabled for debugging
- **IMPROVED:** Now has comprehensive error handling and guards

---

### 📝 Outcomes — (`ui/tabs/outcomes.py`)

**Purpose:** Diagnostic triage and actions management with row-by-row editing capabilities.

**Entry Points:**
- `render()` - Main tab render function

**Functions:**
- `render()` - Main outcomes interface with debugging and guards
- `_get_current_df_and_sheet()` - Legacy data source fallback
- `_render_vocabulary_overview()` - Vocabulary statistics
- `_render_term_definitions()` - Term definition management
- `_render_dictionary_management()` - Dictionary editing tools

**Imports & External Calls:**
- `utils.state as USTATE`, `ui.utils.debug`, `logic.tree.normalize_text`, `ui.utils.rerun.safe_rerun`

**Session State:**
- **Reads:** `work_context`, `gs_workbook`, `upload_workbook`, `sheet_name`, `term_dictionary`
- **Writes:** `upload_workbook`, `gs_workbook`, `out_cur_idx_*`, `term_dictionary`

**UI Components:**
- Row editor, vocabulary overview, term definitions, dictionary management

**Early Returns & Guards:**
- **IMPROVED:** Now has comprehensive guards with visible warnings
- Returns early if no active workbook/sheet
- Returns early if DataFrame is None
- Returns early if headers are invalid

**Error Handling:**
- **IMPROVED:** Debug banners and state dumps
- Visible error reporting

**CSS / Hiding Risks:**
- None detected

**Known Pitfalls / Blank-Tab Risk:**
- **MEDIUM RISK:** Legacy data source fallback logic could be complex
- **IMPROVED:** Now has comprehensive error handling and guards

---

### 📖 Dictionary — (`ui/tabs/dictionary.py`)

**Purpose:** Vocabulary management and term definitions for decision tree terminology.

**Entry Points:**
- `render()` - Main tab render function

**Functions:**
- `render()` - Main dictionary interface
- `_render_vocabulary_overview()` - Vocabulary statistics display
- `_render_term_definitions()` - Term definition editing
- `_render_dictionary_management()` - Dictionary management tools
- `_build_sheet_vocabulary()` - Build vocabulary from sheet data
- `_get_vm_terms()` - Extract Vital Measurement terms
- `_get_node_terms()` - Extract Node terms

**Imports & External Calls:**
- `utils.state` (multiple functions), `ui.utils.rerun.safe_rerun`

**Session State:**
- **Reads:** `term_dictionary`
- **Writes:** `term_dictionary`

**UI Components:**
- Vocabulary overview, term editor, dictionary management interface

**Early Returns & Guards:**
- Returns early if no active workbook/sheet
- Returns early if DataFrame is None
- Returns early if headers are invalid

**Error Handling:**
- Try/except wrapper around main render
- Basic error handling

**CSS / Hiding Risks:**
- None detected

**Known Pitfalls / Blank-Tab Risk:**
- **HIGH RISK:** `from utils.state import X` pattern (line 8) - UnboundLocalError risk
- Silent returns on missing data
- No DataFrame shape validation

---

### 🧮 Calculator — (`ui/tabs/calculator.py`)

**Purpose:** Decision tree path navigation and exploration with interactive controls.

**Entry Points:**
- `render()` - Main tab render function

**Functions:**
- `render()` - Main calculator interface with debugging and guards
- `_get_cached_branch_options()` - Cached branch options computation
- `_calc_build_nested_options()` - Build nested options structure
- `children_for()` - Get children for specific level/path
- `_render_path_navigator()` - Path navigation interface
- `_render_path_results()` - Path result display

**Imports & External Calls:**
- `utils.state as USTATE`, `ui.utils.debug`, `logic.tree.infer_branch_options_with_overrides`

**Session State:**
- **Reads:** `sheet_name`, `calc_nav_*`, `calc_nav_sheet`, `branch_overrides`
- **Writes:** `calc_nav_*`, `calc_nav_sheet`

**UI Components:**
- Path navigator, level selectors, path results display

**Early Returns & Guards:**
- **IMPROVED:** Now has comprehensive guards with visible warnings
- Returns early if no active workbook/sheet
- Returns early if DataFrame is None
- Returns early if headers are invalid

**Error Handling:**
- **IMPROVED:** Debug banners and state dumps
- Visible error reporting

**CSS / Hiding Risks:**
- None detected

**Known Pitfalls / Blank-Tab Risk:**
- **MEDIUM RISK:** Complex path navigation logic
- **IMPROVED:** Now has comprehensive error handling and guards

---

### 🌐 Visualizer — (`ui/tabs/visualizer.py`)

**Purpose:** Decision tree visualization with multiple visualization types and filtering options.

**Entry Points:**
- `render()` - Main tab render function

**Functions:**
- `render()` - Main visualizer interface
- `_render_visualization_options()` - Visualization type selection
- `_render_tree_explorer()` - Tree exploration interface

**Imports & External Calls:**
- `utils.state` (multiple functions)

**Session State:**
- **Reads:** None detected
- **Writes:** None detected

**UI Components:**
- Visualization type selector, filters, tree explorer

**Early Returns & Guards:**
- Returns early if no active workbook/sheet
- Returns early if DataFrame is None
- Returns early if headers are invalid

**Error Handling:**
- Try/except wrapper around main render
- Basic error handling

**CSS / Hiding Risks:**
- None detected

**Known Pitfalls / Blank-Tab Risk:**
- **HIGH RISK:** `from utils.state import X` pattern (line 8) - UnboundLocalError risk
- Silent returns on missing data
- No DataFrame shape validation

---

### 📜 Push Log — (`ui/tabs/push_log.py`)

**Purpose:** Data push operation management and history tracking.

**Entry Points:**
- `render()` - Main tab render function

**Functions:**
- `render()` - Main push log interface
- `_render_push_log_overview()` - Push log statistics
- `_render_push_operations()` - Push operation controls
- `_render_push_history()` - Push history display

**Imports & External Calls:**
- `utils.state` (multiple functions), `ui.utils.rerun.safe_rerun`, `io_utils.sheets.make_push_log_entry`

**Session State:**
- **Reads:** `push_log`
- **Writes:** None detected

**UI Components:**
- Push log overview, operation controls, history display

**Early Returns & Guards:**
- Returns early if no active workbook/sheet
- Returns early if DataFrame is None
- Returns early if headers are invalid

**Error Handling:**
- Try/except wrapper around main render
- Basic error handling

**CSS / Hiding Risks:**
- None detected

**Known Pitfalls / Blank-Tab Risk:**
- **HIGH RISK:** `from utils.state import X` pattern (line 9) - UnboundLocalError risk
- Silent returns on missing data
- No DataFrame shape validation

---

### ⚡ Actions — (`ui/tabs/actions.py`)

**Purpose:** Action decisions and red flag management for decision tree outcomes.

**Entry Points:**
- `render()` - Main tab render function

**Functions:**
- `render()` - Main actions interface
- `_render_actions_overview()` - Actions statistics
- `_render_actions_management()` - Actions editing
- `_render_red_flags_section()` - Red flag management

**Imports & External Calls:**
- `utils.state` (multiple functions), `ui.utils.rerun.safe_rerun`

**Session State:**
- **Reads:** None detected
- **Writes:** None detected

**UI Components:**
- Actions overview, management interface, red flags

**Early Returns & Guards:**
- Returns early if no active workbook/sheet
- Returns early if DataFrame is None
- Returns early if headers are invalid

**Error Handling:**
- Try/except wrapper around main render
- Basic error handling

**CSS / Hiding Risks:**
- None detected

**Known Pitfalls / Blank-Tab Risk:**
- **HIGH RISK:** `from utils.state import X` pattern (line 8) - UnboundLocalError risk
- Silent returns on missing data
- No DataFrame shape validation

---

### 🩺 Diagnostic Triage — (`ui/tabs/triage.py`)

**Purpose:** Triage decision management and analysis for decision tree paths.

**Entry Points:**
- `render()` - Main tab render function

**Functions:**
- `render()` - Main triage interface
- `_render_triage_overview()` - Triage statistics
- `_render_triage_management()` - Triage editing
- `_render_triage_analysis()` - Triage analysis

**Imports & External Calls:**
- `utils.state` (multiple functions), `ui.utils.rerun.safe_rerun`

**Session State:**
- **Reads:** None detected
- **Writes:** None detected

**UI Components:**
- Triage overview, management interface, analysis tools

**Early Returns & Guards:**
- Returns early if no active workbook/sheet
- Returns early if DataFrame is None
- Returns early if headers are invalid

**Error Handling:**
- Try/except wrapper around main render
- Basic error handling

**CSS / Hiding Risks:**
- None detected

**Known Pitfalls / Blank-Tab Risk:**
- **HIGH RISK:** `from utils.state import X` pattern (line 8) - UnboundLocalError risk
- Silent returns on missing data
- No DataFrame shape validation

---

## 3) Cross-Tab Notes

### Common Session Keys Used Across Tabs
- **`branch_overrides`** - Used in: symptoms, conflicts, workspace, outcomes, calculator
- **`current_sheet`** - Used in: all tabs
- **`sheet_name`** - Used in: all tabs
- **`workbook`** - Used in: all tabs
- **`gs_workbook`** - Used in: source, outcomes
- **`upload_workbook`** - Used in: source, outcomes

### Common Imports That May Be Risky If Shadowed
- **`get_active_workbook`** - Imported in 8 tabs
- **`get_current_sheet`** - Imported in 8 tabs
- **`get_active_df`** - Imported in 8 tabs
- **`has_active_workbook`** - Imported in 6 tabs
- **`get_workbook_status`** - Imported in 6 tabs

### Global CSS That Might Affect Multiple Tabs
- **Symptoms tab** had extensive CSS hiding rules (now temporarily disabled)
- No other global CSS detected

## 4) Actionable Findings (Checklist)

### High Priority (Immediate Action Required)
1. **Replace star imports** - Convert `from utils.state import X` to `import utils.state as USTATE` in all tabs to prevent UnboundLocalError
2. **Add DataFrame shape validation** - All tabs should validate `df.shape` before proceeding
3. **Surface guard reasons** - Replace silent returns with visible warnings explaining why tabs can't render

### Medium Priority (Next Sprint)
4. **Standardize error handling** - Implement consistent try/except patterns across all tabs
5. **Add debug banners** - Extend the debug pattern from symptoms/outcomes/calculator to all tabs
6. **Validate session state consistency** - Ensure workbook/sheet state is consistent across tabs

### Low Priority (Future Improvements)
7. **Cache invalidation strategy** - Review `@st.cache_*` usage for potential stale state issues
8. **CSS wrapper restoration** - Re-enable symptoms CSS wrapper once blank tab issues are resolved
9. **Session state cleanup** - Implement cleanup for unused session keys
10. **Performance monitoring** - Add timing metrics for heavy operations

### Implementation Notes
- **Symptoms, Outcomes, and Calculator tabs** have already been updated with the new debugging pattern
- **Source tab** has the most complex state management and should be prioritized for import fixes
- **Workspace tab** has the most session state mutations and should be reviewed for consistency
- **All other tabs** follow a similar pattern and can be updated systematically

### Risk Mitigation Status
- ✅ **Symptoms tab** - Fully protected with debug guards and error handling
- ✅ **Outcomes tab** - Fully protected with debug guards and error handling  
- ✅ **Calculator tab** - Fully protected with debug guards and error handling
- ❌ **All other tabs** - Still vulnerable to UnboundLocalError and silent failures
- ⚠️ **CSS wrapper** - Temporarily disabled in symptoms tab for debugging

---

*Report generated from analysis of `ui/tabs/*.py` files and main app tab registration in `streamlit_app_upload.py`*
