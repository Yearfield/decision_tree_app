# ui/tabs/symptoms.py
import streamlit as st
import pandas as pd
import utils.state as USTATE
from typing import Dict, Any, List, Tuple, Set
from ui.utils.debug import dump_state, banner

from utils import (
    CANON_HEADERS, LEVEL_COLS, normalize_text, validate_headers
)
from utils.constants import MAX_CHILDREN_PER_PARENT, ROOT_PARENT_LABEL, MAX_LEVELS, LEVEL_LABELS
from utils.helpers import normalize_child_set, normalize_text
from ui.utils.rerun import safe_rerun
from logic.tree import infer_branch_options, build_label_children_index, infer_branch_options_with_overrides
from logic.materialize import materialize_children_for_label_group, materialize_children_for_single_parent, materialize_children_for_label_across_tree

# Use imported constants from utils.constants
from utils.constants import NODE_COLS, CANON_HEADERS

# Helper functions for level mapping and column access
def clamp_level(L: int) -> int:
    """Return min(5, max(1, int(L)))."""
    return min(5, max(1, int(L)))

def child_col(L: int) -> str:
    """Return the child column for level L."""
    return NODE_COLS[clamp_level(L) - 1]

def parent_cols(L: int) -> str:
    """Return parent columns for level L (empty list for L=1)."""
    return NODE_COLS[:clamp_level(L) - 1]

def sanitize_parent_tuple(L: int, p) -> Tuple[str, ...]:
    """Robust parent-tuple sanitizer against char-split bugs."""
    if p is None:
        return ()
    
    if isinstance(p, str):
        return (_nz(p),)
    
    if isinstance(p, (list, tuple)):
        # Map _nz over items
        items = [_nz(item) for item in p]
        
        # Anti char-split guard: if L-1 == 1 and every item is length-1, join into single string
        L = clamp_level(L)
        if L - 1 == 1 and all(len(str(x)) == 1 for x in items) and len(items) > 1:
            return (''.join(items),)
        
        # Truncate to length clamp_level(L)-1
        max_len = L - 1
        return tuple(items[:max_len])
    
    return ()

def _nz(s) -> str:
    """Normalizer: return "" if s is None else str(s).strip()."""
    return "" if s is None else str(s).strip()

def _nz_strict(s: object) -> str:
    """Robust normalizer: unicode normalize, strip zero-width, collapse whitespace, trim."""
    # 1) to string
    x = "" if s is None else str(s)
    # 2) unicode normalize
    import unicodedata, re
    x = unicodedata.normalize("NFKC", x)
    # 3) strip zero-width
    x = re.sub(r"[\u200B-\u200D\uFEFF]", "", x)
    # 4) collapse internal whitespace
    x = re.sub(r"\s+", " ", x)
    # 5) trim
    return x.strip()

def _index_to_parent_tuples(grp_index) -> set[tuple[str, ...]]:
    """Convert groupby index to set of parent tuples, handling both single and multi-index."""
    if hasattr(grp_index, 'names') and grp_index.names:  # MultiIndex
        return {tuple(_nz_strict(v) for v in vals) for vals in grp_index.tolist()}
    else:  # Single index
        return {(_nz_strict(v),) for v in grp_index.tolist()}


def get_red_flags_map():
    """Get the red flags map from session state."""
    return st.session_state.setdefault("__red_flags_map", {})  # {label:str -> True}


def is_red_flag(label: str) -> bool:
    """Check if a label is flagged as red."""
    return bool(get_red_flags_map().get(normalize_text(label)))


def set_red_flag(label: str, value: bool):
    """Set or unset a red flag for a label."""
    m = get_red_flags_map()
    key = normalize_text(label)
    if value:
        m[key] = True
    else:
        m.pop(key, None)
    st.session_state["__red_flags_map"] = m


def compute_virtual_parents(store: Dict[str, List[str]]) -> Dict[int, Set[Tuple[str, ...]]]:
    """Build virtual parents by level, ensuring all levels 1-5 are visible."""
    parents_by_level: Dict[int, Set[Tuple[str, ...]]] = {i: set() for i in range(1, 6)}
    # L=1 parents = ROOT (empty tuple)
    parents_by_level[1].add(tuple())

    # expand down from overrides/store
    for key, children in (store or {}).items():
        if "|" not in key:
            continue
        lvl_s, path = key.split("|", 1)
        try:
            L = int(lvl_s[1:])
        except:
            continue
        # Use sanitizer to prevent character-splitting
        raw_tuple = tuple([] if path=="<ROOT>" else path.split(">"))
        parent_tuple = sanitize_parent_tuple(L, raw_tuple)
        if 1 <= L <= 5:
            parents_by_level[L].add(parent_tuple)
            # add all prefix parents to ensure visibility
            for k in range(1, min(L,5)+1):
                prefix_tuple = tuple(parent_tuple[:k-1])
                parents_by_level[k].add(prefix_tuple)

    # Also expand one level forward when children exist, so level 5 parents appear when level 4 has children
    for L in range(1,5):
        for p in list(parents_by_level[L]):
            kids = [x for x in (store.get(f"L{L}|{'>'.join(p) if p else '<ROOT>'}", []) or []) if normalize_text(x)]
            for c in kids:
                parents_by_level[L+1].add(p + (c,))

    return parents_by_level





APP_VERSION = "v6.5"

def render():
    """Render the Symptoms tab for managing symptom quality and branch building."""
    
    # Add guard and debug expander
    from ui.utils.guards import ensure_active_workbook_and_sheet
    ok, df = ensure_active_workbook_and_sheet("Symptoms")
    if not ok:
        return
    
    # Debug state expander
    import json
    with st.expander("🛠 Debug: Session State (tab)", expanded=False):
        ss = {k: type(v).__name__ for k,v in st.session_state.items()}
        st.code(json.dumps(ss, indent=2))
    
    banner("Symptoms RENDER ENTRY")
    dump_state("Session (pre-symptoms)")
    
    try:
        st.header("🧬 Symptoms v6.5")
        
        # Get current sheet name for display
        sheet = USTATE.get_current_sheet()
        
        # Always-on mini debug banner (never returns early)
        try:
            wb, wb_status, wb_detail = USTATE.get_active_workbook_safe()
            wb_keys = list((wb or {}).keys())
        except Exception:
            wb_keys = []
        st.caption(
            f"🚦 Symptoms ENTRY | current_sheet={st.session_state.get('current_sheet')} "
            f"| sheet_name={st.session_state.get('sheet_name')} | wb_keys={wb_keys[:5]}"
        )
        
        # === Use SAFE getters so we know *why* it might be empty ===
        df, status, detail = USTATE.get_active_df_safe()
        
        # Get workbook and sheet info for guards
        wb = USTATE.get_active_workbook()
        st.caption(f"🔎 [Symptoms] start — current_sheet={sheet!r}  sheet_name={st.session_state.get('sheet_name')!r}")

        if not wb:
            st.warning("Symptoms: No active workbook in memory (wb is falsy). Load one in 📂 Source.")
            return
        if not sheet:
            st.warning("Symptoms: No active sheet selected (current_sheet is falsy). Choose a sheet in 🗂 Workspace or Source.")
            return
        
        if status != "ok":
            # Show a more helpful message when no workbook
            if status == "no_wb":
                st.info("📂 **No workbook loaded yet**")
                st.markdown("""
                **To get started:**
                1. Go to the **Source tab** (first tab)
                2. **Upload a workbook** or **connect to Google Sheets**
                3. **Select a sheet** to work with
                
                Once a workbook is loaded, this tab will show your decision tree data.
                """)
                
                # Show current status
                with st.expander("🔍 Current Status", expanded=True):
                    st.write({
                        "workbook_status": status,
                        "detail": detail,
                        "session_keys": [k for k in st.session_state.keys() if "workbook" in k.lower() or "sheet" in k.lower()]
                    })
                
                # Don't show the "Try ensure_active_sheet" button when no workbook
                return
            else:
                st.warning(f"Symptoms not ready: {status} — {detail}")
                colA, colB = st.columns(2)
                with colA:
                    if st.button("🔁 Try ensure_active_sheet()"):
                        picked = USTATE.ensure_active_sheet(default=st.session_state.get("sheet_name"))
                        st.toast(f"Active sheet → {picked}")
                        st.rerun()
                with colB:
                    st.caption("If this persists, check upload step sets `workbook` or `gs_workbook` as a dict of DataFrames.")
                return

        # We have a DataFrame and a sheet name
        st.caption(f"✅ Using sheet: {sheet} | rows={len(df)} | cols={list(df.columns)[:8]}")

        # Guard against None DataFrame
        if df is None:
            st.warning("Symptoms: Active DataFrame is None. (Did the upload complete and was a sheet selected?)")
            dump_state("Session (df is None)")
            return
        else:
            st.caption(f"Symptoms: df shape = {getattr(df, 'shape', None)}")

        # Validate headers early, but tell users *which* are missing
        required = ["Vital Measurement", "Node 1", "Node 2", "Node 3", "Node 4", "Node 5", "Diagnostic Triage", "Actions"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            st.error(f"Missing required columns: {missing}")
            return

        # Build the parent→children store for editing (but not for counting)
        overrides_all: Dict[str, Dict[str, List[str]]] = st.session_state.get("branch_overrides", {}) or {}
        overrides_sheet: Dict[str, List[str]] = overrides_all.get(sheet, {})
        store: Dict[str, List[str]] = infer_branch_options_with_overrides(df, overrides_sheet)

        # Build DataFrame-based queues for accurate counting
        # Ensure required columns exist
        for col in CANON_HEADERS:
            if col not in df.columns:
                df[col] = ""
        
        # Create normalized view: apply robust normalizer to prevent hidden unicode issues
        df_norm = df.copy()
        for col in ["Vital Measurement"] + NODE_COLS:
            df_norm[col] = df_norm[col].map(_nz_strict)
            # Treat empty strings as empty
            df_norm[col] = df_norm[col].replace("", "").fillna("")
        
        # Build queues from the DataFrame (not the store)
        queue_a = []  # No children
        queue_b = []  # ≤5 children (changed from <5 children)
        

        
        for L in range(1, 6):  # 1..5 inclusive
            pcols = parent_cols(L)
            ccol = child_col(L)
            

            
            # Parent set
            if L == 1:
                # Parent set is {()} if the sheet has any rows (or any non-empty Node 1)
                # This ensures ROOT is considered even with zero children
                parent_set = {()} if not df_norm.empty else set()
            else:
                # Parent set is all unique tuples of pcols where every pcols value is non-empty
                mask = (df_norm[pcols] != "").all(axis=1)
                if mask.any():
                    # Make a mask (df_norm[pcols] != "").all(axis=1)
                    # parents = set( tuple(row[c] for c in pcols) for _, row in df_norm[mask][pcols].drop_duplicates().iterrows() )
                    parent_rows = df_norm[mask][pcols].drop_duplicates()
                    # Use strict normalizer to ensure consistent tuple construction
                    parent_set = set()
                    for _, row in parent_rows.iterrows():
                        # Build tuple with strict normalization
                        normalized_tuple = tuple(_nz_strict(row[c]) for c in pcols)
                        parent_set.add(normalized_tuple)
                else:
                    parent_set = set()
            
            if not parent_set:
                continue
            

            
            # Child count per parent
            scope_nonempty = df_norm[df_norm[ccol] != ""]
            
            if L == 1:
                # Treat ROOT specially by counting distinct non-empty Node 1 overall (this yields 0..N)
                n = scope_nonempty[ccol].nunique()
                

                
                # Queue A (no children): n == 0 → include ()
                if n == 0:
                    queue_a.append((L, ()))
                
                # Queue B (≤5 children): if 0 < n ≤ 5 → include ()  # Updated condition
                if 0 < n <= 5:
                    queue_b.append((L, ()))
            else:
                # Group by parent columns and count children
                grp = scope_nonempty.groupby(pcols, dropna=False)[ccol].nunique()
                

                
                # Convert groupby index to normalized tuples for proper comparison
                keys = grp.index
                if pcols and len(pcols) > 1:
                    grp_keys = {tuple(_nz_strict(v) for v in t) for t in keys.tolist()}  # MultiIndex
                else:
                    grp_keys = {(_nz_strict(v),) for v in keys.tolist()}                  # single index
                

                
                # Queue A (no children): parents in parent set not in grp.index
                zero_children = sorted(parent_set - grp_keys)  # both sets are tuples
                for parent_tuple in zero_children:
                    queue_a.append((L, parent_tuple))
                

                
                # Queue B (≤5 children): iterate grp.items() carefully mapped to the tuple keys
                for k, child_count in grp.items():
                    if 0 < child_count <= 5:  # Changed from < 5 to <= 5
                        # Convert the key to a proper tuple using strict normalizer
                        if len(pcols) == 1:
                            parent_tuple = (_nz_strict(k),)
                        else:
                            parent_tuple = tuple(_nz_strict(val) for val in k)
                        queue_b.append((L, parent_tuple))
                

        
        # Sort queues by (L, parent_tuple) for stability
        queue_a.sort(key=lambda x: (x[0], x[1]))
        queue_b.sort(key=lambda x: (x[0], x[1]))
        

        
        # Build summary for editing (using store for pre-filling, not for counting)
        summary: Dict[Tuple[int, Tuple[str, ...]], Dict[str, object]] = {}
        for key, children in (store or {}).items():
            if "|" not in key:
                continue
            lvl_s, path = key.split("|", 1)
            try:
                L = int(lvl_s[1:])
                L = clamp_level(L)  # Ensure L is in range 1..5
            except Exception:
                continue
            parent_tuple = tuple([] if path == "<ROOT>" else path.split(">"))
            # Sanitize parent tuple to prevent Node 6 errors
            parent_tuple = sanitize_parent_tuple(L, parent_tuple)
            kids = [c for c in (children or []) if _nz(c)]
            summary[(L, parent_tuple)] = {"children": kids, "count": len(kids)}

        # Render the streamlined symptoms editor with DataFrame-based queues
        banner("Symptoms about to render streamlined editor")
        
        try:
            _render_streamlined_symptoms_editor(df, df_norm, sheet, summary, queue_a, queue_b)
            banner("Symptoms streamlined editor completed OK")
        except Exception as e:
            st.error(f"Symptoms: streamlined editor crashed: {type(e).__name__}: {e}")
            import traceback as _tb
            st.code(_tb.format_exc())

    except Exception as e:
        st.error(f"Exception in Symptoms.render(): {e}")
        st.exception(e)


def _render_simple_symptoms_editor(df: pd.DataFrame, sheet_name: str, summary: Dict[Tuple[int, Tuple[str, ...]], Dict[str, object]]):
    """Render the Simple mode: parent-first editor for any parent path."""
    st.subheader("✏️ Parent-First Editor")
    
    # Get branch options for parent information
    overrides_all = st.session_state.get("branch_overrides", {})
    overrides_sheet = overrides_all.get(sheet_name, {})
    store = infer_branch_options_with_overrides(df, overrides_sheet)
    
    # Build virtual parents by level (ensuring all levels 1-5 are visible)
    virtual_parents = compute_virtual_parents(store)
    
    if not any(virtual_parents.values()):
        st.info("No parents found")
        return
    
    # Level selector with clear ROOT/Node mapping
    def _level_label(L:int)->str:
        # Parent is ROOT (Vital Measurement) for L=1; children live in Node 1
        # For L=k>1: parent is Node (k-1); children live in Node k
        return "Parent: Vital Measurement → children in Node 1" if L==1 else f"Parent: Node {L-1} → children in Node {L}"

    level = st.selectbox(
        "Which parents do you want to edit (by the column their CHILDREN live in)?",
        [1,2,3,4,5],
        format_func=_level_label,
        key="sym_level"
    )
    
    # Debug expander (optional but helpful)
    with st.expander("🔍 Debug (Level Info)", expanded=False):
        st.write(f"**Selected Level:** {level}")
        st.write(f"**Level Label:** {_level_label(level)}")
        st.write(f"**Virtual parents at this level:** {len(virtual_parents.get(level, []))}")
        if level == 1:
            st.write("**Node 1:** Parent is ROOT (Vital Measurement) → children live in Node 1")
        else:
            st.write(f"**Node {level}:** Parent is Node {level-1} → children live in Node {level}")
        
        # Show (L, parent_tuple) -> child_count for current level
        level_summary = []
        for (L, pth), info in summary.items():
            if L == level:
                child_count = info.get("count", 0)
                parent_display = "ROOT" if not pth else " > ".join(pth)
                level_summary.append(f"({L}, {parent_display}) -> {child_count} children")
        
        if level_summary:
            st.write("**Current level breakdown:**")
            for item in level_summary:
                st.write(f"  • {item}")
    
    # Get parents for selected level
    level_parents = sorted(virtual_parents.get(level, []))
    
    if not level_parents:
        st.info(f"No parents found at level {level}")
        return
    
    # Initialize parent index in session state
    if "sym_simple_parent_index" not in st.session_state:
        st.session_state["sym_simple_parent_index"] = 0
    
    # Section A: Select Parent
    st.markdown("**A) Select Parent**")
    
    # Create searchable dropdown with parent paths
    parent_options = []
    for parent_tuple in level_parents:
        # Get children for this parent
        parent_key = f"L{level}|{'>'.join(parent_tuple) if parent_tuple else '<ROOT>'}"
        children = store.get(parent_key, [])
        
        # Build display text
        if len(parent_tuple) == 0 and level == 1:
            display_text = "Parent: Vital Measurement (root) — children in Node 1"
        else:
            display_text = f"Parent path: {' > '.join(parent_tuple)} — children in Node {level}"
        
        parent_options.append((parent_tuple, children, display_text))
    
    selected_option = st.selectbox(
        "Pick a parent", 
        options=parent_options,
        index=st.session_state["sym_simple_parent_index"] % len(parent_options),
        format_func=lambda x: x[2],  # Show the display text
        key="__symptoms_parent_picker"
    )
    
    if selected_option:
        parent_tuple, children_now, _ = selected_option
        
        # Section B: Edit Children
        st.markdown("**B) Edit Children**")
        st.write(f"**Current children ({len(children_now)}):** {', '.join(children_now) if children_now else 'None'}")
        
        _render_parent_editor_for_symptoms_new(level, parent_tuple, children_now, df, sheet_name, store, summary)
        
        # Section C: Actions
        st.markdown("**C) Actions**")
        
        # Skip to next incomplete button
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Skip ➝ Next incomplete parent", key="__symptoms_skip_next"):
                next_idx = (st.session_state["sym_simple_parent_index"] + 1) % len(parent_options)
                st.session_state["sym_simple_parent_index"] = next_idx
                st.warning("⚠️ Rerun skipped for debugging")
                # safe_rerun()
        
        with col2:
            st.info(f"Parent {st.session_state['sym_simple_parent_index'] + 1} of {len(parent_options)}")


def _render_parent_editor_for_symptoms_new(level: int,
                                           parent_tuple: Tuple[str, ...],
                                           children_now: List[str],
                                           df: pd.DataFrame,
                                           sheet_name: str,
                                           store: Dict[str, List[str]],
                                           summary: Dict[Tuple[int, Tuple[str, ...]], Dict[str, object]]):
    """Render the parent editor for symptoms with the new structure."""
    st.markdown("---")
    st.subheader("✏️ Edit Children")
    
    # Get parent label from path
    if level == 1:
        parent_label = "Vital Measurement (root)"
    else:
        parent_label = " > ".join(parent_tuple)
    
    # Build options from current children and store
    all_children = set(children_now)
    
    # Add children from other variants at this level
    for key, children in store.items():
        if key.startswith(f"L{level}|"):
            all_children.update(children)
    
    union_opts = sorted(all_children)
    default_set = children_now[:MAX_CHILDREN_PER_PARENT]

    # Build stable key seed
    seed = f"symptoms_simple_{level}_{'_'.join(parent_tuple) if parent_tuple else 'root'}"
    
    # Show red flag status for current children
    if children_now:
        flagged_children = [c for c in children_now if is_red_flag(c)]
        if flagged_children:
            st.warning(f"🚩 Red flagged children: {', '.join(flagged_children)}")
    
    chosen = st.multiselect(
        f"Choose up to {MAX_CHILDREN_PER_PARENT} children", 
        options=union_opts, 
        default=default_set, 
        max_selections=MAX_CHILDREN_PER_PARENT,
        key=f"{seed}_ms_children"
    )
    
    new_child = normalize_text(st.text_input("Add new child", key=f"{seed}_ti_add"))
    if new_child and new_child not in chosen:
        chosen = normalize_child_set(chosen + [new_child])
        st.info(f"Preview after add: {', '.join(chosen)}")

    # Red Flags expander
    with st.expander("🚩 Red Flags", expanded=False):
        q = normalize_text(st.text_input("Search symptom/label", key=f"{seed}_rf_search"))
        
        # Build label set from tree summary
        labels = set()
        for (L, pth), info in summary.items():
            if L == 1:
                labels.add(ROOT_PARENT_LABEL)
            else:
                lab = pth.split(">")[-1] if pth else ""
                if lab: 
                    labels.add(lab)
        
        # Add current children to labels
        labels.update(children_now)
        
        # Filter and sort labels
        labels = sorted([x for x in labels if not q or q in normalize_text(x)])
        
        for i, lab in enumerate(labels):
            checked = is_red_flag(lab)
            if st.checkbox(lab, value=checked, key=f"{seed}_rf_{i}"):
                if not checked: 
                    set_red_flag(lab, True)
            else:
                if checked: 
                    set_red_flag(lab, False)

    c1, c2 = st.columns(2)
    
    with c1:
        if st.button("Apply to THIS parent", key=f"{seed}_btn_single"):
            _apply_symptoms_to_single_parent(level, parent_tuple, chosen, df, sheet_name)
    
    with c2:
        if st.button(f"Apply to ALL '{parent_label}' parents across the tree", key=f"{seed}_btn_group"):
            _apply_symptoms_to_label_across_tree(parent_label, chosen, df, sheet_name)


def _apply_symptoms_to_single_parent(level: int, parent_path: str, children: List[str], df: pd.DataFrame, sheet_name: str):
    """Apply children to a single parent path in symptoms."""
    try:
        with st.spinner(f"Applying to single parent at level {level}..."):
            wb = USTATE.get_active_workbook()
            if not wb or sheet_name not in wb:
                st.error("No active sheet.")
                return
            
            df0 = wb[sheet_name]
            
            # Get the parent label from the path
            if level == 1:
                parent_label = ROOT_PARENT_LABEL
            else:
                parent_label = parent_path.split(">")[-1]
            
            # Apply using label-group materializer
            new_df = materialize_children_for_label_group(df0, level, parent_label, children)
            
            # Update workbook
            wb[sheet_name] = new_df
            USTATE.set_active_workbook(wb, default_sheet=sheet_name, source="symptoms_single_parent")
            USTATE.set_current_sheet(sheet_name)
            
            st.success(f"Applied to parent: {parent_path}")
            st.warning("⚠️ Rerun skipped for debugging")
            # safe_rerun()
            
    except Exception as e:
        st.error(f"Error applying to single parent: {e}")
        st.exception(e)


def _apply_symptoms_to_label_group(level: int, parent_label: str, children: List[str], df: pd.DataFrame, sheet_name: str):
    """Apply children to all parents with the same label at the given level in symptoms."""
    try:
        with st.spinner(f"Applying to all '{parent_label}' parents at level {level}..."):
            wb = USTATE.get_active_workbook()
            if not wb or sheet_name not in wb:
                st.error("No active sheet.")
                return
            
            df0 = wb[sheet_name]
            
            # Use the existing label-group materializer
            new_df = materialize_children_for_label_group(df0, level, parent_label, children)
            
            # Update workbook
            wb[sheet_name] = new_df
            USTATE.set_active_workbook(wb, default_sheet=sheet_name, source="symptoms_label_group")
            USTATE.set_current_sheet(sheet_name)
            
            st.success(f"Applied to label-wide group: {parent_label}")
            st.warning("⚠️ Rerun skipped for debugging")
            # safe_rerun()
            
    except Exception as e:
        st.error(f"Error applying to label group: {e}")
        st.exception(e)


def _apply_symptoms_to_label_across_tree(parent_label: str, children: List[str], df: pd.DataFrame, sheet_name: str):
    """Apply children to all parents with the same label across the entire tree in symptoms."""
    try:
        with st.spinner(f"Applying to all '{parent_label}' parents across the tree..."):
            wb = USTATE.get_active_workbook()
            if not wb or sheet_name not in wb:
                st.error("No active sheet.")
                return
            
            df0 = wb[sheet_name]
            
            # Get the summary for the materializer
            res = get_conflict_summary_with_root(df0, USTATE.get_wb_nonce())
            summary = res["summary"]
            
            # Use the across-tree materializer
            new_df = materialize_children_for_label_across_tree(df0, parent_label, children, summary)
            
            # Update workbook
            wb[sheet_name] = new_df
            USTATE.set_active_workbook(wb, default_sheet=sheet_name, source="symptoms_across_tree")
            USTATE.set_current_sheet(sheet_name)
            
            st.success(f"Applied to label-wide group: {parent_label}")
            st.warning("⚠️ Rerun skipped for debugging")
            # safe_rerun()
            
    except Exception as e:
        st.error(f"Error applying across tree: {e}")
        st.exception(e)


def _render_advanced_symptoms(df: pd.DataFrame, symptom_prevalence: Dict, sheet_name: str):
    """Render the Advanced mode: existing symptom quality and branch building functionality."""
    st.subheader("🔬 Advanced Symptoms Analysis")
    
    # Get symptom prevalence data
    symptom_prevalence = symptom_prevalence or {}
    
    # Main sections
    _render_symptom_prevalence_section(df, symptom_prevalence, sheet_name)
    
    st.markdown("---")
    
    _render_branch_building_section(df, symptom_prevalence, sheet_name)
    
    st.markdown("---")
    
    _render_branch_editor_section(df, sheet_name)


def _render_symptom_prevalence_section(df: pd.DataFrame, symptom_prevalence: Dict, sheet_name: str):
    """Render the symptom prevalence management section."""
    st.subheader("🎯 Symptom Prevalence Management")
    
    # Build vocabulary from current sheet
    vocab = _build_sheet_vocabulary(df)
    
    if not vocab:
        st.info("No vocabulary found in the current sheet.")
        return
    
    st.write(f"Found {len(vocab)} unique terms in '{sheet_name}':")
    
    # Display vocabulary with quality scores
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Search/filter vocabulary
        search_term = st.text_input("🔍 Search vocabulary", placeholder="Type to filter...")
        filtered_vocab = [term for term in vocab if search_term.lower() in term.lower()] if search_term else vocab
        
        if filtered_vocab:
            st.write("**Vocabulary terms:**")
            for term in filtered_vocab[:50]:  # Show first 50
                prevalence = symptom_prevalence.get(term, 0)
                st.write(f"• {term} (prevalence: {prevalence})")
            
            if len(filtered_vocab) > 50:
                st.caption(f"... and {len(filtered_vocab) - 50} more terms")
    
    with col2:
        # Prevalence score editor
        st.write("**Set prevalence score:**")
        selected_term = st.selectbox("Select term", [""] + filtered_vocab, key="prevalence_term_selector")
        
        if selected_term:
            current_prevalence = symptom_prevalence.get(selected_term, 0)
            new_prevalence = st.slider(
                "Prevalence score",
                min_value=0,
                max_value=10,
                value=current_prevalence,
                help="0 = rare, 10 = very common"
            )
            
            if new_prevalence != current_prevalence:
                if st.button("Update Prevalence"):
                    symptom_prevalence[selected_term] = new_prevalence
                    st.session_state["__symptom_prevalence"] = symptom_prevalence
                    st.success(f"Updated '{selected_term}' prevalence to {new_prevalence}")
                    st.warning("⚠️ Rerun skipped for debugging")
                    # safe_rerun()
    
    # Bulk prevalence operations
    st.markdown("---")
    st.subheader("📊 Bulk Prevalence Operations")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📈 Auto-assign Prevalence"):
            _auto_assign_prevalence_scores(df, symptom_prevalence)
    
    with col2:
        if st.button("🔄 Reset All Prevalence"):
            if st.checkbox("Confirm reset all prevalence scores"):
                st.session_state["__symptom_prevalence"] = {}
                st.success("All prevalence scores reset!")
                st.warning("⚠️ Rerun skipped for debugging")
                # safe_rerun()
    
    with col3:
        if st.button("💾 Export Prevalence Data"):
            _export_prevalence_data(symptom_prevalence)


def _render_branch_building_section(df: pd.DataFrame, symptom_prevalence: Dict, sheet_name: str):
    """Render the branch building section."""
    st.subheader("🌿 Branch Building")
    
    # Branch building options
    col1, col2 = st.columns(2)
    
    with col1:
        build_mode = st.radio(
            "Build mode",
            ["Manual", "Auto-suggest", "Quality-based"],
            help="Choose how to build branches"
        )
    
    with col2:
        target_level = st.selectbox(
            "Target level",
            options=[1, 2, 3, 4, 5],
            index=0,
            help="Which level to build branches for"
        )
    
    # Branch building interface
    if build_mode == "Manual":
        _render_manual_branch_building(df, target_level, sheet_name)
    elif build_mode == "Auto-suggest":
        _render_auto_suggest_branch_building(df, target_level, symptom_prevalence, sheet_name)
    else:  # Prevalence-based
        _render_prevalence_based_branch_building(df, target_level, symptom_prevalence, sheet_name)


def _build_sheet_vocabulary(df: pd.DataFrame) -> List[str]:
    """Build vocabulary from the current sheet."""
    try:
        vocab = set()
        
        # Collect all non-empty values from node columns
        for col in LEVEL_COLS:
            if col in df.columns:
                values = df[col].map(normalize_text).dropna()
                values = values[values != ""]
                vocab.update(values)
        
        # Also collect from Vital Measurement
        if "Vital Measurement" in df.columns:
            vm_values = df["Vital Measurement"].map(normalize_text).dropna()
            vm_values = vm_values[vm_values != ""]
            vocab.update(vm_values)
        
        return sorted(list(vocab))
    except Exception:
        return []


def _auto_assign_prevalence_scores(df: pd.DataFrame, symptom_prevalence: Dict):
    """Automatically assign quality scores based on data patterns."""
    try:
        with st.spinner("Analyzing data patterns..."):
            # Simple heuristic: terms that appear more frequently get higher scores
            term_counts = {}
            
            for col in LEVEL_COLS + ["Vital Measurement"]:
                if col in df.columns:
                    values = df[col].map(normalize_text).dropna()
                    values = values[values != ""]
                    for term in values:
                        term_counts[term] = term_counts.get(term, 0) + 1
            
            # Assign scores based on frequency
            for term, count in term_counts.items():
                if count == 1:
                    score = 3  # Rare terms get low score
                elif count <= 3:
                    score = 5  # Occasional terms get medium score
                elif count <= 10:
                    score = 7  # Common terms get high score
                else:
                    score = 9  # Very common terms get very high score
                
                symptom_prevalence[term] = score
            
            st.session_state["__symptom_prevalence"] = symptom_prevalence
            st.success(f"Auto-assigned prevalence scores to {len(term_counts)} terms!")
            st.warning("⚠️ Rerun skipped for debugging")
            # safe_rerun()
            
    except Exception as e:
        st.error(f"Error auto-assigning prevalence scores: {e}")


def _export_prevalence_data(symptom_prevalence: Dict):
    """Export prevalence data to a downloadable format."""
    try:
        if not symptom_prevalence:
            st.warning("No prevalence data to export.")
            return
        
        # Create DataFrame
        prevalence_df = pd.DataFrame([
            {"term": term, "prevalence_score": score}
            for term, score in symptom_prevalence.items()
        ])
        
        # Sort by prevalence score
        prevalence_df = prevalence_df.sort_values("prevalence_score", ascending=False)
        
        # Download button
        csv = prevalence_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Prevalence Data (CSV)",
            data=csv,
            file_name="symptom_prevalence.csv",
            mime="text/csv"
        )
        
    except Exception as e:
        st.error(f"Error exporting prevalence data: {e}")


def _render_manual_branch_building(df: pd.DataFrame, target_level: int, sheet_name: str):
    """Render manual branch building interface."""
    st.write("**Manual Branch Building**")
    st.info("Select a parent node and manually specify its children.")
    
    # Get parent options
    parent_col = f"Node {target_level - 1}" if target_level > 1 else "Vital Measurement"
    if parent_col not in df.columns:
        st.warning(f"Parent column '{parent_col}' not found.")
        return
    
    parent_values = df[parent_col].map(normalize_text).dropna().unique()
    parent_values = [v for v in parent_values if v != ""]
    
    if not parent_values:
        st.info(f"No parent values found in '{parent_col}'.")
        return
    
    selected_parent = st.selectbox("Select parent node", parent_values, key=f"manual_parent_{target_level}")
    
    if selected_parent:
        # Input children
        st.write(f"**Children for '{selected_parent}' at level {target_level}:**")
        
        children = []
        for i in range(5):
            child = st.text_input(f"Child {i + 1}", key=f"manual_child_{target_level}_{i}")
            if child.strip():
                children.append(child.strip())
        
        if st.button("Create Branch"):
            if children:
                _create_branch(df, selected_parent, target_level, children, sheet_name)
            else:
                st.warning("Please specify at least one child.")


def _render_auto_suggest_branch_building(df: pd.DataFrame, target_level: int, symptom_prevalence: Dict, sheet_name: str):
    """Render auto-suggest branch building interface."""
    st.write("**Auto-Suggest Branch Building**")
    st.info("Get suggestions for branches based on existing data patterns.")
    
    # Get parent options
    parent_col = f"Node {target_level - 1}" if target_level > 1 else "Vital Measurement"
    if parent_col not in df.columns:
        st.warning(f"Parent column '{parent_col}' not found.")
        return
    
    parent_values = df[parent_col].map(normalize_text).dropna().unique()
    parent_values = [v for v in parent_values if v != ""]
    
    if not parent_values:
        st.info(f"No parent values found in '{parent_col}'.")
        return
    
    selected_parent = st.selectbox("Select parent node", parent_values, key=f"auto_parent_{target_level}")
    
    if selected_parent and st.button("🔍 Get Suggestions"):
        with st.spinner("Analyzing patterns..."):
            suggestions = _get_branch_suggestions(df, selected_parent, target_level, symptom_prevalence)
            
            if suggestions:
                st.write("**Suggested children:**")
                for i, suggestion in enumerate(suggestions):
                    st.write(f"{i + 1}. {suggestion}")
                
                if st.button("Use These Suggestions"):
                    _create_branch(df, selected_parent, target_level, suggestions, sheet_name)
            else:
                st.info("No suggestions found. Try manual input.")


def _render_prevalence_based_branch_building(df: pd.DataFrame, target_level: int, symptom_prevalence: Dict, sheet_name: str):
    """Render quality-based branch building interface."""
    st.write("**Quality-Based Branch Building**")
    st.info("Build branches using high-quality terms from the vocabulary.")
    
    # Get parent options
    parent_col = f"Node {target_level - 1}" if target_level > 1 else "Vital Measurement"
    if parent_col not in df.columns:
        st.warning(f"Parent column '{parent_col}' not found.")
        return
    
    parent_values = df[parent_col].map(normalize_text).dropna().unique()
    parent_values = [v for v in parent_values if v != ""]
    
    if not parent_values:
        st.info(f"No parent values found in '{parent_col}'.")
        return
    
    selected_parent = st.selectbox("Select parent node", parent_values, key=f"quality_parent_{target_level}")
    
    if selected_parent:
        # Show high-prevalence vocabulary
        high_prevalence_terms = [term for term, score in symptom_prevalence.items() if score >= 7]
        
        if high_prevalence_terms:
            st.write("**High-prevalence terms (score ≥ 7):**")
            selected_terms = st.multiselect(
                "Select terms for this branch",
                high_prevalence_terms,
                max_selections=MAX_CHILDREN_PER_PARENT,
                key=f"prevalence_terms_{target_level}"
            )
            
            if selected_terms and st.button("Create Prevalence Branch"):
                _create_branch(df, selected_parent, target_level, selected_terms, sheet_name)
        else:
            st.info("No high-prevalence terms found. Consider improving term prevalence first.")


def _get_branch_suggestions(df: pd.DataFrame, parent: str, target_level: int, symptom_prevalence: Dict) -> List[str]:
    """Get branch suggestions based on data patterns."""
    try:
        # Look for existing patterns
        suggestions = set()
        
        # Check if this parent already has children
        if target_level <= 5:
            target_col = f"Node {target_level}"
            if target_col in df.columns:
                # Find rows where parent matches
                parent_col = f"Node {target_level - 1}" if target_level > 1 else "Vital Measurement"
                mask = df[parent_col].map(normalize_text) == parent
                existing_children = df.loc[mask, target_col].map(normalize_text).dropna()
                existing_children = existing_children[existing_children != ""]
                suggestions.update(existing_children)
        
        # Add high-prevalence vocabulary terms
        high_prevalence_terms = [term for term, score in symptom_prevalence.items() if score >= 6]
        suggestions.update(high_prevalence_terms[:10])  # Top 10
        
        # Convert to list and limit to 5
        result = list(suggestions)[:5]
        return result
        
    except Exception:
        return []


def _create_branch(df: pd.DataFrame, parent: str, target_level: int, children: List[str], sheet_name: str):
    """Create a new branch in the decision tree."""
    try:
        # This would implement the actual branch creation logic
        # For now, just show success message
        st.success(f"Branch created for '{parent}' with {len(children)} children!")
        st.write(f"Children: {', '.join(children)}")
        
        # In a full implementation, this would:
        # 1. Add new rows to the DataFrame
        # 2. Update the workbook in session state
        # 3. Possibly update overrides
        
    except Exception as e:
        st.error(f"Error creating branch: {e}")


def _render_branch_editor_section(df: pd.DataFrame, sheet_name: str):
    """Render the branch editor section for existing parents."""
    st.subheader("✏️ Branch Editor")
    st.markdown("Edit children for existing parents in the decision tree.")
    
    # Guard: require active df
    if df is None or df.empty:
        st.warning("No active DataFrame available.")
        return
    
    # Level selector with clear ROOT/Node mapping
    def _level_label(L:int)->str:
        # Parent is ROOT (Vital Measurement) for L=1; children live in Node 1
        # For L=k>1: parent is Node (k-1); children live in Node k
        return "Parent: Vital Measurement → children in Node 1" if L==1 else f"Parent: Node {L-1} → children in Node {L}"

    level = st.selectbox(
        "Which parents do you want to edit (by the column their CHILDREN live in)?",
        [1,2,3,4,5],
        format_func=_level_label,
        key="branch_editor_level"
    )
    
    # Show hint for Level-1 editing
    if level == 1:
        st.info("💡 **Note:** Level-1 (ROOT) children are managed in the ⚖️ Conflicts tab. Use the Level-1 editor there to set the Node-1 options.")
    
    # Get distinct parent paths at level-1
    parent_paths = _get_parent_paths_at_level(df, level, USTATE.get_wb_nonce())
    
    if not parent_paths:
        st.info(f"No parent paths found at level {level}.")
        return
    
    # Parent picker
    selected_parent_path = st.selectbox(
        "Select parent path",
        parent_paths,
        format_func=lambda x: " > ".join(x) if x else "<ROOT>",
        key="branch_editor_parent"
    )
    
    if not selected_parent_path:
        st.info("Please select a parent path.")
        return
    
    # Show current children for the selected parent
    current_children = _get_current_children_for_parent(df, level, selected_parent_path, USTATE.get_wb_nonce())
    
    # Build clear parent title
    if len(selected_parent_path) == 0 and level == 1:
        parent_title = "Parent: Vital Measurement (root) — children in Node 1"
    else:
        parent_title = f"Parent path: {' > '.join(selected_parent_path)} — children in Node {level}"
    
    st.write(f"**{parent_title}:**")
    if current_children:
        st.write(f"**{len(current_children)} children:** {', '.join(current_children)}")
    else:
        st.write("**No children defined yet.**")
    
    # Multi-select existing children to include
    st.markdown("---")
    st.write("**Select children to include:**")
    
    # Get all possible children at this level from the store
    from logic.tree import infer_branch_options
    store = infer_branch_options(df)
    
    # Find the key for this parent at this level
    parent_key = _build_parent_key(level, selected_parent_path)
    
    # Get existing children from store or current data
    all_possible_children = set()
    if parent_key in store:
        all_possible_children.update(store[parent_key])
    all_possible_children.update(current_children)
    
    if not all_possible_children:
        st.info("No existing children found. You can add new ones below.")
        all_possible_children = set()
    
    # Multi-select existing children (trim to 5)
    selected_existing = st.multiselect(
        "Select from existing children:",
        sorted(all_possible_children),
        max_selections=MAX_CHILDREN_PER_PARENT,
        key="branch_editor_existing"
    )
    
    # Add new child input
    new_child = st.text_input(
        "Add new child (optional):",
        placeholder="Type a new child name...",
        key="branch_editor_new_child"
    )
    
    # Build final children list
    final_children = list(selected_existing)
    
    if new_child.strip():
        new_child_clean = normalize_text(new_child)
        if new_child_clean and new_child_clean not in final_children:
            final_children.append(new_child_clean)
    
    # Use normalize_child_set to cap at MAX_CHILDREN_PER_PARENT
    final_children = normalize_child_set(final_children)
    
    if len(final_children) > MAX_CHILDREN_PER_PARENT:
        st.warning(f"Children list capped to {MAX_CHILDREN_PER_PARENT} (maximum allowed).")
    
    # Show final selection
    if final_children:
        st.write(f"**Final children set ({len(final_children)}):** {', '.join(final_children)}")
        
        # Apply button
        if st.button("Apply Changes", type="primary", key="branch_editor_apply"):
            _apply_branch_editor_changes(df, level, selected_parent_path, final_children, sheet_name)
            
            # Outcomes editor for Diagnostic Triage & Actions
            st.markdown("---")
            st.subheader("Outcomes — Diagnostic Triage & Actions")

            # Scope: filter rows matching this parent's exact path-slot (level, parent_tuple)
            def _match_parent_rows(df0: pd.DataFrame, level: int, parent_tuple: tuple[str, ...]) -> pd.DataFrame:
                dfm = df0.copy()
                # Use helper functions to ensure proper level mapping
                level = clamp_level(level)
                pt = sanitize_parent_tuple(level, parent_tuple)
                pcols = parent_cols(level)
                
                # Respect the parent prefix up to level-1
                for i, col in enumerate(pcols):
                    if i < len(pt):
                        want = _nz(pt[i])
                        dfm = dfm[dfm[col].astype(str).map(_nz) == want]
                # If level==1, parent is ROOT; we'll apply bulk to all Node 1 rows under this VM/sheet
                return dfm

            matched = _match_parent_rows(df, level, selected_parent_path)

            # Bulk-edit boxes
            bcol1, bcol2 = st.columns(2)
            with bcol1:
                bulk_triage = st.text_area("Bulk set: Diagnostic Triage (applies to all matched rows)", key=f"bulk_triage_{level}_{'__'.join(selected_parent_path)}")
            with bcol2:
                bulk_actions = st.text_area("Bulk set: Actions (applies to all matched rows)", key=f"bulk_actions_{level}_{'__'.join(selected_parent_path)}")

            if st.button("💾 Apply bulk Outcomes to matched rows", key=f"bulk_apply_{level}_{'__'.join(selected_parent_path)}"):
                # Snapshot for UNDO
                stack = st.session_state.get("undo_stack", [])
                stack.append({
                    "context": "symptoms_outcomes_bulk",
                    "sheet": sheet_name,
                    "df_before": df.copy(),
                    "overrides_sheet_before": st.session_state.get("branch_overrides", {}).get(sheet_name, {}).copy(),
                })
                st.session_state["undo_stack"] = stack
                st.session_state["redo_stack"] = []

                df_new = df.copy()
                # Identify indices to update: rows that match parent prefix AND have a non-empty node at this level
                idxs = _match_parent_rows(df_new, level, selected_parent_path).index
                if normalize_text(bulk_triage):
                    df_new.loc[idxs, "Diagnostic Triage"] = bulk_triage
                if normalize_text(bulk_actions):
                    df_new.loc[idxs, "Actions"] = bulk_actions

                # Save back
                wb = USTATE.get_active_workbook()
                if wb and sheet_name in wb:
                    wb[sheet_name] = df_new
                    USTATE.set_active_workbook(wb, source="symptoms_outcomes_bulk")
                    st.success(f"Applied Outcomes to {len(idxs)} row(s).")
                else:
                    st.error("Could not update workbook.")

            # Optional: light per-row editor (first 50 matches)
            with st.expander("Per-row edit (first 50 matched rows)", expanded=False):
                # Ensure required columns exist
                for col in CANON_HEADERS:
                    if col not in matched.columns:
                        matched[col] = ""
                
                small = matched.head(50)[CANON_HEADERS].copy()
                edited = st.data_editor(
                    small,
                    hide_index=False,
                    use_container_width=True,
                    column_config={
                        "Vital Measurement": st.column_config.TextColumn("Vital Measurement", disabled=True),
                        "Node 1": st.column_config.TextColumn("Node 1", disabled=True),
                        "Node 2": st.column_config.TextColumn("Node 2", disabled=True),
                        "Node 3": st.column_config.TextColumn("Node 3", disabled=True),
                        "Node 4": st.column_config.TextColumn("Node 4", disabled=True),
                        "Node 5": st.column_config.TextColumn("Node 5", disabled=True),
                        "Diagnostic Triage": st.column_config.TextColumn("Diagnostic Triage"),
                        "Actions": st.column_config.TextColumn("Actions"),
                    },
                    num_rows="dynamic"
                )
                if st.button("💾 Save per-row Outcomes (above table)", key=f"row_save_{level}_{'__'.join(selected_parent_path)}"):
                    # UNDO snapshot
                    stack = st.session_state.get("undo_stack", [])
                    stack.append({
                        "context": "symptoms_outcomes_rows",
                        "sheet": sheet_name,
                        "df_before": df.copy(),
                        "overrides_sheet_before": st.session_state.get("branch_overrides", {}).get(sheet_name, {}).copy(),
                    })
                    st.session_state["undo_stack"] = stack
                    st.session_state["redo_stack"] = []

                    # Merge edits back by positional alignment on the slice
                    df_new = df.copy()
                    # Map: original indices of 'matched.head(50)' to edited values
                    edited_index_map = dict(zip(matched.head(50).index, range(len(edited))))
                    for orig_idx, pos in edited_index_map.items():
                        df_new.at[orig_idx, "Diagnostic Triage"] = edited.iloc[pos]["Diagnostic Triage"]
                        df_new.at[orig_idx, "Actions"] = edited.iloc[pos]["Actions"]

                    # Save back
                    wb = USTATE.get_active_workbook()
                    if wb and sheet_name in wb:
                        wb[sheet_name] = df_new
                        USTATE.set_active_workbook(wb, source="symptoms_outcomes_rows")
                        st.success("Per-row Outcomes saved.")
                    else:
                        st.error("Could not update workbook.")
    else:
        st.info("Please select at least one child to continue.")


@st.cache_data(ttl=600)
def _get_parent_paths_at_level(df: pd.DataFrame, level: int, nonce: str) -> List[Tuple[str, ...]]:
    """Get distinct parent paths at a specific level."""
    try:
        if level < 2:
            return []
        
        # Build parent columns (up to level-1)
        parent_cols = [f"Node {i}" for i in range(1, level)]
        if not all(col in df.columns for col in parent_cols):
            return []
        
        # Get unique parent paths
        parent_paths = df[parent_cols].apply(
            lambda r: tuple(normalize_text(v) for v in r), axis=1
        )
        parent_paths = parent_paths[parent_paths.apply(
            lambda x: all(v != "" for v in x)
        )]
        
        # Return unique paths, sorted
        unique_paths = sorted(parent_paths.unique())
        return unique_paths
        
    except Exception:
        return []


@st.cache_data(ttl=600)
def _get_current_children_for_parent(df: pd.DataFrame, level: int, parent_path: Tuple[str, ...], nonce: str) -> List[str]:
    """Get current children for a specific parent at a level."""
    try:
        if level > 5:
            return []
        
        node_col = f"Node {level}"
        if node_col not in df.columns:
            return []
        
        # Find rows matching the parent path
        matching_rows = df.copy()
        for i, expected_value in enumerate(parent_path):
            col = f"Node {i + 1}"
            if col in df.columns:
                mask = df[col].map(normalize_text) == expected_value
                matching_rows = matching_rows[mask]
        
        if matching_rows.empty:
            return []
        
        # Get children at this level
        children_values = matching_rows[node_col].map(normalize_text).dropna()
        children_values = children_values[children_values != ""]
        
        return sorted(children_values.unique())
        
    except Exception:
        return []


def _build_parent_key(level: int, parent_path: Tuple[str, ...]) -> str:
    """Build the parent key for the store."""
    if not parent_path:
        return f"L{level}|"
    else:
        return f"L{level}|" + ">".join(parent_path)


def _apply_branch_editor_changes(df: pd.DataFrame, level: int, parent_path: Tuple[str, ...], 
                                new_children: List[str], sheet_name: str):
    """Apply branch editor changes using override/materialization pipeline."""
    try:
        with st.spinner("Applying branch changes..."):
            # Create override key
            override_key = (level, parent_path)
            
            # Get current overrides
            overrides_all = st.session_state.get("branch_overrides", {})
            if sheet_name not in overrides_all:
                overrides_all[sheet_name] = {}
            
            # Set the override for this specific parent only
            overrides_all[sheet_name][override_key] = new_children
            st.session_state["branch_overrides"] = overrides_all
            
            # Apply the override using logic.tree
            from logic.tree import build_raw_plus_v630
            
            # Get the active workbook
            active_wb = USTATE.get_active_workbook()
            
            if active_wb and sheet_name in active_wb:
                # Apply overrides and rebuild the sheet
                updated_df = build_raw_plus_v630(df, overrides_all[sheet_name])
                
                # Update the active workbook
                active_wb[sheet_name] = updated_df
                USTATE.set_active_workbook(active_wb, source="symptoms_editor")
                
                # Clear stale caches to ensure immediate refresh
                st.cache_data.clear()
                
                # Show diff/preview
                rows_affected = len(updated_df) - len(df)
                st.success("✅ Branch changes applied successfully!")
                st.info(f"**Delta:** {rows_affected:+d} rows affected.")
                st.write(f"**New children set:** {', '.join(new_children)}")
                
                # Show small preview of changes
                if rows_affected != 0:
                    st.write("**Preview of changes:**")
                    if rows_affected > 0:
                        st.write(f"Added {rows_affected} new rows with the updated children set.")
                    else:
                        st.write(f"Removed {abs(rows_affected)} rows to standardize the children set.")
                
                # Rerun to show updated state
                st.warning("⚠️ Rerun skipped for debugging")
                # safe_rerun()
            else:
                st.error("Could not update active workbook.")
                
    except Exception as e:
        st.error(f"Error applying branch changes: {e}")
        st.exception(e)


def _render_streamlined_symptoms_editor(df: pd.DataFrame, df_norm: pd.DataFrame, sheet_name: str, summary: Dict[Tuple[int, Tuple[str, ...]], Dict[str, object]], queue_a: List[Tuple[int, Tuple[str, ...]]], queue_b: List[Tuple[int, Tuple[str, ...]]]):
    """Render the streamlined symptoms editor with queue-driven navigation."""
    
    # put at the very start of the Symptoms render() (or _render_streamlined_symptoms_editor)
    # TEMPORARILY COMMENTED OUT CSS WRAPPER TO DEBUG BLANK TABS
    # st.markdown("""
    # <style>
    # /* Scope to symptoms only */
    # .symp hr,
    # .symp .stDivider,
    # .symp [data-testid="stMarkdownContainer"] hr {
    #   display: none !important;
    #   margin: 0 !important;
    #   padding: 0 !important;
    #   height: 0 !important;
    #   border: 0 !important;
    # }

    # /* Remove extra spacing right under button rows */
    # .symp .stHorizontalBlock, .symp .stButton > button {
    #   margin-bottom: 0 !important;
    # }

    # /* Only hide empty <p>, don't hide the whole container */
    # .symp [data-testid="stMarkdownContainer"] > p:empty {
    #   display: none !important;
    #   margin: 0 !important;
    #   padding: 0 !important;
    # }

    # .symp .chip {
    #   display:inline-block;
    #   padding:3px 8px;
    #   margin:0 6px 6px 0;
    #   border-radius:12px;
    #   border:1px solid #dbe0e6;
    #   background:#fff;
    #   font-size:12px;
    # }

    # /* Compact, marginless card */
    # .symp .symp-card{
    #   background:#f0f2f6;
    #   border:1px solid #dbe0e6;
    #   border-radius:8px;
    #   padding:16px;
    #   margin:0 !important;          /* kill any outer gap */
    #   display:flex;
    #   gap:24px;
    #   align-items:flex-start;
    # }
    # /* two-column layout */
    # .symp .symp-card__left{ flex:2; min-width:0; }
    # .symp .symp-card__right{ flex:1; min-width:0; }


    # </style>
    # <div class="symp">
    # """, unsafe_allow_html=True)


    
    st.subheader("✏️ Streamlined Symptoms Editor")
    
    # Debug toggle
    show_debug = st.session_state.get("sym_show_debug", False)
    show_debug = st.toggle("Show debug", value=show_debug, key="sym_show_debug")
    
    # Add counters row (metrics) just below the title
    zero_total = len(queue_a)
    lt5_total = len(queue_b)
    # For current target, compute after target selection; for now set child_k = 0
    c1, c2, c3 = st.columns(3)
    m1 = c1.empty()
    c2.metric("Parents with 0 children", f"{zero_total}")
    c3.metric("Parents with ≤5 children", f"{lt5_total}")
    
    # Controls bar with two Next buttons
    # Maintain per-sheet state keys
    posA_key = f"sym_pos_A_{sheet_name}"
    posB_key = f"sym_pos_B_{sheet_name}"
    active_key = f"sym_active_queue_{sheet_name}"  # default to "A" if queue_a non-empty; else "B"
    
    # Two buttons for queue navigation
    col1, col2 = st.columns(2)
    
    with col1:
        if queue_a:
            if st.button("➡️ Next (no children)", key="sym_next_queue_a"):
                st.session_state[posA_key] = (st.session_state.get(posA_key, 0) + 1) % len(queue_a)
                st.session_state[active_key] = "A"
                st.rerun()
        else:
            st.success("No items in this queue 🎉")
    
    with col2:
        if queue_b:
            if st.button("➡️ Next (≤5 children)", key="sym_next_queue_b"):  # Updated label
                st.session_state["sym_next_queue_b"] = True
                st.session_state["sym_queue_b_pos"] = min(len(queue_b) - 1, queue_b_pos + 1)
                st.rerun()
        else:
            st.success("No items in this queue 🎉")
    

    
    # Pick the current target from the stored position
    active = st.session_state.get(active_key, "A" if queue_a else "B")
    if active == "A" and queue_a:
        pos = st.session_state.get(posA_key, 0) % len(queue_a)
        L, pt_raw = queue_a[pos]
    elif active == "B" and queue_b:
        pos = st.session_state.get(posB_key, 0) % len(queue_b)
        L, pt_raw = queue_b[pos]
    else:
        st.success("All parents completed across both queues 🎉")
        return
    
    # Sanitize: pt = sanitize_parent_tuple(L, pt_raw)
    L = clamp_level(L)
    raw_parent = pt_raw  # this is the "raw" tuple coming from the queue
    pt = sanitize_parent_tuple(L, raw_parent)  # this is the sanitized parent tuple you will use everywhere
    pcols = parent_cols(L)
    ccol = child_col(L)
    
    # Context header card (clear and human)
    # Parent:
    if L == 1:
        parent_display = "Parent: ROOT (Vital Measurement) → children in Node 1"
    else:
        parent_display = f"Parent: Node {L-1} = '{pt[-1]}' → children in Node {L}"
    
    # Path: breadcrumb up to L-1
    if L == 1:
        breadcrumb = "Vital Measurement (root)"
    else:
        breadcrumb_parts = []
        for i, val in enumerate(pt):
            if i < len(NODE_COLS):
                breadcrumb_parts.append(f"Node {i+1}='{val}'")
        breadcrumb = " > ".join(breadcrumb_parts)
    
    # Queue info
    if active == "A":
        queue_info = f"Queue A (no children) • position {pos+1} of {len(queue_a)}"
    else:
        queue_info = f"Queue B (≤5 children) • position {pos+1} of {len(queue_b)}"
    
    # Sheet & VM: show active sheet and the first non-empty df["Vital Measurement"]
    vm_label = "—"
    if "Vital Measurement" in df.columns:
        vm_values = df["Vital Measurement"].dropna().astype(str).str.strip()
        vm_values = vm_values[vm_values != ""]
        if not vm_values.empty:
            vm_label = vm_values.iloc[0]
    
    # Rows: len(matches) where matches is the parent mask over pcols
    # Build the parent mask from df_norm using strict normalization
    mask = pd.Series(True, index=df_norm.index)
    for i, col in enumerate(pcols):
        if i < len(pt):
            # Use strict normalization for consistent comparison
            mask &= (df_norm[col] == _nz_strict(pt[i]))
        else:
            mask &= (df_norm[col] == "")
    
    matches = df_norm[mask]
    match_count = len(matches)
    
    # Context Card (compact, marginless design)
    st.markdown(f"""
<div class="symp-card">
  <div class="symp-card__left">
    <div><strong>🎯 {parent_display}</strong></div>
    <div><strong>📍 Path:</strong> {breadcrumb}</div>
    <div><strong>📊 Queue:</strong> {queue_info}</div>
  </div>
  <div class="symp-card__right">
    <div><strong>📁 Sheet:</strong> {sheet_name}</div>
    <div><strong>🔬 VM:</strong> {vm_label}</div>
    <div><strong>📈 Rows:</strong> {match_count} matching</div>
  </div>
</div>
""", unsafe_allow_html=True)
    
    # Existing children (chips/lists)
    # children_sheet: distinct non-empty ccol in matches
    if L == 1:
        # For ROOT level, get all non-empty Node 1 values
        sheet_children = sorted(matches[ccol].loc[matches[ccol] != ""].unique(), key=str.lower)
    else:
        # For other levels, filter by parent columns
        sheet_children = sorted(matches[ccol].loc[matches[ccol] != ""].unique(), key=str.lower)
    
    # children_over: from overrides for key f"L{L}|<ROOT>" or f"L{L}|{'>'.join(pt)}"
    if pt == ():
        override_key = f"L{L}|<ROOT>"
    else:
        override_key = f"L{L}|{'>'.join(pt)}"
    
    # Get children from the summary (which comes from the store)
    override_children = summary.get((L, pt), {}).get("children", [])
    
    # Render two labeled rows
    st.markdown("**Existing Children:**")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**📋 Children in sheet:**")
        if sheet_children:
            st.markdown("".join(f"<span class='chip'>{c}</span>" for c in sheet_children), unsafe_allow_html=True)
        else:
            st.caption("None found in sheet")
    
    with col2:
        st.markdown("**⏳ Children from overrides:**")
        if override_children:
            st.markdown("".join(f"<span class='chip'>{c}</span>" for c in override_children), unsafe_allow_html=True)
        else:
            st.caption("None pending in overrides")
    
    # Prefill and render five inputs (stateful)
    # Build union (order-preserving, de-duplicated):
    prefill = []
    seen = set()
    
    # Add override children first (they take priority)
    for child in override_children:
        if _nz_strict(child) and _nz_strict(child) not in seen:
            prefill.append(_nz_strict(child))
            seen.add(_nz_strict(child))
    
    # Add sheet children (if not already in overrides)
    for child in sheet_children:
        if _nz_strict(child) and _nz_strict(child) not in seen:
            prefill.append(_nz_strict(child))
            seen.add(_nz_strict(child))
    
    # Cap at 5
    prefill = prefill[:5]
    
    # Use a per-target session key so inputs persist
    sess_inputs_key = f"sym_inputs::{sheet_name}::L{L}::{'>'.join(pt) if pt else '<ROOT>'}"
    
    # Initialize st.session_state[sess_inputs_key] = prefill if not set
    if sess_inputs_key not in st.session_state:
        st.session_state[sess_inputs_key] = prefill
    
    # Render 5 st.text_input boxes with only keys
    st.markdown("**Children (up to 5):**")
    children_inputs = []
    for i in range(5):
        child_key = f"{sess_inputs_key}::{i}"
        
        # If child_key not in session yet, set it from st.session_state[sess_inputs_key][i]
        if child_key not in st.session_state:
            st.session_state[child_key] = st.session_state[sess_inputs_key][i] if i < len(st.session_state[sess_inputs_key]) else ""
        
        # Render: st.text_input(f"Child {i+1}", key=child_key)
        input_value = st.text_input(
            f"Child {i+1}",
            key=child_key,
            help=f"Enter child {i+1} for {ccol}"
        )
        children_inputs.append(input_value)
    
    # Keep st.session_state[sess_inputs_key] in sync
    st.session_state[sess_inputs_key] = [st.session_state.get(f"{sess_inputs_key}::{i}", "") for i in range(5)]
    
    # Compute child_k after rendering
    child_k = len([v for v in st.session_state[sess_inputs_key] if _nz_strict(v)])
    
    # Update the Current children metric to show f"{child_k}/5"
    m1.metric("Current children", f"{child_k}/5")
    
    # Clear inputs when the target changes
    last_target_key = st.session_state.get(f"sym_last_target_key_{sheet_name}", "")
    if last_target_key != sess_inputs_key:
        # Purge the old keys
        for i in range(5):
            st.session_state.pop(f"{last_target_key}::{i}", None)
        st.session_state.pop(last_target_key, None)
        
        # Update last target key
        st.session_state[f"sym_last_target_key_{sheet_name}"] = sess_inputs_key
    
    # 6) Buttons & flow
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("💾 Save & Next", key=f"{sess_inputs_key}_save_next"):
            # 5) Save handler: read from state, normalize, save, recompute queues, advance
            # Rebuild values = [st.session_state.get(f"{sess_inputs_key}::{i}", "") for i in range(5)]
            values = [st.session_state.get(f"{sess_inputs_key}::{i}", "") for i in range(5)]
            
            # Normalize → strip → drop blanks → de-dupe (order-preserving) → truncate to 5
            new_children = []
            for value in values:
                trimmed = _nz(value)
                if trimmed and trimmed not in new_children:
                    new_children.append(trimmed)
            
            if len(new_children) > 5:
                st.error("Maximum 5 children allowed!")
                return
            
            # Update branch_overrides[sheet] for the key f"L{L}|<ROOT>" (if L==1) or f"L{L}|{'>'.join(pt)}"
            overrides_all = st.session_state.get("branch_overrides", {})
            if sheet_name not in overrides_all:
                overrides_all[sheet_name] = {}
            
            # Use the existing override schema/keying
            if pt == ():
                override_key = f"L{L}|<ROOT>"
            else:
                override_key = f"L{L}|{'>'.join(pt)}"
            
            overrides_all[sheet_name][override_key] = new_children
            st.session_state["branch_overrides"] = overrides_all
            
            # Call your existing materialize/save routine (the one already used elsewhere) to write to Node L
            try:
                from logic.tree import infer_branch_options_with_overrides
                
                # Get active workbook
                wb = USTATE.get_active_workbook()
                if not wb or sheet_name not in wb:
                    st.error("No active workbook found.")
                    return
                
                # Apply overrides and rebuild
                updated_df = infer_branch_options_with_overrides(df, overrides_all[sheet_name])
                
                # Update workbook
                wb[sheet_name] = updated_df
                USTATE.set_active_workbook(wb, source="symptoms_editor")
                
                # Clear caches
                st.cache_data.clear()
                
                # Toast success
                st.toast(f"Saved Node {L} children for parent: {' > '.join(pt) or 'ROOT'}")
                
                # After save: Clear the per-target input state
                for i in range(5):
                    st.session_state.pop(f"{sess_inputs_key}::{i}", None)
                st.session_state.pop(sess_inputs_key, None)
                
                # Advance the corresponding queue position (wrap allowed) and rerun
                if active == "A" and queue_a:
                    st.session_state[posA_key] = (st.session_state.get(posA_key, 0) + 1) % len(queue_a)
                elif active == "B" and queue_b:
                    st.session_state[posB_key] = (st.session_state.get(posB_key, 0) + 1) % len(queue_a)
                
                # Rerun to update queues
                st.rerun()
                
            except Exception as e:
                st.error(f"Error saving: {e}")
                st.exception(e)
    
    with col2:
        if st.button("💾 Save", key=f"{sess_inputs_key}_save"):
            # Same as Save & Next but without advancing
            # Rebuild values from session state
            values = [st.session_state.get(f"{sess_inputs_key}::{i}", "") for i in range(5)]
            
            # Normalize → strip → drop blanks → de-dupe (order-preserving) → truncate to 5
            new_children = []
            for value in values:
                trimmed = _nz(value)
                if trimmed and trimmed not in new_children:
                    new_children.append(trimmed)
            
            if len(new_children) > 5:
                st.error("Maximum 5 children allowed!")
                return
            
            # Update branch_overrides[sheet] for the key f"L{L}|<ROOT>" (if L==1) or f"L{L}|{'>'.join(pt)}"
            overrides_all = st.session_state.get("branch_overrides", {})
            if sheet_name not in overrides_all:
                overrides_all[sheet_name] = {}
            
            # Use the existing override schema/keying
            if pt == ():
                override_key = f"L{L}|<ROOT>"
            else:
                override_key = f"L{L}|{'>'.join(pt)}"
            
            overrides_all[sheet_name][override_key] = new_children
            st.session_state["branch_overrides"] = overrides_all
            
            # Call your existing materialize/save routine (the one already used elsewhere) to write to Node L
            try:
                from logic.tree import infer_branch_options_with_overrides
                
                # Get active workbook
                wb = USTATE.get_active_workbook()
                if not wb or sheet_name not in wb:
                    st.error("No active workbook found.")
                    return
                
                # Apply overrides and rebuild
                updated_df = infer_branch_options_with_overrides(df, overrides_all[sheet_name])
                
                # Update workbook
                wb[sheet_name] = updated_df
                USTATE.set_active_workbook(wb, source="symptoms_editor")
                
                # Clear caches
                st.cache_data.clear()
                
                # Toast success
                st.toast(f"Saved Node {L} children for parent: {' > '.join(pt) or 'ROOT'}")
                
                # After save: Clear the per-target input state
                for i in range(5):
                    st.session_state.pop(f"{sess_inputs_key}::{i}", None)
                st.session_state.pop(sess_inputs_key, None)
                
                # Just rerun to update queues (no position advance)
                st.rerun()
                
            except Exception as e:
                st.error(f"Error saving: {e}")
                st.exception(e)
    
    with col3:
        if st.button("⏭️ Skip", key=f"{sess_inputs_key}_skip"):
            # 6) Skip handler: advance without saving, clear inputs
            # Clear the same session keys as above
            for i in range(5):
                st.session_state.pop(f"{sess_inputs_key}::{i}", None)
            st.session_state.pop(sess_inputs_key, None)
            
            # Advance the position in the same queue without saving
            if active == "A" and queue_a:
                st.session_state[posA_key] = (st.session_state.get(posA_key, 0) + 1) % len(queue_a)
            elif active == "B" and queue_b:
                st.session_state[posB_key] = (st.session_state.get(posB_key, 0) + 1) % len(queue_b)
            st.rerun()
    
    # 8) Debug (keep collapsed)
    if show_debug:
        with st.expander("🔍 Debug Info", expanded=False):
            st.write(f"**Current target (raw):** ({L}, {raw_parent or 'ROOT'})")
            st.write(f"**Sanitized target:** ({L}, {pt or 'ROOT'})")
            st.write(f"**Parent columns:** {pcols}")
            st.write(f"**Child column:** {ccol}")
            st.write(f"**Children in sheet:** {sheet_children}")
            st.write(f"**Children from overrides:** {override_children}")
            st.write(f"**Queue sizes:** |A|={len(queue_a)}, |B|={len(queue_b)}")
            
            # Get current positions for the active queue
            if active == "A":
                pos = st.session_state.get(posA_key, 0) % len(queue_a)
                st.write(f"**Current position:** A={pos}")
            else:
                pos = st.session_state.get(posB_key, 0) % len(queue_b)
                st.write(f"**Current position:** B={pos}")
            
            # Guard for char-split (dev warning)
            if L - 1 == 1 and isinstance(raw_parent, (tuple, list)) and all(len(str(x)) == 1 for x in raw_parent) and len(raw_parent) > 1:
                st.warning("Detected character-split parent; sanitizing to a single token.")
            
            # Show whether those columns exist in df.columns
            st.write(f"**Parent columns exist:** {all(col in df.columns for col in pcols)}")
            st.write(f"**Child column exists:** {ccol in df.columns}")
            
            # E) Optional (but recommended) tiny debug hooks
            st.write("---")
            st.write("**Queue Debug Hooks:**")
            
            # Show pcols, ccol
            st.write(f"**pcols:** {pcols}")
            st.write(f"**ccol:** {ccol}")
            
            # Show type of grp.index and the coerced grp_keys (first few)
            # This would require recomputing the groupby, so we'll show what we can
            st.write(f"**Parent mask matches:** {len(matches)} rows")
            st.write(f"**Sheet children count:** {len(sheet_children)}")
            st.write(f"**Override children count:** {len(override_children)}")
            
            # Show whether current_target is actually in the correct queue
            if active == "A":
                st.write(f"**In Queue A (no children):** {(L, pt) in queue_a}")
            else:
                st.write(f"**In Queue B (≤5 children):** {(L, pt) in queue_b}")
            
            # 3) Add a one-time debug to verify equality paths
            st.write("---")
            st.write("**Equality Path Debug:**")
            
            # Recompute grp_keys for this specific level to verify equality
            scope_nonempty = df_norm[df_norm[ccol] != ""]
            if pcols:
                grp = scope_nonempty.groupby(pcols, dropna=False)[ccol].nunique()
                keys = grp.index
                if len(pcols) > 1:
                    grp_keys = {tuple(_nz_strict(v) for v in t) for t in keys.tolist()}
                else:
                    grp_keys = {(_nz_strict(v),) for v in keys.tolist()}
            else:
                grp_keys = set()
            
            st.code({
                "pt_tuple": repr(pt),
                "pt_in_grp_keys": pt in grp_keys,
                "sample_grp_key": repr(next(iter(grp_keys)) if grp_keys else None),
            })
            
            # Print repr of first few raw Node-1 values for sanity
            if "Node 1" in df_norm.columns:
                sample_values = df_norm["Node 1"].head(5).tolist()
                st.write(f"**Sample Node 1 values (repr):** {[repr(v) for v in sample_values]}")
    
    # Close the CSS wrapper div
    # TEMPORARILY COMMENTED OUT CSS WRAPPER TO DEBUG BLANK TABS
    # st.markdown("</div>", unsafe_allow_html=True)
