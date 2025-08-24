# ui/tabs/calculator.py
import streamlit as st
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
import utils.state as USTATE
from ui.utils.debug import dump_state, banner
from utils.constants import ROOT_COL, LEVEL_COLS
from utils.helpers import normalize_text

from logic.tree import infer_branch_options_with_overrides


def _nz(s) -> str:
    """Local text normalizer - strips whitespace and handles None/NaN."""
    return "" if s is None else str(s).strip()


def render():
    """Render the Calculator tab for navigating decision tree paths."""
    
    # Add guard and debug expander
    from ui.utils.guards import ensure_active_workbook_and_sheet
    ok, df = ensure_active_workbook_and_sheet("Calculator")
    if not ok:
        return
    
    # Debug state expander
    import json
    with st.expander("🛠 Debug: Session State (tab)", expanded=False):
        ss = {k: type(v).__name__ for k,v in st.session_state.items()}
        st.code(json.dumps(ss, indent=2))
    
    banner("Calculator RENDER ENTRY")
    dump_state("Session (pre-calculator)")
    
    try:
        st.header("🧮 Calculator")
        
        # Get current sheet name for display
        sheet = USTATE.get_current_sheet()
        
        # Ensure Node columns exist (add empty strings if any of Node 1..Node 5 missing)
        from utils.constants import NODE_COLS
        for col in NODE_COLS:
            if col not in df.columns:
                df[col] = ""
        
        # Show Root (VM) as a read-only label only
        vm_values = df["Vital Measurement"].dropna().astype(str).str.strip()
        vm_values = vm_values[vm_values != ""]
        if not vm_values.empty:
            vm_label = vm_values.iloc[0]  # First non-empty value
        else:
            vm_label = "—"
        
        st.caption(f"VM (Root): {vm_label}")

        # Main Path Navigator
        _render_path_navigator(df, sheet)
        
        # Path Results - check if any path is selected
        nav_key = f"calc_nav_{sheet}"
        current_context = st.session_state.get(nav_key, {})
        current_path = []
        for i in range(1, 6):  # Only check N1-N5 (not N6)
            if current_context.get(f"N{i}"):
                current_path.append(current_context[f"N{i}"])
        
        if current_path:
            st.markdown("---")
            _render_path_results(df, sheet, current_path)
        
    except Exception as e:
        st.error(f"Exception in Calculator.render(): {e}")
        st.exception(e)


@st.cache_data(ttl=600)
def _get_cached_branch_options(df: pd.DataFrame, nonce: str) -> Dict[str, Any]:
    """Get cached branch options for the path navigator."""
    return infer_branch_options(df)


def _calc_build_nested_options(df: pd.DataFrame, overrides_sheet: Dict[str, List[str]]) -> Dict[int, Dict[Tuple[str, ...], List[str]]]:
    """
    Build {level -> {parent_tuple -> [children...]}} from df + overrides.
    Level 1 parents = ROOT (empty tuple), children live in Node 1.
    """
    store = infer_branch_options_with_overrides(df, overrides_sheet)
    out: Dict[int, Dict[Tuple[str, ...], List[str]]] = {i: {} for i in range(1, 6)}
    for key, children in (store or {}).items():
        # Expected keys like "L1|<ROOT>", "L2|Headache", "L3|Headache>Thunderclap Headache", ...
        if "|" not in key:
            continue
        lvl_s, path = key.split("|", 1)
        try:
            L = int(lvl_s[1:])
        except Exception:
            continue
        if L < 1 or L > 5:
            continue
        parent_tuple: Tuple[str, ...] = tuple([] if path == "<ROOT>" else path.split(">"))
        out[L][parent_tuple] = [c for c in (children or []) if _nz(c)]
    return out


def children_for(by_parent: Dict[int, Dict[Tuple[str, ...], List[str]]], level: int, path: str) -> List[str]:
    """Get children for a given level and parent path."""
    if level not in by_parent:
        return []
    
    # Convert path string to tuple for lookup
    if path == "<ROOT>":
        parent_tuple = tuple()
    else:
        parent_tuple = tuple(path.split(">"))
    
    return by_parent[level].get(parent_tuple, [])


def _render_path_navigator(df: pd.DataFrame, sheet_name: str):
    """Render the path navigator interface with correct Root/Level mapping and Node 2-5 population."""
    st.subheader("🗺️ Path Navigator")
    st.markdown("Walk through the decision tree by selecting options at each level.")
    
    if df is None or df.empty:
        st.info("Load a workbook in Source.")
        return

    # Build the store and nested lookup "by_parent"
    overrides_sheet = st.session_state.get("branch_overrides", {}).get(sheet_name, {})
    store = infer_branch_options_with_overrides(df, overrides_sheet)
    
    # Parse keys like "L1|<ROOT>", "L2|Headache", "L3|Headache>Thunderclap Headache"
    by_parent: Dict[int, Dict[Tuple[str, ...], List[str]]] = {i: {} for i in range(1, 6)}
    
    for key, children in (store or {}).items():
        if "|" not in key:
            continue
        lstr, path = key.split("|", 1)
        try:
            L = int(lstr[1:])
        except Exception:
            continue
        if L < 1 or L > 5:
            continue
        
        # Parse parent tuple - ensure proper tuple construction
        if path == "<ROOT>":
            parent_tuple = ()
        else:
            # Split by '>' and normalize each part
            parent_tuple = tuple(_nz(part) for part in path.split(">"))
        
        children = [c for c in (children or []) if _nz(c)]
        by_parent[L][parent_tuple] = children
    
    # Root children (Node 1 options)
    root_children = by_parent.get(1, {}).get((), [])
    
    # Fallback: if empty, use sorted unique, non-empty df["Node 1"]
    if not root_children:
        fallback_node1 = sorted(df["Node 1"].astype(str).str.strip().replace("nan", "").unique())
        fallback_node1 = [x for x in fallback_node1 if x and x != ""]
        
        if fallback_node1:
            by_parent[1][()] = fallback_node1
            root_children = fallback_node1
        else:
            st.warning("No Node 1 options found. Check 'Vital Measurement' and 'Node 1' data.")
            return
    
    # State & cascade reset
    nav_key = f"calc_nav_{sheet_name}"
    state = st.session_state.setdefault(nav_key, {"N1": None, "N2": None, "N3": None, "N4": None, "N5": None})
    
    # State hygiene - reset when sheet changes
    if st.session_state.get("calc_nav_sheet") != sheet_name:
        state = {"N1": None, "N2": None, "N3": None, "N4": None, "N5": None}
        st.session_state[nav_key] = state
        st.session_state["calc_nav_sheet"] = sheet_name
    
    # Debug expander (collapsed by default)
    with st.expander("🔍 Debug (Path Navigator)", expanded=False):
        st.write(f"**Root children count:** {len(root_children)}")
        st.write(f"**by_parent[1] has () key:** {() in by_parent.get(1, {})}")
        st.write(f"**by_parent[1][()] children count:** {len(by_parent.get(1, {}).get((), []))}")
        st.write(f"**Sheet:** {sheet_name}")
        st.write(f"**DataFrame shape:** {df.shape}")
    
    # Reset path button
    if st.button("🔄 Reset Path", key="calc_reset_path"):
        state = {"N1": None, "N2": None, "N3": None, "N4": None, "N5": None}
        st.session_state[nav_key] = state
        st.warning("⚠️ Rerun skipped for debugging")
        # st.rerun()
    
    st.markdown("---")
    
    # Helper function for fallback options
    def get_fallback_options(level: int, parent_selections: List[str]) -> List[str]:
        """Get fallback options from DataFrame when store is empty."""
        if level < 2 or level > 5:
            return []
        
        # Build filter mask for parent columns
        mask = pd.Series(True, index=df.index)
        for i, selection in enumerate(parent_selections):
            col_name = f"Node {i+1}"
            if col_name in df.columns:
                # Normalize both DataFrame column and selection for comparison
                mask &= (df[col_name].astype(str).str.strip().replace("nan", "") == _nz(selection))
        
        # Get unique non-empty values from current level column
        level_col = f"Node {level}"
        if level_col in df.columns:
            filtered_values = df[mask][level_col].astype(str).str.strip().replace("nan", "").unique()
            return sorted([x for x in filtered_values if x and x != ""])
        return []
    
    # Node 1 selection
    node1 = st.selectbox(
        "Select Node 1 option",
        [""] + root_children,
        key=f"{nav_key}_node1",
        help="Choose the first node in your path"
    )
    
    # Reset cascade when higher levels change
    if node1 != state.get("N1"):
        state["N1"] = node1
        state["N2"] = None
        state["N3"] = None
        state["N4"] = None
        state["N5"] = None
        st.session_state[nav_key] = state
    
    if node1:
        # Node 2 options
        p1 = _nz(node1)
        key2 = (p1,)
        opts2 = by_parent.get(2, {}).get(key2, [])
        
        # Fallback if opts2 is empty
        if not opts2:
            opts2 = get_fallback_options(2, [p1])
        
        node2 = st.selectbox(
            "Select Node 2 option",
            [""] + opts2,
            key=f"{nav_key}_node2",
            help="Choose the second node in your path"
        )
        
        # Reset cascade when N2 changes
        if node2 != state.get("N2"):
            state["N2"] = node2
            state["N3"] = None
            state["N4"] = None
            state["N5"] = None
            st.session_state[nav_key] = state
        
        if node2:
            # Node 3 options
            p2 = _nz(node2)
            key3 = (p1, p2)
            opts3 = by_parent.get(3, {}).get(key3, [])
            
            # Fallback if opts3 is empty
            if not opts3:
                opts3 = get_fallback_options(3, [p1, p2])
            
            node3 = st.selectbox(
                "Select Node 3 option",
                [""] + opts3,
                key=f"{nav_key}_node3",
                help="Choose the third node in your path"
            )
            
            # Reset cascade when N3 changes
            if node3 != state.get("N3"):
                state["N3"] = node3
                state["N4"] = None
                state["N5"] = None
                st.session_state[nav_key] = state
            
            if node3:
                # Node 4 options
                p3 = _nz(node3)
                key4 = (p1, p2, p3)
                opts4 = by_parent.get(4, {}).get(key4, [])
                
                # Fallback if opts4 is empty
                if not opts4:
                    opts4 = get_fallback_options(4, [p1, p2, p3])
                
                node4 = st.selectbox(
                    "Select Node 4 option",
                    [""] + opts4,
                    key=f"{nav_key}_node4",
                    help="Choose the fourth node in your path"
                )
                
                # Reset cascade when N4 changes
                if node4 != state.get("N4"):
                    state["N4"] = node4
                    state["N5"] = None
                    st.session_state[nav_key] = state
                
                if node4:
                    # Node 5 options
                    p4 = _nz(node4)
                    key5 = (p1, p2, p3, p4)
                    opts5 = by_parent.get(5, {}).get(key5, [])
                    
                    # Fallback if opts5 is empty
                    if not opts5:
                        opts5 = get_fallback_options(5, [p1, p2, p3, p4])
                    
                    node5 = st.selectbox(
                        "Select Node 5 option",
                        [""] + opts5,
                        key=f"{nav_key}_node5",
                        help="Choose the fifth node in your path"
                    )
                    
                    if node5:
                        state["N5"] = node5
                        st.session_state[nav_key] = state
    
    # Debug expander for navigator keys (optional)
    with st.expander("🔍 Debug (Navigator keys)", expanded=False):
        if node1:
            p1_norm = _nz(node1)
            key2_debug = (p1_norm,)
            st.write(f"**Node 2 parent tuple:** {key2_debug}")
            st.write(f"**by_parent[2] contains key:** {key2_debug in by_parent.get(2, {})}")
            st.write(f"**Node 2 options from store:** {len(by_parent.get(2, {}).get(key2_debug, []))}")
            
            # Show fallback values when store is empty
            if not by_parent.get(2, {}).get(key2_debug, []):
                fallback_debug = get_fallback_options(2, [p1_norm])
                st.write(f"**Node 2 fallback DF values:** {fallback_debug[:5]}... (total: {len(fallback_debug)})")
        
        if node1 and node2:
            p2_norm = _nz(node2)
            key3_debug = (_nz(node1), p2_norm)
            st.write(f"**Node 3 parent tuple:** {key3_debug}")
            st.write(f"**by_parent[3] contains key:** {key3_debug in by_parent.get(3, {})}")
            st.write(f"**Node 3 options from store:** {len(by_parent.get(3, {}).get(key3_debug, []))}")
    
    # Current Path should display only the Node selections (do not include VM string in the tuple)
    current_path = []
    for i in range(1, 6):  # Only check N1-N5 (not N6)
        if state.get(f"N{i}"):
            current_path.append(state.get(f"N{i}"))
    
    if current_path:
        st.markdown("---")
        st.subheader("📍 Path Preview")
        path_display = " > ".join(current_path)
        st.info(f"**Current Path:** {path_display}")
        
        # Show current selections
        cols = st.columns(5)  # Only 5 columns for N1-N5
        for i, col in enumerate(cols):
            with col:
                level_name = f"Node {i+1}"
                value = state.get(f"N{i+1}", "")
                st.metric(level_name, value if value else "Not selected")


def _render_path_results(df: pd.DataFrame, sheet_name: str, current_path: List[str]):
    """Render the results for the selected path with row selection and CSV export."""
    st.subheader("📊 Path Results")
    
    if not current_path:
        return
    
    # Build filter mask for the selected path
    filter_mask = _build_path_filter_mask(df, current_path)
    
    if filter_mask is None:
        st.warning("Could not build filter for the selected path.")
        return
    
    # Apply filter to get matching rows
    matching_rows = df[filter_mask]
    
    if matching_rows.empty:
        st.info("No rows match the selected path.")
        return
    
    # Show summary
    st.success(f"Found {len(matching_rows)} matching row(s) for the path.")
    
    # Row selection
    st.subheader("📋 Row Selection")
    idx_options = matching_rows.index.tolist()
    sel_indices = st.multiselect(
        "Select rows to include in export", 
        options=idx_options, 
        default=idx_options, 
        key="calc_sel_rows"
    )
    chosen = matching_rows.loc[sel_indices] if sel_indices else matching_rows.head(0)
    
    if not chosen.empty:
        st.write(f"**Selected {len(chosen)} rows:**")
        st.dataframe(chosen, use_container_width=True)
        
        # CSV Export with Prevalence
        st.subheader("📤 CSV Export")
        _render_csv_export_with_prevalence(chosen)
    else:
        st.info("Please select at least one row for export.")


def _render_csv_export_with_prevalence(chosen_rows: pd.DataFrame):
    """Render CSV export with Prevalence instead of Quality."""
    try:
        # Build export table with 'Diagnosis' header and 'Prevalence' cells
        diag_col = "Diagnostic Triage" if "Diagnostic Triage" in chosen_rows.columns else "Diagnosis"
        prev_col = "Prevalence"  # rename from Quality
        
        # Ensure prevalence column exists (placeholder if missing)
        if prev_col not in chosen_rows.columns:
            chosen_rows[prev_col] = ""
        
        # Export layout:
        # First "header row": Diagnosis as header, then one column per selected row (diagnosis values)
        header = ["Diagnosis"] + chosen_rows[diag_col].astype(str).tolist()
        
        # Then 5 rows, one per Node 1..5, each row has [node label, prevalence per selected row]
        rows = []
        for i, node_col in enumerate(LEVEL_COLS, start=1):
            label_row = [f"Node {i}"]
            # Prevalence per selected row for this node row -> put the chosen[prev_col] value
            label_row += chosen_rows[prev_col].astype(str).tolist()
            rows.append(label_row)
        
        export_df = pd.DataFrame([header] + rows)
        
        st.download_button(
            "Download CSV",
            data=export_df.to_csv(index=False, header=False).encode("utf-8"),
            file_name="calculator_export.csv",
            mime="text/csv",
            key="calc_dl_csv"
        )
        
        st.info("CSV format: First row = Diagnosis values, subsequent rows = Node 1-5 with Prevalence entries")
        
    except Exception as e:
        st.error(f"Error creating CSV export: {e}")


def _build_path_filter_mask(df: pd.DataFrame, path: List[str]) -> Optional[pd.Series]:
    """Build a filter mask for the selected path using VM+Nodes approach."""
    try:
        if not path:
            return None
        
        # Start with all rows as True
        mask = pd.Series([True] * len(df), index=df.index)
        
        # Apply filter for each level in the path
        # path[0] = VM (Root), path[1] = Node 1, path[2] = Node 2, etc.
        for level, value in enumerate(path):
            if level == 0:
                # VM (Root) level
                col_name = ROOT_COL
            else:
                # Node levels (1-5)
                col_name = LEVEL_COLS[level - 1]
            
            if col_name in df.columns:
                # Filter rows where this column matches the path value
                level_mask = df[col_name].map(normalize_text) == value
                mask = mask & level_mask
            else:
                # Column doesn't exist, can't filter
                return None
        
        return mask
        
    except Exception:
        return None


def _display_path_results_table(df: pd.DataFrame):
    """Display the path results in a table format."""
    try:
        # Select relevant columns for display
        display_cols = []
        
        # Always include path columns
        for level in range(1, 6):
            col_name = f"Node {level}"
            if col_name in df.columns:
                display_cols.append(col_name)
        
        # Add diagnostic columns if they exist
        if "Diagnostic Triage" in df.columns:
            display_cols.append("Diagnostic Triage")
        if "Actions" in df.columns:
            display_cols.append("Actions")
        
        # Add Vital Measurement if it exists
        if "Vital Measurement" in df.columns:
            display_cols.insert(0, "Vital Measurement")
        
        # Create display DataFrame
        display_df = df[display_cols].copy()
        
        # Format the display
        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Vital Measurement": st.column_config.TextColumn("Vital Measurement", width="medium"),
                "Node 1": st.column_config.TextColumn("Node 1", width="medium"),
                "Node 2": st.column_config.TextColumn("Node 2", width="medium"),
                "Node 3": st.column_config.TextColumn("Node 3", width="medium"),
                "Node 4": st.column_config.TextColumn("Node 4", width="medium"),
                "Node 5": st.column_config.TextColumn("Node 5", width="medium"),
                "Diagnostic Triage": st.column_config.TextColumn("Diagnostic Triage", width="large"),
                "Actions": st.column_config.TextColumn("Actions", width="large"),
            }
        )
        
    except Exception as e:
        st.error(f"Error displaying results: {e}")
        st.exception(e)
