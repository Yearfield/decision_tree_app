# ui/tabs/calculator.py
import streamlit as st
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional

from utils import (
    CANON_HEADERS, LEVEL_COLS, normalize_text, validate_headers
)
from utils.state import (
    get_active_workbook, get_current_sheet, get_active_df, 
    has_active_workbook, get_workbook_status, get_wb_nonce
)
from ui.utils.rerun import safe_rerun
from logic.tree import infer_branch_options, analyze_decision_tree_with_root
from utils.constants import ROOT_COL, LEVEL_COLS, LEVEL_LABELS


def render():
    """Render the Calculator tab with Path Navigator."""
    try:
        st.header("🧮 Calculator")
        st.markdown("Navigate decision tree paths and explore outcomes.")
        
        # Status badge
        has_wb, sheet_count, current_sheet = get_workbook_status()
        if has_wb and current_sheet:
            st.caption(f"Workbook: ✅ {sheet_count} sheet(s) • Active: **{current_sheet}**")
        else:
            st.caption("Workbook: ❌ not loaded")
        
        # Guard against no active workbook
        wb = get_active_workbook()
        sheet = get_current_sheet()
        if not wb or not sheet:
            st.warning("No active workbook/sheet. Load a workbook in 📂 Source or select a sheet in 🗂 Workspace.")
            return

        # Get active DataFrame
        df = get_active_df()
        if df is None:
            st.warning("No active sheet selected. Please load a workbook in the Source tab and select a sheet.")
            return
        
        if not validate_headers(df):
            st.warning("Active sheet has invalid headers. Please ensure it has the required columns.")
            return

        # Initialize calculator path in session state
        if "calc_path" not in st.session_state:
            st.session_state["calc_path"] = []
        
        # Main Path Navigator
        _render_path_navigator(df, sheet)
        
        # Path Results
        if st.session_state["calc_path"]:
            st.markdown("---")
            _render_path_results(df, sheet)
        
    except Exception as e:
        st.exception(e)


@st.cache_data(ttl=600)
def _get_cached_branch_options(df: pd.DataFrame, nonce: str) -> Dict[str, Any]:
    """Get cached branch options for the path navigator."""
    return infer_branch_options(df)


def _calc_build_nested_options(df: pd.DataFrame) -> Dict[Tuple[int, str], List[str]]:
    """Build nested dict: vm -> node1 children → node2 children → ... node5."""
    res = analyze_decision_tree_with_root(df, get_wb_nonce())
    summary = res["summary"]
    # index: (L,parent_path) -> {children}
    # Build per-level lookup
    by_parent = {(L, p): info["children"] for (L, p), info in summary.items()}
    return by_parent


def children_for(by_parent: Dict[Tuple[int, str], List[str]], level: int, path: str) -> List[str]:
    """Get children for a given level and parent path."""
    return by_parent.get((level, path), [])


def _render_path_navigator(df: pd.DataFrame, sheet_name: str):
    """Render the path navigator interface with VM+Nodes approach."""
    st.subheader("🗺️ Path Navigator")
    st.markdown("Walk through the decision tree by selecting options at each level.")
    
    # Build nested options from tree summary
    by_parent = _calc_build_nested_options(df)
    
    # VM (Root) options - Node 1 under ROOT
    vm_opts = children_for(by_parent, 1, "<ROOT>")
    if not vm_opts:
        st.info("No VM (Root) options found in the decision tree.")
        return
    
    # VM (Root) selection
    vm = st.selectbox(
        f"{LEVEL_LABELS[0]}: Select root option",
        [""] + vm_opts,
        key="calc_vm",
        help="Choose the starting point for your path"
    )
    
    if vm:
        # Update path
        if len(st.session_state["calc_path"]) == 0:
            st.session_state["calc_path"] = [vm]
        else:
            st.session_state["calc_path"][0] = vm
            st.session_state["calc_path"] = st.session_state["calc_path"][:1]
        
        # Node 1 options based on VM choice
        n1_path = vm
        n1_opts = children_for(by_parent, 2, n1_path)
        n1 = st.selectbox(
            f"{LEVEL_LABELS[1]}: Select option",
            [""] + n1_opts,
            key="calc_n1",
            help="Choose Node 1 option"
        )
        
        if n1:
            # Update path
            if len(st.session_state["calc_path"]) < 2:
                st.session_state["calc_path"].append(n1)
            else:
                st.session_state["calc_path"][1] = n1
                st.session_state["calc_path"] = st.session_state["calc_path"][:2]
            
            # Node 2 options based on VM + Node 1
            n2_path = ">".join([vm, n1])
            n2_opts = children_for(by_parent, 3, n2_path)
            n2 = st.selectbox(
                f"{LEVEL_LABELS[2]}: Select option",
                [""] + n2_opts,
                key="calc_n2",
                help="Choose Node 2 option"
            )
            
            if n2:
                # Update path
                if len(st.session_state["calc_path"]) < 3:
                    st.session_state["calc_path"].append(n2)
                else:
                    st.session_state["calc_path"][2] = n2
                    st.session_state["calc_path"] = st.session_state["calc_path"][:3]
                
                # Node 3 options based on VM + Node 1 + Node 2
                n3_path = ">".join([vm, n1, n2])
                n3_opts = children_for(by_parent, 4, n3_path)
                n3 = st.selectbox(
                    f"{LEVEL_LABELS[3]}: Select option",
                    [""] + n3_opts,
                    key="calc_n3",
                    help="Choose Node 3 option"
                )
                
                if n3:
                    # Update path
                    if len(st.session_state["calc_path"]) < 4:
                        st.session_state["calc_path"].append(n3)
                    else:
                        st.session_state["calc_path"][3] = n3
                        st.session_state["calc_path"] = st.session_state["calc_path"][:4]
                    
                    # Node 4 options based on VM + Node 1 + Node 2 + Node 3
                    n4_path = ">".join([vm, n1, n2, n3])
                    n4_opts = children_for(by_parent, 5, n4_path)
                    n4 = st.selectbox(
                        f"{LEVEL_LABELS[4]}: Select option",
                        [""] + n4_opts,
                        key="calc_n4",
                        help="Choose Node 4 option"
                    )
                    
                    if n4:
                        # Update path
                        if len(st.session_state["calc_path"]) < 5:
                            st.session_state["calc_path"].append(n4)
                        else:
                            st.session_state["calc_path"][4] = n4
                            st.session_state["calc_path"] = st.session_state["calc_path"][:5]
                        
                        # Node 5 options based on full path
                        n5_path = ">".join([vm, n1, n2, n3, n4])
                        n5_opts = children_for(by_parent, 6, n5_path)
                        n5 = st.selectbox(
                            f"{LEVEL_LABELS[5]}: Select option",
                            [""] + n5_opts,
                            key="calc_n5",
                            help="Choose Node 5 option"
                        )
                        
                        if n5:
                            # Update path
                            if len(st.session_state["calc_path"]) < 6:
                                st.session_state["calc_path"].append(n5)
                            else:
                                st.session_state["calc_path"][5] = n5
                                st.session_state["calc_path"] = st.session_state["calc_path"][:6]
                                st.session_state["calc_path"][5] = n5
                                st.session_state["calc_path"] = st.session_state["calc_path"][:6]
    
    # Live Path Preview
    if st.session_state["calc_path"]:
        st.markdown("---")
        st.subheader("📍 Path Preview")
        path_display = " > ".join(st.session_state["calc_path"])
        st.info(f"**Current Path:** {path_display}")
        
        # Reset button
        if st.button("🔄 Reset Path", key="calc_reset"):
            st.session_state["calc_path"] = []
            safe_rerun()


def _render_path_results(df: pd.DataFrame, sheet_name: str):
    """Render the results for the selected path with row selection and CSV export."""
    st.subheader("📊 Path Results")
    
    if not st.session_state["calc_path"]:
        return
    
    # Build filter mask for the selected path
    filter_mask = _build_path_filter_mask(df, st.session_state["calc_path"])
    
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
