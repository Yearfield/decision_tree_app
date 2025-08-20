# ui/tabs/calculator.py
import streamlit as st
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional

from utils import (
    CANON_HEADERS, LEVEL_COLS, normalize_text, validate_headers
)
from utils.state import (
    get_active_workbook, get_current_sheet, get_active_df, 
    has_active_workbook, get_workbook_status
)
from ui.utils.rerun import safe_rerun
from logic.tree import infer_branch_options


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


def _render_path_navigator(df: pd.DataFrame, sheet_name: str):
    """Render the path navigator interface."""
    st.subheader("🗺️ Path Navigator")
    st.markdown("Walk through the decision tree by selecting options at each level.")
    
    # Get branch options from logic.tree (cached)
    from utils.state import get_wb_nonce
    store = _get_cached_branch_options(df, get_wb_nonce())
    
    # Level 1: Root level options
    level1_options = store.get("L1|", [])
    if not level1_options:
        st.info("No Level 1 options found in the decision tree.")
        return
    
    # Level 1 selection
    level1_choice = st.selectbox(
        "Level 1: Select root option",
        [""] + level1_options,
        key="calc_level1",
        help="Choose the starting point for your path"
    )
    
    if level1_choice:
        # Update path
        if len(st.session_state["calc_path"]) == 0:
            st.session_state["calc_path"] = [level1_choice]
        else:
            st.session_state["calc_path"][0] = level1_choice
            st.session_state["calc_path"] = st.session_state["calc_path"][:1]
        
        # Level 2: Options based on Level 1 choice
        level2_key = f"L2|{level1_choice}"
        level2_options = store.get(level2_key, [])
        
        if level2_options:
            level2_choice = st.selectbox(
                "Level 2: Select option",
                [""] + level2_options,
                key="calc_level2",
                help="Choose the second level option"
            )
            
            if level2_choice:
                # Update path
                if len(st.session_state["calc_path"]) < 2:
                    st.session_state["calc_path"].append(level2_choice)
                else:
                    st.session_state["calc_path"][1] = level2_choice
                    st.session_state["calc_path"] = st.session_state["calc_path"][:2]
                
                # Level 3: Options based on Level 1 + Level 2
                level3_key = f"L3|{level1_choice}>{level2_choice}"
                level3_options = store.get(level3_key, [])
                
                if level3_options:
                    level3_choice = st.selectbox(
                        "Level 3: Select option",
                        [""] + level3_options,
                        key="calc_level3",
                        help="Choose the third level option"
                    )
                    
                    if level3_choice:
                        # Update path
                        if len(st.session_state["calc_path"]) < 3:
                            st.session_state["calc_path"].append(level3_choice)
                        else:
                            st.session_state["calc_path"][2] = level3_choice
                            st.session_state["calc_path"] = st.session_state["calc_path"][:3]
                        
                        # Level 4: Options based on Level 1 + Level 2 + Level 3
                        level4_key = f"L4|{level1_choice}>{level2_choice}>{level3_choice}"
                        level4_options = store.get(level4_key, [])
                        
                        if level4_options:
                            level4_choice = st.selectbox(
                                "Level 4: Select option",
                                [""] + level4_options,
                                key="calc_level4",
                                help="Choose the fourth level option"
                            )
                            
                            if level4_choice:
                                # Update path
                                if len(st.session_state["calc_path"]) < 4:
                                    st.session_state["calc_path"].append(level4_choice)
                                else:
                                    st.session_state["calc_path"][3] = level4_choice
                                    st.session_state["calc_path"] = st.session_state["calc_path"][:4]
                                
                                # Level 5: Options based on Level 1 + Level 2 + Level 3 + Level 4
                                level5_key = f"L5|{level1_choice}>{level2_choice}>{level3_choice}>{level4_choice}"
                                level5_options = store.get(level5_key, [])
                                
                                if level5_options:
                                    level5_choice = st.selectbox(
                                        "Level 5: Select option",
                                        [""] + level5_options,
                                        key="calc_level5",
                                        help="Choose the fifth level option"
                                    )
                                    
                                    if level5_choice:
                                        # Update path
                                        if len(st.session_state["calc_path"]) < 5:
                                            st.session_state["calc_path"].append(level5_choice)
                                        else:
                                            st.session_state["calc_path"][4] = level5_choice
                                            st.session_state["calc_path"] = st.session_state["calc_path"][:5]
    
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
    """Render the results for the selected path."""
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
    
    # Display results
    if len(matching_rows) <= 50:
        st.write(f"**All {len(matching_rows)} matching rows:**")
        _display_path_results_table(matching_rows)
    else:
        st.write(f"**Showing top 50 of {len(matching_rows)} matching rows:**")
        _display_path_results_table(matching_rows.head(50))
        st.info(f"... and {len(matching_rows) - 50} more rows. Use filters to narrow down results.")


def _build_path_filter_mask(df: pd.DataFrame, path: List[str]) -> Optional[pd.Series]:
    """Build a filter mask for the selected path."""
    try:
        if not path:
            return None
        
        # Start with all rows as True
        mask = pd.Series([True] * len(df), index=df.index)
        
        # Apply filter for each level in the path
        for level, value in enumerate(path, 1):
            col_name = f"Node {level}"
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
