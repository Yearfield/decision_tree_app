# ui/tabs/workspace.py
import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, Any

from utils import (
    CANON_HEADERS, LEVEL_COLS, normalize_text, validate_headers
)


def render():
    """Render the Workspace Selection tab for choosing and previewing sheets."""
    try:
        st.header("🗂 Workspace Selection")
        st.markdown("Choose a sheet to work with and preview its contents.")

        # Get active DataFrame
        df = get_active_df()
        if df is None:
            st.warning("No active sheet selected. Please load a workbook in the Source tab and select a sheet.")
            return
        
        if not validate_headers(df):
            st.warning("Active sheet has invalid headers. Please ensure it has the required columns.")
            return

        # Get sheet name from context
        ctx = st.session_state.get("work_context", {})
        sheet = ctx.get("sheet", "Unknown")

        # Summary + preview
        st.write(f"Active sheet: **{sheet}** ({len(df)} rows)")
        
        # Calculate metrics
        ok_p, total_p = _compute_parent_depth_score_counts(df)
        ok_r, total_r = _compute_row_path_score_counts(df)
        
        p1, p2 = st.columns(2)
        with p1:
            st.metric("Parents with 5 children", f"{ok_p}/{total_p}")
            st.progress(0 if total_p == 0 else ok_p / total_p)
        with p2:
            st.metric("Rows with full path", f"{ok_r}/{total_r}")
            st.progress(0 if total_r == 0 else ok_r / total_r)

        # Preview section
        _render_preview_section(df, sheet)

        # Group rows controls
        st.markdown("---")
        _render_grouping_controls(df)

    except Exception as e:
        st.exception(e)


def get_active_df():
    """Get the currently active DataFrame from session state."""
    wb_u = st.session_state.get("upload_workbook", {})
    wb_g = st.session_state.get("gs_workbook", {})
    ctx = st.session_state.get("work_context", {})
    sheet = ctx.get("sheet")
    if sheet and sheet in wb_u: 
        return wb_u[sheet]
    if sheet and sheet in wb_g: 
        return wb_g[sheet]
    return None


def _compute_parent_depth_score_counts(df: pd.DataFrame) -> tuple[int, int]:
    """Compute parent depth score counts."""
    try:
        total = 0
        ok = 0
        
        for level in range(1, 5):  # Check levels 1-4 for having 5 children
            if f"Node {level}" not in df.columns or f"Node {level + 1}" not in df.columns:
                continue
                
            # Group by parent path
            parent_cols = [f"Node {i}" for i in range(1, level + 1)]
            if not all(col in df.columns for col in parent_cols):
                continue
                
            # Count unique parent paths
            parent_paths = df[parent_cols].apply(lambda r: tuple(normalize_text(v) for v in r), axis=1)
            parent_paths = parent_paths[parent_paths.apply(lambda x: all(v != "" for v in x))]
            unique_parents = parent_paths.unique()
            
            for parent in unique_parents:
                total += 1
                # Count children for this parent
                mask = parent_paths == parent
                children = df.loc[mask, f"Node {level + 1}"].map(normalize_text)
                children = children[children != ""]
                if len(children.unique()) == 5:
                    ok += 1
                    
        return ok, total
    except Exception:
        return 0, 0


def _compute_row_path_score_counts(df: pd.DataFrame) -> tuple[int, int]:
    """Compute row path score counts."""
    try:
        total = len(df)
        ok = 0
        
        for _, row in df.iterrows():
            path_complete = True
            for col in LEVEL_COLS:
                if col in df.columns:
                    if normalize_text(row.get(col, "")) == "":
                        path_complete = False
                        break
            if path_complete:
                ok += 1
                
        return ok, total
    except Exception:
        return 0, 0


def _render_preview_section(df: pd.DataFrame, sheet_name: str):
    """Render the preview section with pagination."""
    total_rows = len(df)
    st.markdown("#### Preview (50 rows)")
    
    if total_rows <= 50:
        st.caption(f"Showing all {total_rows} rows.")
        st.dataframe(df, use_container_width=True)
    else:
        state_key = f"preview_start_{sheet_name}"
        start_idx = int(st.session_state.get(state_key, 0))
        
        cprev, cnum, cnext = st.columns([1, 2, 1])
        with cprev:
            if st.button("◀ Previous 50", key=f"prev50_{sheet_name}"):
                start_idx = max(0, start_idx - 50)
        with cnum:
            start_1based = st.number_input(
                "Start row (1-based)",
                min_value=1,
                max_value=max(1, total_rows - 49),
                value=start_idx + 1,
                step=50,
                help="Pick where to start the 50-row preview.",
                key=f"startnum_{sheet_name}"
            )
            start_idx = int(start_1based) - 1
        with cnext:
            if st.button("Next 50 ▶", key=f"next50_{sheet_name}"):
                start_idx = min(max(0, total_rows - 50), start_idx + 50)
        
        st.session_state[state_key] = start_idx
        end_idx = min(start_idx + 50, total_rows)
        st.caption(f"Showing rows **{start_idx + 1}–{end_idx}** of **{total_rows}**.")
        st.dataframe(df.iloc[start_idx:end_idx], use_container_width=True)


def _render_grouping_controls(df: pd.DataFrame):
    """Render the grouping controls section."""
    with st.expander("🧩 Group rows (cluster identical labels together)"):
        st.caption("Group rows so identical **Node 1** and **Node 2** values are contiguous.")
        
        if df.empty or not validate_headers(df):
            st.info("Load a valid sheet first.")
            return
            
        scope = st.radio("Grouping scope", ["Whole sheet", "Within Vital Measurement"], horizontal=True, key="ws_group_scope_sel")
        rf_scope = st.radio("Red-Flag priority", ["Node 2 only", "Any node (Dictionary)"], horizontal=True, key="ws_group_rf_scope")
        preview = st.checkbox("Show preview (does not modify data)", value=True, key="ws_group_preview_sel")

        if preview:
            grouped_df = _grouped_df(df, scope, rf_scope)
            st.dataframe(grouped_df.head(100), use_container_width=True)
            st.caption(f"Showing first 100 rows of grouped data. Total: {len(grouped_df)} rows.")


def _grouped_df(df: pd.DataFrame, scope_mode: str, rf_scope_sel: str) -> pd.DataFrame:
    """Create a grouped DataFrame based on the specified parameters."""
    try:
        if rf_scope_sel == "Node 2 only":
            # Simple sort by Node 1, Node 2
            df2 = df.sort_values(["Node 1", "Node 2"], kind="stable")
        else:
            # Sort by all nodes
            sort_cols = ["Vital Measurement"] + LEVEL_COLS
            sort_cols = [col for col in sort_cols if col in df.columns]
            df2 = df.sort_values(sort_cols, kind="stable")
            
        if scope_mode == "Within Vital Measurement":
            df2["_vm"] = df2["Vital Measurement"].map(normalize_text)
            df2["_row"] = np.arange(len(df2))
            df2 = df2.sort_values(["_vm", "Node 1", "Node 2", "Node 3", "Node 4", "Node 5", "_row"], kind="stable").drop(columns=["_vm", "_row"])
            
        return df2
    except Exception:
        return df
