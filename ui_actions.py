# ui_actions.py

import io
import json
from typing import Dict, List, Tuple, Optional, Set, Any
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

from utils import (
    CANON_HEADERS, LEVEL_COLS, MAX_LEVELS,
    normalize_text, validate_headers,
    friendly_parent_label, level_key_tuple,
)

# Import logic functions if they exist
try:
    from logic_actions import (
        filter_actions_data,
        compute_actions_metrics,
        validate_actions_data,
    )
    HAVE_ACTIONS_LOGIC = True
except ImportError:
    HAVE_ACTIONS_LOGIC = False

# Import sheets functions for Google Sheets integration
try:
    from app.sheets import get_gspread_client_from_secrets, open_spreadsheet, write_dataframe
    HAVE_SHEETS = True
except ImportError:
    HAVE_SHEETS = False


# Common helpers
def _ensure_cols(df, cols):
    import pandas as pd
    if df is None or df.empty:
        return pd.DataFrame({c: [] for c in cols})
    out = df.copy()
    for c in cols:
        if c not in out.columns:
            out[c] = ""
    return out


@st.cache_data(show_spinner=False, ttl=600)
def _actions_metrics(df):
    if df is None or df.empty:
        return {"total": 0, "filled": 0, "coverage_pct": 0.0, "missing": 0}
    total = len(df)
    filled = int(df["Actions"].astype(str).str.strip().ne("").sum())
    missing = total - filled
    pct = 0.0 if total == 0 else round(100.0*filled/total, 1)
    return {"total": total, "filled": filled, "coverage_pct": pct, "missing": missing}


def _apply_filters(df, vm_filter, q):
    from typing import List
    import pandas as pd
    if df is None or df.empty:
        return df
    view = df.copy()
    if vm_filter and vm_filter != "(All)":
        view = view[view["Vital Measurement"].astype(str).str.strip().str.lower() == str(vm_filter).strip().lower()]
    if q:
        ql = str(q).strip().lower()
        cols = ["Vital Measurement","Node 1","Node 2","Node 3","Node 4","Node 5"]
        mask = False
        for c in cols:
            if c in view.columns:
                mask = mask | view[c].astype(str).str.lower().str.contains(ql, na=False)
        view = view[mask]
    return view


@st.cache_data(show_spinner=False, ttl=600)
def _cached_compute_actions_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    """Cached version of compute_actions_metrics to prevent recomputation."""
    if HAVE_ACTIONS_LOGIC and compute_actions_metrics:
        return compute_actions_metrics(df)
    else:
        # Fallback metrics calculation
        total_rows = len(df)
        actions_rows = df["Actions"].notna().sum()
        actions_rows = actions_rows + (df["Actions"].astype(str).str.strip() != "").sum()
        coverage_pct = (actions_rows / total_rows * 100) if total_rows > 0 else 0
        
        # Analyze action types
        action_types = _analyze_action_types(df)
        
        return {
            "total_rows": total_rows,
            "actions_rows": actions_rows,
            "coverage_pct": coverage_pct,
            "remaining": total_rows - actions_rows,
            "action_types": action_types
        }


@st.cache_data(show_spinner=False, ttl=600)
def _cached_analyze_action_types(actions_df: pd.DataFrame) -> Dict[str, int]:
    """Cached version of _analyze_action_types to prevent recomputation."""
    return _analyze_action_types(actions_df)


def render(df: pd.DataFrame):
    """
    Render the Actions tab.
    
    Args:
        df: The current decision tree DataFrame (or None if not loaded).
    """
    st.header("⚡ Actions")
    
    if df is None or df.empty:
        st.warning("⚠️ No decision tree data loaded. Please upload or connect to a sheet.")
        return
    
    if not validate_headers(df):
        st.error("❌ Invalid data format. Expected canonical headers.")
        return
    
    # Ensure required columns exist
    required_cols = ["Vital Measurement","Node 1","Node 2","Node 3","Node 4","Node 5","Actions"]
    df = _ensure_cols(df, required_cols)
    
    # Filters row
    col1, col2 = st.columns([1, 2])
    with col1:
        vm_options = ["(All)"] + sorted(df["Vital Measurement"].astype(str).unique().tolist())
        vm_filter = st.selectbox("Vital Measurement", vm_options, key="actions_vm_filter")
    with col2:
        search_query = st.text_input("Search", placeholder="Search in all columns...", key="actions_search")
    
    # Apply filters
    filtered_df = _apply_filters(df, vm_filter, search_query)
    
    # Compute metrics
    metrics = _actions_metrics(filtered_df)
    
    # Show KPIs prominently
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total rows", metrics["total"])
    with col2:
        st.metric("Filled actions", metrics["filled"])
    with col3:
        st.metric("Coverage %", f"{metrics['coverage_pct']}%")
    
    # Progress bar
    st.progress(metrics["coverage_pct"]/100.0)
    
    st.markdown("---")
    
    # Editor
    if filtered_df.empty:
        st.info("ℹ️ No data matches the current filters.")
        return
    
    # Limit to first 100 rows for performance
    edit_df = filtered_df.head(100)
    
    # Column configuration
    column_config = {
        "Vital Measurement": st.column_config.TextColumn("Vital Measurement", disabled=True),
        "Node 1": st.column_config.TextColumn("Node 1", disabled=True),
        "Node 2": st.column_config.TextColumn("Node 2", disabled=True),
        "Node 3": st.column_config.TextColumn("Node 3", disabled=True),
        "Node 4": st.column_config.TextColumn("Node 4", disabled=True),
        "Node 5": st.column_config.TextColumn("Node 5", disabled=True),
        "Actions": st.column_config.TextColumn("Actions", max_chars=None),
    }
    
    # Data editor
    edited_df = st.data_editor(
        edit_df,
        column_config=column_config,
        use_container_width=True,
        key="actions_editor"
    )
    
    # Save controls
    st.markdown("---")
    col1, col2 = st.columns([1, 1])
    with col1:
        save_mode = st.radio("Save mode", ["Overwrite", "Append"], index=0, key="actions_save_mode")
    with col2:
        st.download_button(
            "Download CSV",
            data=filtered_df.to_csv(index=False).encode("utf-8"),
            file_name="actions_data.csv",
            mime="text/csv"
        )
    
    # Save to session button
    if st.button("💾 Save to session", key="actions_save_session"):
        if filtered_df.empty:
            st.warning("⚠️ No data to save.")
            return
        
        # Get current work context
        work_context = st.session_state.get("work_context", {})
        source = work_context.get("source")
        sheet_name = work_context.get("sheet")
        
        if not source or not sheet_name:
            st.error("❌ No active sheet context. Please select a sheet first.")
            return
        
        # Update the underlying DataFrame
        try:
            if source == "upload":
                wb = st.session_state.get("upload_workbook", {})
                if sheet_name in wb:
                    # Update rows in the original DataFrame
                    original_df = wb[sheet_name]
                    for idx, row in edited_df.iterrows():
                        # Find matching row in original DataFrame
                        mask = (
                            (original_df["Vital Measurement"] == row["Vital Measurement"]) &
                            (original_df["Node 1"] == row["Node 1"]) &
                            (original_df["Node 2"] == row["Node 2"]) &
                            (original_df["Node 3"] == row["Node 3"]) &
                            (original_df["Node 4"] == row["Node 4"]) &
                            (original_df["Node 5"] == row["Node 5"])
                        )
                        if mask.any():
                            original_df.loc[mask, "Actions"] = row["Actions"]
                    
                    wb[sheet_name] = original_df
                    st.session_state["upload_workbook"] = wb
                    st.success("✅ Actions data saved to session successfully!")
                    st.rerun()
                else:
                    st.error("❌ Sheet not found in upload workbook.")
            elif source == "gs":
                wb = st.session_state.get("gs_workbook", {})
                if sheet_name in wb:
                    # Similar update logic for Google Sheets workbook
                    original_df = wb[sheet_name]
                    for idx, row in edited_df.iterrows():
                        mask = (
                            (original_df["Vital Measurement"] == row["Vital Measurement"]) &
                            (original_df["Node 1"] == row["Node 1"]) &
                            (original_df["Node 2"] == row["Node 2"]) &
                            (original_df["Node 3"] == row["Node 3"]) &
                            (original_df["Node 4"] == row["Node 4"]) &
                            (original_df["Node 5"] == row["Node 5"])
                        )
                        if mask.any():
                            original_df.loc[mask, "Actions"] = row["Actions"]
                    
                    wb[sheet_name] = original_df
                    st.session_state["gs_workbook"] = wb
                    st.success("✅ Actions data saved to session successfully!")
                    st.rerun()
                else:
                    st.error("❌ Sheet not found in Google Sheets workbook.")
        except Exception as e:
            st.error(f"❌ Error saving to session: {str(e)}")
    
    # Push to Google Sheets button
    if HAVE_SHEETS and "gcp_service_account" in st.secrets:
        spreadsheet_id = st.session_state.get("gs_spreadsheet_id")
        if spreadsheet_id and sheet_name:
            if st.button("☁️ Push to Google Sheets", key="actions_push_gsheets"):
                if filtered_df.empty:
                    st.warning("⚠️ No data to push.")
                    return
                
                try:
                    with st.spinner("Pushing to Google Sheets..."):
                        client = get_gspread_client_from_secrets(st.secrets["gcp_service_account"])
                        spreadsheet = open_spreadsheet(client, spreadsheet_id)
                        
                        # Get the full DataFrame for pushing
                        full_df = st.session_state.get("gs_workbook", {}).get(sheet_name, pd.DataFrame())
                        if full_df.empty:
                            st.error("❌ No data available for pushing.")
                            return
                        
                        write_dataframe(spreadsheet, sheet_name, full_df, mode=save_mode.lower())
                        st.success("✅ Successfully pushed to Google Sheets!")
                        st.rerun()
                except Exception as e:
                    st.error(f"❌ Error pushing to Google Sheets: {str(e)}")


def _prepare_actions_view(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare DataFrame for actions view, filtering relevant columns.
    """
    # Select relevant columns for actions
    actions_columns = ["Vital Measurement"] + LEVEL_COLS + ["Actions"]
    actions_df = df[actions_columns].copy()
    
    # Filter out completely empty rows
    actions_df = actions_df.dropna(subset=["Vital Measurement"] + LEVEL_COLS, how="all")
    
    return actions_df


def _render_actions_summary(actions_df: pd.DataFrame):
    """
    Render summary metrics for actions data.
    """
    st.subheader("📊 Actions Overview")
    
    # Get cached metrics
    metrics = _cached_compute_actions_metrics(actions_df)
    
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Rows", metrics["total_rows"])
    
    with col2:
        st.metric("With Actions", metrics["actions_rows"])
    
    with col3:
        st.metric("Coverage", f"{metrics['coverage_pct']:.1f}%")
    
    with col4:
        st.metric("Missing Actions", metrics["remaining"])
    
    # Progress bar
    if metrics["total_rows"] > 0:
        st.progress(metrics["coverage_pct"] / 100)
        st.caption(f"Actions coverage: {metrics['actions_rows']}/{metrics['total_rows']} rows have actions")
    
    # Action types breakdown
    if metrics["action_types"]:
        st.subheader("🔍 Action Types Breakdown")
        action_df = pd.DataFrame(metrics["action_types"].items(), columns=["Action Type", "Count"])
        # Limit preview to first 100 rows for speed
        st.dataframe(action_df.head(100), use_container_width=True)


