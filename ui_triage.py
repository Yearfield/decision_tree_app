# ui_triage.py

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
    from logic_triage import (
        filter_triage_data,
        compute_triage_metrics,
        validate_triage_data,
    )
    HAVE_TRIAGE_LOGIC = True
except ImportError:
    HAVE_TRIAGE_LOGIC = False


@st.cache_data(show_spinner=False, ttl=600)
def _cached_compute_triage_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    """Cached version of compute_triage_metrics to prevent recomputation."""
    if HAVE_TRIAGE_LOGIC and compute_triage_metrics:
        return compute_triage_metrics(df)
    else:
        # Fallback metrics calculation
        total_rows = len(df)
        triaged_rows = df["Diagnostic Triage"].notna().sum()
        triaged_rows = triaged_rows + (df["Diagnostic Triage"].astype(str).str.strip() != "").sum()
        coverage_pct = (triaged_rows / total_rows * 100) if total_rows > 0 else 0
        
        return {
            "total_rows": total_rows,
            "triaged_rows": triaged_rows,
            "coverage_pct": coverage_pct,
            "remaining": total_rows - triaged_rows
        }


def render(df: pd.DataFrame):
    """
    Render the Diagnostic Triage tab.
    
    Args:
        df: The current decision tree DataFrame (or None if not loaded).
    """
    st.header("🩺 Diagnostic Triage")
    
    if df is None or df.empty:
        st.warning("⚠️ No decision tree data loaded. Please upload or connect to a sheet.")
        return
    
    if not validate_headers(df):
        st.error("❌ Invalid data format. Expected canonical headers.")
        return
    
    # Check if Diagnostic Triage column exists
    if "Diagnostic Triage" not in df.columns:
        st.error("❌ 'Diagnostic Triage' column not found in the data.")
        return
    
    # Filter data for triage view
    triage_df = _prepare_triage_view(df)
    
    # Display summary metrics
    _render_triage_summary(triage_df)
    
    # Main triage interface
    _render_triage_editor(triage_df)
    
    # Triage notes form
    _render_triage_notes_form()


def _prepare_triage_view(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare DataFrame for triage view, filtering relevant columns.
    """
    # Select relevant columns for triage
    triage_columns = ["Vital Measurement"] + LEVEL_COLS + ["Diagnostic Triage"]
    triage_df = df[triage_columns].copy()
    
    # Filter out completely empty rows
    triage_df = triage_df.dropna(subset=["Vital Measurement"] + LEVEL_COLS, how="all")
    
    return triage_df


def _render_triage_summary(triage_df: pd.DataFrame):
    """
    Render summary metrics for triage data.
    """
    st.subheader("📊 Triage Overview")
    
    # Get cached metrics
    metrics = _cached_compute_triage_metrics(triage_df)
    
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Rows", metrics["total_rows"])
    
    with col2:
        st.metric("Triaged Rows", metrics["triaged_rows"])
    
    with col3:
        st.metric("Coverage", f"{metrics['coverage_pct']:.1f}%")
    
    with col4:
        st.metric("Remaining", metrics["remaining"])
    
    # Progress bar
    if metrics["total_rows"] > 0:
        st.progress(metrics["coverage_pct"] / 100)
        st.caption(f"Triage progress: {metrics['triaged_rows']}/{metrics['total_rows']} rows completed")


def _render_triage_editor(triage_df: pd.DataFrame):
    """
    Render the main triage data editor.
    """
    st.subheader("✏️ Edit Triage Data")
    
    # Show current triage data in editable format
    edited_triage_df = st.data_editor(
        triage_df,
        use_container_width=True,
        num_rows="dynamic",
        column_config={
            "Vital Measurement": st.column_config.TextColumn(
                "Vital Measurement",
                help="The vital measurement being assessed",
                disabled=True
            ),
            "Node 1": st.column_config.TextColumn(
                "Node 1",
                help="First decision node",
                disabled=True
            ),
            "Node 2": st.column_config.TextColumn(
                "Node 2", 
                help="Second decision node",
                disabled=True
            ),
            "Node 3": st.column_config.TextColumn(
                "Node 3",
                help="Third decision node", 
                disabled=True
            ),
            "Node 4": st.column_config.TextColumn(
                "Node 4",
                help="Fourth decision node",
                disabled=True
            ),
            "Node 5": st.column_config.TextColumn(
                "Node 5",
                help="Fifth decision node",
                disabled=True
            ),
            "Diagnostic Triage": st.column_config.TextColumn(
                "Diagnostic Triage",
                help="Triage priority and notes (e.g., 'High Priority', 'Urgent', 'Routine')",
                placeholder="Enter triage priority...",
                max_chars=None
            )
        }
    )
    
    # Check for changes
    if not triage_df.equals(edited_triage_df):
        st.session_state["triage_has_changes"] = True
        st.session_state["edited_triage_df"] = edited_triage_df
        st.success("✅ Changes detected. Use the save button below to persist changes.")
    else:
        st.session_state["triage_has_changes"] = False
    
    # Save controls
    _render_triage_save_controls(edited_triage_df)


def _render_triage_notes_form():
    """
    Render form for adding triage notes and comments.
    """
    st.subheader("📝 Triage Notes")
    
    with st.form("triage_notes_form"):
        triage_note = st.text_area(
            "Add general triage notes or comments:",
            placeholder="Enter notes about triage process, priorities, or general guidelines...",
            height=100
        )
        
        triage_priority = st.selectbox(
            "Default triage priority:",
            options=["", "High Priority", "Medium Priority", "Low Priority", "Urgent", "Routine"],
            help="Select a default priority level for new triage entries"
        )
        
        submitted = st.form_submit_button("💾 Save Notes", type="primary")
        
        if submitted:
            if triage_note.strip() or triage_priority:
                st.session_state["triage_notes"] = triage_note
                st.session_state["triage_priority"] = triage_priority
                st.success("✅ Triage notes saved successfully!")
            else:
                st.warning("⚠️ Please enter some notes or select a priority.")


def _render_triage_save_controls(edited_df: pd.DataFrame):
    """
    Render save controls for triage data.
    """
    st.subheader("💾 Save Changes")
    
    # Check if there are changes to save
    has_changes = st.session_state.get("triage_has_changes", False)
    
    if not has_changes:
        st.info("ℹ️ No changes to save.")
        return
    
    # Save mode selection
    save_mode = st.radio(
        "Save mode:",
        options=["overwrite", "append"],
        format_func=lambda x: "Overwrite existing data" if x == "overwrite" else "Append new data",
        horizontal=True,
        index=0
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Download backup
        csv_data = edited_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download CSV backup",
            data=csv_data,
            file_name="triage_data_backup.csv",
            mime="text/csv",
            help="Download current triage data as CSV before saving"
        )
    
    with col2:
        # Save button
        if st.button("💾 Save Triage Data", type="primary"):
            with st.spinner("💾 Saving triage data..."):
                success = _save_triage_data(edited_df, save_mode)
                if success:
                    st.success("✅ Triage data saved successfully!")
                    st.session_state["triage_has_changes"] = False
                    st.rerun()
                else:
                    st.error("❌ Failed to save triage data. Please try again.")


def _save_triage_data(edited_df: pd.DataFrame, save_mode: str) -> bool:
    """
    Save triage data back to the current workbook.
    
    Args:
        edited_df: The edited triage DataFrame
        save_mode: "overwrite" or "append"
        
    Returns:
        bool: True if save was successful
    """
    try:
        # Get current workbook
        wb_upload = st.session_state.get("upload_workbook", {})
        wb_gs = st.session_state.get("gs_workbook", {})
        
        if wb_upload:
            # Save to upload workbook
            current_sheet = st.session_state.get("current_sheet")
            if current_sheet and current_sheet in wb_upload:
                if save_mode == "overwrite":
                    # Update the triage column in the original DataFrame
                    original_df = wb_upload[current_sheet]
                    original_df["Diagnostic Triage"] = edited_df["Diagnostic Triage"]
                    wb_upload[current_sheet] = original_df
                else:
                    # Append mode - add new rows
                    wb_upload[current_sheet] = pd.concat([wb_upload[current_sheet], edited_df], ignore_index=True)
                
                st.session_state["upload_workbook"] = wb_upload
                return True
                
        elif wb_gs:
            # Save to Google Sheets
            current_sheet = st.session_state.get("current_sheet")
            spreadsheet_id = st.session_state.get("current_spreadsheet_id")
            
            if current_sheet and spreadsheet_id:
                from app.sheets import get_gspread_client_from_secrets, open_spreadsheet, write_dataframe
                
                client = get_gspread_client_from_secrets(st.secrets["gcp_service_account"])
                spreadsheet = open_spreadsheet(client, spreadsheet_id)
                
                if save_mode == "overwrite":
                    # Update the triage column
                    original_df = wb_gs[current_sheet]
                    original_df["Diagnostic Triage"] = edited_df["Diagnostic Triage"]
                    write_dataframe(spreadsheet, current_sheet, original_df, mode="overwrite")
                else:
                    # Append mode
                    write_dataframe(spreadsheet, current_sheet, edited_df, mode="append")
                
                # Update local copy
                wb_gs[current_sheet] = edited_df
                st.session_state["gs_workbook"] = wb_gs
                return True
        
        return False
        
    except Exception as e:
        st.error(f"❌ Error saving triage data: {str(e)}")
        return False
