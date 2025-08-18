# ui_actions.py

import io
import json
from typing import Dict, List, Tuple, Optional, Set
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


def render(df: pd.DataFrame):
    """
    Render the Actions tab.
    
    Args:
        df: The current decision tree DataFrame (or None if not loaded).
    """
    st.header("⚡ Actions")
    
    if df is None or df.empty:
        st.warning("No decision tree data loaded. Please upload or connect to a sheet.")
        return
    
    if not validate_headers(df):
        st.error("❌ Invalid data format. Expected canonical headers.")
        return
    
    # Check if Actions column exists
    if "Actions" not in df.columns:
        st.error("❌ 'Actions' column not found in the data.")
        return
    
    # Filter data for actions view
    actions_df = _prepare_actions_view(df)
    
    # Display summary metrics
    _render_actions_summary(actions_df)
    
    # Main actions interface
    _render_actions_editor(actions_df)
    
    # Add new actions form
    _render_add_actions_form()


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
    
    # Calculate metrics
    total_rows = len(actions_df)
    actions_rows = actions_df["Actions"].notna().sum()
    actions_rows = actions_rows + (actions_df["Actions"].astype(str).str.strip() != "").sum()
    coverage_pct = (actions_rows / total_rows * 100) if total_rows > 0 else 0
    
    # Count action types
    action_types = _analyze_action_types(actions_df)
    
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Rows", total_rows)
    
    with col2:
        st.metric("With Actions", actions_rows)
    
    with col3:
        st.metric("Coverage", f"{coverage_pct:.1f}%")
    
    with col4:
        remaining = total_rows - actions_rows
        st.metric("Missing Actions", remaining)
    
    # Progress bar
    if total_rows > 0:
        st.progress(actions_rows / total_rows)
        st.caption(f"Actions coverage: {actions_rows}/{total_rows} rows have actions")
    
    # Action types breakdown
    if action_types:
        st.subheader("🔍 Action Types Breakdown")
        action_df = pd.DataFrame(action_types.items(), columns=["Action Type", "Count"])
        st.dataframe(action_df, use_container_width=True)


def _analyze_action_types(actions_df: pd.DataFrame) -> Dict[str, int]:
    """
    Analyze the types of actions present in the data.
    """
    action_types = {}
    
    for _, row in actions_df.iterrows():
        actions = normalize_text(row.get("Actions", ""))
        if actions:
            # Split by common delimiters and count types
            action_list = [a.strip() for a in actions.replace(";", ",").replace("|", ",").split(",") if a.strip()]
            for action in action_list:
                # Extract action type (first word or common prefixes)
                action_type = action.split()[0] if action else "Unknown"
                action_types[action_type] = action_types.get(action_type, 0) + 1
    
    return dict(sorted(action_types.items(), key=lambda x: x[1], reverse=True))


def _render_actions_editor(actions_df: pd.DataFrame):
    """
    Render the main actions data editor.
    """
    st.subheader("✏️ Edit Actions Data")
    
    # Show current actions data in editable format
    edited_actions_df = st.data_editor(
        actions_df,
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
            "Actions": st.column_config.TextColumn(
                "Actions",
                help="Actions to take (e.g., 'Refer to specialist', 'Order tests', 'Monitor')",
                placeholder="Enter actions..."
            )
        }
    )
    
    # Check for changes
    if not actions_df.equals(edited_actions_df):
        st.session_state["actions_has_changes"] = True
        st.session_state["edited_actions_df"] = edited_actions_df
        st.success("✅ Changes detected. Use the save button below to persist changes.")
    else:
        st.session_state["actions_has_changes"] = False
    
    # Save controls
    _render_actions_save_controls(edited_actions_df)


def _render_add_actions_form():
    """
    Render form for adding new actions.
    """
    st.subheader("➕ Add New Actions")
    
    with st.form("add_actions_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            action_label = st.text_input(
                "Action label:",
                placeholder="e.g., 'Refer to specialist'",
                help="Enter a descriptive action label"
            )
            
            action_category = st.selectbox(
                "Action category:",
                options=["", "Diagnostic", "Treatment", "Follow-up", "Referral", "Monitoring", "Other"],
                help="Categorize the action type"
            )
        
        with col2:
            action_priority = st.selectbox(
                "Priority:",
                options=["", "High", "Medium", "Low", "Urgent"],
                help="Set the priority level for this action"
            )
            
            action_duration = st.text_input(
                "Duration/Timeframe:",
                placeholder="e.g., 'Within 24 hours'",
                help="When this action should be completed"
            )
        
        action_description = st.text_area(
            "Detailed description:",
            placeholder="Provide additional details about this action...",
            height=80
        )
        
        submitted = st.form_submit_button("➕ Add Action", type="primary")
        
        if submitted:
            if action_label.strip():
                # Store the new action for potential use
                new_action = {
                    "label": action_label,
                    "category": action_category,
                    "priority": action_priority,
                    "duration": action_duration,
                    "description": action_description
                }
                
                # Add to session state for potential use
                if "new_actions" not in st.session_state:
                    st.session_state["new_actions"] = []
                st.session_state["new_actions"].append(new_action)
                
                st.success(f"✅ Action '{action_label}' added successfully!")
                st.info("💡 Use the action label in the Actions column above to apply this action to specific rows.")
            else:
                st.warning("⚠️ Please enter an action label.")


def _render_actions_save_controls(edited_df: pd.DataFrame):
    """
    Render save controls for actions data.
    """
    st.subheader("💾 Save Changes")
    
    # Check if there are changes to save
    has_changes = st.session_state.get("actions_has_changes", False)
    
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
            file_name="actions_data_backup.csv",
            mime="text/csv",
            help="Download current actions data as CSV before saving"
        )
    
    with col2:
        # Save button
        if st.button("💾 Save Actions Data", type="primary"):
            with st.spinner("💾 Saving actions data..."):
                success = _save_actions_data(edited_df, save_mode)
                if success:
                    st.success("✅ Actions data saved successfully!")
                    st.session_state["actions_has_changes"] = False
                    st.rerun()
                else:
                    st.error("❌ Failed to save actions data. Please try again.")


def _save_actions_data(edited_df: pd.DataFrame, save_mode: str) -> bool:
    """
    Save actions data back to the current workbook.
    
    Args:
        edited_df: The edited actions DataFrame
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
                    # Update the actions column in the original DataFrame
                    original_df = wb_upload[current_sheet]
                    original_df["Actions"] = edited_df["Actions"]
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
                    # Update the actions column
                    original_df = wb_gs[current_sheet]
                    original_df["Actions"] = edited_df["Actions"]
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
        st.error(f"❌ Error saving actions data: {str(e)}")
        return False
