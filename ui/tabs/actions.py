# ui/tabs/actions.py
import streamlit as st
import pandas as pd
from typing import Dict, Any, List

from utils import (
    CANON_HEADERS, LEVEL_COLS, normalize_text, validate_headers
)


def render():
    """Render the Actions tab for managing action decisions and red flags."""
    try:
        st.header("⚡ Actions")

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

        # Main sections
        _render_actions_overview(df, sheet)
        
        st.markdown("---")
        
        _render_actions_management(df, sheet)
        
        st.markdown("---")
        
        _render_red_flags_section(df, sheet)

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


def _render_actions_overview(df: pd.DataFrame, sheet_name: str):
    """Render the actions overview section."""
    st.subheader("📊 Actions Overview")
    
    # Check if Actions column exists
    if "Actions" not in df.columns:
        st.warning("No 'Actions' column found in the sheet.")
        return
    
    # Get actions statistics
    actions_values = df["Actions"].map(normalize_text).dropna()
    actions_values = actions_values[actions_values != ""]
    
    if actions_values.empty:
        st.info("No actions defined in the current sheet.")
        return
    
    # Display statistics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        total_actions = len(actions_values)
        st.metric("Total Actions", total_actions)
    
    with col2:
        unique_actions = len(actions_values.unique())
        st.metric("Unique Action Types", unique_actions)
    
    with col3:
        coverage = (len(actions_values) / len(df)) * 100
        st.metric("Actions Coverage", f"{coverage:.1f}%")
    
    # Show actions breakdown
    st.write("**Actions breakdown:**")
    actions_counts = actions_values.value_counts()
    
    for action_type, count in actions_counts.items():
        percentage = (count / len(actions_values)) * 100
        st.write(f"• {action_type}: {count} ({percentage:.1f}%)")
    
    # Red flag analysis
    red_flag_count = actions_values.str.contains("red flag|urgent|emergency", case=False, na=False).sum()
    if red_flag_count > 0:
        st.warning(f"🚨 Found {red_flag_count} rows with red flag indicators")


def _render_actions_management(df: pd.DataFrame, sheet_name: str):
    """Render the actions management section."""
    st.subheader("✏️ Actions Management")
    
    # Actions editing options
    col1, col2 = st.columns(2)
    
    with col1:
        edit_mode = st.radio(
            "Edit mode",
            ["Individual rows", "Bulk operations", "Template-based"],
            help="Choose how to edit actions"
        )
    
    with col2:
        if edit_mode == "Individual rows":
            _render_individual_actions_editing(df, sheet_name)
        elif edit_mode == "Bulk operations":
            _render_bulk_actions_operations(df, sheet_name)
        else:  # Template-based
            _render_template_based_actions(df, sheet_name)


def _render_individual_actions_editing(df: pd.DataFrame, sheet_name: str):
    """Render individual actions editing interface."""
    st.write("**Individual Row Editing**")
    
    # Row selector
    if len(df) > 100:
        st.warning("Sheet has many rows. Consider using bulk operations for large datasets.")
        return
    
    # Show first 50 rows for editing
    display_df = df.head(50).copy()
    
    # Add row index for reference
    display_df["Row #"] = range(1, len(display_df) + 1)
    
    # Show editable actions column
    st.write("**Edit actions (first 50 rows):**")
    
    # Create a form for editing
    with st.form("actions_edit_form"):
        edited_actions = {}
        
        for idx, row in display_df.iterrows():
            current_action = normalize_text(row.get("Actions", ""))
            new_action = st.text_input(
                f"Row {idx + 1}: {row.get('Vital Measurement', 'N/A')}",
                value=current_action,
                key=f"actions_edit_{idx}",
                placeholder="Enter action..."
            )
            edited_actions[idx] = new_action
        
        if st.form_submit_button("💾 Save All Changes"):
            _save_actions_changes(df, edited_actions, sheet_name)


def _render_bulk_actions_operations(df: pd.DataFrame, sheet_name: str):
    """Render bulk actions operations interface."""
    st.write("**Bulk Actions Operations**")
    
    # Bulk operations
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📝 Set Actions by Pattern")
        
        # Pattern-based actions
        pattern_col = st.selectbox(
            "Match column",
            ["Vital Measurement"] + LEVEL_COLS,
            help="Select column to match against"
        )
        
        pattern_value = st.text_input(
            "Match value",
            placeholder="Enter value to match..."
        )
        
        action_value = st.text_input(
            "Action",
            placeholder="Enter action..."
        )
        
        if st.button("Apply to Matching Rows"):
            if pattern_value and action_value:
                _apply_bulk_actions(df, pattern_col, pattern_value, action_value, sheet_name)
            else:
                st.warning("Please fill in all fields.")
    
    with col2:
        st.subheader("🔄 Bulk Operations")
        
        # Clear all actions
        if st.button("🗑️ Clear All Actions"):
            if st.checkbox("Confirm clear all actions"):
                _clear_all_actions(df, sheet_name)
        
        # Copy actions from another column
        copy_from = st.selectbox(
            "Copy actions from",
            [""] + ["Vital Measurement"] + LEVEL_COLS,
            help="Copy values from another column as actions"
        )
        
        if copy_from and st.button("📋 Copy as Actions"):
            _copy_column_as_actions(df, copy_from, sheet_name)


def _render_template_based_actions(df: pd.DataFrame, sheet_name: str):
    """Render template-based actions interface."""
    st.write("**Template-Based Actions**")
    
    # Action templates
    templates = {
        "Standard": "Standard action required",
        "Red Flag": "🚨 RED FLAG - Immediate attention required",
        "Follow-up": "Schedule follow-up appointment",
        "Referral": "Refer to specialist",
        "Monitoring": "Continue monitoring",
        "Discharge": "Patient can be discharged"
    }
    
    # Template selector
    selected_template = st.selectbox(
        "Select action template",
        [""] + list(templates.keys())
    )
    
    if selected_template:
        template_action = templates[selected_template]
        st.write(f"**Template:** {template_action}")
        
        # Customize template
        custom_action = st.text_area(
            "Customize action (optional)",
            value=template_action,
            height=100,
            help="Modify the template action as needed"
        )
        
        # Apply template
        col1, col2 = st.columns(2)
        
        with col1:
            pattern_col = st.selectbox(
                "Match column",
                ["Vital Measurement"] + LEVEL_COLS,
                key="template_pattern_col"
            )
        
        with col2:
            pattern_value = st.text_input(
                "Match value",
                placeholder="Enter value to match...",
                key="template_pattern_value"
            )
        
        if st.button("Apply Template"):
            if pattern_value:
                _apply_bulk_actions(df, pattern_col, pattern_value, custom_action, sheet_name)
            else:
                st.warning("Please specify a pattern value.")


def _render_red_flags_section(df: pd.DataFrame, sheet_name: str):
    """Render the red flags section."""
    st.subheader("🚨 Red Flags Management")
    
    if "Actions" not in df.columns:
        st.info("No actions column to analyze for red flags.")
        return
    
    # Find red flag rows
    actions_col = df["Actions"].map(normalize_text)
    red_flag_mask = actions_col.str.contains("red flag|urgent|emergency|critical", case=False, na=False)
    red_flag_rows = df[red_flag_mask]
    
    if red_flag_rows.empty:
        st.success("✅ No red flags found in the current sheet.")
        return
    
    st.warning(f"🚨 Found {len(red_flag_rows)} rows with red flag indicators")
    
    # Show red flag details
    with st.expander("🔍 Red Flag Details", expanded=False):
        for idx, row in red_flag_rows.iterrows():
            vm = row.get("Vital Measurement", "N/A")
            action = row.get("Actions", "")
            st.write(f"**{vm}**: {action}")
    
    # Red flag management
    st.markdown("---")
    st.subheader("🛠️ Red Flag Management")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📊 Red Flag Report"):
            _generate_red_flag_report(df, red_flag_rows)
    
    with col2:
        if st.button("🔧 Bulk Red Flag Actions"):
            _bulk_red_flag_actions(df, sheet_name)


def _save_actions_changes(df: pd.DataFrame, edited_actions: Dict, sheet_name: str):
    """Save actions changes to the DataFrame."""
    try:
        changes_made = 0
        
        for row_idx, new_action in edited_actions.items():
            if new_action != normalize_text(df.loc[row_idx, "Actions"]):
                df.loc[row_idx, "Actions"] = new_action
                changes_made += 1
        
        if changes_made > 0:
            # Update the workbook in session state
            ctx = st.session_state.get("work_context", {})
            src = ctx.get("source")
            
            if src == "upload":
                wb = st.session_state.get("upload_workbook", {})
                wb[sheet_name] = df
                st.session_state["upload_workbook"] = wb
            else:
                wb = st.session_state.get("gs_workbook", {})
                wb[sheet_name] = df
                st.session_state["gs_workbook"] = wb
            
            st.success(f"Saved {changes_made} action changes!")
            st.rerun()
        else:
            st.info("No changes to save.")
            
    except Exception as e:
        st.error(f"Error saving action changes: {e}")


def _apply_bulk_actions(df: pd.DataFrame, pattern_col: str, pattern_value: str, action_value: str, sheet_name: str):
    """Apply action to rows matching a pattern."""
    try:
        if pattern_col not in df.columns:
            st.error(f"Column '{pattern_col}' not found.")
            return
        
        # Find matching rows
        mask = df[pattern_col].map(normalize_text) == normalize_text(pattern_value)
        matching_rows = mask.sum()
        
        if matching_rows == 0:
            st.warning(f"No rows found matching '{pattern_value}' in '{pattern_col}'.")
            return
        
        # Apply action
        df.loc[mask, "Actions"] = action_value
        
        # Update workbook
        _update_workbook(df, sheet_name)
        
        st.success(f"Applied action to {matching_rows} rows!")
        st.rerun()
        
    except Exception as e:
        st.error(f"Error applying bulk actions: {e}")


def _clear_all_actions(df: pd.DataFrame, sheet_name: str):
    """Clear all actions."""
    try:
        df["Actions"] = ""
        _update_workbook(df, sheet_name)
        st.success("Cleared all actions!")
        st.rerun()
        
    except Exception as e:
        st.error(f"Error clearing actions: {e}")


def _copy_column_as_actions(df: pd.DataFrame, source_col: str, sheet_name: str):
    """Copy values from another column as actions."""
    try:
        if source_col not in df.columns:
            st.error(f"Column '{source_col}' not found.")
            return
        
        # Copy values
        df["Actions"] = df[source_col]
        _update_workbook(df, sheet_name)
        
        st.success(f"Copied values from '{source_col}' as actions!")
        st.rerun()
        
    except Exception as e:
        st.error(f"Error copying column: {e}")


def _update_workbook(df: pd.DataFrame, sheet_name: str):
    """Update the workbook in session state."""
    try:
        ctx = st.session_state.get("work_context", {})
        src = ctx.get("source")
        
        if src == "upload":
            wb = st.session_state.get("upload_workbook", {})
            wb[sheet_name] = df
            st.session_state["upload_workbook"] = wb
        else:
            wb = st.session_state.get("gs_workbook", {})
            wb[sheet_name] = df
            st.session_state["gs_workbook"] = wb
            
    except Exception as e:
        st.error(f"Error updating workbook: {e}")


def _generate_red_flag_report(df: pd.DataFrame, red_flag_rows: pd.DataFrame):
    """Generate a comprehensive red flag report."""
    try:
        st.subheader("📊 Red Flag Report")
        
        # Summary statistics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            total_rows = len(df)
            st.metric("Total Rows", total_rows)
        
        with col2:
            red_flag_count = len(red_flag_rows)
            st.metric("Red Flag Rows", red_flag_count)
        
        with col3:
            red_flag_percentage = (red_flag_count / total_rows) * 100
            st.metric("Red Flag %", f"{red_flag_percentage:.1f}%")
        
        # Red flag by Vital Measurement
        st.write("**Red Flags by Vital Measurement:**")
        vm_red_flags = red_flag_rows.groupby("Vital Measurement").size().sort_values(ascending=False)
        
        for vm, count in vm_red_flags.items():
            st.write(f"• {vm}: {count} red flag(s)")
        
        # Red flag patterns
        st.write("**Red Flag Patterns:**")
        actions_col = red_flag_rows["Actions"].map(normalize_text)
        
        # Count different types of red flags
        red_flag_types = {
            "red flag": actions_col.str.contains("red flag", case=False, na=False).sum(),
            "urgent": actions_col.str.contains("urgent", case=False, na=False).sum(),
            "emergency": actions_col.str.contains("emergency", case=False, na=False).sum(),
            "critical": actions_col.str.contains("critical", case=False, na=False).sum()
        }
        
        for flag_type, count in red_flag_types.items():
            if count > 0:
                st.write(f"• {flag_type.title()}: {count}")
        
    except Exception as e:
        st.error(f"Error generating red flag report: {e}")


def _bulk_red_flag_actions(df: pd.DataFrame, sheet_name: str):
    """Perform bulk operations on red flag rows."""
    try:
        st.subheader("🔧 Bulk Red Flag Operations")
        
        # Find red flag rows
        actions_col = df["Actions"].map(normalize_text)
        red_flag_mask = actions_col.str.contains("red flag|urgent|emergency|critical", case=False, na=False)
        red_flag_rows = df[red_flag_mask]
        
        if red_flag_rows.empty:
            st.info("No red flags to process.")
            return
        
        st.write(f"Found {len(red_flag_rows)} red flag rows to process.")
        
        # Bulk operations
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📋 Standardize Red Flag Format"):
                _standardize_red_flags(df, red_flag_mask, sheet_name)
        
        with col2:
            if st.button("🔍 Add Red Flag Tags"):
                _add_red_flag_tags(df, red_flag_mask, sheet_name)
        
    except Exception as e:
        st.error(f"Error in bulk red flag operations: {e}")


def _standardize_red_flags(df: pd.DataFrame, red_flag_mask: pd.Series, sheet_name: str):
    """Standardize red flag formatting."""
    try:
        # Standardize red flag format
        df.loc[red_flag_mask, "Actions"] = df.loc[red_flag_mask, "Actions"].apply(
            lambda x: f"🚨 RED FLAG: {x}" if not x.startswith("🚨") else x
        )
        
        _update_workbook(df, sheet_name)
        st.success("Standardized red flag formatting!")
        st.rerun()
        
    except Exception as e:
        st.error(f"Error standardizing red flags: {e}")


def _add_red_flag_tags(df: pd.DataFrame, red_flag_mask: pd.Series, sheet_name: str):
    """Add red flag tags to actions."""
    try:
        # Add red flag tags
        df.loc[red_flag_mask, "Actions"] = df.loc[red_flag_mask, "Actions"].apply(
            lambda x: f"{x} [RED_FLAG]" if "[RED_FLAG]" not in x else x
        )
        
        _update_workbook(df, sheet_name)
        st.success("Added red flag tags!")
        st.rerun()
        
    except Exception as e:
        st.error(f"Error adding red flag tags: {e}")
