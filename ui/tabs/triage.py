# ui/tabs/triage.py
import streamlit as st
import pandas as pd
from typing import Dict, Any, List

from utils import (
    CANON_HEADERS, LEVEL_COLS, normalize_text, validate_headers
)
from utils.state import (
    get_active_workbook, get_current_sheet, get_active_df, 
    has_active_workbook, get_workbook_status, set_active_workbook
)
from ui.utils.rerun import safe_rerun


def render():
    """Render the Diagnostic Triage tab for managing triage decisions."""
    try:
        st.header("🩺 Diagnostic Triage")
        
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

        # Main sections
        _render_triage_overview(df, sheet)
        
        st.markdown("---")
        
        _render_triage_management(df, sheet)
        
        st.markdown("---")
        
        _render_triage_analysis(df, sheet)

    except Exception as e:
        st.exception(e)


def _render_triage_overview(df: pd.DataFrame, sheet_name: str):
    """Render the triage overview section."""
    st.subheader("📊 Triage Overview")
    
    # Check if Diagnostic Triage column exists
    if "Diagnostic Triage" not in df.columns:
        st.warning("No 'Diagnostic Triage' column found in the sheet.")
        return
    
    # Get triage statistics
    triage_values = df["Diagnostic Triage"].map(normalize_text).dropna()
    triage_values = triage_values[triage_values != ""]
    
    if triage_values.empty:
        st.info("No triage decisions found in the current sheet.")
        return
    
    # Display statistics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        total_triage = len(triage_values)
        st.metric("Total Triage Decisions", total_triage)
    
    with col2:
        unique_triage = len(triage_values.unique())
        st.metric("Unique Triage Types", unique_triage)
    
    with col3:
        coverage = (len(triage_values) / len(df)) * 100
        st.metric("Triage Coverage", f"{coverage:.1f}%")
    
    # Show triage breakdown
    st.write("**Triage breakdown:**")
    triage_counts = triage_values.value_counts()
    
    for triage_type, count in triage_counts.items():
        percentage = (count / len(triage_values)) * 100
        st.write(f"• {triage_type}: {count} ({percentage:.1f}%)")


def _render_triage_management(df: pd.DataFrame, sheet_name: str):
    """Render the triage management section."""
    st.subheader("✏️ Triage Management")
    
    # Triage editing options
    col1, col2 = st.columns(2)
    
    with col1:
        edit_mode = st.radio(
            "Edit mode",
            ["Individual rows", "Bulk operations"],
            help="Choose how to edit triage decisions"
        )
    
    with col2:
        if edit_mode == "Individual rows":
            _render_individual_triage_editing(df, sheet_name)
        else:
            _render_bulk_triage_operations(df, sheet_name)


def _render_individual_triage_editing(df: pd.DataFrame, sheet_name: str):
    """Render individual triage editing interface."""
    st.write("**Individual Row Editing**")
    
    # Row selector
    if len(df) > 100:
        st.warning("Sheet has many rows. Consider using bulk operations for large datasets.")
        return
    
    # Show first 50 rows for editing
    display_df = df.head(50).copy()
    
    # Add row index for reference
    display_df["Row #"] = range(1, len(display_df) + 1)
    
    # Show editable triage column
    st.write("**Edit triage decisions (first 50 rows):**")
    
    # Create a form for editing
    with st.form("triage_edit_form"):
        edited_triage = {}
        
        for idx, row in display_df.iterrows():
            current_triage = normalize_text(row.get("Diagnostic Triage", ""))
            new_triage = st.text_input(
                f"Row {idx + 1}: {row.get('Vital Measurement', 'N/A')}",
                value=current_triage,
                key=f"triage_edit_{idx}",
                placeholder="Enter triage decision..."
            )
            edited_triage[idx] = new_triage
        
        if st.form_submit_button("💾 Save All Changes"):
            _save_triage_changes(df, edited_triage, sheet_name)


def _render_bulk_triage_operations(df: pd.DataFrame, sheet_name: str):
    """Render bulk triage operations interface."""
    st.write("**Bulk Triage Operations**")
    
    # Bulk operations
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📝 Set Triage by Pattern")
        
        # Pattern-based triage
        pattern_col = st.selectbox(
            "Match column",
            ["Vital Measurement"] + LEVEL_COLS,
            help="Select column to match against"
        )
        
        pattern_value = st.text_input(
            "Match value",
            placeholder="Enter value to match..."
        )
        
        triage_decision = st.text_input(
            "Triage decision",
            placeholder="Enter triage decision..."
        )
        
        if st.button("Apply to Matching Rows"):
            if pattern_value and triage_decision:
                _apply_bulk_triage(df, pattern_col, pattern_value, triage_decision, sheet_name)
            else:
                st.warning("Please fill in all fields.")
    
    with col2:
        st.subheader("🔄 Bulk Operations")
        
        # Clear all triage
        if st.button("🗑️ Clear All Triage"):
            if st.checkbox("Confirm clear all triage decisions"):
                _clear_all_triage(df, sheet_name)
        
        # Copy triage from another column
        copy_from = st.selectbox(
            "Copy triage from",
            [""] + ["Vital Measurement"] + LEVEL_COLS,
            help="Copy values from another column as triage decisions"
        )
        
        if copy_from and st.button("📋 Copy as Triage"):
            _copy_column_as_triage(df, copy_from, sheet_name)


def _render_triage_analysis(df: pd.DataFrame, sheet_name: str):
    """Render the triage analysis section."""
    st.subheader("🔍 Triage Analysis")
    
    if "Diagnostic Triage" not in df.columns:
        st.info("No triage data to analyze.")
        return
    
    # Analysis options
    analysis_type = st.selectbox(
        "Analysis type",
        ["Triage patterns", "Missing triage", "Triage quality"],
        help="Choose what to analyze"
    )
    
    if analysis_type == "Triage patterns":
        _analyze_triage_patterns(df)
    elif analysis_type == "Missing triage":
        _analyze_missing_triage(df)
    else:  # Triage quality
        _analyze_triage_quality(df)


def _save_triage_changes(df: pd.DataFrame, edited_triage: Dict, sheet_name: str):
    """Save triage changes to the DataFrame."""
    try:
        changes_made = 0
        
        for row_idx, new_triage in edited_triage.items():
            if new_triage != normalize_text(df.loc[row_idx, "Diagnostic Triage"]):
                df.loc[row_idx, "Diagnostic Triage"] = new_triage
                changes_made += 1
        
        if changes_made > 0:
            # Update the active workbook using canonical API
            active_wb = get_active_workbook()
            if active_wb and sheet_name in active_wb:
                active_wb[sheet_name] = df
                set_active_workbook(active_wb, source="triage_editor")
                
                # Clear stale caches to ensure immediate refresh
                st.cache_data.clear()
                
                st.success(f"Saved {changes_made} triage changes!")
                safe_rerun()
            else:
                st.error("Could not update active workbook.")
        else:
            st.info("No changes to save.")
            
    except Exception as e:
        st.error(f"Error saving triage changes: {e}")


def _apply_bulk_triage(df: pd.DataFrame, pattern_col: str, pattern_value: str, triage_decision: str, sheet_name: str):
    """Apply triage decision to rows matching a pattern."""
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
        
        # Apply triage
        df.loc[mask, "Diagnostic Triage"] = triage_decision
        
        # Update workbook
        _update_workbook(df, sheet_name)
        
        st.success(f"Applied triage decision to {matching_rows} rows!")
        safe_rerun()
        
    except Exception as e:
        st.error(f"Error applying bulk triage: {e}")


def _clear_all_triage(df: pd.DataFrame, sheet_name: str):
    """Clear all triage decisions."""
    try:
        df["Diagnostic Triage"] = ""
        _update_workbook(df, sheet_name)
        st.success("Cleared all triage decisions!")
        safe_rerun()
        
    except Exception as e:
        st.error(f"Error clearing triage: {e}")


def _copy_column_as_triage(df: pd.DataFrame, source_col: str, sheet_name: str):
    """Copy values from another column as triage decisions."""
    try:
        if source_col not in df.columns:
            st.error(f"Column '{source_col}' not found.")
            return
        
        # Copy values
        df["Diagnostic Triage"] = df[source_col]
        _update_workbook(df, sheet_name)
        
        st.success(f"Copied values from '{source_col}' as triage decisions!")
        safe_rerun()
        
    except Exception as e:
        st.error(f"Error copying column: {e}")


def _update_workbook(df: pd.DataFrame, sheet_name: str):
    """Update the workbook in session state using canonical API."""
    try:
        from utils.state import get_active_workbook, set_active_workbook
        active_wb = get_active_workbook()
        if active_wb and sheet_name in active_wb:
            active_wb[sheet_name] = df
            set_active_workbook(active_wb, source="triage_bulk")
            
            # Clear stale caches to ensure immediate refresh
            st.cache_data.clear()
        else:
            st.error("Could not update active workbook.")
            
    except Exception as e:
        st.error(f"Error updating workbook: {e}")


def _analyze_triage_patterns(df: pd.DataFrame):
    """Analyze triage patterns in the data."""
    try:
        triage_values = df["Diagnostic Triage"].map(normalize_text).dropna()
        triage_values = triage_values[triage_values != ""]
        
        if triage_values.empty:
            st.info("No triage data to analyze.")
            return
        
        # Show patterns
        st.write("**Triage patterns by Vital Measurement:**")
        
        vm_triage = df.groupby("Vital Measurement")["Diagnostic Triage"].apply(
            lambda x: x.map(normalize_text).dropna().value_counts().to_dict()
        )
        
        for vm, triage_counts in vm_triage.items():
            if triage_counts:
                with st.expander(f"**{vm}**", expanded=False):
                    for triage, count in triage_counts.items():
                        if triage != "":
                            st.write(f"• {triage}: {count}")
        
    except Exception as e:
        st.error(f"Error analyzing triage patterns: {e}")


def _analyze_missing_triage(df: pd.DataFrame):
    """Analyze missing triage decisions."""
    try:
        missing_mask = (df["Diagnostic Triage"].map(normalize_text) == "")
        missing_count = missing_mask.sum()
        total_count = len(df)
        
        st.write(f"**Missing Triage Analysis:**")
        st.metric("Missing Triage", f"{missing_count}/{total_count}")
        st.metric("Coverage", f"{((total_count - missing_count) / total_count) * 100:.1f}%")
        
        if missing_count > 0:
            st.write("**Rows missing triage:**")
            missing_df = df[missing_mask][["Vital Measurement"] + LEVEL_COLS[:2]].head(20)
            st.dataframe(missing_df, use_container_width=True)
            
            if missing_count > 20:
                st.caption(f"... and {missing_count - 20} more rows")
        
    except Exception as e:
        st.error(f"Error analyzing missing triage: {e}")


def _analyze_triage_quality(df: pd.DataFrame):
    """Analyze the quality of triage decisions."""
    try:
        triage_values = df["Diagnostic Triage"].map(normalize_text).dropna()
        triage_values = triage_values[triage_values != ""]
        
        if triage_values.empty:
            st.info("No triage data to analyze.")
            return
        
        st.write("**Triage Quality Analysis:**")
        
        # Length analysis
        triage_lengths = [len(str(t)) for t in triage_values]
        avg_length = sum(triage_lengths) / len(triage_lengths)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Average Triage Length", f"{avg_length:.1f} chars")
        with col2:
            st.metric("Total Triage Decisions", len(triage_values))
        
        # Show sample triage decisions
        st.write("**Sample triage decisions:**")
        sample_triage = triage_values.head(10).tolist()
        for i, triage in enumerate(sample_triage, 1):
            st.write(f"{i}. {triage}")
        
        if len(triage_values) > 10:
            st.caption(f"... and {len(triage_values) - 10} more")
        
    except Exception as e:
        st.error(f"Error analyzing triage quality: {e}")
