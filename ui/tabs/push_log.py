# ui/tabs/push_log.py
import streamlit as st
import pandas as pd
from typing import Dict, Any, List
from datetime import datetime

from utils import (
    CANON_HEADERS, LEVEL_COLS, normalize_text, validate_headers
)
import utils.state as USTATE
from ui.utils.rerun import safe_rerun
from io_utils.sheets import make_push_log_entry


def render():
    """Render the Push Log tab for managing data push operations."""
    
    # Add guard and debug expander
    from ui.utils.guards import ensure_active_workbook_and_sheet
    ok, df = ensure_active_workbook_and_sheet("Push Log")
    if not ok:
        return
    
    # Debug state expander
    import json
    with st.expander("🛠 Debug: Session State (tab)", expanded=False):
        ss = {k: type(v).__name__ for k,v in st.session_state.items()}
        st.code(json.dumps(ss, indent=2))
    
    try:
        st.header("📜 Push Log")
        
        # Get current sheet name for display
        sheet = USTATE.get_current_sheet()
        
        # Status badge
        has_wb, sheet_count, current_sheet = USTATE.get_workbook_status()
        if has_wb and current_sheet:
            st.caption(f"Workbook: ✅ {sheet_count} sheet(s) • Active: **{current_sheet}**")
        else:
            st.caption("Workbook: ❌ not loaded")

        # Get source (legacy - could be updated later)
        src = "upload"  # Default for now

        # Main sections
        _render_push_log_overview(sheet)
        
        st.markdown("---")
        
        _render_push_operations(df, sheet, src)
        
        st.markdown("---")
        
        _render_push_history(sheet)

    except Exception as e:
        st.exception(e)


def _render_push_log_overview(sheet_name: str):
    """Render the push log overview section."""
    st.subheader("📊 Push Log Overview")
    
    # Get push log data
    push_log = st.session_state.get("push_log", [])
    
    if not push_log:
        st.info("No push operations recorded yet.")
        return
    
    # Filter by current sheet
    sheet_logs = [log for log in push_log if log.get("sheet") == sheet_name]
    
    if not sheet_logs:
        st.info(f"No push operations recorded for '{sheet_name}'.")
        return
    
    # Display statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_pushes = len(sheet_logs)
        st.metric("Total Pushes", total_pushes)
    
    with col2:
        total_rows_written = sum(int(log.get("rows_written", 0)) for log in sheet_logs)
        st.metric("Total Rows Written", total_rows_written)
    
    with col3:
        total_new_rows = sum(int(log.get("new_rows_added", 0)) for log in sheet_logs)
        st.metric("New Rows Added", total_new_rows)
    
    with col4:
        latest_push = max(sheet_logs, key=lambda x: x.get("ts", ""))
        latest_time = latest_push.get("ts", "Never")
        st.metric("Last Push", latest_time.split()[0] if latest_time != "Never" else "Never")
    
    # Show recent pushes
    st.write("**Recent pushes:**")
    recent_logs = sorted(sheet_logs, key=lambda x: x.get("ts", ""), reverse=True)[:5]
    
    for log in recent_logs:
        with st.expander(f"📝 {log.get('ts', 'Unknown time')} - {log.get('target_tab', 'Unknown tab')}", expanded=False):
            st.write(f"**Sheet:** {log.get('sheet', 'Unknown')}")
            st.write(f"**Target:** {log.get('target_tab', 'Unknown')}")
            st.write(f"**Rows written:** {log.get('rows_written', '0')}")
            st.write(f"**New rows:** {log.get('new_rows_added', '0')}")
            st.write(f"**Scope:** {log.get('scope', 'Unknown')}")
            
            # Show extra fields if any
            extra_fields = {k: v for k, v in log.items() if k not in ['ts', 'sheet', 'target_tab', 'spreadsheet_id', 'rows_written', 'new_rows_added', 'scope']}
            if extra_fields:
                st.write("**Additional info:**")
                for key, value in extra_fields.items():
                    st.write(f"• {key}: {value}")


def _render_push_operations(df: pd.DataFrame, sheet_name: str, source: str):
    """Render the push operations section."""
    st.subheader("🚀 Push Operations")
    
    # Push options
    col1, col2 = st.columns(2)
    
    with col1:
        push_mode = st.radio(
            "Push mode",
            ["Full sheet", "Selected rows", "Incremental"],
            help="Choose what to push"
        )
    
    with col2:
        target_type = st.selectbox(
            "Target type",
            ["Google Sheets", "Export file"],
            help="Choose where to push the data"
        )
    
    # Push configuration
    if push_mode == "Full sheet":
        _render_full_sheet_push(df, sheet_name, source, target_type)
    elif push_mode == "Selected rows":
        _render_selected_rows_push(df, sheet_name, source, target_type)
    else:  # Incremental
        _render_incremental_push(df, sheet_name, source, target_type)


def _render_full_sheet_push(df: pd.DataFrame, sheet_name: str, source: str, target_type: str):
    """Render full sheet push interface."""
    st.write("**Full Sheet Push**")
    
    # Show sheet summary
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Rows", len(df))
    
    with col2:
        vm_count = df["Vital Measurement"].map(normalize_text).dropna().nunique()
        st.metric("Vital Measurements", vm_count)
    
    with col3:
        non_empty_nodes = sum(1 for col in LEVEL_COLS if col in df.columns and df[col].map(normalize_text).dropna().nunique() > 0)
        st.metric("Active Node Levels", non_empty_nodes)
    
    # Push options
    if target_type == "Google Sheets":
        _render_google_sheets_push(df, sheet_name, source, "full")
    else:  # Export file
        _render_export_file_push(df, sheet_name, "full")


def _render_selected_rows_push(df: pd.DataFrame, sheet_name: str, source: str, target_type: str):
    """Render selected rows push interface."""
    st.write("**Selected Rows Push**")
    
    # Row selection options
    selection_method = st.radio(
        "Selection method",
        ["By Vital Measurement", "By Node values", "Manual selection"],
        help="Choose how to select rows"
    )
    
    if selection_method == "By Vital Measurement":
        vm_values = df["Vital Measurement"].map(normalize_text).dropna().unique()
        vm_values = [v for v in vm_values if v != ""]
        
        if vm_values:
            selected_vms = st.multiselect(
                "Select Vital Measurements",
                sorted(vm_values),
                help="Choose which vital measurements to push"
            )
            
            if selected_vms:
                selected_df = df[df["Vital Measurement"].map(normalize_text).isin(selected_vms)]
                st.write(f"Selected {len(selected_df)} rows")
                
                if target_type == "Google Sheets":
                    _render_google_sheets_push(selected_df, sheet_name, source, "selected")
                else:
                    _render_export_file_push(selected_df, sheet_name, "selected")
    
    elif selection_method == "By Node values":
        # Node-based selection
        node_col = st.selectbox("Select node column", LEVEL_COLS[:3], help="Choose node level for filtering")
        
        if node_col in df.columns:
            node_values = df[node_col].map(normalize_text).dropna().unique()
            node_values = [v for v in node_values if v != ""]
            
            if node_values:
                selected_nodes = st.multiselect(
                    f"Select {node_col} values",
                    sorted(node_values),
                    help=f"Choose which {node_col} values to push"
                )
                
                if selected_nodes:
                    selected_df = df[df[node_col].map(normalize_text).isin(selected_nodes)]
                    st.write(f"Selected {len(selected_df)} rows")
                    
                    if target_type == "Google Sheets":
                        _render_google_sheets_push(selected_df, sheet_name, source, "selected")
                    else:
                        _render_export_file_push(selected_df, sheet_name, "selected")
    
    else:  # Manual selection
        st.write("**Manual Row Selection**")
        st.info("Use the data editor below to select specific rows, then push the filtered data.")
        
        # Show data with selection
        edited_df = st.data_editor(df, num_rows="dynamic")
        
        if st.button("Push Selected Data"):
            if target_type == "Google Sheets":
                _render_google_sheets_push(edited_df, sheet_name, source, "manual")
            else:
                _render_export_file_push(edited_df, sheet_name, "manual")


def _render_incremental_push(df: pd.DataFrame, sheet_name: str, source: str, target_type: str):
    """Render incremental push interface."""
    st.write("**Incremental Push**")
    
    # Get last push info
    push_log = st.session_state.get("push_log", [])
    sheet_logs = [log for log in push_log if log.get("sheet") == sheet_name]
    
    if not sheet_logs:
        st.info("No previous pushes found. Use 'Full sheet' mode for first push.")
        return
    
    latest_push = max(sheet_logs, key=lambda x: x.get("ts", ""))
    last_push_time = latest_push.get("ts", "")
    last_rows_written = int(latest_push.get("rows_written", 0))
    
    st.write(f"**Last push:** {last_push_time}")
    st.write(f"**Rows in last push:** {last_rows_written}")
    
    # Incremental options
    incremental_mode = st.radio(
        "Incremental mode",
        ["New rows only", "Modified rows only", "New + modified"],
        help="Choose what to include in incremental push"
    )
    
    if incremental_mode == "New rows only":
        # Simple approach: push rows beyond last count
        if len(df) > last_rows_written:
            new_rows = df.iloc[last_rows_written:]
            st.write(f"Found {len(new_rows)} new rows to push")
            
            if st.button("Push New Rows"):
                _execute_push(new_rows, sheet_name, source, target_type, "incremental_new", len(new_rows))
        else:
            st.info("No new rows found since last push.")
    
    elif incremental_mode == "Modified rows only":
        st.info("Modified row detection requires tracking changes. Use 'Full sheet' mode for now.")
    
    else:  # New + modified
        st.info("Combined incremental push requires change tracking. Use 'Full sheet' mode for now.")


def _render_google_sheets_push(df: pd.DataFrame, sheet_name: str, source: str, push_type: str):
    """Render Google Sheets push interface."""
    st.write("**Google Sheets Push**")
    
    # Check if Google Sheets is configured
    if "gcp_service_account" not in st.secrets:
        st.error("Google Sheets not configured. Add your service account JSON under [gcp_service_account].")
        return
    
    # Get spreadsheet ID
    spreadsheet_id = st.text_input(
        "Spreadsheet ID",
        value=st.session_state.get("gs_spreadsheet_id", ""),
        help="Enter the Google Sheets spreadsheet ID"
    )
    
    if spreadsheet_id:
        st.session_state["gs_spreadsheet_id"] = spreadsheet_id
    
    # Target sheet name
    target_sheet = st.text_input(
        "Target sheet name",
        value=sheet_name,
        help="Name of the sheet to push to"
    )
    
    # Push options
    col1, col2 = st.columns(2)
    
    with col1:
        push_mode = st.selectbox(
            "Push mode",
            ["overwrite", "append"],
            help="Choose whether to overwrite or append to existing data"
        )
    
    with col2:
        create_backup = st.checkbox(
            "Create backup before push",
            value=True,
            help="Create a backup of the target sheet before pushing"
        )
    
    # Execute push
    if st.button("🚀 Push to Google Sheets", type="primary"):
        if spreadsheet_id and target_sheet:
            _execute_push(df, sheet_name, source, "google_sheets", push_type, len(df), 
                         spreadsheet_id=spreadsheet_id, target_sheet=target_sheet, 
                         push_mode=push_mode, create_backup=create_backup)
        else:
            st.warning("Please provide both spreadsheet ID and target sheet name.")


def _render_export_file_push(df: pd.DataFrame, sheet_name: str, push_type: str):
    """Render export file push interface."""
    st.write("**Export File Push**")
    
    # Export options
    col1, col2 = st.columns(2)
    
    with col1:
        export_format = st.selectbox(
            "Export format",
            ["Excel (.xlsx)", "CSV", "JSON"],
            help="Choose the export format"
        )
    
    with col2:
        filename = st.text_input(
            "Filename",
            value=f"{sheet_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            help="Base filename for export"
        )
    
    # Execute export
    if st.button("📤 Export File", type="primary"):
        _execute_push(df, sheet_name, "local", push_type, len(df), 
                     export_format=export_format, filename=filename)


def _render_push_history(sheet_name: str):
    """Render the push history section."""
    st.subheader("📚 Push History")
    
    # Get push log data
    push_log = st.session_state.get("push_log", [])
    sheet_logs = [log for log in push_log if log.get("sheet") == sheet_name]
    
    if not sheet_logs:
        st.info("No push history found.")
        return
    
    # Sort by timestamp
    sheet_logs.sort(key=lambda x: x.get("ts", ""), reverse=True)
    
    # Show all pushes
    st.write(f"**Complete push history for '{sheet_name}':**")
    
    for i, log in enumerate(sheet_logs):
        with st.expander(f"📝 {log.get('ts', 'Unknown time')} - {log.get('target_tab', 'Unknown tab')}", expanded=i < 3):
            # Create a nice display
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.write(f"**Target:** {log.get('target_tab', 'Unknown')}")
                st.write(f"**Source:** {log.get('sheet', 'Unknown')}")
                st.write(f"**Scope:** {log.get('scope', 'Unknown')}")
                
                if log.get('spreadsheet_id'):
                    st.write(f"**Spreadsheet ID:** {log.get('spreadsheet_id')}")
            
            with col2:
                st.metric("Rows Written", log.get('rows_written', '0'))
                st.metric("New Rows", log.get('new_rows_added', '0'))
            
            # Show extra fields
            extra_fields = {k: v for k, v in log.items() if k not in ['ts', 'sheet', 'target_tab', 'spreadsheet_id', 'rows_written', 'new_rows_added', 'scope']}
            if extra_fields:
                st.write("**Additional details:**")
                for key, value in extra_fields.items():
                    st.write(f"• {key}: {value}")
    
    # Export push history
    st.markdown("---")
    if st.button("📥 Export Push History"):
        _export_push_history(sheet_logs, sheet_name)


def _execute_push(
    df: pd.DataFrame, 
    sheet_name: str, 
    source: str, 
    target_type: str,   # e.g. "google_sheets"
    push_type: str,     # "full" | "delta" | etc.
    rows_count: int | None = None,
    **kwargs
):
    """
    Execute a push to the given target. If rows_count is provided, use it
    for logging/UI; otherwise compute from df.
    """
    try:
        # Compute rows_count if not provided
        if rows_count is None:
            try:
                rows_count = len(df)
            except Exception:
                rows_count = 0
        
        with st.spinner("Executing push operation..."):
            # Handle Google Sheets push
            if target_type == "google_sheets":
                _execute_google_sheets_push(df, sheet_name, source, push_type, rows_count, **kwargs)
            else:
                # Handle other export types (local files, etc.)
                _execute_local_export(df, sheet_name, source, push_type, rows_count, **kwargs)
            
            # Create push log entry
            log_entry = make_push_log_entry(
                sheet=sheet_name,
                target_tab=kwargs.get('target_sheet', 'export'),
                spreadsheet_id=kwargs.get('spreadsheet_id', 'local'),
                rows_written=rows_count,
                new_rows_added=rows_count,  # Simplified for now
                scope=push_type,
                extra={
                    "source": source,
                    "push_type": push_type,
                    "export_format": kwargs.get('export_format', 'N/A'),
                    "filename": kwargs.get('filename', 'N/A'),
                    "push_mode": kwargs.get('push_mode', 'N/A')
                }
            )
            
            # Add to push log
            push_log = st.session_state.get("push_log", [])
            push_log.append(log_entry)
            st.session_state["push_log"] = push_log
            
            # Show success message
            if target_type == "google_sheets":
                st.success(f"✅ Successfully pushed {rows_count} rows to Google Sheets!")
            else:
                st.success(f"✅ Successfully exported {rows_count} rows to file!")
            
            safe_rerun()
            
    except Exception as e:
        st.error(f"❌ Push operation failed: {e}")
        st.exception(e)


def _execute_google_sheets_push(df: pd.DataFrame, sheet_name: str, source: str, push_type: str, rows_count: int, **kwargs):
    """Execute Google Sheets push using proven resize-then-write semantics."""
    try:
        from io_utils.sheets import push_to_google_sheets
        
        spreadsheet_id = kwargs.get('spreadsheet_id')
        target_sheet = kwargs.get('target_sheet', sheet_name)
        push_mode = kwargs.get('push_mode', 'overwrite')
        create_backup = kwargs.get('create_backup', True)
        
        if not spreadsheet_id:
            raise ValueError("Spreadsheet ID is required for Google Sheets push")
        
        # Get service account credentials
        if "gcp_service_account" not in st.secrets:
            raise ValueError("Google Sheets not configured. Add your service account JSON under [gcp_service_account].")
        
        secrets_dict = st.secrets["gcp_service_account"]
        
        # Create backup if requested
        if create_backup:
            try:
                from io_utils.sheets import backup_sheet_copy
                backup_name = backup_sheet_copy(spreadsheet_id, target_sheet, secrets_dict)
                if backup_name:
                    st.info(f"📋 Created backup: {backup_name}")
            except Exception as e:
                st.warning(f"⚠️ Backup creation failed: {e}")
        
        # Execute the push using proven Sheets semantics
        success = push_to_google_sheets(
            spreadsheet_id=spreadsheet_id,
            sheet_name=target_sheet,
            df=df,
            secrets_dict=secrets_dict,
            mode=push_mode
        )
        
        if not success:
            raise Exception("Google Sheets push operation failed")
            
    except Exception as e:
        st.error(f"❌ Google Sheets push failed: {e}")
        raise


def _execute_local_export(df: pd.DataFrame, sheet_name: str, source: str, push_type: str, rows_count: int, **kwargs):
    """Execute local file export."""
    try:
        export_format = kwargs.get('export_format', 'Excel (.xlsx)')
        filename = kwargs.get('filename', f"{sheet_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        
        if export_format == "Excel (.xlsx)":
            from io_utils.sheets import export_dataframe_to_excel_bytes
            file_data = export_dataframe_to_excel_bytes(df, sheet_name)
            file_extension = ".xlsx"
            mime_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            
        elif export_format == "CSV":
            from io_utils.sheets import export_dataframe_to_csv_bytes
            file_data = export_dataframe_to_csv_bytes(df)
            file_extension = ".csv"
            mime_type = "text/csv"
            
        elif export_format == "JSON":
            import json
            file_data = json.dumps(df.to_dict('records'), indent=2).encode('utf-8')
            file_extension = ".json"
            mime_type = "application/json"
            
        else:
            raise ValueError(f"Unsupported export format: {export_format}")
        
        # Create download button
        st.download_button(
            label=f"📥 Download {export_format}",
            data=file_data,
            file_name=f"{filename}{file_extension}",
            mime=mime_type
        )
        
    except Exception as e:
        st.error(f"❌ Local export failed: {e}")
        raise


def _export_push_history(push_logs: List[Dict], sheet_name: str):
    """Export push history to a file."""
    try:
        if not push_logs:
            st.warning("No push history to export.")
            return
        
        # Create DataFrame
        history_df = pd.DataFrame(push_logs)
        
        # Download button
        csv = history_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Push History (CSV)",
            data=csv,
            file_name=f"{sheet_name}_push_history.csv",
            mime="text/csv"
        )
        
    except Exception as e:
        st.error(f"Error exporting push history: {e}")
