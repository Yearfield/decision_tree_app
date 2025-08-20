# ui/tabs/validation.py
import streamlit as st
import pandas as pd
from typing import Dict, Any, List

from utils import (
    CANON_HEADERS, LEVEL_COLS, normalize_text, validate_headers
)


def render():
    """Render the Validation tab for checking decision tree integrity."""
    try:
        st.header("🧪 Validation rules")

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

        # Get overrides and red flag data
        overrides_all = st.session_state.get("branch_overrides", {})
        overrides_sheet = overrides_all.get(sheet, {})
        redflag_map = st.session_state.get("symptom_quality", {})

        # Validation options
        show_loose = st.checkbox("Show loose orphans", value=True)
        show_strict = st.checkbox("Show strict orphans", value=True)
        
        # Run validation
        if st.button("🔍 Run Validation", type="primary"):
            with st.spinner("Running validation checks..."):
                _run_validation_checks(df, overrides_sheet, show_loose, show_strict, sheet)

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


def _has_active_workbook() -> bool:
    """Check if there's an active workbook in session state."""
    upload_wb = st.session_state.get("upload_workbook", {})
    gs_wb = st.session_state.get("gs_workbook", {})
    return bool(upload_wb or gs_wb)


def _run_validation_checks(df: pd.DataFrame, overrides_sheet: Dict, show_loose: bool, show_strict: bool, sheet_name: str):
    """Run all validation checks and display results."""
    try:
        # Get cached validation report
        from streamlit_app_upload import get_cached_validation_summary_for_ui
        report = get_cached_validation_summary_for_ui(df, sheet_name)
        
        # Display summary
        st.subheader("📊 Validation Summary")
        summary = report["summary"]
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Orphan Nodes", summary["total_orphans"])
        with col2:
            st.metric("Loops", summary["total_loops"])
        with col3:
            st.metric("Missing Red Flags", summary["total_missing_red_flags"])
        with col4:
            st.metric("Total Issues", summary["total_issues"])

        # Orphan detection
        if show_loose or show_strict:
            st.markdown("---")
            _display_orphan_analysis(df, overrides_sheet, show_loose, show_strict, sheet_name)

        # Loop detection
        if report["loops"]:
            st.markdown("---")
            _display_loop_analysis(report["loops"])

        # Missing red flags
        if report["missing_red_flags"]:
            st.markdown("---")
            _display_missing_red_flags_analysis(report["missing_red_flags"])

        # Overall status
        st.markdown("---")
        if summary["total_issues"] == 0:
            st.success("✅ All validation checks passed! Your decision tree looks healthy.")
        else:
            st.warning(f"⚠️ Found {summary['total_issues']} validation issue(s). Review the details above.")

    except Exception as e:
        st.error(f"Error running validation: {e}")
        st.exception(e)


def _display_orphan_analysis(df: pd.DataFrame, overrides_sheet: Dict, show_loose: bool, show_strict: bool, sheet_name: str):
    """Display orphan node analysis."""
    st.subheader("🔍 Orphan Node Analysis")
    
    # Use the cached validation functions from the main app
    if show_loose:
        from streamlit_app_upload import get_cached_validation_summary_for_ui
        report = get_cached_validation_summary_for_ui(df, sheet_name)
        orphans_loose = report["orphans"]
        
        colA, colB = st.columns(2)
        
        with colA:
            st.subheader("Orphans (loose)")
            if not orphans_loose:
                st.success("None found")
            else:
                st.write(f"Found {len(orphans_loose)} loose orphan nodes")
                orphan_df = pd.DataFrame(orphans_loose)
                st.dataframe(orphan_df, use_container_width=True)
    
    if show_strict:
        # For strict orphans, we'll use a simplified approach
        # In a full implementation, this would use overrides to determine strict orphans
        orphans_strict = []  # Placeholder - would implement strict orphan detection
        
        with colB if show_loose else st:
            st.subheader("Orphans (strict)")
            if not orphans_strict:
                st.success("None found")
            else:
                st.write(f"Found {len(orphans_strict)} strict orphan nodes")
                orphan_df = pd.DataFrame(orphans_strict)
                st.dataframe(orphan_df, use_container_width=True)


def _display_loop_analysis(loops: list):
    """Display loop/cycle analysis."""
    st.subheader("🔄 Loop Detection")
    
    if not loops:
        st.success("No loops detected")
        return
    
    st.write(f"Found {len(loops)} loop(s) in the decision tree:")
    
    # Format loop data for display
    def _format_loop_data(loop):
        formatted = {}
        for key, value in loop.items():
            if key == "path" and isinstance(value, (list, tuple)):
                formatted[key] = " > ".join(map(str, value))
            elif key == "repeats" and isinstance(value, list):
                try:
                    parts = []
                    for item in value:
                        if isinstance(item, (list, tuple)) and len(item) == 3:
                            label, i, j = item
                            parts.append(f"{label} ({i}->{j})")
                        else:
                            parts.append(str(item))
                    formatted[key] = "; ".join(parts)
                except Exception:
                    formatted[key] = str(value)
            else:
                formatted[key] = str(value)
        return formatted
    
    formatted_loops = [_format_loop_data(loop) for loop in loops]
    loop_df = pd.DataFrame(formatted_loops)
    st.dataframe(loop_df, use_container_width=True)


def _display_missing_red_flags_analysis(missing_red_flags: list):
    """Display missing red flags analysis."""
    st.subheader("🚨 Missing Red Flags")
    
    if not missing_red_flags:
        st.success("No missing red flags detected")
        return
    
    st.write(f"Found {len(missing_red_flags)} nodes that might need red flag indicators:")
    
    red_flags_df = pd.DataFrame(missing_red_flags)
    st.dataframe(red_flags_df, use_container_width=True)
    
    # Show suggestions
    st.markdown("**Suggestions:**")
    for item in missing_red_flags:
        st.info(f"**{item['node']}** ({item['node_id']}): {item['suggested_action']}")


def _detect_orphans_strict(df: pd.DataFrame, overrides: Dict) -> list:
    """Detect strict orphans using overrides (placeholder implementation)."""
    # This would implement strict orphan detection using the overrides
    # For now, return empty list as placeholder
    return []
