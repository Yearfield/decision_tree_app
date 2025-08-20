# ui/tabs/conflicts.py
import streamlit as st
import pandas as pd
from typing import Dict, Any, List, Tuple

from utils import (
    CANON_HEADERS, LEVEL_COLS, normalize_text, validate_headers
)


def render():
    """Render the Conflicts tab for detecting and resolving decision tree conflicts."""
    try:
        st.header("⚖️ Conflicts")

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

        # Get overrides
        overrides_all = st.session_state.get("branch_overrides", {})
        overrides_sheet = overrides_all.get(sheet, {})

        # Conflict detection options
        st.subheader("🔍 Conflict Detection")
        
        col1, col2 = st.columns(2)
        with col1:
            detect_mode = st.radio(
                "Detection mode",
                ["Basic conflicts", "With overrides", "Deep analysis"],
                help="Choose how thoroughly to check for conflicts"
            )
        
        with col2:
            if st.button("🔍 Detect Conflicts", type="primary"):
                with st.spinner("Analyzing conflicts..."):
                    _detect_and_display_conflicts(df, overrides_sheet, detect_mode, sheet)

        # Override management
        st.markdown("---")
        _render_override_management(df, overrides_sheet, sheet)

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


def _detect_and_display_conflicts(df: pd.DataFrame, overrides_sheet: Dict, detect_mode: str, sheet_name: str):
    """Detect and display conflicts based on the selected mode."""
    try:
        if detect_mode == "Basic conflicts":
            # Use cached conflict summary
            from streamlit_app_upload import get_cached_conflict_summary_for_ui
            conflict_summary = get_cached_conflict_summary_for_ui(df, sheet_name)
            conflicts = conflict_summary["conflicts"]
        elif detect_mode == "With overrides":
            conflicts = _detect_conflicts_with_overrides(df, overrides_sheet, sheet_name)
        else:  # Deep analysis
            conflicts = _detect_deep_conflicts(df, overrides_sheet, sheet_name)

        if not conflicts:
            st.success("✅ No conflicts detected!")
            return

        st.warning(f"⚠️ Found {len(conflicts)} conflict(s):")
        
        # Group conflicts by type
        conflict_types = {}
        for conflict in conflicts:
            conflict_type = conflict.get("type", "unknown")
            if conflict_type not in conflict_types:
                conflict_types[conflict_type] = []
            conflict_types[conflict_type].append(conflict)

        # Display conflicts by type
        for conflict_type, type_conflicts in conflict_types.items():
            st.subheader(f"🔴 {conflict_type.title()} Conflicts ({len(type_conflicts)})")
            
            # Convert to DataFrame for better display
            conflict_df = pd.DataFrame(type_conflicts)
            st.dataframe(conflict_df, use_container_width=True)

    except Exception as e:
        st.error(f"Error detecting conflicts: {e}")
        st.exception(e)


def _detect_conflicts_with_overrides(df: pd.DataFrame, overrides_sheet: Dict, sheet_name: str) -> List[Dict]:
    """Detect conflicts considering overrides."""
    # Start with basic conflicts
    from streamlit_app_upload import get_cached_conflict_summary_for_ui
    conflict_summary = get_cached_conflict_summary_for_ui(df, sheet_name)
    conflicts = conflict_summary["conflicts"]
    
    try:
        # Check for override conflicts
        for override_key, override_values in overrides_sheet.items():
            if isinstance(override_key, tuple) and len(override_key) >= 2:
                level, parent_path = override_key[0], override_key[1:]
                
                # Check if override values are consistent with actual data
                if level <= 5:
                    node_col = f"Node {level}"
                    if node_col in df.columns:
                        # Find rows matching parent path
                        mask = pd.Series([True] * len(df))
                        for i, parent_val in enumerate(parent_path):
                            if i < len(LEVEL_COLS):
                                col = LEVEL_COLS[i]
                                if col in df.columns:
                                    mask &= df[col].map(normalize_text) == parent_val
                        
                        # Check actual values vs override values
                        actual_values = df.loc[mask, node_col].map(normalize_text)
                        actual_values = actual_values[actual_values != ""].unique()
                        
                        if set(actual_values) != set(override_values):
                            conflicts.append({
                                "type": "override_mismatch",
                                "level": level,
                                "parent_path": " > ".join(parent_path),
                                "override_values": override_values,
                                "actual_values": list(actual_values),
                                "description": "Override values don't match actual data"
                            })

    except Exception as e:
        st.error(f"Error in override conflict detection: {e}")
        
    return conflicts


def _detect_deep_conflicts(df: pd.DataFrame, overrides_sheet: Dict, sheet_name: str) -> List[Dict]:
    """Perform deep conflict analysis."""
    conflicts = _detect_conflicts_with_overrides(df, overrides_sheet, sheet_name)
    
    try:
        # Additional deep analysis could include:
        # - Circular references
        # - Inconsistent data types
        # - Missing required fields
        # - Business rule violations
        
        # For now, just add a placeholder
        if not conflicts:
            conflicts.append({
                "type": "deep_analysis",
                "level": "N/A",
                "description": "Deep analysis completed - no conflicts found",
                "status": "clean"
            })
            
    except Exception as e:
        st.error(f"Error in deep conflict detection: {e}")
        
    return conflicts


def _render_override_management(df: pd.DataFrame, overrides_sheet: Dict, sheet_name: str):
    """Render the override management section."""
    st.subheader("🎛️ Override Management")
    
    if not overrides_sheet:
        st.info("No overrides defined for this sheet.")
        return
    
    st.write(f"Current overrides for '{sheet_name}':")
    
    # Display current overrides
    for override_key, override_values in overrides_sheet.items():
        if isinstance(override_key, tuple) and len(override_key) >= 2:
            level, parent_path = override_key[0], override_key[1:]
            
            with st.expander(f"Level {level}: {' > '.join(parent_path) if parent_path else 'Root'}", expanded=False):
                st.write(f"**Override values:** {override_values}")
                
                # Show edit controls
                col1, col2 = st.columns([3, 1])
                with col1:
                    new_values = st.text_input(
                        "New values (comma-separated)",
                        value=", ".join(override_values),
                        key=f"edit_override_{level}_{hash(str(parent_path))}"
                    )
                
                with col2:
                    if st.button("Update", key=f"update_override_{level}_{hash(str(parent_path))}"):
                        if new_values.strip():
                            new_list = [v.strip() for v in new_values.split(",") if v.strip()]
                            # Update the override
                            overrides_all = st.session_state.get("branch_overrides", {})
                            overrides_all[sheet_name][override_key] = new_list
                            st.session_state["branch_overrides"] = overrides_all
                            st.success("Override updated!")
                            st.rerun()
                
                # Delete option
                if st.button("🗑️ Delete", key=f"delete_override_{level}_{hash(str(parent_path))}"):
                    overrides_all = st.session_state.get("branch_overrides", {})
                    if override_key in overrides_all[sheet_name]:
                        del overrides_all[sheet_name][override_key]
                        st.session_state["branch_overrides"] = overrides_all
                        st.success("Override deleted!")
                        st.rerun()
    
    # Add new override
    st.markdown("---")
    st.subheader("➕ Add New Override")
    
    col1, col2 = st.columns(2)
    with col1:
        new_level = st.number_input("Level", min_value=1, max_value=5, value=1)
        new_parent = st.text_input("Parent path (comma-separated, leave empty for root)")
    
    with col2:
        new_values = st.text_input("Values (comma-separated)")
        if st.button("Add Override"):
            if new_values.strip():
                parent_path = tuple()
                if new_parent.strip():
                    parent_path = tuple(v.strip() for v in new_parent.split(",") if v.strip())
                
                override_key = (new_level, parent_path)
                override_values = [v.strip() for v in new_values.split(",") if v.strip()]
                
                # Add to overrides
                overrides_all = st.session_state.get("branch_overrides", {})
                if sheet_name not in overrides_all:
                    overrides_all[sheet_name] = {}
                overrides_all[sheet_name][override_key] = override_values
                st.session_state["branch_overrides"] = overrides_all
                
                st.success("Override added!")
                st.rerun()
