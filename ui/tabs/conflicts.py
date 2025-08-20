# ui/tabs/conflicts.py
import streamlit as st
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional

from utils import (
    CANON_HEADERS, LEVEL_COLS, normalize_text, validate_headers
)
from utils.state import (
    get_active_workbook, get_current_sheet, get_active_df, 
    has_active_workbook, get_workbook_status, set_active_workbook, set_current_sheet
)
from utils.helpers import normalize_text, normalize_child_set
from utils.constants import ROOT_PARENT_LABEL, MAX_CHILDREN_PER_PARENT
from logic.tree import infer_branch_options, set_level1_children
from ui.utils.rerun import safe_rerun


def render():
    """Render the Conflicts tab for detecting and resolving decision tree conflicts."""
    try:
        st.header("⚖️ Conflicts")
        
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
        
        # Resolve children tool
        _render_resolve_children_tool(df, sheet)

    except Exception as e:
        st.exception(e)


def _render_level1_root_editor(df: pd.DataFrame, sheet_name: str):
    """Render the Level 1 (ROOT) editor for Node-1 options."""
    st.subheader("🌱 Level 1 (ROOT) Children Editor")
    st.markdown("Manage the top-level Node-1 options that serve as the root of your decision tree.")
    
    # Get the store to see observed Node-1 options
    store = infer_branch_options(df)
    observed = store.get(f"L1|{ROOT_PARENT_LABEL}", [])
    
    if not observed:
        st.warning("No Node-1 options found. Check headers/data.")
        return
    
    st.write(f"**Observed Node-1 options ({len(observed)}):** {', '.join(observed)}")
    
    # Custom selector (limit to MAX_CHILDREN_PER_PARENT)
    chosen = st.multiselect(
        "Custom set (choose up to 5)", 
        options=observed, 
        default=observed[:MAX_CHILDREN_PER_PARENT], 
        max_selections=MAX_CHILDREN_PER_PARENT,
        key="__root_custom"
    )
    
    # Add new option
    new_opt = normalize_text(st.text_input("Add new Node-1 option"))
    if new_opt and new_opt not in chosen:
        chosen = normalize_child_set(chosen + [new_opt])
        st.info(f"**Preview after add:** {', '.join(chosen)}")
    
    # Action buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("✅ Apply: Keep observed (for all)", type="primary"):
            _apply_root_children(observed, sheet_name)
    with col2:
        if st.button("🔧 Apply: Use custom set (for all)", type="primary"):
            _apply_root_children(chosen, sheet_name)


def _apply_root_children(children: list[str], sheet_name: str):
    """Apply the new Level 1 children set to the entire sheet."""
    try:
        with st.spinner("Applying Level 1 changes..."):
            # Get current workbook and DataFrame
            wb = get_active_workbook()
            if not wb or sheet_name not in wb:
                st.error("No active sheet.")
                return
            
            df = wb[sheet_name]
            
            # Apply the new children set using logic.tree.set_level1_children
            new_df = set_level1_children(df, children)
            
            # Update the workbook with the new DataFrame
            wb[sheet_name] = new_df
            
            # Refresh the active workbook state
            set_active_workbook(wb, default_sheet=sheet_name, source="conflicts_level1_editor")
            
            # Bump nonce via re-setting current sheet
            set_current_sheet(sheet_name)
            
            st.success(f"✅ Updated Level-1 (ROOT) children to: {', '.join(children)}")
            st.info("The sheet has been updated. Other tabs will reflect these changes.")
            
            # Rerun to refresh the UI
            safe_rerun()
            
    except Exception as e:
        st.error(f"Error applying Level 1 changes: {str(e)}")
        st.exception(e)


def _detect_and_display_conflicts(df: pd.DataFrame, overrides_sheet: Dict, detect_mode: str, sheet_name: str):
    """Detect and display conflicts based on the selected mode."""
    try:
        if detect_mode == "Basic conflicts":
            # Use cached conflict summary
            from streamlit_app_upload import get_cached_conflict_summary_for_ui
            from utils.state import get_wb_nonce
            conflict_summary = get_cached_conflict_summary_for_ui(df, sheet_name, get_wb_nonce())
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
    from utils.state import get_wb_nonce
    conflict_summary = get_cached_conflict_summary_for_ui(df, sheet_name, get_wb_nonce())
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
                            safe_rerun()
                
                # Delete option
                if st.button("🗑️ Delete", key=f"delete_override_{level}_{hash(str(parent_path))}"):
                    overrides_all = st.session_state.get("branch_overrides", {})
                    if override_key in overrides_all[sheet_name]:
                        del overrides_all[sheet_name][override_key]
                        st.session_state["branch_overrides"] = overrides_all
                        st.success("Override deleted!")
                        safe_rerun()
    
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
                safe_rerun()


def _render_resolve_children_tool(df: pd.DataFrame, sheet_name: str):
    """Render the resolve children across sheet tool."""
    st.markdown("---")
    st.subheader("🔧 Resolve Children Across Sheet")
    st.markdown("Standardize child sets for all parents with the same label at a specific level.")
    
    # Level selection
    level = st.selectbox("Node level", [1, 2, 3, 4, 5], key="resolve_level")
    
    # Get parent labels for this level
    parent_labels = _get_parent_labels_at_level(df, level, "resolve_children_tool")
    
    if not parent_labels:
        st.info(f"No parent labels found at level {level}.")
        return
    
    # Parent label selection
    if level == 1:
        # For Level 1, show the special ROOT editor
        _render_level1_root_editor(df, sheet_name)
        return
    else:
        parent_label = st.selectbox("Parent label", parent_labels, key="resolve_parent_label")
        # Find the parent path for this label
        parent_path = _find_parent_path_for_label(df, level, parent_label, "resolve_children_tool")
    
    if not parent_label or (level > 1 and not parent_path):
        st.info("Please select a valid parent label.")
        return
    
    # Analyze current children for this parent label
    current_children_variants = _analyze_children_variants(df, level, parent_path, parent_label, "resolve_children_tool")
    
    if not current_children_variants:
        st.info(f"No children found for parent label '{parent_label}' at level {level}.")
        return
    
    st.write(f"**Current children variants for '{parent_label}' at level {level}:**")
    
    # Display variants
    for i, (children_set, count) in enumerate(current_children_variants):
        st.write(f"**Variant {i+1}** ({count} parents): {', '.join(children_set)}")
    
    # Action selection
    st.markdown("---")
    action = st.radio(
        "Choose action:",
        ["Keep Set For All", "Custom Set For All"],
        key="resolve_action"
    )
    
    if action == "Keep Set For All":
        # Keep existing variant
        if len(current_children_variants) == 1:
            st.info("Only one variant exists - no action needed.")
            return
        
        variant_choice = st.selectbox(
            "Select variant to keep:",
            [f"Variant {i+1}: {', '.join(children)} ({count} parents)" 
             for i, (children, count) in enumerate(current_children_variants)],
            key="variant_choice"
        )
        
        if st.button("Apply Keep Set For All", type="primary"):
            variant_index = int(variant_choice.split(":")[0].split()[1]) - 1
            selected_children = current_children_variants[variant_index][0]
            _apply_children_resolution(df, level, parent_path, parent_label, selected_children, sheet_name, "keep")
    
    else:  # Custom Set For All
        # Get union of all observed children
        all_children = set()
        for children_set, _ in current_children_variants:
            all_children.update(children_set)
        
        st.write(f"**Union of observed children:** {', '.join(sorted(all_children))}")
        
        # Multi-select up to 5 children
        selected_children = st.multiselect(
            "Select up to 5 children:",
            sorted(all_children),
            max_selections=5,
            key="custom_children"
        )
        
        # Free-text input for new children
        new_child = st.text_input("Add new child (optional):", key="new_child_input")
        if new_child.strip():
            new_child_clean = normalize_text(new_child)
            if new_child_clean and new_child_clean not in selected_children:
                selected_children.append(new_child_clean)
        
        if len(selected_children) > 5:
            selected_children = selected_children[:5]
        
        if selected_children:
            st.write(f"**Final children set:** {', '.join(selected_children)}")
            
            if st.button("Apply Custom Set For All", type="primary"):
                _apply_children_resolution(df, level, parent_path, parent_label, selected_children, sheet_name, "custom")


@st.cache_data(ttl=600)
def _get_parent_labels_at_level(df: pd.DataFrame, level: int, nonce: str) -> List[str]:
    """Get parent labels at a specific level."""
    try:
        if level <= 1:
            return []
        
        parent_col = f"Node {level - 1}"
        if parent_col not in df.columns:
            return []
        
        values = df[parent_col].map(normalize_text).dropna()
        values = values[values != ""]
        return sorted(values.unique())
        
    except Exception:
        return []


@st.cache_data(ttl=600)
def _find_parent_path_for_label(df: pd.DataFrame, level: int, label: str, nonce: str) -> Optional[Tuple[str, ...]]:
    """Find the parent path for a specific label at a level."""
    try:
        if level <= 1:
            return tuple()
        
        parent_col = f"Node {level - 1}"
        if parent_col not in df.columns:
            return None
        
        # Find rows with this label
        mask = df[parent_col].map(normalize_text) == label
        matching_rows = df[mask]
        
        if matching_rows.empty:
            return None
        
        # Get the parent path from the first matching row
        parent_cols = [f"Node {i}" for i in range(1, level - 1)]
        if not parent_cols:
            return tuple()
        
        parent_path = []
        for col in parent_cols:
            if col in df.columns:
                value = normalize_text(matching_rows.iloc[0].get(col, ""))
                if value:
                    parent_path.append(value)
        
        return tuple(parent_path)
        
    except Exception:
        return None


@st.cache_data(ttl=600)
def _analyze_children_variants(df: pd.DataFrame, level: int, parent_path: Tuple[str, ...], parent_label: str, nonce: str) -> List[Tuple[List[str], int]]:
    """Analyze current children variants for a parent label."""
    try:
        if level > 5:
            return []
        
        node_col = f"Node {level}"
        if node_col not in df.columns:
            return []
        
        # Find rows matching the parent path
        matching_rows = df.copy()
        for i, expected_value in enumerate(parent_path):
            col = f"Node {i + 1}"
            if col in df.columns:
                mask = df[col].map(normalize_text) == expected_value
                matching_rows = matching_rows[mask]
        
        if matching_rows.empty:
            return []
        
        # Get children at this level
        children_values = matching_rows[node_col].map(normalize_text).dropna()
        children_values = children_values[children_values != ""]
        
        if children_values.empty:
            return []
        
        # Group by unique children sets
        children_variants = {}
        for _, row in matching_rows.iterrows():
            child_value = normalize_text(row.get(node_col, ""))
            if child_value:
                # Create a key for this row's parent path
                row_parent_path = []
                for i in range(1, level):
                    col = f"Node {i}"
                    if col in row.index:
                        value = normalize_text(row.get(col, ""))
                        if value:
                            row_parent_path.append(value)
                
                parent_key = tuple(row_parent_path)
                if parent_key not in children_variants:
                    children_variants[parent_key] = set()
                children_variants[parent_key].add(child_value)
        
        # Count variants
        variant_counts = {}
        for children_set in children_variants.values():
            children_tuple = tuple(sorted(children_set))
            variant_counts[children_tuple] = variant_counts.get(children_tuple, 0) + 1
        
        # Return sorted by count
        result = [(list(children), count) for children, count in variant_counts.items()]
        result.sort(key=lambda x: x[1], reverse=True)
        
        return result
        
    except Exception:
        return []


def _apply_children_resolution(df: pd.DataFrame, level: int, parent_path: Tuple[str, ...], parent_label: str, 
                              new_children: List[str], sheet_name: str, action_type: str):
    """Apply children resolution across the sheet."""
    try:
        with st.spinner(f"Applying {action_type} resolution..."):
            # Create override key
            override_key = (level, parent_path)
            
            # Get current overrides
            overrides_all = st.session_state.get("branch_overrides", {})
            if sheet_name not in overrides_all:
                overrides_all[sheet_name] = {}
            
            # Set the override
            overrides_all[sheet_name][override_key] = new_children
            st.session_state["branch_overrides"] = overrides_all
            
            # Apply the override using logic.tree
            from logic.tree import build_raw_plus_v630
            
            # Get the active workbook
            from utils.state import get_active_workbook, set_active_workbook
            active_wb = get_active_workbook()
            
            if active_wb and sheet_name in active_wb:
                # Apply overrides and rebuild the sheet
                updated_df = build_raw_plus_v630(df, overrides_all[sheet_name])
                
                # Update the active workbook
                active_wb[sheet_name] = updated_df
                set_active_workbook(active_wb, source="conflicts_resolution")
                
                # Clear stale caches to ensure immediate refresh
                st.cache_data.clear()
                
                # Show delta preview
                rows_affected = len(updated_df) - len(df)
                st.success(f"✅ {action_type} resolution applied successfully!")
                st.info(f"**Delta:** {rows_affected:+d} rows affected. Sheet updated with new children set: {', '.join(new_children)}")
                
                # Rerun to show updated state
                safe_rerun()
            else:
                st.error("Could not update active workbook.")
                
    except Exception as e:
        st.error(f"Error applying resolution: {e}")
        st.exception(e)
