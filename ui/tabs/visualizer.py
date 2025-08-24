# ui/tabs/visualizer.py
import streamlit as st
import pandas as pd
from typing import Dict, Any, List

from utils import (
    CANON_HEADERS, LEVEL_COLS, normalize_text, validate_headers
)
import utils.state as USTATE


def render():
    """Render the Visualizer tab for decision tree visualization."""
    
    # Add guard and debug expander
    from ui.utils.guards import ensure_active_workbook_and_sheet
    ok, df = ensure_active_workbook_and_sheet("Visualizer")
    if not ok:
        return
    
    # Debug state expander
    import json
    with st.expander("🛠 Debug: Session State (tab)", expanded=False):
        ss = {k: type(v).__name__ for k,v in st.session_state.items()}
        st.code(json.dumps(ss, indent=2))
    
    try:
        st.header("🌐 Visualizer")
        
        # Get current sheet name for display
        sheet = USTATE.get_current_sheet()
        
        # Status badge
        has_wb, sheet_count, current_sheet = USTATE.get_workbook_status()
        if has_wb and current_sheet:
            st.caption(f"Workbook: ✅ {sheet_count} sheet(s) • Active: **{current_sheet}**")
        else:
            st.caption("Workbook: ❌ not loaded")

        # Main sections
        _render_visualization_options(df, sheet)
        
        st.markdown("---")
        
        _render_tree_explorer(df, sheet)

    except Exception as e:
        st.exception(e)


def _render_visualization_options(df: pd.DataFrame, sheet_name: str):
    """Render the visualization options section."""
    st.subheader("🎨 Visualization Options")
    
    # Visualization type
    viz_type = st.selectbox(
        "Visualization type",
        ["Tree Explorer", "Path Analysis", "Node Distribution", "Simple Text Tree"],
        help="Choose how to visualize the decision tree"
    )
    
    # Filtering options
    col1, col2 = st.columns(2)
    
    with col1:
        # Filter by Vital Measurement
        vm_values = df["Vital Measurement"].map(normalize_text).dropna().unique()
        vm_values = [v for v in vm_values if v != ""]
        
        if vm_values:
            selected_vm = st.selectbox(
                "Filter by Vital Measurement",
                ["All"] + sorted(vm_values),
                help="Focus on specific vital measurements"
            )
        else:
            selected_vm = "All"
    
    with col2:
        # Max depth filter
        max_depth = st.slider(
            "Max depth to show",
            min_value=1,
            max_value=5,
            value=3,
            help="Limit tree depth for better visualization"
        )
    
    # Apply filters
    filtered_df = df.copy()
    
    if selected_vm != "All":
        filtered_df = filtered_df[filtered_df["Vital Measurement"].map(normalize_text) == selected_vm]
    
    # Generate visualization based on type
    if viz_type == "Tree Explorer":
        _render_tree_explorer(filtered_df, sheet_name, max_depth)
    elif viz_type == "Path Analysis":
        _render_path_analysis(filtered_df, sheet_name)
    elif viz_type == "Node Distribution":
        _render_node_distribution(filtered_df, sheet_name)
    else:  # Simple Text Tree
        _render_simple_text_tree(filtered_df, sheet_name, max_depth)


def _render_tree_explorer(df: pd.DataFrame, sheet_name: str, max_depth: int = 3):
    """Render the tree explorer visualization."""
    st.subheader("🌳 Tree Explorer")
    
    if df.empty:
        st.info("No data to visualize after filtering.")
        return
    
    # Build tree structure
    tree_data = _build_tree_structure(df, max_depth)
    
    if not tree_data:
        st.info("No tree structure found in the data.")
        return
    
    # Display tree
    st.write(f"**Tree structure for '{sheet_name}' (max depth: {max_depth}):**")
    
    for vm, vm_data in tree_data.items():
        with st.expander(f"🌿 {vm}", expanded=False):
            _display_tree_node(vm_data, 0, max_depth)


def _render_path_analysis(df: pd.DataFrame, sheet_name: str):
    """Render path analysis visualization."""
    st.subheader("🛤️ Path Analysis")
    
    if df.empty:
        st.info("No data to analyze.")
        return
    
    # Analyze paths
    paths = _analyze_paths(df)
    
    if not paths:
        st.info("No complete paths found in the data.")
        return
    
    # Show path statistics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        total_paths = len(paths)
        st.metric("Total Paths", total_paths)
    
    with col2:
        avg_path_length = sum(len(path) for path in paths) / len(paths)
        st.metric("Avg Path Length", f"{avg_path_length:.1f}")
    
    with col3:
        max_path_length = max(len(path) for path in paths)
        st.metric("Max Path Length", max_path_length)
    
    # Show sample paths
    st.write("**Sample paths:**")
    for i, path in enumerate(paths[:10], 1):
        path_str = " → ".join(path)
        st.write(f"{i}. {path_str}")
    
    if len(paths) > 10:
        st.caption(f"... and {len(paths) - 10} more paths")


def _render_node_distribution(df: pd.DataFrame, sheet_name: str):
    """Render node distribution visualization."""
    st.subheader("📊 Node Distribution")
    
    if df.empty:
        st.info("No data to analyze.")
        return
    
    # Analyze node distribution
    node_stats = _analyze_node_distribution(df)
    
    if not node_stats:
        st.info("No node data found.")
        return
    
    # Display statistics
    for level, stats in node_stats.items():
        with st.expander(f"**Level {level}**", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Unique Nodes", stats["unique_count"])
                st.metric("Total Occurrences", stats["total_count"])
            
            with col2:
                st.metric("Avg Occurrences", f"{stats['avg_occurrences']:.1f}")
                st.metric("Max Occurrences", stats["max_occurrences"])
            
            # Show top nodes
            st.write("**Top nodes by frequency:**")
            for node, count in stats["top_nodes"][:5]:
                st.write(f"• {node}: {count} times")


def _render_simple_text_tree(df: pd.DataFrame, sheet_name: str, max_depth: int = 3):
    """Render a simple text-based tree visualization."""
    st.subheader("📝 Simple Text Tree")
    
    if df.empty:
        st.info("No data to visualize.")
        return
    
    # Build simple tree
    tree_text = _build_simple_tree_text(df, max_depth)
    
    if not tree_text:
        st.info("No tree structure found.")
        return
    
    # Display tree
    st.write("**Tree structure:**")
    st.code(tree_text, language="text")


def _build_tree_structure(df: pd.DataFrame, max_depth: int) -> Dict:
    """Build tree structure from DataFrame."""
    try:
        tree = {}
        
        for _, row in df.iterrows():
            vm = normalize_text(row.get("Vital Measurement", ""))
            if not vm:
                continue
            
            if vm not in tree:
                tree[vm] = {}
            
            current_node = tree[vm]
            
            for level in range(1, min(max_depth + 1, len(LEVEL_COLS) + 1)):
                node_col = f"Node {level}"
                if node_col in df.columns:
                    node_value = normalize_text(row.get(node_col, ""))
                    if node_value:
                        if level not in current_node:
                            current_node[level] = {}
                        if node_value not in current_node[level]:
                            current_node[level][node_value] = {}
                        current_node = current_node[level][node_value]
        
        return tree
    except Exception:
        return {}


def _display_tree_node(node_data: Dict, level: int, max_depth: int):
    """Display a tree node recursively."""
    if level >= max_depth:
        return
    
    for node_type, children in node_data.items():
        if isinstance(children, dict) and children:
            # This is a level with children
            st.write(f"{'  ' * level}├─ {node_type}")
            _display_tree_node(children, level + 1, max_depth)
        else:
            # This is a leaf node
            st.write(f"{'  ' * level}└─ {node_type}")


def _analyze_paths(df: pd.DataFrame) -> List[List[str]]:
    """Analyze complete paths in the decision tree."""
    try:
        paths = []
        
        for _, row in df.iterrows():
            path = []
            
            # Add Vital Measurement
            vm = normalize_text(row.get("Vital Measurement", ""))
            if vm:
                path.append(vm)
            
            # Add Node values
            for col in LEVEL_COLS:
                if col in df.columns:
                    node_val = normalize_text(row.get(col, ""))
                    if node_val:
                        path.append(node_val)
                    else:
                        break  # Stop at first empty node
            
            if len(path) > 1:  # At least VM + 1 node
                paths.append(path)
        
        return paths
    except Exception:
        return []


def _analyze_node_distribution(df: pd.DataFrame) -> Dict:
    """Analyze distribution of nodes across levels."""
    try:
        stats = {}
        
        for level in range(1, 6):
            node_col = f"Node {level}"
            if node_col in df.columns:
                node_values = df[node_col].map(normalize_text).dropna()
                node_values = node_values[node_values != ""]
                
                if not node_values.empty:
                    value_counts = node_values.value_counts()
                    
                    stats[level] = {
                        "unique_count": len(value_counts),
                        "total_count": len(node_values),
                        "avg_occurrences": len(node_values) / len(value_counts),
                        "max_occurrences": value_counts.max(),
                        "top_nodes": value_counts.head(10).items()
                    }
        
        return stats
    except Exception:
        return {}


def _build_simple_tree_text(df: pd.DataFrame, max_depth: int) -> str:
    """Build a simple text representation of the tree."""
    try:
        tree_lines = []
        
        # Group by Vital Measurement
        vm_groups = df.groupby("Vital Measurement")
        
        for vm, vm_df in vm_groups:
            if normalize_text(vm) == "":
                continue
                
            tree_lines.append(f"🌿 {vm}")
            
            # Build tree for this VM
            vm_tree = _build_tree_structure(vm_df, max_depth)
            if vm in vm_tree:
                _add_tree_lines(vm_tree[vm], tree_lines, 1, max_depth)
            
            tree_lines.append("")  # Empty line between VMs
        
        return "\n".join(tree_lines)
    except Exception:
        return ""


def _add_tree_lines(node_data: Dict, lines: List[str], level: int, max_depth: int):
    """Add tree lines recursively."""
    if level > max_depth:
        return
    
    for node_type, children in node_data.items():
        indent = "  " * level
        if isinstance(children, dict) and children:
            lines.append(f"{indent}├─ {node_type}")
            _add_tree_lines(children, lines, level + 1, max_depth)
        else:
            lines.append(f"{indent}└─ {node_type}")
