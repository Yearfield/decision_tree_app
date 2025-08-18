"""
logic_validation_functions.py

Pure logic functions for validation operations.
No Streamlit dependencies - can be imported by both logic and UI modules.
"""

from typing import Dict, List, Tuple, Optional, Set, Any
import pandas as pd
import numpy as np

from utils import (
    CANON_HEADERS, LEVEL_COLS, MAX_LEVELS,
    normalize_text, validate_headers, friendly_parent_label,
    level_key_tuple, parent_key_from_row_strict, infer_branch_options,
)


def detect_orphan_nodes(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Detect orphan nodes in the decision tree (nodes whose parent does not exist).
    
    Args:
        df: DataFrame with decision tree data
        
    Returns:
        List of dicts with orphan node details:
        {
            "label": str,        # The orphan node label
            "node_id": str,      # Node level identifier (e.g., "Node 2")
            "row_index": int,    # First occurrence row index
            "level": int,        # Node level (1-5)
            "appears_as_child_in": List[str]  # Parent paths where this appears as child
        }
        
    Failure modes:
        - Returns empty list for non-DataFrame inputs
        - Returns empty list for empty DataFrames
        - Returns empty list if required columns missing
        - Returns empty list if processing fails
    """
    try:
        if not isinstance(df, pd.DataFrame) or df.empty:
            return []
        
        # Check for required columns
        if not any(col in df.columns for col in LEVEL_COLS):
            return []
        
        orphan_nodes = []
        
        for level in range(1, MAX_LEVELS):
            child_col = LEVEL_COLS[level - 1]  # Current level (appears as child)
            parent_col = LEVEL_COLS[level]     # Next level (appears as parent)
            
            if child_col not in df.columns or parent_col not in df.columns:
                continue
                
            # Get all unique child labels at this level
            child_labels = df[child_col].map(normalize_text).replace("", np.nan).dropna().unique()
            
            # Get all unique parent labels at the next level
            parent_labels = df[parent_col].map(normalize_text).replace("", np.nan).dropna().unique()
            
            # Find children that never appear as parents
            for child_label in child_labels:
                if child_label not in parent_labels:
                    # Find first occurrence row index
                    row_index = -1
                    for idx, row in df.iterrows():
                        if normalize_text(row.get(child_col, "")) == child_label:
                            row_index = idx
                            break
                    
                    # Find parent paths where this appears as child
                    parent_paths = []
                    for idx, row in df.iterrows():
                        if normalize_text(row.get(child_col, "")) == child_label:
                            # Build parent path
                            parent_path = []
                            for i in range(level - 1):
                                col = LEVEL_COLS[i]
                                if col in df.columns:
                                    val = normalize_text(row.get(col, ""))
                                    if val:
                                        parent_path.append(val)
                            if parent_path:
                                parent_paths.append(" > ".join(parent_path))
                            else:
                                parent_paths.append("Root")
                    
                    orphan_nodes.append({
                        "label": child_label,
                        "node_id": f"Node {level}",
                        "row_index": row_index,
                        "level": level,
                        "appears_as_child_in": list(set(parent_paths))  # Remove duplicates
                    })
        
        return orphan_nodes
    except Exception:
        return []


def detect_loops(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Detect loops (cycles) in the decision tree.
    
    Args:
        df: DataFrame with decision tree data
        
    Returns:
        List of dicts with loop details:
        {
            "path": List[str],    # The cycle path (node labels)
            "length": int,        # Length of the cycle
            "start_node": str,    # Starting node of the cycle
            "cycle_type": str     # Type of cycle detected
        }
        
    Failure modes:
        - Returns empty list for non-DataFrame inputs
        - Returns empty list for empty DataFrames
        - Returns empty list if required columns missing
        - Returns empty list if processing fails
    """
    try:
        if not isinstance(df, pd.DataFrame) or df.empty:
            return []
        
        # Check for required columns
        if not any(col in df.columns for col in LEVEL_COLS):
            return []
        
        # This is a simplified implementation
        # A more sophisticated version would build a graph and detect cycles
        # For now, return empty list as placeholder
        # TODO: Implement proper cycle detection using graph algorithms
        
        loops = []
        
        # Placeholder: detect simple self-references (a node pointing to itself)
        for level in range(1, MAX_LEVELS):
            node_col = LEVEL_COLS[level - 1]
            if node_col not in df.columns:
                continue
                
            # Check for self-references (simplified loop detection)
            for idx, row in df.iterrows():
                node_label = normalize_text(row.get(node_col, ""))
                if node_label:
                    # Check if this node appears as a child of itself in deeper levels
                    for deeper_level in range(level, MAX_LEVELS):
                        deeper_col = LEVEL_COLS[deeper_level]
                        if deeper_col in df.columns:
                            for check_idx, check_row in df.iterrows():
                                # Build parent path to check if it contains the node
                                parent_path = []
                                for i in range(deeper_level):
                                    col = LEVEL_COLS[i]
                                    if col in df.columns:
                                        val = normalize_text(check_row.get(col, ""))
                                        if val:
                                            parent_path.append(val)
                                
                                # Check if the node appears in its own parent path
                                if node_label in parent_path:
                                    loops.append({
                                        "path": parent_path + [node_label],
                                        "length": len(parent_path) + 1,
                                        "start_node": node_label,
                                        "cycle_type": "self_reference"
                                    })
        
        # Remove duplicates based on path
        unique_loops = []
        seen_paths = set()
        for loop in loops:
            path_key = tuple(loop["path"])
            if path_key not in seen_paths:
                seen_paths.add(path_key)
                unique_loops.append(loop)
        
        return unique_loops
    except Exception:
        return []


def detect_missing_red_flags(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Detect missing red flag indicators in the decision tree.
    
    Args:
        df: DataFrame with decision tree data
        
    Returns:
        List of dicts with missing red flag details:
        {
            "node_id": str,       # Node level identifier (e.g., "Node 2")
            "label": str,         # The node label that's missing red flag
            "row_index": int,     # Row index where this occurs
            "level": int,         # Node level (1-5)
            "issue_type": str,    # Type of red flag issue
            "suggested_action": str  # Suggested action to fix
        }
        
    Failure modes:
        - Returns empty list for non-DataFrame inputs
        - Returns empty list for empty DataFrames
        - Returns empty list if required columns missing
        - Returns empty list if processing fails
    """
    try:
        if not isinstance(df, pd.DataFrame) or df.empty:
            return []
        
        # Check for required columns
        if not any(col in df.columns for col in LEVEL_COLS):
            return []
        
        missing_red_flags = []
        
        # Check for nodes that might need red flag indicators
        # This is a simplified implementation - in a real system, you'd have
        # specific rules for what constitutes a "red flag" scenario
        
        for level in range(1, MAX_LEVELS + 1):
            node_col = LEVEL_COLS[level - 1]
            if node_col not in df.columns:
                continue
                
            # Check for nodes that appear frequently but might need red flag coverage
            node_counts = {}
            for idx, row in df.iterrows():
                node_label = normalize_text(row.get(node_col, ""))
                if node_label:
                    node_counts[node_label] = node_counts.get(node_label, 0) + 1
            
            # Identify potential red flag candidates (nodes with high frequency)
            # that don't have explicit red flag indicators
            for node_label, count in node_counts.items():
                if count > 3:  # Arbitrary threshold - nodes appearing more than 3 times
                    # Check if this node has any red flag indicators in the Actions column
                    has_red_flag = False
                    row_index = -1
                    
                    for idx, row in df.iterrows():
                        if normalize_text(row.get(node_col, "")) == node_label:
                            row_index = idx
                            actions = normalize_text(row.get("Actions", ""))
                            if "red flag" in actions.lower() or "urgent" in actions.lower():
                                has_red_flag = True
                                break
                    
                    if not has_red_flag and row_index >= 0:
                        missing_red_flags.append({
                            "node_id": f"Node {level}",
                            "label": node_label,
                            "row_index": row_index,
                            "level": level,
                            "issue_type": "high_frequency_no_red_flag",
                            "suggested_action": f"Consider adding red flag indicators for '{node_label}' (appears {count} times)"
                        })
        
        return missing_red_flags
    except Exception:
        return []


def compute_validation_report(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute a comprehensive validation report for the decision tree.
    
    Args:
        df: DataFrame with decision tree data
        
    Returns:
        Dictionary containing validation results with structure:
        {
            "orphan_nodes": List[Dict[str, Any]],      # Orphan node details
            "loops": List[Dict[str, Any]],             # Loop details
            "missing_red_flags": List[Dict[str, Any]], # Missing red flag details
            "summary": Dict[str, Any]                  # Summary statistics
        }
        
    Failure modes:
        - Returns empty report structure for non-DataFrame inputs
        - Returns empty report structure for empty DataFrames
        - Returns empty report structure if processing fails
    """
    try:
        if not isinstance(df, pd.DataFrame) or df.empty:
            return {
                "orphan_nodes": [],
                "loops": [],
                "missing_red_flags": [],
                "summary": {
                    "total_orphans": 0,
                    "total_loops": 0,
                    "total_missing_red_flags": 0,
                    "total_issues": 0
                }
            }
        
        orphan_nodes = detect_orphan_nodes(df)
        loops = detect_loops(df)
        missing_red_flags = detect_missing_red_flags(df)
        
        return {
            "orphan_nodes": orphan_nodes,
            "loops": loops,
            "missing_red_flags": missing_red_flags,
            "summary": {
                "total_orphans": len(orphan_nodes),
                "total_loops": len(loops),
                "total_missing_red_flags": len(missing_red_flags),
                "total_issues": len(orphan_nodes) + len(loops) + len(missing_red_flags)
            }
        }
    except Exception:
        return {
            "orphan_nodes": [],
            "loops": [],
            "missing_red_flags": [],
            "summary": {
                "total_orphans": 0,
                "total_loops": 0,
                "total_missing_red_flags": 0,
                "total_issues": 0
            }
        }
