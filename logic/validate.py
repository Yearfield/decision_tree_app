# logic/validate.py
"""
Pure validation logic functions for decision tree operations.
No Streamlit dependencies - can be imported by both logic and UI modules.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Set, Any, Tuple

from utils import (
    LEVEL_COLS, MAX_LEVELS, normalize_text
)


def detect_orphan_nodes(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Detect orphan nodes (nodes without proper parent relationships).
    
    Args:
        df: DataFrame with decision tree data
        
    Returns:
        List of orphan node information dictionaries
    """
    try:
        if not isinstance(df, pd.DataFrame) or df.empty:
            return []
        if not any(col in df.columns for col in LEVEL_COLS):
            return []

        orphans = []
        
        # Check each level for orphan nodes
        for level in range(1, MAX_LEVELS + 1):
            node_col = f"Node {level}"
            if node_col not in df.columns:
                continue
                
            # Get unique values at this level
            node_values = df[node_col].map(normalize_text).dropna()
            node_values = node_values[node_values != ""]
            unique_nodes = node_values.unique()
            
            for node in unique_nodes:
                # Check if this node has a proper parent
                has_parent = False
                
                if level == 1:
                    # Level 1 nodes only need Vital Measurement
                    vm_col = "Vital Measurement"
                    if vm_col in df.columns:
                        vm_mask = df[vm_col].map(normalize_text) == node
                        if vm_mask.any():
                            has_parent = True
                else:
                    # Higher level nodes need parent path
                    parent_cols = [f"Node {i}" for i in range(1, level)]
                    if all(col in df.columns for col in parent_cols):
                        # Find rows with this node
                        node_mask = df[node_col].map(normalize_text) == node
                        node_rows = df[node_mask]
                        
                        # Check if any of these rows have complete parent paths
                        for _, row in node_rows.iterrows():
                            parent_complete = True
                            for col in parent_cols:
                                if normalize_text(row.get(col, "")) == "":
                                    parent_complete = False
                                    break
                            
                            if parent_complete:
                                has_parent = True
                                break
                
                if not has_parent:
                    orphans.append({
                        "node": node,
                        "level": level,
                        "node_column": node_col,
                        "type": "orphan",
                        "description": f"Node '{node}' at level {level} has no valid parent path"
                    })
        
        return orphans
        
    except Exception:
        return []


def detect_loops(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Detect loops (cycles) in the decision tree using optimized set-based approach.
    
    Args:
        df: DataFrame with decision tree data
        
    Returns:
        List of loop information dictionaries
    """
    try:
        if not isinstance(df, pd.DataFrame) or df.empty:
            return []
        if not any(col in df.columns for col in LEVEL_COLS):
            return []

        loops = []
        node_paths: Dict[str, Set[Tuple[str, ...]]] = {}

        # Build node path mapping
        for idx, row in df.iterrows():
            path = []
            for level in range(MAX_LEVELS):
                col = LEVEL_COLS[level]
                if col in df.columns:
                    val = normalize_text(row.get(col, ""))
                    if val:
                        path.append(val)
                        if val not in node_paths:
                            node_paths[val] = set()
                        node_paths[val].add(tuple(path[:-1]))

        # Detect cycles
        for node, paths in node_paths.items():
            for path in paths:
                if node in path:
                    path_list = list(path)
                    cycle_start_idx = path_list.index(node)
                    cycle_path = path_list[cycle_start_idx:] + [node]
                    cycle_info = {
                        "path": cycle_path,
                        "length": len(cycle_path),
                        "start_node": node,
                        "cycle_type": "ancestor_cycle"
                    }
                    loops.append(cycle_info)

        # Remove duplicate cycles
        unique_loops = []
        seen_paths: Set[Tuple[str, ...]] = set()
        for loop in loops:
            normalized_path = tuple(sorted(loop["path"]))
            if normalized_path not in seen_paths:
                seen_paths.add(normalized_path)
                unique_loops.append(loop)
                
        return unique_loops
        
    except Exception:
        return []


def detect_missing_red_flags(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Detect nodes that might need red flag indicators.
    
    Args:
        df: DataFrame with decision tree data
        
    Returns:
        List of missing red flag information dictionaries
    """
    try:
        if not isinstance(df, pd.DataFrame) or df.empty:
            return []
        if not any(col in df.columns for col in LEVEL_COLS):
            return []

        missing_red_flags = []
        
        # Check for nodes that might need red flags
        # This is a simplified implementation - in practice, you'd have business rules
        for level in range(1, MAX_LEVELS + 1):
            node_col = f"Node {level}"
            if node_col not in df.columns:
                continue
                
            # Get unique values at this level
            node_values = df[node_col].map(normalize_text).dropna()
            node_values = node_values[node_values != ""]
            unique_nodes = node_values.unique()
            
            for node in unique_nodes:
                # Simple heuristic: nodes with certain keywords might need red flags
                node_lower = node.lower()
                if any(keyword in node_lower for keyword in ["urgent", "critical", "emergency", "warning"]):
                    missing_red_flags.append({
                        "node": node,
                        "level": level,
                        "node_id": f"L{level}_{node}",
                        "suggested_action": "Add red flag indicator",
                        "reason": "Contains urgency keywords"
                    })
        
        return missing_red_flags
        
    except Exception:
        return []


def compute_validation_report(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute a comprehensive validation report.
    
    Args:
        df: DataFrame with decision tree data
        
    Returns:
        Dictionary containing validation summary and details
    """
    try:
        if not isinstance(df, pd.DataFrame) or df.empty:
            return {
                "summary": {
                    "total_orphans": 0,
                    "total_loops": 0,
                    "total_missing_red_flags": 0,
                    "total_issues": 0
                },
                "orphans": [],
                "loops": [],
                "missing_red_flags": []
            }

        # Run all validation checks
        orphans = detect_orphan_nodes(df)
        loops = detect_loops(df)
        missing_red_flags = detect_missing_red_flags(df)
        
        # Compute summary
        total_orphans = len(orphans)
        total_loops = len(loops)
        total_missing_red_flags = len(missing_red_flags)
        total_issues = total_orphans + total_loops + total_missing_red_flags
        
        summary = {
            "total_orphans": total_orphans,
            "total_loops": total_loops,
            "total_missing_red_flags": total_missing_red_flags,
            "total_issues": total_issues
        }
        
        return {
            "summary": summary,
            "orphans": orphans,
            "loops": loops,
            "missing_red_flags": missing_red_flags
        }
        
    except Exception:
        return {
            "summary": {
                "total_orphans": 0,
                "total_loops": 0,
                "total_missing_red_flags": 0,
                "total_issues": 0
            },
            "orphans": [],
            "loops": [],
            "missing_red_flags": []
        }


# Cached wrapper functions for UI layer
def get_cached_orphan_nodes(df: pd.DataFrame, cache_key: Tuple) -> List[Dict[str, Any]]:
    """
    Cached wrapper for detect_orphan_nodes.
    
    Args:
        df: DataFrame with decision tree data
        cache_key: Tuple for caching (should include sheet name and version)
        
    Returns:
        List of orphan node information dictionaries
    """
    # This function will be decorated with @st.cache_data in the UI layer
    # to avoid circular imports
    return detect_orphan_nodes(df)


def get_cached_loops(df: pd.DataFrame, cache_key: Tuple) -> List[Dict[str, Any]]:
    """
    Cached wrapper for detect_loops.
    
    Args:
        df: DataFrame with decision tree data
        cache_key: Tuple for caching (should include sheet name and version)
        
    Returns:
        List of loop information dictionaries
    """
    # This function will be decorated with @st.cache_data in the UI layer
    # to avoid circular imports
    return detect_loops(df)


def get_cached_validation_report(df: pd.DataFrame, cache_key: Tuple) -> Dict[str, Any]:
    """
    Cached wrapper for compute_validation_report.
    
    Args:
        df: DataFrame with decision tree data
        cache_key: Tuple for caching (should include sheet name and version)
        
    Returns:
        Dictionary containing validation summary and details
    """
    # This function will be decorated with @st.cache_data in the UI layer
    # to avoid circular imports
    return compute_validation_report(df)