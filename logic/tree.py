# logic/tree.py
"""
Pure logic functions for tree manipulation operations.
No Streamlit dependencies - can be imported by both logic and UI modules.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional

from utils.constants import LEVEL_COLS, MAX_LEVELS, ROOT_PARENT_LABEL
from utils.helpers import normalize_text, normalize_child_set, validate_headers


def infer_branch_options(df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Infer branch options for each level in the decision tree.
    
    Args:
        df: DataFrame with decision tree data
        
    Returns:
        Dictionary mapping level keys to lists of possible values
    """
    try:
        if not isinstance(df, pd.DataFrame) or df.empty:
            return {}
        
        # Only validate headers if they exist, but don't fail if they don't
        # This allows the function to work with partial DataFrames for testing
        has_canonical_headers = validate_headers(df) if len(df.columns) >= len(LEVEL_COLS) else False
        
        store = {}
        
        # Ensure all Node columns exist and normalize ragged rows
        df_normalized = df.copy()
        for level in range(1, MAX_LEVELS + 1):
            node_col = f"Node {level}"
            if node_col not in df_normalized.columns:
                df_normalized[node_col] = ""
            else:
                # Normalize node columns
                df_normalized[node_col] = df_normalized[node_col].map(normalize_text)
        
        # Always build L1|<ROOT> from all non-empty Node 1 labels
        node1_col = LEVEL_COLS[0]  # "Node 1"
        if node1_col in df_normalized.columns:
            node1_values = df_normalized[node1_col].dropna()
            node1_values = node1_values[node1_values != ""]
            unique_node1_values = sorted(node1_values.unique())
            if unique_node1_values:
                store[f"L1|{ROOT_PARENT_LABEL}"] = unique_node1_values
        
        # Process each level
        for level in range(1, MAX_LEVELS + 1):
            node_col = f"Node {level}"
            if node_col not in df_normalized.columns:
                continue
            
            # Get unique values at this level
            values = df_normalized[node_col].dropna()
            values = values[values != ""]
            unique_values = sorted(values.unique())
            
            if unique_values:
                # Store as level key
                level_key = f"L{level}|"
                store[level_key] = unique_values
                
                # Also store with parent context
                if level > 1:
                    parent_cols = [f"Node {i}" for i in range(1, level)]
                    if all(col in df_normalized.columns for col in parent_cols):
                        # Group by parent path
                        parent_paths = df_normalized[parent_cols].apply(
                            lambda r: tuple(v for v in r), axis=1
                        )
                        parent_paths = parent_paths[parent_paths.apply(
                            lambda x: all(v != "" for v in x)
                        )]
                        
                        for parent_path in parent_paths.unique():
                            if parent_path:
                                key = f"L{level}|" + ">".join(parent_path)
                                mask = parent_paths == parent_path
                                children = df_normalized.loc[mask, node_col]
                                children = children[children != ""]
                                store[key] = sorted(children.unique())
        
        return store
        
    except Exception:
        return {}


def infer_branch_options_with_overrides(df: pd.DataFrame, overrides: Dict) -> Dict[str, List[str]]:
    """
    Infer branch options with overrides applied.
    
    Args:
        df: DataFrame with decision tree data
        overrides: Dictionary of override values
        
    Returns:
        Dictionary mapping level keys to lists of possible values
    """
    try:
        base_options = infer_branch_options(df)
        
        # Apply overrides
        for override_key, override_values in overrides.items():
            if isinstance(override_key, tuple) and len(override_key) >= 2:
                level, parent_path = override_key[0], override_key[1:]
                
                # Build the key
                if parent_path:
                    key = f"L{level}|" + ">".join(parent_path)
                else:
                    key = f"L{level}|"
                
                # Update with override values
                if key in base_options:
                    base_options[key] = list(override_values)
                else:
                    base_options[key] = list(override_values)
        
        return base_options
        
    except Exception:
        return {}


def build_label_children_index(store: Dict[str, List[str]]) -> Dict[str, Dict[str, List[str]]]:
    """
    Build an index of labels to their children.
    
    Args:
        store: Dictionary from infer_branch_options
        
    Returns:
        Dictionary mapping labels to their children
    """
    try:
        index = {}
        
        for key, values in store.items():
            if "|" in key:
                parts = key.split("|")
                if len(parts) == 2:
                    level_info = parts[0]
                    parent_path = parts[1]
                    
                    if parent_path:
                        # This is a child level
                        parent_label = parent_path.split(">")[-1]
                        if parent_label not in index:
                            index[parent_label] = []
                        index[parent_label].extend(values)
        
        return index
        
    except Exception:
        return {}


def _rows_match_parent(row: pd.Series, parent_key: Tuple[str, ...], level: int) -> bool:
    """Check if a row matches a parent key."""
    try:
        if level <= 1:
            return True
        
        for i, expected_value in enumerate(parent_key):
            col = LEVEL_COLS[i]
            if col in row.index:
                actual_value = normalize_text(row.get(col, ""))
                if actual_value != expected_value:
                    return False
        
        return True
        
    except Exception:
        return False


def _present_children_at_level(df: pd.DataFrame, parent_key: Tuple[str, ...], level: int) -> List[str]:
    """Get children present at a specific level for a parent."""
    try:
        if level > MAX_LEVELS:
            return []
        
        node_col = f"Node {level}"
        if node_col not in df.columns:
            return []
        
        # Filter rows matching parent
        mask = df.apply(lambda row: _rows_match_parent(row, parent_key, level - 1), axis=1)
        matching_rows = df[mask]
        
        # Get unique values
        values = matching_rows[node_col].map(normalize_text).dropna()
        values = values[values != ""]
        return sorted(values.unique())
        
    except Exception:
        return []


def _find_anchor_index(store: Dict[str, List[str]], anchor_key: str) -> Optional[int]:
    """Find the index of an anchor in the store."""
    try:
        if anchor_key in store:
            return 0  # Anchor found at index 0
        
        # Search for anchor in other keys
        for key, values in store.items():
            if anchor_key in values:
                return values.index(anchor_key)
        
        return None
        
    except Exception:
        return None


def _emit_row_from_prefix(prefix: Tuple[str, ...], level: int, 
                         store: Dict[str, List[str]]) -> List[Dict[str, str]]:
    """Emit rows from a prefix."""
    try:
        if level > MAX_LEVELS:
            return []
        
        rows = []
        node_col = f"Node {level}"
        
        # Get children for this prefix
        children = _present_children_at_level(
            pd.DataFrame(), prefix, level
        )
        
        for child in children:
            row_data = {}
            for i, val in enumerate(prefix):
                col = LEVEL_COLS[i]
                row_data[col] = val
            
            row_data[node_col] = child
            rows.append(row_data)
        
        return rows
        
    except Exception:
        return []


def _children_from_store(store: Dict[str, List[str]], parent_key: Tuple[str, ...], 
                        level: int) -> List[str]:
    """Get children from store for a parent at a specific level."""
    try:
        if level <= 0:
            return []
        
        # Build the key
        if parent_key:
            key = f"L{level}|" + ">".join(parent_key)
        else:
            key = f"L{level}|"
        
        return store.get(key, [])
        
    except Exception:
        return []


def expand_parent_nextnode_anchor_reuse_for_vm(df: pd.DataFrame, parent_key: Tuple[str, ...], 
                                             level: int, store: Dict[str, List[str]]) -> List[Dict[str, str]]:
    """
    Expand a parent node with next level children, reusing anchors.
    
    Args:
        df: DataFrame with decision tree data
        parent_key: Parent key tuple
        level: Current level
        store: Branch options store
        
    Returns:
        List of row dictionaries
    """
    try:
        if level > MAX_LEVELS:
            return []
        
        children = _children_from_store(store, parent_key, level)
        if not children:
            return []
        
        rows = []
        for child in children:
            row_data = {}
            
            # Add parent values
            for i, val in enumerate(parent_key):
                col = LEVEL_COLS[i]
                row_data[col] = val
            
            # Add child value
            node_col = f"Node {level}"
            row_data[node_col] = child
            
            rows.append(row_data)
        
        return rows
        
    except Exception:
        return []


def cascade_anchor_reuse_full(df: pd.DataFrame, store: Dict[str, List[str]]) -> List[Dict[str, str]]:
    """
    Perform full cascade with anchor reuse.
    
    Args:
        df: DataFrame with decision tree data
        store: Branch options store
        
    Returns:
        List of expanded row dictionaries
    """
    try:
        expanded_rows = []
        
        # Start with root level
        root_children = _children_from_store(store, (), 1)
        
        for root_child in root_children:
            # Expand this root child
            parent_key = (root_child,)
            rows = expand_parent_nextnode_anchor_reuse_for_vm(df, parent_key, 2, store)
            expanded_rows.extend(rows)
            
            # Continue expanding for deeper levels
            for level in range(3, MAX_LEVELS + 1):
                new_rows = []
                for row in rows:
                    # Extract parent key from this row
                    row_parent_key = tuple(
                        normalize_text(row.get(f"Node {i}", ""))
                        for i in range(1, level)
                        if normalize_text(row.get(f"Node {i}", "")) != ""
                    )
                    
                    if len(row_parent_key) == level - 1:
                        level_rows = expand_parent_nextnode_anchor_reuse_for_vm(
                            df, row_parent_key, level, store
                        )
                        new_rows.extend(level_rows)
                
                if not new_rows:
                    break
                
                rows = new_rows
                expanded_rows.extend(new_rows)
        
        return expanded_rows
        
    except Exception:
        return []


def build_raw_plus_v630(df: pd.DataFrame, overrides: Dict = None, 
                        include_scope: bool = True, 
                        edited_keys_for_sheet: List[str] = None) -> pd.DataFrame:
    """
    Build raw plus v630 decision tree.
    
    Args:
        df: Input DataFrame
        overrides: Branch overrides
        include_scope: Whether to include scope information
        edited_keys_for_sheet: List of edited keys
        
    Returns:
        Enhanced DataFrame
    """
    try:
        if overrides is None:
            overrides = {}
        
        # Get branch options
        store = infer_branch_options_with_overrides(df, overrides)
        
        # Build expanded rows
        expanded_rows = cascade_anchor_reuse_full(df, store)
        
        # Convert to DataFrame
        if expanded_rows:
            result_df = pd.DataFrame(expanded_rows)
            
            # Ensure all canonical columns are present
            for col in ["Vital Measurement"] + LEVEL_COLS + ["Diagnostic Triage", "Actions"]:
                if col not in result_df.columns:
                    result_df[col] = ""
            
            return result_df
        else:
            return df.copy()
            
    except Exception:
        return df.copy()


def order_decision_tree(df: pd.DataFrame) -> pd.DataFrame:
    """
    Order the decision tree for consistent display.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Ordered DataFrame
    """
    try:
        if df.empty:
            return df
        
        # Sort by Vital Measurement first, then by Node columns
        sort_cols = ["Vital Measurement"] + LEVEL_COLS
        sort_cols = [col for col in sort_cols if col in df.columns]
        
        if sort_cols:
            df_sorted = df.sort_values(sort_cols, kind="stable")
            return df_sorted
        
        return df
        
    except Exception:
        return df


# Cached wrapper for infer_branch_options
def get_cached_branch_options(df: pd.DataFrame, cache_key: Tuple) -> Dict[str, List[str]]:
    """
    Cached wrapper for infer_branch_options.
    
    Args:
        df: DataFrame with decision tree data
        cache_key: Tuple for caching (should include sheet name and version)
        
    Returns:
        Dictionary mapping level keys to lists of possible values
    """
    # This function will be decorated with @st.cache_data in the UI layer
    # to avoid circular imports
    return infer_branch_options(df)


def set_level1_children(df: pd.DataFrame, children: list[str]) -> pd.DataFrame:
    """
    Return a new DataFrame where Node 1 values are restricted to the given 'children'
    (normalized, deduped, capped to MAX_CHILDREN_PER_PARENT). Rows with Node 1
    values not in the new set are mapped to the first child (or left blank if empty df).
    This is a simple, deterministic policy mirroring monolith behavior.
    """
    new_children = normalize_child_set(children)
    if df is None or df.empty:
        return df
    if not new_children:
        return df
    df2 = df.copy()
    n1 = LEVEL_COLS[0]
    # Map any Node-1 value not in new_children to the 1st element (stable remap)
    preferred = new_children[0]
    df2[n1] = df2[n1].apply(lambda x: preferred if normalize_text(x) not in new_children else normalize_text(x))
    return df2
