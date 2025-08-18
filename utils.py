"""
utils.py

Shared constants and helper utilities used across the Decision Tree Builder app.

This module is intentionally Streamlit-free so it can be imported by both
pure logic modules and UI modules without side effects.
"""

from __future__ import annotations

from typing import Dict, List, Tuple, Optional, Set
import pandas as pd
import numpy as np


# ========= Canonical schema & settings =========

APP_VERSION_DEFAULT = "v6.3.0"

CANON_HEADERS: List[str] = [
    "Vital Measurement", "Node 1", "Node 2", "Node 3", "Node 4", "Node 5",
    "Diagnostic Triage", "Actions",
]
LEVEL_COLS: List[str] = ["Node 1","Node 2","Node 3","Node 4","Node 5"]
MAX_LEVELS: int = 5

# Friendly wording for the root parent label (Node 1 parents)
FRIENDLY_ROOT: str = "Top-level (Node 1) options"


# ========= Basic text / validation helpers =========

def normalize_text(x: object) -> str:
    """
    Return a stripped string, converting NaN/None to "".
    
    Args:
        x: Any object to convert to string
        
    Returns:
        Stripped string, empty string for None/NaN values
        
    Failure modes:
        - Returns empty string for any unconvertible objects
    """
    try:
        if x is None:
            return ""
        if isinstance(x, float) and np.isnan(x):
            return ""
    except Exception:
        return ""
    return str(x).strip()


def validate_headers(df: pd.DataFrame) -> bool:
    """
    Verify that the first len(CANON_HEADERS) columns match the canonical schema.
    
    Args:
        df: DataFrame to validate
        
    Returns:
        True if headers match canonical schema, False otherwise
        
    Failure modes:
        - Returns False for non-DataFrame inputs
        - Returns False for empty DataFrames
        - Returns False for DataFrames with insufficient columns
    """
    try:
        if not isinstance(df, pd.DataFrame) or df.empty:
            return False
        if len(df.columns) < len(CANON_HEADERS):
            return False
        return list(df.columns[:len(CANON_HEADERS)]) == CANON_HEADERS
    except Exception:
        return False


def ensure_canon_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure the DataFrame has all canonical columns (creates missing as "") and in order.
    Does not drop any extra columns the caller may have added.
    
    Args:
        df: DataFrame to ensure canonical columns for
        
    Returns:
        DataFrame with all canonical columns present and in order
        
    Failure modes:
        - Returns empty DataFrame with canonical columns for non-DataFrame inputs
        - Creates missing columns as empty strings
    """
    try:
        if not isinstance(df, pd.DataFrame):
            return pd.DataFrame(columns=CANON_HEADERS)
        
        df2 = df.copy()
        for c in CANON_HEADERS:
            if c not in df2.columns:
                df2[c] = ""
        return df2[CANON_HEADERS]
    except Exception:
        return pd.DataFrame(columns=CANON_HEADERS)


def drop_fully_blank_paths(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop rows where Vital Measurement + Node1..Node5 are all blank.
    
    Args:
        df: DataFrame to filter
        
    Returns:
        DataFrame with fully blank rows removed
        
    Failure modes:
        - Returns empty DataFrame for non-DataFrame inputs
        - Returns original DataFrame if required columns missing
    """
    try:
        if not isinstance(df, pd.DataFrame) or df.empty:
            return pd.DataFrame()
        
        node_block = ["Vital Measurement"] + LEVEL_COLS
        # Check if required columns exist
        missing_cols = [col for col in node_block if col not in df.columns]
        if missing_cols:
            return df.copy()
        
        mask_blank = df[node_block].apply(lambda r: all(normalize_text(v) == "" for v in r), axis=1)
        return df[~mask_blank].copy()
    except Exception:
        return df.copy() if isinstance(df, pd.DataFrame) else pd.DataFrame()


# ========= Keys & paths =========

def level_key_tuple(level: int, parent: Tuple[str, ...]) -> str:
    """
    Build the canonical store key for a parent tuple at a given level:
        "L{level}|<ROOT>" or "L{level}|A>B>C"
    """
    return f"L{level}|" + (">".join(parent) if parent else "<ROOT>")


def parent_key_from_row_strict(row: pd.Series, upto_level: int) -> Optional[Tuple[str, ...]]:
    """
    For a row and a target 'upto_level', return the parent tuple of length (upto_level-1).
    If any required Node column is blank, return None.
    L=1 => parent is () (root).
    
    Args:
        row: DataFrame row to extract parent from
        upto_level: Target level to build parent for
        
    Returns:
        Parent tuple or None if incomplete
        
    Failure modes:
        - Returns None for invalid inputs
        - Returns None if required columns missing
    """
    try:
        if upto_level <= 1:
            return tuple()
        parent: List[str] = []
        for c in LEVEL_COLS[:upto_level-1]:
            v = normalize_text(row.get(c, ""))
            if v == "":
                return None
            parent.append(v)
        return tuple(parent)
    except Exception:
        return None


def order_decision_tree(df: pd.DataFrame) -> pd.DataFrame:
    """
    Order decision tree DataFrame so that:
      - All Node 1 branches appear first, grouped together
      - Children follow directly after their parents
      - Ordering cascades through Node 2 → Node 3 → Node 4 → Node 5
    Falls back gracefully if structure is incomplete.
    
    Args:
        df: DataFrame to order
        
    Returns:
        Ordered DataFrame with logical tree structure
        
    Failure modes:
        - Returns original DataFrame for non-DataFrame inputs
        - Returns original DataFrame if required columns missing
        - Returns original DataFrame if ordering fails
    """
    try:
        # Defensive: return df if invalid
        if not isinstance(df, pd.DataFrame) or df.empty:
            return df

        # Defensive: check for required node columns
        node_cols = [c for c in LEVEL_COLS if c in df.columns]
        if not node_cols:
            return df

        # Build an ordered list of indices
        ordered_indices = []
        visited = set()

        def add_branch(parent_path, depth=1):
            """
            Recursively add parent + children in order
            """
            nonlocal ordered_indices
            parent_col = LEVEL_COLS[depth - 1]  # Use LEVEL_COLS instead of hardcoded names
            if parent_col not in df.columns:
                return

            # Filter rows where this node matches parent_path
            mask = df[parent_col] == parent_path if parent_path else df[parent_col].notna()
            rows = df[mask]

            for idx, row in rows.iterrows():
                if idx not in visited:
                    ordered_indices.append(idx)
                    visited.add(idx)

                    # Recurse into next depth
                    if depth < MAX_LEVELS:
                        next_col = LEVEL_COLS[depth] if depth < len(LEVEL_COLS) else None
                        if next_col and next_col in df.columns:
                            next_val = row.get(next_col, None)
                            if pd.notna(next_val) and normalize_text(next_val):
                                add_branch(next_val, depth + 1)

        # Start with Node 1 roots
        if LEVEL_COLS[0] in df.columns:
            roots = df[LEVEL_COLS[0]].dropna().unique()
            for root in roots:
                if normalize_text(root):
                    add_branch(root, 1)

        # Add any remaining unvisited rows
        for idx in df.index:
            if idx not in visited:
                ordered_indices.append(idx)

        # Reindex DataFrame
        if ordered_indices:
            return df.loc[ordered_indices].reset_index(drop=True)
        else:
            return df
            
    except Exception:
        return df


def friendly_parent_label(level: int, parent_tuple: Tuple[str, ...]) -> str:
    """
    Human-friendly label for the parent path feeding Node {level} children.
    Provides FRIENDLY_ROOT for the top-level.
    """
    if level == 1 and not parent_tuple:
        return FRIENDLY_ROOT
    return " > ".join(parent_tuple) if parent_tuple else FRIENDLY_ROOT


def enforce_k_five(opts: List[str]) -> List[str]:
    """
    Trim/pad a list of options to exactly 5, removing blanks and normalizing text.
    Pads with "" to length 5.
    """
    clean = [normalize_text(o) for o in (opts or []) if normalize_text(o) != ""]
    if len(clean) > 5:
        clean = clean[:5]
    elif len(clean) < 5:
        clean = clean + [""]*(5-len(clean))
    return clean


# ========= Store (parent -> children) =========

def infer_branch_options(df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Build a map: "L{level}|{parent_path or <ROOT>}" -> [children].
    Deduplicates while preserving first-seen order across the entire sheet.
    """
    store: Dict[str, List[str]] = {}
    if df is None or df.empty:
        return store

    for level in range(1, MAX_LEVELS+1):
        parent_to_children: Dict[Tuple[str, ...], List[str]] = {}
        child_col = LEVEL_COLS[level-1]
        for _, row in df.iterrows():
            child = normalize_text(row.get(child_col, ""))
            if child == "":
                continue
            parent = parent_key_from_row_strict(row, level)
            if parent is None:
                continue
            parent_to_children.setdefault(parent, []).append(child)

        for parent, children in parent_to_children.items():
            uniq, seen = [], set()
            for c in children:
                if c not in seen:
                    seen.add(c); uniq.append(c)
            store[level_key_tuple(level, parent)] = uniq

    return store


def infer_branch_options_with_overrides(df: pd.DataFrame, overrides: Dict[str, List[str]]) -> Dict[str, List[str]]:
    """
    Merge inferred store with overrides (overrides win).
    """
    base = infer_branch_options(df)
    merged = dict(base)
    if overrides:
        for k, v in overrides.items():
            if not isinstance(v, list):
                vals = [normalize_text(v)]
            else:
                vals = [normalize_text(x) for x in v]
            merged[k] = vals
    return merged


def build_label_children_index(store: Dict[str, List[str]]) -> Dict[Tuple[int, str], List[str]]:
    """
    Build (level, parent_label) -> children list.

    Interpretation:
      - For key "L{L}|{parent_tuple}", the children in 'store[key]' are the options for Node L.
      - The "parent label" for level L is:
          * "<ROOT>" when L==1
          * the last element of parent_tuple when L>1
    """
    idx: Dict[Tuple[int, str], List[str]] = {}
    for key, children in store.items():
        if "|" not in key:
            continue
        lvl_s, path = key.split("|", 1)
        try:
            L = int(lvl_s[1:])
        except Exception:
            continue
        if not (1 <= L <= MAX_LEVELS):
            continue
        if path == "<ROOT>":
            parent_label = "<ROOT>"
        else:
            parent_tuple = tuple(path.split(">"))
            parent_label = parent_tuple[-1] if parent_tuple else "<ROOT>"
        idx[(L, parent_label)] = [normalize_text(c) for c in children if normalize_text(c) != ""]
    return idx


# ========= DataFrame convenience =========

def vital_measurements(df: pd.DataFrame) -> List[str]:
    """
    Unique, normalized Vital Measurement names in ascending order (blanks removed).
    """
    if df is None or df.empty or "Vital Measurement" not in df.columns:
        return []
    vals = (
        df["Vital Measurement"]
        .map(normalize_text)
        .replace("", np.nan)
        .dropna()
        .unique()
        .tolist()
    )
    return sorted(vals)


def children_counts(df: pd.DataFrame, vm: str, parent: Tuple[str, ...], next_level: int) -> Dict[str, int]:
    """
    Count Node {next_level} children occurrences under (vm, parent).
    """
    if df is None or df.empty or not (1 <= next_level <= MAX_LEVELS):
        return {}
    m = (df["Vital Measurement"].map(normalize_text) == normalize_text(vm))
    for i, val in enumerate(parent, 1):
        m = m & (df[f"Node {i}"].map(normalize_text) == normalize_text(val))
    sub = df[m].copy()
    if sub.empty:
        return {}
    col = f"Node {next_level}"
    s = sub[col].map(normalize_text)
    vc = s[s != ""].value_counts(dropna=True)
    return {k: int(v) for k, v in vc.items()}


def cluster_by_node(df: pd.DataFrame, node_col: str) -> pd.DataFrame:
    """
    Return a copy sorted such that identical labels in `node_col` appear contiguously.
    Stable on other columns.

    Example: cluster_by_node(df, "Node 2")
    """
    if node_col not in df.columns:
        return df.copy()
    df2 = df.copy()
    df2[node_col] = df2[node_col].map(normalize_text)
    return df2.sort_values([node_col] + [c for c in CANON_HEADERS if c != node_col], kind="stable").reset_index(drop=True)


# ========= Validation & scoring =========

def compute_parent_depth_score(df: pd.DataFrame) -> Tuple[int, int]:
    """
    Compute how many parent nodes have exactly 5 children (the ideal branching factor).
    
    Returns:
        Tuple[int, int]: (ok_count, total_count) where ok_count is parents with exactly 5 children
    """
    store = infer_branch_options(df)
    total = 0
    ok = 0
    for level in range(1, MAX_LEVELS+1):
        parents = set()
        for _, row in df.iterrows():
            p = parent_key_from_row_strict(row, level)
            if p is not None:
                parents.add(p)
        for p in parents:
            total += 1
            key = f"L{level}|" + (">".join(p) if p else "<ROOT>")
            if len([x for x in store.get(key, []) if normalize_text(x) != ""]) == 5:
                ok += 1
    return ok, total


def compute_row_path_score(df: pd.DataFrame) -> Tuple[int, int]:
    """
    Compute how many rows have complete paths (all Node columns filled).
    
    Returns:
        Tuple[int, int]: (complete_count, total_count) where complete_count is rows with full paths
    """
    if df.empty:
        return (0, 0)
    nodes = df[LEVEL_COLS].applymap(normalize_text)
    full = nodes.ne("").all(axis=1)
    return int(full.sum()), int(len(df))
