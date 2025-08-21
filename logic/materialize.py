# logic/materialize.py
"""
Materializer utility for applying canonical children sets using monolith row-multiplication semantics.
"""

import pandas as pd
from typing import List, Tuple, Optional, Dict, Any
from utils.constants import ROOT_COL, LEVEL_COLS, MAX_LEVELS, MAX_CHILDREN_PER_PARENT, ROOT_PARENT_LABEL
from utils.helpers import normalize_text, normalize_child_set


def _path_cols(level: int) -> List[str]:
    """Get columns for parent prefix at given level.
    
    Args:
        level: Level 1..5 (child lives in Node L)
        
    Returns:
        List of column names for parent prefix: [ROOT_COL, Node1, ..., Node(L-1)]
    """
    return [ROOT_COL] + LEVEL_COLS[:max(0, level-1)]


def _child_col(level: int) -> str:
    """Get the child column name for a given level.
    
    Args:
        level: Level 1..5
        
    Returns:
        Column name where children live (e.g., 'Node 1' for level 1)
    """
    return LEVEL_COLS[level-1]  # Node L


def _deeper_cols(level: int) -> List[str]:
    """Get columns that should be cleared when materializing at a given level.
    
    Args:
        level: Level where we're setting children
        
    Returns:
        List of columns to clear: [Node L+1, ..., Node 5, Diagnostic Triage, Actions]
    """
    start = level
    return LEVEL_COLS[start:] + ["Diagnostic Triage", "Actions"]


def _row_matches_parent_prefix(row: pd.Series, parent_prefix: List[str]) -> bool:
    """Check if a row matches a given parent prefix.
    
    Args:
        row: DataFrame row to check
        parent_prefix: List of expected values for parent prefix columns
        
    Returns:
        True if row matches the parent prefix
    """
    for col, val in zip(_path_cols(len(parent_prefix)-1), parent_prefix[:-1]):
        if normalize_text(row.get(col, "")) != normalize_text(val):
            return False
    # last label of parent prefix is parent_label itself (for L>=2)
    return True


def _parent_path_string(prefix_vals: List[str]) -> str:
    """Convert parent prefix values to a string representation.
    
    Args:
        prefix_vals: List of values for parent prefix columns
        
    Returns:
        String representation with non-empty values joined by '>'
    """
    # For grouping: join non-empty values
    return ">".join([v for v in prefix_vals if normalize_text(v)])


def materialize_children_for_label_group(
    df: pd.DataFrame,
    level: int,
    parent_label: str,
    new_children: List[str],
) -> pd.DataFrame:
    """
    Apply canonical children to all parent paths at a given (level, parent_label),
    using monolith row-multiplication semantics.
    
    Args:
        df: Input DataFrame with decision tree data
        level: Level 1..5 where children will be set
        parent_label: Parent label to match (for L==1, this is synthetic <ROOT>)
        new_children: List of children to apply (will be normalized/deduped/capped to 5)
        
    Returns:
        New DataFrame with materialized children following monolith rules
    """
    assert 1 <= level <= MAX_LEVELS
    out = df.copy()
    kids = normalize_child_set(new_children)
    if out.empty or not kids:
        return out

    # Build a prefix matcher for each parent path at this level that has the requested parent_label
    # parent prefix values = [ROOT, Node1, ..., Node(L-1)]
    # note: ROOT lives in ROOT_COL (Vital Measurement)
    used_cols = [ROOT_COL] + LEVEL_COLS
    for c in used_cols:
        if c not in out.columns:
            out[c] = ""

    # Enumerate all rows with their path values
    rows = out[used_cols].astype(str).map(normalize_text)

    # Build all parent paths at L that end with parent_label
    # parent prefix length in columns: ROOT + Node1..Node(L-1) => (L) cells
    # parent_label lives in: ROOT if L==1? No — by design L==1's parent is synthetic ROOT.
    parent_prefix_cols = _path_cols(level)
    parent_paths = rows[parent_prefix_cols].apply(lambda s: [normalize_text(x) for x in s.tolist()], axis=1)

    # Determine which rows belong to any parent whose last element equals parent_label (for L>=2)
    # For L==1, parent_label is synthetic <ROOT> and applies to all rows.
    mask_rows_for_label = []
    if level == 1:
        mask_rows_for_label = [True] * len(out)
    else:
        mask_rows_for_label = [
            (pp and normalize_text(pp[-1]) == normalize_text(parent_label)) for pp in parent_paths.tolist()
        ]

    # For each distinct parent path (prefix values), compress its current children and re-expand to kids
    # Build map: parent_prefix_tuple -> list(row_indices)
    from collections import defaultdict
    groups = defaultdict(list)
    for i, use in enumerate(mask_rows_for_label):
        if not use:
            continue
        prefix_vals = parent_paths.iat[i]  # list length == len(parent_prefix_cols)
        key = tuple(prefix_vals)  # stable
        groups[key].append(i)

    # Helper to clear deeper columns
    def _clear_deeper_block(df_block: pd.DataFrame, lvl: int):
        """Clear deeper columns in a DataFrame block."""
        for col in _deeper_cols(lvl):
            if col in df_block.columns:
                df_block[col] = ""
        return df_block

    # We will rebuild rows by accumulating into a list of DataFrames then concat
    rebuilt = []
    consumed = set()

    for prefix, idxs in groups.items():
        # rows currently under this parent (in any order)
        subset = out.iloc[idxs].copy()
        consumed.update(idxs)

        # sort subset by appearance to keep stable order; collapse to the first row as base
        base = subset.iloc[[0]].copy()

        # set the prefix columns to the normalized prefix values (clean)
        for col, val in zip(parent_prefix_cols, prefix):
            base[col] = normalize_text(val)

        # clear deeper columns on the base
        base = _clear_deeper_block(base, level)

        # slot first child into base row
        base[_child_col(level)] = kids[0]

        rebuilt.append(base)

        # add k-1 extra rows for remaining kids
        for ch in kids[1:]:
            extra = base.copy()
            extra[_child_col(level)] = ch
            rebuilt.append(extra)

    # keep all rows not in consumed as-is
    keep_mask = [i not in consumed for i in range(len(out))]
    kept = out.iloc[keep_mask].copy()
    result = pd.concat([kept] + rebuilt, ignore_index=True)

    return result


def materialize_children_for_single_parent(
    df: pd.DataFrame,
    level: int,
    parent_path: str,
    new_children: List[str],
) -> pd.DataFrame:
    """
    Same semantics as the label-group materializer, but targets only the given parent_path
    (parent_path is 'root>Node1>...>Node(L-1)' for L>=2; for L==1 treat as <ROOT>).
    """
    if df is None or df.empty:
        return df
    kids = normalize_child_set(new_children)
    if not kids:
        return df

    out = df.copy()
    used_cols = [ROOT_COL] + LEVEL_COLS + ["Diagnostic Triage", "Actions"]
    for c in used_cols:
        if c not in out.columns:
            out[c] = ""

    # Build each row's parent_path at this level
    parent_cols = [ROOT_COL] + LEVEL_COLS[:max(0, level-1)]
    import numpy as np

    norm_parent = (
        out[parent_cols]
        .astype(str)
        .map(normalize_text)
        .agg(lambda s: ">".join([x for x in s.tolist() if x]), axis=1)
    )

    target_key = parent_path if level >= 2 else "<ROOT>"

    # Indices belonging to this parent_path
    if level == 1:
        idxs = list(range(len(out)))  # L1 parent is synthetic ROOT; apply to all top rows
    else:
        idxs = np.where(norm_parent == target_key)[0].tolist()

    if not idxs:
        return out

    # CLEAR deeper columns function
    def _clear_deeper_block(df_block: pd.DataFrame, lvl: int):
        deeper = LEVEL_COLS[lvl:] + ["Diagnostic Triage", "Actions"]
        for col in deeper:
            if col in df_block.columns:
                df_block[col] = ""
        return df_block

    # Rebuild block for this single parent
    subset = out.iloc[idxs].copy()
    subset = subset.sort_index()
    base = subset.iloc[[0]].copy()

    # Ensure parent prefix is normalized
    prefix_cols = parent_cols
    prefix_vals = (
        base[prefix_cols]
        .astype(str)
        .map(normalize_text)
        .iloc[0]
        .tolist()
    )
    # Clear deeper cols and assign first child
    base = _clear_deeper_block(base, level)
    base[LEVEL_COLS[level-1]] = kids[0]

    rebuilt = [base]
    for ch in kids[1:]:
        extra = base.copy()
        extra[LEVEL_COLS[level-1]] = ch
        rebuilt.append(extra)

    # Keep rows not in idxs and append rebuilt
    keep_mask = [i not in set(idxs) for i in range(len(out))]
    kept = out.iloc[keep_mask].copy()
    result = pd.concat([kept] + rebuilt, ignore_index=True)
    return result


def materialize_children_for_label_across_tree(
    df: pd.DataFrame,
    label: str,
    new_children: List[str],
    index_summary: Dict[Tuple[int, str], Dict[str, Any]] | None = None,
) -> pd.DataFrame:
    """
    Apply the same 5-set to *all* parent_paths whose parent_label == label across the entire tree.
    Implementation:
      - Find all (level, parent_path) whose last segment equals label (L>=2)
      - For L==1 treat label as ROOT and update Node 1 set
      - For each: use single-parent materializer to avoid unintended cross-parent overwrites
    """
    kids = normalize_child_set(new_children)
    if df is None or df.empty or not kids:
        return df

    out = df.copy()

    # If index_summary provided (summary from summarize_children_sets), we can iterate its keys;
    # otherwise recompute against df here (but prefer passing it).
    if index_summary is None:
        from .tree import build_parent_child_index_with_root, summarize_children_sets
        index_summary = summarize_children_sets(build_parent_child_index_with_root(out))

    targets: List[Tuple[int, str]] = []
    for (L, parent_path), _info in index_summary.items():
        if L == 1 and label == ROOT_PARENT_LABEL:
            targets.append((L, parent_path))
        elif L >= 2:
            plabel = parent_path.split(">")[-1] if parent_path else ""
            if normalize_text(plabel) == normalize_text(label):
                targets.append((L, parent_path))

    # Apply per target using single-parent materializer
    for (L, pth) in targets:
        out = materialize_children_for_single_parent(out, L, pth if L >= 2 else "<ROOT>", kids)

    return out






