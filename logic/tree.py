# logic/tree.py
"""
Pure logic functions for tree manipulation operations.
No Streamlit dependencies - can be imported by both logic and UI modules.

DUPLICATE ROWS POLICY:
This app treats each row as a full path. Duplicate prefixes (first N nodes) are expected; 
we compute unique children per parent by set semantics. Downstream multiplication does not 
inflate child counts.

The indexer maintains counters for display purposes but uses unique labels to judge 
child-set size and conflicts. This ensures accurate conflict detection regardless of 
row duplication patterns.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from collections import Counter, defaultdict

from utils.constants import LEVEL_COLS, MAX_LEVELS, ROOT_PARENT_LABEL, MAX_CHILDREN_PER_PARENT
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


def build_parent_child_index(df: pd.DataFrame) -> Dict[Tuple[int, str], Counter]:
    """
    For each level L=1..MAX_LEVELS, build a mapping from (L, parent_path) to Counter(child -> row_count).
    - L=1: parent_path is <ROOT>, child is Node 1
    - L>=2: parent_path is 'Node 1>...>Node L-1' (non-empty segments only)
    Each counter counts how many rows contribute that child, but the caller should treat presence as unique.
    This is tolerant of row multiplication at deeper levels.
    """
    
    idx: Dict[Tuple[int, str], Counter] = {}
    if df is None or df.empty:
        return idx

    # Normalize & protect missing columns
    nodes = [c for c in LEVEL_COLS if c in df.columns]
    if not nodes:
        return idx

    # L=1 (ROOT → Node1)
    c0 = nodes[0]
    key_root = (1, ROOT_PARENT_LABEL)
    ctr_root = Counter()
    for val in df[c0].astype(str).tolist():
        ch = normalize_text(val)
        if ch:
            ctr_root[ch] += 1
    if ctr_root:
        idx[key_root] = ctr_root

    # L=2..MAX_LEVELS
    for L in range(2, min(MAX_LEVELS, len(nodes)) + 1):
        parent_cols = nodes[:L-1]
        child_col = nodes[L-1]
        # Build parent_path string per row
        parent_paths = df[parent_cols].astype(str).map(normalize_text).agg(lambda s: ">".join([x for x in s.tolist() if x]), axis=1)
        children = df[child_col].astype(str).map(normalize_text)
        # Accumulate counts
        for pth, ch in zip(parent_paths.tolist(), children.tolist()):
            if not pth or not ch:
                continue
            key = (L, pth)
            idx.setdefault(key, Counter())
            idx[key][ch] += 1
    return idx


def summarize_children_sets(idx: Dict[Tuple[int, str], Counter]) -> Dict[Tuple[int, str], Dict]:
    """
    Convert counters to ordered unique child lists and simple metrics.
    Returns mapping: (level, parent_path) -> {
      'children': children,
      'count': int,   # unique child count
      'over5': bool,
      'exact5': bool
    }
    """
    
    out: Dict[Tuple[int, str], Dict] = {}
    for key, ctr in idx.items():
        # preserve order by frequency desc then alpha
        items = sorted(list(ctr.items()), key=lambda x: (-x[1], x[0]))
        children = [c for c, _ in items]
        children = [c for c in children if c]  # strip blanks (already normalized)
        out[key] = {
            "children": children,
            "count": len(children),
            "over5": len(children) > MAX_CHILDREN_PER_PARENT,
            "exact5": len(children) == MAX_CHILDREN_PER_PARENT,
        }
    return out


def group_by_parent_label_at_level(summary: Dict[Tuple[int, str], Dict]) -> Dict[Tuple[int, str], List[Tuple[str, List[str]]]]:
    """
    For a given level L>=2, group parents by their **last label** (parent label).
    Example: all parents at L=3 whose last label == 'Confusion' are grouped together,
    yielding a list of (parent_path, child_set). This lets us detect 'same label, different 5-sets'.
    For L=1, the label is ROOT (only one group).
    Returns mapping: (L, parent_label) -> [ (parent_path, children_list) ... ]
    """
    
    buckets: Dict[Tuple[int, str], List[Tuple[str, List[str]]]] = defaultdict(list)
    for (L, parent_path), info in summary.items():
        if L == 1:
            buckets[(1, ROOT_PARENT_LABEL)].append((parent_path, info["children"]))
            continue
        # parent_path like 'Node1>Node2>...>Node(L-1)'
        label = parent_path.split(">")[-1] if parent_path else ""
        if label:
            buckets[(L, label)].append((parent_path, info["children"]))
    return buckets


def find_label_set_mismatches(summary: Dict[Tuple[int, str], Dict]) -> Dict[Tuple[int, str], Dict]:
    """
    For each (L, parent_label), check if all parent paths sharing that label have the **same 5 children**.
    Report groups where:
      - some have >5 children
      - or they do not all share the exact same 5-set
    Returns: (L, parent_label) -> {
      'variants': List[ {'parent_path': str, 'children': List[str]} ],
      'has_over5': bool,
      'all_exact5_same': bool
    }
    """
    
    buckets = group_by_parent_label_at_level(summary)
    out: Dict[Tuple[int, str], Dict] = {}
    for key, items in buckets.items():
        sets = [tuple(sorted(v[1])) for v in items]  # sorted for set-equality
        unique_sets = set(sets)
        has_over5 = any(len(v[1]) > MAX_CHILDREN_PER_PARENT for v in items)
        all_exact5_same = (len(unique_sets) == 1) and all(len(v[1]) == MAX_CHILDREN_PER_PARENT for v in items)
        out[key] = {
            "variants": [{"parent_path": p, "children": c} for p, c in items],
            "has_over5": has_over5,
            "all_exact5_same": all_exact5_same,
        }
    return out


def majority_vote_5set(variants: List[List[str]]) -> List[str]:
    """
    Suggest a canonical 5-set per (level, parent_label) by majority vote across parents sharing that label.
    
    Args:
        variants: List of child lists from different parent paths sharing the same label
        
    Returns:
        List of up to 5 children, ordered by frequency (descending) then alphabetically
    """
    if not variants:
        return []
    
    # Flatten all children and count frequencies
    flat = [c for vs in variants for c in vs]
    freq = Counter(flat)
    
    # Rank by frequency (descending) then alphabetically
    ranked = [c for c, _ in freq.most_common()]
    
    # Apply normalization (dedupe, cap to 5)
    return normalize_child_set(ranked)


def _row_nodes_with_root(row: pd.Series) -> List[str]:
    """
    Return a 6-length list: [root, n1, n2, n3, n4, n5]
    with normalization and safe blanks for ragged rows.
    """
    from utils.constants import ROOT_COL
    
    vals = [normalize_text(row.get(ROOT_COL, ""))]
    for c in LEVEL_COLS:
        vals.append(normalize_text(row.get(c, "")))
    # ensure length is 6
    if len(vals) < (1 + MAX_LEVELS):
        vals += [""] * ((1 + MAX_LEVELS) - len(vals))
    return vals[: (1 + MAX_LEVELS)]  # [root, n1..n5]


def build_parent_child_index_with_root(df: pd.DataFrame) -> Dict[Tuple[int, str], Counter]:
    """
    For each level L=1..MAX_LEVELS, build (L, parent_path) -> Counter(child -> rows_count).
    - L=1 : parent is ROOT (<ROOT>), child is Node 1.
    - L>=2: parent path is 'root>Node1>...>Node(L-1)', child is Node L.
    Row multiplication at deeper levels is tolerated: we aggregate via Counter and
    derive unique children from the counter keys (not total row counts).
    """
    from utils.constants import ROOT_COL
    
    idx: Dict[Tuple[int, str], Counter] = {}
    if df is None or df.empty:
        return idx

    # ROOT -> Node1
    ctr_root = Counter()
    for _, row in df.iterrows():
        root, n1, *_rest = _row_nodes_with_root(row)
        if normalize_text(n1):
            ctr_root[normalize_text(n1)] += 1
    if ctr_root:
        idx[(1, ROOT_PARENT_LABEL)] = ctr_root

    # L=2..5
    for L in range(2, MAX_LEVELS + 1):
        ctr_level: Dict[str, Counter] = {}
        for _, row in df.iterrows():
            nodes = _row_nodes_with_root(row)  # [root, n1...n5]
            parent_chain = [normalize_text(x) for x in nodes[:L]]  # includes root ... Node(L-1)
            child = normalize_text(nodes[L])  # Node L
            if not child:
                continue
            # parent_path is root>n1>...>n(L-1) with blanks removed
            parent_path = ">".join([x for x in parent_chain if x])
            if not parent_path:
                continue
            ctr_level.setdefault(parent_path, Counter())
            ctr_level[parent_path][child] += 1
        for pth, ctr in ctr_level.items():
            idx[(L, pth)] = ctr

    return idx


def detect_full_path_duplicates(df: pd.DataFrame) -> List[Tuple[str, int]]:
    """
    Return [(full_path_string, count), ...] for any full path (root + 5 nodes) that occurs >1 times.
    """
    from utils.constants import ROOT_COL
    
    if df is None or df.empty:
        return []
    cols = [ROOT_COL] + LEVEL_COLS
    cols = [c for c in cols if c in df.columns]
    if len(cols) < 2:
        return []
    path_series = df[cols].astype(str).map(normalize_text).agg(lambda s: ">".join([x for x in s.tolist() if x]), axis=1)
    counts = path_series.value_counts()
    dups = [(p, int(n)) for p, n in counts.items() if n > 1]
    return dups


def group_across_tree_by_parent_label(summary: Dict[Tuple[int, str], Dict]) -> Dict[str, List[Tuple[int, str, List[str]]]]:
    """
    Whole-tree: bucket all parents by their *parent_label* regardless of level.
    Key = parent_label (the last segment of parent_path; ROOT for level 1)
    Value = list of tuples: (child_level, parent_path, children_list)
    """
    buckets: Dict[str, List[Tuple[int, str, List[str]]]] = defaultdict(list)
    for (L, parent_path), info in summary.items():
        if L == 1:
            label = ROOT_PARENT_LABEL
        else:
            label = parent_path.split(">")[-1] if parent_path else ""
        if not label:
            continue
        buckets[label].append((L, parent_path, info["children"]))
    return buckets


def find_treewide_label_mismatches(summary: Dict[Tuple[int, str], Dict]) -> Dict[str, Dict[str, Any]]:
    """
    For each parent_label across the *entire* tree, report if:
      - any parent_path under this label has >5 children, or
      - not all sets are the same exact 5.
    Returns: label -> {
        variants: [ { level, parent_path, children, unique_count } ],
        has_over5: bool,
        all_exact5_same: bool
    }
    """
    buckets = group_across_tree_by_parent_label(summary)
    out: Dict[str, Dict[str, Any]] = {}
    for label, items in buckets.items():
        sets = [tuple(sorted(children)) for (_L, _p, children) in items]
        uniq = set(sets)
        has_over5 = any(len(children) > MAX_CHILDREN_PER_PARENT for (_L, _p, children) in items)
        all_exact5_same = (len(uniq) == 1) and all(len(children) == MAX_CHILDREN_PER_PARENT for (_L, _p, children) in items)
        out[label] = {
            "variants": [
                {"level": L, "parent_path": p, "children": children, "unique_count": len(children)}
                for (L, p, children) in items
            ],
            "has_over5": has_over5,
            "all_exact5_same": all_exact5_same,
        }
    return out


def analyze_decision_tree_with_root(df: pd.DataFrame) -> Dict[str, Any]:
    """
    UI-ready summary of the sheet:
      - parent/child index (tolerant of multiplication)
      - children-set summary per parent
      - mismatches by parent label (same label, different 5-sets or >5)
      - root set
      - full-path duplicates
      - top-level counts
    """
    idx = build_parent_child_index_with_root(df)
    summary = summarize_children_sets(idx)
    mismatches = find_label_set_mismatches(summary)
    root_children = summary.get((1, ROOT_PARENT_LABEL), {}).get("children", [])
    over5 = sum(1 for v in summary.values() if v["over5"])
    not_exact5 = sum(1 for v in summary.values() if not v["exact5"])
    dups = detect_full_path_duplicates(df)
    return {
        "index": idx,
        "summary": summary,
        "mismatches": mismatches,
        "treewide_mismatches": find_treewide_label_mismatches(summary),
        "root_children": root_children,
        "duplicates_full_path": dups,
        "counts": {
            "parents_over5": over5,
            "parents_not_exact5": not_exact5,
            "total_parents": len(summary),
        }
    }
