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
    """
    try:
        if x is None:
            return ""
        if isinstance(x, float) and np.isnan(x):
            return ""
    except Exception:
        pass
    return str(x).strip()


def validate_headers(df: pd.DataFrame) -> bool:
    """
    Verify that the first len(CANON_HEADERS) columns match the canonical schema.
    """
    return list(df.columns[:len(CANON_HEADERS)]) == CANON_HEADERS


def ensure_canon_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure the DataFrame has all canonical columns (creates missing as "") and in order.
    Does not drop any extra columns the caller may have added.
    """
    df2 = df.copy()
    for c in CANON_HEADERS:
        if c not in df2.columns:
            df2[c] = ""
    return df2[CANON_HEADERS]


def drop_fully_blank_paths(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop rows where Vital Measurement + Node1..Node5 are all blank.
    """
    node_block = ["Vital Measurement"] + LEVEL_COLS
    mask_blank = df[node_block].apply(lambda r: all(normalize_text(v) == "" for v in r), axis=1)
    return df[~mask_blank].copy()


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
    """
    if upto_level <= 1:
        return tuple()
    parent: List[str] = []
    for c in LEVEL_COLS[:upto_level-1]:
        v = normalize_text(row.get(c, ""))
        if v == "":
            return None
        parent.append(v)
    return tuple(parent)


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
