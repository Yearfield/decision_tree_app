from typing import Dict, List, Tuple, Optional, Set
import pandas as pd
import numpy as np

# ----------------- constants -----------------

CANON_HEADERS = [
    "Vital Measurement", "Node 1", "Node 2", "Node 3", "Node 4", "Node 5",
    "Diagnostic Triage", "Actions"
]
LEVEL_COLS = ["Node 1","Node 2","Node 3","Node 4","Node 5"]
MAX_LEVELS = 5


# ----------------- basic helpers -----------------

def normalize_text(x: str) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()


def validate_headers(df: pd.DataFrame) -> bool:
    return list(df.columns[:len(CANON_HEADERS)]) == CANON_HEADERS


def level_key_tuple(level: int, parent: Tuple[str, ...]) -> str:
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


def enforce_k_five(opts: List[str]) -> List[str]:
    """Trim/pad to exactly 5 (blank padded)."""
    clean = [normalize_text(o) for o in opts if normalize_text(o) != ""]
    if len(clean) > 5:
        clean = clean[:5]
    elif len(clean) < 5:
        clean = clean + [""]*(5-len(clean))
    return clean


def friendly_parent_label(level: int, parent_tuple: Tuple[str, ...]) -> str:
    """Human-friendly label for the parent path feeding Node {level} children."""
    if level == 1 and not parent_tuple:
        return "Top-level (Node 1) options"
    return " > ".join(parent_tuple) if parent_tuple else "Top-level (Node 1) options"


# ----------------- store (parent->children) -----------------

def infer_branch_options(df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Build a map: "L{level}|{parent_path or <ROOT>}" -> [children].
    Deduplicates while preserving first-seen order.
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


# ----------------- conflict computation -----------------

def compute_conflicts(store: Dict[str, List[str]], friendly_labels: bool = True
    ) -> Dict[Tuple[int, str], Dict[Tuple[str, ...], List[Tuple[str, ...]]]]:
    """
    Build a structure:
      key: (level, last_label) where last_label is the parent's last element
           or a friendly root string (if friendly_labels=True) else "<ROOT>"
      value: dict mapping child_set(tuple-of-children) -> list of parent tuples that use that set

    A "variant" is a distinct tuple(child labels) used for the same (level, last_label).
    """
    conflicts: Dict[Tuple[int, str], Dict[Tuple[str, ...], List[Tuple[str, ...]]]] = {}
    for key, children in store.items():
        if "|" not in key:
            continue
        lvl_s, path = key.split("|", 1)
        try:
            level = int(lvl_s[1:])
        except Exception:
            continue
        if not (1 <= level <= MAX_LEVELS):
            continue
        parent_tuple: Tuple[str, ...] = tuple([] if path == "<ROOT>" else path.split(">"))
        if parent_tuple:
            last_label = parent_tuple[-1]
        else:
            last_label = friendly_parent_label(level, parent_tuple) if friendly_labels else "<ROOT>"

        child_set = tuple([normalize_text(c) for c in children if normalize_text(c) != ""])
        conflicts.setdefault((level, last_label), {})
        conflicts[(level, last_label)].setdefault(child_set, [])
        conflicts[(level, last_label)][child_set].append(parent_tuple)
    return conflicts


def parents_for_label_at_level(store: Dict[str, List[str]], level: int, target_label: str, friendly_labels: bool = True
    ) -> List[Tuple[str, ...]]:
    """
    Return all parent tuples at 'level' whose last-label equals target_label.
    For root, target_label should be "Top-level (Node 1) options" if friendly_labels=True, else "<ROOT>".
    """
    parents: List[Tuple[str, ...]] = []
    for key in store.keys():
        if "|" not in key:
            continue
        lvl_s, path = key.split("|", 1)
        try:
            lvl = int(lvl_s[1:])
        except Exception:
            continue
        if lvl != level:
            continue
        parent_tuple = tuple([] if path == "<ROOT>" else path.split(">"))
        last_label = parent_tuple[-1] if parent_tuple else (friendly_parent_label(lvl, parent_tuple) if friendly_labels else "<ROOT>")
        if last_label == target_label:
            parents.append(parent_tuple)
    return parents


def conflict_summary(conflicts: Dict[Tuple[int, str], Dict[Tuple[str, ...], List[Tuple[str, ...]]]], level: Optional[int] = None
    ) -> List[Dict[str, object]]:
    """
    Flatten conflict structure into rows for export/reporting.

    Returns list of dict rows with:
      - Node
      - Parent Label
      - Variants
      - Affected Parents
      - Union Children (list)
    """
    rows: List[Dict[str, object]] = []
    for (lvl, lbl), variants in conflicts.items():
        if level is not None and lvl != level:
            continue
        n_var = len(variants)
        tot_parents = sum(len(v) for v in variants.values())
        union_children = sorted({c for childset in variants.keys() for c in childset})
        rows.append({
            "Node": lvl,
            "Parent Label": lbl,
            "Variants": n_var,
            "Affected Parents": tot_parents,
            "Union Children": union_children
        })
    return rows


# ----------------- resolution operations -----------------

def resolve_keep_set_for_all(
    overrides_sheet: Dict[str, List[str]],
    store: Dict[str, List[str]],
    level: int,
    parent_label: str,
    childset_to_keep: List[str],
    friendly_labels: bool = True
) -> Tuple[Dict[str, List[str]], int]:
    """
    Apply resolution: assign the given childset to ALL parents whose last-label == parent_label at Node 'level'.
    Returns (updated_overrides_sheet, n_affected_parents).

    - overrides_sheet: the overrides dict for a single sheet (key -> list-of-children).
    - store: merged df+overrides store, used to find affected parents.
    """
    chosen = enforce_k_five(childset_to_keep)
    affected_parents = parents_for_label_at_level(store, level, parent_label, friendly_labels=friendly_labels)

    updated = overrides_sheet.copy()
    for p in affected_parents:
        k = level_key_tuple(level, p)
        updated[k] = chosen
    return updated, len(affected_parents)


def resolve_custom_set_for_all(
    overrides_sheet: Dict[str, List[str]],
    store: Dict[str, List[str]],
    level: int,
    parent_label: str,
    custom_children: List[str],
    friendly_labels: bool = True
) -> Tuple[Dict[str, List[str]], int]:
    """
    Same as keep_set, but with an arbitrary selection of up to 5 children (trim/pad applied).
    Returns (updated_overrides_sheet, n_affected_parents).
    """
    return resolve_keep_set_for_all(overrides_sheet, store, level, parent_label, custom_children, friendly_labels=friendly_labels)


# ----------------- convenience API -----------------

def build_store(df: pd.DataFrame, overrides_sheet: Optional[Dict[str, List[str]]] = None) -> Dict[str, List[str]]:
    """
    Helper: validate headers and return merged store from df + overrides_sheet.
    """
    assert validate_headers(df), "Headers must match canonical schema."
    return infer_branch_options_with_overrides(df, overrides_sheet or {})


def detect_conflicts_for_level(
    df: pd.DataFrame,
    overrides_sheet: Optional[Dict[str, List[str]]] = None,
    level: Optional[int] = None,
    friendly_labels: bool = True
) -> Tuple[Dict[Tuple[int, str], Dict[Tuple[str, ...], List[Tuple[str, ...]]]], List[Dict[str, object]]]:
    """
    End-to-end:
      - Build store
      - Compute conflicts (friendly labels by default)
      - Optionally filter by level
      - Return both the raw conflicts structure and a flattened summary rows list
    """
    store = build_store(df, overrides_sheet)
    conf = compute_conflicts(store, friendly_labels=friendly_labels)
    if level is not None:
        conf = {k:v for k,v in conf.items() if k[0] == level}
    summary = conflict_summary(conf, level=level)
    return conf, summary
