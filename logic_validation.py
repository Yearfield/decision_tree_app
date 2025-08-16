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
    """Merge inferred store with overrides (overrides win)."""
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


# ----------------- validations -----------------

def detect_orphans(
    df: pd.DataFrame,
    overrides: Optional[Dict[str, List[str]]] = None,
    strict: bool = False
) -> List[Dict[str, object]]:
    """
    Detect orphans: children that never appear as parents.

    Modes:
      - strict=False (default, 'loose'): a child label 'c' at Node L is an orphan if
        there is NO entry anywhere for (L+1, parent_label == 'c') in the label index.
      - strict=True: a child is orphan if there is NO exact parent-tuple (parent + (c,))
        defined at Node (L+1) in the store.

    Returns list of dicts:
      {
        "level": L,
        "parent": parent_tuple,
        "child": child_label,
        "mode": "loose" | "strict"
      }
    """
    assert validate_headers(df), "Headers must match canonical schema."
    store = infer_branch_options_with_overrides(df, overrides or {})
    label_idx = build_label_children_index(store)

    results: List[Dict[str, object]] = []

    for key, children in store.items():
        if "|" not in key:
            continue
        lvl_s, path = key.split("|", 1)
        try:
            L = int(lvl_s[1:])
        except Exception:
            continue
        if not (1 <= L <= MAX_LEVELS-1):  # only levels that can have children below
            continue

        parent_tuple = tuple([] if path == "<ROOT>" else path.split(">"))
        next_level = L + 1

        for c in [normalize_text(x) for x in children if normalize_text(x) != ""]:
            if strict:
                # exact parent tuple must exist at next level
                k_exact = level_key_tuple(next_level, parent_tuple + (c,))
                if k_exact not in store:
                    results.append({"level": L, "parent": parent_tuple, "child": c, "mode": "strict"})
            else:
                # loose by label: label 'c' must appear as a parent label at next level
                if len(label_idx.get((next_level, c), [])) == 0:
                    results.append({"level": L, "parent": parent_tuple, "child": c, "mode": "loose"})

    return results


def detect_loops(df: pd.DataFrame) -> List[Dict[str, object]]:
    """
    Detect circular branches within a single row:
    i.e., a label that repeats later in the same path (Node 1..5).
    Returns a list of dicts:
      {
        "row_index": idx,
        "vm": <Vital Measurement>,
        "path": [Node 1..5 values],
        "repeats": [(label, first_pos, later_pos), ...]
      }
    """
    assert validate_headers(df), "Headers must match canonical schema."
    results: List[Dict[str, object]] = []

    nodes = df[LEVEL_COLS].applymap(normalize_text)
    for idx, row in nodes.iterrows():
        path = [row[c] for c in LEVEL_COLS if normalize_text(row[c]) != ""]
        if not path:
            continue
        seen_pos: Dict[str, int] = {}
        repeats: List[Tuple[str, int, int]] = []
        for i, label in enumerate(path):
            if label in seen_pos:
                repeats.append((label, seen_pos[label]+1, i+1))  # 1-based positions
            else:
                seen_pos[label] = i
        if repeats:
            results.append({
                "row_index": idx,
                "vm": normalize_text(df.at[idx, "Vital Measurement"]),
                "path": path,
                "repeats": repeats
            })
    return results


def detect_missing_redflag_coverage(
    df: pd.DataFrame,
    overrides: Optional[Dict[str, List[str]]] = None,
    redflag_map: Optional[Dict[str, str]] = None
) -> List[Dict[str, object]]:
    """
    For every parent (Node L) with non-empty children, verify at least one child is flagged 'Red Flag'.
    redflag_map format: {label: "Red Flag"|"Normal"} (case-insensitive on lookup).
    Returns list of dicts:
      {
        "level": L,
        "parent": parent_tuple,
        "children": [list_of_children],
        "redflag_present": False
      }
    """
    assert validate_headers(df), "Headers must match canonical schema."
    store = infer_branch_options_with_overrides(df, overrides or {})
    results: List[Dict[str, object]] = []

    if not redflag_map:
        return results  # nothing to check against

    # normalize RF map keys once
    rf = {normalize_text(k): ("Red Flag" if str(v).lower().strip() == "red flag" else "Normal")
          for k, v in redflag_map.items()}

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
        non_empty = [normalize_text(x) for x in children if normalize_text(x) != ""]
        if not non_empty:
            continue
        has_rf = any(rf.get(x, "Normal") == "Red Flag" for x in non_empty)
        if not has_rf:
            parent_tuple = tuple([] if path == "<ROOT>" else path.split(">"))
            results.append({
                "level": L,
                "parent": parent_tuple,
                "children": non_empty,
                "redflag_present": False
            })

    return results


# ----------------- aggregator -----------------

def compute_validation_report(
    df: pd.DataFrame,
    overrides: Optional[Dict[str, List[str]]] = None,
    redflag_map: Optional[Dict[str, str]] = None
) -> Dict[str, object]:
    """
    Run all validations and return a structured report:
      {
        "orphans_loose": [...],
        "orphans_strict": [...],
        "loops": [...],
        "missing_redflag": [...],
        "counts": {
            "orphans_loose": N,
            "orphans_strict": N,
            "loops": N,
            "missing_redflag": N
        }
      }
    """
    assert validate_headers(df), "Headers must match canonical schema."

    orphans_loose = detect_orphans(df, overrides=overrides, strict=False)
    orphans_strict = detect_orphans(df, overrides=overrides, strict=True)
    loops = detect_loops(df)
    missing_rf = detect_missing_redflag_coverage(df, overrides=overrides, redflag_map=redflag_map)

    report = {
        "orphans_loose": orphans_loose,
        "orphans_strict": orphans_strict,
        "loops": loops,
        "missing_redflag": missing_rf,
        "counts": {
            "orphans_loose": len(orphans_loose),
            "orphans_strict": len(orphans_strict),
            "loops": len(loops),
            "missing_redflag": len(missing_rf),
        }
    }
    return report
