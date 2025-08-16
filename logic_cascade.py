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


def build_label_children_index(store: Dict[str, List[str]]) -> Dict[Tuple[int, str], List[str]]:
    """
    Build (level, parent_label) -> children list.

    Interpretation:
      - For key "L{L}|{parent_tuple}", the children in 'store[key]' are the options for Node L.
      - The "parent label" for level L is:
          * "<ROOT>" when L==1 (friendly top-level)
          * the last element of parent_tuple when L>1
      - We store that as index[(L, parent_label)] = children.

    This is used to auto-attach: when we create a child at Node L with label 'X',
    we will look up (L+1, 'X') to fetch the known Node (L+1) options for that child label.
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


# ----------------- anchor reuse primitives -----------------

def _rows_match_parent(df: pd.DataFrame, vm: str, parent: Tuple[str,...], level: int) -> pd.DataFrame:
    """
    Filter df to rows matching VM and the given parent tuple for *this* level (i.e., Node 1..Node L-1 fixed).
    """
    if df is None or df.empty:
        return df
    mask = (df["Vital Measurement"].map(normalize_text) == normalize_text(vm))
    for i, val in enumerate(parent, 1):
        mask = mask & (df[f"Node {i}"].map(normalize_text) == normalize_text(val))
    return df[mask].copy()


def _present_children_at_level(df: pd.DataFrame, vm: str, parent: Tuple[str,...], level: int) -> Set[str]:
    """
    Existing (non-empty) children values present in df at Node {level} under (vm,parent).
    """
    if level < 1 or level > MAX_LEVELS:
        return set()
    sub = _rows_match_parent(df, vm, parent, level)
    col = f"Node {level}"
    vals = sub[col].map(normalize_text).replace("", np.nan).dropna().unique().tolist()
    return set(vals)


def _find_anchor_index(df: pd.DataFrame, vm: str, parent: Tuple[str,...], level: int) -> Optional[int]:
    """
    Find an anchor row under (vm,parent) with Node{level} blank to fill in-place.
    """
    target_col = f"Node {level}"
    sub_idx = _rows_match_parent(df, vm, parent, level).index.tolist()
    for ix in sub_idx:
        if normalize_text(df.at[ix, target_col]) == "":
            return ix
    return None


def _emit_row_from_prefix(vm_val: str, pref: Tuple[str,...]) -> Dict[str,str]:
    """
    Construct a canonical row dict with Vital Measurement and Node 1..5 filled along prefix;
    remaining Nodes blank; Diagnostic/Actions blank.
    """
    row = {"Vital Measurement": vm_val}
    for i, val in enumerate(pref, 1):
        row[f"Node {i}"] = val
    for i in range(1, MAX_LEVELS+1):
        row.setdefault(f"Node {i}", "")
    row["Diagnostic Triage"] = ""
    row["Actions"] = ""
    return row


# ----------------- core expansion -----------------

def _children_from_store(store: Dict[str, List[str]], level: int, parent: Tuple[str,...]) -> List[str]:
    """
    Fetch defined children for Node {level} under this parent tuple from store.
    """
    opts_raw = store.get(level_key_tuple(level, parent), [])
    return [normalize_text(o) for o in opts_raw if normalize_text(o)!=""]


def _expand_children_with_label_map(children: List[str], next_level: int, label_idx: Dict[Tuple[int,str], List[str]]) -> Dict[str, List[str]]:
    """
    For each child label 'c' at Node {next_level}, if (next_level+1, c) exists in label_idx,
    include those as the next-level children for the new parent '... -> c'.
    Returns a map: child_label -> known_children_for_next_level (may be []).
    """
    result: Dict[str, List[str]] = {}
    for c in children:
        result[c] = label_idx.get((next_level+1, c), [])
    return result


def expand_parent_nextnode_anchor_reuse_for_vm(
    df: pd.DataFrame,
    store: Dict[str, List[str]],
    label_idx: Dict[Tuple[int,str], List[str]],
    vm: str,
    parent: Tuple[str,...]
) -> Tuple[pd.DataFrame, Dict[str,int], List[Tuple[str,...]]]:
    """
    Ensure Node L children exist under 'parent' for a single VM, using anchor-reuse,
    and leverage label_idx to immediately seed known next-level children for each new child created.

    Returns:
      df_updated,
      stats = {'new_rows','inplace_filled'},
      child_parents_confirmed = list of parent tuples (parent + child)
    """
    stats = {"new_rows": 0, "inplace_filled": 0}
    L = len(parent) + 1
    if L > MAX_LEVELS:
        return df, stats, []

    # children defined for this parent (Node L)
    children = _children_from_store(store, L, parent)
    if not children:
        return df, stats, []

    # current children present in df
    present = _present_children_at_level(df, vm, parent, L)
    missing = [c for c in children if c not in present]
    child_parents_confirmed: List[Tuple[str,...]] = []

    # Anchor fill (one missing)
    if missing:
        anchor_ix = _find_anchor_index(df, vm, parent, L)
        if anchor_ix is not None:
            df.at[anchor_ix, f"Node {L}"] = missing[0]
            stats["inplace_filled"] += 1
            child_parents_confirmed.append(parent + (missing[0],))
            missing = missing[1:]

    # Rows for remaining missing options
    new_rows = []
    for m in missing:
        row = _emit_row_from_prefix(vm, parent + (m,))
        new_rows.append(row)
        child_parents_confirmed.append(parent + (m,))

    if new_rows:
        df = pd.concat([df, pd.DataFrame(new_rows, columns=CANON_HEADERS)], ignore_index=True)
        stats["new_rows"] += len(new_rows)

    # Include already-present children as confirmed
    for c in children:
        if c in present:
            child_parents_confirmed.append(parent + (c,))

    # Dedup
    seen = set()
    uniq = []
    for tup in child_parents_confirmed:
        if tup not in seen:
            seen.add(tup); uniq.append(tup)

    # Auto-attach: if each child label exists as a parent label at next_level, we don't create rows here yet;
    # we return confirmed child parents so the caller can recurse with those options defined via store/label_idx.
    return df, stats, uniq


def cascade_anchor_reuse_full(
    df: pd.DataFrame,
    store: Dict[str, List[str]],
    vm_scope: List[str],
    start_parents: List[Tuple[str,...]],
) -> Tuple[pd.DataFrame, Dict[str,int]]:
    """
    Deep cascade to Node 5 using anchor-reuse at EACH level and global label map.
    - Completes partial parents (adds rows for any defined missing options).
    - For each new/confirmed child 'c' at Node L, if (L+1, c) exists in label map,
      recursion will seed Node (L+1) options under (..., c).
    """
    total = {"new_rows":0, "inplace_filled":0}
    label_idx = build_label_children_index(store)

    # BFS/queue over parents
    stack: List[Tuple[str,...]] = list(start_parents)
    visited: Set[Tuple[str,...]] = set()

    while stack:
        parent = stack.pop(0)
        if parent in visited:
            continue
        visited.add(parent)

        L = len(parent) + 1
        if L > MAX_LEVELS:
            continue

        # children options defined?
        children_defined = _children_from_store(store, L, parent)
        if not children_defined:
            # nothing defined for next node; stop at this parent
            continue

        next_child_parents_all_vms: Set[Tuple[str,...]] = set()

        for vm in vm_scope:
            df, stats, child_parents = expand_parent_nextnode_anchor_reuse_for_vm(df, store, label_idx, vm, parent)
            total["new_rows"] += stats["new_rows"]
            total["inplace_filled"] += stats["inplace_filled"]
            for cp in child_parents:
                next_child_parents_all_vms.add(cp)

        # For each child parent cp, if it has known children by label map at next level,
        # or explicit children in store, enqueue for deeper expansion.
        for cp in sorted(next_child_parents_all_vms):
            next_level = len(cp) + 1
            if next_level <= MAX_LEVELS:
                # If explicit store says there are children OR label_idx provides children for (next_level, last_label)
                last_label = cp[-1] if cp else "<ROOT>"
                has_label_defined = len(label_idx.get((next_level, last_label), [])) > 0
                has_store_defined = len(_children_from_store(store, next_level, cp)) > 0
                if has_label_defined or has_store_defined:
                    stack.append(cp)

    return df, total


# ----------------- top-level builder (v6.3.0) -----------------

def build_raw_plus_v630(
    df: pd.DataFrame,
    overrides: Dict[str, List[str]],
    include_scope: str,
    edited_keys_for_sheet: Set[str],
) -> Tuple[pd.DataFrame, Dict[str,int]]:
    """
    Always deep-cascade with anchor-reuse for selected scope, and auto-attachment via label-level map.
    - include_scope: 'session' or 'all'
    """
    if df is None:
        df = pd.DataFrame(columns=CANON_HEADERS)
    assert validate_headers(df), "Headers must match canonical schema."

    store = infer_branch_options_with_overrides(df, overrides or {})
    vms = sorted(set(df["Vital Measurement"].map(normalize_text).replace("", np.nan).dropna().unique().tolist()))
    df_aug = df.copy()
    stats_total = {"generated":0, "new_added":0, "duplicates_skipped":0, "final_total":len(df_aug), "inplace_filled":0}

    # Parents to process
    parent_keys: List[Tuple[int,Tuple[str,...]]] = []
    if include_scope == "session" and edited_keys_for_sheet:
        for keyname in edited_keys_for_sheet:
            if "|" not in keyname:
                continue
            lvl_s, path = keyname.split("|", 1)
            try:
                L = int(lvl_s[1:])
            except Exception:
                continue
            parent = tuple([] if path == "<ROOT>" else path.split(">"))
            parent_keys.append((L, parent))
    else:
        for key in list(store.keys()):
            if "|" not in key:
                continue
            lvl_s, path = key.split("|", 1)
            try:
                L = int(lvl_s[1:])
            except Exception:
                continue
            parent = tuple([] if path == "<ROOT>" else path.split(">"))
            parent_keys.append((L, parent))

    start_parents = sorted({p for (_, p) in parent_keys})
    df_before = len(df_aug)
    df_aug, stx = cascade_anchor_reuse_full(df_aug, store, vms, start_parents)
    stats_total["inplace_filled"] += stx["inplace_filled"]
    stats_total["generated"] += (len(df_aug) - df_before) + stx["inplace_filled"]

    # Compute new_added / duplicates_skipped against original df
    def make_key(rowlike) -> Tuple[str,...]:
        return tuple(normalize_text(rowlike.get(c, "")) for c in ["Vital Measurement"] + LEVEL_COLS)

    original_keys = set()
    for _, r in df.iterrows():
        original_keys.add(make_key(r))

    now_keys = set()
    for _, r in df_aug.iterrows():
        now_keys.add(make_key(r))

    stats_total["new_added"] = len(now_keys - original_keys)
    stats_total["duplicates_skipped"] = max(0, stats_total["generated"] - stats_total["new_added"])
    stats_total["final_total"] = len(df_aug)
    return df_aug, stats_total
