# streamlit_app_upload.py — v6.3.1 (monolith)
# Single-file build: all logic + UI integrated

import io
import os
import re
import json
import math
import random
from typing import Dict, List, Tuple, Optional, Set

from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

# Optional visualizer
try:
    from pyvis.network import Network
    HAS_PYVIS = True
except Exception:
    HAS_PYVIS = False

# ============ VERSION / CONFIG ============
APP_VERSION = "v6.3.1"

CANON_HEADERS = [
    "Vital Measurement", "Node 1", "Node 2", "Node 3", "Node 4", "Node 5",
    "Diagnostic Triage", "Actions"
]
LEVEL_COLS = ["Node 1","Node 2","Node 3","Node 4","Node 5"]
MAX_LEVELS = 5

st.set_page_config(page_title=f"Decision Tree Builder {APP_VERSION}", page_icon="🌳", layout="wide")


# ============ Session helpers ============
def ss_get(key, default):
    if key not in st.session_state:
        st.session_state[key] = default
    return st.session_state[key]

def ss_set(key, value):
    st.session_state[key] = value

def mark_session_edit(sheet: str, keyname: str):
    ek = ss_get("session_edited_keys", {})
    cur = set(ek.get(sheet, []))
    cur.add(keyname)
    ek[sheet] = list(cur)
    ss_set("session_edited_keys", ek)


# ============ Core helpers ============
def normalize_text(x: str) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()

def validate_headers(df: pd.DataFrame) -> bool:
    return list(df.columns[:len(CANON_HEADERS)]) == CANON_HEADERS

def parent_key_from_row_strict(row: pd.Series, upto_level: int) -> Optional[Tuple[str, ...]]:
    if upto_level <= 1:
        return tuple()
    parent = []
    for c in LEVEL_COLS[:upto_level-1]:
        v = normalize_text(row.get(c, ""))
        if v == "":
            return None
        parent.append(v)
    return tuple(parent)

def level_key_tuple(level: int, parent: Tuple[str, ...]) -> str:
    return f"L{level}|" + (">".join(parent) if parent else "<ROOT>")

def friendly_parent_label(level: int, parent_tuple: Tuple[str, ...]) -> str:
    if level == 1 and not parent_tuple:
        return "Top-level (Node 1) options"
    return " > ".join(parent_tuple) if parent_tuple else "Top-level (Node 1) options"

def enforce_k_five(opts: List[str]) -> List[str]:
    clean = [normalize_text(o) for o in opts if normalize_text(o) != ""]
    if len(clean) > 5:
        clean = clean[:5]
    elif len(clean) < 5:
        clean = clean + [""]*(5-len(clean))
    return clean


# ============ Google Sheets helpers ============
def _gs_client():
    import gspread
    from google.oauth2.service_account import Credentials
    sa_info = st.secrets["gcp_service_account"]
    scopes = ["https://www.googleapis.com/auth/spreadsheets","https://www.googleapis.com/auth/drive"]
    creds = Credentials.from_service_account_info(sa_info, scopes=scopes)
    return gspread.authorize(creds)

def push_to_google_sheets(spreadsheet_id: str, sheet_name: str, df: pd.DataFrame) -> bool:
    try:
        import gspread
        from google.oauth2.service_account import Credentials
        if "gcp_service_account" not in st.secrets:
            st.error("Google Sheets not configured. Add your service account JSON under [gcp_service_account].")
            return False

        client = _gs_client()
        sh = client.open_by_key(spreadsheet_id)

        df = df.fillna("")
        headers = list(df.columns)
        values = [headers] + df.astype(str).values.tolist()
        n_rows = len(values)
        n_cols = max(1, len(headers))

        try:
            ws = sh.worksheet(sheet_name)
            ws.clear()
            ws.resize(rows=max(n_rows, 200), cols=max(n_cols, 8))
        except Exception:
            ws = sh.add_worksheet(title=sheet_name, rows=max(n_rows, 200), cols=max(n_cols, 8))

        ws.update('A1', values, value_input_option="RAW")
        return True

    except Exception as e:
        st.error(f"Push to Google Sheets failed: {e}")
        return False

def backup_sheet_copy(spreadsheet_id: str, source_sheet: str) -> Optional[str]:
    try:
        if "gcp_service_account" not in st.secrets:
            return None
        client = _gs_client()
        sh = client.open_by_key(spreadsheet_id)
        try:
            ws = sh.worksheet(source_sheet)
        except Exception:
            return None
        values = ws.get_all_values()
        ts = datetime.now().strftime("%Y-%m-%d %H%M")
        backup_title_full = f"{source_sheet} (backup {ts})"
        backup_title = backup_title_full[:99]
        rows = max(len(values), 100)
        cols = max(len(values[0]) if values else 8, 8)
        ws_bak = sh.add_worksheet(title=backup_title, rows=rows, cols=cols)
        if values:
            ws_bak.update('A1', values, value_input_option="RAW")
        return backup_title
    except Exception as e:
        st.error(f"Backup failed: {e}")
        return None


# ============ Inference / store building ============
def infer_branch_options(df: pd.DataFrame) -> Dict[str, List[str]]:
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
    base = infer_branch_options(df)
    merged = dict(base)
    for k, v in (overrides or {}).items():
        vals = [normalize_text(x) for x in (v if isinstance(v, list) else [v])]
        merged[k] = vals
    return merged

def build_label_children_index(store: Dict[str, List[str]]) -> Dict[Tuple[int, str], List[str]]:
    """
    Map of (level, parent_label) -> children.
    E.g., ("L2|(<ROOT>)" has label "<ROOT>"), for L>1 parent_label is the last segment of the tuple.
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


# ============ Cascade engine (anchor-reuse + dictionary-driven recursion) ============
def _rows_match_parent(df: pd.DataFrame, vm: str, parent: Tuple[str,...], level: int) -> pd.DataFrame:
    mask = (df["Vital Measurement"].map(normalize_text) == vm)
    for i, val in enumerate(parent, 1):
        mask = mask & (df[f"Node {i}"].map(normalize_text) == val)
    return df[mask].copy()

def _present_children_at_level(df: pd.DataFrame, vm: str, parent: Tuple[str,...], level: int) -> Set[str]:
    sub = _rows_match_parent(df, vm, parent, level)
    if level <= MAX_LEVELS:
        col = f"Node {level}"
        return set(sub[col].map(normalize_text).replace("", np.nan).dropna().unique().tolist())
    return set()

def _find_anchor_index(df: pd.DataFrame, vm: str, parent: Tuple[str,...], level: int) -> Optional[int]:
    target_col = f"Node {level}"
    sub_idx = _rows_match_parent(df, vm, parent, level).index.tolist()
    for ix in sub_idx:
        if normalize_text(df.at[ix, target_col]) == "":
            return ix
    return None

def _emit_row_from_prefix(vm_val: str, pref: Tuple[str,...]) -> Dict[str,str]:
    row = {"Vital Measurement": vm_val}
    for i, val in enumerate(pref, 1):
        row[f"Node {i}"] = val
    for i in range(1, MAX_LEVELS+1):
        row.setdefault(f"Node {i}", "")
    row["Diagnostic Triage"] = ""
    row["Actions"] = ""
    return row

def _children_from_store(store: Dict[str, List[str]], label_idx: Dict[Tuple[int,str], List[str]],
                         level: int, parent: Tuple[str,...]) -> List[str]:
    """
    Enhanced: Prefer exact tuple key; if not present or empty, fall back to label-based map.
    This allows auto-attaching known children by parent label across the dataset.
    """
    key = level_key_tuple(level, parent)
    direct = [normalize_text(o) for o in store.get(key, []) if normalize_text(o)!=""]
    if direct:
        return direct
    # Fallback: label index (dictionary-driven)
    parent_label = "<ROOT>" if (level == 1 and not parent) else (parent[-1] if parent else "<ROOT>")
    return [normalize_text(o) for o in label_idx.get((level, parent_label), []) if normalize_text(o)!=""]

def expand_parent_nextnode_anchor_reuse_for_vm(
    df: pd.DataFrame,
    store: Dict[str, List[str]],
    label_idx: Dict[Tuple[int,str], List[str]],
    vm: str,
    parent: Tuple[str,...]
) -> Tuple[pd.DataFrame, Dict[str,int], List[Tuple[str,...]]]:
    """
    Ensure next-node children exist under 'parent' for a single VM, using anchor-reuse.
    Uses label-index fallback so that if a child label exists as a parent elsewhere,
    its known children propagate immediately.
    """
    stats = {"new_rows": 0, "inplace_filled": 0}
    L = len(parent) + 1
    if L > MAX_LEVELS:
        return df, stats, []

    children = _children_from_store(store, label_idx, L, parent)
    if not children:
        return df, stats, []

    present = _present_children_at_level(df, vm, parent, L)
    missing = [c for c in children if c not in present]
    child_parents_confirmed: List[Tuple[str,...]] = []

    # Anchor fill (one)
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

    # Also include already-present
    for c in children:
        if c in present:
            child_parents_confirmed.append(parent + (c,))

    # Deduplicate confirmed list
    seen = set()
    uniq = []
    for tup in child_parents_confirmed:
        if tup not in seen:
            seen.add(tup); uniq.append(tup)

    return df, stats, uniq

def cascade_anchor_reuse_full(
    df: pd.DataFrame,
    store: Dict[str, List[str]],
    vm_scope: List[str],
    start_parents: List[Tuple[str,...]],
) -> Tuple[pd.DataFrame, Dict[str,int]]:
    """
    Deep cascade to Node 5 using anchor-reuse at EACH level + label-index fallback.
    Completes parents (adds rows for missing options) and recurses while child options exist.
    """
    total = {"new_rows":0, "inplace_filled":0}
    stack: List[Tuple[str,...]] = list(start_parents)
    label_idx = build_label_children_index(store)

    while stack:
        parent = stack.pop(0)
        L = len(parent)+1
        if L > MAX_LEVELS:
            continue

        # Are there children defined (via exact or label idx)?
        children_defined = _children_from_store(store, label_idx, L, parent)
        if not children_defined:
            continue

        next_child_parents_all_vms: Set[Tuple[str,...]] = set()

        for vm in vm_scope:
            df, stats, child_parents = expand_parent_nextnode_anchor_reuse_for_vm(df, store, label_idx, vm, parent)
            total["new_rows"] += stats["new_rows"]
            total["inplace_filled"] += stats["inplace_filled"]
            for cp in child_parents:
                next_child_parents_all_vms.add(cp)

        # Recurse deeper if next-level options exist (again, via direct or label)
        for cp in sorted(next_child_parents_all_vms):
            next_level = len(cp) + 1
            if next_level <= MAX_LEVELS:
                if _children_from_store(store, label_idx, next_level, cp):
                    stack.append(cp)

    return df, total

def build_raw_plus_v631(
    df: pd.DataFrame,
    overrides: Dict[str, List[str]],
    include_scope: str,
    edited_keys_for_sheet: Set[str],
) -> Tuple[pd.DataFrame, Dict[str,int]]:
    """
    Always deep-cascade with anchor-reuse for selected scope (v6.3.1).
    """
    store = infer_branch_options_with_overrides(df, overrides)
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
            try: L = int(lvl_s[1:])
            except: continue
            parent = tuple([] if path == "<ROOT>" else path.split(">"))
            parent_keys.append((L, parent))
    else:
        for key in list(store.keys()):
            if "|" not in key: 
                continue
            lvl_s, path = key.split("|", 1)
            try: L = int(lvl_s[1:])
            except: continue
            parent = tuple([] if path == "<ROOT>" else path.split(">"))
            parent_keys.append((L, parent))

    start_parents = sorted({p for (_, p) in parent_keys})
    df_before = len(df_aug)
    df_aug, stx = cascade_anchor_reuse_full(df_aug, store, vms, start_parents)
    stats_total["inplace_filled"] += stx["inplace_filled"]
    stats_total["generated"] += (len(df_aug) - df_before) + stx["inplace_filled"]

    # Compute new_added / duplicates_skipped vs original df
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


# ============ Progress metrics ============
def compute_parent_depth_score(df: pd.DataFrame) -> Tuple[int, int]:
    store = infer_branch_options(df)
    total = 0; ok = 0
    for level in range(1, MAX_LEVELS+1):
        parents = set()
        for _, row in df.iterrows():
            p = parent_key_from_row_strict(row, level)
            if p is not None:
                parents.add(p)
        for p in parents:
            total += 1
            key = level_key_tuple(level, p)
            if len([x for x in store.get(key, []) if normalize_text(x)!=""]) == 5:
                ok += 1
    return ok, total

def compute_row_path_score(df: pd.DataFrame) -> Tuple[int, int]:
    if df.empty:
        return (0,0)
    nodes = df[LEVEL_COLS].map(normalize_text)
    full = nodes.ne("").all(axis=1)
    return int(full.sum()), int(len(df))

def progress_badge_html(df: pd.DataFrame) -> str:
    ok_p, total_p = compute_parent_depth_score(df)
    ok_r, total_r = compute_row_path_score(df)
    pct_p = 0 if total_p==0 else int(round(100*ok_p/total_p))
    pct_r = 0 if total_r==0 else int(round(100*ok_r/total_r))
    bar_css = """
    <style>
      .badge-wrap { display:inline-block; padding:8px 10px; border-radius:12px; border:1px solid #dbe0e6; background:#f8fafc; }
      .badge-row { font: 12px/1.2 -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Helvetica, Arial, sans-serif; margin:4px 0 6px 0; color:#0f172a;}
      .bar { position:relative; height:10px; background:#eef2f6; border:1px solid #dbe0e6; border-radius:8px; overflow:hidden; }
      .fill { position:absolute; left:0; top:0; bottom:0; width:VAR%; background:linear-gradient(90deg,#60a5fa,#2563eb); }
      .fill2 { position:absolute; left:0; top:0; bottom:0; width:VAR2%; background:linear-gradient(90deg,#34d399,#059669); }
      .badge-top { display:flex; justify-content:space-between; }
      .muted { color:#334155; }
      .strong { font-weight:600; }
    </style>
    """
    html = f"""
    {bar_css}
    <div class='badge-wrap'>
      <div class='badge-row'>
        <div class='badge-top'><span class='muted'>Parents 5/5</span><span class='strong'>{ok_p}/{total_p}</span></div>
        <div class='bar'><div class='fill' style='width:{pct_p}%;'></div></div>
      </div>
      <div class='badge-row'>
        <div class='badge-top'><span class='muted'>Rows full path</span><span class='strong'>{ok_r}/{total_r}</span></div>
        <div class='bar'><div class='fill2' style='width:{pct_r}%;'></div></div>
      </div>
    </div>
    """
    return html


# ============ Validation rules (orphans, loops, red flag coverage) ============
def detect_orphans(
    df: pd.DataFrame,
    overrides: Optional[Dict[str, List[str]]] = None,
    strict: bool = False
) -> List[Dict[str, object]]:
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
        if not (1 <= L <= MAX_LEVELS-1):
            continue

        parent_tuple = tuple([] if path == "<ROOT>" else path.split(">"))
        next_level = L + 1

        for c in [normalize_text(x) for x in children if normalize_text(x) != ""]:
            if strict:
                k_exact = level_key_tuple(next_level, parent_tuple + (c,))
                if k_exact not in store:
                    results.append({"level": L, "parent": parent_tuple, "child": c, "mode": "strict"})
            else:
                if len(label_idx.get((next_level, c), [])) == 0:
                    results.append({"level": L, "parent": parent_tuple, "child": c, "mode": "loose"})
    return results

def detect_loops(df: pd.DataFrame) -> List[Dict[str, object]]:
    assert validate_headers(df), "Headers must match canonical schema."
    results: List[Dict[str, object]] = []
    nodes = df[LEVEL_COLS].map(normalize_text)
    for idx, row in nodes.iterrows():
        path = [row[c] for c in LEVEL_COLS if normalize_text(row[c]) != ""]
        if not path:
            continue
        seen_pos: Dict[str, int] = {}
        repeats: List[Tuple[str, int, int]] = []
        for i, label in enumerate(path):
            if label in seen_pos:
                repeats.append((label, seen_pos[label]+1, i+1))  # 1-based
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
    assert validate_headers(df), "Headers must match canonical schema."
    store = infer_branch_options_with_overrides(df, overrides or {})
    results: List[Dict[str, object]] = []
    if not redflag_map:
        return results
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

def compute_validation_report(
    df: pd.DataFrame,
    overrides: Optional[Dict[str, List[str]]] = None,
    redflag_map: Optional[Dict[str, str]] = None
) -> Dict[str, object]:
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


# ============ UI: Source ============
def render_source():
    st.header("📂 Source")
    st.markdown("Select how you want to load or create your decision tree data.")

    # Upload workbook
    st.subheader("📤 Upload Workbook")
    up = st.file_uploader("Upload an Excel workbook (.xlsx)", type=["xlsx"], key="src_uploader")
    if up is not None:
        try:
            xls = pd.ExcelFile(up)
            sheets = {}
            for name in xls.sheet_names:
                dfx = xls.parse(name)
                dfx.columns = [normalize_text(c) for c in dfx.columns]
                for c in CANON_HEADERS:
                    if c not in dfx.columns:
                        dfx[c] = ""
                dfx = dfx[CANON_HEADERS]
                node_block = ["Vital Measurement"] + LEVEL_COLS
                dfx = dfx[~dfx[node_block].apply(lambda r: all(normalize_text(v) == "" for v in r), axis=1)].copy()
                sheets[name] = dfx
            ss_set("upload_workbook", {k: v.copy() for k, v in sheets.items()})
            ss_set("upload_filename", up.name)
            st.success(f"Loaded {len(sheets)} sheet(s) from **{up.name}**.")
            # Set work context default to first sheet
            if sheets:
                first = next(iter(sheets))
                ss_set("work_context", {"source":"upload", "sheet": first})
        except Exception as e:
            st.error(f"Upload failed: {e}")

    st.markdown("---")

    # Load or refresh from Google Sheets
    st.subheader("🔄 Google Sheets")
    st.caption("Requires service account JSON under [gcp_service_account] in `secrets.toml`.")
    sid = st.text_input("Spreadsheet ID", value=ss_get("gs_spreadsheet_id",""), key="src_gs_sid")
    if sid:
        ss_set("gs_spreadsheet_id", sid)
    sheet_to_load = st.text_input("Sheet name to load (e.g., BP)", value="", key="src_gs_sheetname")

    if st.button("Load / Refresh from Google Sheets", key="src_gs_load_btn"):
        if not sid or not sheet_to_load:
            st.warning("Enter Spreadsheet ID and the sheet name to load.")
        else:
            try:
                client = _gs_client()
                sh = client.open_by_key(sid)
                ws = sh.worksheet(sheet_to_load)
                values = ws.get_all_values()
                if not values:
                    st.error("Selected sheet is empty."); return
                header = [normalize_text(c) for c in values[0]]
                rows = values[1:]
                df_g = pd.DataFrame(rows, columns=header)
                if not validate_headers(df_g):
                    # Fill missing headers
                    for c in CANON_HEADERS:
                        if c not in df_g.columns:
                            df_g[c] = ""
                    df_g = df_g[CANON_HEADERS]
                # Strip empty rows
                node_block = ["Vital Measurement"] + LEVEL_COLS
                df_g = df_g[~df_g[node_block].apply(lambda r: all(normalize_text(v) == "" for v in r), axis=1)].copy()
                wb_g = ss_get("gs_workbook", {}); wb_g[sheet_to_load] = df_g; ss_set("gs_workbook", wb_g)
                st.success(f"Loaded '{sheet_to_load}' from Google Sheets.")
                ss_set("work_context", {"source":"gs", "sheet": sheet_to_load})
            except Exception as e:
                st.error(f"Google Sheets error: {e}")

    st.markdown("---")

    # VM Builder (light)
    st.subheader("🧩 VM Builder (create Vital Measurements and auto-cascade)")
    with st.expander("Open VM Builder"):
        wb_choice = st.radio("Target workbook", ["Upload workbook","Google Sheets workbook"], horizontal=True, key="src_vmb_wb")
        if wb_choice == "Upload workbook":
            wb = ss_get("upload_workbook", {})
            override_root = "branch_overrides_upload"
            source_code = "upload"
        else:
            wb = ss_get("gs_workbook", {})
            override_root = "branch_overrides_gs"
            source_code = "gs"
        if not wb:
            st.info("No workbook loaded. Upload or load from Google Sheets first.")
        else:
            sheet_name = st.selectbox("Sheet", list(wb.keys()), key="src_vmb_sheet")
            df_in = wb.get(sheet_name, pd.DataFrame()).copy()
            overrides_all = ss_get(override_root, {})
            overrides_sheet = overrides_all.get(sheet_name, {}).copy()

            mode = st.radio("Mode", ["Create VM with 5 Node-1 options",
                                     "Create VM with one Node-1 and its 5 Node-2 options"], horizontal=False, key="src_vmb_mode")
            if mode == "Create VM with 5 Node-1 options":
                vm_name = st.text_input("Vital Measurement name", key="src_vmb_vm1")
                cols = st.columns(5)
                vals_n1 = [cols[i].text_input(f"Node-1 option {i+1}", key=f"src_vmb_n1_{i}") for i in range(5)]
                if st.button("Create VM and Auto-cascade", key="src_vmb_create1"):
                    if not vm_name.strip():
                        st.error("Enter a Vital Measurement name.")
                    else:
                        if df_in[df_in["Vital Measurement"].map(normalize_text) == vm_name].empty:
                            anchor = {"Vital Measurement": vm_name}
                            for c in LEVEL_COLS: anchor[c] = ""
                            anchor["Diagnostic Triage"] = ""; anchor["Actions"] = ""
                            df_in = pd.concat([df_in, pd.DataFrame([anchor], columns=CANON_HEADERS)], ignore_index=True)
                        k1 = level_key_tuple(1, tuple())
                        overrides_sheet[k1] = enforce_k_five(vals_n1)
                        overrides_all[sheet_name] = overrides_sheet
                        ss_set(override_root, overrides_all)
                        mark_session_edit(sheet_name, k1)
                        store = infer_branch_options_with_overrides(df_in, overrides_sheet)
                        df_new, tstats = cascade_anchor_reuse_full(df_in, store, [vm_name], [tuple()])
                        df_in = df_new
                        wb[sheet_name] = df_in
                        if source_code == "upload": ss_set("upload_workbook", wb)
                        else: ss_set("gs_workbook", wb)
                        st.success(f"VM '{vm_name}' created. Auto-cascade added {tstats['new_rows']} rows, filled {tstats['inplace_filled']} anchors.")
            else:
                vm_name = st.text_input("Vital Measurement name", key="src_vmb_vm2")
                n1 = st.text_input("Node-1 value", key="src_vmb_n1_val")
                cols = st.columns(5)
                vals_n2 = [cols[i].text_input(f"Node-2 option {i+1}", key=f"src_vmb_n2_{i}") for i in range(5)]
                if st.button("Create VM + Node-1 + Node-2 and Auto-cascade", key="src_vmb_create2"):
                    if not vm_name.strip() or not n1.strip():
                        st.error("Enter a Vital Measurement and Node-1.")
                    else:
                        if df_in[df_in["Vital Measurement"].map(normalize_text) == vm_name].empty:
                            anchor = {"Vital Measurement": vm_name}
                            for c in LEVEL_COLS: anchor[c] = ""
                            anchor["Diagnostic Triage"] = ""; anchor["Actions"] = ""
                            df_in = pd.concat([df_in, pd.DataFrame([anchor], columns=CANON_HEADERS)], ignore_index=True)
                        k1 = level_key_tuple(1, tuple())
                        existing = [x for x in overrides_sheet.get(k1, []) if normalize_text(x)!=""]
                        if n1 not in existing:
                            existing.append(n1)
                        overrides_sheet[k1] = enforce_k_five(existing)
                        k2 = level_key_tuple(2, (n1,))
                        overrides_sheet[k2] = enforce_k_five(vals_n2)
                        overrides_all[sheet_name] = overrides_sheet
                        ss_set(override_root, overrides_all)
                        mark_session_edit(sheet_name, k1); mark_session_edit(sheet_name, k2)
                        store = infer_branch_options_with_overrides(df_in, overrides_sheet)
                        df_new, tstats = cascade_anchor_reuse_full(df_in, store, [vm_name], [(n1,)])
                        df_in = df_new
                        wb[sheet_name] = df_in
                        if source_code == "upload": ss_set("upload_workbook", wb)
                        else: ss_set("gs_workbook", wb)
                        st.success(f"VM '{vm_name}' with Node-1 '{n1}' created. Auto-cascade added {tstats['new_rows']} rows, filled {tstats['inplace_filled']} anchors.")

    st.markdown("---")

    # Wizard placeholder
    st.subheader("🧙 VM Build Wizard (step-by-step to Node 5)")
    st.info("Wizard coming soon: guided steps to define Node 1 → Node 5 with validations and previews.")


# ============ UI: Workspace Selection ============
def render_workspace():
    st.header("🗂 Workspace Selection")

    sources = []
    if ss_get("upload_workbook", {}):
        sources.append("Upload workbook")
    if ss_get("gs_workbook", {}):
        sources.append("Google Sheets workbook")

    if not sources:
        st.info("Load a workbook in the **Source** tab first (upload file or Google Sheets).")
        return

    ctx = ss_get("work_context", {})
    default_src_label = {"upload": "Upload workbook", "gs": "Google Sheets workbook"}.get(ctx.get("source"))
    init_idx = sources.index(default_src_label) if default_src_label in sources else 0
    source_ws = st.radio("Choose data source", sources, horizontal=True, index=init_idx, key="ws_source_sel")

    if source_ws == "Upload workbook":
        wb_ws = ss_get("upload_workbook", {})
        override_root = "branch_overrides_upload"
        current_source_code = "upload"
    else:
        wb_ws = ss_get("gs_workbook", {})
        override_root = "branch_overrides_gs"
        current_source_code = "gs"

    if not wb_ws:
        st.warning("No sheets found in the selected source. Load data from the **Source** tab.")
        return

    default_sheet = ctx.get("sheet")
    sheet_names = list(wb_ws.keys())
    sheet_idx = sheet_names.index(default_sheet) if default_sheet in sheet_names else 0
    sheet_ws = st.selectbox("Sheet", sheet_names, index=sheet_idx, key="ws_sheet_sel")
    df_ws = wb_ws.get(sheet_ws, pd.DataFrame())

    # Remember work context
    ss_set("work_context", {"source": current_source_code, "sheet": sheet_ws})

    if df_ws.empty or not validate_headers(df_ws):
        st.info("Selected sheet is empty or headers mismatch.")
    else:
        st.write(f"Found {len(wb_ws)} sheet(s). Choose one to process:")

        ok_p, total_p = compute_parent_depth_score(df_ws)
        ok_r, total_r = compute_row_path_score(df_ws)
        p1, p2 = st.columns(2)
        with p1:
            st.metric("Parents with 5 children", f"{ok_p}/{total_p}")
            st.progress(0 if total_p==0 else ok_p/total_p)
        with p2:
            st.metric("Rows with full path", f"{ok_r}/{total_r}")
            st.progress(0 if total_r==0 else ok_r/total_r)

        total_rows = len(df_ws)
        st.markdown("#### Preview (50 rows)")
        if total_rows <= 50:
            st.caption(f"Showing all {total_rows} rows.")
            st.dataframe(df_ws, use_container_width=True)
        else:
            state_key = f"preview_start_{sheet_ws}"
            start_idx = int(ss_get(state_key, 0))
            cprev, cnum, cnext = st.columns([1,2,1])
            with cprev:
                if st.button("◀ Previous 50", key=f"prev50_{sheet_ws}"):
                    start_idx = max(0, start_idx - 50)
            with cnum:
                start_1based = st.number_input(
                    "Start row (1-based)",
                    min_value=1,
                    max_value=max(1, total_rows-49),
                    value=start_idx+1,
                    step=50,
                    help="Pick where to start the 50-row preview.",
                    key=f"startnum_{sheet_ws}"
                )
                start_idx = int(start_1based) - 1
            with cnext:
                if st.button("Next 50 ▶", key=f"next50_{sheet_ws}"):
                    start_idx = min(max(0, total_rows - 50), start_idx + 50)
            ss_set(state_key, start_idx)
            end_idx = min(start_idx + 50, total_rows)
            st.caption(f"Showing rows **{start_idx+1}–{end_idx}** of **{total_rows}**.")
            st.dataframe(df_ws.iloc[start_idx:end_idx], use_container_width=True)

    st.markdown("---")

    # Export / Import Overrides
    with st.expander("📦 Export / Import Overrides (JSON)"):
        overrides_all = ss_get(override_root, {})
        overrides_sheet = overrides_all.get(sheet_ws, {})
        col1, col2 = st.columns([1,2])
        with col1:
            data = json.dumps(overrides_sheet, indent=2).encode("utf-8")
            st.download_button("Export overrides.json", data=data, file_name=f"{sheet_ws}_overrides.json", mime="application/json")
        with col2:
            upfile = st.file_uploader("Import overrides.json", type=["json"], key="ws_imp_json_sel")
            import_mode = st.radio("Import mode", ["Replace", "Merge (prefer import)", "Merge (prefer existing)"], horizontal=True, key="ws_imp_mode_sel")
            if upfile is not None and st.button("Apply Import", key="ws_imp_apply_sel"):
                try:
                    imported = json.loads(upfile.getvalue().decode("utf-8"))
                    if not isinstance(imported, dict):
                        st.error("Invalid JSON: expected an object mapping keys to lists.")
                    else:
                        if import_mode == "Replace":
                            new_over = {}
                            for k,v in imported.items():
                                new_over[k] = enforce_k_five(v if isinstance(v, list) else [v])
                            overrides_all[sheet_ws] = new_over
                        elif import_mode == "Merge (prefer import)":
                            cur = overrides_all.get(sheet_ws, {}).copy()
                            for k,v in imported.items():
                                cur[k] = enforce_k_five(v if isinstance(v, list) else [v])
                            overrides_all[sheet_ws] = cur
                        else:
                            cur = overrides_all.get(sheet_ws, {}).copy()
                            for k,v in imported.items():
                                if k not in cur:
                                    cur[k] = enforce_k_five(v if isinstance(v, list) else [v])
                            overrides_all[sheet_ws] = cur
                        ss_set(override_root, overrides_all)
                        st.success("Overrides imported (stored in-session).")
                except Exception as e:
                    st.error(f"Import failed: {e}")

    # Data quality tools
    with st.expander("🧼 Data quality tools (applies to this sheet)"):
        if df_ws.empty or not validate_headers(df_ws):
            st.info("Load a valid sheet first.")
        else:
            ok_p, total_p = compute_parent_depth_score(df_ws)
            ok_r, total_r = compute_row_path_score(df_ws)
            st.write(f"Parents with 5 children: **{ok_p}/{total_p}**")
            st.write(f"Rows with full path: **{ok_r}/{total_r}**")

            case_mode = st.selectbox("Case normalization", ["None","Title","lower","UPPER"], index=0, key="ws_case_mode_sel")
            syn_text = st.text_area("Synonym map (one per line: A => B)", key="ws_syn_map_text_sel")

            def parse_synonym_map(text: str) -> Dict[str,str]:
                mapping = {}
                for line in text.splitlines():
                    if "=>" in line:
                        a,b = line.split("=>",1)
                        a = a.strip(); b = b.strip()
                        if a and b: mapping[a] = b
                return mapping

            def normalize_label_case(s: str, case_mode: str) -> str:
                s = s.strip()
                if case_mode == "lower": return s.lower()
                if case_mode == "UPPER": return s.upper()
                if case_mode == "Title": return s.title()
                return s

            def normalize_sheet_df(df0: pd.DataFrame, case_mode: str, syn_map: Dict[str,str]) -> pd.DataFrame:
                df2 = df0.copy()
                for col in ["Vital Measurement"] + LEVEL_COLS:
                    if col in df2.columns:
                        def _apply(v):
                            v = normalize_text(v)
                            v = syn_map.get(v, v)
                            v = normalize_label_case(v, case_mode) if case_mode!="None" else v
                            return v
                        df2[col] = df2[col].map(_apply)
                for col in ["Diagnostic Triage","Actions"]:
                    if col in df2.columns:
                        df2[col] = df2[col].map(normalize_text)
                return df2

            if st.button("Normalize sheet now (in-session)", key="ws_norm_sel"):
                syn_map = parse_synonym_map(syn_text)
                df_norm = normalize_sheet_df(df_ws, case_mode, syn_map)
                wb_ws[sheet_ws] = df_norm
                if current_source_code == "upload":
                    ss_set("upload_workbook", wb_ws)
                    st.success("Sheet normalized in-session. Download from Source tab to persist locally.")
                else:
                    ss_set("gs_workbook", wb_ws)
                    st.success("Sheet normalized in-session. Use Push Settings below to write to Google Sheets.")

    # Group rows
    with st.expander("🧩 Group rows (cluster identical labels together)"):
        st.caption("Group rows so identical **Node 1** or **Node 2** values are contiguous. This is a stable grouping.")
        if df_ws.empty or not validate_headers(df_ws):
            st.info("Load a valid sheet first.")
        else:
            group_mode = st.radio("Grouping mode", ["Off", "Node 1", "Node 2"], horizontal=True, key="ws_group_mode_sel")
            scope = st.radio("Grouping scope", ["Whole sheet", "Within Vital Measurement"], horizontal=True, key="ws_group_scope_sel")
            preview = st.checkbox("Show preview (does not modify data)", value=True, key="ws_group_preview_sel")

            def grouped_df(df0: pd.DataFrame, by_col: Optional[str], scope_mode: str) -> pd.DataFrame:
                if by_col is None:
                    return df0.copy()
                df2 = df0.copy()
                df2["_orig_idx"] = np.arange(len(df2))
                key_cols = []
                if scope_mode == "Within Vital Measurement":
                    key_cols = ["Vital Measurement"]
                df2["_gkey"] = df2[by_col].map(lambda x: normalize_text(x).lower())
                sort_by = key_cols + ["_gkey", "_orig_idx"]
                df2 = df2.sort_values(sort_by, kind="stable").drop(columns=["_gkey","_orig_idx"])
                return df2

            by_col = None
            if group_mode == "Node 1":
                by_col = "Node 1"
            elif group_mode == "Node 2":
                by_col = "Node 2"

            df_prev = grouped_df(df_ws, by_col, scope)

            if preview:
                st.dataframe(df_prev.head(100), use_container_width=True)

            csvprev = df_prev.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download grouped view (CSV)",
                data=csvprev,
                file_name=f"{sheet_ws}_grouped_{(by_col or 'none').replace(' ','_').lower()}.csv",
                mime="text/csv"
            )

            colg1, colg2 = st.columns([1,1])
            with colg1:
                if st.button("Apply grouping (in-session)", key="ws_group_apply_sel"):
                    wb_ws[sheet_ws] = df_prev
                    if current_source_code == "upload":
                        ss_set("upload_workbook", wb_ws)
                        st.success("Applied grouping in-session (Upload workbook).")
                    else:
                        ss_set("gs_workbook", wb_ws)
                        st.success("Applied grouping in-session (Google Sheets workbook).")
            with colg2:
                sid_group = ss_get("gs_spreadsheet_id","")
                if current_source_code == "gs" and sid_group and st.button("Apply & push grouping to Google Sheets", key="ws_group_push_sel"):
                    ok = push_to_google_sheets(sid_group, sheet_ws, df_prev)
                    if ok: st.success("Grouping pushed to Google Sheets.")

    # Push settings (bottom)
    st.markdown("---")
    st.subheader("🔧 Google Sheets Push Settings")
    sid_push = st.text_input("Spreadsheet ID", value=ss_get("gs_spreadsheet_id",""), key="ws_push_sid_sel")
    if sid_push:
        ss_set("gs_spreadsheet_id", sid_push)
    default_tab = ss_get("saved_targets", {}).get(sheet_ws, {}).get("tab", f"{sheet_ws}")
    target_tab = st.text_input("Target tab name", value=default_tab, key="ws_push_target_sel")
    include_scope = st.radio("Include scope", ["All completed parents","Only parents edited this session"], horizontal=True, key="ws_push_scope_sel")
    push_backup = st.checkbox("Create a backup tab before overwrite", value=True, key="ws_push_backup_sel")
    dry_run = st.checkbox("Dry-run (build but don't write to Sheets)", value=False, key="ws_push_dry_sel")
    st.caption("This writes the **current view**. (Raw+ build is applied elsewhere when used.)")

    if st.button("📤 Push current view to Google Sheets", type="primary", key="ws_push_btn_sel"):
        if not sid_push or not target_tab:
            st.error("Missing Spreadsheet ID or target tab.")
        elif df_ws.empty or not validate_headers(df_ws):
            st.error("Current sheet is empty or headers mismatch.")
        else:
            if dry_run:
                st.success("Dry-run complete. No changes written to Google Sheets.")
                st.dataframe(df_ws.head(50), use_container_width=True)
                buffer = io.BytesIO()
                with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
                    df_ws.to_excel(writer, index=False, sheet_name=sheet_ws[:31] or "Sheet1")
                st.download_button("Download current view workbook", data=buffer.getvalue(),
                                   file_name="decision_tree_current_view.xlsx",
                                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            else:
                if push_backup:
                    bak = backup_sheet_copy(sid_push, target_tab)
                    if bak:
                        st.info(f"Backed up current '{target_tab}' to '{bak}'.")
                ok = push_to_google_sheets(sid_push, target_tab, df_ws)
                if ok:
                    log = ss_get("push_log", [])
                    log.append({
                        "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "sheet": sheet_ws,
                        "target_tab": target_tab,
                        "spreadsheet_id": sid_push,
                        "rows_written": len(df_ws),
                        "new_rows_added": 0,
                        "scope": "session" if include_scope.endswith("session") else "all",
                    })
                    ss_set("push_log", log)
                    saved = ss_get("saved_targets", {})
                    saved.setdefault(sheet_ws, {})
                    saved[sheet_ws]["tab"] = target_tab
                    ss_set("saved_targets", saved)
                    st.success(f"Pushed {len(df_ws)} rows to '{target_tab}'.")


# ============ UI: Symptoms ============
def render_symptoms():
    st.header("🩺 Symptoms — browse & edit child branches, auto-cascade, undo")

    # Resolve active dataset
    ctx = ss_get("work_context", {})
    wb = None
    override_root = None
    if ctx.get("source") == "upload":
        wb = ss_get("upload_workbook", {})
        override_root = "branch_overrides_upload"
    elif ctx.get("source") == "gs":
        wb = ss_get("gs_workbook", {})
        override_root = "branch_overrides_gs"

    if not wb:
        st.info("Select a sheet in **Workspace** first.")
        return

    sheet = ctx.get("sheet")
    if sheet not in wb:
        st.info("Select a sheet in **Workspace** first.")
        return

    df = wb.get(sheet, pd.DataFrame())
    if df.empty or not validate_headers(df):
        st.info("Selected sheet is empty or headers mismatch.")
        return

    overrides_all = ss_get(override_root, {})
    overrides_sheet = overrides_all.get(sheet, {})

    store = infer_branch_options_with_overrides(df, overrides_sheet)

    # Build virtual parent list
    def compute_virtual_parents(store: Dict[str, List[str]]) -> Dict[int, Set[Tuple[str, ...]]]:
        parents_by_level: Dict[int, Set[Tuple[str, ...]]] = {i: set() for i in range(1, MAX_LEVELS+1)}
        parents_by_level[1].add(tuple())
        for L in range(1, MAX_LEVELS):
            for p in list(parents_by_level[L]):
                key = level_key_tuple(L, p)
                children = [x for x in store.get(key, []) if normalize_text(x)!=""]
                for c in children:
                    parents_by_level[L+1].add(p + (c,))
        for key in store.keys():
            if "|" not in key: continue
            lvl_s, path = key.split("|", 1)
            try: L = int(lvl_s[1:])
            except: continue
            parent_tuple = tuple([] if path=="<ROOT>" else path.split(">"))
            if 1 <= L <= MAX_LEVELS:
                parents_by_level[L].add(parent_tuple)
                for k in range(1, min(L, MAX_LEVELS)+1):
                    parents_by_level.setdefault(k, set())
                    parents_by_level[k].add(tuple(parent_tuple[:k-1]))
        return parents_by_level

    parents_by_level = compute_virtual_parents(store)

    st.subheader("Inspect & Edit")
    level = st.selectbox("Level to inspect (child options of...)", [1,2,3,4,5], format_func=lambda x: f"Node {x}", key="sym_level")

    # Search / quick-jump / compact
    _pending = st.session_state.pop("sym_search_pending", None)
    if _pending is not None:
        st.session_state["sym_search"] = _pending
    top_cols = st.columns([2,1,1,1,2])
    with top_cols[0]:
        search = st.text_input("Search parent symptom/path", key="sym_search").strip().lower()
    with top_cols[1]:
        if st.button("Next Missing", key="sym_next_missing"):
            for pt in sorted(parents_by_level.get(level, set())):
                key = level_key_tuple(level, pt)
                if len([x for x in store.get(key, []) if normalize_text(x)!=""]) == 0:
                    st.session_state["sym_search_pending"] = (" > ".join(pt) or "Top-level (Node 1) options").lower()
                    st.rerun()
    with top_cols[2]:
        if st.button("Next Symptom left out", key="sym_next_leftout"):
            for pt in sorted(parents_by_level.get(level, set())):
                key = level_key_tuple(level, pt)
                n = len([x for x in store.get(key, []) if normalize_text(x)!=""])
                if 1 <= n < 5:
                    st.session_state["sym_search_pending"] = (" > ".join(pt) or "Top-level (Node 1) options").lower()
                    st.rerun()
    with top_cols[3]:
        compact = st.checkbox("Compact mode", value=True, key="sym_compact")
    with top_cols[4]:
        parent_choices = ["(select parent)"] + [(" > ".join(p) or "Top-level (Node 1) options") for p in sorted(parents_by_level.get(level, set()))]
        pick_parent = st.selectbox("Quick jump", parent_choices, key="sym_quick_jump")
        if pick_parent and pick_parent != "(select parent)":
            st.session_state["sym_search_pending"] = pick_parent.lower()
            st.rerun()

    sort_mode = st.radio("Sort by", ["Problem severity (issues first)", "Alphabetical (parent path)"], horizontal=True, key="sym_sort")

    # Build entries
    entries = []
    label_childsets: Dict[Tuple[int,str], set] = {}
    for parent_tuple in sorted(parents_by_level.get(level, set())):
        parent_text = " > ".join(parent_tuple) if parent_tuple else "Top-level (Node 1) options"
        if search and (search not in parent_text.lower()):
            continue
        keyname = level_key_tuple(level, parent_tuple)
        children = [x for x in store.get(keyname, []) if normalize_text(x)!=""]
        n = len(children)
        if n == 0: status = "No group of symptoms"
        elif n < 5: status = "Symptom left out"
        elif n == 5: status = "OK"
        else: status = "Overspecified"
        entries.append((parent_tuple, children, status))
        last_label = parent_tuple[-1] if parent_tuple else "<ROOT>"
        label_childsets.setdefault((level, last_label), set()).add(tuple(sorted([c for c in children])))

    inconsistent_labels = {k for k, v in label_childsets.items() if len(v) > 1}
    status_rank = {"No group of symptoms":0, "Symptom left out":1, "Overspecified":2, "OK":3}
    if sort_mode.startswith("Problem"):
        entries.sort(key=lambda e: (status_rank[e[2]], e[0]))
    else:
        entries.sort(key=lambda e: e[0])

    # Vocabulary from sheet
    def build_vocabulary(df0: pd.DataFrame) -> List[str]:
        vocab = set()
        for col in LEVEL_COLS:
            if col in df0.columns:
                for x in df0[col].dropna().astype(str):
                    x = normalize_text(x)
                    if x: vocab.add(x)
        return sorted(vocab)
    vocab = build_vocabulary(df)
    vocab_opts = ["(pick suggestion)"] + vocab

    # Undo
    if st.button("↩️ Undo last branch edit (session)", key="sym_undo"):
        stack = ss_get("undo_stack", [])
        if not stack:
            st.info("Nothing to undo.")
        else:
            last = stack.pop()
            ss_set("undo_stack", stack)
            context = last.get("context")
            if context == "symptoms":
                overrides_all2 = ss_get(last["override_root"], {})
                overrides_all2[last["sheet"]] = last["overrides_sheet_before"]
                ss_set(last["override_root"], overrides_all2)
                if last.get("df_before") is not None:
                    if ctx.get("source") == "upload":
                        src = ss_get("upload_workbook", {})
                        src[last["sheet"]] = last["df_before"]
                        ss_set("upload_workbook", src)
                    else:
                        src = ss_get("gs_workbook", {})
                        src[last["sheet"]] = last["df_before"]
                        ss_set("gs_workbook", src)
                st.success(f"Undid edit on sheet '{last['sheet']}'.")

    # Render entries with inline editing
    for parent_tuple, children, status in entries:
        keyname = level_key_tuple(level, parent_tuple)
        subtitle = f"{' > '.join(parent_tuple) if parent_tuple else 'Top-level (Node 1) options'} — {status} {'⚠️' if (level, (parent_tuple[-1] if parent_tuple else '<ROOT>')) in inconsistent_labels else ''}"

        with st.expander(subtitle):
            selected_vals = []
            if compact:
                for i in range(5):
                    default_val = children[i] if i < len(children) else ""
                    txt_key = f"sym_txt_{level}_{'__'.join(parent_tuple)}_{i}"
                    sel_key = f"sym_sel_{level}_{'__'.join(parent_tuple)}_{i}"
                    txt = st.text_input(f"Child {i+1}", value=default_val, key=txt_key)
                    pick = st.selectbox("Pick suggestion", options=vocab_opts, index=0, key=sel_key, label_visibility="collapsed")
                    selected_vals.append((txt, pick))
            else:
                cols = st.columns(5)
                for i in range(5):
                    default_val = children[i] if i < len(children) else ""
                    with cols[i]:
                        txt_key = f"sym_txt_{level}_{'__'.join(parent_tuple)}_{i}"
                        sel_key = f"sym_sel_{level}_{'__'.join(parent_tuple)}_{i}"
                        txt = st.text_input(f"Child {i+1}", value=default_val, key=txt_key)
                        pick = st.selectbox("Pick suggestion", options=vocab_opts, index=0, key=sel_key, label_visibility="collapsed")
                    selected_vals.append((txt, pick))

            fill_other = st.checkbox("Fill remaining blanks with 'Other' on save", key=f"sym_other_{level}_{'__'.join(parent_tuple)}")
            enforce_unique = st.checkbox("Enforce uniqueness across the 5", value=True, key=f"sym_unique_{level}_{'__'.join(parent_tuple)}")

            def build_final_values():
                vals = []
                for (txt, pick) in selected_vals:
                    val = pick if pick != "(pick suggestion)" else txt
                    vals.append(normalize_text(val))
                vals = vals[:5] + [""] * max(0, 5 - len(vals))
                if fill_other: vals = [v if v else "Other" for v in vals]
                if enforce_unique:
                    seen = set(); uniq = []
                    for v in vals:
                        if v and v not in seen:
                            uniq.append(v); seen.add(v)
                    vals = uniq + [""] * max(0, 5 - len(uniq))
                    if fill_other: vals = [v if v else "Other" for v in vals]
                return enforce_k_five(vals)

            if st.button("Save 5 branches for this parent", key=f"sym_save_{level}_{'__'.join(parent_tuple)}"):
                fixed = build_final_values()
                overrides_all2 = ss_get(override_root, {})
                overrides_sheet2 = overrides_all2.get(sheet, {}).copy()
                stack = ss_get("undo_stack", [])
                stack.append({
                    "context": "symptoms",
                    "override_root": override_root,
                    "sheet": sheet,
                    "level": level,
                    "parent": parent_tuple,
                    "overrides_sheet_before": overrides_all2.get(sheet, {}).copy(),
                    "df_before": df.copy()
                })
                ss_set("undo_stack", stack)
                overrides_sheet2[keyname] = fixed
                overrides_all2[sheet] = overrides_sheet2
                ss_set(override_root, overrides_all2)
                mark_session_edit(sheet, keyname)
                # Auto-cascade deep
                store2 = infer_branch_options_with_overrides(df, overrides_sheet2)
                vms = sorted(set(df["Vital Measurement"].map(normalize_text).replace("", np.nan).dropna().unique().tolist()))
                df_new, tstats = cascade_anchor_reuse_full(df, store2, vms, [parent_tuple])
                wb[sheet] = df_new
                if ctx.get("source") == "upload":
                    ss_set("upload_workbook", wb)
                else:
                    ss_set("gs_workbook", wb)
                st.success(f"Saved and auto-cascaded: added {tstats['new_rows']} rows, filled {tstats['inplace_filled']} anchors.")


# ============ UI: Conflicts Inspector ============
def render_conflicts():
    st.header("🧩 Conflicts Inspector — unify child sets and resolve to 5")

    # Current sheet
    ctx = ss_get("work_context", {})
    if ctx.get("source") == "upload":
        wb = ss_get("upload_workbook", {})
        override_root = "branch_overrides_upload"
    else:
        wb = ss_get("gs_workbook", {})
        override_root = "branch_overrides_gs"

    if not wb:
        st.info("Select a sheet in **Workspace** first."); return

    sheet = ctx.get("sheet")
    df = wb.get(sheet, pd.DataFrame())
    if df is None or df.empty or not validate_headers(df):
        st.info("Selected sheet is empty or headers mismatch."); return

    overrides_all = ss_get(override_root, {})
    overrides_sheet = overrides_all.get(sheet, {})
    store = infer_branch_options_with_overrides(df, overrides_sheet)

    # Build label->sets index
    def label_children_sets(store: Dict[str, List[str]]) -> Dict[Tuple[int, str], Set[Tuple[str,...]]]:
        idx: Dict[Tuple[int,str], Set[Tuple[str,...]]] = {}
        for key, children in store.items():
            if "|" not in key: continue
            lvl_s, path = key.split("|", 1)
            try: L = int(lvl_s[1:])
            except: continue
            if not (1<=L<=MAX_LEVELS): continue
            if path == "<ROOT>":
                parent_label = "<ROOT>"
            else:
                pt = tuple(path.split(">"))
                parent_label = pt[-1] if pt else "<ROOT>"
            clist = tuple(sorted([normalize_text(c) for c in children if normalize_text(c)!=""]))
            idx.setdefault((L, parent_label), set()).add(clist)
        return idx

    idx = label_children_sets(store)
    conflicts = [(k,v) for k,v in idx.items() if len(v) > 1]
    if not conflicts:
        st.success("No conflicts found — every parent label maps to a single child set.")
        return

    st.caption("A conflict means the same parent **label** maps to different child sets at the same level.")

    for (L, plabel), sets in sorted(conflicts):
        with st.expander(f"Node {L} parent label: {plabel} — {len(sets)} different child sets", expanded=False):
            # Aggregate all unique children across sets to allow picking any of them
            all_children: List[str] = sorted({c for s in sets for c in s})
            st.write("All children observed across sets:")
            st.code(", ".join(all_children) if all_children else "(none)")

            # Provide a chooser for up to 5 children (ensure user can pick any present)
            chosen = st.multiselect(
                "Choose up to 5 children to keep",
                options=all_children,
                default=list(all_children)[:5],
                key=f"conflict_choose_{L}_{plabel}"
            )
            chosen = chosen[:5]
            st.caption(f"Selected {len(chosen)} / 5")

            # Optionally save as child set override for each exact parent tuple that has this label
            if st.button("💾 Save chosen child set for all matching parents", key=f"conflict_save_{L}_{plabel}"):
                if not chosen:
                    st.warning("Pick at least one child (up to 5).")
                else:
                    overrides_all2 = ss_get(override_root, {})
                    overrides_sheet2 = overrides_all2.get(sheet, {}).copy()

                    # Apply the chosen set to every parent tuple whose last label == plabel at that level
                    affected = 0
                    for key in list(store.keys()):
                        if "|" not in key: continue
                        lvl_s, path = key.split("|", 1)
                        try: LL = int(lvl_s[1:])
                        except: continue
                        if LL != L: continue
                        parent_tuple = tuple([] if path=="<ROOT>" else path.split(">"))
                        last_label = "<ROOT>" if (L==1 and not parent_tuple) else (parent_tuple[-1] if parent_tuple else "<ROOT>")
                        if last_label == plabel:
                            overrides_sheet2[level_key_tuple(L, parent_tuple)] = enforce_k_five(chosen)
                            affected += 1

                    overrides_all2[sheet] = overrides_sheet2
                    ss_set(override_root, overrides_all2)
                    mark_session_edit(sheet, f"L{L}|*{plabel}*")
                    st.success(f"Saved chosen set for {affected} parent(s) labeled '{plabel}'. ✅")


# ============ UI: Dictionary ============
def highlight_text(s: str, q: str) -> str:
    if not q:
        return s
    try:
        pattern = re.compile(re.escape(q), re.IGNORECASE)
        def _mark(m): 
            return f"<mark>{m.group(0)}</mark>"
        return pattern.sub(_mark, s)
    except Exception:
        return s.replace(q, f"<mark>{q}</mark>")

def render_dictionary():
    st.header("📖 Dictionary — all symptom labels")

    # Sources
    sources_avail = []
    if ss_get("upload_workbook", {}): sources_avail.append("Upload workbook")
    if ss_get("gs_workbook", {}): sources_avail.append("Google Sheets workbook")
    if not sources_avail:
        st.info("Load data in the **Source** tab first.")
        return

    st.subheader("Sources")
    source_choice = st.multiselect("Include sources", sources_avail, default=sources_avail, key="dict_sources")

    dfs: List[pd.DataFrame] = []
    if "Upload workbook" in source_choice:
        wb_u = ss_get("upload_workbook", {})
        pick_u = st.multiselect("Sheets (Upload)", list(wb_u.keys()), default=list(wb_u.keys()), key="dict_pick_u")
        for nm in pick_u:
            if nm in wb_u:
                dfs.append(wb_u[nm])
    if "Google Sheets workbook" in source_choice:
        wb_g = ss_get("gs_workbook", {})
        pick_g = st.multiselect("Sheets (Google Sheets)", list(wb_g.keys()), default=list(wb_g.keys()), key="dict_pick_g")
        for nm in pick_g:
            if nm in wb_g:
                dfs.append(wb_g[nm])

    if not dfs:
        st.info("Select at least one sheet.")
        return

    # Build dictionary
    counts: Dict[str,int] = {}
    levels_map: Dict[str, Set[int]] = {}
    for df0 in dfs:
        if df0 is None or df0.empty or not validate_headers(df0):
            continue
        for lvl, col in enumerate(LEVEL_COLS, start=1):
            if col in df0.columns:
                for val in df0[col].astype(str).map(normalize_text):
                    if not val: 
                        continue
                    counts[val] = counts.get(val, 0) + 1
                    levels_map.setdefault(val, set()).add(lvl)

    if not counts:
        st.info("No symptom labels found."); return

    quality_map = ss_get("symptom_quality", {})

    rows = []
    for symptom, cnt in counts.items():
        levels_list = sorted(list(levels_map.get(symptom, set())))
        quality = quality_map.get(symptom, "Normal")
        rows.append({
            "Symptom": symptom,
            "Count": cnt,
            "Levels": ", ".join([f"Node {i}" for i in levels_list]),
            "RedFlag": (quality == "Red Flag")
        })
    dict_df = pd.DataFrame(rows).sort_values(["Symptom"]).reset_index(drop=True)

    # Controls
    st.subheader("Search & Filter")
    c1, c2, c3, c4 = st.columns([2,1,1,1])
    with c1:
        q = st.text_input("Search symptom (case-insensitive)", key="dict_search").strip()
    with c2:
        show_only_rf = st.checkbox("Show only Red Flags", value=False, key="dict_only_rf")
    with c3:
        list_mode = st.checkbox("List mode with highlights", value=False, help="Show a simple list with <mark>-highlighted matches. Turn off for the editable table.")
    with c4:
        sort_mode = st.selectbox("Sort by", ["A → Z", "Count ↓", "Red Flags first"], index=0, key="dict_sort")

    view = dict_df.copy()
    if q:
        view = view[view["Symptom"].str.contains(q, case=False, na=False)]
    if show_only_rf:
        view = view[view["RedFlag"] == True]

    if sort_mode == "Count ↓":
        view = view.sort_values(["Count","Symptom"], ascending=[False, True])
    elif sort_mode == "Red Flags first":
        view = view.sort_values(["RedFlag","Symptom"], ascending=[False, True])
    else:
        view = view.sort_values(["Symptom"], ascending=[True])

    st.caption(f"{len(view)} symptoms match the current filters.")

    if list_mode:
        st.markdown("### Results (highlighted)")
        page_size = st.selectbox("Items per page", [25, 50, 100, 200], index=1, key="dict_list_pagesize")
        total = len(view)
        max_page = max(1, math.ceil(total / page_size))
        page = st.number_input("Page", min_value=1, max_value=max_page, value=1, step=1, key="dict_list_page")
        start = (page - 1) * page_size
        end = min(start + page_size, total)
        slice_df = view.iloc[start:end]

        for _, row in slice_df.iterrows():
            sym = row["Symptom"]
            hl = highlight_text(sym, q) if q else sym
            badge = "🔴" if row["RedFlag"] else "🟢"
            st.markdown(f"- {badge} <strong>{hl}</strong> · Count: **{int(row['Count'])}** · Levels: {row['Levels']}", unsafe_allow_html=True)

        csv_data = view.to_csv(index=False).encode("utf-8")
        st.download_button("Download current view (CSV)", data=csv_data, file_name="dictionary_filtered.csv", mime="text/csv")
        st.info("Switch off 'List mode' to edit Red Flags in a table.")
        return

    st.markdown("### Edit Red Flags")
    page_size = st.selectbox("Rows per page", [25, 50, 100, 200], index=1, key="dict_table_pagesize")
    total = len(view)
    max_page = max(1, math.ceil(total / page_size))
    page = st.number_input("Page", min_value=1, max_value=max_page, value=1, step=1, key="dict_table_page")
    start = (page - 1) * page_size
    end = min(start + page_size, total)
    slice_df = view.iloc[start:end].reset_index(drop=True)

    qa1, qa2, qa3 = st.columns([1,1,3])
    with qa1:
        if st.button("Select all on page", key="dict_select_all_page"):
            slice_df["RedFlag"] = True
    with qa2:
        if st.button("Clear all on page", key="dict_clear_all_page"):
            slice_df["RedFlag"] = False
    with qa3:
        st.caption("Tip: Use filters and pagination to focus on specific subsets.")

    edited = st.data_editor(
        slice_df,
        num_rows="dynamic",
        use_container_width=True,
        hide_index=True,
        column_config={
            "RedFlag": st.column_config.CheckboxColumn("Red Flag"),
            "Symptom": st.column_config.TextColumn("Symptom", disabled=True),
            "Count": st.column_config.NumberColumn("Count", disabled=True),
            "Levels": st.column_config.TextColumn("Levels", disabled=True),
        }
    )

    colS1, colS2 = st.columns([1,3])
    with colS1:
        if st.button("💾 Save changes", key="dict_save_changes"):
            view.loc[view.index[start:end], "RedFlag"] = edited["RedFlag"].values

            new_quality = {}
            new_quality.update(quality_map)

            dict_df_updates = dict_df.set_index("Symptom")
            view_updates = view.set_index("Symptom")["RedFlag"]
            dict_df_updates.loc[view_updates.index, "RedFlag"] = view_updates.values
            dict_df2 = dict_df_updates.reset_index()

            for _, r in dict_df2.iterrows():
                new_quality[r["Symptom"]] = "Red Flag" if bool(r["RedFlag"]) else "Normal"

            ss_set("symptom_quality", new_quality)
            st.success("Red Flags saved for this session.")

    with colS2:
        csv_data = view.to_csv(index=False).encode("utf-8")
        st.download_button("Download current view (CSV)", data=csv_data, file_name="dictionary_filtered.csv", mime="text/csv")


# ============ UI: Validation ============
def render_validation():
    st.header("🧪 Validation")

    ctx = ss_get("work_context", {})
    wb = ss_get("upload_workbook", {}) if ctx.get("source")=="upload" else ss_get("gs_workbook", {})
    if not wb:
        st.info("Select a sheet in **Workspace** first."); return
    sheet = ctx.get("sheet")
    df = wb.get(sheet, pd.DataFrame())
    if df is None or df.empty or not validate_headers(df):
        st.info("Selected sheet is empty or headers mismatch."); return

    override_root = "branch_overrides_upload" if ctx.get("source")=="upload" else "branch_overrides_gs"
    overrides = ss_get(override_root, {}).get(sheet, {})
    redflag_map = ss_get("symptom_quality", {})

    report = compute_validation_report(df, overrides=overrides, redflag_map=redflag_map)
    counts = report["counts"]

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Orphans (loose)", counts["orphans_loose"])
    c2.metric("Orphans (strict)", counts["orphans_strict"])
    c3.metric("Loops", counts["loops"])
    c4.metric("Missing Red Flag", counts["missing_redflag"])

    with st.expander("Orphans (loose)"):
        if counts["orphans_loose"] == 0:
            st.success("None 🎉")
        else:
            st.dataframe(pd.DataFrame(report["orphans_loose"]), use_container_width=True, height=300)

    with st.expander("Orphans (strict)"):
        if counts["orphans_strict"] == 0:
            st.success("None 🎉")
        else:
            st.dataframe(pd.DataFrame(report["orphans_strict"]), use_container_width=True, height=300)

    with st.expander("Loops"):
        if counts["loops"] == 0:
            st.success("None 🎉")
        else:
            st.dataframe(pd.DataFrame(report["loops"]), use_container_width=True, height=300)

    with st.expander("Missing Red Flag coverage"):
        if counts["missing_redflag"] == 0:
            st.success("All parents have at least one Red Flag child. ✅")
        else:
            st.dataframe(pd.DataFrame(report["missing_redflag"]), use_container_width=True, height=300)


# ============ UI: Calculator (placeholder utilities) ============
def render_calculator():
    st.header("🧮 Calculator (utilities)")
    st.caption("Quick helpers for your tree. We’ll grow this into a full toolset in v6.3.x.")

    ctx = ss_get("work_context", {})
    wb = ss_get("upload_workbook", {}) if ctx.get("source")=="upload" else ss_get("gs_workbook", {})
    if not wb:
        st.info("Select a sheet in **Workspace** first."); return
    sheet = ctx.get("sheet")
    df = wb.get(sheet, pd.DataFrame())
    if df is None or df.empty or not validate_headers(df):
        st.info("Selected sheet is empty or headers mismatch."); return

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Count unique labels by level")
        for i, col in enumerate(LEVEL_COLS, start=1):
            uniq = df[col].map(normalize_text).replace("", np.nan).dropna().nunique()
            st.write(f"Node {i}: **{uniq}** unique labels")

    with c2:
        st.subheader("Row filter (contains label)")
        label_q = st.text_input("Contains label (case-insensitive)", key="calc_label_q").strip().lower()
        if label_q:
            mask = False
            for col in ["Vital Measurement"] + LEVEL_COLS:
                mask = (df[col].astype(str).str.lower().str.contains(label_q)) | mask
            st.write(f"Matching rows: **{int(mask.sum())}**")
            st.dataframe(df[mask].head(100), use_container_width=True)


# ============ UI: Push Log ============
def render_pushlog():
    st.header("📜 Push Log")
    log = ss_get("push_log", [])
    if not log:
        st.info("No pushes recorded this session.")
    else:
        df_log = pd.DataFrame(log)
        st.dataframe(df_log, use_container_width=True)
        csv = df_log.to_csv(index=False).encode("utf-8")
        st.download_button("Download push log (CSV)", data=csv, file_name="push_log.csv", mime="text/csv")


# ============ UI: Visualizer ============
def _get_current_df_for_visualizer() -> Tuple[Optional[pd.DataFrame], str]:
    ctx = ss_get("work_context", {})
    src = ctx.get("source")
    sheet = ctx.get("sheet")

    if src == "upload":
        wb = ss_get("upload_workbook", {})
        if sheet in wb:
            return wb[sheet], sheet
    elif src == "gs":
        wb = ss_get("gs_workbook", {})
        if sheet in wb:
            return wb[sheet], sheet

    wb_u = ss_get("upload_workbook", {})
    if wb_u:
        name = next(iter(wb_u))
        return wb_u[name], name

    wb_g = ss_get("gs_workbook", {})
    if wb_g:
        name = next(iter(wb_g))
        return wb_g[name], name

    return None, "(no sheet loaded)"

def _unique_vm_values(df: pd.DataFrame) -> List[str]:
    if "Vital Measurement" not in df.columns:
        return []
    vals = [normalize_text(x) for x in df["Vital Measurement"].dropna().astype(str)]
    uniq = []
    seen = set()
    for v in vals:
        if v and v not in seen:
            seen.add(v); uniq.append(v)
    return sorted(uniq)

def _build_edges_for_visualizer(
    df: pd.DataFrame,
    limit_rows: int = 20000,
    scope_vm: Optional[str] = None,
    collapse_by_label_per_level: bool = True,
) -> Tuple[Set[str], List[Tuple[str, str]], Dict[str, Dict[str, str]]]:
    nodes: Set[str] = set()
    edges: List[Tuple[str, str]] = []
    node_attrs: Dict[str, Dict[str, str]] = {}

    if df is None or df.empty or not validate_headers(df):
        return nodes, edges, node_attrs

    df2 = df.copy()
    if scope_vm:
        df2 = df2[df2["Vital Measurement"].astype(str).map(normalize_text) == normalize_text(scope_vm)]

    for i, (_, row) in enumerate(df2.iterrows()):
        if i >= limit_rows:
            break
        vm = normalize_text(row.get("Vital Measurement", ""))
        path = [normalize_text(row.get(c, "")) for c in LEVEL_COLS]
        prev_id = None
        if vm:
            vm_id = f"L0:{vm}" if collapse_by_label_per_level else f"L0:{vm}:{i}"
            if vm_id not in nodes:
                nodes.add(vm_id)
                node_attrs[vm_id] = {"label": vm, "title": f"Vital Measurement: {vm}"}
            prev_id = vm_id

        for li, label in enumerate(path, start=1):
            if not label:
                break
            node_id = f"L{li}:{label}" if collapse_by_label_per_level else f"L{li}:{label}:{i}"
            if node_id not in nodes:
                nodes.add(node_id)
                node_attrs[node_id] = {"label": label, "title": f"Node {li}: {label}"}
            if prev_id is not None:
                edges.append((prev_id, node_id))
            prev_id = node_id

    edges = list({(a, b) for (a, b) in edges})
    return nodes, edges, node_attrs

def _apply_pyvis_options(net: 'Network', hierarchical: bool):
    if hierarchical:
        options = {
            "layout": {
                "hierarchical": {"enabled": True, "direction": "UD", "sortMethod": "directed"}
            },
            "physics": {"enabled": False},
            "nodes": {"shape": "dot", "size": 12},
            "edges": {"arrows": {"to": {"enabled": True}}}
        }
    else:
        options = {
            "physics": {"enabled": True, "stabilization": {"enabled": True}},
            "nodes": {"shape": "dot", "size": 12},
            "edges": {"arrows": {"to": {"enabled": True}}}
        }
    net.set_options(json.dumps(options))

def render_visualizer():
    st.header("🌳 Visualizer")
    if not HAS_PYVIS:
        st.info("PyVis is not installed; visualizer unavailable. Add `pyvis` to requirements to enable.")
        return

    df, sheet_name = _get_current_df_for_visualizer()
    if df is None or df.empty or not validate_headers(df):
        st.info("No valid sheet found. Load data in **Source** and select a sheet in **Workspace**.")
        return

    st.caption(f"Showing sheet: **{sheet_name}**")

    c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
    with c1:
        hierarchical = st.checkbox("Hierarchical layout", value=True, help="Top-to-bottom layered view.")
    with c2:
        collapse = st.checkbox("Merge same labels per level", value=True, help="If ON, identical labels at the same level are merged.")
    with c3:
        vms = _unique_vm_values(df)
        vm_scope = st.selectbox("Filter by Vital Measurement", options=["(All)"] + vms, index=0)
        vm_sel = None if vm_scope == "(All)" else vm_scope
    with c4:
        limit = st.number_input("Row limit", min_value=100, max_value=100000, value=5000, step=500,
                                help="Maximum rows to scan when building the graph.")

    st.markdown("---")

    nodes, edges, node_attrs = _build_edges_for_visualizer(df, limit_rows=int(limit), scope_vm=vm_sel,
                                                           collapse_by_label_per_level=collapse)
    if not nodes:
        st.info("No nodes to visualize with the current filters.")
        return

    net = Network(height="650px", width="100%", directed=True, notebook=False)
    _apply_pyvis_options(net, hierarchical=hierarchical)

    for nid in nodes:
        info = node_attrs.get(nid, {})
        label = info.get("label", nid)
        title = info.get("title", label)
        try:
            level_prefix = nid.split(":")[0]
            level_num = int(level_prefix[1:])
        except Exception:
            level_num = 0
        color = ["#4f46e5","#2563eb","#059669","#16a34a","#d97706","#dc2626"][min(level_num,5)]
        net.add_node(nid, label=label, title=title, color=color)

    for (src, dst) in edges:
        net.add_edge(src, dst)

    try:
        html = net.generate_html()
        st.components.v1.html(html, height=680, scrolling=True)
    except Exception:
        import tempfile
        with tempfile.TemporaryDirectory() as tmpd:
            out = os.path.join(tmpd, "graph.html")
            net.write_html(out)
            with open(out, "r", encoding="utf-8") as f:
                st.components.v1.html(f.read(), height=680, scrolling=True)

    with st.expander("Legend & Tips"):
        st.markdown(
            "- **Colors by level** (VM → Node 1 → Node 2 → Node 3 → Node 4 → Node 5).\n"
            "- Turn **Merge same labels per level** OFF to see every occurrence (more crowded).\n"
            "- Use **Filter by Vital Measurement** to focus a single tree.\n"
            "- Increase **Row limit** for larger sheets."
        )


# ============ MAIN APP LAYOUT ============
left, right = st.columns([1,2])
with left:
    st.title(f"🌳 Decision Tree Builder — {APP_VERSION}")
with right:
    badges = ""
    wb_upload = ss_get("upload_workbook", {})
    if wb_upload:
        first_sheet = next(iter(wb_upload))
        df0 = wb_upload[first_sheet]
        if validate_headers(df0) and not df0.empty:
            badges = progress_badge_html(df0)
    st.markdown(badges, unsafe_allow_html=True)
    if "gcp_service_account" not in st.secrets:
        st.error("Google Sheets not configured. Add your service account JSON under [gcp_service_account].")
    else:
        st.caption("Google Sheets linked ✓")

st.caption("Canonical headers: Vital Measurement, Node 1, Node 2, Node 3, Node 4, Node 5, Diagnostic Triage, Actions")

tabs = st.tabs([
    "📂 Source",
    "🗂 Workspace",
    "🩺 Symptoms",
    "🧩 Conflicts",
    "📖 Dictionary",
    "🧪 Validation",
    "🧮 Calculator",
    "📜 Push Log",
    "🌳 Visualizer"
])

with tabs[0]:
    render_source()

with tabs[1]:
    render_workspace()

with tabs[2]:
    render_symptoms()

with tabs[3]:
    render_conflicts()

with tabs[4]:
    render_dictionary()

with tabs[5]:
    render_validation()

with tabs[6]:
    render_calculator()

with tabs[7]:
    render_pushlog()

with tabs[8]:
    render_visualizer()
