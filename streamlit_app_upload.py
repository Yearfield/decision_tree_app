# streamlit_app_upload.py
# Decision Tree Builder — Unified Monolith
# Version: v6.3.1.2

from __future__ import annotations

import io
import json
from typing import Dict, List, Tuple, Optional, Set

import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime

# Optional visualizer dependency (handled gracefully if missing)
try:
    from pyvis.network import Network
    _HAS_PYVIS = True
except Exception:
    _HAS_PYVIS = False


# ==============================
# VERSION / CONFIG
# ==============================
APP_VERSION = "v6.3.1.2"
CANON_HEADERS = [
    "Vital Measurement", "Node 1", "Node 2", "Node 3", "Node 4", "Node 5",
    "Diagnostic Triage", "Actions"
]
LEVEL_COLS = ["Node 1","Node 2","Node 3","Node 4","Node 5"]
MAX_LEVELS = 5

st.set_page_config(page_title=f"Decision Tree Builder — {APP_VERSION}", page_icon="🌳", layout="wide")


# ==============================
# Session helpers
# ==============================
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


# ==============================
# Core helpers
# ==============================
def normalize_text(x: str) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    return str(x).strip()

def validate_headers(df: pd.DataFrame) -> bool:
    # allow extra columns, but verify canonical ones exist
    return all(c in df.columns for c in CANON_HEADERS)

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

def enforce_k_five(opts: List[str]) -> List[str]:
    clean = [normalize_text(o) for o in opts if normalize_text(o) != ""]
    if len(clean) > 5:
        clean = clean[:5]
    elif len(clean) < 5:
        clean = clean + [""]*(5-len(clean))
    return clean


# ==============================
# Inference (parent -> children) and metrics
# ==============================
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

def compute_parent_depth_score_counts(df: pd.DataFrame) -> Tuple[int, int]:
    """
    Count how many parents (at all levels) currently have exactly 5 non-empty children.
    Returns (ok_count, total_parent_count).
    """
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

def compute_row_path_score_counts(df: pd.DataFrame) -> Tuple[int, int]:
    if df is None or df.empty:
        return (0,0)
    # avoid applymap deprecation — use per-column map:
    nodes = df[LEVEL_COLS].apply(lambda s: s.map(normalize_text))
    full = nodes.ne("").all(axis=1)
    return int(full.sum()), int(len(df))


# ==============================
# Validation rules
# ==============================
def build_label_children_index(store: Dict[str, List[str]]) -> Dict[Tuple[int, str], List[str]]:
    """
    (level, parent_label) -> children list
    For level L, parent_label is "<ROOT>" when L==1 else the last label of the parent tuple.
    """
    idx: Dict[Tuple[int, str], List[str]] = {}
    for key, children in (store or {}).items():
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

def detect_orphans(
    df: pd.DataFrame,
    overrides: Optional[Dict[str, List[str]]] = None,
    strict: bool = False
) -> List[Dict[str, object]]:
    """
    Orphans: a child label that never appears as a parent for the next level.
    strict=False => by label; strict=True => by exact parent tuple.
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

def detect_orphan_nodes(df: pd.DataFrame) -> List[str]:
    """
    Simpler row-based orphan check: any Node k that has an empty Node k-1 on its row.
    """
    problems = []
    for _, row in df.iterrows():
        for i in range(1, len(LEVEL_COLS)):
            cur = normalize_text(row.get(LEVEL_COLS[i], ""))
            prev = normalize_text(row.get(LEVEL_COLS[i-1], ""))
            if cur and not prev:
                problems.append(f"{LEVEL_COLS[i]}:{cur}")
    return problems

def detect_loops(df: pd.DataFrame) -> List[Dict[str, object]]:
    """
    Circular branches within a single row (a label repeats later in the same path).
    """
    assert validate_headers(df), "Headers must match canonical schema."
    results: List[Dict[str, object]] = []
    nodes = df[LEVEL_COLS].apply(lambda s: s.map(normalize_text))
    for idx, row in nodes.iterrows():
        path = [row[c] for c in LEVEL_COLS if normalize_text(row[c]) != ""]
        if not path:
            continue
        seen_pos: Dict[str, int] = {}
        repeats: List[Tuple[str, int, int]] = []
        for i, label in enumerate(path):
            if label in seen_pos:
                repeats.append((label, seen_pos[label]+1, i+1))
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
    For every parent with children, verify at least one child is 'Red Flag'.
    redflag_map format: {label -> "Red Flag"|"Normal"} (exact label keys).
    """
    assert validate_headers(df), "Headers must match canonical schema."
    store = infer_branch_options_with_overrides(df, overrides or {})
    results: List[Dict[str, object]] = []

    if not redflag_map:
        return results

    rf = {normalize_text(k): ("Red Flag" if str(v).strip().lower()=="red flag" else "Normal")
          for k, v in redflag_map.items()}

    for key, children in store.items():
        if "|" not in key:
            continue
        try:
            L = int(key.split("|", 1)[0][1:])
        except Exception:
            continue
        non_empty = [normalize_text(x) for x in children if normalize_text(x) != ""]
        if not non_empty:
            continue
        has_rf = any(rf.get(x, "Normal") == "Red Flag" for x in non_empty)
        if not has_rf:
            path = key.split("|", 1)[1]
            parent_tuple = tuple([] if path == "<ROOT>" else path.split(">"))
            results.append({
                "level": L,
                "parent": parent_tuple,
                "children": non_empty,
                "redflag_present": False
            })

    return results


# ==============================
# Google Sheets helpers
# ==============================
def push_to_google_sheets(spreadsheet_id: str, sheet_name: str, df: pd.DataFrame) -> bool:
    try:
        import gspread
        from google.oauth2.service_account import Credentials
        if "gcp_service_account" not in st.secrets:
            st.error("Google Sheets not configured. Add your service account JSON under [gcp_service_account].")
            return False

        sa_info = st.secrets["gcp_service_account"]
        scopes = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
        creds = Credentials.from_service_account_info(sa_info, scopes=scopes)
        client = gspread.authorize(creds)

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
        import gspread
        from google.oauth2.service_account import Credentials
        if "gcp_service_account" not in st.secrets:
            return None
        sa_info = st.secrets["gcp_service_account"]
        scopes = ["https://www.googleapis.com/auth/spreadsheets","https://www.googleapis.com/auth/drive"]
        creds = Credentials.from_service_account_info(sa_info, scopes=scopes)
        client = gspread.authorize(creds)
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


# ==============================
# Sorting helpers
# ==============================
def sort_sheet_for_view(df: pd.DataFrame, redflag_map: Optional[Dict[str, str]] = None) -> pd.DataFrame:
    """
    Sort so that Node 1 is grouped, and within Node 2:
      - Red Flags appear first
      - then alphabetical by Node 2
    Then by Node 3..5 alphabetically. Keeps VM grouping natural.
    """
    if df is None or df.empty:
        return df

    df2 = df.copy()
    # Ensure columns exist (non-destructive for extra columns)
    for c in CANON_HEADERS:
        if c not in df2.columns:
            df2[c] = ""

    # Build RF priority for Node 2 (0 if Red Flag, 1 otherwise)
    rf_map = {k: v for k, v in (redflag_map or {}).items()}
    def rf_priority(val: str) -> int:
        lab = normalize_text(val)
        status = rf_map.get(val, rf_map.get(lab, "Normal"))
        return 0 if str(status).strip().lower() == "red flag" else 1

    df2["_rf2"] = df2["Node 2"].map(rf_priority) if "Node 2" in df2.columns else 1
    # Stable sort keys
    sort_cols = ["Vital Measurement", "Node 1", "_rf2", "Node 2", "Node 3", "Node 4", "Node 5"]
    sort_cols = [c for c in sort_cols if c in df2.columns]
    df2 = df2.sort_values(sort_cols, kind="stable").drop(columns=["_rf2"], errors="ignore").reset_index(drop=True)
    return df2

def order_rows_for_push(df: pd.DataFrame, redflag_map: Optional[Dict[str, str]] = None) -> pd.DataFrame:
    """
    Push ordering that:
      - Preserves Vital Measurement grouping,
      - Groups by Node 1,
      - Optionally lifts any row containing a Red Flag label across Node 1–5 (Dictionary-driven),
      - Then sorts Node 2..5 alphabetically for stability.
    """
    if df is None or df.empty:
        return df

    df2 = df.copy()
    for c in CANON_HEADERS:
        if c not in df2.columns:
            df2[c] = ""

    # Base stable keys
    df2["_vm_key"] = df2.get("Vital Measurement", "").astype(str).str.strip().str.lower()
    df2["_n1_key"] = df2.get("Node 1", "").astype(str).str.strip().str.lower()

    rf = None
    if isinstance(redflag_map, dict) and len(redflag_map) > 0:
        rf = {str(k).strip().lower(): str(v).strip().lower() for k, v in redflag_map.items()}

    if rf:
        def _rf_hit(row):
            for col in ("Node 1","Node 2","Node 3","Node 4","Node 5"):
                val = str(row.get(col, "")).strip().lower()
                if val and rf.get(val) == "red flag":
                    return 1
            return 0
        df2["_rf_score"] = df2.apply(_rf_hit, axis=1)
        sort_cols = ["_vm_key", "_rf_score", "_n1_key", "Node 2", "Node 3", "Node 4", "Node 5"]
        ascending = [True, False, True, True, True, True, True]
    else:
        sort_cols = ["_vm_key", "_n1_key", "Node 2", "Node 3", "Node 4", "Node 5"]
        ascending = [True, True, True, True, True, True]

    sort_cols = [c for c in sort_cols if c in df2.columns]
    df2 = df2.sort_values(by=sort_cols, ascending=ascending[:len(sort_cols)], kind="stable")
    return df2.drop(columns=["_vm_key","_n1_key","_rf_score"], errors="ignore").reset_index(drop=True)


# ==============================
# UI: Header / Badges
# ==============================
left, right = st.columns([1,2])
with left:
    st.title(f"🌳 Decision Tree Builder — {APP_VERSION}")
with right:
    # If a workbook is present in session, show quick badge metrics for first sheet
    badges_html = ""
    wb_upload = ss_get("upload_workbook", {})
    if wb_upload:
        first_sheet = next(iter(wb_upload))
        df0 = wb_upload[first_sheet]
        if validate_headers(df0) and not df0.empty:
            ok_p, total_p = compute_parent_depth_score_counts(df0)
            ok_r, total_r = compute_row_path_score_counts(df0)
            pct_p = 0 if total_p==0 else int(round(100*ok_p/total_p))
            pct_r = 0 if total_r==0 else int(round(100*ok_r/total_r))
            badges_html = f"""
            <style>
            .badge-wrap {{ display:flex; gap:16px; flex-wrap: wrap; }}
            .b {{ padding:8px 10px; border-radius:12px; border:1px solid #dbe0e6; background:#f8fafc; 
                 font: 12px/1.2 -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Helvetica, Arial, sans-serif; color:#0f172a; }}
            .bar {{ position:relative; height:8px; background:#eef2f6; border:1px solid #dbe0e6; border-radius:8px; overflow:hidden; width:160px; }}
            .fill1 {{ position:absolute; left:0; top:0; bottom:0; width:{pct_p}%; background:linear-gradient(90deg,#60a5fa,#2563eb); }}
            .fill2 {{ position:absolute; left:0; top:0; bottom:0; width:{pct_r}%; background:linear-gradient(90deg,#34d399,#059669); }}
            .strong {{ font-weight:600; }}
            </style>
            <div class='badge-wrap'>
              <div class='b'>
                <div>Parents 5/5 — <span class='strong'>{ok_p}/{total_p}</span></div>
                <div class='bar'><div class='fill1'></div></div>
              </div>
              <div class='b'>
                <div>Rows full path — <span class='strong'>{ok_r}/{total_r}</span></div>
                <div class='bar'><div class='fill2'></div></div>
              </div>
            </div>
            """
    st.markdown(badges_html, unsafe_allow_html=True)

if "gcp_service_account" not in st.secrets:
    st.error("Google Sheets not configured. Add your service account JSON under [gcp_service_account].")
else:
    st.caption("Google Sheets linked ✓")

st.caption("Canonical headers: Vital Measurement, Node 1, Node 2, Node 3, Node 4, Node 5, Diagnostic Triage, Actions")


# ==============================
# Tabs
# ==============================
tabs = st.tabs([
    "📂 Source",
    "🗂 Workspace",
    "🧬 Symptoms",
    "📖 Dictionary",
    "⚖️ Conflicts",
    "🧪 Validation",
    "🧮 Calculator",
    "🌳 Visualizer",
    "📜 Push Log",
])


# ==============================
# Tab: Source
# ==============================
with tabs[0]:
    st.header("📂 Source")
    st.markdown("Load/create your decision tree data.")

    # Upload workbook
    st.subheader("📤 Upload Workbook")
    file = st.file_uploader("Upload XLSX or CSV", type=["xlsx","xls","csv"])
    if file is not None:
        if file.name.lower().endswith(".csv"):
            df = pd.read_csv(file)
            # Guarantee canonical columns
            for c in CANON_HEADERS:
                if c not in df.columns:
                    df[c] = ""
            df = df[CANON_HEADERS]
            # Remove empty node blocks
            node_block = ["Vital Measurement"] + LEVEL_COLS
            df = df[~df[node_block].apply(lambda r: all(normalize_text(v) == "" for v in r), axis=1)].copy()
            # Sort for view using RF map (if any)
            df = sort_sheet_for_view(df, ss_get("symptom_quality", {}))
            wb = {"Sheet1": df}
        else:
            xls = pd.ExcelFile(file)
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
                dfx = sort_sheet_for_view(dfx, ss_get("symptom_quality", {}))
                sheets[name] = dfx
            wb = sheets
        ss_set("upload_workbook", wb)
        ss_set("upload_filename", file.name)
        st.success(f"Loaded {len(wb)} sheet(s) into session.")

    # Load or refresh from Google Sheets
    st.markdown("---")
    st.subheader("🔄 Google Sheets")
    sid = st.text_input("Spreadsheet ID", value=ss_get("gs_spreadsheet_id",""))
    if sid:
        ss_set("gs_spreadsheet_id", sid)
    gs_sheet = st.text_input("Sheet name to load (e.g., BP)", value=ss_get("gs_default_sheet","BP"))
    if st.button("Load / Refresh from Google Sheets"):
        try:
            import gspread
            from google.oauth2.service_account import Credentials
            if "gcp_service_account" not in st.secrets:
                st.error("Google Sheets not configured. Add your service account JSON under [gcp_service_account].")
            else:
                sa_info = st.secrets["gcp_service_account"]
                scopes = ["https://www.googleapis.com/auth/spreadsheets","https://www.googleapis.com/auth/drive"]
                creds = Credentials.from_service_account_info(sa_info, scopes=scopes)
                client = gspread.authorize(creds)
                sh = client.open_by_key(sid)
                ws = sh.worksheet(gs_sheet)
                values = ws.get_all_values()
                if not values:
                    st.error("Selected sheet is empty.")
                else:
                    header = values[0]; rows = values[1:]
                    df_g = pd.DataFrame(rows, columns=header)
                    for c in CANON_HEADERS:
                        if c not in df_g.columns:
                            df_g[c] = ""
                    df_g = df_g[CANON_HEADERS]
                    node_block = ["Vital Measurement"] + LEVEL_COLS
                    df_g = df_g[~df_g[node_block].apply(lambda r: all(normalize_text(v) == "" for v in r), axis=1)].copy()
                    df_g = sort_sheet_for_view(df_g, ss_get("symptom_quality", {}))
                    wb_g = ss_get("gs_workbook", {})
                    wb_g[gs_sheet] = df_g
                    ss_set("gs_workbook", wb_g)
                    st.success(f"Loaded '{gs_sheet}' from Google Sheets.")
        except Exception as e:
            st.error(f"Google Sheets error: {e}")

    # VM builder stubs
    st.markdown("---")
    st.subheader("🧩 VM Builder (create Vital Measurements and auto-cascade)")
    st.caption("Stub for now in this monolith. Use Symptoms/Dictionary for edits.")

    st.markdown("---")
    st.subheader("🧙 VM Build Wizard (step-by-step to Node 5)")
    st.caption("Stub for now in this monolith. Coming in future releases.")


# ==============================
# Tab: Workspace
# ==============================
with tabs[1]:
    st.header("🗂 Workspace Selection")

    # Choose data source
    sources = []
    if ss_get("upload_workbook", {}):
        sources.append("Upload workbook")
    if ss_get("gs_workbook", {}):
        sources.append("Google Sheets workbook")

    if not sources:
        st.info("Load a workbook in the **Source** tab first (upload file or Google Sheets).")
    else:
        source_ws = st.radio("Choose data source", sources, horizontal=True, key="ws_source_sel")
        if source_ws == "Upload workbook":
            wb_ws = ss_get("upload_workbook", {})
            current_source_code = "upload"
        else:
            wb_ws = ss_get("gs_workbook", {})
            current_source_code = "gs"

        if not wb_ws:
            st.warning("No sheets found in the selected source.")
        else:
            sheet_ws = st.selectbox("Sheet", list(wb_ws.keys()), key="ws_sheet_sel")
            df_ws = wb_ws.get(sheet_ws, pd.DataFrame())

            # Remember context
            ss_set("work_context", {"source": current_source_code, "sheet": sheet_ws})

            # Summary + preview
            if df_ws.empty or not validate_headers(df_ws):
                st.info("Selected sheet is empty or headers mismatch.")
            else:
                st.write(f"Found {len(wb_ws)} sheet(s). Choose one to process:")
                ok_p, total_p = compute_parent_depth_score_counts(df_ws)
                ok_r, total_r = compute_row_path_score_counts(df_ws)
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

            # Group rows controls (Node 1 / Node 2)
            st.markdown("---")
            with st.expander("🧩 Group rows (cluster identical labels together)"):
                st.caption("Group rows so identical **Node 1** and **Node 2** values are contiguous. Red Flags in Node 2 appear on top.")
                if df_ws.empty or not validate_headers(df_ws):
                    st.info("Load a valid sheet first.")
                else:
                    scope = st.radio("Grouping scope", ["Whole sheet", "Within Vital Measurement"], horizontal=True, key="ws_group_scope_sel")
                    preview = st.checkbox("Show preview (does not modify data)", value=True, key="ws_group_preview_sel")

                    def grouped_df(df0: pd.DataFrame, scope_mode: str) -> pd.DataFrame:
                        df2 = sort_sheet_for_view(df0, ss_get("symptom_quality", {}))
                        if scope_mode == "Within Vital Measurement":
                            # Stable sort within VM, preserving overall VM groups
                            df2["_vm"] = df2["Vital Measurement"].map(normalize_text)
                            df2["_row"] = np.arange(len(df2))
                            df2 = df2.sort_values(["_vm","Node 1","Node 2","Node 3","Node 4","Node 5","_row"], kind="stable").drop(columns=["_vm","_row"])
                        return df2

                    df_prev = grouped_df(df_ws, scope)
                    if preview:
                        st.dataframe(df_prev.head(100), use_container_width=True)
                    csvprev = df_prev.to_csv(index=False).encode("utf-8")
                    st.download_button("Download grouped view (CSV)", data=csvprev, file_name=f"{sheet_ws}_grouped.csv", mime="text/csv")

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
                        sid2 = ss_get("gs_spreadsheet_id","")
                        if current_source_code == "gs" and sid2 and st.button("Apply & push grouping to Google Sheets", key="ws_group_push_sel"):
                            ok = push_to_google_sheets(sid2, sheet_ws, df_prev)
                            if ok: st.success("Grouping pushed to Google Sheets.")

            # Push settings at bottom
            st.markdown("---")
            st.subheader("🔧 Google Sheets Push Settings (current view)")
            sid = st.text_input("Spreadsheet ID", value=ss_get("gs_spreadsheet_id",""), key="ws_push_sid_sel")
            if sid:
                ss_set("gs_spreadsheet_id", sid)
            default_tab = ss_get("saved_targets", {}).get(sheet_ws, {}).get("tab", f"{sheet_ws}")
            target_tab = st.text_input("Target tab", value=default_tab, key="ws_push_target_sel")
            include_scope = st.radio("Include scope (for stats only)", ["All completed parents","Only parents edited this session"], horizontal=True, key="ws_push_scope_sel")
            push_backup = st.checkbox("Create a backup tab before overwrite", value=True, key="ws_push_backup_sel")
            sort_rf_first = st.checkbox("Sort Red Flags first (Dictionary-wide lift across Node 1–5)", value=True, key="ws_push_rf_first_sel")
            dry_run = st.checkbox("Dry-run (build but don't write to Sheets)", value=False, key="ws_push_dry_sel")

            if st.button("📤 Push current view to Google Sheets", type="primary", key="ws_push_btn_sel"):
                if not sid or not target_tab:
                    st.error("Missing Spreadsheet ID or target tab.")
                elif df_ws.empty or not validate_headers(df_ws):
                    st.error("Current sheet is empty or headers mismatch.")
                else:
                    # Build push view: preserve VM grouping, group Node 1, optionally lift RF anywhere in Node1–5
                    redflag_map = ss_get("symptom_quality", {}) if sort_rf_first else None
                    view_df = order_rows_for_push(df_ws, redflag_map=redflag_map)

                    if dry_run:
                        st.success("Dry-run complete. No changes written to Google Sheets.")
                        st.dataframe(view_df.head(50), use_container_width=True)
                        buffer = io.BytesIO()
                        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
                            view_df.to_excel(writer, index=False, sheet_name=sheet_ws[:31] or "Sheet1")
                        st.download_button("Download current view workbook", data=buffer.getvalue(),
                                           file_name="decision_tree_current_view.xlsx",
                                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
                    else:
                        if push_backup:
                            bak = backup_sheet_copy(sid, target_tab)
                            if bak:
                                st.info(f"Backed up current '{target_tab}' to '{bak}'.")
                        ok = push_to_google_sheets(sid, target_tab, view_df)
                        if ok:
                            log = ss_get("push_log", [])
                            log.append({
                                "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                "sheet": sheet_ws,
                                "target_tab": target_tab,
                                "spreadsheet_id": sid,
                                "rows_written": len(view_df),
                                "new_rows_added": 0,
                                "scope": "session" if include_scope.endswith("session") else "all",
                            })
                            ss_set("push_log", log)
                            saved = ss_get("saved_targets", {})
                            saved.setdefault(sheet_ws, {})
                            saved[sheet_ws]["tab"] = target_tab
                            ss_set("saved_targets", saved)
                            st.success(f"Pushed {len(view_df)} rows to '{target_tab}'.")


# ==============================
# Tab: Symptoms (simplified inline editor with vocabulary)
# ==============================
with tabs[2]:
    st.header("🧬 Symptoms — inline edit of child sets")

    # Choose source/sheet
    sources = []
    if ss_get("upload_workbook", {}): sources.append("Upload workbook")
    if ss_get("gs_workbook", {}): sources.append("Google Sheets workbook")

    if not sources:
        st.info("Load a workbook first (Source tab).")
    else:
        src = st.radio("Choose data source", sources, horizontal=True, key="sym_source")
        if src == "Upload workbook":
            wb = ss_get("upload_workbook", {})
            where = "upload"
        else:
            wb = ss_get("gs_workbook", {})
            where = "gs"

        if not wb:
            st.info("No sheets found.")
        else:
            sheet = st.selectbox("Sheet", list(wb.keys()), key="sym_sheet")
            df = wb.get(sheet, pd.DataFrame())
            if df.empty or not validate_headers(df):
                st.info("Selected sheet is empty or headers mismatch.")
            else:
                overrides_all = ss_get("branch_overrides", {})
                overrides_sheet = overrides_all.get(sheet, {})
                store = infer_branch_options_with_overrides(df, overrides_sheet)

                # Build parents list by level
                def compute_virtual_parents(store: Dict[str, List[str]]) -> Dict[int, Set[Tuple[str, ...]]]:
                    parents_by_level: Dict[int, Set[Tuple[str, ...]]] = {i: set() for i in range(1, MAX_LEVELS+1)}
                    parents_by_level[1].add(tuple())  # <ROOT>
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
                level = st.selectbox("Level to inspect (child options of...)", [1,2,3,4,5], format_func=lambda x: f"Node {x}", key="sym_level")

                # Vocabulary from sheet (all node labels)
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

                # Search / status sort
                top_cols = st.columns([2,1,1,1,2])
                with top_cols[0]:
                    search = st.text_input("Search parent symptom/path", key="sym_search").strip().lower()
                with top_cols[1]:
                    compact = st.checkbox("Compact mode", value=True)
                with top_cols[2]:
                    sort_mode = st.selectbox("Sort by", ["Problem severity (issues first)", "Alphabetical (parent path)"], index=0)
                with top_cols[3]:
                    fill_other_default = st.checkbox("Fill blanks with 'Other' on save", value=False)
                with top_cols[4]:
                    enforce_unique_default = st.checkbox("Enforce uniqueness among the 5", value=True)

                # Build entries
                entries = []
                label_childsets: Dict[Tuple[int,str], set] = {}
                for parent_tuple in sorted(parents_by_level.get(level, set())):
                    parent_text = " > ".join(parent_tuple)
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

                status_rank = {"No group of symptoms":0, "Symptom left out":1, "Overspecified":2, "OK":3}
                if sort_mode.startswith("Problem"):
                    entries.sort(key=lambda e: (status_rank[e[2]], e[0]))
                else:
                    entries.sort(key=lambda e: e[0])

                # Render entries
                for parent_tuple, children, status in entries:
                    keyname = level_key_tuple(level, parent_tuple)
                    subtitle = f"{' > '.join(parent_tuple) or 'Top-level (Node 1) options'} — {status}"
                    with st.expander(subtitle):
                        selected_vals = []
                        if compact:
                            for i in range(5):
                                default_val = children[i] if i < len(children) else ""
                                txt_key = f"sym_txt_{level}_{'__'.join(parent_tuple)}_{i}"
                                sel_key = f"sym_sel_{level}_{'__'.join(parent_tuple)}_{i}"
                                txt = st.text_input(f"Child {i+1}", value=default_val, key=txt_key)
                                pick = st.selectbox("Pick suggestion", options=vocab_opts, index=0, key=sel_key)
                                selected_vals.append((txt, pick))
                        else:
                            cols = st.columns(5)
                            for i in range(5):
                                default_val = children[i] if i < len(children) else ""
                                with cols[i]:
                                    txt_key = f"sym_txt_{level}_{'__'.join(parent_tuple)}_{i}"
                                    sel_key = f"sym_sel_{level}_{'__'.join(parent_tuple)}_{i}"
                                    txt = st.text_input(f"Child {i+1}", value=default_val, key=txt_key)
                                    pick = st.selectbox("Suggestion", options=vocab_opts, index=0, key=sel_key)
                                selected_vals.append((txt, pick))

                        fill_other = st.checkbox("Fill remaining blanks with 'Other' on save", value=fill_other_default, key=f"sym_other_{level}_{'__'.join(parent_tuple)}")
                        enforce_unique = st.checkbox("Enforce uniqueness across the 5", value=enforce_unique_default, key=f"sym_unique_{level}_{'__'.join(parent_tuple)}")

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
                            overrides_all = ss_get("branch_overrides", {})
                            overrides_sheet = overrides_all.get(sheet, {}).copy()
                            # undo stack
                            stack = ss_get("undo_stack", [])
                            stack.append({
                                "context": "symptoms",
                                "sheet": sheet,
                                "level": level,
                                "parent": parent_tuple,
                                "overrides_sheet_before": overrides_all.get(sheet, {}).copy(),
                                "df_before": df.copy()
                            })
                            ss_set("undo_stack", stack)
                            overrides_sheet[keyname] = fixed
                            overrides_all[sheet] = overrides_sheet
                            ss_set("branch_overrides", overrides_all)
                            mark_session_edit(sheet, keyname)

                            # Update view df using overrides (no heavy cascade)
                            wb[sheet] = sort_sheet_for_view(df, ss_get("symptom_quality", {}))
                            if where == "upload":
                                ss_set("upload_workbook", wb)
                            else:
                                ss_set("gs_workbook", wb)
                            st.success("Saved and updated in-session view (grouping by Node 1 with RF on Node 2).")

            # Undo
            st.markdown("---")
            if st.button("↩️ Undo last branch edit (session)"):
                stack = ss_get("undo_stack", [])
                if not stack:
                    st.info("Nothing to undo.")
                else:
                    last = stack.pop()
                    ss_set("undo_stack", stack)
                    if last.get("context") == "symptoms":
                        overrides_all = ss_get("branch_overrides", {})
                        overrides_all[last["sheet"]] = last["overrides_sheet_before"]
                        ss_set("branch_overrides", overrides_all)
                        wb2 = ss_get("upload_workbook", {}) if where == "upload" else ss_get("gs_workbook", {})
                        if last.get("df_before") is not None and last["sheet"] in wb2:
                            wb2[last["sheet"]] = last["df_before"]
                            if where == "upload":
                                ss_set("upload_workbook", wb2)
                            else:
                                ss_set("gs_workbook", wb2)
                        st.success(f"Undid overrides for sheet '{last['sheet']}'.")


# ==============================
# Tab: Dictionary (search, highlight, red flag tagging)
# ==============================
with tabs[3]:
    st.header("📖 Dictionary — all symptom labels")

    # Pick sources
    sources_avail = []
    if ss_get("upload_workbook", {}): sources_avail.append("Upload workbook")
    if ss_get("gs_workbook", {}): sources_avail.append("Google Sheets workbook")

    if not sources_avail:
        st.info("Load data in the **Source** tab first (upload a workbook or load Google Sheets).")
    else:
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
        else:
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

            quality_map = ss_get("symptom_quality", {})  # {label: "Red Flag"|"Normal"}

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

            # Search & filter & sort
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

            def highlight_text(s: str, q: str) -> str:
                if not q: return s
                try:
                    import re
                    pattern = re.compile(re.escape(q), re.IGNORECASE)
                    return pattern.sub(lambda m: f"<mark>{m.group(0)}</mark>", s)
                except Exception:
                    return s.replace(q, f"<mark>{q}</mark>")

            if list_mode:
                st.markdown("### Results (highlighted)")
                page_size = st.selectbox("Items per page", [25, 50, 100, 200], index=1, key="dict_list_pagesize")
                total = len(view)
                import math
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
            else:
                st.markdown("### Edit Red Flags")
                page_size = st.selectbox("Rows per page", [25, 50, 100, 200], index=1, key="dict_table_pagesize")
                total = len(view)
                import math
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
                        dict_df = dict_df_updates.reset_index()
                        for _, r in dict_df.iterrows():
                            new_quality[r["Symptom"]] = "Red Flag" if bool(r["RedFlag"]) else "Normal"
                        ss_set("symptom_quality", new_quality)
                        st.success("Red Flags saved for this session.")
                with colS2:
                    csv_data = view.to_csv(index=False).encode("utf-8")
                    st.download_button("Download current view (CSV)", data=csv_data, file_name="dictionary_filtered.csv", mime="text/csv")


# ==============================
# Tab: Conflicts (simple inspector & resolver)
# ==============================
with tabs[4]:
    st.header("⚖️ Conflicts inspector")
    # Use current context sheet if available
    ctx = ss_get("work_context", {})
    src = ctx.get("source")
    sheet = ctx.get("sheet")

    if src == "upload":
        wb = ss_get("upload_workbook", {})
    else:
        wb = ss_get("gs_workbook", {})

    if not sheet or sheet not in wb:
        st.info("Select a sheet in Workspace first.")
    else:
        df = wb[sheet]
        if df.empty or not validate_headers(df):
            st.info("Sheet empty or invalid.")
        else:
            overrides_all = ss_get("branch_overrides", {})
            overrides_sheet = overrides_all.get(sheet, {})
            store = infer_branch_options_with_overrides(df, overrides_sheet)

            # Build conflicts by (level, parent_label)
            label_childsets: Dict[Tuple[int,str], List[Tuple[str,...]]] = {}
            for key, children in store.items():
                if "|" not in key: continue
                lvl_s, path = key.split("|", 1)
                try: L = int(lvl_s[1:])
                except: continue
                parent_tuple = tuple([] if path=="<ROOT>" else path.split(">"))
                parent_label = parent_tuple[-1] if parent_tuple else "<ROOT>"
                label_childsets.setdefault((L, parent_label), [])
                label_childsets[(L, parent_label)].append(tuple(sorted([c for c in children if normalize_text(c)!=""])))

            conflicts = {k:v for k,v in label_childsets.items() if len(set(v)) > 1}
            if not conflicts:
                st.success("No conflicts detected across identical parent labels.")
            else:
                st.warning(f"Found {len(conflicts)} parent label(s) with conflicting child sets.")
                for (L, plabel), sets in conflicts.items():
                    with st.expander(f"Node {L} parent '{plabel}' — {len(set(sets))} different child sets"):
                        # show each set
                        union_children = sorted({c for tup in sets for c in tup})
                        st.write("All observed child sets:")
                        for i, tup in enumerate(set(sets), start=1):
                            st.write(f"- Set {i}: {', '.join(tup)}")

                        st.markdown("**Resolve by choosing exactly 5 children from the union:**")
                        selected = st.multiselect(f"Pick 5 for '{plabel}' (Node {L})", union_children, max_selections=5, key=f"conf_pick_{L}_{plabel}")
                        if st.button("Save resolved child set", key=f"conf_save_{L}_{plabel}"):
                            if len(selected) != 5:
                                st.error("Please pick exactly 5 options.")
                            else:
                                # Save under the canonical key L|parent_tuple for each distinct parent tuple that shares this label
                                # We resolve *by label*, so apply to *all* parent tuples whose last label == plabel at this level.
                                new_overrides = overrides_all.get(sheet, {}).copy()
                                # find all parent tuples with last label == plabel
                                for key2 in store.keys():
                                    if "|" not in key2: continue
                                    lvl_s2, path2 = key2.split("|", 1)
                                    try: L2 = int(lvl_s2[1:])
                                    except: continue
                                    if L2 != L: continue
                                    ptuple = tuple([] if path2=="<ROOT>" else path2.split(">"))
                                    last_label = ptuple[-1] if ptuple else "<ROOT>"
                                    if last_label == plabel:
                                        new_overrides[level_key_tuple(L, ptuple)] = enforce_k_five(selected)
                                overrides_all[sheet] = new_overrides
                                ss_set("branch_overrides", overrides_all)
                                st.success("Resolved and saved for all matching parents.")


# ==============================
# Tab: Validation
# ==============================
with tabs[5]:
    st.header("🧪 Validation rules")

    # Use current context
    ctx = ss_get("work_context", {})
    src = ctx.get("source")
    sheet = ctx.get("sheet")

    if src == "upload":
        wb = ss_get("upload_workbook", {})
    else:
        wb = ss_get("gs_workbook", {})

    if not sheet or sheet not in wb:
        st.info("Select a sheet in Workspace first.")
    else:
        df = wb[sheet]
        if df.empty or not validate_headers(df):
            st.info("Sheet empty or invalid.")
        else:
            overrides_all = ss_get("branch_overrides", {})
            overrides_sheet = overrides_all.get(sheet, {})
            redflag_map = ss_get("symptom_quality", {})

            orphans_loose = detect_orphans(df, overrides=overrides_sheet, strict=False)
            orphans_strict = detect_orphans(df, overrides=overrides_sheet, strict=True)
            loops = detect_loops(df)
            missing_rf = detect_missing_redflag_coverage(df, overrides=overrides_sheet, redflag_map=redflag_map)

            colA, colB = st.columns(2)
            with colA:
                st.subheader("Orphans (loose)")
                if not orphans_loose:
                    st.success("None")
                else:
                    st.write(pd.DataFrame(orphans_loose))
            with colB:
                st.subheader("Orphans (strict)")
                if not orphans_strict:
                    st.success("None")
                else:
                    st.write(pd.DataFrame(orphans_strict))

            st.subheader("Loops")
            if not loops:
                st.success("None")
            else:
                # Flatten Arrow/CSV-unfriendly objects to strings
                def _fmt_path(x):
                    if isinstance(x, (list, tuple)):
                        return " > ".join(map(str, x))
                    return str(x)

                def _fmt_repeats(x):
                    # x ~ [(label, first_pos, later_pos), ...]
                    if isinstance(x, list):
                        try:
                            parts = []
                            for tup in x:
                                if isinstance(tup, (list, tuple)) and len(tup) == 3:
                                    label, i, j = tup
                                    parts.append(f"{label} ({i}->{j})")
                                else:
                                    parts.append(str(tup))
                            return "; ".join(parts)
                        except Exception:
                            return json.dumps(x, ensure_ascii=False)
                    return str(x)

                df_loops = pd.DataFrame(loops)
                if "path" in df_loops.columns:
                    df_loops["path"] = df_loops["path"].map(_fmt_path)
                if "repeats" in df_loops.columns:
                    df_loops["repeats"] = df_loops["repeats"].map(_fmt_repeats)
                for c in df_loops.columns:
                    df_loops[c] = df_loops[c].astype(str)

                st.dataframe(df_loops, use_container_width=True)

            st.subheader("Parents missing a Red Flag child")
            if not missing_rf:
                st.success("All parents have at least one Red Flag child.")
            else:
                st.write(pd.DataFrame(missing_rf))


# ==============================
# Tab: Calculator (placeholder)
# ==============================
with tabs[6]:
    st.header("🧮 Calculator (placeholder)")
    st.caption("A staging ground for v6.3.2 features — quick estimators, RF coverage %, etc.")
    n_rows = st.number_input("Rows", 0, 100000, 0, step=100)
    n_parents = st.number_input("Parents", 0, 100000, 0, step=100)
    if st.button("Compute simple ratio"):
        if n_parents == 0:
            st.warning("Parents must be > 0")
        else:
            st.success(f"Rows per parent: {n_rows / max(1, n_parents):.2f}")


# ==============================
# Tab: Visualizer
# ==============================
with tabs[7]:
    st.header("🌳 Visualizer")
    if not _HAS_PYVIS:
        st.info("PyVis is not installed. Add `pyvis>=0.3.2` to requirements to enable the visualizer.")
    else:
        # Resolve current sheet
        def _get_current_df() -> Tuple[Optional[pd.DataFrame], str]:
            ctx = ss_get("work_context", {})
            src = ctx.get("source")
            sheet = ctx.get("sheet")
            if src == "upload":
                wb = ss_get("upload_workbook", {})
                if sheet in wb: return wb[sheet], sheet
            elif src == "gs":
                wb = ss_get("gs_workbook", {})
                if sheet in wb: return wb[sheet], sheet
            wb_u = ss_get("upload_workbook", {})
            if wb_u:
                name = next(iter(wb_u))
                return wb_u[name], name
            wb_g = ss_get("gs_workbook", {})
            if wb_g:
                name = next(iter(wb_g))
                return wb_g[name], name
            return None, "(no sheet loaded)"

        dfv, sheet_name = _get_current_df()
        if dfv is None or dfv.empty or not validate_headers(dfv):
            st.info("No valid sheet found. Load data in **Source** and select a sheet in **Workspace**.")
        else:
            st.caption(f"Showing sheet: **{sheet_name}**")
            c1, c2, c3, c4 = st.columns([1,1,1,1])
            with c1:
                hierarchical = st.checkbox("Hierarchical layout", value=True, help="Top-to-bottom layered view.")
            with c2:
                collapse = st.checkbox("Merge same labels per level", value=True, help="If ON, identical labels at the same level are merged.")
            with c3:
                vms = sorted(set(dfv["Vital Measurement"].astype(str).map(normalize_text)))
                vm_scope = st.selectbox("Filter by Vital Measurement", options=["(All)"] + vms, index=0)
                vm_sel = None if vm_scope == "(All)" else vm_scope
            with c4:
                limit = st.number_input("Row limit", min_value=100, max_value=100000, value=5000, step=500)

            # Build graph data
            def _build_edges(df: pd.DataFrame, limit_rows: int, scope_vm: Optional[str], collapse_by_label_per_level: bool):
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

            def _apply_pyvis_options(net: Network, hierarchical: bool):
                if hierarchical:
                    options = {
                        "layout": {"hierarchical": {"enabled": True, "direction": "UD", "sortMethod": "directed"}},
                        "physics": {"enabled": False},
                        "nodes": {"shape": "dot", "size": 12},
                        "edges": {"arrows": {"to": {"enabled": True}}},
                    }
                else:
                    options = {
                        "physics": {"enabled": True, "stabilization": {"enabled": True}},
                        "nodes": {"shape": "dot", "size": 12},
                        "edges": {"arrows": {"to": {"enabled": True}}},
                    }
                net.set_options(json.dumps(options))

            nodes, edges, node_attrs = _build_edges(dfv, int(limit), vm_sel, collapse)
            if not nodes:
                st.info("No nodes to visualize with the current filters.")
            else:
                net = Network(height="650px", width="100%", directed=True, notebook=False)
                _apply_pyvis_options(net, hierarchical=hierarchical)

                for nid in nodes:
                    info = node_attrs.get(nid, {})
                    label = info.get("label", nid)
                    title = info.get("title", label)
                    try:
                        level_num = int(nid.split(":")[0][1:])
                    except Exception:
                        level_num = 0
                    color = [
                        "#4f46e5",  # L0
                        "#2563eb",  # L1
                        "#059669",  # L2
                        "#16a34a",  # L3
                        "#d97706",  # L4
                        "#dc2626",  # L5
                    ][min(level_num, 5)]
                    net.add_node(nid, label=label, title=title, color=color)
                for (src_id, dst_id) in edges:
                    net.add_edge(src_id, dst_id)

                try:
                    html = net.generate_html()
                    st.components.v1.html(html, height=680, scrolling=True)
                except Exception:
                    import tempfile, os
                    with tempfile.TemporaryDirectory() as tmpd:
                        out = os.path.join(tmpd, "graph.html")
                        net.write_html(out)
                        with open(out, "r", encoding="utf-8") as f:
                            st.components.v1.html(f.read(), height=680, scrolling=True)

                with st.expander("Legend & Tips", expanded=False):
                    st.markdown(
                        """
                        - **Colors by level** (VM → Node 1 → Node 2 → Node 3 → Node 4 → Node 5).
                        - Turn **Merge same labels per level** OFF to see every occurrence (more crowded).
                        - Use **Filter by Vital Measurement** to focus a single tree.
                        - Increase **Row limit** if your sheet is large and you need deeper coverage.
                        """
                    )


# ==============================
# Tab: Push Log
# ==============================
with tabs[8]:
    st.header("📜 Push Log")
    log = ss_get("push_log", [])
    if not log:
        st.info("No pushes recorded this session.")
    else:
        df_log = pd.DataFrame(log)
        st.dataframe(df_log, use_container_width=True)
        csv = df_log.to_csv(index=False).encode("utf-8")
        st.download_button("Download push log (CSV)", data=csv, file_name="push_log.csv", mime="text/csv")


# ==============================
# Footer
# ==============================
st.markdown(f"---\nApp Version: **{APP_VERSION}**")