# streamlit_app_upload.py — Version 6.2.6

import io
import json
import random
from typing import Dict, List, Tuple, Optional, Set
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime

# ============ VERSION / CONFIG ============
APP_VERSION = "v6.2.6"
CANON_HEADERS = ["Vital Measurement","Node 1","Node 2","Node 3","Node 4","Node 5","Diagnostic Triage","Actions"]
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
        v = normalize_text(row[c])
        if v == "":
            return None
        parent.append(v)
    return tuple(parent)

def level_key_tuple(level: int, parent: Tuple[str, ...]) -> str:
    return f"L{level}|" + (">".join(parent) if parent else "<ROOT>")

def infer_branch_options(df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Build a store of explicit parent->children from the dataframe.
    Keys are level-specific:  f"L{level}|{'>'.join(parent_tuple) or '<ROOT>'}"
    """
    store: Dict[str, List[str]] = {}
    for level in range(1, MAX_LEVELS+1):
        parent_to_children: Dict[Tuple[str, ...], List[str]] = {}
        for _, row in df.iterrows():
            child_col = LEVEL_COLS[level-1]
            if child_col not in df.columns:
                continue
            child = normalize_text(row[child_col])
            if child == "":
                continue
            parent = parent_key_from_row_strict(row, level)
            if parent is None:
                continue
            parent_to_children.setdefault(parent, [])
            parent_to_children[parent].append(child)
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

def enforce_k_five(opts: List[str]) -> List[str]:
    clean = [normalize_text(o) for o in opts if normalize_text(o) != ""]
    if len(clean) > 5:
        clean = clean[:5]
    elif len(clean) < 5:
        clean = clean + [""]*(5-len(clean))
    return clean

# ===== Utilities for keying rows to detect duplicates & depth =====
def _row_key_from_like(rowlike) -> Tuple[str, ...]:
    return tuple(normalize_text(rowlike.get(c, "")) for c in ["Vital Measurement"] + LEVEL_COLS)

def make_keyset(df: pd.DataFrame) -> Set[Tuple[str, ...]]:
    ks = set()
    for _, r in df.iterrows():
        ks.add(_row_key_from_like(r))
    return ks

def compute_branch_depth(df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    nodes = df[LEVEL_COLS].applymap(normalize_text)
    return int(nodes.apply(lambda r: sum(1 for v in r if v != ""), axis=1).max())

# ============ Google Sheets helpers (RESIZE BEFORE UPDATE) ============
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

# ============ Global label→children map (for recursive cascade) ============
def build_label_child_map(store: Dict[str, List[str]]) -> Dict[str, List[str]]:
    """
    Build a global mapping of label -> children across all contexts.
    For each parent tuple (at any level), we take its last label and collect its children.
    Excludes ROOT.
    """
    label_to_children: Dict[str, List[str]] = {}
    for key, children in store.items():
        # key like "L2|parent1>parent2" OR "L1|<ROOT>"
        try:
            _lvl_s, path = key.split("|", 1)
        except ValueError:
            continue
        if path == "<ROOT>":
            continue
        parent_tuple = tuple(path.split(">"))
        parent_label = parent_tuple[-1] if parent_tuple else None
        if not parent_label:
            continue
        cur = label_to_children.get(parent_label, [])
        # keep order & dedupe
        seen = set(cur)
        for c in children:
            c = normalize_text(c)
            if c and c not in seen:
                cur.append(c); seen.add(c)
        label_to_children[parent_label] = cur[:5]  # keep max 5 consistently
    return label_to_children

def union_children_for_parent(parent: Tuple[str, ...],
                              level: int,
                              store: Dict[str, List[str]],
                              label_map: Dict[str, List[str]]) -> List[str]:
    """Union of:
      - store-defined children for this (level, parent)
      - global label_map for the last label of this parent (if parent non-root)
    """
    opts_store = store.get(level_key_tuple(level, parent), []) if level <= MAX_LEVELS else []
    opts_label = label_map.get(parent[-1], []) if parent else []
    # Keep order: store first, then label additions
    merged, seen = [], set()
    for src in (opts_store, opts_label):
        for x in src:
            x = normalize_text(x)
            if x and x not in seen:
                merged.append(x); seen.add(x)
    return enforce_k_five(merged)

# ============ Expansion primitives (anchor-reuse, deep cascade, recursive) ============
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
            # lenient anchor: accept if node L is blank (deeper nodes can be anything)
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

def expand_parent_nextnode_anchor_reuse_for_vm(
    df: pd.DataFrame,
    store: Dict[str, List[str]],
    label_map: Dict[str, List[str]],
    vm: str,
    parent: Tuple[str,...]
) -> Tuple[pd.DataFrame, Dict[str,int], List[Tuple[str,...]]]:
    """
    Ensure next-node children exist under 'parent' for a single VM, using anchor-reuse:
    - compute union children from (store for this parent) U (label_map for last label)
    - reuse one anchor row (Node L blank) to fill the first missing option
    - create rows for the remaining missing options
    Returns updated df, stats {'new_rows','inplace_filled'}, and list of child parent-tuples we created/confirmed.
    """
    stats = {"new_rows": 0, "inplace_filled": 0}
    L = len(parent) + 1
    if L > MAX_LEVELS:
        return df, stats, []

    # union children
    children = union_children_for_parent(parent, L, store, label_map)
    children = [c for c in children if c]  # remove blanks

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

    # Also include already-present children as confirmed
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
    label_map: Dict[str, List[str]],
    vm_scope: List[str],
    start_parents: List[Tuple[str,...]],
) -> Tuple[pd.DataFrame, Dict[str,int]]:
    """
    Deep cascade to Node 5 using anchor-reuse at EACH level.
    - Completes partial parents (adds rows for any defined missing options).
    - Proceeds deeper as long as child options are defined in the store OR label_map for the child's label.
    - Recursive effect: label_map ensures entire subtrees propagate wherever a label appears.
    """
    total = {"new_rows":0, "inplace_filled":0}
    stack: List[Tuple[str,...]] = list(start_parents)

    while stack:
        parent = stack.pop(0)
        L = len(parent)+1
        if L > MAX_LEVELS:
            continue

        # union to decide whether to proceed at this level
        children_defined = union_children_for_parent(parent, L, store, label_map)
        if not [x for x in children_defined if x]:
            continue

        next_child_parents_all_vms: Set[Tuple[str,...]] = set()

        for vm in vm_scope:
            df, stats, child_parents = expand_parent_nextnode_anchor_reuse_for_vm(df, store, label_map, vm, parent)
            total["new_rows"] += stats["new_rows"]
            total["inplace_filled"] += stats["inplace_filled"]
            for cp in child_parents:
                next_child_parents_all_vms.add(cp)

        # Push deeper for each child that has next-level options defined via store or label_map
        for cp in sorted(next_child_parents_all_vms):
            next_level = len(cp) + 1
            if next_level <= MAX_LEVELS:
                # If store has next-level options OR the child's label has known children, keep cascading
                has_next_store = bool([x for x in store.get(level_key_tuple(next_level, cp), []) if normalize_text(x)!=""])
                has_next_label = bool(label_map.get(cp[-1], []))
                if has_next_store or has_next_label:
                    stack.append(cp)

    return df, total

# ============ Raw+ builder (v6.2.6) ============
def build_raw_plus_v626(
    df: pd.DataFrame,
    overrides: Dict[str, List[str]],
    include_scope: str,
    edited_keys_for_sheet: Set[str],
) -> Tuple[pd.DataFrame, Dict[str,int], pd.DataFrame]:
    """
    Always deep-cascade with anchor-reuse for selected scope.
    Returns (df_aug, stats, duplicates_df_preview)
    """
    store = infer_branch_options_with_overrides(df, overrides)
    label_map = build_label_child_map(store)
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
            except:
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
            except:
                continue
            parent = tuple([] if path == "<ROOT>" else path.split(">"))
            parent_keys.append((L, parent))

    start_parents = sorted({p for (_, p) in parent_keys})
    df_before = len(df_aug)
    df_aug, stx = cascade_anchor_reuse_full(df_aug, store, label_map, vms, start_parents)
    stats_total["inplace_filled"] += stx["inplace_filled"]
    stats_total["generated"] += (len(df_aug) - df_before) + stx["inplace_filled"]

    # Compute new_added / duplicates_skipped against original df
    original_keys = make_keyset(df)
    now_keys = make_keyset(df_aug)
    stats_total["new_added"] = len(now_keys - original_keys)
    stats_total["duplicates_skipped"] = max(0, stats_total["generated"] - stats_total["new_added"])
    stats_total["final_total"] = len(df_aug)

    # Build duplicates preview DF
    dup_mask = []
    for _, r in df_aug.iterrows():
        dup_mask.append(_row_key_from_like(r) in original_keys)
    duplicates_df = df_aug[pd.Series(dup_mask, index=df_aug.index)].copy()

    return df_aug, stats_total, duplicates_df

# ======== Progress / depth metrics and badge ========
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
    nodes = df[LEVEL_COLS].applymap(normalize_text)
    full = nodes.ne("").all(axis=1)
    return int(full.sum()), int(len(df))

def progress_badge_html(df: pd.DataFrame) -> str:
    ok_p, total_p = compute_parent_depth_score(df)
    ok_r, total_r = compute_row_path_score(df)
    depth = compute_branch_depth(df)
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
      .depth { font-weight:600; color:#0e7490; }
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
      <div class='badge-row'>
        <span class='muted'>Branch depth</span> <span class='depth'>{depth}/{MAX_LEVELS}</span>
      </div>
    </div>
    """
    return html

# ======== PDF export (ReportLab) ========
def build_symptoms_pdf(store: Dict[str, List[str]], parents: List[Tuple[str,...]], level: int, sheet_name: str) -> bytes:
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    from reportlab.lib.units import cm

    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4
    margin = 1.7 * cm
    x = margin
    y = height - margin

    title = f"Symptoms & Branches — Sheet: {sheet_name} — Node {level}"
    c.setFont("Helvetica-Bold", 14)
    c.drawString(x, y, title)
    y -= 1.0*cm

    c.setFont("Helvetica", 10)
    for parent in parents:
        key = level_key_tuple(level, parent)
        children = store.get(key, [])
        parent_text = " > ".join(parent) if parent else "<ROOT>"
        c.setFont("Helvetica-Bold", 10)
        c.drawString(x, y, f"{parent_text}")
        y -= 0.5*cm
        c.setFont("Helvetica", 10)
        if not children:
            c.drawString(x+0.5*cm, y, "- (no children)")
            y -= 0.4*cm
        else:
            for idx, ch in enumerate(children[:5], start=1):
                c.drawString(x+0.5*cm, y, f"{idx}. {ch}")
                y -= 0.4*cm
        y -= 0.3*cm
        if y < margin + 2*cm:
            c.showPage()
            y = height - margin
            c.setFont("Helvetica-Bold", 14)
            c.drawString(x, y, title)
            y -= 1.0*cm
            c.setFont("Helvetica", 10)

    c.showPage()
    c.save()
    return buffer.getvalue()

# ============ HEADER ============
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

with st.sidebar:
    st.header("❓ Tips")
    st.markdown("""
- **Recursive deep cascade:** Uses a global label→children map to propagate entire subtrees; anchor-reuse prevents duplicates.
- **VM Builder & Wizard:** Add a Vital Measurement and its branches; auto-cascade forward.
- **Dictionary:** Filter/search + toggle **Red Flag** per symptom; export CSV.
- Use **Dry-run** to preview; keep **Backup** on for easy rollback.
""")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "⬆️ Upload Excel/CSV",
    "📄 Google Sheets Mode",
    "🧪 Fill Diagnostic & Actions",
    "🧬 Symptoms",
    "📜 Push Log",
    "📚 Dictionary"
])

# ---------- Upload mode ----------
with tab1:
    st.subheader("Upload your workbook")
    file = st.file_uploader("Upload XLSX or CSV", type=["xlsx","xls","csv"])

    st.markdown("**Create a new sheet (in-session)**")
    new_sheet_name = st.text_input("New sheet name (upload workbook)", key="new_upload_sheet")
    if st.button("Create sheet (upload workbook)"):
        wb = ss_get("upload_workbook", {})
        if not new_sheet_name:
            st.warning("Please enter a sheet name.")
        elif new_sheet_name in wb:
            st.warning("A sheet with that name already exists in the current session workbook.")
        else:
            empty = pd.DataFrame(columns=CANON_HEADERS)
            wb[new_sheet_name] = empty
            ss_set("upload_workbook", wb)
            st.success(f"Created new sheet '{new_sheet_name}' in-session. Remember to Download workbook to save it locally.")

    if file is not None:
        if file.name.lower().endswith(".csv"):
            df = pd.read_csv(file)
            for c in CANON_HEADERS:
                if c not in df.columns:
                    df[c] = ""
            df = df[CANON_HEADERS]
            sheets = {"Sheet1": df}
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
                sheets[name] = dfx
        ss_set("upload_workbook", {k: v.copy() for k, v in sheets.items()})
        ss_set("upload_filename", file.name)

    wb = ss_get("upload_workbook", {})
    if not wb:
        st.info("Upload a workbook to continue.")
    else:
        st.write(f"Found {len(wb)} sheet(s). Choose one to process:")
        sheet_name = st.selectbox("Sheet", list(wb.keys()))
        df_in = wb[sheet_name].copy()

        if not validate_headers(df_in):
            st.error("Headers mismatch. First 8 columns must be: " + ", ".join(CANON_HEADERS)); st.stop()

        # Progress summary
        ok_p, total_p = compute_parent_depth_score(df_in)
        ok_r, total_r = compute_row_path_score(df_in)
        p1, p2 = st.columns(2)
        with p1:
            st.metric("Parents with 5 children", f"{ok_p}/{total_p}")
            st.progress(0 if total_p==0 else ok_p/total_p)
        with p2:
            st.metric("Rows with full path", f"{ok_r}/{total_r}")
            st.progress(0 if total_r==0 else ok_r/total_r)

        # ===== Preview control: choose which 50 rows =====
        total_rows = len(df_in)
        st.markdown("#### Preview (50 rows)")
        if total_rows <= 50:
            st.caption(f"Showing all {total_rows} rows.")
            st.dataframe(df_in, use_container_width=True)
        else:
            # Persisted start index with Next/Previous buttons
            state_key = f"preview_start_{sheet_name}"
            start_idx = int(ss_get(state_key, 0))
            # Controls
            cprev, cnum, cnext = st.columns([1,2,1])
            with cprev:
                if st.button("◀ Previous 50"):
                    start_idx = max(0, start_idx - 50)
            with cnum:
                start_1based = st.number_input(
                    "Start row (1-based)",
                    min_value=1,
                    max_value=max(1, total_rows-49),
                    value=start_idx+1,
                    step=50,
                    help="Pick where to start the 50-row preview."
                )
                start_idx = int(start_1based) - 1
            with cnext:
                if st.button("Next 50 ▶"):
                    start_idx = min(max(0, total_rows - 50), start_idx + 50)
            ss_set(state_key, start_idx)
            end_idx = min(start_idx + 50, total_rows)
            st.caption(f"Showing rows **{start_idx+1}–{end_idx}** of **{total_rows}**.")
            st.dataframe(df_in.iloc[start_idx:end_idx], use_container_width=True)

        # ===== VM builder (quick) =====
        with st.expander("🧩 VM Builder (create Vital Measurements and auto-cascade)"):
            vm_mode = st.radio("Mode", ["Create VM with 5 Node-1 options","Create VM with one Node-1 and its 5 Node-2 options"], horizontal=False)
            overrides_upload_all = ss_get("branch_overrides_upload", {})
            overrides_upload = overrides_upload_all.get(sheet_name, {}).copy()

            if vm_mode == "Create VM with 5 Node-1 options":
                vm_name = st.text_input("Vital Measurement name", key="vm_new_name")
                cols = st.columns(5)
                vals_n1 = [cols[i].text_input(f"Node-1 option {i+1}", key=f"vm_n1_{i}") for i in range(5)]
                if st.button("Create VM and Auto-cascade", key="vm_create_n1"):
                    if not vm_name.strip():
                        st.error("Enter a Vital Measurement name.")
                    else:
                        # Ensure at least one anchor row exists for this VM
                        if df_in[df_in["Vital Measurement"].map(normalize_text) == vm_name].empty:
                            anchor = {"Vital Measurement": vm_name}
                            for c in LEVEL_COLS: anchor[c] = ""
                            anchor["Diagnostic Triage"] = ""; anchor["Actions"] = ""
                            df_in = pd.concat([df_in, pd.DataFrame([anchor], columns=CANON_HEADERS)], ignore_index=True)

                        # Save L1|<ROOT>
                        k1 = level_key_tuple(1, tuple())
                        overrides_upload[k1] = enforce_k_five(vals_n1)

                        overrides_upload_all[sheet_name] = overrides_upload
                        ss_set("branch_overrides_upload", overrides_upload_all)
                        mark_session_edit(sheet_name, k1)

                        # Auto-cascade from <ROOT> limited to this VM
                        store = infer_branch_options_with_overrides(df_in, overrides_upload)
                        label_map = build_label_child_map(store)
                        df_new, tstats = cascade_anchor_reuse_full(df_in, store, label_map, [vm_name], [tuple()])
                        df_in = df_new
                        wb[sheet_name] = df_in; ss_set("upload_workbook", wb)
                        st.success(f"VM '{vm_name}' created. Auto-cascade added {tstats['new_rows']} rows, filled {tstats['inplace_filled']} anchors.")

            else:
                vm_name = st.text_input("Vital Measurement name", key="vm2_new_name")
                n1 = st.text_input("Node-1 value", key="vm2_n1")
                cols = st.columns(5)
                vals_n2 = [cols[i].text_input(f"Node-2 option {i+1}", key=f"vm2_n2_{i}") for i in range(5)]
                if st.button("Create VM + Node-1 + Node-2 and Auto-cascade", key="vm_create_n1n2"):
                    if not vm_name.strip() or not n1.strip():
                        st.error("Enter a Vital Measurement and Node-1.")
                    else:
                        # Ensure anchor row for VM exists
                        if df_in[df_in["Vital Measurement"].map(normalize_text) == vm_name].empty:
                            anchor = {"Vital Measurement": vm_name}
                            for c in LEVEL_COLS: anchor[c] = ""
                            anchor["Diagnostic Triage"] = ""; anchor["Actions"] = ""
                            df_in = pd.concat([df_in, pd.DataFrame([anchor], columns=CANON_HEADERS)], ignore_index=True)

                        # L1|<ROOT>
                        k1 = level_key_tuple(1, tuple())
                        existing = [x for x in overrides_upload.get(k1, []) if normalize_text(x)!=""]
                        if n1 and n1 not in existing:
                            existing.append(n1)
                        overrides_upload[k1] = enforce_k_five(existing)
                        # L2|<n1>
                        k2 = level_key_tuple(2, (n1,))
                        overrides_upload[k2] = enforce_k_five(vals_n2)

                        overrides_upload_all[sheet_name] = overrides_upload
                        ss_set("branch_overrides_upload", overrides_upload_all)
                        mark_session_edit(sheet_name, k1); mark_session_edit(sheet_name, k2)

                        # Auto-cascade from parent=(n1) limited to this VM
                        store = infer_branch_options_with_overrides(df_in, overrides_upload)
                        label_map = build_label_child_map(store)
                        df_new, tstats = cascade_anchor_reuse_full(df_in, store, label_map, [vm_name], [(n1,)])
                        df_in = df_new
                        wb[sheet_name] = df_in; ss_set("upload_workbook", wb)
                        st.success(f"VM '{vm_name}' with Node-1 '{n1}' created. Auto-cascade added {tstats['new_rows']} rows, filled {tstats['inplace_filled']} anchors.")

        # ===== 🧙 VM Build Wizard (end-to-end) =====
        with st.expander("🧙 VM Build Wizard (step-by-step to Node 5)"):
            wizard = ss_get("vm_wizard", {"vm":"","queue":[()], "overrides":{}, "active": False})
            vm_input = st.text_input("VM name (applies cascade/rows for this VM only)", value=wizard.get("vm",""))
            if not wizard["active"]:
                if st.button("Start Wizard"):
                    wizard = {"vm": vm_input.strip(), "queue":[()], "overrides":{}, "active": True}
                    ss_set("vm_wizard", wizard)
                    st.rerun()
            else:
                if vm_input.strip() != wizard["vm"]:
                    wizard["vm"] = vm_input.strip()
                    ss_set("vm_wizard", wizard)

                if not wizard["queue"]:
                    st.success("Wizard queue is empty. You can Finish & Apply or Reset.")
                else:
                    current_parent = wizard["queue"][0]
                    L = len(current_parent)+1
                    st.write(f"**Current parent path:** `{ ' > '.join(current_parent) or '<ROOT>' }` — define children for **Node {L}**")
                    cols = st.columns(5)
                    vals = [cols[i].text_input(f"Child {i+1}", key=f"wiz_{L}_{'__'.join(current_parent)}_{i}") for i in range(5)]
                    fill_other = st.checkbox("Fill remaining blanks with 'Other'", key=f"wiz_other_{L}_{'__'.join(current_parent)}")
                    if fill_other:
                        vals = [v if v.strip() else "Other" for v in vals]
                    vals = enforce_k_five(vals)

                    c1, c2, c3 = st.columns(3)
                    with c1:
                        if st.button("Save children & Next"):
                            key = level_key_tuple(L, current_parent)
                            wizard["overrides"][key] = vals
                            # extend queue with next parents
                            if L < MAX_LEVELS:
                                for v in [normalize_text(x) for x in vals if normalize_text(x)!=""]:
                                    wizard["queue"].append(current_parent + (v,))
                            # pop current
                            wizard["queue"].pop(0)
                            ss_set("vm_wizard", wizard)
                            st.rerun()
                    with c2:
                        if st.button("Skip this parent"):
                            wizard["queue"].pop(0)
                            ss_set("vm_wizard", wizard)
                            st.rerun()
                    with c3:
                        if st.button("Reset Wizard"):
                            ss_set("vm_wizard", {"vm":"","queue":[()], "overrides":{}, "active": False})
                            st.experimental_rerun()

                st.markdown("---")
                colA, colB = st.columns(2)
                with colA:
                    if st.button("Finish & Apply to this sheet"):
                        # merge into sheet overrides and cascade for this VM
                        overrides_upload_all = ss_get("branch_overrides_upload", {})
                        current_sheet_overrides = overrides_upload_all.get(sheet_name, {}).copy()
                        for k,v in wizard["overrides"].items():
                            current_sheet_overrides[k] = enforce_k_five(v)
                            mark_session_edit(sheet_name, k)
                        overrides_upload_all[sheet_name] = current_sheet_overrides
                        ss_set("branch_overrides_upload", overrides_upload_all)

                        # ensure anchor row for VM
                        vmn = wizard.get("vm","").strip() or "New VM"
                        if df_in[df_in["Vital Measurement"].map(normalize_text) == vmn].empty:
                            anchor = {"Vital Measurement": vmn}
                            for c in LEVEL_COLS: anchor[c] = ""
                            anchor["Diagnostic Triage"] = ""; anchor["Actions"] = ""
                            df_in = pd.concat([df_in, pd.DataFrame([anchor], columns=CANON_HEADERS)], ignore_index=True)

                        # cascade from ROOT for this VM
                        store = infer_branch_options_with_overrides(df_in, current_sheet_overrides)
                        label_map = build_label_child_map(store)
                        df_new, tstats = cascade_anchor_reuse_full(df_in, store, label_map, [vmn], [tuple()])
                        df_in = df_new
                        wb[sheet_name] = df_in; ss_set("upload_workbook", wb)

                        st.success(f"Wizard applied. Auto-cascade: +{tstats['new_rows']} rows, {tstats['inplace_filled']} anchors filled.")
                        # reset wizard
                        ss_set("vm_wizard", {"vm":"","queue":[()], "overrides":{}, "active": False})
                with colB:
                    if st.button("Cancel & Reset Wizard"):
                        ss_set("vm_wizard", {"vm":"","queue":[()], "overrides":{}, "active": False})
                        st.experimental_rerun()

        # Overrides for this sheet (Upload)
        overrides_upload_all = ss_get("branch_overrides_upload", {})
        overrides_upload = overrides_upload_all.get(sheet_name, {})
        store = infer_branch_options_with_overrides(df_in, overrides_upload)

        # Diagnostics of store
        def compute_virtual_parents(store: Dict[str, List[str]]) -> Dict[int, Set[Tuple[str, ...]]]:
            parents_by_level: Dict[int, Set[Tuple[str, ...]]] = {i: set() for i in range(1, MAX_LEVELS+1)}
            parents_by_level[1].add(tuple())  # <ROOT>
            for level in range(1, MAX_LEVELS):
                for p in list(parents_by_level[level]):
                    key = level_key_tuple(level, p)
                    children = [x for x in store.get(key, []) if normalize_text(x)!=""]
                    for c in children:
                        if c != "":
                            parents_by_level[level+1].add(p + (c,))
            # include explicit
            for key in store.keys():
                if "|" not in key: continue
                lvl_s, path = key.split("|", 1)
                try:
                    lvl = int(lvl_s[1:])
                except:
                    continue
                parent_tuple = tuple([] if path == "<ROOT>" else path.split(">"))
                if 1 <= lvl <= MAX_LEVELS:
                    parents_by_level[lvl].add(parent_tuple)
                    for k in range(1, min(lvl, MAX_LEVELS)+1):
                        parents_by_level.setdefault(k, set())
                        parents_by_level[k].add(tuple(parent_tuple[:k-1]))
            return parents_by_level

        parents_by_level = compute_virtual_parents(store)

        # Prepare issues
        missing, overspec, incomplete = [], [], []
        for level in range(1, MAX_LEVELS+1):
            for parent in sorted(parents_by_level.get(level, set())):
                key = level_key_tuple(level, parent); opts = store.get(key, [])
                non_empty = [x for x in opts if normalize_text(x)!=""]
                if len(non_empty)==0: missing.append((level,parent,key))
                elif len(non_empty)<5: incomplete.append((level,parent,key,opts))
                elif len(non_empty)>5: overspec.append((level,parent,key,opts))

        st.markdown("#### Review & Complete")
        # Undo (Upload)
        if st.button("↩️ Undo last edit (Upload tab)"):
            stack = ss_get("undo_stack", [])
            if not stack:
                st.info("Nothing to undo.")
            else:
                last = stack.pop(); ss_set("undo_stack", stack)
                if last.get("context") == "upload" and last.get("sheet") == sheet_name:
                    bo_all = ss_get("branch_overrides_upload", {})
                    bo_all[sheet_name] = last["overrides_sheet_before"]
                    ss_set("branch_overrides_upload", bo_all)
                    if last.get("df_before") is not None:
                        wb[sheet_name] = last["df_before"]
                        ss_set("upload_workbook", wb)
                    st.success("Restored overrides and sheet to previous state for this sheet.")
                else:
                    st.info("Last undo snapshot was for a different tab/sheet.")

        # Push settings (Raw+ for this sheet)
        st.info("Sheets Push (Raw+) — set your target below:")
        cols_push = st.columns([2,2,2,1,1])
        with cols_push[0]:
            up_sid_inline = st.text_input("Spreadsheet ID (Upload)", value=ss_get("gs_spreadsheet_id",""), key="up_sid_inline")
            if up_sid_inline: ss_set("gs_spreadsheet_id", up_sid_inline)
        with cols_push[1]:
            default_tab = ss_get("saved_targets", {}).get(sheet_name, {}).get("tab", f"{sheet_name}")
            up_target_inline = st.text_input("Target tab", value=default_tab, key="up_target_inline")
        with cols_push[2]:
            include_scope = st.radio("Include scope", ["All completed parents","Only parents edited this session"], horizontal=True, key="up_scope")
        with cols_push[3]:
            up_backup_inline = st.checkbox("Backup", value=True, key="up_backup_inline")
        with cols_push[4]:
            dry_run_inline = st.checkbox("Dry-run", value=False, key="up_dry_run_inline")

        if up_target_inline:
            saved = ss_get("saved_targets", {})
            saved.setdefault(sheet_name, {})
            saved[sheet_name]["tab"] = up_target_inline
            ss_set("saved_targets", saved)

        user_fixes: Dict[str, List[str]] = {}

        # --- No group of symptoms (0 options) ---
        if missing:
            st.warning(f"No group of symptoms: {len(missing)}")
            for (level, parent, key) in missing:
                with st.expander(f"{' > '.join(parent) or '<ROOT>'} — Node {level} — No group of symptoms"):
                    cols = st.columns(5)
                    edit_keys = [f"miss_{key}_{i}" for i in range(5)]
                    edit = [
                        cols[i].text_input(
                            f"Option {i+1}",
                            value=st.session_state.get(edit_keys[i], ""),
                            placeholder="Enter option",
                            key=edit_keys[i],
                        ) for i in range(5)
                    ]
                    user_fixes[key] = [normalize_text(x) for x in edit]

                    if st.button("Save 5 branches", key=f"up_save_{key}"):
                        stack = ss_get("undo_stack", [])
                        stack.append({
                            "context": "upload",
                            "label": f"Save 5 (missing) L{level}",
                            "sheet": sheet_name,
                            "overrides_sheet_before": overrides_upload.copy(),
                            "df_before": df_in.copy()
                        })
                        ss_set("undo_stack", stack)
                        overrides_upload[key] = enforce_k_five(edit)
                        overrides_upload_all[sheet_name] = overrides_upload
                        ss_set("branch_overrides_upload", overrides_upload_all)
                        mark_session_edit(sheet_name, key)
                        store2 = infer_branch_options_with_overrides(df_in, overrides_upload)
                        label_map2 = build_label_child_map(store2)
                        df_new, tstats = cascade_anchor_reuse_full(df_in, store2, label_map2, sorted(set(df_in["Vital Measurement"].map(normalize_text))), [parent])
                        df_in = df_new
                        wb[sheet_name] = df_in; ss_set("upload_workbook", wb)
                        st.success(f"Saved 5 and auto-cascaded: added {tstats['new_rows']} rows, filled {tstats['inplace_filled']} anchors.")

        # --- Symptom left out (<5 options) ---
        if incomplete:
            st.warning(f"Symptom left out (<5 options): {len(incomplete)}")
            for (level,parent,key,opts) in incomplete:
                count_non_empty = len([x for x in opts if normalize_text(x)!=""])
                with st.expander(f"{' > '.join(parent) or '<ROOT>'} — Node {level} — {count_non_empty} / 5"):
                    padded = enforce_k_five(opts)
                    cols = st.columns(5)
                    edit_keys = [f"incomp_{key}_{i}" for i in range(5)]
                    edit = [
                        cols[i].text_input(
                            f"Option {i+1}",
                            value=st.session_state.get(edit_keys[i], padded[i]),
                            placeholder="Enter option",
                            key=edit_keys[i],
                        ) for i in range(5)
                    ]
                    user_fixes[key] = [normalize_text(x) for x in edit]

                    if st.button("Save 5 branches", key=f"up_save_{key}"):
                        stack = ss_get("undo_stack", [])
                        stack.append({
                            "context": "upload",
                            "label": f"Save 5 (incomplete) L{level}",
                            "sheet": sheet_name,
                            "overrides_sheet_before": overrides_upload.copy(),
                            "df_before": df_in.copy()
                        })
                        ss_set("undo_stack", stack)
                        overrides_upload[key] = enforce_k_five(edit)
                        overrides_upload_all[sheet_name] = overrides_upload
                        ss_set("branch_overrides_upload", overrides_upload_all)
                        mark_session_edit(sheet_name, key)
                        store2 = infer_branch_options_with_overrides(df_in, overrides_upload)
                        label_map2 = build_label_child_map(store2)
                        df_new, tstats = cascade_anchor_reuse_full(df_in, store2, label_map2, sorted(set(df_in["Vital Measurement"].map(normalize_text))), [parent])
                        df_in = df_new
                        wb[sheet_name] = df_in; ss_set("upload_workbook", wb)
                        st.success(f"Saved 5 and auto-cascaded: added {tstats['new_rows']} rows, filled {tstats['inplace_filled']} anchors.")

        if overspec:
            st.error(f"Overspecified branches (>5 options): {len(overspec)} — choose exactly 5")
            for (level,parent,key,opts) in overspec:
                with st.expander(f"{' > '.join(parent) or '<ROOT>'} — Node {level} — Overspecified"):
                    chosen = st.multiselect("Select 5 options", opts, default=opts[:5], key=f"over_{key}")
                    fix = chosen[:5] if len(chosen) >= 5 else enforce_k_five(chosen)
                    user_fixes[key] = [normalize_text(x) for x in fix]
                    if st.button("Save 5 branches", key=f"up_save_over_{key}"):
                        stack = ss_get("undo_stack", [])
                        stack.append({
                            "context": "upload",
                            "label": f"Save 5 (overspec) L{level}",
                            "sheet": sheet_name,
                            "overrides_sheet_before": overrides_upload.copy(),
                            "df_before": df_in.copy()
                        })
                        ss_set("undo_stack", stack)
                        overrides_upload[key] = enforce_k_five(fix)
                        overrides_upload_all[sheet_name] = overrides_upload
                        ss_set("branch_overrides_upload", overrides_upload_all)
                        mark_session_edit(sheet_name, key)
                        store2 = infer_branch_options_with_overrides(df_in, overrides_upload)
                        label_map2 = build_label_child_map(store2)
                        df_new, tstats = cascade_anchor_reuse_full(df_in, store2, label_map2, sorted(set(df_in["Vital Measurement"].map(normalize_text))), [parent])
                        df_in = df_new
                        wb[sheet_name] = df_in; ss_set("upload_workbook", wb)
                        st.success(f"Saved 5 and auto-cascaded: added {tstats['new_rows']} rows, filled {tstats['inplace_filled']} anchors.")

        st.markdown("---")
        st.markdown("#### Push to Google Sheets (Raw+ augmented)")
        up_spreadsheet_id = st.text_input("Spreadsheet ID (Upload tab)", value=ss_get("gs_spreadsheet_id",""), key="up_sheet_id_main")
        if up_spreadsheet_id: ss_set("gs_spreadsheet_id", up_spreadsheet_id)
        default_tab_main = ss_get("saved_targets", {}).get(sheet_name, {}).get("tab", f"{sheet_name}")
        up_target_tab = st.text_input("Target tab name", value=default_tab_main, key="up_target_tab")
        include_scope_main = st.radio("Include scope", ["All completed parents","Only parents edited this session"], horizontal=True, key="up_scope_main")
        up_confirm = st.checkbox("I confirm I want to overwrite the target tab.", key="up_confirm")
        up_backup = st.checkbox("Create a backup tab before overwriting", value=True, key="up_backup")
        up_dry = st.checkbox("Dry-run (build but don't write to Sheets)", value=False, key="up_dry")
        show_dups = st.checkbox("Show rows that already exist (duplicates that will be skipped)", value=False, key="up_show_dups")

        edited_keys_for_sheet = set(ss_get("session_edited_keys", {}).get(sheet_name, []))

        if st.button("Push Raw+ (augmented)"):
            if not up_spreadsheet_id or not up_target_tab:
                st.warning("Enter Spreadsheet ID and target tab name.")
            elif not up_confirm and not up_dry:
                st.warning("Please tick the confirmation checkbox before pushing.")
            else:
                merged_overrides = {**overrides_upload, **user_fixes}
                scope_flag = "session" if include_scope_main.endswith("session") else "all"
                df_aug, stats, dups_df = build_raw_plus_v626(df_in, merged_overrides, scope_flag, edited_keys_for_sheet)

                st.info(f"Delta preview — Generated: **{stats['generated']}**, New added: **{stats['new_added']}**, In-place filled: **{stats['inplace_filled']}**, Duplicates skipped: **{stats['duplicates_skipped']}**, Final total: **{stats['final_total']}**.")
                st.caption(f"Will write **{len(df_aug)} rows × {len(df_aug.columns)} cols** to tab **{up_target_tab}**.")

                if show_dups and not dups_df.empty:
                    st.warning(f"Previewing {len(dups_df)} rows that already exist and will be treated as duplicates.")
                    st.dataframe(dups_df.head(200), use_container_width=True)
                    csvd = dups_df.to_csv(index=False).encode("utf-8")
                    st.download_button("Download duplicates (CSV)", data=csvd, file_name="duplicates_preview.csv", mime="text/csv")

                if up_dry:
                    st.success("Dry-run complete. No changes written to Google Sheets.")
                    st.dataframe(df_aug.head(50), use_container_width=True)
                    buffer = io.BytesIO()
                    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
                        df_aug.to_excel(writer, index=False, sheet_name=sheet_name[:31] or "Sheet1")
                    st.download_button("Download augmented (Raw+) workbook", data=buffer.getvalue(),
                                       file_name="decision_tree_raw_plus.xlsx",
                                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
                else:
                    if up_backup:
                        backup_name = backup_sheet_copy(up_spreadsheet_id, up_target_tab)
                        if backup_name: st.info(f"Backed up current '{up_target_tab}' to '{backup_name}'.")
                    ok = push_to_google_sheets(up_spreadsheet_id, up_target_tab, df_aug)
                    if ok:
                        log = ss_get("push_log", [])
                        log.append({
                            "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "sheet": sheet_name,
                            "target_tab": up_target_tab,
                            "spreadsheet_id": up_spreadsheet_id,
                            "rows_written": len(df_aug),
                            "new_rows_added": stats["new_added"],
                            "scope": scope_flag,
                        })
                        ss_set("push_log", log)
                        saved = ss_get("saved_targets", {})
                        saved.setdefault(sheet_name, {})
                        saved[sheet_name]["tab"] = up_target_tab
                        ss_set("saved_targets", saved)
                        st.success(f"Pushed {len(df_aug)} rows to Google Sheets.")

# ---------- Google Sheets mode ----------
with tab2:
    st.subheader("Google Sheets (optional)")
    if "gcp_service_account" not in st.secrets:
        st.error("Google Sheets not configured. Add your service account JSON under [gcp_service_account].")
    else:
        st.caption("Service account ready ✓")
    spreadsheet_id = st.text_input("Spreadsheet ID", value=ss_get("gs_spreadsheet_id",""), key="gs_id")
    if spreadsheet_id: ss_set("gs_spreadsheet_id", spreadsheet_id)
    sheet_name_g = st.text_input("Sheet name to process (e.g., BP)", value="BP")
    run_btn = st.button("Load")

    def get_gsheet_client():
        import gspread
        from google.oauth2.service_account import Credentials
        sa_info = st.secrets["gcp_service_account"]
        scopes = ["https://www.googleapis.com/auth/spreadsheets","https://www.googleapis.com/auth/drive"]
        creds = Credentials.from_service_account_info(sa_info, scopes=scopes)
        return gspread.authorize(creds)

    if run_btn:
        try:
            client = get_gsheet_client()
            sh = client.open_by_key(spreadsheet_id)
            ws = sh.worksheet(sheet_name_g)
            values = ws.get_all_values()
            if not values: st.error("Selected sheet is empty."); st.stop()
            header = values[0]; rows = values[1:]
            df_g = pd.DataFrame(rows, columns=header)
            df_g.columns = [normalize_text(c) for c in df_g.columns]
            if not validate_headers(df_g): st.error("Sheet does not match canonical headers."); st.stop()
            for c in CANON_HEADERS: df_g[c] = df_g[c].map(normalize_text)
            node_block = ["Vital Measurement"] + LEVEL_COLS
            df_g = df_g[~df_g[node_block].apply(lambda r: all(v == "" for v in r), axis=1)].copy()
            wb_g = ss_get("gs_workbook", {}); wb_g[sheet_name_g] = df_g; ss_set("gs_workbook", wb_g)
            st.success(f"Loaded '{sheet_name_g}' from Google Sheets.")
            st.dataframe(df_g.head(50), use_container_width=True)
        except Exception as e:
            st.error(f"Google Sheets error: {e}")

# ---------- Interactive completion tab ----------
with tab3:
    st.subheader("Fill Diagnostic & Actions (random)")
    st.caption("Works across ALL currently loaded sheets (Upload or Google Sheets). Select a source below.")
    sources = []
    if ss_get("upload_workbook", {}): sources.append("Upload workbook")
    if ss_get("gs_workbook", {}): sources.append("Google Sheets workbook")
    if not sources: st.info("Load a workbook first (Upload or Google Sheets tabs).")
    else:
        source = st.radio("Choose data source", sources, horizontal=True)
        if source == "Upload workbook":
            wb = ss_get("upload_workbook", {}); where = "upload"; override_root = "branch_overrides_upload"
        else:
            wb = ss_get("gs_workbook", {}); where = "gs"; override_root = "branch_overrides_gs"

        selected_sheets = st.multiselect("Limit to sheets", list(wb.keys()), default=list(wb.keys()))
        candidates = []
        for nm in selected_sheets:
            df = wb[nm]
            if not validate_headers(df): continue
            tri_empty = df["Diagnostic Triage"].map(normalize_text) == ""
            act_empty = df["Actions"].map(normalize_text) == ""
            path_complete = df[LEVEL_COLS].applymap(normalize_text).ne("").all(axis=1)
            mask = path_complete & (tri_empty | act_empty)
            idxs = list(df[mask].index)
            for ix in idxs: candidates.append((nm, ix))
        if not candidates: st.success("No rows found with empty Diagnostic Triage & Actions in the selected sheets. 🎉")
        else:
            if st.button("Pick a random incomplete row"):
                pick = random.choice(candidates); ss_set("act_sheet", pick[0]); ss_set("act_index", pick[1])
            sheet_cur = st.session_state.get("act_sheet"); idx_cur = st.session_state.get("act_index")
            if sheet_cur is not None and idx_cur is not None and sheet_cur in wb and idx_cur in wb[sheet_cur].index:
                st.markdown(f"**Selected sheet:** `{sheet_cur}` — **Row index:** {idx_cur}")
                row = wb[sheet_cur].loc[idx_cur]
                with st.expander("Symptom path for this row"):
                    vm = normalize_text(row["Vital Measurement"]); nodes = [normalize_text(row[c]) for c in LEVEL_COLS]
                    st.write(f"Vital Measurement: **{vm or '(blank)'}**")
                    for i, val in enumerate(nodes, start=1):
                        if val != "": st.write(f"Node {i}: **{val}**")
                tri_default = row.get("Diagnostic Triage", ""); act_default = row.get("Actions", "")
                tri = st.text_input("Diagnostic Triage", value=tri_default, key="tri_input")
                act = st.text_area("Actions", value=act_default, key="act_input")
                if st.button("Save to sheet NOW"):
                    wb[sheet_cur].loc[idx_cur, "Diagnostic Triage"] = tri; wb[sheet_cur].loc[idx_cur, "Actions"] = act
                    if where == "upload":
                        ss_set("upload_workbook", wb)
                        buffer = io.BytesIO()
                        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
                            for nm, d in wb.items():
                                if not validate_headers(d) and d.empty: d = pd.DataFrame(columns=CANON_HEADERS)
                                d.to_excel(writer, index=False, sheet_name=nm[:31] or "Sheet1")
                        st.download_button("Download UPDATED workbook (with your changes)", data=buffer.getvalue(), file_name="decision_tree_updated.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
                        st.success("Saved in-session. Download to persist locally.")
                    else:
                        sid = ss_get("gs_spreadsheet_id", "")
                        if not sid: st.error("Missing Spreadsheet ID in session. Reload in the Google Sheets tab.")
                        else:
                            # Deep cascade Raw+ push for this sheet
                            overrides_all = ss_get(override_root, {})
                            overrides_sheet = overrides_all.get(sheet_cur, {})
                            edited_keys_for_sheet = set(ss_get("session_edited_keys", {}).get(sheet_cur, []))
                            df_aug, stats, dups_df = build_raw_plus_v626(wb[sheet_cur], overrides_sheet, include_scope="all", edited_keys_for_sheet=edited_keys_for_sheet)
                            st.caption(f"Will write **{len(df_aug)} rows × {len(df_aug.columns)} cols** to tab **{sheet_cur}**.")
                            ok = push_to_google_sheets(sid, sheet_cur, df_aug)
                            if ok:
                                log = ss_get("push_log", [])
                                log.append({
                                    "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                    "sheet": sheet_cur,
                                    "target_tab": sheet_cur,
                                    "spreadsheet_id": sid,
                                    "rows_written": len(df_aug),
                                    "new_rows_added": stats["new_added"],
                                    "scope": "all",
                                })
                                ss_set("push_log", log)
                                st.success("Changes pushed to Google Sheets (Raw+).")
            else: st.info("Click 'Pick a random incomplete row' to begin.")

# ---------- Symptoms browser & editor ----------
with tab4:
    st.subheader("Symptoms — browse, check consistency, edit child branches, auto-cascade, push Raw+, export PDF")

    # Push settings
    with st.expander("🔧 Google Sheets Push Settings"):
        sid_current = ss_get("gs_spreadsheet_id", "")
        sid_input = st.text_input("Spreadsheet ID for pushes", value=sid_current, key="sym_sid")
        if sid_input and sid_input != sid_current:
            ss_set("gs_spreadsheet_id", sid_input)
        include_scope_sym = st.radio("Include scope for Raw+", ["All completed parents","Only parents edited this session"], horizontal=True, key="sym_scope")
        st.caption("Recursive deep cascade with anchor-reuse is the default.")
        push_backup = st.checkbox("Create a backup tab before overwrite", value=True, key="sym_push_backup")
        dry_run_sym = st.checkbox("Dry-run (build but don't write to Sheets)", value=False, key="sym_dry_run")
        show_dups_sym = st.checkbox("Show duplicates preview after build", value=False, key="sym_show_dups")

    # Export/Import Overrides (per-sheet)
    with st.expander("📦 Export / Import Overrides (JSON)"):
        overrides_all = ss_get("branch_overrides_upload", {})
        overrides_sheet = overrides_all.get(st.session_state.get("sym_sheet", ss_get('sym_sheet','')), {})
        st.caption("Export the current sheet’s overrides to a JSON file, or import to replace/merge.")
        export_col, import_col = st.columns([1,2])
        with export_col:
            st.write("Use after selecting a sheet (below).")
        with import_col:
            st.write("Upload JSON after selecting a sheet (below).")

    # Undo stack controls
    if st.button("↩️ Undo last branch edit (session)"):
        stack = ss_get("undo_stack", [])
        if not stack:
            st.info("Nothing to undo.")
        else:
            last = stack.pop()
            ss_set("undo_stack", stack)
            context = last.get("context")
            override_root = last.get("override_root")
            if context == "upload":
                bo_all = ss_get("branch_overrides_upload", {})
                bo_all[last["sheet"]] = last["overrides_sheet_before"]
                ss_set("branch_overrides_upload", bo_all)
                if last.get("df_before") is not None:
                    wb = ss_get("upload_workbook", {})
                    wb[last["sheet"]] = last["df_before"]
                    ss_set("upload_workbook", wb)
                st.success(f"Undid Upload override changes for sheet '{last['sheet']}'.")
            else:
                overrides_all = ss_get(override_root, {})
                overrides_all[last["sheet"]] = last["overrides_sheet_before"]
                ss_set(override_root, overrides_all)
                if last.get("df_before") is not None:
                    src = ss_get("upload_workbook", {})
                    if last["sheet"] in src:
                        src[last["sheet"]] = last["df_before"]
                        ss_set("upload_workbook", src)
                st.success(f"Undid Symptoms tab edit on sheet '{last['sheet']}'.")

    sources = []
    if ss_get("upload_workbook", {}): sources.append("Upload workbook")
    if ss_get("gs_workbook", {}): sources.append("Google Sheets workbook")

    if not sources:
        st.info("Load a workbook first (Upload or Google Sheets tabs).")
    else:
        source = st.radio("Choose data source", sources, horizontal=True, key="sym_source")
        if source == "Upload workbook":
            wb = ss_get("upload_workbook", {})
            override_root = "branch_overrides_upload"
        else:
            wb = ss_get("gs_workbook", {})
            override_root = "branch_overrides_gs"

        sheet = st.selectbox("Sheet", list(wb.keys()), key="sym_sheet")
        df = wb.get(sheet, pd.DataFrame())

        # Wire up Export/Import now that we know 'sheet'
        with st.expander("📦 Export / Import Overrides (JSON)"):
            overrides_all = ss_get(override_root, {})
            overrides_sheet = overrides_all.get(sheet, {})
            col1, col2 = st.columns([1,2])
            with col1:
                data = json.dumps(overrides_sheet, indent=2).encode("utf-8")
                st.download_button("Export overrides.json", data=data, file_name=f"{sheet}_overrides.json", mime="application/json")
            with col2:
                upfile = st.file_uploader("Import overrides.json", type=["json"], key="imp_json")
                import_mode = st.radio("Import mode", ["Replace", "Merge (prefer import)", "Merge (prefer existing)"], horizontal=True)
                auto_cascade = st.checkbox("Auto-cascade after import", value=True)
                if upfile is not None and st.button("Apply Import"):
                    try:
                        imported = json.loads(upfile.getvalue().decode("utf-8"))
                        if not isinstance(imported, dict):
                            st.error("Invalid JSON: expected an object mapping keys to lists.")
                        else:
                            stack = ss_get("undo_stack", [])
                            stack.append({
                                "context": "symptoms",
                                "label": "Import overrides JSON",
                                "override_root": override_root,
                                "sheet": sheet,
                                "overrides_sheet_before": overrides_all.get(sheet, {}).copy(),
                                "df_before": df.copy()
                            })
                            ss_set("undo_stack", stack)

                            if import_mode == "Replace":
                                new_over = {}
                                for k,v in imported.items():
                                    new_over[k] = enforce_k_five(v if isinstance(v, list) else [v])
                                    mark_session_edit(sheet, k)
                                overrides_all[sheet] = new_over
                            elif import_mode == "Merge (prefer import)":
                                cur = overrides_all.get(sheet, {}).copy()
                                for k,v in imported.items():
                                    cur[k] = enforce_k_five(v if isinstance(v, list) else [v])
                                    mark_session_edit(sheet, k)
                                overrides_all[sheet] = cur
                            else:  # prefer existing
                                cur = overrides_all.get(sheet, {}).copy()
                                for k,v in imported.items():
                                    if k not in cur:
                                        cur[k] = enforce_k_five(v if isinstance(v, list) else [v])
                                        mark_session_edit(sheet, k)
                                overrides_all[sheet] = cur

                            ss_set(override_root, overrides_all)
                            st.success("Overrides imported.")
                            if auto_cascade and not df.empty and validate_headers(df):
                                store2 = infer_branch_options_with_overrides(df, overrides_all[sheet])
                                label_map2 = build_label_child_map(store2)
                                vms = sorted(set(df["Vital Measurement"].map(normalize_text).replace("", np.nan).dropna().unique().tolist()))
                                df_new, tstats = cascade_anchor_reuse_full(df, store2, label_map2, vms, [tuple()])
                                wb[sheet] = df_new
                                if source == "Upload workbook":
                                    ss_set("upload_workbook", wb)
                                else:
                                    ss_set("gs_workbook", wb)
                                st.info(f"Cascaded: +{tstats['new_rows']} rows, {tstats['inplace_filled']} anchors filled.")
                    except Exception as e:
                        st.error(f"Import failed: {e}")

        # Build store + parents
        overrides_current = ss_get(override_root, {}).get(sheet, {})
        store = infer_branch_options_with_overrides(df, overrides_current)

        # Data Quality Tools
        if not df.empty and validate_headers(df):
            with st.expander("🧼 Data quality tools (applies to this sheet)"):
                ok_p, total_p = compute_parent_depth_score(df)
                ok_r, total_r = compute_row_path_score(df)
                st.write(f"Parents with 5 children: **{ok_p}/{total_p}**")
                st.write(f"Rows with full path: **{ok_r}/{total_r}**")
                case_mode = st.selectbox("Case normalization", ["None","Title","lower","UPPER"], index=0)
                syn_text = st.text_area("Synonym map (one per line: A => B)", key="sym_syn_map_text")
                def parse_synonym_map(text: str) -> Dict[str,str]:
                    mapping = {}
                    for line in text.splitlines():
                        if "=>" in line:
                            a,b = line.split("=>",1)
                            a = a.strip(); b = b.strip()
                            if a and b: mapping[a] = b
                    return mapping
                def normalize_label(s: str, case_mode: str) -> str:
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
                                v = normalize_label(v, case_mode) if case_mode!="None" else v
                                return v
                            df2[col] = df2[col].map(_apply)
                    for col in ["Diagnostic Triage","Actions"]:
                        if col in df2.columns:
                            df2[col] = df2[col].map(normalize_text)
                    return df2
                if st.button("Normalize sheet now"):
                    syn_map = parse_synonym_map(syn_text)
                    df_norm = normalize_sheet_df(df, case_mode, syn_map)
                    wb[sheet] = df_norm
                    if source == "Upload workbook":
                        ss_set("upload_workbook", wb)
                        st.success("Sheet normalized in-session. Download in Upload tab to persist locally.")
                    else:
                        sid = ss_get("gs_spreadsheet_id","")
                        if not sid:
                            st.error("Missing Spreadsheet ID in session.")
                        else:
                            st.caption(f"Will write **{len(df_norm)} rows × {len(df_norm.columns)} cols** to tab **{sheet}**.")
                            ok = push_to_google_sheets(sid, sheet, df_norm)
                            if ok: st.success("Normalized sheet pushed to Google Sheets.")

        # Controls (Symptoms list)
        level = st.selectbox("Level to inspect (child options of...)", [1,2,3,4,5], format_func=lambda x: f"Node {x}", key="sym_level")

        # Build parents list
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

        # Persistent search helpers
        _pending = st.session_state.pop("sym_search_pending", None)
        if _pending is not None:
            st.session_state["sym_search"] = _pending
        top_cols = st.columns([2,1,1,1,2])
        with top_cols[0]:
            search = st.text_input("Search parent symptom/path", key="sym_search").strip().lower()
        with top_cols[1]:
            if st.button("Next Missing"):
                for pt in sorted(parents_by_level.get(level, set())):
                    key = level_key_tuple(level, pt)
                    if len([x for x in store.get(key, []) if normalize_text(x)!=""]) == 0:
                        st.session_state["sym_search_pending"] = (" > ".join(pt) or "<ROOT>").lower()
                        st.rerun()
        with top_cols[2]:
            if st.button("Next Symptom left out"):
                for pt in sorted(parents_by_level.get(level, set())):
                    key = level_key_tuple(level, pt)
                    n = len([x for x in store.get(key, []) if normalize_text(x)!=""])
                    if 1 <= n < 5:
                        st.session_state["sym_search_pending"] = (" > ".join(pt) or "<ROOT>").lower()
                        st.rerun()
        with top_cols[3]:
            compact = st.checkbox("Compact mode", value=True)
        with top_cols[4]:
            parent_choices = ["(select parent)"] + [(" > ".join(p) or "<ROOT>") for p in sorted(parents_by_level.get(level, set()))]
            pick_parent = st.selectbox("Quick jump", parent_choices)
            if pick_parent and pick_parent != "(select parent)":
                st.session_state["sym_search_pending"] = pick_parent.lower()
                st.rerun()

        sort_mode = st.radio("Sort by", ["Problem severity (issues first)", "Alphabetical (parent path)"], horizontal=True)

        # Build entries
        entries = []
        label_childsets: Dict[Tuple[int,str], set] = {}
        for parent_tuple in sorted(parents_by_level.get(level, set())):
            parent_text = " > ".join(parent_tuple)
            if search and (search not in parent_text.lower()):
                continue
            key = level_key_tuple(level, parent_tuple)
            children = [x for x in store.get(key, []) if normalize_text(x)!=""]
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

        # Vocabulary
        def build_vocabulary(df0: pd.DataFrame) -> List[str]:
            vocab = set()
            for col in LEVEL_COLS:
                if col in df0.columns:
                    for x in df0[col].dropna().astype(str):
                        x = x.strip()
                        if x: vocab.add(x)
            return sorted(vocab)
        vocab = build_vocabulary(df)
        vocab_opts = ["(pick suggestion)"] + vocab

        # Render entries with inline editing + auto-cascade
        for parent_tuple, children, status in entries:
            keyname = level_key_tuple(level, parent_tuple)
            subtitle = f"{' > '.join(parent_tuple) or '<ROOT>'} — {status} {'⚠️' if (level, (parent_tuple[-1] if parent_tuple else '<ROOT>')) in inconsistent_labels else ''}"

            with st.expander(subtitle):
                selected_vals = []
                if compact:
                    for i in range(5):
                        default_val = children[i] if i < len(children) else ""
                        txt_key = f"sym_txt_{level}_{'__'.join(parent_tuple)}_{i}"
                        sel_key = f"sym_sel_{level}_{'__'.join(parent_tuple)}_{i}"
                        txt = st.text_input(f"Child {i+1}", value=default_val, key=txt_key)
                        pick = st.selectbox("", options=vocab_opts, index=0, key=sel_key, label_visibility="collapsed")
                        selected_vals.append((txt, pick))
                else:
                    cols = st.columns(5)
                    for i in range(5):
                        default_val = children[i] if i < len(children) else ""
                        with cols[i]:
                            txt_key = f"sym_txt_{level}_{'__'.join(parent_tuple)}_{i}"
                            sel_key = f"sym_sel_{level}_{'__'.join(parent_tuple)}_{i}"
                            txt = st.text_input(f"Child {i+1}", value=default_val, key=txt_key)
                            pick = st.selectbox("Pick", options=vocab_opts, index=0, key=sel_key)
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
                    overrides_all = ss_get(override_root, {})
                    overrides_sheet = overrides_all.get(sheet, {}).copy()
                    stack = ss_get("undo_stack", [])
                    stack.append({
                        "context": "symptoms",
                        "label": f"Save parent {(' > '.join(parent_tuple) or '<ROOT>')} (L{level})",
                        "override_root": override_root,
                        "sheet": sheet,
                        "level": level,
                        "parent": parent_tuple,
                        "overrides_sheet_before": overrides_all.get(sheet, {}).copy(),
                        "df_before": df.copy()
                    })
                    ss_set("undo_stack", stack)
                    overrides_sheet[keyname] = fixed
                    overrides_all[sheet] = overrides_sheet
                    ss_set(override_root, overrides_all)
                    mark_session_edit(sheet, keyname)
                    # Auto-cascade deep with anchor-reuse + label propagation
                    store2 = infer_branch_options_with_overrides(df, overrides_sheet)
                    label_map2 = build_label_child_map(store2)
                    vms = sorted(set(df["Vital Measurement"].map(normalize_text).replace("", np.nan).dropna().unique().tolist()))
                    df_new, tstats = cascade_anchor_reuse_full(df, store2, label_map2, vms, [parent_tuple])
                    wb[sheet] = df_new
                    if source == "Upload workbook":
                        ss_set("upload_workbook", wb)
                    else:
                        ss_set("gs_workbook", wb)
                    st.success(f"Saved and auto-cascaded: added {tstats['new_rows']} rows, filled {tstats['inplace_filled']} anchors.")
                    st.caption("Tip: use Bulk Push below to write to Google Sheets.")

        # ---------- Conflicts Inspector ----------
        with st.expander("🧯 Conflicts Inspector (same parent label → different child sets)"):
            # Build label -> list of (level, parent_tuple, child_set)
            label_to_sets: Dict[str, List[Tuple[int, Tuple[str,...], Tuple[str,...]]]] = {}
            for key, children in store.items():
                if "|" not in key: continue
                lvl_s, path = key.split("|", 1)
                try: L = int(lvl_s[1:])
                except: continue
                if path == "<ROOT>": continue
                parent_tuple = tuple(path.split(">"))
                label = parent_tuple[-1]
                child_set = tuple([c for c in children if normalize_text(c)!=""])
                label_to_sets.setdefault(label, []).append((L, parent_tuple, tuple(sorted(child_set))))

            conflicts = []
            for label, entries_list in label_to_sets.items():
                variants = sorted(set([s for (_,_,s) in entries_list]))
                if len(variants) > 1:
                    conflicts.append((label, variants, entries_list))

            if not conflicts:
                st.success("No conflicts detected 🎉")
            else:
                st.warning(f"Found {len(conflicts)} conflicting labels.")
                resolve_ops = {}
                for i, (label, variants, entries_list) in enumerate(conflicts, start=1):
                    with st.expander(f"{i}. '{label}' has {len(variants)} different child sets"):
                        st.write("Variants:")
                        for j, var in enumerate(variants, start=1):
                            st.code(f"{j}. {list(var)}")
                        union_children = []
                        seen = set()
                        for var in variants:
                            for c in var:
                                if c not in seen and c!="":
                                    union_children.append(c); seen.add(c)
                        union_children = enforce_k_five(union_children)
                        sel = st.multiselect("Resolve to EXACTLY 5 children (choose up to 5)", union_children, default=union_children[:5], key=f"ci_{label}")
                        sel = enforce_k_five(sel)
                        st.caption(f"Will apply to all occurrences of parent label '{label}' across levels (affecting next-level children).")
                        resolve_ops[label] = sel

                if st.button("Apply resolutions across this sheet"):
                    overrides_all = ss_get(override_root, {})
                    overrides_sheet = overrides_all.get(sheet, {}).copy()
                    stack = ss_get("undo_stack", [])
                    stack.append({
                        "context": "symptoms",
                        "label": "Conflicts resolution",
                        "override_root": override_root,
                        "sheet": sheet,
                        "overrides_sheet_before": overrides_all.get(sheet, {}).copy(),
                        "df_before": df.copy()
                    })
                    ss_set("undo_stack", stack)
                    # For each key in store, if parent label matches and not ROOT, override its children
                    for key in list(store.keys()):
                        if "|" not in key: continue
                        lvl_s, path = key.split("|", 1)
                        if path == "<ROOT>": continue
                        parent_tuple = tuple(path.split(">"))
                        label = parent_tuple[-1]
                        if label in resolve_ops:
                            overrides_sheet[key] = enforce_k_five(resolve_ops[label])
                            mark_session_edit(sheet, key)
                    overrides_all[sheet] = overrides_sheet
                    ss_set(override_root, overrides_all)

                    # cascade
                    store2 = infer_branch_options_with_overrides(df, overrides_sheet)
                    label_map2 = build_label_child_map(store2)
                    vms = sorted(set(df["Vital Measurement"].map(normalize_text).replace("", np.nan).dropna().unique().tolist()))
                    df_new, tstats = cascade_anchor_reuse_full(df, store2, label_map2, vms, [tuple()])
                    wb[sheet] = df_new
                    if source == "Upload workbook":
                        ss_set("upload_workbook", wb)
                    else:
                        ss_set("gs_workbook", wb)
                    st.success(f"Applied resolutions. Cascaded: +{tstats['new_rows']} rows, {tstats['inplace_filled']} anchors.")

        st.markdown("---")
        st.markdown("#### Bulk Push Raw+ for this sheet")
        colA, colB = st.columns([1,2])
        with colA:
            sid = ss_get("gs_spreadsheet_id", "")
            sid = st.text_input("Spreadsheet ID", value=sid, key="sym_sid_bulk") or sid
            if sid: ss_set("gs_spreadsheet_id", sid)

            default_tab = ss_get("saved_targets", {}).get(sheet, {}).get("tab", f"{sheet}")
            target_tab = st.text_input("Target tab", value=default_tab, key="sym_target_tab")
            scope_flag = "session" if include_scope_sym.endswith("session") else "all"
            edited_keys_for_sheet = set(ss_get("session_edited_keys", {}).get(sheet, []))

            if st.button("📤 Bulk Push Raw+ (build/overwrite)", type="primary"):
                if not sid or not target_tab:
                    st.error("Missing Spreadsheet ID or target tab.")
                else:
                    df_aug, stats, dups_df = build_raw_plus_v626(df, overrides_current, scope_flag, edited_keys_for_sheet)
                    st.info(f"Delta preview — Generated: **{stats['generated']}**, New added: **{stats['new_added']}**, In-place filled: **{stats['inplace_filled']}**, Duplicates skipped: **{stats['duplicates_skipped']}**, Final total: **{stats['final_total']}**.")
                    st.caption(f"Will write **{len(df_aug)} rows × {len(df_aug.columns)} cols** to tab **{target_tab}**.")
                    if dry_run_sym:
                        st.success("Dry-run complete. No changes written to Google Sheets.")
                        st.dataframe(df_aug.head(50), use_container_width=True)
                        buffer = io.BytesIO()
                        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
                            df_aug.to_excel(writer, index=False, sheet_name=sheet[:31] or "Sheet1")
                        st.download_button("Download augmented (Raw+) workbook", data=buffer.getvalue(),
                                           file_name="decision_tree_raw_plus.xlsx",
                                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
                    else:
                        if ss_get("sym_push_backup", True) and push_backup:
                            backup_name = backup_sheet_copy(sid, target_tab)
                            if backup_name: st.info(f"Backed up current '{target_tab}' to '{backup_name}'.")
                        ok = push_to_google_sheets(sid, target_tab, df_aug)
                        if ok:
                            log = ss_get("push_log", [])
                            log.append({
                                "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                "sheet": sheet,
                                "target_tab": target_tab,
                                "spreadsheet_id": sid,
                                "rows_written": len(df_aug),
                                "new_rows_added": stats["new_added"],
                                "scope": scope_flag,
                            })
                            ss_set("push_log", log)
                            saved = ss_get("saved_targets", {})
                            saved.setdefault(sheet, {})
                            saved[sheet]["tab"] = target_tab
                            ss_set("saved_targets", saved)
                            st.success(f"Pushed {len(df_aug)} rows to '{target_tab}'.")
        with colB:
            st.caption("Raw+ uses recursive deep cascade with anchor-reuse in v6.2.6.")
            if show_dups_sym and 'dups_df' in locals():
                if not dups_df.empty:
                    st.warning(f"Previewing {len(dups_df)} duplicate rows (already existed).")
                    st.dataframe(dups_df.head(200), use_container_width=True)
                    csvd = dups_df.to_csv(index=False).encode("utf-8")
                    st.download_button("Download duplicates (CSV)", data=csvd, file_name="duplicates_preview.csv", mime="text/csv")

# ---------- Push Log ----------
with tab5:
    st.subheader("Push Log")
    log = ss_get("push_log", [])
    if not log:
        st.info("No pushes recorded this session.")
    else:
        df_log = pd.DataFrame(log)
        st.dataframe(df_log, use_container_width=True)
        csv = df_log.to_csv(index=False).encode("utf-8")
        st.download_button("Download push log (CSV)", data=csv, file_name="push_log.csv", mime="text/csv")

# ---------- Dictionary ----------
with tab6:
    st.subheader("Dictionary — all symptom labels (branches)")

    sources_avail = []
    if ss_get("upload_workbook", {}): sources_avail.append("Upload workbook")
    if ss_get("gs_workbook", {}): sources_avail.append("Google Sheets workbook")

    if not sources_avail:
        st.info("Load a workbook in the Upload or Google Sheets tab first.")
    else:
        source_choice = st.multiselect("Include sources", sources_avail, default=sources_avail)
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
                for lvl, col in enumerate(LEVEL_COLS, start=1):
                    if col in df0.columns:
                        for val in df0[col].astype(str).map(normalize_text):
                            if not val: continue
                            counts[val] = counts.get(val, 0) + 1
                            levels_map.setdefault(val, set()).add(lvl)

            # Quality store (in-session)
            quality_map = ss_get("symptom_quality", {})  # {symptom: "Red Flag"|"Normal"}

            # Construct table
            rows = []
            for symptom, cnt in counts.items():
                levels_list = sorted(list(levels_map.get(symptom, set())))
                rows.append({
                    "Symptom": symptom,
                    "Count": cnt,
                    "Levels": ", ".join([f"Node {i}" for i in levels_list]),
                    "Red Flag": (quality_map.get(symptom, "Normal") == "Red Flag")
                })
            dict_df = pd.DataFrame(rows).sort_values(["Symptom"]).reset_index(drop=True)

            # --- Filters & Search ---
            with st.expander("Filters / Search", expanded=True):
                q = st.text_input("Search Symptom (substring, case-insensitive)", key="dict_search").strip().lower()
                levels_all = sorted({i for s in levels_map.values() for i in s})
                level_filter = st.multiselect("Filter by Levels (optional)", [f"Node {i}" for i in levels_all], default=[], key="dict_level_filter")
                red_only = st.checkbox("Show only Red Flags", value=False, key="dict_red_only")
                min_count = st.number_input("Min Count", min_value=0, max_value=int(dict_df["Count"].max() if not dict_df.empty else 0), value=0, step=1)

            filtered = dict_df.copy()
            if q:
                filtered = filtered[filtered["Symptom"].str.lower().str.contains(q)]
            if level_filter:
                mask = filtered["Levels"].apply(lambda s: any(lv in s for lv in level_filter))
                filtered = filtered[mask]
            if red_only:
                filtered = filtered[filtered["Red Flag"] == True]
            if min_count > 0:
                filtered = filtered[filtered["Count"] >= min_count]

            st.markdown("#### Dictionary (editable Red Flag column)")
            edited = st.data_editor(
                filtered,
                use_container_width=True,
                num_rows="dynamic",
                column_config={
                    "Symptom": st.column_config.TextColumn(disabled=True),
                    "Count": st.column_config.NumberColumn(disabled=True),
                    "Levels": st.column_config.TextColumn(disabled=True),
                    "Red Flag": st.column_config.CheckboxColumn(help="Toggle to mark/unmark this symptom as a Red Flag")
                },
                hide_index=True,
                key="dict_editor"
            )

            # Persist Red Flag changes back to session map
            if st.button("💾 Save Red Flag changes"):
                edited_map = quality_map.copy()
                for _, row in edited.iterrows():
                    edited_map[row["Symptom"]] = "Red Flag" if bool(row["Red Flag"]) else "Normal"
                ss_set("symptom_quality", edited_map)
                st.success("Symptom Red Flags saved for this session.")

            # Download CSV (with current filter view and Red Flag boolean)
            csv = filtered.to_csv(index=False).encode("utf-8")
            st.download_button("Download current view (CSV)", data=csv, file_name="symptom_dictionary.csv", mime="text/csv")
