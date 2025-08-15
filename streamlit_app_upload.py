# streamlit_app_upload.py — Version 6.2.1

import io
import random
from typing import Dict, List, Tuple, Optional, Set
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime

# ============ VERSION / CONFIG ============
APP_VERSION = "v6.2.1"
CANON_HEADERS = ["Vital Measurement","Node 1","Node 2","Node 3","Node 4","Node 5","Diagnostic Triage","Actions"]
LEVEL_COLS = ["Node 1","Node 2","Node 3","Node 4","Node 5"]
SHEET_COMPLETED_SUFFIX = " (Completed)"  # kept only for backward compatibility with old tabs; not used as a mode now
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
    """Remember which parent-keys were edited during this session (per sheet)."""
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
    # Allow extra trailing columns but first 8 must match canonical exactly
    return list(df.columns[:len(CANON_HEADERS)]) == CANON_HEADERS

def parent_key_from_row_strict(row: pd.Series, upto_level: int) -> Optional[Tuple[str, ...]]:
    """Return (level-1)-length parent tuple only if ALL earlier nodes are non-empty."""
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
    """From existing rows, collect distinct child options observed under each (level,parent_path)."""
    store: Dict[str, List[str]] = {}
    for level in range(1, MAX_LEVELS+1):
        parent_to_children: Dict[Tuple[str, ...], List[str]] = {}
        for _, row in df.iterrows():
            if LEVEL_COLS[level-1] not in df.columns: 
                continue
            child = normalize_text(row[LEVEL_COLS[level-1]])
            if child == "":
                continue
            parent = parent_key_from_row_strict(row, level)
            if parent is None:
                continue
            parent_to_children.setdefault(parent, [])
            parent_to_children[parent].append(child)
        for parent, children in parent_to_children.items():
            uniq = []
            seen = set()
            for c in children:
                if c not in seen:
                    seen.add(c); uniq.append(c)
            store[level_key_tuple(level, parent)] = uniq
    return store

def infer_branch_options_with_overrides(df: pd.DataFrame, overrides: Dict[str, List[str]]) -> Dict[str, List[str]]:
    """Merge observed store with overrides. Do NOT enforce 5 here."""
    base = infer_branch_options(df)
    merged = dict(base)
    for k, v in (overrides or {}).items():
        vals = [normalize_text(x) for x in (v if isinstance(v, list) else [v])]
        merged[k] = vals
    return merged

def enforce_k_five(opts: List[str]) -> List[str]:
    """Ensure exactly 5 options (trim or pad with blanks)."""
    clean = [normalize_text(o) for o in opts if normalize_text(o) != ""]
    if len(clean) > 5:
        clean = clean[:5]
    elif len(clean) < 5:
        clean = clean + [""]*(5-len(clean))
    return clean

def compute_virtual_parents(store: Dict[str, List[str]]) -> Dict[int, Set[Tuple[str, ...]]]:
    """Expand parents forward using known store (observed + overrides)."""
    parents_by_level: Dict[int, Set[Tuple[str, ...]]] = {i: set() for i in range(1, MAX_LEVELS+1)}
    parents_by_level[1].add(tuple())  # <ROOT> for Node 1
    for level in range(1, MAX_LEVELS):
        for p in list(parents_by_level[level]):
            key = level_key_tuple(level, p)
            children = store.get(key, [])
            for c in children:
                if c != "":
                    parents_by_level[level+1].add(p + (c,))
    # include explicit parents from store keys
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

# ============ Google Sheets helpers (RESIZE BEFORE UPDATE) ============
def push_to_google_sheets(spreadsheet_id: str, sheet_name: str, df: pd.DataFrame) -> bool:
    """Overwrite or create target sheet with df using service account in secrets.
    Ensures the worksheet is resized so rows are never truncated.
    """
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

        # Prepare data
        df = df.fillna("")
        headers = list(df.columns)
        values = [headers] + df.astype(str).values.tolist()
        n_rows = len(values)  # includes header row
        n_cols = max(1, len(headers))

        try:
            ws = sh.worksheet(sheet_name)
            ws.clear()
            ws.resize(rows=max(n_rows, 200), cols=max(n_cols, 8))
        except Exception:
            ws = sh.add_worksheet(
                title=sheet_name,
                rows=max(n_rows, 200),
                cols=max(n_cols, 8)
            )

        ws.update('A1', values, value_input_option="RAW")
        return True

    except Exception as e:
        st.error(f"Push to Google Sheets failed: {e}")
        return False

def backup_sheet_copy(spreadsheet_id: str, source_sheet: str) -> Optional[str]:
    """Create a backup tab by copying values from source_sheet into a new '(backup YYYY-MM-DD HHMM)' tab."""
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
        backup_title = backup_title_full[:99]  # Sheets allows up to 100 chars

        rows = max(len(values), 100)
        cols = max(len(values[0]) if values else 8, 8)

        ws_bak = sh.add_worksheet(title=backup_title, rows=rows, cols=cols)
        if values:
            ws_bak.update('A1', values, value_input_option="RAW")
        return backup_title
    except Exception as e:
        st.error(f"Backup failed: {e}")
        return None

# ============ RAW+ (augmented) builder ============
def build_raw_plus(
    df: pd.DataFrame,
    overrides: Dict[str, List[str]],
    include_scope: str,
    edited_keys_for_sheet: Set[str]
) -> Tuple[pd.DataFrame, Dict[str,int]]:
    """
    Return the raw dataframe augmented with extra rows generated from (observed + overrides)
    for 5/5-complete subtrees. Dedup by (VM + Node1..5). New rows have empty Triage/Actions.
    include_scope: 'all' or 'session' (only subtrees rooted at parents edited this session).
    """
    # Build store and enforce exactly 5 where specified
    store = infer_branch_options_with_overrides(df, overrides)
    for k in list(store.keys()):
        store[k] = enforce_k_five(store[k])

    all_vm = sorted(set(normalize_text(x) for x in df["Vital Measurement"] if normalize_text(x) != ""))

    generated_rows = []

    def expand_from_prefix(vm_val: str, start_level: int, prefix: Tuple[str, ...]):
        def dfs(level: int, pref: Tuple[str, ...]):
            if level > MAX_LEVELS:
                row = {"Vital Measurement": vm_val}
                for i, val in enumerate(pref, 1):
                    row[f"Node {i}"] = val
                for i in range(1, MAX_LEVELS+1):
                    row.setdefault(f"Node {i}", "")
                row["Diagnostic Triage"] = ""
                row["Actions"] = ""
                generated_rows.append(row); return
            key = level_key_tuple(level, pref)
            opts = store.get(key, [])
            if len(opts) != 5 or any(o == "" for o in opts):
                return
            for o in opts:
                dfs(level+1, pref + (o,))
        dfs(start_level, prefix)

    # Choose prefixes based on scope
    prefixes: List[Tuple[int, Tuple[str, ...]]] = []
    if include_scope == "session":
        for keyname in edited_keys_for_sheet:
            if "|" not in keyname: 
                continue
            lvl_s, path = keyname.split("|", 1)
            try:
                lvl = int(lvl_s[1:])
            except:
                continue
            parent_tuple = tuple([] if path == "<ROOT>" else path.split(">"))
            prefixes.append((lvl, parent_tuple))
    else:
        prefixes = [(1, tuple())]  # start from root

    # Generate rows from selected prefixes for every VM
    for vm_val in all_vm:
        for (lvl, parent) in prefixes:
            expand_from_prefix(vm_val, lvl, parent)

    # Deduplicate against existing rows
    keycols = ["Vital Measurement"] + LEVEL_COLS
    def make_key(rowlike) -> Tuple[str,...]:
        return tuple(normalize_text(rowlike.get(c, "")) for c in keycols)

    existing_keys = set()
    for _, r in df.iterrows():
        existing_keys.add(make_key(r))

    to_append = []
    duplicates = 0
    for r in generated_rows:
        k = make_key(r)
        if k in existing_keys:
            duplicates += 1
        else:
            to_append.append(r)
            existing_keys.add(k)

    df_aug = pd.concat([df, pd.DataFrame(to_append, columns=CANON_HEADERS)], ignore_index=True)
    stats = {
        "generated": len(generated_rows),
        "new_added": len(to_append),
        "duplicates_skipped": duplicates,
        "final_total": len(df_aug)
    }
    return df_aug, stats

# ======== Progress / depth metrics (for badges/quality) ========
def compute_parent_depth_score(df: pd.DataFrame) -> Tuple[int, int]:
    """Parents with exactly 5 children across all levels."""
    store = infer_branch_options(df)
    total = 0; ok = 0
    for level in range(1, MAX_LEVELS+1):
        # parents observed at this level
        parents = set()
        for _, row in df.iterrows():
            p = parent_key_from_row_strict(row, level)
            if p is not None:
                parents.add(p)
        for p in parents:
            total += 1
            key = level_key_tuple(level, p)
            if len(store.get(key, [])) == 5:
                ok += 1
    return ok, total

def compute_row_path_score(df: pd.DataFrame) -> Tuple[int, int]:
    """Rows where Node 1..5 are all non-empty (full path)."""
    if df.empty:
        return (0,0)
    nodes = df[LEVEL_COLS].applymap(normalize_text)
    full = nodes.ne("").all(axis=1)
    return int(full.sum()), int(len(df))

def branch_depth_badge_html(df: pd.DataFrame) -> str:
    ok_p, total_p = compute_parent_depth_score(df)
    ok_r, total_r = compute_row_path_score(df)
    return (
        f"<div style='display:inline-block;background:#eef6ff;"
        f"padding:6px 10px;border-radius:8px;border:1px solid #cfe2ff;margin-left:8px;'>"
        f"Parents 5/5: <b>{ok_p}/{total_p}</b> &nbsp;|&nbsp; Rows full path: <b>{ok_r}/{total_r}</b>"
        f"</div>"
    )

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
        # If a sheet is selected later, we’ll recompute; here show for first sheet if exists
        first_sheet = next(iter(wb_upload))
        df0 = wb_upload[first_sheet]
        if validate_headers(df0) and not df0.empty:
            badges = branch_depth_badge_html(df0)
    st.markdown(badges, unsafe_allow_html=True)
    if "gcp_service_account" not in st.secrets:
        st.error("Google Sheets not configured. Add your service account JSON under [gcp_service_account].")
    else:
        st.caption("Google Sheets linked ✓")

st.caption("Canonical headers: Vital Measurement, Node 1, Node 2, Node 3, Node 4, Node 5, Diagnostic Triage, Actions")

with st.sidebar:
    st.header("❓ Tips")
    st.markdown("""
- **Push (Raw+)** writes your current sheet **plus** new rows generated from your **saved branch overrides** (5/5 parents only), with no duplicates.
- New rows will have **empty** Diagnostic Triage / Actions so you can fill them later.
- Use the **Data Quality** tools to reach 5/5 where needed; Raw+ only expands fully specified parents.
""")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "⬆️ Upload Excel/CSV",
    "📄 Google Sheets Mode",
    "🧪 Fill Diagnostic & Actions",
    "🧬 Symptoms",
    "📜 Push Log"
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
            # Ensure canonical columns (add if missing)
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
                # Drop fully blank placeholder rows (all node cells empty)
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

        st.markdown("#### Preview (first 50 rows)")
        st.dataframe(df_in.head(50), use_container_width=True)

        # Overrides for this sheet (Upload)
        overrides_upload_all = ss_get("branch_overrides_upload", {})
        overrides_upload = overrides_upload_all.get(sheet_name, {})

        # Store with overrides + virtual parents
        store = infer_branch_options_with_overrides(df_in, overrides_upload)
        parents_by_level = compute_virtual_parents(store)

        # Prepare issues
        missing, overspec, incomplete = [], [], []
        for level in range(1, MAX_LEVELS+1):
            for parent in sorted(parents_by_level.get(level, set())):
                key = level_key_tuple(level, parent); opts = store.get(key, [])
                if len(opts)==0: missing.append((level,parent,key))
                elif len(opts)<5: incomplete.append((level,parent,key,opts))
                elif len(opts)>5: overspec.append((level,parent,key,opts))

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
                    st.success("Restored overrides to previous state for this sheet.")
                else:
                    st.info("Last undo snapshot was for a different tab/sheet.")

        # Push settings row (applies to Raw+ pushes)
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

        # Save target defaults per sheet
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
                    non_empty = [normalize_text(x) for x in edit if normalize_text(x)]
                    if len(non_empty) < 5:
                        st.info("Please fill all 5 options.")
                    elif len(set(non_empty)) < 5:
                        st.warning("Duplicate values detected among the 5 options.")
                    st.caption("Enter the 5 options for this parent.")
                    user_fixes[key] = [normalize_text(x) for x in edit]

                    # Save overrides in-session (no direct push here)
                    if st.button("Save 5 branches", key=f"up_save_{key}"):
                        stack = ss_get("undo_stack", [])
                        stack.append({
                            "context": "upload",
                            "sheet": sheet_name,
                            "overrides_sheet_before": overrides_upload.copy()
                        })
                        ss_set("undo_stack", stack)
                        overrides_upload[key] = enforce_k_five(edit)
                        overrides_upload_all[sheet_name] = overrides_upload
                        ss_set("branch_overrides_upload", overrides_upload_all)
                        mark_session_edit(sheet_name, key)
                        st.success("Saved 5 branches for this parent.")

        # --- Symptom left out (<5 options) ---
        if incomplete:
            st.warning(f"Symptom left out (<5 options): {len(incomplete)}")
            for (level,parent,key,opts) in incomplete:
                with st.expander(f"{' > '.join(parent) or '<ROOT>'} — Node {level} — {len(opts)}/5"):
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
                    non_empty = [normalize_text(x) for x in edit if normalize_text(x)]
                    if len(set([x for x in non_empty])) < len(non_empty):
                        st.warning("Duplicate values detected among the non-empty options.")
                    if len(non_empty) < 5:
                        st.info("Fill the remaining boxes so there are exactly 5 options.")
                    user_fixes[key] = [normalize_text(x) for x in edit]

                    if st.button("Save 5 branches", key=f"up_save_{key}"):
                        stack = ss_get("undo_stack", [])
                        stack.append({
                            "context": "upload",
                            "sheet": sheet_name,
                            "overrides_sheet_before": overrides_upload.copy()
                        })
                        ss_set("undo_stack", stack)
                        overrides_upload[key] = enforce_k_five(edit)
                        overrides_upload_all[sheet_name] = overrides_upload
                        ss_set("branch_overrides_upload", overrides_upload_all)
                        mark_session_edit(sheet_name, key)
                        st.success("Saved 5 branches for this parent.")

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
                            "sheet": sheet_name,
                            "overrides_sheet_before": overrides_upload.copy()
                        })
                        ss_set("undo_stack", stack)
                        overrides_upload[key] = enforce_k_five(fix)
                        overrides_upload_all[sheet_name] = overrides_upload
                        ss_set("branch_overrides_upload", overrides_upload_all)
                        mark_session_edit(sheet_name, key)
                        st.success("Saved 5 branches for this parent.")

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

        # Precompute edited keys for this sheet
        edited_keys_for_sheet = set(ss_get("session_edited_keys", {}).get(sheet_name, []))

        if st.button("Push Raw+ (augmented)"):
            if not up_spreadsheet_id or not up_target_tab:
                st.warning("Enter Spreadsheet ID and target tab name.")
            elif not up_confirm and not up_dry:
                st.warning("Please tick the confirmation checkbox before pushing.")
            else:
                merged_overrides = {**overrides_upload, **user_fixes}
                scope_flag = "session" if include_scope_main.endswith("session") else "all"
                df_aug, stats = build_raw_plus(df_in, merged_overrides, scope_flag, edited_keys_for_sheet)

                st.info(f"Delta preview — Generated: **{stats['generated']}**, New added: **{stats['new_added']}**, Duplicates skipped: **{stats['duplicates_skipped']}**, Final total: **{stats['final_total']}**.")
                st.caption(f"Will write **{len(df_aug)} rows × {len(df_aug.columns)} cols** to tab **{up_target_tab}**.")

                if up_dry:
                    st.success("Dry-run complete. No changes written to Google Sheets.")
                    st.dataframe(df_aug.head(50), use_container_width=True)
                    # Optional: download the augmented table
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
                        # Push log
                        log = ss_get("push_log", [])
                        log.append({
                            "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "sheet": sheet_name,
                            "target_tab": up_target_tab,
                            "spreadsheet_id": up_spreadsheet_id,
                            "rows_written": len(df_aug),
                            "new_rows_added": stats["new_added"],
                            "scope": scope_flag
                        })
                        ss_set("push_log", log)
                        # Save target for this sheet
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
                            # Push Raw+ (augmented) for this sheet
                            overrides_all = ss_get(override_root, {})
                            overrides_sheet = overrides_all.get(sheet_cur, {})
                            scope_flag = "all"  # default for this quick push
                            edited_keys_for_sheet = set(ss_get("session_edited_keys", {}).get(sheet_cur, []))
                            df_aug, stats = build_raw_plus(wb[sheet_cur], overrides_sheet, scope_flag, edited_keys_for_sheet)
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
                                    "scope": scope_flag
                                })
                                ss_set("push_log", log)
                                st.success("Changes pushed to Google Sheets (Raw+).")
            else: st.info("Click 'Pick a random incomplete row' to begin.")

# ---------- Symptoms browser & editor ----------
with tab4:
    st.subheader("Symptoms — browse, check consistency, edit child branches, push Raw+, export PDF")

    # Push settings (used by per-parent save/push and Bulk push)
    with st.expander("🔧 Google Sheets Push Settings"):
        sid_current = ss_get("gs_spreadsheet_id", "")
        sid_input = st.text_input("Spreadsheet ID for pushes", value=sid_current, key="sym_sid")
        if sid_input and sid_input != sid_current:
            ss_set("gs_spreadsheet_id", sid_input)
        include_scope_sym = st.radio("Include scope for Raw+", ["All completed parents","Only parents edited this session"], horizontal=True, key="sym_scope")
        st.caption("Raw+ = Source rows + new rows generated from saved branch overrides (5/5 parents only).")
        push_backup = st.checkbox("Create a backup tab before overwrite", value=True, key="sym_push_backup")
        dry_run_sym = st.checkbox("Dry-run (build but don't write to Sheets)", value=False, key="sym_dry_run")

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
                st.success(f"Undid Upload override changes for sheet '{last['sheet']}'.")
            else:
                overrides_all = ss_get(override_root, {})
                overrides_all[last["sheet"]] = last["overrides_sheet_before"]
                ss_set(override_root, overrides_all)
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

        # Build store + parents
        overrides_current = ss_get(override_root, {}).get(sheet, {})
        store = infer_branch_options_with_overrides(df, overrides_current)
        parents_by_level = compute_virtual_parents(store)

        # Data Quality Tools
        if not df.empty and validate_headers(df):
            with st.expander("🧼 Data quality tools (applies to this sheet)"):
                ok_p, total_p = compute_parent_depth_score(df)
                ok_r, total_r = compute_row_path_score(df)
                st.write(f"Parents with 5 children: **{ok_p}/{total_p}**")
                st.write(f"Rows with full path: **{ok_r}/{total_r}**")
                # Inline normalization (optional)
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
                        if col not in df2.columns: continue
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

        # Persistent search
        _pending = st.session_state.pop("sym_search_pending", None)
        if _pending is not None:
            st.session_state["sym_search"] = _pending

        top_cols = st.columns([2,1,1,1,2])
        with top_cols[0]:
            search = st.text_input("Search parent symptom/path", key="sym_search").strip().lower()
        with top_cols[1]:
            if st.button("Next Missing"):
                cand = []
                for pt in sorted(parents_by_level.get(level, set())):
                    key = level_key_tuple(level, pt)
                    if len(store.get(key, [])) == 0:
                        cand.append(pt)
                if cand:
                    path = " > ".join(cand[0]) or "<ROOT>"
                    st.session_state["sym_search_pending"] = path.lower()
                    st.rerun()
        with top_cols[2]:
            if st.button("Next Symptom left out"):
                cand = []
                for pt in sorted(parents_by_level.get(level, set())):
                    key = level_key_tuple(level, pt); n = len(store.get(key, []))
                    if 1 <= n < 5: cand.append(pt)
                if cand:
                    path = " > ".join(cand[0]) or "<ROOT>"
                    st.session_state["sym_search_pending"] = path.lower()
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

        # Inconsistency detection & entries (including virtual parents)
        label_childsets: Dict[Tuple[int,str], set] = {}
        entries = []  # (parent_tuple, children, status)
        for parent_tuple in sorted(parents_by_level.get(level, set())):
            parent_text = " > ".join(parent_tuple)
            if search and (search not in parent_text.lower()):
                continue
            key = level_key_tuple(level, parent_tuple)
            children = store.get(key, [])
            n = len(children)
            if n == 0: status = "No group of symptoms"
            elif n < 5: status = "Symptom left out"
            elif n == 5: status = "OK"
            else: status = "Overspecified"
            entries.append((parent_tuple, children, status))
            last_label = parent_tuple[-1] if parent_tuple else "<ROOT>"
            label_childsets.setdefault((level, last_label), set()).add(tuple(sorted([c for c in children])))

        inconsistent_labels = {k for k, v in label_childsets.items() if len(v) > 1}

        # Sort
        status_rank = {"No group of symptoms":0, "Symptom left out":1, "Overspecified":2, "OK":3}
        if sort_mode.startswith("Problem"):
            entries.sort(key=lambda e: (status_rank[e[2]], e[0]))
        else:
            entries.sort(key=lambda e: e[0])

        # Vocabulary for suggestions (for inline editing)
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

        render_pdf_colA, render_pdf_colB = st.columns([1,3])
        with render_pdf_colA:
            # PDF export for selected Node level
            level_for_pdf = st.selectbox("PDF: Level to export (parents at...)", [1,2,3,4,5], format_func=lambda x: f"Node {x}")
            if st.button("📄 Download Symptoms + Branches (PDF)"):
                parents_list = sorted(parents_by_level.get(level_for_pdf, set()))
                pdf_bytes = build_symptoms_pdf(store, parents_list, level_for_pdf, sheet)
                st.download_button(
                    "Download PDF",
                    data=pdf_bytes,
                    file_name=f"{sheet}_Node{level_for_pdf}_Symptoms.pdf",
                    mime="application/pdf"
                )

        with render_pdf_colB:
            st.caption("Edit child branches and push **Raw+** (augmented) to Google Sheets when ready.")

        # Render entries with inline editing
        for parent_tuple, children, status in entries:
            keyname = level_key_tuple(level, parent_tuple)
            last_label = parent_tuple[-1] if parent_tuple else "<ROOT>"
            inconsistent_flag = (level, last_label) in inconsistent_labels
            subtitle = f"{' > '.join(parent_tuple) or '<ROOT>'} — {status} {'⚠️' if inconsistent_flag else ''}"

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

                # Save in-session (overrides)
                if st.button("Save 5 branches for this parent", key=f"sym_save_{level}_{'__'.join(parent_tuple)}"):
                    fixed = build_final_values()
                    overrides_all = ss_get(override_root, {})
                    overrides_sheet = overrides_all.get(sheet, {}).copy()
                    # undo snapshot
                    stack = ss_get("undo_stack", [])
                    stack.append({
                        "context": "symptoms",
                        "override_root": override_root,
                        "sheet": sheet,
                        "level": level,
                        "parent": parent_tuple,
                        "overrides_sheet_before": overrides_all.get(sheet, {}).copy()
                    })
                    ss_set("undo_stack", stack)
                    overrides_sheet[keyname] = fixed
                    overrides_all[sheet] = overrides_sheet
                    ss_set(override_root, overrides_all)
                    mark_session_edit(sheet, keyname)
                    # update local view
                    store[keyname] = fixed
                    st.success("Saved in-session. (Undo available above)")

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
                    df_aug, stats = build_raw_plus(df, overrides_current, scope_flag, edited_keys_for_sheet)
                    st.info(f"Delta preview — Generated: **{stats['generated']}**, New added: **{stats['new_added']}**, Duplicates skipped: **{stats['duplicates_skipped']}**, Final total: **{stats['final_total']}**.")
                    st.caption(f"Will write **{len(df_aug)} rows × {len(df_aug.columns)} cols** to tab **{target_tab}**.")
                    if dry_run_sym:
                        st.success("Dry-run complete. No changes written to Google Sheets.")
                        st.dataframe(df_aug.head(50), use_container_width=True)
                    else:
                        if ss_get("sym_push_backup", True) and push_backup:
                            backup_name = backup_sheet_copy(sid, target_tab)
                            if backup_name: st.info(f"Backed up current '{target_tab}' to '{backup_name}'.")
                        ok = push_to_google_sheets(sid, target_tab, df_aug)
                        if ok:
                            # push log + save target
                            log = ss_get("push_log", [])
                            log.append({
                                "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                "sheet": sheet,
                                "target_tab": target_tab,
                                "spreadsheet_id": sid,
                                "rows_written": len(df_aug),
                                "new_rows_added": stats["new_added"],
                                "scope": scope_flag
                            })
                            ss_set("push_log", log)
                            saved = ss_get("saved_targets", {})
                            saved.setdefault(sheet, {})
                            saved[sheet]["tab"] = target_tab
                            ss_set("saved_targets", saved)
                            st.success(f"Pushed {len(df_aug)} rows to '{target_tab}'.")
        with colB:
            st.caption("Raw+ respects your Include Scope choice.")

# ---------- Push Log ----------
with tab5:
    st.subheader("Push Log")
    log = ss_get("push_log", [])
    if not log:
        st.info("No pushes recorded this session.")
    else:
        df_log = pd.DataFrame(log)
        st.dataframe(df_log, use_container_width=True)
        # Download log
        csv = df_log.to_csv(index=False).encode("utf-8")
        st.download_button("Download push log (CSV)", data=csv, file_name="push_log.csv", mime="text/csv")
