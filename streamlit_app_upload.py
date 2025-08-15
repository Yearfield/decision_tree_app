import io
import random
from typing import Dict, List, Tuple, Optional, Set
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime

# ============ CONFIG ============
CANON_HEADERS = ["Vital Measurement","Node 1","Node 2","Node 3","Node 4","Node 5","Diagnostic Triage","Actions"]
LEVEL_COLS = ["Node 1","Node 2","Node 3","Node 4","Node 5"]
SHEET_COMPLETED_SUFFIX = " (Completed)"
MAX_LEVELS = 5

st.set_page_config(page_title="Decision Tree Builder v4.1", page_icon="🌳", layout="wide")

# ============ Session helpers ============
def ss_get(key, default):
    if key not in st.session_state:
        st.session_state[key] = default
    return st.session_state[key]

def ss_set(key, value):
    st.session_state[key] = value

# ============ Core helpers ============
def normalize_text(x: str) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()

def validate_headers(df: pd.DataFrame) -> bool:
    return list(df.columns[:len(CANON_HEADERS)]) == CANON_HEADERS

def parent_key_from_row(row: pd.Series, upto_level: int) -> Tuple[str, ...]:
    # Legacy (non-strict) version kept for reference
    return tuple(normalize_text(row[c]) for c in LEVEL_COLS[:upto_level-1] if normalize_text(row[c]) != "")

def parent_key_from_row_strict(row: pd.Series, upto_level: int) -> Optional[Tuple[str, ...]]:
    """
    Return the (level-1)-length parent tuple only if ALL earlier nodes are non-empty.
    If any earlier node is blank, return None (i.e., this row doesn't define a parent context).
    For level=1, returns () for <ROOT>.
    """
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
    """
    Merge observed store with overrides for the selected sheet.
    We don't enforce exactly 5 here; shorter lists show as incomplete.
    """
    base = infer_branch_options(df)
    merged = dict(base)
    for k, v in (overrides or {}).items():
        if isinstance(v, list):
            vals = [normalize_text(x) for x in v]
        else:
            vals = [normalize_text(v)]
        merged[k] = vals
    return merged

def propose_autofill(level: int, parent: Tuple[str, ...], existing_store: Dict[str, List[str]]) -> List[str]:
    """(Kept for reference; not used in Missing/Incomplete sections after your request)."""
    context = parent[:-1] if len(parent) > 0 else tuple()
    level_prefix = f"L{level}|"
    ctx_counts: Dict[Tuple[str, ...], int] = {}
    lvl_counts: Dict[Tuple[str, ...], int] = {}

    def tup5(lst: List[str]) -> Tuple[str, ...]:
        return tuple(lst[:5])

    for key, opts in existing_store.items():
        if not key.startswith(level_prefix):
            continue
        _, path = key.split("|", 1)
        parent_tuple = tuple([] if path == "<ROOT>" else path.split(">"))
        t = tup5([o for o in opts if o != ""])
        if parent_tuple[:-1] == context and len(t) > 0:
            ctx_counts[t] = ctx_counts.get(t, 0) + 1
        if len(t) > 0:
            lvl_counts[t] = lvl_counts.get(t, 0) + 1

    def top_by(d):
        if not d: return None
        return list(sorted(d.items(), key=lambda kv: (-kv[1], kv[0]))[0][0])

    res = top_by(ctx_counts) or top_by(lvl_counts) or []
    return list(res)

def enforce_k_five(opts: List[str]) -> Tuple[List[str], List[str]]:
    """Ensure exactly 5 options (trim or pad with blanks)."""
    clean = [normalize_text(o) for o in opts if normalize_text(o) != ""]
    extras = []
    if len(clean) > 5:
        extras = clean[5:]
        clean = clean[:5]
    elif len(clean) < 5:
        clean = clean + [""]*(5-len(clean))
    return clean, extras

def mark_duplicates(df_completed: pd.DataFrame) -> pd.DataFrame:
    keycols = ["Vital Measurement"] + LEVEL_COLS
    dup_mask = df_completed.duplicated(subset=keycols, keep=False)
    df_completed = df_completed.copy()
    if "Duplicate" not in df_completed.columns:
        df_completed.insert(len(df_completed.columns), "Duplicate", dup_mask)
    else:
        df_completed["Duplicate"] = dup_mask
    return df_completed

def build_completed_sheet(df: pd.DataFrame, user_fixes: Dict[str, List[str]]) -> pd.DataFrame:
    """Expand to full 5-way tree using observed + user-approved options. Merge triage/actions where exact path existed."""
    observed = infer_branch_options(df)
    for k, v in user_fixes.items():
        observed[k] = v
    for k in list(observed.keys()):
        opts5, _ = enforce_k_five(observed[k])
        observed[k] = opts5

    all_vm = sorted(set(normalize_text(x) for x in df["Vital Measurement"] if normalize_text(x) != ""))
    completed_rows = []

    for vm_val in all_vm:
        def dfs(level: int, prefix: Tuple[str, ...]):
            if level > MAX_LEVELS:
                row = {"Vital Measurement": vm_val}
                for i, val in enumerate(prefix, 1):
                    row[f"Node {i}"] = val
                for i in range(1, MAX_LEVELS+1):
                    row.setdefault(f"Node {i}", "")
                row["Diagnostic Triage"] = ""
                row["Actions"] = ""
                completed_rows.append(row); return
            key = level_key_tuple(level, prefix)
            opts = observed.get(key, [])
            if len(opts) != 5 or any(o == "" for o in opts):
                return
            for o in opts:
                dfs(level+1, prefix + (o,))
        dfs(1, tuple())

    if not completed_rows:
        out = pd.DataFrame(columns=CANON_HEADERS + ["Duplicate","Auto-filled"])
    else:
        out = pd.DataFrame(completed_rows, columns=CANON_HEADERS)
        # Merge existing triage/actions where exact path matches
        keycols = ["Vital Measurement"] + LEVEL_COLS
        df_keyed = df.copy()
        for c in CANON_HEADERS:
            if c not in df_keyed.columns:
                df_keyed[c] = ""
        df_keyed = df_keyed[CANON_HEADERS]

        # Compute original keys to mark autofilled paths
        orig_keys = set(tuple(row) for row in df_keyed[keycols].itertuples(index=False, name=None))
        out_keys = [tuple(row) for row in out[keycols].itertuples(index=False, name=None)]
        auto_mask = [k not in orig_keys for k in out_keys]
        out["Auto-filled"] = auto_mask

        out = out.merge(df_keyed, on=keycols, how="left", suffixes=("", "_old"))
        for col in ["Diagnostic Triage","Actions"]:
            out[col] = np.where(out[f"{col}_old"].notna() & (out[f"{col}_old"] != ""), out[f"{col}_old"], out[col])
            out.drop(columns=[f"{col}_old"], inplace=True)

    out = mark_duplicates(out)
    return out

def compute_virtual_parents(store: Dict[str, List[str]]) -> Dict[int, Set[Tuple[str, ...]]]:
    """
    Using the known store (observed + overrides), compute parent contexts for each level
    by expanding children forward. Guarantees <ROOT> at level 1.
    """
    parents_by_level: Dict[int, Set[Tuple[str, ...]]] = {i: set() for i in range(1, MAX_LEVELS+1)}
    parents_by_level[1].add(tuple())  # <ROOT> for Node 1
    for level in range(1, MAX_LEVELS):
        for p in list(parents_by_level[level]):
            key = level_key_tuple(level, p)
            children = store.get(key, [])
            for c in children:
                if c != "":
                    parents_by_level[level+1].add(p + (c,))
    # Also include any explicit parents present in store keys (e.g., only overrides at deeper levels)
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
            # ensure ancestors up to lvl exist too
            for k in range(1, min(lvl, MAX_LEVELS)+1):
                parents_by_level.setdefault(k, set())
                parents_by_level[k].add(tuple(parent_tuple[:k-1]))
    return parents_by_level

# ======== Google Sheets helpers ========
def push_to_google_sheets(spreadsheet_id: str, sheet_name: str, df: pd.DataFrame) -> bool:
    """Overwrite or create target sheet with df using service account in secrets."""
    try:
        import gspread
        from google.oauth2.service_account import Credentials
        sa_info = st.secrets["gcp_service_account"]
        scopes = ["https://www.googleapis.com/auth/spreadsheets","https://www.googleapis.com/auth/drive"]
        creds = Credentials.from_service_account_info(sa_info, scopes=scopes)
        client = gspread.authorize(creds)
        sh = client.open_by_key(spreadsheet_id)
        try:
            ws = sh.worksheet(sheet_name); ws.clear()
        except Exception:
            ws = sh.add_worksheet(title=sheet_name, rows=2000, cols=26)
        df = df.fillna("")
        values = [list(df.columns)] + df.astype(str).values.tolist()
        ws.update(values); return True
    except Exception as e:
        st.error(f"Push to Google Sheets failed: {e}"); return False

def backup_sheet_copy(spreadsheet_id: str, source_sheet: str) -> Optional[str]:
    """Create a backup tab by copying values from source_sheet into a new '(backup YYYY-MM-DD HHMM)' tab. Returns backup name or None."""
    try:
        import gspread
        from google.oauth2.service_account import Credentials
        sa_info = st.secrets["gcp_service_account"]
        scopes = ["https://www.googleapis.com/auth/spreadsheets","https://www.googleapis.com/auth/drive"]
        creds = Credentials.from_service_account_info(sa_info, scopes=scopes)
        client = gspread.authorize(creds)
        sh = client.open_by_key(spreadsheet_id)
        try:
            ws = sh.worksheet(source_sheet)
        except Exception:
            return None  # nothing to back up
        values = ws.get_all_values()
        ts = datetime.now().strftime("%Y-%m-%d %H%M")
        backup_title_full = f"{source_sheet} (backup {ts})"
        backup_title = backup_title_full[:31]  # Sheets tab name limit
        ws_bak = sh.add_worksheet(title=backup_title, rows=max(len(values),100), cols=max(len(values[0]) if values else 8, 8))
        if values:
            ws_bak.update(values)
        return backup_title
    except Exception as e:
        st.error(f"Backup failed: {e}")
        return None

# ======== Pretty message for branch issues ========
def describe_branch(level: int, parent: Tuple[str, ...], have_n: int, label: str) -> str:
    parts = []
    for i, val in enumerate(parent, start=1):
        parts.append(f"Node {i} '{val}'")
    ctx = ""
    if parts:
        under = parts[-1]
        froms = ", ".join(parts[:-1])
        ctx = f" under {under}" + (f" (from {froms})" if froms else "")
    else:
        ctx = " under <ROOT>"
    return f"Node {level}{ctx} has {have_n}/5 options. [{label}]"

# ======== Styling for autofilled preview ========
def style_completed(df: pd.DataFrame) -> "pd.io.formats.style.Styler":
    if "Auto-filled" not in df.columns:
        return df.style
    def highlight_row(row):
        return ["background-color: #eaffea" if row.get("Auto-filled", False) else "" for _ in row]
    return df.style.apply(lambda r: highlight_row(r), axis=1)

# ======== Progress helpers ========
def compute_branch_progress(df: pd.DataFrame) -> Tuple[int, int]:
    store = infer_branch_options(df)
    total = 0
    ok = 0
    for level in range(1, MAX_LEVELS+1):
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

def compute_outcome_progress(df: pd.DataFrame) -> Tuple[int, int]:
    if df.empty:
        return 0, 0
    tri = df["Diagnostic Triage"].map(normalize_text)
    act = df["Actions"].map(normalize_text)
    done = ((tri != "") & (act != "")).sum()
    return int(done), int(len(df))

# ======== Data quality helpers ========
def build_vocabulary(df: pd.DataFrame) -> List[str]:
    vocab = set()
    for col in LEVEL_COLS:
        for x in df[col].dropna().astype(str):
            x = x.strip()
            if x:
                vocab.add(x)
    return sorted(vocab)

def normalize_label(s: str, case_mode: str) -> str:
    s = s.strip()
    if case_mode == "lower":
        return s.lower()
    if case_mode == "UPPER":
        return s.upper()
    if case_mode == "Title":
        return s.title()
    return s  # None

def apply_synonyms(s: str, syn_map: Dict[str,str]) -> str:
    return syn_map.get(s, s)

def normalize_sheet_df(df: pd.DataFrame, case_mode: str, syn_map: Dict[str,str]) -> pd.DataFrame:
    df2 = df.copy()
    for col in ["Vital Measurement"] + LEVEL_COLS:
        if col not in df2.columns:
            continue
        df2[col] = df2[col].astype(str).map(lambda v: apply_synonyms(normalize_label(v, case_mode), syn_map) if v else v).map(normalize_text)
    for col in ["Diagnostic Triage","Actions"]:
        if col in df2.columns:
            df2[col] = df2[col].map(normalize_text)
    return df2

def parse_synonym_map(text: str) -> Dict[str,str]:
    mapping = {}
    for line in text.splitlines():
        if "=>" in line:
            a,b = line.split("=>",1)
            a = a.strip()
            b = b.strip()
            if a and b:
                mapping[a] = b
    return mapping

# ======== UI badges / legend ========
def render_badge_legend():
    st.markdown(
        "<div style='font-size:0.95rem'>"
        "✅ <b>OK</b> &nbsp;&nbsp; "
        "🟦 <b>No group of symptoms</b> &nbsp;&nbsp; "
        "🟨 <b>Symptom left out</b> &nbsp;&nbsp; "
        "⛔ <b>Overspecified</b> &nbsp;&nbsp; "
        "⚠️ <b>Inconsistent</b>"
        "</div>",
        unsafe_allow_html=True
    )

def status_badge(status: str, inconsistent: bool=False) -> str:
    base = {"OK":"✅","No group of symptoms":"🟦","Symptom left out":"🟨","Overspecified":"⛔"}.get(status,"")
    inc = " ⚠️" if inconsistent else ""
    return f"{base} {status}{inc}"

# ============ UI ============
st.title("🌳 Decision Tree Builder — v4.1 (virtual parents)")
st.caption("Canonical headers: Vital Measurement, Node 1, Node 2, Node 3, Node 4, Node 5, Diagnostic Triage, Actions")

with st.sidebar:
    st.header("❓ Help & Legend")
    render_badge_legend()
    st.markdown("""
**What’s new in v4.1**
- Overrides now create **virtual parents** immediately, so adding Node 1 options makes Node 2 contexts appear to fill.
- **Strict parent detection**: placeholder rows won’t produce fake deeper-level parents.
- Random action filler ignores placeholder rows; only fully specified paths are eligible.
""")

tab1, tab2, tab3, tab4 = st.tabs(["⬆️ Upload Excel/CSV", "📄 Google Sheets Mode", "🧪 Fill Diagnostic & Actions", "🧬 Symptoms"])

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
            df = pd.read_csv(file); sheets = {"Sheet1": df}
        else:
            xls = pd.ExcelFile(file); sheets = {name: xls.parse(name) for name in xls.sheet_names}
        # Drop fully blank placeholder rows (VM + Nodes all blank)
        for nm in list(sheets.keys()):
            dfx = sheets[nm].copy()
            dfx.columns = [normalize_text(c) for c in dfx.columns]
            # pad missing canonical cols
            for c in CANON_HEADERS:
                if c not in dfx.columns:
                    dfx[c] = ""
            dfx = dfx[CANON_HEADERS]
            node_block = ["Vital Measurement"] + LEVEL_COLS
            dfx = dfx[~dfx[node_block].apply(lambda r: all(normalize_text(v) == "" for v in r), axis=1)].copy()
            sheets[nm] = dfx
        ss_set("upload_workbook", {k: v.copy() for k, v in sheets.items()})
        ss_set("upload_filename", file.name)
        st.write(f"Found {len(sheets)} sheet(s). Choose one to process:")
        sheet_name = st.selectbox("Sheet", list(sheets.keys()))
        df_in = sheets[sheet_name].copy()
        df_in.columns = [normalize_text(c) for c in df_in.columns]
        if not validate_headers(df_in):
            st.error("Headers mismatch. First 8 columns must be: " + ", ".join(CANON_HEADERS)); st.stop()
        for c in CANON_HEADERS: df_in[c] = df_in[c].map(normalize_text)

        ok, total = compute_branch_progress(df_in)
        done, total_rows = compute_outcome_progress(df_in)
        st.markdown("#### Progress")
        st.write(f"Branch definitions complete: **{ok}/{total}**"); st.progress(0 if total==0 else ok/total)
        st.write(f"Outcomes filled (Diagnostic Triage + Actions): **{done}/{total_rows}**"); st.progress(0 if total_rows==0 else done/total_rows)

        st.markdown("#### Preview (first 50 rows)")
        st.dataframe(df_in.head(50), use_container_width=True)

        # Merge overrides to create virtual parents
        overrides_upload = ss_get("branch_overrides_upload", {}).get(sheet_name, {})
        store = infer_branch_options_with_overrides(df_in, overrides_upload)
        parents_by_level = compute_virtual_parents(store)

        missing, overspec, incomplete = [], [], []
        for level in range(1, MAX_LEVELS+1):
            for parent in sorted(parents_by_level.get(level, set())):
                key = level_key_tuple(level, parent); opts = store.get(key, [])
                if len(opts)==0: missing.append((level,parent,key))
                elif len(opts)<5: incomplete.append((level,parent,key,opts))
                elif len(opts)>5: overspec.append((level,parent,key,opts))

        st.markdown("#### Review & Complete")
        render_badge_legend()  # badges above list

        user_fixes: Dict[str, List[str]] = {}

        # --- No group of symptoms (0 options) ---
        if missing:
            st.warning(f"No group of symptoms: {len(missing)}")
            for (level, parent, key) in missing:
                with st.expander(describe_branch(level, parent, 0, "No group of symptoms")):
                    cols = st.columns(5)
                    # Show 5 empty boxes; you fill them
                    edit = [
                        cols[i].text_input(
                            f"Option {i+1}",
                            value="",
                            placeholder="Enter option",
                            key=f"miss_{key}_{i}",
                        ) for i in range(5)
                    ]
                    st.caption("Enter the 5 options for this parent.")
                    user_fixes[key] = [normalize_text(x) for x in edit]

        # --- Symptom left out (<5 options) ---
        if incomplete:
            st.warning(f"Symptom left out (<5 options): {len(incomplete)}")
            for (level,parent,key,opts) in incomplete:
                with st.expander(describe_branch(level, parent, len(opts), "Symptom left out")):
                    padded = (opts[:5] + [""] * (5 - len(opts))) if len(opts) < 5 else opts[:5]
                    cols = st.columns(5)
                    edit = [
                        cols[i].text_input(
                            f"Option {i+1}",
                            value=padded[i],
                            placeholder="Enter option",
                            key=f"incomp_{key}_{i}",
                        ) for i in range(5)
                    ]
                    st.caption("Fill the remaining boxes so there are exactly 5 options.")
                    user_fixes[key] = [normalize_text(x) for x in edit]

        if overspec:
            st.error(f"Overspecified branches (>5 options): {len(overspec)} — choose exactly 5")
            for (level,parent,key,opts) in overspec:
                with st.expander(describe_branch(level, parent, len(opts), "Overspecified")):
                    chosen = st.multiselect("Select 5 options", opts, default=opts[:5], key=f"over_{key}")
                    if len(chosen)!=5: st.warning("Please select exactly 5.")
                    user_fixes[key] = [normalize_text(x) for x in chosen] if len(chosen)==5 else [normalize_text(x) for x in opts[:5]]

        st.markdown("---")
        completed = None
        if st.button("Build Completed Sheet"):
            overrides_all = ss_get("branch_overrides_upload", {})
            overrides_sheet = overrides_all.get(sheet_name, {})
            merged = {**overrides_sheet, **user_fixes}
            completed = build_completed_sheet(df_in, merged)
            st.success(f"Completed rows: {len(completed)}")
            styled = style_completed(completed)
            st.dataframe(styled, use_container_width=True)
            completed_map = ss_get("upload_completed", {}); completed_map[sheet_name] = completed.copy(); ss_set("upload_completed", completed_map)
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
                wb = ss_get("upload_workbook", {sheet_name: df_in})
                for nm, d in wb.items():
                    d.to_excel(writer, index=False, sheet_name=nm[:31] or "Sheet1")
                out_name = f"{sheet_name}{SHEET_COMPLETED_SUFFIX}"; completed.to_excel(writer, index=False, sheet_name=out_name[:31])
            st.download_button("Download updated workbook", data=buffer.getvalue(), file_name="decision_tree_completed.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

        st.markdown("#### Push to Google Sheets (overwrite)")
        st.caption("Requires service account credentials in app secrets under [gcp_service_account].")
        up_spreadsheet_id = st.text_input("Spreadsheet ID (Upload tab)", key="up_sheet_id")
        up_target_tab = st.text_input("Target tab name", value=f"{sheet_name}{SHEET_COMPLETED_SUFFIX}", key="up_target_tab")
        up_confirm = st.checkbox("I confirm I want to overwrite the target tab.", key="up_confirm")
        up_backup = st.checkbox("Create a backup tab before overwriting", value=True, key="up_backup")
        if st.button("Push (Upload tab)"):
            if completed is None: st.warning("Build the Completed sheet first.")
            elif not up_spreadsheet_id or not up_target_tab: st.warning("Enter Spreadsheet ID and target tab name.")
            elif not up_confirm: st.warning("Please tick the confirmation checkbox before pushing.")
            else:
                if up_backup:
                    backup_name = backup_sheet_copy(up_spreadsheet_id, up_target_tab)
                    if backup_name: st.info(f"Backed up current '{up_target_tab}' to '{backup_name}'.")
                ok = push_to_google_sheets(up_spreadsheet_id, up_target_tab, completed)
                if ok: st.success("Pushed to Google Sheets.")

# ---------- Google Sheets mode ----------
with tab2:
    st.subheader("Google Sheets (optional)")
    st.caption("Add a service account JSON to app secrets under key 'gcp_service_account'.")
    spreadsheet_id = st.text_input("Spreadsheet ID", key="gs_id")
    load_mode = st.radio("Load scope", ["Single sheet", "All sheets"], horizontal=True)
    sheet_name_g = st.text_input("Sheet name to process (for single sheet)", value="BP")
    run_btn = st.button("Load")
    def get_gsheet_client():
        import gspread
        from google.oauth2.service_account import Credentials
        sa_info = st.secrets["gcp_service_account"]
        scopes = ["https://www.googleapis.com/auth/spreadsheets","https://www.googleapis.com/auth/drive"]
        creds = Credentials.from_service_account_info(sa_info, scopes=scopes)
        return gspread.authorize(creds)
    st.markdown("**Create a new sheet (Google Sheets)**")
    new_gs_sheet = st.text_input("New sheet name (Google Sheets)", key="new_gs_sheet")
    if st.button("Create sheet (Google Sheets)"):
        try:
            client = get_gsheet_client(); sh = client.open_by_key(spreadsheet_id)
            ws = sh.add_worksheet(title=new_gs_sheet, rows=100, cols=26); ws.update([CANON_HEADERS])
            st.success(f"Created new tab '{new_gs_sheet}' with canonical headers.")
        except Exception as e:
            st.error(f"Google Sheets error: {e}")
    if run_btn:
        try:
            client = get_gsheet_client(); sh = client.open_by_key(spreadsheet_id)
            if load_mode == "Single sheet":
                ws = sh.worksheet(sheet_name_g); values = ws.get_all_values()
                if not values: st.error("Selected sheet is empty."); st.stop()
                header = values[0]; rows = values[1:]
                df_g = pd.DataFrame(rows, columns=header)
                df_g.columns = [normalize_text(c) for c in df_g.columns]
                if not validate_headers(df_g): st.error("Sheet does not match canonical headers."); st.stop()
                for c in CANON_HEADERS: df_g[c] = df_g[c].map(normalize_text)
                # Drop fully blank placeholder rows
                node_block = ["Vital Measurement"] + LEVEL_COLS
                df_g = df_g[~df_g[node_block].apply(lambda r: all(v == "" for v in r), axis=1)].copy()

                st.markdown("#### Progress")
                ok, total = compute_branch_progress(df_g); done, total_rows = compute_outcome_progress(df_g)
                st.write(f"Branch definitions complete: **{ok}/{total}**"); st.progress(0 if total==0 else ok/total)
                st.write(f"Outcomes filled (Diagnostic Triage + Actions): **{done}/{total_rows}**"); st.progress(0 if total_rows==0 else done/total_rows)
                st.markdown("#### Preview (first 50 rows)"); st.dataframe(df_g.head(50), use_container_width=True)

                # Merge overrides to create virtual parents
                overrides_gs = ss_get("branch_overrides_gs", {}).get(sheet_name_g, {})
                store = infer_branch_options_with_overrides(df_g, overrides_gs)
                parents_by_level = compute_virtual_parents(store)

                missing, overspec, incomplete = [], [], []
                for level in range(1, MAX_LEVELS+1):
                    for parent in sorted(parents_by_level.get(level, set())):
                        key = level_key_tuple(level, parent); opts = store.get(key, [])
                        if len(opts)==0: missing.append((level,parent,key))
                        elif len(opts)<5: incomplete.append((level,parent,key,opts))
                        elif len(opts)>5: overspec.append((level,parent,key,opts))

                st.markdown("#### Review & Complete")
                render_badge_legend()

                user_fixes: Dict[str, List[str]] = {}
                # --- No group of symptoms (0 options) ---
                if missing:
                    st.warning(f"No group of symptoms: {len(missing)}")
                    for (level,parent,key) in missing:
                        with st.expander(describe_branch(level, parent, 0, "No group of symptoms")):
                            cols = st.columns(5)
                            edit = [
                                cols[i].text_input(
                                    f"Option {i+1}",
                                    value="",
                                    placeholder="Enter option",
                                    key=f"g_miss_{key}_{i}",
                                ) for i in range(5)
                            ]
                            st.caption("Enter the 5 options for this parent.")
                            user_fixes[key] = [normalize_text(x) for x in edit]
                # --- Symptom left out (<5 options) ---
                if incomplete:
                    st.warning(f"Symptom left out (<5 options): {len(incomplete)}")
                    for (level,parent,key,opts) in incomplete:
                        with st.expander(describe_branch(level, parent, len(opts), "Symptom left out")):
                            padded = (opts[:5] + [""] * (5 - len(opts))) if len(opts) < 5 else opts[:5]
                            cols = st.columns(5)
                            edit = [
                                cols[i].text_input(
                                    f"Option {i+1}",
                                    value=padded[i],
                                    placeholder="Enter option",
                                    key=f"g_incomp_{key}_{i}",
                                ) for i in range(5)
                            ]
                            st.caption("Fill the remaining boxes so there are exactly 5 options.")
                            user_fixes[key] = [normalize_text(x) for x in edit]
                if overspec:
                    st.error(f"Overspecified branches (>5 options): {len(overspec)} — choose exactly 5")
                    for (level,parent,key,opts) in overspec:
                        with st.expander(describe_branch(level, parent, len(opts), "Overspecified")):
                            chosen = st.multiselect("Select 5 options", opts, default=opts[:5], key=f"g_over_{key}")
                            if len(chosen)!=5: st.warning("Please select exactly 5.")
                            user_fixes[key] = [normalize_text(x) for x in chosen] if len(chosen)==5 else [normalize_text(x) for x in opts[:5]]

                completed_g = None
                if st.button("Build Completed (Sheets)"):
                    # merge current overrides + fixes
                    overrides_all = ss_get("branch_overrides_gs", {})
                    overrides_sheet = overrides_all.get(sheet_name_g, {})
                    merged = {**overrides_sheet, **user_fixes}
                    completed_g = build_completed_sheet(df_g, merged)
                    st.success(f"Completed rows: {len(completed_g)}"); st.dataframe(completed_g.head(50), use_container_width=True)
                    buffer = io.BytesIO()
                    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
                        df_g.to_excel(writer, index=False, sheet_name=sheet_name_g)
                        completed_g.to_excel(writer, index=False, sheet_name=f"{sheet_name_g}{SHEET_COMPLETED_SUFFIX}")
                    st.download_button("Download updated workbook", data=buffer.getvalue(), file_name=f"{sheet_name_g}_completed.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

                st.markdown("#### Push to Google Sheets (overwrite)")
                st.caption("Uses the same Spreadsheet ID; choose a target tab to overwrite or create.")
                target_tab = st.text_input("Target tab name", value=f"{sheet_name_g}{SHEET_COMPLETED_SUFFIX}", key="g_target_tab")
                if st.button("Push (Sheets tab)"):
                    if completed_g is None: st.warning("Build the Completed sheet first.")
                    elif not spreadsheet_id or not target_tab: st.warning("Enter Spreadsheet ID and target tab name.")
                    else:
                        ok = push_to_google_sheets(spreadsheet_id, target_tab, completed_g)
                        if ok: st.success("Pushed to Google Sheets.")

            else:
                wb = {}
                for ws in sh.worksheets():
                    values = ws.get_all_values()
                    if not values: df_g = pd.DataFrame(columns=CANON_HEADERS)
                    else:
                        header = values[0]; rows = values[1:]
                        df_g = pd.DataFrame(rows, columns=header)
                        df_g.columns = [normalize_text(c) for c in df_g.columns]
                        if validate_headers(df_g):
                            for c in CANON_HEADERS: df_g[c] = df_g[c].map(normalize_text)
                            # Drop fully blank placeholder rows
                            node_block = ["Vital Measurement"] + LEVEL_COLS
                            df_g = df_g[~df_g[node_block].apply(lambda r: all(v == "" for v in r), axis=1)].copy()
                        else:
                            if df_g.empty: df_g = pd.DataFrame(columns=CANON_HEADERS)
                            else: st.warning(f"Sheet '{ws.title}' does not match canonical headers; loaded raw values.")
                    wb[ws.title] = df_g
                ss_set("gs_workbook", wb); ss_set("gs_spreadsheet_id", spreadsheet_id)
                st.success(f"Loaded {len(wb)} Sheets from Google Sheets. Use the '🧪 Fill Diagnostic & Actions' and '🧬 Symptoms' tabs.")
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
            wb = ss_get("upload_workbook", {}); where = "upload"
        else:
            wb = ss_get("gs_workbook", {}); where = "gs"
        selected_sheets = st.multiselect("Limit to sheets", list(wb.keys()), default=list(wb.keys()))
        candidates = []
        for nm in selected_sheets:
            df = wb[nm]
            if not validate_headers(df): continue
            tri_empty = df["Diagnostic Triage"].map(normalize_text) == ""
            act_empty = df["Actions"].map(normalize_text) == ""
            path_complete = df[LEVEL_COLS].applymap(normalize_text).ne("").all(axis=1)
            mask = path_complete & tri_empty & act_empty
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
                            ok = push_to_google_sheets(sid, sheet_cur, wb[sheet_cur])
                            if ok: st.success("Changes pushed to Google Sheets.")
            else: st.info("Click 'Pick a random incomplete row' to begin.")

# ---------- Symptoms browser & editor ----------
with tab4:
    st.subheader("Symptoms — browse, check consistency, and edit child branches")

    # Undo stack controls (editing safety)
    if st.button("↩️ Undo last branch edit (session)"):
        stack = ss_get("undo_stack", [])
        if not stack:
            st.info("Nothing to undo.")
        else:
            last = stack.pop()
            ss_set("undo_stack", stack)
            # Restore overrides
            override_root = last["override_root"]
            overrides_all = ss_get(override_root, {})
            overrides_all[last["sheet"]] = last["overrides_sheet_before"]
            ss_set(override_root, overrides_all)
            st.success(f"Undid last edit on sheet '{last['sheet']}' (Node {last['level']}, parent '{' > '.join(last['parent']) or '<ROOT>'}').")

    sources = []
    if ss_get("upload_workbook", {}):
        sources.append("Upload workbook")
    if ss_get("gs_workbook", {}):
        sources.append("Google Sheets workbook")

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

        if df.empty or not validate_headers(df):
            st.warning("Selected sheet is empty or headers are not canonical.")
        else:
            # ----- Data Quality Tools (sheet-level) -----
            with st.expander("🧼 Data quality tools (applies to this sheet)"):
                case_mode = st.selectbox("Case normalization", ["None","Title","lower","UPPER"], index=0)
                syn_text = st.text_area("Synonym map (one per line: A => B)", key="syn_map_text")
                if st.button("Normalize sheet now"):
                    syn_map = parse_synonym_map(syn_text)
                    df_norm = normalize_sheet_df(df, case_mode if case_mode!="None" else "None", syn_map)
                    wb[sheet] = df_norm
                    if source == "Upload workbook":
                        ss_set("upload_workbook", wb)
                        st.success("Sheet normalized in-session. Remember to download to save locally.")
                    else:
                        sid = ss_get("gs_spreadsheet_id","")
                        if not sid:
                            st.error("Missing Spreadsheet ID in session. Reload in the Google Sheets tab.")
                        else:
                            ok = push_to_google_sheets(sid, sheet, df_norm)
                            if ok:
                                st.success("Normalized sheet pushed to Google Sheets.")

            # Build merged store and virtual parents
            overrides_current = ss_get(override_root, {}).get(sheet, {})
            store = infer_branch_options_with_overrides(df, overrides_current)
            parents_by_level = compute_virtual_parents(store)

            # Controls
            level = st.selectbox("Level to inspect (child options of...)", [1,2,3,4,5], format_func=lambda x: f"Node {x}", key="sym_level")

            # Apply deferred search change before creating the widget
            _pending = st.session_state.pop("sym_search_pending", None)
            if _pending is not None:
                st.session_state["sym_search"] = _pending

            search = st.text_input("Search parent symptom/path", key="sym_search").strip().lower()
            compact = st.checkbox("Compact edit mode (mobile-friendly)", value=True)
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
                if n == 0:
                    status = "No group of symptoms"
                elif n < 5:
                    status = "Symptom left out"
                elif n == 5:
                    status = "OK"
                else:
                    status = "Overspecified"
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

            # Legend + Next best action
            render_badge_legend()
            if entries:
                # Next best action chooses first by severity, then prioritize inconsistent ones.
                def score(entry):
                    parent_tuple, children, status = entry
                    inc = ((level, (parent_tuple[-1] if parent_tuple else "<ROOT>")) in inconsistent_labels)
                    return (status_rank[status], 0 if inc else 1, len(parent_tuple))  # severity, inconsistent first, shallower first
                nba = sorted(entries, key=score)[0]
                nba_path = " > ".join(nba[0]) or "<ROOT>"
                st.info(f"Next best action suggestion: **{status_badge(nba[2], ((level, (nba[0][-1] if nba[0] else '<ROOT>'))) in inconsistent_labels)}** at **{nba_path}**")
                if st.button("Jump to next best action"):
                    # Defer change so we can set search before the widget is recreated
                    st.session_state["sym_search_pending"] = nba_path.lower()
                    st.rerun()

            st.markdown(f"#### {len(entries)} parent contexts at Node {level}")

            # Vocabulary for suggestions
            vocab = build_vocabulary(df)
            vocab_opts = ["(pick suggestion)"] + vocab

            # Render entries
            for parent_tuple, children, status in entries:
                last_label = parent_tuple[-1] if parent_tuple else "<ROOT>"
                inconsistent_flag = (level, last_label) in inconsistent_labels
                subtitle = f"{' > '.join(parent_tuple) or '<ROOT>'} — {status_badge(status, inconsistent_flag)}"
                with st.expander(subtitle):
                    # Inputs
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
                        vals, _ = enforce_k_five(vals); return vals

                    # Save
                    if st.button("Save 5 branches for this parent", key=f"sym_save_{level}_{'__'.join(parent_tuple)}"):
                        fixed = build_final_values()
                        overrides_all = ss_get(override_root, {})
                        overrides_sheet = overrides_all.get(sheet, {}).copy()
                        keyname = level_key_tuple(level, parent_tuple)

                        # push to undo stack BEFORE change
                        stack = ss_get("undo_stack", [])
                        stack.append({
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
                        st.success("Saved. These 5 options will be used when building the Completed sheet. (Undo available above)")

                    # Bulk apply — use parents_by_level to include virtual contexts too
                    if parent_tuple:
                        last_label = parent_tuple[-1]
                        if st.button(f"Apply these 5 to ALL contexts with label '{last_label}' at Node {level}", key=f"sym_bulk_{level}_{'__'.join(parent_tuple)}"):
                            fixed = build_final_values()
                            overrides_all = ss_get(override_root, {})
                            overrides_sheet = overrides_all.get(sheet, {}).copy()

                            # push to undo stack BEFORE change
                            stack = ss_get("undo_stack", [])
                            stack.append({
                                "override_root": override_root,
                                "sheet": sheet,
                                "level": level,
                                "parent": parent_tuple,
                                "overrides_sheet_before": overrides_all.get(sheet, {}).copy()
                            })
                            ss_set("undo_stack", stack)

                            for pt in parents_by_level.get(level, set()):
                                if pt and pt[-1] == last_label:
                                    overrides_sheet[level_key_tuple(level, pt)] = fixed
                            overrides_all[sheet] = overrides_sheet
                            ss_set(override_root, overrides_all)
                            st.success(f"Applied to all Node {level} contexts with label '{last_label}'. (Undo available above)")

            # Build & Push section for Symptoms tab (optional)
            st.markdown("---")
            st.markdown("#### Build / Push from Symptoms")
            overrides_current = ss_get(override_root, {}).get(sheet, {})
            if st.button("Build Completed (with current overrides)"):
                completed_now = build_completed_sheet(df, overrides_current)
                st.success(f"Completed rows: {len(completed_now)}")
                st.dataframe(style_completed(completed_now), use_container_width=True)
                ss_set("sym_completed_preview", completed_now)

            target_tab = st.text_input("Target tab to overwrite/create", value=f"{sheet}{SHEET_COMPLETED_SUFFIX}", key=f"sym_target_{sheet}")
            sym_confirm = st.checkbox("I confirm I want to overwrite the target tab.", key=f"sym_confirm_{sheet}")
            sym_backup = st.checkbox("Create a backup tab before overwriting", value=True, key=f"sym_backup_{sheet}")
            if st.button("Push completed to Google Sheets now"):
                sid = ss_get("gs_spreadsheet_id", "")
                completed_now = ss_get("sym_completed_preview", None)
                if not sid:
                    st.error("Missing Spreadsheet ID in session. Load sheets in the Google Sheets tab first.")
                elif completed_now is None:
                    st.error("Please build the Completed preview first.")
                elif not sym_confirm:
                    st.warning("Please tick the confirmation checkbox before pushing.")
                else:
                    if sym_backup:
                        backup_name = backup_sheet_copy(sid, target_tab)
                        if backup_name: st.info(f"Backed up current '{target_tab}' to '{backup_name}'.")
                    ok = push_to_google_sheets(sid, target_tab, completed_now)
                    if ok:
                        st.success("Pushed to Google Sheets.")
