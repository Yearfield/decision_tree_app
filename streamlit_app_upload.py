import io
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np
import streamlit as st

# ============ CONFIG ============
CANON_HEADERS = ["Vital Measurement","Node 1","Node 2","Node 3","Node 4","Node 5","Diagnostic Triage","Actions"]
LEVEL_COLS = ["Node 1","Node 2","Node 3","Node 4","Node 5"]
SHEET_COMPLETED_SUFFIX = " (Completed)"
MAX_LEVELS = 5

st.set_page_config(page_title="Decision Tree Builder (Upload + Sheets)", page_icon="🌳", layout="wide")

# ============ Helpers ============

def normalize_text(x: str) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()

def validate_headers(df: pd.DataFrame) -> bool:
    return list(df.columns[:len(CANON_HEADERS)]) == CANON_HEADERS

def parent_key_from_row(row: pd.Series, upto_level: int) -> Tuple[str, ...]:
    return tuple(normalize_text(row[c]) for c in LEVEL_COLS[:upto_level-1] if normalize_text(row[c]) != "")

def level_key_tuple(level: int, parent: Tuple[str, ...]) -> str:
    return f"L{level}|" + (">".join(parent) if parent else "<ROOT>")

def infer_branch_options(df: pd.DataFrame) -> Dict[str, List[str]]:
    store: Dict[str, List[str]] = {}
    for level in range(1, MAX_LEVELS+1):
        parent_to_children: Dict[Tuple[str, ...], List[str]] = {}
        for _, row in df.iterrows():
            child = normalize_text(row[LEVEL_COLS[level-1]])
            if child == "":
                continue
            parent = parent_key_from_row(row, level)
            parent_to_children.setdefault(parent, [])
            parent_to_children[parent].append(child)
        uniq_store = {}
        for parent, children in parent_to_children.items():
            uniq = []
            seen = set()
            for c in children:
                if c not in seen:
                    seen.add(c); uniq.append(c)
            uniq_store[parent] = uniq
        for parent, opts in uniq_store.items():
            store[level_key_tuple(level, parent)] = opts
    return store

def propose_autofill(level: int, parent: Tuple[str, ...], existing_store: Dict[str, List[str]]) -> List[str]:
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
        t = tup5(opts)
        if parent_tuple[:-1] == context:
            ctx_counts[t] = ctx_counts.get(t, 0) + 1
        lvl_counts[t] = lvl_counts.get(t, 0) + 1

    def top_by(d):
        if not d: return None
        return list(sorted(d.items(), key=lambda kv: (-kv[1], kv[0]))[0][0])

    return top_by(ctx_counts) or top_by(lvl_counts) or []

def enforce_k_five(opts: List[str]) -> Tuple[List[str], List[str]]:
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
        out = pd.DataFrame(columns=CANON_HEADERS + ["Duplicate"])
    else:
        out = pd.DataFrame(completed_rows, columns=CANON_HEADERS[:-2] + ["Diagnostic Triage","Actions"])
        keycols = ["Vital Measurement"] + LEVEL_COLS
        df_keyed = df.copy()
        for c in CANON_HEADERS:
            if c not in df_keyed.columns:
                df_keyed[c] = ""
        df_keyed = df_keyed[CANON_HEADERS]
        out = out.merge(df_keyed, on=keycols, how="left", suffixes=("", "_old"))
        for col in ["Diagnostic Triage","Actions"]:
            out[col] = np.where(out[f"{col}_old"].notna() & (out[f"{col}_old"] != ""), out[f"{col}_old"], out[col])
            out.drop(columns=[f"{col}_old"], inplace=True)
    out = mark_duplicates(out); return out

def push_to_google_sheets(spreadsheet_id: str, sheet_name: str, df: pd.DataFrame) -> bool:
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
            ws = sh.add_worksheet(title=sheet_name, rows=100, cols=26)
        df = df.fillna("")
        values = [list(df.columns)] + df.astype(str).values.tolist()
        ws.update(values); return True
    except Exception as e:
        st.error(f"Push to Google Sheets failed: {e}"); return False

# ============ UI ============
st.title("🌳 Decision Tree Builder — Upload + Google Sheets (Force 5 options)")
st.caption("Canonical headers: Vital Measurement, Node 1, Node 2, Node 3, Node 4, Node 5, Diagnostic Triage, Actions")

tab1, tab2 = st.tabs(["⬆️ Upload Excel/CSV", "📄 Google Sheets Mode"])

# ---------- Upload mode ----------
with tab1:
    st.subheader("Upload your workbook")
    file = st.file_uploader("Upload XLSX or CSV", type=["xlsx","xls","csv"])
    if file is not None:
        if file.name.lower().endswith(".csv"):
            df = pd.read_csv(file); sheets = {"Sheet1": df}
        else:
            xls = pd.ExcelFile(file); sheets = {name: xls.parse(name) for name in xls.sheet_names}
        st.write(f"Found {len(sheets)} sheet(s). Choose one:")
        sheet_name = st.selectbox("Sheet", list(sheets.keys()))
        df_in = sheets[sheet_name].copy()
        df_in.columns = [normalize_text(c) for c in df_in.columns]
        if not validate_headers(df_in):
            st.error("Headers mismatch. First 8 columns must be: " + ", ".join(CANON_HEADERS)); st.stop()
        for c in CANON_HEADERS: df_in[c] = df_in[c].map(normalize_text)
        st.markdown("#### Preview (first 50 rows)"); st.dataframe(df_in.head(50), use_container_width=True)
        store = infer_branch_options(df_in)
        missing, overspec, incomplete = [], [], []
        for level in range(1, MAX_LEVELS+1):
            parents = set(parent_key_from_row(row, level) for _, row in df_in.iterrows())
            for parent in sorted(parents):
                key = level_key_tuple(level, parent); opts = store.get(key, [])
                if len(opts)==0: missing.append((level,parent,key))
                elif len(opts)<5: incomplete.append((level,parent,key,opts))
                elif len(opts)>5: overspec.append((level,parent,key,opts))
        st.markdown("#### Autofill & Fix")
        user_fixes: Dict[str, List[str]] = {}
        if missing:
            st.warning(f"Missing branches: {len(missing)}")
            for (level, parent, key) in missing:
                with st.expander(f"Missing {key}"):
                    proposal = propose_autofill(level, parent, store); proposal5,_ = enforce_k_five(proposal)
                    cols = st.columns(5); edit = [cols[i].text_input(f"Option {i+1}", value=proposal5[i] if i<len(proposal5) else "", key=f"miss_{key}_{i}") for i in range(5)]
                    st.caption("Approve/edit to set 5 options."); user_fixes[key] = [normalize_text(x) for x in edit]
        if incomplete:
            st.warning(f"Incomplete branches (<5): {len(incomplete)}")
            for (level,parent,key,opts) in incomplete:
                with st.expander(f"Incomplete {key}"):
                    proposal = (opts + propose_autofill(level, parent, store))[:5]; proposal5,_ = enforce_k_five(proposal)
                    cols = st.columns(5); edit = [cols[i].text_input(f"Option {i+1}", value=proposal5[i] if i<len(proposal5) else "", key=f"incomp_{key}_{i}") for i in range(5)]
                    st.caption("Fill remaining blanks to reach 5."); user_fixes[key] = [normalize_text(x) for x in edit]
        if overspec:
            st.error(f"Overspecified branches (>5): {len(overspec)} — choose exactly 5")
            for (level,parent,key,opts) in overspec:
                with st.expander(f"Overspecified {key}"):
                    chosen = st.multiselect("Select 5 options", opts, default=opts[:5], key=f"over_{key}")
                    if len(chosen)!=5: st.warning("Please select exactly 5.")
                    user_fixes[key] = [normalize_text(x) for x in chosen] if len(chosen)==5 else [normalize_text(x) for x in opts[:5]]
        st.markdown("---")
        completed = None
        if st.button("Build Completed Sheet"):
            completed = build_completed_sheet(df_in, user_fixes)
            st.success(f"Completed rows: {len(completed)}"); st.dataframe(completed.head(50), use_container_width=True)
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
                df_in.to_excel(writer, index=False, sheet_name=sheet_name)
                completed.to_excel(writer, index=False, sheet_name=f"{sheet_name}{SHEET_COMPLETED_SUFFIX}")
            st.download_button("Download updated workbook", data=buffer.getvalue(), file_name="decision_tree_completed.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        st.markdown("#### Push to Google Sheets (overwrite)")
        st.caption("Requires [gcp_service_account] secrets in Streamlit.")
        up_spreadsheet_id = st.text_input("Spreadsheet ID (Upload tab)", key="up_sheet_id")
        up_target_tab = st.text_input("Target tab name", value=f"{sheet_name}{SHEET_COMPLETED_SUFFIX}", key="up_target_tab")
        if st.button("Push (Upload tab)"):
            if completed is None: st.warning("Build the Completed sheet first.")
            elif not up_spreadsheet_id or not up_target_tab: st.warning("Enter Spreadsheet ID and target tab name.")
            else:
                ok = push_to_google_sheets(up_spreadsheet_id, up_target_tab, completed)
                if ok: st.success("Pushed to Google Sheets.")

# ---------- Google Sheets mode ----------
with tab2:
    st.subheader("Google Sheets (optional)")
    st.caption("Add a service account JSON to app secrets under key 'gcp_service_account'.")
    spreadsheet_id = st.text_input("Spreadsheet ID")
    sheet_name_g = st.text_input("Sheet name to process (e.g., BP)", value="BP")
    run_btn = st.button("Load sheet")
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
            st.markdown("#### Preview (first 50 rows)"); st.dataframe(df_g.head(50), use_container_width=True)
            store = infer_branch_options(df_g)
            missing, overspec, incomplete = [], [], []
            for level in range(1, MAX_LEVELS+1):
                parents = set(parent_key_from_row(row, level) for _, row in df_g.iterrows())
                for parent in sorted(parents):
                    key = level_key_tuple(level, parent); opts = store.get(key, [])
                    if len(opts)==0: missing.append((level,parent,key))
                    elif len(opts)<5: incomplete.append((level,parent,key,opts))
                    elif len(opts)>5: overspec.append((level,parent,key,opts))
            st.markdown("#### Autofill & Fix")
            user_fixes: Dict[str, List[str]] = {}
            if missing:
                st.warning(f"Missing branches: {len(missing)}")
                for (level,parent,key) in missing:
                    with st.expander(f"Missing {key}"):
                        proposal = propose_autofill(level, parent, store); proposal5,_ = enforce_k_five(proposal)
                        cols = st.columns(5)
                        edit = [cols[i].text_input(f"Option {i+1}", value=proposal5[i] if i<len(proposal5) else "", key=f"g_miss_{key}_{i}") for i in range(5)]
                        user_fixes[key] = [normalize_text(x) for x in edit]
            if incomplete:
                st.warning(f"Incomplete branches (<5): {len(incomplete)}")
                for (level,parent,key,opts) in incomplete:
                    with st.expander(f"Incomplete {key}"):
                        proposal = (opts + propose_autofill(level, parent, store))[:5]; proposal5,_ = enforce_k_five(proposal)
                        cols = st.columns(5)
                        edit = [cols[i].text_input(f"Option {i+1}", value=proposal5[i] if i<len(proposal5) else "", key=f"g_incomp_{key}_{i}") for i in range(5)]
                        user_fixes[key] = [normalize_text(x) for x in edit]
            if overspec:
                st.error(f"Overspecified branches (>5): {len(overspec)} — choose exactly 5")
                for (level,parent,key,opts) in overspec:
                    with st.expander(f"Overspecified {key}"):
                        chosen = st.multiselect("Select 5 options", opts, default=opts[:5], key=f"g_over_{key}")
                        if len(chosen)!=5: st.warning("Please select exactly 5.")
                        user_fixes[key] = [normalize_text(x) for x in chosen] if len(chosen)==5 else [normalize_text(x) for x in opts[:5]]
            completed_g = None
            if st.button("Build Completed (Sheets)"):
                completed_g = build_completed_sheet(df_g, user_fixes)
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
        except Exception as e:
            st.error(f"Error loading sheet: {e}")
            print(e)  # For debugging purposes
            st.exception(e)
# End of streamlit_app_upload.py