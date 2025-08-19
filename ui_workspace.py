# ui_workspace.py

# TODO[Step10]: UX consistency pass:
# - Standardize header icon text, KPI row, and Save/Push controls
# - Ensure previews cap at .head(100) and metrics are cached

import io
import json
from typing import Dict, List, Tuple, Optional, Set
from datetime import datetime


import numpy as np
import pandas as pd
import streamlit as st

from utils import (
    CANON_HEADERS, LEVEL_COLS, MAX_LEVELS,
    normalize_text, validate_headers,
    compute_parent_depth_score, compute_row_path_score,
    friendly_parent_label, level_key_tuple, enforce_k_five,
    order_decision_tree,
)
from ui_helpers import render_preview_caption, st_success, st_warning, st_error, st_info


@st.cache_data(show_spinner=False, ttl=600)
def _cached_compute_parent_depth_score(df: pd.DataFrame) -> Tuple[int, int]:
    """Cached version of compute_parent_depth_score to prevent recomputation."""
    return compute_parent_depth_score(df)


@st.cache_data(show_spinner=False, ttl=600)
def _cached_compute_row_path_score(df: pd.DataFrame) -> Tuple[int, int]:
    """Cached version of compute_row_path_score to prevent recomputation."""
    return compute_row_path_score(df)


# --- Google Sheets helpers (using app.sheets module) ---
def _push_to_google_sheets(spreadsheet_id: str, sheet_name: str, df: pd.DataFrame) -> bool:
    """Push DataFrame to Google Sheets using app.sheets module."""
    try:
        from app.sheets import get_gspread_client_from_secrets, open_spreadsheet, write_dataframe
        
        # Get client and open spreadsheet
        client = get_gspread_client_from_secrets(st.secrets["gcp_service_account"])
        spreadsheet = open_spreadsheet(client, spreadsheet_id)
        
        # Prepare DataFrame for writing
        df = df.fillna("")
        
        # Write data using the new helper
        write_dataframe(spreadsheet, sheet_name, df, mode="overwrite")
        
        return True

    except Exception as e:
        st.error(f"❌ Push to Google Sheets failed: {e}")
        return False


def _backup_sheet_copy(spreadsheet_id: str, source_sheet: str) -> Optional[str]:
    """Create backup of Google Sheet using app.sheets module."""
    try:
        from app.sheets import get_gspread_client_from_secrets, open_spreadsheet, backup_worksheet
        
        # Get client and open spreadsheet
        client = get_gspread_client_from_secrets(st.secrets["gcp_service_account"])
        spreadsheet = open_spreadsheet(client, spreadsheet_id)
        
        # Create backup using the new helper
        backup_title = backup_worksheet(spreadsheet, source_sheet)
        return backup_title
        
    except Exception as e:
        st.error(f"❌ Backup failed: {e}")
        return None


# ---------- UI: Workspace Selection ----------

def render():
    st.header("🗂 Workspace Selection")

    # Choose data source
    sources = []
    if st.session_state.get("upload_workbook", {}):
        sources.append("Upload workbook")
    if st.session_state.get("gs_workbook", {}):
        sources.append("Google Sheets workbook")

    if not sources:
        st_info("Load a workbook in the **Source** tab first (upload file or Google Sheets).")
        return

    # Default from work_context if available
    ctx = st.session_state.get("work_context", {})
    default_src_label = {"upload": "Upload workbook", "gs": "Google Sheets workbook"}.get(ctx.get("source"))
    init_idx = sources.index(default_src_label) if default_src_label in sources else 0

    source_ws = st.radio("Choose data source", sources, horizontal=True, index=init_idx, key="ws_source_sel")

    if source_ws == "Upload workbook":
        wb_ws = st.session_state.get("upload_workbook", {})
        override_root = "branch_overrides_upload"
        current_source_code = "upload"
    else:
        wb_ws = st.session_state.get("gs_workbook", {})
        override_root = "branch_overrides_gs"
        current_source_code = "gs"

    if not wb_ws:
        st_warning("No sheets found in the selected source. Load data from the **Source** tab.")
        return

    # Sheet picker (default to context if present)
    default_sheet = ctx.get("sheet")
    sheet_names = list(wb_ws.keys())
    sheet_idx = sheet_names.index(default_sheet) if default_sheet in sheet_names else 0
    sheet_ws = st.selectbox("Sheet", sheet_names, index=sheet_idx, key="ws_sheet_sel")
    df_ws = wb_ws.get(sheet_ws, pd.DataFrame())

    # Remember current work context for other tabs
    st.session_state["work_context"] = {"source": current_source_code, "sheet": sheet_ws}
    
    # Propagate current DataFrame to session state for downstream tabs
    st.session_state["current_df"] = df_ws
    st.info(f"ℹ️ current_df updated with {len(df_ws)} rows")

    # ===== Summary + Preview =====
    if df_ws.empty or not validate_headers(df_ws):
        st_info("Selected sheet is empty or headers mismatch.")
    else:
        st.write(f"Found {len(wb_ws)} sheet(s). Choose one to process:")

        ok_p, total_p = _cached_compute_parent_depth_score(df_ws)
        ok_r, total_r = _cached_compute_row_path_score(df_ws)
        p1, p2 = st.columns(2)
        with p1:
            st.metric("Parents with 5 children", f"{ok_p}/{total_p}")
            st.progress(0 if total_p==0 else ok_p/total_p)
        with p2:
            st.metric("Rows with full path", f"{ok_r}/{total_r}")
            st.progress(0 if total_r==0 else ok_r/total_r)

        # Apply branch ordering for logical display
        try:
            df_ws_ordered = order_decision_tree(df_ws)
            if not df_ws_ordered.equals(df_ws):
                st_info("Data displayed in logical tree order (parents followed by children)")
        except Exception as e:
            st_warning(f"Could not apply tree ordering: {e}")
            df_ws_ordered = df_ws
        
        total_rows = len(df_ws_ordered)
        st.markdown("#### Preview (50 rows)")
        if total_rows <= 50:
            st.dataframe(df_ws_ordered, use_container_width=True)
            render_preview_caption(df_ws_ordered, df_ws_ordered, max_rows=50)
        elif total_rows <= 100:
            st.dataframe(df_ws_ordered.head(100), use_container_width=True)
            render_preview_caption(df_ws_ordered.head(100), df_ws_ordered, max_rows=100)
        else:
            state_key = f"preview_start_{sheet_ws}"
            start_idx = int(st.session_state.get(state_key, 0))
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
            st.session_state[state_key] = start_idx
            end_idx = min(start_idx + 50, total_rows)
            st.dataframe(df_ws_ordered.iloc[start_idx:end_idx], use_container_width=True)
            render_preview_caption(df_ws_ordered.iloc[start_idx:end_idx], df_ws_ordered, max_rows=50)

    st.markdown("---")

    # ========== 📦 Export / Import Overrides (JSON) ==========
    with st.expander("📦 Export / Import Overrides (JSON)"):
        overrides_all = st.session_state.get(override_root, {})
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
                        st.error("❌ Invalid JSON: expected an object mapping keys to lists.")
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
                        else:  # prefer existing
                            cur = overrides_all.get(sheet_ws, {}).copy()
                            for k,v in imported.items():
                                if k not in cur:
                                    cur[k] = enforce_k_five(v if isinstance(v, list) else [v])
                            overrides_all[sheet_ws] = cur
                        st.session_state[override_root] = overrides_all
                        st.success("✅ Overrides imported (stored in-session).")
                except Exception as e:
                    st.error(f"❌ Import failed: {e}")

    # ========== 🧼 Data quality tools ==========
    with st.expander("🧼 Data quality tools (applies to this sheet)"):
        if df_ws.empty or not validate_headers(df_ws):
            st.info("ℹ️ Load a valid sheet first.")
        else:
            ok_p, total_p = _cached_compute_parent_depth_score(df_ws)
            ok_r, total_r = _cached_compute_row_path_score(df_ws)
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
                        # Use DataFrame.map (applymap deprecated)
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
                    st.session_state["upload_workbook"] = wb_ws
                    st.success("✅ Sheet normalized in-session. Download from Source tab to persist locally.")
                else:
                    st.session_state["gs_workbook"] = wb_ws
                    st.success("✅ Sheet normalized in-session. Use Push Settings below to write to Google Sheets.")

    # ========== 🧩 Group rows (Node 1 / Node 2) ==========
    with st.expander("🧩 Group rows (cluster identical labels together)"):
        st.caption("Group rows so identical **Node 1** or **Node 2** values are contiguous. This is a stable grouping.")
        if df_ws.empty or not validate_headers(df_ws):
            st.info("ℹ️ Load a valid sheet first.")
        else:
            group_mode = st.radio("Grouping mode", ["Off", "Node 1", "Node 2"], horizontal=True, key="ws_group_mode_sel")
            scope = st.radio("Grouping scope", ["Whole sheet", "Within Vital Measurement"], horizontal=True, key="ws_group_scope_sel")
            preview = st.checkbox("Show preview (does not modify data)", value=True, key="ws_group_preview_sel")

            # TODO Step 9: Implement branch grouping so Node 1 children are displayed/exported consecutively.
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
                        st.session_state["upload_workbook"] = wb_ws
                        st.success("✅ Applied grouping in-session (Upload workbook).")
                    else:
                        st.session_state["gs_workbook"] = wb_ws
                        st.success("✅ Applied grouping in-session (Google Sheets workbook).")
            with colg2:
                sid_group = st.session_state.get("gs_spreadsheet_id","")
                if current_source_code == "gs" and sid_group and st.button("Apply & push grouping to Google Sheets", key="ws_group_push_sel"):
                    ok = _push_to_google_sheets(sid_group, sheet_ws, df_prev)
                    if ok: st.success("✅ Grouping pushed to Google Sheets.")

    # ========== 🔧 Google Sheets Push Settings (BOTTOM) ==========
    st.markdown("---")
    st.subheader("🔧 Google Sheets Push Settings")

    sid = st.text_input("Spreadsheet ID", value=st.session_state.get("gs_spreadsheet_id",""), key="ws_push_sid_sel")
    if sid:
        st.session_state["gs_spreadsheet_id"] = sid

    default_tab = st.session_state.get("saved_targets", {}).get(sheet_ws, {}).get("tab", f"{sheet_ws}")
    target_tab = st.text_input("Target tab name", value=default_tab, key="ws_push_target_sel")

    include_scope = st.radio("Include scope", ["All completed parents","Only parents edited this session"], horizontal=True, key="ws_push_scope_sel")
    push_backup = st.checkbox("Create a backup tab before overwrite", value=True, key="ws_push_backup_sel")
    dry_run = st.checkbox("Dry-run (build but don't write to Sheets)", value=False, key="ws_push_dry_sel")

    st.caption("Note: In this refactor step, push writes the **current view**. Raw+ build will be wired when logic modules are added.")

    if st.button("📤 Push current view to Google Sheets", type="primary", key="ws_push_btn_sel"):
        if not sid or not target_tab:
            st.error("❌ Missing Spreadsheet ID or target tab.")
        elif df_ws.empty or not validate_headers(df_ws):
            st.error("❌ Current sheet is empty or headers mismatch.")
        else:
            if dry_run:
                st.success("✅ Dry-run complete. No changes written to Google Sheets.")
                # Limit preview to first 100 rows for speed
                st.dataframe(df_ws.head(100), use_container_width=True)
                buffer = io.BytesIO()
                with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
                    df_ws.to_excel(writer, index=False, sheet_name=sheet_ws[:31] or "Sheet1")
                st.download_button("Download current view workbook", data=buffer.getvalue(),
                                   file_name="decision_tree_current_view.xlsx",
                                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            else:
                if push_backup:
                    bak = _backup_sheet_copy(sid, target_tab)
                    if bak:
                        st.info(f"ℹ️ Backed up current '{target_tab}' to '{bak}'.")
                ok = _push_to_google_sheets(sid, target_tab, df_ws)
                if ok:
                    log = st.session_state.get("push_log", [])
                    log.append({
                        "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "sheet": sheet_ws,
                        "target_tab": target_tab,
                        "spreadsheet_id": sid,
                        "rows_written": len(df_ws),
                        "new_rows_added": 0,  # Raw+ stats will come later
                        "scope": "session" if include_scope.endswith("session") else "all",
                    })
                    st.session_state["push_log"] = log
                    saved = st.session_state.get("saved_targets", {})
                    saved.setdefault(sheet_ws, {})
                    saved[sheet_ws]["tab"] = target_tab
                    st.session_state["saved_targets"] = saved
                    st.success(f"✅ Pushed {len(df_ws)} rows to '{target_tab}'.")
