# ui_workspace.py

from typing import Dict, List, Tuple, Optional, Set
from datetime import datetime
import io

import numpy as np
import pandas as pd
import streamlit as st

# Shared utils + logic modules
from utils import (
    CANON_HEADERS, LEVEL_COLS, MAX_LEVELS,
    normalize_text, validate_headers,
    infer_branch_options, friendly_parent_label,
    cluster_by_node,
)
from logic_export import (
    push_to_google_sheets, backup_sheet_copy,
    export_overrides_json, import_overrides_json,
    export_dataframe_to_excel_bytes, make_push_log_entry,
)
from logic_cascade import build_raw_plus_v630


# ----------------- local metrics (using utils.store) -----------------

def compute_parent_depth_score(df: pd.DataFrame) -> Tuple[int, int]:
    """
    Count how many parents (at all levels) have exactly 5 non-empty children.
    """
    store = infer_branch_options(df)
    total = 0
    ok = 0
    for level in range(1, MAX_LEVELS + 1):
        parents = set()
        for _, row in df.iterrows():
            # parent tuple for this level (Node 1..level-1)
            if level <= 1:
                p = tuple()
            else:
                path = []
                valid = True
                for c in LEVEL_COLS[:level - 1]:
                    v = normalize_text(row.get(c, ""))
                    if v == "":
                        valid = False
                        break
                    path.append(v)
                p = tuple(path) if valid else None
            if p is not None:
                parents.add(p)

        for p in parents:
            total += 1
            key = f"L{level}|" + (">".join(p) if p else "<ROOT>")
            non_empty_children = [x for x in (store.get(key, []) or []) if normalize_text(x) != ""]
            if len(non_empty_children) == 5:
                ok += 1
    return ok, total


def compute_row_path_score(df: pd.DataFrame) -> Tuple[int, int]:
    """
    Rows that have a full path (Node1..Node5 all non-empty).
    """
    if df.empty:
        return (0, 0)
    nodes = df[LEVEL_COLS].applymap(normalize_text)
    full = nodes.ne("").all(axis=1)
    return int(full.sum()), int(len(df))


# ----------------- Workspace Selection UI -----------------

def render():
    st.header("🗂 Workspace Selection")

    # Discover available sources
    sources = []
    if st.session_state.get("upload_workbook", {}):
        sources.append("Upload workbook")
    if st.session_state.get("gs_workbook", {}):
        sources.append("Google Sheets workbook")

    if not sources:
        st.info("Load a workbook in the **Source** tab first (upload file or Google Sheets).")
        return

    # Source selector
    source_ws = st.radio("Choose data source", sources, horizontal=True, key="ws_source_sel")

    if source_ws == "Upload workbook":
        wb_ws = st.session_state.get("upload_workbook", {})
        override_root = "branch_overrides_upload"
        source_code = "upload"
    else:
        wb_ws = st.session_state.get("gs_workbook", {})
        override_root = "branch_overrides_gs"
        source_code = "gs"

    if not wb_ws:
        st.warning("No sheets found in the selected source. Load data from the **Source** tab.")
        return

    # Sheet selector
    sheet_ws = st.selectbox("Sheet", list(wb_ws.keys()), key="ws_sheet_sel")
    df_ws = wb_ws.get(sheet_ws, pd.DataFrame())

    # Remember current work context for other tabs
    st.session_state["work_context"] = {"source": source_code, "sheet": sheet_ws}

    # ===== Summary + Preview =====
    if df_ws.empty or not validate_headers(df_ws):
        st.info("Selected sheet is empty or headers mismatch.")
    else:
        st.write(f"Found {len(wb_ws)} sheet(s). Choose one to process:")

        ok_p, total_p = compute_parent_depth_score(df_ws)
        ok_r, total_r = compute_row_path_score(df_ws)
        p1, p2 = st.columns(2)
        with p1:
            st.metric("Parents with 5 children", f"{ok_p}/{total_p}")
            st.progress(0 if total_p == 0 else ok_p / total_p)
        with p2:
            st.metric("Rows with full path", f"{ok_r}/{total_r}")
            st.progress(0 if total_r == 0 else ok_r / total_r)

        total_rows = len(df_ws)
        st.markdown("#### Preview (50 rows)")
        if total_rows <= 50:
            st.caption(f"Showing all {total_rows} rows.")
            st.dataframe(df_ws, use_container_width=True)
        else:
            state_key = f"preview_start_{sheet_ws}"
            start_idx = int(st.session_state.get(state_key, 0))
            cprev, cnum, cnext = st.columns([1, 2, 1])
            with cprev:
                if st.button("◀ Previous 50", key=f"prev50_{sheet_ws}"):
                    start_idx = max(0, start_idx - 50)
            with cnum:
                start_1based = st.number_input(
                    "Start row (1-based)",
                    min_value=1,
                    max_value=max(1, total_rows - 49),
                    value=start_idx + 1,
                    step=50,
                    help="Pick where to start the 50-row preview.",
                    key=f"startnum_{sheet_ws}",
                )
                start_idx = int(start_1based) - 1
            with cnext:
                if st.button("Next 50 ▶", key=f"next50_{sheet_ws}"):
                    start_idx = min(max(0, total_rows - 50), start_idx + 50)
            st.session_state[state_key] = start_idx
            end_idx = min(start_idx + 50, total_rows)
            st.caption(f"Showing rows **{start_idx + 1}–{end_idx}** of **{total_rows}**.")
            st.dataframe(df_ws.iloc[start_idx:end_idx], use_container_width=True)

    st.markdown("---")

    # ========== 📦 Export / Import Overrides (JSON) ==========
    with st.expander("📦 Export / Import Overrides (JSON)"):
        overrides_all = st.session_state.get(override_root, {})
        overrides_sheet = overrides_all.get(sheet_ws, {})

        col1, col2 = st.columns([1, 2])
        with col1:
            # Use shared exporter
            data = export_overrides_json(overrides_sheet)
            st.download_button(
                "Export overrides.json",
                data=data,
                file_name=f"{sheet_ws}_overrides.json",
                mime="application/json",
            )

        with col2:
            upfile = st.file_uploader("Import overrides.json", type=["json"], key="ws_imp_json_sel")
            import_mode = st.radio(
                "Import mode",
                ["Replace", "Merge (prefer import)", "Merge (prefer existing)"],
                horizontal=True,
                key="ws_imp_mode_sel",
            )
            if upfile is not None and st.button("Apply Import", key="ws_imp_apply_sel"):
                try:
                    mode_key = {"Replace": "replace", "Merge (prefer import)": "merge_import", "Merge (prefer existing)": "merge_existing"}[import_mode]
                    merged = import_overrides_json(overrides_sheet, upfile.getvalue(), mode=mode_key)
                    overrides_all[sheet_ws] = merged
                    st.session_state[override_root] = overrides_all
                    st.success("Overrides imported (stored in-session).")
                except Exception as e:
                    st.error(f"Import failed: {e}")

    # ========== 🧼 Data quality tools ==========
    with st.expander("🧼 Data quality tools (applies to this sheet)"):
        if df_ws.empty or not validate_headers(df_ws):
            st.info("Load a valid sheet first.")
        else:
            ok_p, total_p = compute_parent_depth_score(df_ws)
            ok_r, total_r = compute_row_path_score(df_ws)
            st.write(f"Parents with 5 children: **{ok_p}/{total_p}**")
            st.write(f"Rows with full path: **{ok_r}/{total_r}**")

            case_mode = st.selectbox("Case normalization", ["None", "Title", "lower", "UPPER"], index=0, key="ws_case_mode_sel")
            syn_text = st.text_area("Synonym map (one per line: A => B)", key="ws_syn_map_text_sel")

            def parse_synonym_map(text: str) -> Dict[str, str]:
                mapping = {}
                for line in text.splitlines():
                    if "=>" in line:
                        a, b = line.split("=>", 1)
                        a = a.strip()
                        b = b.strip()
                        if a and b:
                            mapping[a] = b
                return mapping

            def normalize_label(s: str, case_mode: str) -> str:
                s = s.strip()
                if case_mode == "lower":
                    return s.lower()
                if case_mode == "UPPER":
                    return s.upper()
                if case_mode == "Title":
                    return s.title()
                return s

            def normalize_sheet_df(df0: pd.DataFrame, case_mode: str, syn_map: Dict[str, str]) -> pd.DataFrame:
                df2 = df0.copy()
                for col in ["Vital Measurement"] + LEVEL_COLS:
                    if col in df2.columns:
                        def _apply(v):
                            v = normalize_text(v)
                            v = syn_map.get(v, v)
                            v = normalize_label(v, case_mode) if case_mode != "None" else v
                            return v
                        df2[col] = df2[col].map(_apply)
                for col in ["Diagnostic Triage", "Actions"]:
                    if col in df2.columns:
                        df2[col] = df2[col].map(normalize_text)
                return df2

            if st.button("Normalize sheet now (in-session)", key="ws_norm_sel"):
                syn_map = parse_synonym_map(syn_text)
                df_norm = normalize_sheet_df(df_ws, case_mode, syn_map)
                wb_ws[sheet_ws] = df_norm
                if source_code == "upload":
                    st.session_state["upload_workbook"] = wb_ws
                    st.success("Sheet normalized in-session. Download from Source tab to persist locally.")
                else:
                    st.session_state["gs_workbook"] = wb_ws
                    st.success("Sheet normalized in-session. Use Push Settings below to write to Google Sheets.")

    # ========== 🧩 Group rows (cluster identical labels together) ==========
    with st.expander("🧩 Group rows (cluster identical labels together)"):
        st.caption("Group rows so identical **Node 1** or **Node 2** values are contiguous. This is a stable grouping.")

        if df_ws.empty or not validate_headers(df_ws):
            st.info("Load a valid sheet first.")
        else:
            # Default to Node 1 (per 6.3.0 scope)
            group_mode = st.radio(
                "Grouping mode",
                ["Off", "Node 1", "Node 2"],
                index=1,
                horizontal=True,
                key="ws_group_mode_sel",
            )
            scope = st.radio(
                "Grouping scope",
                ["Whole sheet", "Within Vital Measurement"],
                horizontal=True,
                key="ws_group_scope_sel",
            )
            preview = st.checkbox("Show preview (does not modify data)", value=True, key="ws_group_preview_sel")

            def grouped_df(df0: pd.DataFrame, by_col: Optional[str], scope_mode: str) -> pd.DataFrame:
                if by_col is None:
                    return df0.copy()
                if scope_mode == "Within Vital Measurement":
                    # Group within each VM block to keep VMs separate
                    parts = []
                    for vm_name, df_part in df0.groupby(df0["Vital Measurement"].map(normalize_text), sort=False):
                        parts.append(cluster_by_node(df_part, by_col))
                    return pd.concat(parts, ignore_index=True) if parts else df0.copy()
                else:
                    return cluster_by_node(df0, by_col)

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
                file_name=f"{sheet_ws}_grouped_{(by_col or 'none').replace(' ', '_').lower()}.csv",
                mime="text/csv",
            )

            colg1, colg2 = st.columns([1, 1])
            with colg1:
                if st.button("Apply grouping (in-session)", key="ws_group_apply_sel"):
                    wb_ws[sheet_ws] = df_prev
                    if source_code == "upload":
                        st.session_state["upload_workbook"] = wb_ws
                        st.success("Applied grouping in-session (Upload workbook).")
                    else:
                        st.session_state["gs_workbook"] = wb_ws
                        st.success("Applied grouping in-session (Google Sheets workbook).")

            with colg2:
                sid = st.session_state.get("gs_spreadsheet_id", "")
                if source_code == "gs" and sid and st.button("Apply & push grouping to Google Sheets", key="ws_group_push_sel"):
                    ok = push_to_google_sheets(sid, sheet_ws, df_prev)
                    if ok:
                        st.success("Grouping pushed to Google Sheets.")

    # ========== 🔧 Google Sheets Push Settings (BOTTOM) ==========
    st.markdown("---")
    st.subheader("🔧 Google Sheets Push Settings")

    sid = st.text_input("Spreadsheet ID", value=st.session_state.get("gs_spreadsheet_id", ""), key="ws_push_sid_sel")
    if sid:
        st.session_state["gs_spreadsheet_id"] = sid

    default_tab = st.session_state.get("saved_targets", {}).get(sheet_ws, {}).get("tab", f"{sheet_ws}")
    target_tab = st.text_input("Target tab", value=default_tab, key="ws_push_target_sel")

    include_scope_label = st.radio(
        "Include scope for Raw+",
        ["All completed parents", "Only parents edited this session"],
        horizontal=True,
        key="ws_push_scope_sel",
    )
    include_scope_flag = "session" if include_scope_label.endswith("session") else "all"

    push_backup = st.checkbox("Create a backup tab before overwrite", value=True, key="ws_push_backup_sel")
    dry_run = st.checkbox("Dry-run (build but don't write to Sheets)", value=False, key="ws_push_dry_sel")

    # Buttons: push current view, or build & push Raw+
    c_push1, c_push2 = st.columns([1, 1])

    with c_push1:
        if st.button("📤 Push current view (overwrite)", key="ws_push_current_btn"):
            if not sid or not target_tab:
                st.error("Missing Spreadsheet ID or target tab.")
            elif df_ws.empty or not validate_headers(df_ws):
                st.error("Current sheet is empty or headers mismatch.")
            else:
                if dry_run:
                    st.success("Dry-run complete. No changes written to Google Sheets.")
                    st.dataframe(df_ws.head(50), use_container_width=True)
                    xlsx = export_dataframe_to_excel_bytes(df_ws, sheet_name=sheet_ws[:31] or "Sheet1")
                    st.download_button(
                        "Download current view workbook",
                        data=xlsx,
                        file_name="decision_tree_current_view.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    )
                else:
                    if push_backup:
                        bak = backup_sheet_copy(sid, target_tab)
                        if bak:
                            st.info(f"Backed up current '{target_tab}' to '{bak}'.")
                    ok = push_to_google_sheets(sid, target_tab, df_ws)
                    if ok:
                        log = st.session_state.get("push_log", [])
                        log.append(make_push_log_entry(
                            sheet=sheet_ws, target_tab=target_tab, spreadsheet_id=sid,
                            rows_written=len(df_ws), new_rows_added=0,
                            scope=include_scope_flag
                        ))
                        st.session_state["push_log"] = log
                        saved = st.session_state.get("saved_targets", {})
                        saved.setdefault(sheet_ws, {})
                        saved[sheet_ws]["tab"] = target_tab
                        st.session_state["saved_targets"] = saved
                        st.success(f"Pushed {len(df_ws)} rows to '{target_tab}'.")

    with c_push2:
        if st.button("🧠 Build & Push Raw+ (cascade)", type="primary", key="ws_push_raw_btn"):
            if not sid or not target_tab:
                st.error("Missing Spreadsheet ID or target tab.")
            elif df_ws.empty or not validate_headers(df_ws):
                st.error("Current sheet is empty or headers mismatch.")
            else:
                # Collect overrides + session-edited keys
                overrides_all = st.session_state.get(override_root, {})
                overrides_sheet = overrides_all.get(sheet_ws, {})
                edited_keys_for_sheet = set(st.session_state.get("session_edited_keys", {}).get(sheet_ws, []))

                # Build Raw+ with enhanced cascade (v6.3.0)
                try:
                    df_aug, stats = build_raw_plus_v630(
                        df_ws, overrides_sheet, include_scope_flag, edited_keys_for_sheet
                    )
                except AssertionError as e:
                    st.error(str(e))
                    return

                st.info(
                    f"Delta — Generated: **{stats['generated']}**, New added: **{stats['new_added']}**, "
                    f"In-place filled: **{stats['inplace_filled']}**, Duplicates skipped: **{stats['duplicates_skipped']}**, "
                    f"Final total: **{stats['final_total']}**."
                )
                st.caption(f"Will write **{len(df_aug)} rows × {len(df_aug.columns)} cols** to tab **{target_tab}**.")

                if dry_run:
                    st.success("Dry-run complete. No changes written to Google Sheets.")
                    st.dataframe(df_aug.head(50), use_container_width=True)
                    xlsx = export_dataframe_to_excel_bytes(df_aug, sheet_name=sheet_ws[:31] or "Sheet1")
                    st.download_button(
                        "Download augmented (Raw+) workbook",
                        data=xlsx,
                        file_name="decision_tree_raw_plus.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    )
                else:
                    if push_backup:
                        bak = backup_sheet_copy(sid, target_tab)
                        if bak:
                            st.info(f"Backed up current '{target_tab}' to '{bak}'.")
                    ok = push_to_google_sheets(sid, target_tab, df_aug)
                    if ok:
                        log = st.session_state.get("push_log", [])
                        log.append(make_push_log_entry(
                            sheet=sheet_ws, target_tab=target_tab, spreadsheet_id=sid,
                            rows_written=len(df_aug), new_rows_added=stats.get("new_added", 0),
                            scope=include_scope_flag
                        ))
                        st.session_state["push_log"] = log
                        saved = st.session_state.get("saved_targets", {})
                        saved.setdefault(sheet_ws, {})
                        saved[sheet_ws]["tab"] = target_tab
                        st.session_state["saved_targets"] = saved
                        st.success(f"Pushed {len(df_aug)} rows to '{target_tab}'.")
