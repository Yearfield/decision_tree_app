# ui_source.py

from typing import Dict, List, Tuple, Optional
import io

import pandas as pd
import numpy as np
import streamlit as st

from utils import (
    CANON_HEADERS, LEVEL_COLS, MAX_LEVELS,
    normalize_text, validate_headers, ensure_canon_columns, drop_fully_blank_paths,
    enforce_k_five, level_key_tuple,
)
from logic_export import (
    read_google_sheet,
    export_dataframe_to_excel_bytes,
)
from logic_cascade import build_raw_plus_v630


# ----------------- session helpers -----------------



def _mark_session_edit(sheet: str, keyname: str):
    ek = st.session_state.get("session_edited_keys", {})
    cur = set(ek.get(sheet, []))
    cur.add(keyname)
    ek[sheet] = list(cur)
    st.session_state["session_edited_keys"] = ek


# ----------------- core IO helpers -----------------

def _load_xlsx_to_workbook(file) -> Dict[str, pd.DataFrame]:
    xls = pd.ExcelFile(file)
    sheets: Dict[str, pd.DataFrame] = {}
    for name in xls.sheet_names:
        dfx = xls.parse(name)
        # normalize headers & ensure canon columns
        dfx.columns = [normalize_text(c) for c in dfx.columns]
        for c in CANON_HEADERS:
            if c not in dfx.columns:
                dfx[c] = ""
        dfx = dfx[CANON_HEADERS]
        dfx = drop_fully_blank_paths(dfx)
        sheets[name] = dfx
    return sheets

def _load_csv_to_workbook(file) -> Dict[str, pd.DataFrame]:
    df = pd.read_csv(file)
    df.columns = [normalize_text(c) for c in df.columns]
    for c in CANON_HEADERS:
        if c not in df.columns:
            df[c] = ""
    df = df[CANON_HEADERS]
    df = drop_fully_blank_paths(df)
    return {"Sheet1": df}


# ----------------- UI: Source Tab -----------------

def render():
    st.header("📂 Source")
    st.markdown("Load or create your decision tree data. This tab manages **input data** and quick builders.")

    # Show service account hint
    if "gcp_service_account" in st.secrets:
        st.caption("Google Sheets linked ✓")
    else:
        st.caption("Google Sheets not configured (optional). Add your service account under `[gcp_service_account]` in secrets.")

    # ----------------- Upload workbook -----------------
    st.subheader("📤 Upload Workbook")
    file = st.file_uploader("Upload XLSX or CSV", type=["xlsx", "xls", "csv"], key="source_upload_file")

    cols_up = st.columns([1,1,1])
    with cols_up[0]:
        new_sheet_name = st.text_input("Create in-session sheet (optional)", key="source_new_sheet_name", help="Adds a blank sheet to the Upload workbook.")
    with cols_up[1]:
        if st.button("Create sheet", key="source_create_sheet_btn"):
            wb = st.session_state.get("upload_workbook", {})
            if not new_sheet_name:
                st.warning("Enter a sheet name.")
            elif new_sheet_name in wb:
                st.warning("That sheet already exists.")
            else:
                empty = pd.DataFrame(columns=CANON_HEADERS)
                wb[new_sheet_name] = empty
                st.session_state["upload_workbook"] = wb
                st.success(f"Created blank sheet '{new_sheet_name}' in Upload workbook.")
    with cols_up[2]:
        if st.button("Download current Upload workbook", key="source_download_upload_book_btn"):
            wb = st.session_state.get("upload_workbook", {})
            if not wb:
                st.info("Upload workbook is empty.")
            else:
                buffer = io.BytesIO()
                with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
                    for nm, d in wb.items():
                        dfw = ensure_canon_columns(d)
                        dfw.to_excel(writer, index=False, sheet_name=(nm or "Sheet1")[:31])
                st.download_button(
                    "Download .xlsx",
                    data=buffer.getvalue(),
                    file_name="upload_workbook.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )

    if file is not None:
        try:
            if file.name.lower().endswith(".csv"):
                sheets = _load_csv_to_workbook(file)
            else:
                sheets = _load_xlsx_to_workbook(file)
            # merge into session upload_workbook
            cur = st.session_state.get("upload_workbook", {})
            cur.update({k: v.copy() for k, v in sheets.items()})
            st.session_state["upload_workbook"] = cur
            st.success(f"Loaded {len(sheets)} sheet(s) into **Upload workbook**.")
            st.caption("Tip: Switch to **Workspace Selection** to inspect metrics and preview.")
        except Exception as e:
            st.error(f"Failed to load workbook: {e}")

    # Show quick summary of in-session upload workbook
    wb_u = st.session_state.get("upload_workbook", {})
    if wb_u:
        st.write(f"Upload workbook now contains **{len(wb_u)}** sheet(s): {', '.join(wb_u.keys())}")

    st.markdown("---")

    # ----------------- Google Sheets Load/Refresh -----------------
    st.subheader("🔄 Google Sheets")
    cgs1, cgs2 = st.columns([2,1])
    with cgs1:
        spreadsheet_id = st.text_input("Spreadsheet ID", value=st.session_state.get("gs_spreadsheet_id", ""), key="source_gs_sid")
        if spreadsheet_id:
            st.session_state["gs_spreadsheet_id"] = spreadsheet_id
        sheet_name = st.text_input("Sheet name to load (e.g., BP)", value="BP", key="source_gs_sheet")
    with cgs2:
        if st.button("Load / Refresh sheet", key="source_gs_load_btn"):
            if not spreadsheet_id or not sheet_name:
                st.warning("Enter Spreadsheet ID and sheet name.")
            else:
                try:
                    df_g = read_google_sheet(spreadsheet_id, sheet_name)
                    wb_g = st.session_state.get("gs_workbook", {})
                    wb_g[sheet_name] = df_g
                    st.session_state["gs_workbook"] = wb_g
                    st.success(f"Loaded '{sheet_name}' from Google Sheets.")
                    st.dataframe(df_g.head(50), use_container_width=True)
                except Exception as e:
                    st.error(f"Google Sheets error: {e}")

    wb_g = st.session_state.get("gs_workbook", {})
    if wb_g:
        st.write(f"Google Sheets workbook holds **{len(wb_g)}** sheet(s): {', '.join(wb_g.keys())}")

    st.markdown("---")

    # ----------------- 🧩 VM Builder (auto-cascade) -----------------
    st.subheader("🧩 VM Builder (create Vital Measurements and auto-cascade)")

    # Target: which workbook / sheet to update
    possible_sources = []
    if st.session_state.get("upload_workbook", {}): possible_sources.append("Upload workbook")
    if st.session_state.get("gs_workbook", {}): possible_sources.append("Google Sheets workbook")

    if not possible_sources:
        st.info("Load data first (Upload or Google Sheets).")
    else:
        vm_target_src = st.radio("Apply to", possible_sources, horizontal=True, key="source_vm_target_src")

        if vm_target_src == "Upload workbook":
            vm_wb = st.session_state.get("upload_workbook", {})
            override_root = "branch_overrides_upload"
            src_code = "upload"
        else:
            vm_wb = st.session_state.get("gs_workbook", {})
            override_root = "branch_overrides_gs"
            src_code = "gs"

        vm_sheet = st.selectbox("Target sheet", list(vm_wb.keys()), key="source_vm_target_sheet")
        df_in = vm_wb.get(vm_sheet, pd.DataFrame())

        vm_mode = st.radio(
            "Mode",
            ["Create VM with 5 Node-1 options", "Create VM with one Node-1 and its 5 Node-2 options"],
            horizontal=False,
            key="source_vm_mode",
        )

        overrides_all = st.session_state.get(override_root, {})
        overrides_sheet = overrides_all.get(vm_sheet, {}).copy()

        if vm_mode == "Create VM with 5 Node-1 options":
            vm_name = st.text_input("Vital Measurement name", key="source_vm_new_name")
            cols = st.columns(5)
            vals_n1 = [cols[i].text_input(f"Node-1 option {i+1}", key=f"source_vm_n1_{i}") for i in range(5)]

            if st.button("Create VM and Auto-cascade", key="source_vm_create_n1_btn"):
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
                    overrides_sheet[k1] = enforce_k_five(vals_n1)
                    overrides_all[vm_sheet] = overrides_sheet
                    st.session_state[override_root] = overrides_all
                    _mark_session_edit(vm_sheet, k1)

                    # Build Raw+ (session scope is enough; uses edited keys)
                    edited_keys_for_sheet = set(st.session_state.get("session_edited_keys", {}).get(vm_sheet, []))
                    df_aug, stats = build_raw_plus_v630(df_in, overrides_sheet, include_scope="session", edited_keys_for_sheet=edited_keys_for_sheet)

                    # Persist in-session
                    vm_wb[vm_sheet] = df_aug
                    if src_code == "upload":
                        st.session_state["upload_workbook"] = vm_wb
                    else:
                        st.session_state["gs_workbook"] = vm_wb

                    st.success(f"VM '{vm_name}' created. Auto-cascade added {stats['new_added']} new unique rows; filled {stats['inplace_filled']} anchors.")

        else:
            vm_name = st.text_input("Vital Measurement name", key="source_vm2_new_name")
            n1 = st.text_input("Node-1 value", key="source_vm2_n1")
            cols = st.columns(5)
            vals_n2 = [cols[i].text_input(f"Node-2 option {i+1}", key=f"source_vm2_n2_{i}") for i in range(5)]

            if st.button("Create VM + Node-1 + Node-2 and Auto-cascade", key="source_vm_create_n1n2_btn"):
                if not vm_name.strip() or not n1.strip():
                    st.error("Enter a Vital Measurement and Node-1.")
                else:
                    # Ensure anchor row for VM exists
                    if df_in[df_in["Vital Measurement"].map(normalize_text) == vm_name].empty:
                        anchor = {"Vital Measurement": vm_name}
                        for c in LEVEL_COLS: anchor[c] = ""
                        anchor["Diagnostic Triage"] = ""; anchor["Actions"] = ""
                        df_in = pd.concat([df_in, pd.DataFrame([anchor], columns=CANON_HEADERS)], ignore_index=True)

                    # L1|<ROOT> — add n1 (preserve existing)
                    k1 = level_key_tuple(1, tuple())
                    existing = [x for x in overrides_sheet.get(k1, []) if normalize_text(x) != ""]
                    if n1 not in existing:
                        existing.append(n1)
                    overrides_sheet[k1] = enforce_k_five(existing)

                    # L2|<n1>
                    k2 = level_key_tuple(2, (n1,))
                    overrides_sheet[k2] = enforce_k_five(vals_n2)

                    overrides_all[vm_sheet] = overrides_sheet
                    st.session_state[override_root] = overrides_all
                    _mark_session_edit(vm_sheet, k1)
                    _mark_session_edit(vm_sheet, k2)

                    # Build Raw+ (session scope)
                    edited_keys_for_sheet = set(st.session_state.get("session_edited_keys", {}).get(vm_sheet, []))
                    df_aug, stats = build_raw_plus_v630(df_in, overrides_sheet, include_scope="session", edited_keys_for_sheet=edited_keys_for_sheet)

                    # Persist in-session
                    vm_wb[vm_sheet] = df_aug
                    if src_code == "upload":
                        st.session_state["upload_workbook"] = vm_wb
                    else:
                        st.session_state["gs_workbook"] = vm_wb

                    st.success(f"VM '{vm_name}' with Node-1 '{n1}' created. Auto-cascade added {stats['new_added']} new unique rows; filled {stats['inplace_filled']} anchors.")

    st.markdown("---")

    # ----------------- 🧙 VM Build Wizard (stub) -----------------
    st.subheader("🧙 VM Build Wizard (step-by-step to Node 5) — Preview")
    st.caption("This is a lightweight scaffold. We’ll flesh out full step flows in a later patch.")

    wiz_state = st.session_state.get("vm_wizard", {"step": 1, "vm": "", "n1": [""]*5, "n2": {}, "target": "Upload workbook", "sheet": None})

    c1, c2 = st.columns([2,1])
    with c1:
        target_src = st.radio("Target source", ["Upload workbook", "Google Sheets workbook"], index=0 if st.session_state.get("upload_workbook", {}) else 1, horizontal=True, key="wiz_target_src")
        wiz_state["target"] = target_src
        if target_src == "Upload workbook":
            wbX = st.session_state.get("upload_workbook", {})
        else:
            wbX = st.session_state.get("gs_workbook", {})
        if not wbX:
            st.info("Load a workbook first to use the wizard.")
        else:
            wiz_state["sheet"] = st.selectbox("Target sheet", list(wbX.keys()), key="wiz_target_sheet")

    with c2:
        step = st.number_input("Step", min_value=1, max_value=5, value=wiz_state.get("step", 1), help="1: VM name, 2: Node-1 options, 3: Node-2 for each, etc.")
        wiz_state["step"] = int(step)

    # Step content (preview only)
    if wiz_state["step"] == 1:
        wiz_state["vm"] = st.text_input("Vital Measurement name", value=wiz_state.get("vm", ""))
        if st.button("Next ▶", key="wiz_next_1"):
            wiz_state["step"] = 2
    elif wiz_state["step"] == 2:
        cols = st.columns(5)
        for i in range(5):
            wiz_state["n1"][i] = cols[i].text_input(f"Node-1 option {i+1}", value=wiz_state["n1"][i], key=f"wiz_n1_{i}")
        if st.button("Next ▶", key="wiz_next_2"):
            wiz_state["step"] = 3
    elif wiz_state["step"] == 3:
        st.caption("Provide Node-2 options for each Node-1 that you filled.")
        for i, n1v in enumerate([x for x in wiz_state["n1"] if normalize_text(x)]):
            st.markdown(f"**Node-1: {n1v}**")
            c = st.columns(5)
            wiz_state["n2"].setdefault(n1v, [""]*5)
            for j in range(5):
                wiz_state["n2"][n1v][j] = c[j].text_input(f"{n1v} → Node-2 {j+1}", value=wiz_state["n2"][n1v][j], key=f"wiz_n2_{i}_{j}")
        if st.button("Next ▶", key="wiz_next_3"):
            wiz_state["step"] = 4
    else:
        st.info("Further steps (Node-3..5) will be added in the full wizard implementation.")

    st.session_state["vm_wizard"] = wiz_state
    st.caption("Note: The wizard does not write to the sheet yet; it’s a scaffold to be completed in the next patch.")
