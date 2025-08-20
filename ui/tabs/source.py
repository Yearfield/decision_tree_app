# ui/tabs/source.py
import streamlit as st
import pandas as pd
from typing import Dict, Any

from utils import (
    CANON_HEADERS, LEVEL_COLS, normalize_text, validate_headers
)
from logic.tree import build_raw_plus_v630 # Not fully used yet, but imported
from io_utils.sheets import read_google_sheet


def render():
    """Render the Source tab for loading and creating decision tree data."""
    try:
        st.header("📂 Source")
        st.markdown("Load/create your decision tree data.")

        _render_upload_section()
        st.markdown("---")
        _render_google_sheets_section()
        st.markdown("---")
        _render_vm_builder_section()
        st.markdown("---")
        # New Sheet Wizard
        _render_new_sheet_wizard_section()

    except Exception as e:
        st.exception(e)


def _has_active_workbook() -> bool:
    """Check if there's an active workbook in session state."""
    upload_wb = st.session_state.get("upload_workbook", {})
    gs_wb = st.session_state.get("gs_workbook", {})
    return bool(upload_wb or gs_wb)


def _render_upload_section():
    """Render the file upload section."""
    st.subheader("📤 Upload Workbook")
    file = st.file_uploader("Upload XLSX or CSV", type=["xlsx", "xls", "csv"])
    
    if file is not None:
        if file.name.lower().endswith(".csv"):
            df = pd.read_csv(file)
            for c in CANON_HEADERS:
                if c not in df.columns:
                    df[c] = ""
            df = df[CANON_HEADERS]
            node_block = ["Vital Measurement"] + LEVEL_COLS
            df = df[~df[node_block].apply(lambda r: all(normalize_text(v) == "" for v in r), axis=1)].copy()
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
                sheets[name] = dfx
            wb = sheets
        
        st.session_state["upload_workbook"] = wb
        st.session_state["upload_filename"] = file.name
        st.success(f"Loaded {len(wb)} sheet(s) into session.")


def _render_google_sheets_section():
    """Render the Google Sheets loading section."""
    st.subheader("🔄 Google Sheets")
    sid = st.text_input("Spreadsheet ID", value=st.session_state.get("gs_spreadsheet_id", ""))
    if sid:
        st.session_state["gs_spreadsheet_id"] = sid
    
    gs_sheet = st.text_input("Sheet name to load (e.g., BP)", value=st.session_state.get("gs_default_sheet", "BP"))
    
    if st.button("Load / Refresh from Google Sheets"):
        try:
            if "gcp_service_account" not in st.secrets:
                st.error("Google Sheets not configured. Add your service account JSON under [gcp_service_account].")
            else:
                df_g = read_google_sheet(sid, gs_sheet, st.secrets["gcp_service_account"])
                if df_g.empty and gs_sheet:
                    st.warning("Selected sheet is empty or not found.")
                else:
                    wb_g = st.session_state.get("gs_workbook", {})
                    wb_g[gs_sheet] = df_g
                    st.session_state["gs_workbook"] = wb_g
                    st.session_state["gs_default_sheet"] = gs_sheet
                    st.success(f"Loaded '{gs_sheet}' from Google Sheets.")
        except Exception as e:
            st.error(f"Google Sheets error: {e}")


def _render_vm_builder_section():
    """Render the VM Builder section."""
    st.subheader("🧩 VM Builder (add VMs to an existing sheet)")
    
    # Pick target workbook + sheet
    vm_target_src = st.radio("Target workbook", ["Upload workbook", "Google Sheets workbook"], horizontal=True, key="vm_builder_target")
    target_wb = (st.session_state.get("upload_workbook", {}) if vm_target_src == "Upload workbook" 
                 else st.session_state.get("gs_workbook", {}))
    
    if not target_wb:
        st.info("Load a workbook first above.")
    else:
        tgt_sheet = st.selectbox("Target sheet", list(target_wb.keys()), key="vm_builder_sheet")
        add_vm_cols = st.columns([3, 1])
        
        with add_vm_cols[0]:
            vm_new = st.text_input("Add VM (free type; press Enter to enqueue)", key="vm_builder_new")
        with add_vm_cols[1]:
            if st.button("➕ Enqueue VM"):
                vm_list = st.session_state.get("vm_builder_queue", [])
                if normalize_text(vm_new):
                    vm_list.append(normalize_text(vm_new))
                    st.session_state["vm_builder_queue"] = vm_list
                    st.session_state["vm_builder_new"] = ""
        
        vm_list = st.session_state.get("vm_builder_queue", [])
        if vm_list:
            st.write("Queued VMs:", ", ".join(vm_list))
            if st.button("Create rows for these VMs"):
                wb = (st.session_state.get("upload_workbook", {}) if vm_target_src == "Upload workbook" 
                      else st.session_state.get("gs_workbook", {}))
                df = wb.get(tgt_sheet, pd.DataFrame(columns=CANON_HEADERS)).copy()
                for vm in vm_list:
                    new_row = {c: "" for c in CANON_HEADERS}
                    new_row["Vital Measurement"] = vm
                    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
                wb[tgt_sheet] = df
                if vm_target_src == "Upload workbook":
                    st.session_state["upload_workbook"] = wb
                else:
                    st.session_state["gs_workbook"] = wb
                st.session_state["vm_builder_queue"] = []
                st.success(f"Added {len(vm_list)} VM base row(s) to '{tgt_sheet}'. Use Symptoms/Conflicts or Wizard to build branches.")
        else:
            st.caption("No VMs queued yet.")


def _render_new_sheet_wizard_section():
    """Render the New Sheet Wizard section."""
    st.subheader("🧙 New Sheet Wizard (create sheet + seed branches)")
    
    wiz_src = st.radio("Create in workbook", ["Upload workbook", "Google Sheets workbook"], horizontal=True, key="wiz_target_src")
    wiz_wb = (st.session_state.get("upload_workbook", {}) if wiz_src == "Upload workbook" 
              else st.session_state.get("gs_workbook", {}))
    
    # Step 1: name sheet
    if "wiz_step" not in st.session_state:
        st.session_state["wiz_step"] = 1
    step = st.session_state.get("wiz_step", 1)
    st.caption(f"Step {step} of 4")

    if step == 1:
        name = st.text_input("New sheet name", key="wiz_sheet_name")
        if st.button("Next →", key="wiz_next1"):
            if not normalize_text(name):
                st.warning("Enter a sheet name.")
            elif name in wiz_wb:
                st.warning("A sheet with this name already exists.")
            else:
                st.session_state["wiz_step"] = 2

    if step >= 2:
        st.markdown("**Step 2 — Add root VMs**")
        cols_vm = st.columns([3, 1])
        with cols_vm[0]:
            vm_entry = st.text_input("Add VM (free type; press Enter to enqueue)", key="wiz_vm_entry")
        with cols_vm[1]:
            if st.button("➕ Add VM", key="wiz_add_vm"):
                lst = st.session_state.get("wiz_vms", [])
                if normalize_text(vm_entry):
                    lst.append(normalize_text(vm_entry))
                    st.session_state["wiz_vms"] = lst
                    st.session_state["wiz_vm_entry"] = ""
        
        vms = st.session_state.get("wiz_vms", [])
        if vms:
            st.write("VMs:", ", ".join(vms))
        
        col_nav = st.columns([1, 1, 1])
        with col_nav[0]:
            if st.button("Back", key="wiz_back2"):
                st.session_state["wiz_step"] = 1
        with col_nav[2]:
            if st.button("Next →", key="wiz_next2"):
                if not vms:
                    st.warning("Please add at least one VM.")
                else:
                    st.session_state["wiz_step"] = 3

    if step >= 3:
        st.markdown("**Step 3 — Choose Node 1 (5 children for this sheet)**")
        # Simplified vocabulary building for now
        vocab = ["suggestion1", "suggestion2"]  # Placeholder
        sugg = ["(pick suggestion)"] + vocab
        n1_vals = []
        
        for i in range(5):
            c1, c2 = st.columns([2, 1])
            with c1:
                txt = st.text_input(f"Node 1 — option {i+1}", key=f"wiz_n1_txt_{i}")
            with c2:
                sel = st.selectbox("Suggest", options=sugg, index=0, key=f"wiz_n1_sel_{i}")
            n1_vals.append(normalize_text(sel if sel != "(pick suggestion)" else txt))
        
        st.caption("You can leave blanks; materializer will enforce 5 by ignoring empties.")
        col_nav3 = st.columns([1, 1, 1])
        with col_nav3[0]:
            if st.button("Back", key="wiz_back3"):
                st.session_state["wiz_step"] = 2
        with col_nav3[2]:
            if st.button("Next →", key="wiz_next3"):
                if not any(normalize_text(x) for x in n1_vals):
                    st.warning("Enter at least one Node 1 child.")
                else:
                    st.session_state["wiz_n1_vals"] = n1_vals
                    st.session_state["wiz_step"] = 4

    if step >= 4:
        st.markdown("**Step 4 — (Optional) Node 2 for each Node 1**")
        n1_vals = st.session_state.get("wiz_n1_vals", [])
        vocab = ["suggestion1", "suggestion2"]  # Placeholder
        sugg = ["(pick suggestion)"] + vocab
        node2_map: Dict[str, list] = {}

        for n1 in [x for x in n1_vals if normalize_text(x)]:
            with st.expander(f"Node 1 = {n1} → set 5 Node 2 children", expanded=False):
                vals = []
                for i in range(5):
                    c1, c2 = st.columns([2, 1])
                    with c1:
                        txt = st.text_input(f"{n1} → Node 2 option {i+1}", key=f"wiz_n2_txt_{n1}_{i}")
                    with c2:
                        sel = st.selectbox("Suggest", options=sugg, index=0, key=f"wiz_n2_sel_{n1}_{i}")
                    vals.append(normalize_text(sel if sel != "(pick suggestion)" else txt))
                node2_map[n1] = vals

        col_nav4 = st.columns([1, 1, 2])
        with col_nav4[0]:
            if st.button("Back", key="wiz_back4"):
                st.session_state["wiz_step"] = 3
        with col_nav4[2]:
            if st.button("✅ Create sheet", key="wiz_create"):
                # Create empty DF with one base row per VM
                name = st.session_state.get("wiz_sheet_name", "NewSheet")
                vms = st.session_state.get("wiz_vms", [])
                df_new = pd.DataFrame(columns=CANON_HEADERS)
                
                for vm in vms:
                    new_row = {c: "" for c in CANON_HEADERS}
                    new_row["Vital Measurement"] = normalize_text(vm)
                    df_new = pd.concat([df_new, pd.DataFrame([new_row])], ignore_index=True)

                # Save overrides for the new sheet
                ov_all = st.session_state.get("branch_overrides", {})
                ov_sheet = {}
                # Node 1:
                ov_sheet[(1, tuple())] = [x for x in n1_vals if normalize_text(x)]
                # Node 2:
                for n1, kids in node2_map.items():
                    if any(normalize_text(k) for k in kids):
                        ov_sheet[(2, (n1,))] = [x for x in kids if normalize_text(x)]
                ov_all[name] = ov_sheet
                st.session_state["branch_overrides"] = ov_all

                # Attach to workbook
                wiz_wb[name] = df_new
                if wiz_src == "Upload workbook":
                    st.session_state["upload_workbook"] = wiz_wb
                else:
                    st.session_state["gs_workbook"] = wiz_wb

                # Set context to new sheet
                st.session_state["work_context"] = {"source": "upload" if wiz_src == "Upload workbook" else "gs", "sheet": name}

                st.success(
                    f"Created '{name}' with {len(vms)} VM(s). "
                    f"Use Symptoms/Conflicts or Wizard to build branches."
                )
                # Reset wizard
                for k in list(st.session_state.keys()):
                    if k.startswith("wiz_"):
                        del st.session_state[k]
