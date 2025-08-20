# ui/tabs/source.py
import streamlit as st
import pandas as pd
from typing import Dict, Any

from utils import (
    CANON_HEADERS, LEVEL_COLS, normalize_text, validate_headers
)
from utils.state import (
    set_active_workbook, get_active_workbook, set_current_sheet, 
    get_current_sheet, has_active_workbook, get_workbook_status
)
from logic.tree import build_raw_plus_v630 # Not fully used yet, but imported
from io_utils.sheets import read_google_sheet


def render():
    """Render the Source tab for loading and creating decision tree data."""
    try:
        st.header("📂 Source")
        st.markdown("Load/create your decision tree data.")
        
        # Status badge
        has_wb, sheet_count, current_sheet = get_workbook_status()
        if has_wb and current_sheet:
            st.caption(f"Workbook: ✅ {sheet_count} sheet(s) • Active: **{current_sheet}**")
        else:
            st.caption("Workbook: ❌ not loaded")

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
        
        # Validate that we got proper DataFrames
        from utils.state import coerce_workbook_to_dataframes
        clean_wb = coerce_workbook_to_dataframes(wb)
        if not clean_wb:
            st.error("Uploaded workbook did not contain any valid sheets (as DataFrames).")
            return
        
        # Store filename for display
        st.session_state["upload_filename"] = file.name
        
        # Set as active workbook using canonical API
        set_active_workbook(clean_wb, source="upload")
        
        # Clear stale caches to ensure immediate refresh
        st.cache_data.clear()
        
        # Sanity assertions (temporary; safe to remove later)
        from utils.state import get_active_workbook, get_current_sheet
        wb_check, sheet_check = get_active_workbook(), get_current_sheet()
        assert wb_check is not None, "active workbook missing after upload"
        assert isinstance(wb_check, dict), "active workbook should be Dict[str, DataFrame]"
        assert sheet_check is None or sheet_check in wb_check, "current_sheet must be a key of active workbook"
        
        st.success(f"Loaded {len(clean_wb)} sheet(s) into session.")


def _render_google_sheets_section():
    """Render the Google Sheets loading section."""
    st.subheader("🔄 Google Sheets")
    
    # Re-sync button for current sheet
    if has_active_workbook() and get_current_sheet():
        if st.button("🔄 Re-sync current sheet"):
            sheet_id = st.session_state.get("sheet_id")
            sheet_name = st.session_state.get("sheet_name")
            if sheet_id and sheet_name:
                try:
                    if "gcp_service_account" not in st.secrets:
                        st.error("Google Sheets not configured. Add your service account JSON under [gcp_service_account].")
                    else:
                        # Re-read sheet and replace that entry in active workbook
                        with st.spinner("Re-syncing..."):
                            new_df = read_google_sheet(sheet_id, sheet_name, st.secrets["gcp_service_account"])
                            if not new_df.empty:
                                wb = get_active_workbook() or {}
                                wb[sheet_name] = new_df
                                set_active_workbook(wb, source="sheets")
                                
                                # Clear stale caches to ensure immediate refresh
                                st.cache_data.clear()
                                
                                # Sanity assertions (temporary; safe to remove later)
                                from utils.state import get_active_workbook, get_current_sheet
                                wb_check, sheet_check = get_active_workbook(), get_current_sheet()
                                assert wb_check is not None, "active workbook missing after re-sync"
                                assert isinstance(wb_check, dict), "active workbook should be Dict[str, DataFrame]"
                                assert sheet_check is None or sheet_check in wb_check, "current_sheet must be a key of active workbook"
                                
                                st.success(f"Re-synced '{sheet_name}' from Google Sheets.")
                            else:
                                st.warning("Selected sheet is empty or not found.")
                except Exception as e:
                    st.error(f"Google Sheets error: {e}")
            else:
                st.info("No stored Google Sheet ID/name to re-sync.")
    
    sid = st.text_input("Spreadsheet ID", value=st.session_state.get("sheet_id", ""))
    if sid:
        st.session_state["sheet_id"] = sid
    
    gs_sheet = st.text_input("Sheet name to load (e.g., BP)", value=st.session_state.get("sheet_name", "BP"))
    if gs_sheet:
        st.session_state["sheet_name"] = gs_sheet
    
    if st.button("Load / Refresh from Google Sheets"):
        try:
            if "gcp_service_account" not in st.secrets:
                st.error("Google Sheets not configured. Add your service account JSON under [gcp_service_account].")
            else:
                df_g = read_google_sheet(sid, gs_sheet, st.secrets["gcp_service_account"])
                if df_g.empty and gs_sheet:
                    st.warning("Selected sheet is empty or not found.")
                else:
                    # Validate that we got a proper DataFrame
                    if not isinstance(df_g, pd.DataFrame):
                        st.error("Google Sheets loader returned invalid data type. Expected DataFrame.")
                        return
                    
                    # Create workbook with the loaded sheet
                    wb_g = {gs_sheet: df_g}
                    
                    # Use the new verification utilities to ensure proper DataFrame storage
                    from utils.state import set_active_workbook, verify_active_workbook, coerce_workbook_to_dataframes
                    
                    clean_wb = coerce_workbook_to_dataframes(wb_g)
                    if not clean_wb:
                        st.error("Loaded workbook contained no valid (non-empty) sheets. Check the sheet content.")
                        return
                    
                    # Choose default sheet deterministically:
                    default_sheet = gs_sheet if gs_sheet in clean_wb else list(clean_wb.keys())[0]
                    
                    # Store sheet metadata
                    st.session_state["sheet_id"] = sid
                    st.session_state["sheet_name"] = default_sheet
                    
                    # Set as active workbook using canonical API
                    set_active_workbook(clean_wb, default_sheet=default_sheet, source="sheets")
                    
                    # Clear stale caches - the nonce-based system should handle this automatically,
                    # but we'll clear explicitly to ensure immediate refresh
                    st.cache_data.clear()
                    
                    # Show verification report for debugging
                    rep = verify_active_workbook()
                    with st.expander("Workbook verification (loader)", expanded=False):
                        st.json(rep)
                    
                    # Sanity assertions (temporary; safe to remove later)
                    from utils.state import get_active_workbook, get_current_sheet
                    wb, sheet = get_active_workbook(), get_current_sheet()
                    assert wb is not None, "active workbook missing after load"
                    assert isinstance(wb, dict), "active workbook should be Dict[str, DataFrame]"
                    assert sheet is None or sheet in wb, "current_sheet must be a key of active workbook"
                    
                    # As a sanity test for BP sheet
                    if "BP" in clean_wb:
                        st.write("BP head():")
                        st.dataframe(clean_wb["BP"].head(5), use_container_width=True)
                    
                    st.success(f"Loaded '{default_sheet}' from Google Sheets and stored in session state.")
        except Exception as e:
            st.error(f"Google Sheets error: {e}")


def _render_vm_builder_section():
    """Render the VM Builder section."""
    st.subheader("🧩 VM Builder (add VMs to an existing sheet)")
    
    # Check active workbook status
    active_wb = get_active_workbook()
    current_sheet = get_current_sheet()
    
    if not active_wb:
        st.info("No active workbook. Paste a Google Sheet ID in this tab or upload a file.")
        return
    
    # Show active workbook status
    st.info(f"Active workbook detected. VM Builder will write overrides to the current sheet ({current_sheet}).")
    
    # Use the active workbook directly instead of legacy selection
    target_wb = active_wb
    tgt_sheet = current_sheet
    
    # Allow sheet selection if multiple sheets exist
    if len(target_wb) > 1:
        tgt_sheet = st.selectbox("Target sheet", list(target_wb.keys()), index=list(target_wb.keys()).index(current_sheet), key="vm_builder_sheet")
    
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
            df = target_wb.get(tgt_sheet, pd.DataFrame(columns=CANON_HEADERS)).copy()
            for vm in vm_list:
                new_row = {c: "" for c in CANON_HEADERS}
                new_row["Vital Measurement"] = vm
                df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            
            # Update the active workbook directly
            target_wb[tgt_sheet] = df
            set_active_workbook(target_wb, source="vm_builder")
            
            # Clear stale caches to ensure immediate refresh
            st.cache_data.clear()
            
            st.session_state["vm_builder_queue"] = []
            st.success(f"Added {len(vm_list)} VM base row(s) to '{tgt_sheet}'. Use Symptoms/Conflicts or Wizard to build branches.")
    else:
        st.caption("No VMs queued yet.")


def _render_new_sheet_wizard_section():
    """Render the New Sheet Wizard section."""
    st.subheader("🧙 New Sheet Wizard (create sheet + seed branches)")
    
    # Use canonical workbook instead of legacy access
    active_wb = get_active_workbook()
    if not active_wb:
        st.info("No active workbook. Please load a workbook first (upload or Google Sheets).")
        return
    
    wiz_src = "Active workbook"  # Simplified since we're using canonical workbook
    wiz_wb = active_wb  # Use the canonical workbook directly
    
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
                
                # Set as active workbook using canonical API
                set_active_workbook(wiz_wb, default_sheet=name, source="wizard")
                
                # Clear stale caches to ensure immediate refresh
                st.cache_data.clear()
                
                # Set context to new sheet
                st.session_state["work_context"] = {"source": "wizard", "sheet": name}

                st.success(
                    f"Created '{name}' with {len(vms)} VM(s). "
                    f"Use Symptoms/Conflicts or Wizard to build branches."
                )
                # Reset wizard
                for k in list(st.session_state.keys()):
                    if k.startswith("wiz_"):
                        del st.session_state[k]
