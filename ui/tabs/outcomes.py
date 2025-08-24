# ui/tabs/outcomes.py
from typing import Dict, List, Tuple, Optional
import pandas as pd
import streamlit as st
import utils.state as USTATE
from ui.utils.debug import dump_state, banner

from utils.helpers import normalize_text
from ui.utils.rerun import safe_rerun


def _get_current_df_and_sheet():
    """Get current DataFrame and sheet from active context."""
    # Try to get from active workbook first
    wb = USTATE.get_active_workbook()
    sheet = USTATE.get_current_sheet()
    if wb and sheet and sheet in wb:
        df = wb[sheet]
        if df is not None and not df.empty:
            return df, sheet, "active"
    
    # Fall back to session state workbooks
    ctx = st.session_state.get("work_context", {})
    src = ctx.get("source")
    sheet = ctx.get("sheet")
    
    if src == "gs":
        wb = st.session_state.get("gs_workbook", {})
        if sheet in wb:
            return wb[sheet], sheet, "gs"
    elif src == "upload":
        wb = st.session_state.get("upload_workbook", {})
        if sheet in wb:
            return wb[sheet], sheet, "upload"
    
    # Last resort: use any available workbook
    wb_g = st.session_state.get("gs_workbook", {})
    if wb_g:
        name = next(iter(wb_g))
        return wb_g[name], name, "gs"
    
    wb_u = st.session_state.get("upload_workbook", {})
    if wb_u:
        name = next(iter(wb_u))
        return wb_u[name], name, "upload"
    
    return None, None, None


def render():
    """Render the Outcomes tab for managing diagnostic triage and actions."""
    
    # Add guard and debug expander
    from ui.utils.guards import ensure_active_workbook_and_sheet
    ok, df = ensure_active_workbook_and_sheet("Outcomes")
    if not ok:
        return
    
    # Debug state expander
    import json
    with st.expander("🛠 Debug: Session State (tab)", expanded=False):
        ss = {k: type(v).__name__ for k,v in st.session_state.items()}
        st.code(json.dumps(ss, indent=2))
    
    banner("Outcomes RENDER ENTRY")
    dump_state("Session (pre-outcomes)")
    
    try:
        st.header("📝 Outcomes")
        
        # Get current sheet name for display
        sheet = USTATE.get_current_sheet()
        
        # Check if we have a workbook before proceeding
        df, status, detail = USTATE.get_active_df_safe()
        
        if status != "ok":
            if status == "no_wb":
                st.info("📂 **No workbook loaded yet**")
                st.markdown("""
                **To get started:**
                1. Go to the **Source tab** (first tab)
                2. **Upload a workbook** or **connect to Google Sheets**
                3. **Select a sheet** to work with
                
                Once a workbook is loaded, this tab will show your decision tree data.
                """)
                return
            else:
                st.warning(f"Outcomes not ready: {status} — {detail}")
                return

        # Guard against None DataFrame
        if df is None:
            st.warning("Outcomes: Active DataFrame is None. (Did the upload complete and was a sheet selected?)")
            dump_state("Session (df is None)")
            return
        else:
            st.caption(f"Outcomes: df shape = {getattr(df, 'shape', None)}")

    except Exception as dbg_e:
        st.error(f"🔥 Debug tracer failed: {dbg_e}")
        st.exception(dbg_e)
    
    try:
        st.header("📝 Outcomes — Diagnostic Triage & Actions")

        # Use the df we already validated above, but also check the legacy function
        df_legacy, sheet_legacy, src_code = _get_current_df_and_sheet()
        if df_legacy is None or sheet_legacy is None:
            st.info("Select a sheet in **Workspace** (or load one in **Source**).")
            st.warning("🚧 Outcomes: Exiting early guard check")
            return

        # Build a wb-like view from the active source so the sheet pickers still work
        if src_code == "gs":
            wb = st.session_state.get("gs_workbook", {})
            where = "gs"
        elif src_code == "upload":
            wb = st.session_state.get("upload_workbook", {})
            where = "upload"
        else:
            # active context
            wb = get_active_workbook() or {}
            where = "active"
        
        if not wb or sheet not in wb:
            # fallback: synthesize a single-sheet wb from the active df
            wb = {sheet: df}
        
        # Ensure columns exist before editing, using CANON_HEADERS order
        CANON_HEADERS = ["Vital Measurement","Node 1","Node 2","Node 3","Node 4","Node 5","Diagnostic Triage","Actions"]
        df_cur = wb.get(sheet, pd.DataFrame()).copy()
        for c in CANON_HEADERS:
            if c not in df_cur.columns:
                df_cur[c] = ""
        df_cur = df_cur[CANON_HEADERS].copy()

        st.markdown("Filter rows that need Outcomes:")
        colf1, colf2, colf3, colf4 = st.columns([1,1,1,2])
        with colf1:
            miss_triage = st.checkbox("Missing Triage", value=True)
        with colf2:
            miss_actions = st.checkbox("Missing Actions", value=True)
        with colf3:
            only_full_paths = st.checkbox("Only full Node1..Node5", value=False)
        with colf4:
            search = st.text_input("Search (VM or any Node)")

        view = df_cur.copy()
        if only_full_paths:
            view = view[(view[["Node 1","Node 2","Node 3","Node 4","Node 5"]] != "").all(axis=1)]
        mask = pd.Series(True, index=view.index)
        if miss_triage:
            mask &= (view["Diagnostic Triage"].astype(str).str.strip() == "")
        if miss_actions:
            mask &= (view["Actions"].astype(str).str.strip() == "")
        if search.strip():
            q = search.strip().lower()
            cols = ["Vital Measurement","Node 1","Node 2","Node 3","Node 4","Node 5"]
            mask &= view[cols].apply(lambda r: any(q in str(v).lower() for v in r), axis=1)
        view = view[mask]

        st.caption(f"{len(view)} row(s) match the current filters.")
        if len(view) == 0:
            st.success("No incomplete rows with current filters.")
            return



        st.markdown("### Per-row edit (first 200)")
        edit_cols = ["Vital Measurement","Node 1","Node 2","Node 3","Node 4","Node 5","Diagnostic Triage","Actions"]
        
        # Start around current position if available
        incomplete_idxs = list(view.index)
        if incomplete_idxs and len(incomplete_idxs) > 0:
            pos_key = f"out_pos_{sheet}"
            cur_pos = st.session_state.get(pos_key, 0) % len(incomplete_idxs)
            slice_order = incomplete_idxs[cur_pos:] + incomplete_idxs[:cur_pos]
            slice_df = view.loc[slice_order].head(200)[edit_cols].copy()
        else:
            slice_df = view.head(200)[edit_cols].copy()
        edited = st.data_editor(
            slice_df,
            use_container_width=True,
            num_rows="dynamic",
            column_config={
                "Vital Measurement": st.column_config.TextColumn("Vital Measurement", disabled=True),
                "Node 1": st.column_config.TextColumn("Node 1", disabled=True),
                "Node 2": st.column_config.TextColumn("Node 2", disabled=True),
                "Node 3": st.column_config.TextColumn("Node 3", disabled=True),
                "Node 4": st.column_config.TextColumn("Node 4", disabled=True),
                "Node 5": st.column_config.TextColumn("Node 5", disabled=True),
                "Diagnostic Triage": st.column_config.TextColumn("Diagnostic Triage"),
                "Actions": st.column_config.TextColumn("Actions"),
            }
        )
        if st.button("💾 Save per-row Outcomes (above table)"):
            df_new = df_cur.copy()
            # align back by original index (positionally over the slice)
            for i, (_, row) in enumerate(edited.iterrows()):
                orig_idx = slice_df.index[i]
                df_new.at[orig_idx, "Diagnostic Triage"] = row["Diagnostic Triage"]
                df_new.at[orig_idx, "Actions"] = row["Actions"]
            wb[sheet] = df_new
            if where == "upload":
                st.session_state["upload_workbook"] = wb
            elif where == "gs":
                st.session_state["gs_workbook"] = wb
            else:
                # active context - update the active workbook
                set_active_workbook(wb, source="outcomes_rows")
            st.success("Per-row Outcomes saved.")

        st.markdown("### Navigate")

        # Build stable list of incomplete row indices under current filters
        incomplete_idxs = list(view.index)

        # Build full list of all incomplete rows in the whole sheet (ignores filters)
        all_incomplete_idxs = df.index[
            (df["Diagnostic Triage"].astype(str).str.strip() == "") |
            (df["Actions"].astype(str).str.strip() == "")
        ].tolist()

        if not incomplete_idxs:
            st.info("No incomplete rows match the current filters.")
        else:
            # Remember position within the current incomplete set
            pos_key = f"out_pos_{sheet}"
            if pos_key not in st.session_state:
                st.session_state[pos_key] = 0

            # Current index within the filtered set
            cur_pos = st.session_state[pos_key] % len(incomplete_idxs)
            cur_df_index = incomplete_idxs[cur_pos]

            coln1, coln2, coln3 = st.columns([1,1,2])
            with coln1:
                if st.button("➡️ Next incomplete"):
                    st.session_state[pos_key] = (cur_pos + 1) % len(incomplete_idxs)
                    st.warning("⚠️ Rerun skipped for debugging")
                    # st.rerun()

            with coln2:
                if st.button("🎲 Random incomplete (anywhere in sheet)"):
                    import random
                    if all_incomplete_idxs:
                        choice = random.choice(all_incomplete_idxs)
                        # Try to set pos_key to this choice if it's in the current filtered list
                        if choice in incomplete_idxs:
                            st.session_state[pos_key] = incomplete_idxs.index(choice)
                        else:
                            # If not in filtered view, reset filter position to 0 and toast info
                            st.session_state[pos_key] = 0
                            st.toast("Random row not in current filter; showing first filtered row instead.")
                        st.session_state["out_cur_idx_random"] = choice
                    st.warning("⚠️ Rerun skipped for debugging")
                    # st.rerun()

            with coln3:
                st.caption(f"Current incomplete row (DataFrame index): **{cur_df_index}**  •  {cur_pos+1}/{len(incomplete_idxs)}")

    except Exception as e:
        st.error(f"Exception in Outcomes.render(): {e}")
        st.exception(e)
