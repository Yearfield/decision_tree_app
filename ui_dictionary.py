# ui_dictionary.py

import math
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd
import streamlit as st

from utils import (
    CANON_HEADERS, LEVEL_COLS,
    normalize_text, validate_headers,
)


# ----------------- helpers -----------------

def build_dictionary(dfs: List[pd.DataFrame]) -> Tuple[Dict[str, int], Dict[str, Set[int]]]:
    """Aggregate symptom labels across Node 1..5 from a list of dataframes."""
    counts: Dict[str, int] = {}
    levels_map: Dict[str, Set[int]] = {}
    for df0 in dfs:
        if df0 is None or df0.empty or not validate_headers(df0):
            continue
        for lvl, col in enumerate(LEVEL_COLS, start=1):
            if col in df0.columns:
                for val in df0[col].astype(str).map(normalize_text):
                    if not val:
                        continue
                    counts[val] = counts.get(val, 0) + 1
                    levels_map.setdefault(val, set()).add(lvl)
    return counts, levels_map


def highlight_text(s: str, q: str) -> str:
    """Return HTML with <mark> around case-insensitive matches of q inside s."""
    if not q:
        return s
    try:
        import re
        pattern = re.compile(re.escape(q), re.IGNORECASE)
        def _mark(m):
            return f"<mark>{m.group(0)}</mark>"
        return pattern.sub(_mark, s)
    except Exception:
        # Fallback: simple replace (case-sensitive)
        return s.replace(q, f"<mark>{q}</mark>")


# ----------------- UI -----------------

def render():
    st.header("📖 Dictionary")

    # Pick sources
    sources_avail = []
    if st.session_state.get("upload_workbook", {}):
        sources_avail.append("Upload workbook")
    if st.session_state.get("gs_workbook", {}):
        sources_avail.append("Google Sheets workbook")

    if not sources_avail:
        st.info("Load data in the **Source** tab first (upload a workbook or load Google Sheets).")
        return

    st.subheader("Sources")
    source_choice = st.multiselect(
        "Include sources",
        sources_avail,
        default=sources_avail,
        key="dict_sources"
    )

    dfs: List[pd.DataFrame] = []

    if "Upload workbook" in source_choice:
        wb_u = st.session_state.get("upload_workbook", {})
        pick_u = st.multiselect("Sheets (Upload)", list(wb_u.keys()), default=list(wb_u.keys()), key="dict_pick_u")
        for nm in pick_u:
            if nm in wb_u:
                dfs.append(wb_u[nm])

    if "Google Sheets workbook" in source_choice:
        wb_g = st.session_state.get("gs_workbook", {})
        pick_g = st.multiselect("Sheets (Google Sheets)", list(wb_g.keys()), default=list(wb_g.keys()), key="dict_pick_g")
        for nm in pick_g:
            if nm in wb_g:
                dfs.append(wb_g[nm])

    if not dfs:
        st.info("Select at least one sheet.")
        return

    # Build dictionary
    counts, levels_map = build_dictionary(dfs)
    if not counts:
        st.info("No symptom labels found.")
        return

    # Load existing quality map (in-session memory)
    quality_map = st.session_state.get("symptom_quality", {})  # {symptom: "Red Flag"|"Normal"}

    # Build base dataframe
    rows = []
    for symptom, cnt in counts.items():
        levels_list = sorted(list(levels_map.get(symptom, set())))
        quality = quality_map.get(symptom, "Normal")
        rows.append({
            "Symptom": symptom,
            "Count": int(cnt),
            "Levels": ", ".join([f"Node {i}" for i in levels_list]),
            "RedFlag": (quality == "Red Flag"),
        })

    dict_df = pd.DataFrame(rows).sort_values(["Symptom"]).reset_index(drop=True)

    # ----------------- Controls: search, filter, sort -----------------

    st.subheader("Search & Filter")

    c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
    with c1:
        q = st.text_input("Search symptom (case-insensitive)", key="dict_search").strip()
    with c2:
        show_only_rf = st.checkbox("Show only Red Flags", value=False, key="dict_only_rf")
    with c3:
        list_mode = st.checkbox("List mode with highlights", value=False, help="Show a simple list with <mark>-highlighted matches. Turn off for the editable table.")
    with c4:
        sort_mode = st.selectbox("Sort by", ["A → Z", "Count ↓", "Red Flags first"], index=0, key="dict_sort")

    # Filter
    view = dict_df.copy()
    if q:
        view = view[view["Symptom"].str.contains(q, case=False, na=False)]
    if show_only_rf:
        view = view[view["RedFlag"] == True]

    # Sort
    if sort_mode == "Count ↓":
        view = view.sort_values(["Count", "Symptom"], ascending=[False, True])
    elif sort_mode == "Red Flags first":
        view = view.sort_values(["RedFlag", "Symptom"], ascending=[False, True])
    else:
        view = view.sort_values(["Symptom"], ascending=[True])

    total_visible = len(view)
    rf_visible = int(view["RedFlag"].sum()) if total_visible else 0
    st.caption(f"{total_visible} symptoms match the current filters • Red Flags among visible: {rf_visible}")

    # ----------------- List mode (with highlight) -----------------
    if list_mode:
        st.markdown("### Results (highlighted)")
        # Simple paginated list
        page_size = st.selectbox("Items per page", [25, 50, 100, 200], index=1, key="dict_list_pagesize")
        total = len(view)
        max_page = max(1, math.ceil(total / page_size))
        page = st.number_input("Page", min_value=1, max_value=max_page, value=1, step=1, key="dict_list_page")
        start = (page - 1) * page_size
        end = min(start + page_size, total)
        slice_df = view.iloc[start:end]

        # Render
        for _, row in slice_df.iterrows():
            sym = row["Symptom"]
            hl = highlight_text(sym, q) if q else sym
            badge = "🔴" if row["RedFlag"] else "🟢"
            st.markdown(
                f"- {badge} <strong>{hl}</strong> &nbsp;·&nbsp; Count: **{int(row['Count'])}** &nbsp;·&nbsp; Levels: {row['Levels']}",
                unsafe_allow_html=True,
            )

        # CSV export of current view
        csv_data = view.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download current view (CSV)",
            data=csv_data,
            file_name="dictionary_filtered.csv",
            mime="text/csv",
        )

        # Also export just the Red Flags (visible)
        if rf_visible:
            rf_csv = view[view["RedFlag"] == True][["Symptom", "Levels", "Count"]].to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download visible Red Flags (CSV)",
                data=rf_csv,
                file_name="dictionary_red_flags.csv",
                mime="text/csv",
            )

        st.info("Switch off 'List mode' to edit Red Flags in a table.")
        return

    # ----------------- Editable table mode -----------------
    st.markdown("### Edit Red Flags")

    # Pagination
    page_size = st.selectbox("Rows per page", [25, 50, 100, 200], index=1, key="dict_table_pagesize")
    total = len(view)
    max_page = max(1, math.ceil(total / page_size))
    page = st.number_input("Page", min_value=1, max_value=max_page, value=1, step=1, key="dict_table_page")

    start = (page - 1) * page_size
    end = min(start + page_size, total)
    slice_df = view.iloc[start:end].reset_index(drop=True)

    # Quick actions
    qa1, qa2, qa3 = st.columns([1, 1, 3])
    with qa1:
        if st.button("Select all on page", key="dict_select_all_page"):
            slice_df["RedFlag"] = True
    with qa2:
        if st.button("Clear all on page", key="dict_clear_all_page"):
            slice_df["RedFlag"] = False
    with qa3:
        st.caption("Tip: Use filters and pagination to focus on specific subsets. Changes apply to the current filtered view.")

    edited = st.data_editor(
        slice_df,
        num_rows="fixed",
        use_container_width=True,
        hide_index=True,
        column_config={
            "RedFlag": st.column_config.CheckboxColumn("Red Flag"),
            "Symptom": st.column_config.TextColumn("Symptom", disabled=True),
            "Count": st.column_config.NumberColumn("Count", disabled=True),
            "Levels": st.column_config.TextColumn("Levels", disabled=True),
        },
        key="dict_editor",
    )

    # Save changes back to full view and then to session_state
    colS1, colS2, colS3 = st.columns([1, 2, 2])
    with colS1:
        if st.button("💾 Save changes", key="dict_save_changes"):
            # Apply edited flags back into 'view' via the slice indices
            view.loc[view.index[start:end], "RedFlag"] = edited["RedFlag"].values

            # Merge back into full dict_df by Symptom
            dict_df_updates = dict_df.set_index("Symptom")
            view_updates = view.set_index("Symptom")["RedFlag"]
            common = dict_df_updates.index.intersection(view_updates.index)
            dict_df_updates.loc[common, "RedFlag"] = view_updates.loc[common].values
            merged_df = dict_df_updates.reset_index()

            # Build quality map from merged result
            new_quality = {}
            for _, r in merged_df.iterrows():
                new_quality[r["Symptom"]] = "Red Flag" if bool(r["RedFlag"]) else "Normal"

            st.session_state["symptom_quality"] = new_quality

            rf_total = sum(1 for v in new_quality.values() if v == "Red Flag")
            st.success(f"Red Flags saved for this session. Total flagged: {rf_total}.")

    with colS2:
        # CSV export of the current filtered view
        csv_data = view.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download current view (CSV)",
            data=csv_data,
            file_name="dictionary_filtered.csv",
            mime="text/csv",
        )

    with colS3:
        # Also export just the visible Red Flags
        if rf_visible:
            rf_csv = view[view["RedFlag"] == True][["Symptom", "Levels", "Count"]].to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download visible Red Flags (CSV)",
                data=rf_csv,
                file_name="dictionary_red_flags.csv",
                mime="text/csv",
            )
