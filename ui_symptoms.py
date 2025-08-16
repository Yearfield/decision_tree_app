# ui_symptoms.py

from typing import Dict, List, Tuple, Optional, Set
import pandas as pd
import numpy as np
import streamlit as st

from utils import (
    CANON_HEADERS, LEVEL_COLS, MAX_LEVELS,
    normalize_text, validate_headers,
    infer_branch_options_with_overrides, infer_branch_options,
    level_key_tuple, enforce_k_five, friendly_parent_label,
)
from logic_cascade import build_raw_plus_v630


# ----------------- session helpers -----------------

def _ss_get(key, default):
    if key not in st.session_state:
        st.session_state[key] = default
    return st.session_state[key]

def _mark_session_edit(sheet: str, keyname: str):
    ek = _ss_get("session_edited_keys", {})
    cur = set(ek.get(sheet, []))
    cur.add(keyname)
    ek[sheet] = list(cur)
    st.session_state["session_edited_keys"] = ek


# ----------------- local builders -----------------

def _compute_virtual_parents(store: Dict[str, List[str]]) -> Dict[int, Set[Tuple[str, ...]]]:
    """
    From a store (parent->children), compute all parent tuples that *exist or are implied*
    at each level (1..MAX_LEVELS). Level 1 parent is the empty tuple ().
    """
    parents_by_level: Dict[int, Set[Tuple[str, ...]]] = {i: set() for i in range(1, MAX_LEVELS + 1)}
    parents_by_level[1].add(tuple())  # <ROOT>

    # Walk down using explicit store children
    for L in range(1, MAX_LEVELS):
        for p in list(parents_by_level[L]):
            key = level_key_tuple(L, p)
            children = [x for x in (store.get(key, []) or []) if normalize_text(x) != ""]
            for c in children:
                parents_by_level[L + 1].add(p + (c,))

    # Include explicit keys directly (ensures coverage even if the parent had no path continuity)
    for key in list(store.keys()):
        if "|" not in key:
            continue
        lvl_s, path = key.split("|", 1)
        try:
            L = int(lvl_s[1:])
        except Exception:
            continue
        if not (1 <= L <= MAX_LEVELS):
            continue
        parent_tuple = tuple([] if path == "<ROOT>" else path.split(">"))
        parents_by_level[L].add(parent_tuple)
        # ensure all prefix parents exist too
        for k in range(1, min(L, MAX_LEVELS) + 1):
            parents_by_level[k].add(tuple(parent_tuple[:k - 1]))

    return parents_by_level


def _build_vocabulary(df0: pd.DataFrame) -> List[str]:
    """
    Build a sorted vocabulary across Node 1..5 in the given DataFrame.
    """
    vocab = set()
    for col in LEVEL_COLS:
        if col in df0.columns:
            for x in df0[col].dropna().astype(str):
                x = normalize_text(x)
                if x:
                    vocab.add(x)
    return sorted(vocab)


# ----------------- UI: Symptoms Browser & Editor -----------------

def render():
    st.header("🧬 Symptoms")

    # Choose data source
    sources = []
    if _ss_get("upload_workbook", {}):
        sources.append("Upload workbook")
    if _ss_get("gs_workbook", {}):
        sources.append("Google Sheets workbook")

    if not sources:
        st.info("Load a workbook first in the **Source** tab.")
        return

    source = st.radio("Choose data source", sources, horizontal=True, key="sym_source_sel")

    if source == "Upload workbook":
        wb = _ss_get("upload_workbook", {})
        override_root = "branch_overrides_upload"
    else:
        wb = _ss_get("gs_workbook", {})
        override_root = "branch_overrides_gs"

    if not wb:
        st.warning("No sheets found in the selected source.")
        return

    # Sheet selector
    sheet = st.selectbox("Sheet", list(wb.keys()), key="sym_sheet_sel")
    df = wb.get(sheet, pd.DataFrame())

    if df.empty or not validate_headers(df):
        st.info("Selected sheet is empty or headers mismatch.")
        return

    # Build store + parents
    overrides_all = _ss_get(override_root, {})
    overrides_sheet = overrides_all.get(sheet, {})
    store = infer_branch_options_with_overrides(df, overrides_sheet)

    # Undo stack controls
    if st.button("↩️ Undo last Symptoms edit", key="sym_undo_btn"):
        stack = _ss_get("undo_stack", [])
        if not stack:
            st.info("Nothing to undo.")
        else:
            last = stack.pop()
            st.session_state["undo_stack"] = stack
            if last.get("context") == "symptoms" and last.get("sheet") == sheet and last.get("override_root") == override_root:
                # restore overrides + df
                all_over = _ss_get(override_root, {})
                all_over[sheet] = last.get("overrides_sheet_before", {})
                st.session_state[override_root] = all_over
                if last.get("df_before") is not None:
                    wb[sheet] = last["df_before"]
                    if source == "Upload workbook":
                        st.session_state["upload_workbook"] = wb
                    else:
                        st.session_state["gs_workbook"] = wb
                st.success(f"Restored previous state for '{sheet}'.")
            else:
                st.info("Last undo snapshot was for a different tab/sheet.")

    # Controls: Level to inspect
    level = st.selectbox("Level to inspect (child options of...)", [1, 2, 3, 4, 5],
                         format_func=lambda x: f"Node {x}", key="sym_level_sel")

    # Compute parents by level (virtual)
    parents_by_level = _compute_virtual_parents(store)

    # Persistent search helpers
    _pending = st.session_state.pop("sym_search_pending", None)
    if _pending is not None:
        st.session_state["sym_search"] = _pending

    top_cols = st.columns([2, 1, 1, 1, 2])
    with top_cols[0]:
        search = st.text_input("Search parent symptom/path", key="sym_search").strip().lower()
    with top_cols[1]:
        if st.button("Next Missing", key="sym_next_missing"):
            for pt in sorted(parents_by_level.get(level, set())):
                key = level_key_tuple(level, pt)
                if len([x for x in (store.get(key, []) or []) if normalize_text(x) != ""]) == 0:
                    st.session_state["sym_search_pending"] = (" > ".join(pt) or friendly_parent_label(level, tuple()))
                    st.experimental_rerun()
    with top_cols[2]:
        if st.button("Next Symptom left out", key="sym_next_leftout"):
            for pt in sorted(parents_by_level.get(level, set())):
                key = level_key_tuple(level, pt)
                n = len([x for x in (store.get(key, []) or []) if normalize_text(x) != ""])
                if 1 <= n < 5:
                    st.session_state["sym_search_pending"] = (" > ".join(pt) or friendly_parent_label(level, tuple()))
                    st.experimental_rerun()
    with top_cols[3]:
        compact = st.checkbox("Compact mode", value=True, key="sym_compact_mode")
    with top_cols[4]:
        parent_choices = ["(select parent)"] + [(" > ".join(p) or friendly_parent_label(level, tuple())) for p in sorted(parents_by_level.get(level, set()))]
        pick_parent = st.selectbox("Quick jump", parent_choices, key="sym_quick_jump")
        if pick_parent and pick_parent != "(select parent)":
            st.session_state["sym_search_pending"] = pick_parent.lower()
            st.experimental_rerun()

    sort_mode = st.radio("Sort by", ["Problem severity (issues first)", "Alphabetical (parent path)"],
                         horizontal=True, key="sym_sort_mode")

    # Build entries
    entries = []
    label_childsets: Dict[Tuple[int, str], set] = {}

    for parent_tuple in sorted(parents_by_level.get(level, set())):
        parent_text = " > ".join(parent_tuple) if parent_tuple else friendly_parent_label(level, parent_tuple)
        if search and (search not in parent_text.lower()):
            continue
        key = level_key_tuple(level, parent_tuple)
        children = [x for x in (store.get(key, []) or []) if normalize_text(x) != ""]
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
        label_childsets.setdefault((level, last_label), set()).add(tuple(sorted(children)))

    # Simple inconsistency flag: same last label → different sets
    inconsistent_labels = {k for k, v in label_childsets.items() if len(v) > 1}

    status_rank = {"No group of symptoms": 0, "Symptom left out": 1, "Overspecified": 2, "OK": 3}
    if sort_mode.startswith("Problem"):
        entries.sort(key=lambda e: (status_rank[e[2]], e[0]))
    else:
        entries.sort(key=lambda e: e[0])

    # Vocabulary suggestions from current sheet
    vocab = _build_vocabulary(df)
    vocab_opts = ["(pick suggestion)"] + vocab

    # Render entries
    for parent_tuple, children, status in entries:
        keyname = level_key_tuple(level, parent_tuple)
        subtitle = f"{(' > '.join(parent_tuple)) if parent_tuple else friendly_parent_label(level, parent_tuple)} — {status}"
        if (level, (parent_tuple[-1] if parent_tuple else "<ROOT>")) in inconsistent_labels:
            subtitle += "  ⚠️"

        with st.expander(subtitle, expanded=False):
            selected_vals: List[Tuple[str, str]] = []

            if compact:
                for i in range(5):
                    default_val = children[i] if i < len(children) else ""
                    # text + suggestion for slot i
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

            fill_other = st.checkbox(
                "Fill remaining blanks with 'Other' on save",
                key=f"sym_other_{level}_{'__'.join(parent_tuple)}"
            )
            enforce_unique = st.checkbox(
                "Enforce uniqueness across the 5",
                value=True,
                key=f"sym_unique_{level}_{'__'.join(parent_tuple)}"
            )

            def build_final_values() -> List[str]:
                vals = []
                for (txt, pick) in selected_vals:
                    val = pick if pick != "(pick suggestion)" else txt
                    vals.append(normalize_text(val))
                # ensure 5 slots
                vals = vals[:5] + [""] * max(0, 5 - len(vals))

                # fill others before uniqueness so 'Other' can be deduped if desired
                if fill_other:
                    vals = [v if v else "Other" for v in vals]

                if enforce_unique:
                    seen = set()
                    uniq = []
                    for v in vals:
                        if v and v not in seen:
                            uniq.append(v)
                            seen.add(v)
                    vals = uniq + [""] * max(0, 5 - len(uniq))
                    if fill_other:
                        vals = [v if v else "Other" for v in vals]

                return enforce_k_five(vals)

            # Save button
            if st.button("Save 5 branches for this parent", key=f"sym_save_{level}_{'__'.join(parent_tuple)}"):
                fixed = build_final_values()

                # Snapshot for undo
                stack = _ss_get("undo_stack", [])
                stack.append({
                    "context": "symptoms",
                    "override_root": override_root,
                    "sheet": sheet,
                    "level": level,
                    "parent": parent_tuple,
                    "overrides_sheet_before": overrides_sheet.copy(),
                    "df_before": df.copy(),
                })
                st.session_state["undo_stack"] = stack

                # Update overrides
                overrides_all = _ss_get(override_root, {})
                overrides_sheet = overrides_all.get(sheet, {}).copy()
                overrides_sheet[keyname] = fixed
                overrides_all[sheet] = overrides_sheet
                st.session_state[override_root] = overrides_all
                _mark_session_edit(sheet, keyname)

                # Auto-cascade with enhanced engine
                try:
                    vms = sorted(set(df["Vital Measurement"].map(normalize_text).replace("", np.nan).dropna().unique().tolist()))
                    edited_keys_for_sheet = set(_ss_get("session_edited_keys", {}).get(sheet, []))
                    df_new, tstats = build_raw_plus_v630(df, overrides_sheet, include_scope="session", edited_keys_for_sheet=edited_keys_for_sheet)
                except AssertionError as e:
                    st.error(str(e))
                    continue

                # Persist in-session
                wb[sheet] = df_new
                if source == "Upload workbook":
                    st.session_state["upload_workbook"] = wb
                else:
                    st.session_state["gs_workbook"] = wb

                # Visual confirmation with a small thumb
                st.success(f"Saved and auto-cascaded: added {tstats['new_rows']} rows, filled {tstats['inplace_filled']} anchors.  👍")

    st.caption("Tip: Use **Workspace Selection** to preview grouped views or push to Google Sheets.")
