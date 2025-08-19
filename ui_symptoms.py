# ui_symptoms.py

from typing import Dict, List, Tuple, Optional, Set, Any
import json
import numpy as np
import pandas as pd
import streamlit as st

from utils import (
    CANON_HEADERS, LEVEL_COLS, MAX_LEVELS,
    normalize_text, validate_headers, friendly_parent_label, level_key_tuple,
)

# -----------------------------
# Robust imports from logic_cascade with graceful fallbacks
# -----------------------------

_infer_with_overrides = None
_cascade_fn = None

try:
    # Preferred names (v6.3.x)
    from logic_cascade import infer_branch_options_with_overrides as _infer_with_overrides
except Exception:
    _infer_with_overrides = None

try:
    # Newer engine name first
    from logic_cascade import cascade_deep_known_parent_attach_anchor_reuse as _cascade_fn
except Exception:
    # Older name fallback
    try:
        from logic_cascade import cascade_anchor_reuse_full as _cascade_fn
    except Exception:
        _cascade_fn = None


def _infer_local(df: pd.DataFrame, overrides: Dict[str, List[str]]) -> Dict[str, List[str]]:
    """Local fallback: build store + merge overrides."""
    store: Dict[str, List[str]] = {}
    if df is None or df.empty:
        return overrides or {}

    def parent_key_from_row_strict(row: pd.Series, upto_level: int) -> Optional[Tuple[str, ...]]:
        if upto_level <= 1:
            return tuple()
        parent = []
        for c in LEVEL_COLS[:upto_level-1]:
            v = normalize_text(row.get(c, ""))
            if v == "":
                return None
            parent.append(v)
        return tuple(parent)

    def level_key_tuple_local(level: int, parent: Tuple[str, ...]) -> str:
        return f"L{level}|" + (">".join(parent) if parent else "<ROOT>")

    # Infer from data
    for level in range(1, MAX_LEVELS+1):
        parent_to_children: Dict[Tuple[str, ...], List[str]] = {}
        child_col = LEVEL_COLS[level-1]
        for _, row in df.iterrows():
            if child_col not in df.columns:
                continue
            child = normalize_text(row.get(child_col, ""))
            if child == "":
                continue
            parent = parent_key_from_row_strict(row, level)
            if parent is None:
                continue
            parent_to_children.setdefault(parent, []).append(child)

        for parent, children in parent_to_children.items():
            uniq, seen = [], set()
            for c in children:
                if c not in seen:
                    seen.add(c)
                    uniq.append(c)
            store[level_key_tuple_local(level, parent)] = uniq

    # Merge overrides (overrides win)
    merged = dict(store)
    for k, v in (overrides or {}).items():
        vals = v if isinstance(v, list) else [v]
        merged[k] = [normalize_text(x) for x in vals]
    return merged


def _cascade_local(
    df: pd.DataFrame,
    store: Dict[str, List[str]],
    vm_scope: List[str],
    start_parents: List[Tuple[str, ...]],
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Very simple fallback cascade:
      - ensures each specified parent has 5 children rows for each VM in-scope
      - no deep known-parent attach; only anchor reuse + row append for missing children
    Stats: {"new_rows": int, "inplace_filled": int}
    """
    if df is None or df.empty:
        return df, {"new_rows": 0, "inplace_filled": 0}

    def _rows_match_parent(df0: pd.DataFrame, vm: str, parent: Tuple[str, ...], level: int) -> pd.DataFrame:
        mask = (df0["Vital Measurement"].astype(str).map(normalize_text) == normalize_text(vm))
        for i, val in enumerate(parent, 1):
            col = f"Node {i}"
            mask = mask & (df0[col].astype(str).map(normalize_text) == normalize_text(val))
        return df0[mask].copy()

    def _present_children(df0: pd.DataFrame, vm: str, parent: Tuple[str, ...], level: int) -> Set[str]:
        sub = _rows_match_parent(df0, vm, parent, level)
        col = f"Node {level}"
        return set(
            sub[col].astype(str).map(normalize_text).replace("", np.nan).dropna().unique().tolist()
        )

    def _find_anchor_index(df0: pd.DataFrame, vm: str, parent: Tuple[str, ...], level: int) -> Optional[int]:
        target_col = f"Node {level}"
        sub_idx = _rows_match_parent(df0, vm, parent, level).index.tolist()
        for ix in sub_idx:
            if normalize_text(df0.at[ix, target_col]) == "":
                return ix
        return None

    def _emit_row_from_prefix(vm_val: str, pref: Tuple[str, ...]) -> Dict[str, str]:
        row = {"Vital Measurement": vm_val}
        for i, val in enumerate(pref, 1):
            row[f"Node {i}"] = val
        for i in range(1, MAX_LEVELS+1):
            row.setdefault(f"Node {i}", "")
        row["Diagnostic Triage"] = ""
        row["Actions"] = ""
        return row

    def _children_from_store(store0: Dict[str, List[str]], level: int, parent: Tuple[str, ...]) -> List[str]:
        key = level_key_tuple(level, parent)
        return [normalize_text(o) for o in store0.get(key, []) if normalize_text(o) != ""]

    total = {"new_rows": 0, "inplace_filled": 0}
    stack: List[Tuple[str, ...]] = list(start_parents)

    while stack:
        parent = stack.pop(0)
        L = len(parent) + 1
        if L > MAX_LEVELS:
            continue
        children_defined = _children_from_store(store, L, parent)
        if not children_defined:
            continue

        next_child_parents: Set[Tuple[str, ...]] = set()

        for vm in vm_scope:
            present = _present_children(df, vm, parent, L)
            missing = [c for c in children_defined if c not in present]

            # anchor reuse: fill at most 1
            if missing:
                anchor_ix = _find_anchor_index(df, vm, parent, L)
                if anchor_ix is not None:
                    df.at[anchor_ix, f"Node {L}"] = missing[0]
                    total["inplace_filled"] += 1
                    next_child_parents.add(parent + (missing[0],))
                    missing = missing[1:]

            new_rows = []
            for m in missing:
                row = _emit_row_from_prefix(vm, parent + (m,))
                new_rows.append(row)
                next_child_parents.add(parent + (m,))

            if new_rows:
                df = pd.concat([df, pd.DataFrame(new_rows, columns=CANON_HEADERS)], ignore_index=True)
                total["new_rows"] += len(new_rows)

            # also propagate for already-present
            for c in children_defined:
                if c in present:
                    next_child_parents.add(parent + (c,))

        # Only go deeper if the next level has defined children in the store
        for cp in sorted(next_child_parents):
            next_level = len(cp) + 1
            if next_level <= MAX_LEVELS:
                if _children_from_store(store, next_level, cp):
                    stack.append(cp)

    return df, total


# -----------------------------
# Small session helpers
# -----------------------------




def _mark_session_edit(sheet: str, keyname: str):
    ek = st.session_state.get("session_edited_keys", {})
    cur = set(ek.get(sheet, []))
    cur.add(keyname)
    ek[sheet] = list(cur)
    st.session_state["session_edited_keys"] = ek


# -----------------------------
# UI helpers
# -----------------------------

def enforce_k_five(opts: List[str]) -> List[str]:
    clean = [normalize_text(o) for o in opts if normalize_text(o) != ""]
    if len(clean) > 5:
        clean = clean[:5]
    elif len(clean) < 5:
        clean = clean + [""] * (5 - len(clean))
    return clean


def _compute_virtual_parents(store: Dict[str, List[str]]) -> Dict[int, Set[Tuple[str, ...]]]:
    """Collect all parent tuples reachable via store keys (virtual graph expansion)."""
    parents_by_level: Dict[int, Set[Tuple[str, ...]]] = {i: set() for i in range(1, MAX_LEVELS+1)}
    parents_by_level[1].add(tuple())  # root

    # forward expansion using children lists
    for L in range(1, MAX_LEVELS):
        for p in list(parents_by_level[L]):
            key = level_key_tuple(L, p)
            children = [x for x in store.get(key, []) if normalize_text(x) != ""]
            for c in children:
                parents_by_level[L+1].add(p + (c,))

    # also include explicit parents present in store keys
    for key in store.keys():
        if "|" not in key:
            continue
        lvl_s, path = key.split("|", 1)
        try:
            L = int(lvl_s[1:])
        except Exception:
            continue
        parent_tuple = tuple([] if path == "<ROOT>" else path.split(">"))
        if 1 <= L <= MAX_LEVELS:
            parents_by_level[L].add(parent_tuple)
            for k in range(1, min(L, MAX_LEVELS)+1):
                parents_by_level.setdefault(k, set())
                parents_by_level[k].add(tuple(parent_tuple[:k-1]))
    return parents_by_level


def _build_vocabulary(df0: pd.DataFrame) -> List[str]:
    vocab = set()
    for col in LEVEL_COLS:
        if col in df0.columns:
            for x in df0[col].dropna().astype(str):
                xv = normalize_text(x)
                if xv:
                    vocab.add(xv)
    return sorted(vocab)


# -----------------------------
# Main UI
# -----------------------------

def render():
    st.header("🧬 Symptoms — browse & edit child sets (auto-cascade)")

    # Determine source & sheet (prefer current work_context)
    ctx = st.session_state.get("work_context", {})
    default_src = {"upload": "Upload workbook", "gs": "Google Sheets workbook"}.get(ctx.get("source"))
    sources_avail = []
    if st.session_state.get("upload_workbook", {}):
        sources_avail.append("Upload workbook")
    if st.session_state.get("gs_workbook", {}):
        sources_avail.append("Google Sheets workbook")

    if not sources_avail:
        st.info("Load data first in **Source** and pick your sheet in **Workspace**.")
        return

    source = st.radio("Choose data source", sources_avail, horizontal=True, index=(sources_avail.index(default_src) if default_src in sources_avail else 0), key="sym_source_sel")

    if source == "Upload workbook":
        wb = st.session_state.get("upload_workbook", {})
        override_root = "branch_overrides_upload"
        current_source_code = "upload"
    else:
        wb = st.session_state.get("gs_workbook", {})
        override_root = "branch_overrides_gs"
        current_source_code = "gs"

    if not wb:
        st.warning("No sheets found in the selected source.")
        return

    # Sheet selection (default from work_context)
    default_sheet = ctx.get("sheet")
    sheet_names = list(wb.keys())
    sheet_idx = sheet_names.index(default_sheet) if default_sheet in sheet_names else 0
    sheet = st.selectbox("Sheet", sheet_names, index=sheet_idx, key="sym_sheet_sel")

    df = wb.get(sheet, pd.DataFrame())
    if df is None or df.empty or not validate_headers(df):
        st.info("Selected sheet is empty or headers mismatch.")
        return

    # Build store (inferred + overrides)
    overrides_all = st.session_state.get(override_root, {})
    overrides_sheet = overrides_all.get(sheet, {})
    if _infer_with_overrides:
        store = _infer_with_overrides(df, overrides_sheet)
    else:
        store = _infer_local(df, overrides_sheet)

    parents_by_level = _compute_virtual_parents(store)

    # Persistent search helpers
    _pending = st.session_state.pop("sym_search_pending", None)
    if _pending is not None:
        st.session_state["sym_search"] = _pending

    # Top controls
    top_cols = st.columns([2, 1, 1, 1, 2])
    with top_cols[0]:
        search = st.text_input("Search parent symptom/path", key="sym_search").strip().lower()
    with top_cols[1]:
        if st.button("Next Missing"):
            for pt in sorted(parents_by_level.get(1, set())) + sorted(parents_by_level.get(2, set())) + \
                       sorted(parents_by_level.get(3, set())) + sorted(parents_by_level.get(4, set())) + \
                       sorted(parents_by_level.get(5, set())):
                # find first parent with 0 children
                L = len(pt) + 1
                key = level_key_tuple(L, pt)
                non_empty = [x for x in store.get(key, []) if normalize_text(x) != ""]
                if len(non_empty) == 0:
                    st.session_state["sym_search"] = (" > ".join(pt) or friendly_parent_label(L, tuple())).lower()
                    st.rerun()
    with top_cols[2]:
        if st.button("Next Incomplete"):
            for L in range(1, MAX_LEVELS+1):
                for pt in sorted(parents_by_level.get(L, set())):
                    key = level_key_tuple(L, pt)
                    n = len([x for x in store.get(key, []) if normalize_text(x) != ""])
                    if 1 <= n < 5:
                        st.session_state["sym_search"] = (" > ".join(pt) or friendly_parent_label(L, tuple())).lower()
                        st.rerun()
    with top_cols[3]:
        compact = st.checkbox("Compact mode", value=True)
    with top_cols[4]:
        level = st.selectbox("Level to edit children for", [1, 2, 3, 4, 5], index=0, format_func=lambda x: f"Node {x}", key="sym_level_sel")

    # Sort & filter entries
    entries = []
    for parent_tuple in sorted(parents_by_level.get(level, set())):
        parent_text = " > ".join(parent_tuple)
        # filter by search (case-insensitive)
        if search and (search not in parent_text.lower()) and not (not parent_tuple and search in friendly_parent_label(level, tuple()).lower()):
            continue
        key = level_key_tuple(level, parent_tuple)
        children = [x for x in store.get(key, []) if normalize_text(x) != ""]
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

    # Vocab suggestions (built once)
    vocab = _build_vocabulary(df)
    vocab_opts = ["(pick suggestion)"] + vocab

    # Undo control (session-wide)
    if st.button("↩️ Undo last edit in this tab"):
        stack = st.session_state.get("undo_stack", [])
        if not stack:
            st.info("Nothing to undo.")
        else:
            last = stack.pop()
            st.session_state["undo_stack"] = stack
            if last.get("context") == "symptoms" and last.get("sheet") == sheet and last.get("override_root") == override_root:
                # restore overrides & df snapshot
                overrides_all = st.session_state.get(override_root, {})
                overrides_all[sheet] = last.get("overrides_sheet_before", {})
                st.session_state[override_root] = overrides_all
                if last.get("df_before") is not None:
                    wb[sheet] = last["df_before"]
                    if current_source_code == "upload":
                        st.session_state["upload_workbook"] = wb
                    else:
                        st.session_state["gs_workbook"] = wb
                st.success("Reverted last Symptoms edit for this sheet.")
                st.rerun()
            else:
                st.info("Last undo snapshot does not belong to this sheet/tab.")

    st.markdown("---")

    # Render entries
    if not entries:
        st.info("No matching parents at this level with current filters.")
        return

    # A place to remember which parent we just saved (to show a 👍)
    saved_parents_marks: Set[str] = set(st.session_state.get("sym_saved_marks", set()))

    for parent_tuple, children, status in entries:
        keyname = level_key_tuple(level, parent_tuple)
        subtitle = f"{' > '.join(parent_tuple) or friendly_parent_label(level, tuple())} — {status}"
        if keyname in saved_parents_marks:
            subtitle += "  ✅"

        with st.expander(subtitle):
            selected_vals: List[Tuple[str, str]] = []

            if compact:
                # Five rows of (text input + suggestion select)
                for i in range(5):
                    default_val = children[i] if i < len(children) else ""
                    cols = st.columns([2, 1])
                    txt_key = f"sym_txt_{level}_{'__'.join(parent_tuple)}_{i}"
                    sel_key = f"sym_sel_{level}_{'__'.join(parent_tuple)}_{i}"
                    with cols[0]:
                        txt = st.text_input(f"Child {i+1}", value=default_val, key=txt_key)
                    with cols[1]:
                        # Non-empty label to silence Streamlit warning
                        pick = st.selectbox(
                            "Suggestion",
                            options=vocab_opts,
                            index=0,
                            key=sel_key,
                            label_visibility="collapsed",
                        )
                    selected_vals.append((txt, pick))
            else:
                cols = st.columns(5)
                for i in range(5):
                    default_val = children[i] if i < len(children) else ""
                    with cols[i]:
                        txt_key = f"sym_txt_{level}_{'__'.join(parent_tuple)}_{i}"
                        sel_key = f"sym_sel_{level}_{'__'.join(parent_tuple)}_{i}"
                        txt = st.text_input(f"Child {i+1}", value=default_val, key=txt_key)
                        pick = st.selectbox(
                            "Suggestion",
                            options=vocab_opts,
                            index=0,
                            key=sel_key,
                            label_visibility="collapsed",
                        )
                    selected_vals.append((txt, pick))

            fill_other = st.checkbox(
                "Fill remaining blanks with ‘Other’ on save",
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
                # pad to 5
                vals = vals[:5] + [""] * max(0, 5 - len(vals))
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

            colA, colB = st.columns([1, 3])
            with colA:
                if st.button("💾 Save 5 branches", key=f"sym_save_{level}_{'__'.join(parent_tuple)}"):
                    fixed = build_final_values()

                    # Snapshot for undo
                    stack = st.session_state.get("undo_stack", [])
                    stack.append({
                        "context": "symptoms",
                        "override_root": override_root,
                        "sheet": sheet,
                        "level": level,
                        "parent": parent_tuple,
                        "overrides_sheet_before": overrides_all.get(sheet, {}).copy(),
                        "df_before": df.copy(),
                    })
                    st.session_state["undo_stack"] = stack

                    # Update overrides
                    overrides_all = st.session_state.get(override_root, {})
                    overrides_sheet = overrides_all.get(sheet, {}).copy()
                    overrides_sheet[keyname] = fixed
                    overrides_all[sheet] = overrides_sheet
                    st.session_state[override_root] = overrides_all
                    _mark_session_edit(sheet, keyname)

                    # Auto-cascade (preferred engine if available)
                    store2 = (_infer_with_overrides(df, overrides_sheet) if _infer_with_overrides
                              else _infer_local(df, overrides_sheet))
                    vms = sorted(
                        set(df["Vital Measurement"].astype(str).map(normalize_text).replace("", np.nan).dropna().unique().tolist())
                    )
                    start_parents = [parent_tuple]
                    if _cascade_fn:
                        df_new, tstats = _cascade_fn(df, store2, vms, start_parents)
                    else:
                        df_new, tstats = _cascade_local(df, store2, vms, start_parents)

                    wb[sheet] = df_new
                    if current_source_code == "upload":
                        st.session_state["upload_workbook"] = wb
                    else:
                        st.session_state["gs_workbook"] = wb

                    # Remember saved mark
                    saved_parents_marks.add(keyname)
                    st.session_state["sym_saved_marks"] = saved_parents_marks

                    st.success(f"Saved and auto-cascaded: added {tstats['new_rows']} rows, filled {tstats['inplace_filled']} anchors.")
                    st.toast("Child set saved ✅", icon="✅")

            with colB:
                # Show a read-only preview of final saved values
                st.caption("Preview to be saved")
                st.write(", ".join(build_final_values()))

    st.markdown("---")
    st.caption("Tip: Use search and ‘Next Missing / Next Incomplete’ to move through parents efficiently.")
