# ui_conflicts.py

from typing import Dict, List, Tuple, Optional, Set
import pandas as pd
import numpy as np
import streamlit as st

from utils import (
    CANON_HEADERS, LEVEL_COLS, MAX_LEVELS,
    normalize_text, validate_headers, enforce_k_five, level_key_tuple, friendly_parent_label,
)
from logic_conflicts import (
    build_store, compute_conflicts, conflict_summary,
    resolve_keep_set_for_all, resolve_custom_set_for_all,
    parents_for_label_at_level,
)
from logic_cascade import build_raw_plus_v630


@st.cache_data(show_spinner=False, ttl=600)
def _cached_compute_conflicts(store: Dict, friendly_labels: bool = True) -> Dict:
    """Cached version of compute_conflicts to prevent recomputation."""
    return compute_conflicts(store, friendly_labels=friendly_labels)


@st.cache_data(show_spinner=False, ttl=600)
def _cached_conflict_summary(conf_map: Dict, level: Optional[int] = None) -> List[Dict]:
    """Cached version of conflict_summary to prevent recomputation."""
    return conflict_summary(conf_map, level=level)


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


# ----------------- UI: Conflicts Inspector -----------------

def render():
    st.header("⚖️ Conflicts Inspector")

    # Source + sheet selection
    sources = []
    if _ss_get("upload_workbook", {}):
        sources.append("Upload workbook")
    if _ss_get("gs_workbook", {}):
        sources.append("Google Sheets workbook")
    if not sources:
        st.info("ℹ️ Load a workbook first in the **Source** tab.")
        return

    source = st.radio("Choose data source", sources, horizontal=True, key="conf_source_sel")
    if source == "Upload workbook":
        wb = _ss_get("upload_workbook", {})
        override_root = "branch_overrides_upload"
    else:
        wb = _ss_get("gs_workbook", {})
        override_root = "branch_overrides_gs"

    if not wb:
        st.warning("⚠️ No sheets found in the selected source.")
        return

    sheet = st.selectbox("Sheet", list(wb.keys()), key="conf_sheet_sel")
    df = wb.get(sheet, pd.DataFrame())
    if df.empty or not validate_headers(df):
        st.info("ℹ️ Selected sheet is empty or headers mismatch.")
        return

    # Build store & conflicts
    overrides_all = _ss_get(override_root, {})
    overrides_sheet = overrides_all.get(sheet, {})
    store = build_store(df, overrides_sheet)

    # Filters
    colf1, colf2, colf3, colf4 = st.columns([1, 1, 2, 1])
    with colf1:
        level_filter = st.selectbox("Node", ["All", 1, 2, 3, 4, 5], key="conf_level_filter")
    with colf2:
        only_conflicts = st.checkbox("Only items with conflicts", value=True, key="conf_only_conf")
    with colf3:
        search = st.text_input("Search parent label", key="conf_search").strip().lower()
    with colf4:
        exp_all_toggle = st.checkbox("Expand all", value=False, key="conf_expand_all_toggle")

    conf_map = _cached_compute_conflicts(store, friendly_labels=True)
    # Build summary
    summary_rows = _cached_conflict_summary(conf_map, level=None if level_filter == "All" else level_filter)
    # Dataframe for export
    if summary_rows:
        df_summary = pd.DataFrame(summary_rows)
        st.dataframe(df_summary.head(100), use_container_width=True, height=220)
        st.download_button("Download conflict summary (CSV)",
                           data=df_summary.to_csv(index=False).encode("utf-8"),
                           file_name=f"{sheet}_conflicts_summary.csv",
                           mime="text/csv")
    else:
        st.success("✅ No conflicts found at the selected scope. 🎉")

    # Persisted expand state & focus handling to prevent UI jump
    open_keys: List[Tuple[int, str]] = _ss_get("conflicts_open_keys", [])
    focus_key = st.session_state.get("conflicts_focus_key")  # (level, parent_label) after save

    # Iterate conflict groups
    groups = []
    for (lvl, label), variants in conf_map.items():
        if level_filter != "All" and lvl != level_filter:
            continue
        # decide if it's a "conflict"
        if only_conflicts and len(variants) <= 1:
            continue
        if search and (search not in str(label).lower()):
            continue
        groups.append((lvl, label, variants))

    # Sort groups by severity (more variants first), then label
    groups.sort(key=lambda g: (-len(g[2]), g[0], str(g[1])))

    # Presets store
    presets_root = _ss_get("conflict_presets", {})  # {sheet: {(lvl,label): [list_of_sets]}}
    presets_sheet = presets_root.get(sheet, {})

    # Navigation helpers
    st.markdown("---")
    st.caption("Click a group to expand. You can apply an existing variant, build a custom 5-child set from the union, or use presets you saved.")
    st.caption("When you save, the app keeps this group expanded to avoid scroll jumps. 👍")

    for (lvl, label, variants) in groups:
        key_group = (lvl, label)
        # compute metrics
        num_variants = len(variants)
        affected_parents = sum(len(v) for v in variants.values())
        union_children = sorted({c for childset in variants.keys() for c in childset})

        # expansion state
        expanded_default = exp_all_toggle or (focus_key == key_group) or (key_group in open_keys)
        exp_label = f"Node {lvl} — {label}  •  {num_variants} variant(s)  •  {affected_parents} affected parent(s)"
        with st.expander(exp_label, expanded=expanded_default):
            # remember open state
            if expanded_default and key_group not in open_keys:
                open_keys.append(key_group)

            # Show affected parents (toggle)
            with st.container():
                st.markdown("**Show affected parents (paths)**")
                for childset, parents in variants.items():
                    st.markdown(f"- **Variant ({len(childset)}):** {', '.join(childset) if childset else '(empty)'}")
                    for p in parents:
                        pretty = " > ".join(p) if p else friendly_parent_label(lvl, p)
                        st.code(pretty)

            # --- Existing variants (apply buttons)
            st.markdown("**Variants**")
            vcols = st.columns(min(3, max(1, num_variants)))
            v_list = list(variants.items())
            for i, (childset, parents) in enumerate(v_list):
                with vcols[i % len(vcols)]:
                    st.write(f"Variant {i+1}:")
                    if childset:
                        for idx, ch in enumerate(childset, 1):
                            st.write(f"{idx}. {ch}")
                    else:
                        st.write("(no children)")
                    if st.button(f"Apply Variant {i+1} to all", key=f"conf_apply_variant_{lvl}_{label}_{i}"):
                        # Resolve: keep this set for ALL parents of this last-label at this level
                        new_overrides, n_aff = resolve_keep_set_for_all(
                            overrides_sheet, store, level=lvl, parent_label=label,
                            childset_to_keep=list(childset), friendly_labels=True
                        )
                        # Mark each affected parent as edited for cascade scope
                        parents_affected = parents_for_label_at_level(store, lvl, label, friendly_labels=True)
                        for p in parents_affected:
                            _mark_session_edit(sheet, level_key_tuple(lvl, p))

                        # Persist overrides
                        overrides_all = _ss_get(override_root, {})
                        overrides_all[sheet] = new_overrides
                        st.session_state[override_root] = overrides_all

                        # Auto-cascade with enhanced engine
                        edited_keys_for_sheet = set(_ss_get("session_edited_keys", {}).get(sheet, []))
                        df_new, tstats = build_raw_plus_v630(df, new_overrides, include_scope="session", edited_keys_for_sheet=edited_keys_for_sheet)

                        # Persist workbook
                        wb[sheet] = df_new
                        if source == "Upload workbook":
                            st.session_state["upload_workbook"] = wb
                        else:
                            st.session_state["gs_workbook"] = wb

                        # Keep this group open after re-run
                        st.session_state["conflicts_focus_key"] = key_group
                        st.session_state["conflicts_open_keys"] = open_keys
                        st.success(f"✅ Applied Variant {i+1} to {n_aff} parent(s). Added {tstats['new_rows']} row(s), filled {tstats['inplace_filled']} anchors. 👍")
                        st.rerun()  # updated from st.experimental_rerun()

                    # Save this variant as a preset for later reuse
                    if st.button(f"Save Variant {i+1} as preset", key=f"conf_save_preset_{lvl}_{label}_{i}"):
                        presets_sheet.setdefault(key_group, [])
                        preset_set = [c for c in childset]
                        if preset_set not in presets_sheet[key_group]:
                            presets_sheet[key_group].append(preset_set)
                            presets_root[sheet] = presets_sheet
                            st.session_state["conflict_presets"] = presets_root
                            st.success("✅ Preset saved. 👍")
                        else:
                            st.info("ℹ️ Preset already saved for this group.")

            st.markdown("---")

            # --- Presets section
            preset_sets = presets_sheet.get(key_group, [])
            with st.container():
                st.markdown("**Presets for this group**")
                if not preset_sets:
                    st.caption("No presets saved yet. Use “Save Variant as preset” or save a custom set below.")
                else:
                    for j, pset in enumerate(preset_sets):
                        cols_p = st.columns([2, 1, 1])
                        with cols_p[0]:
                            st.write(f"Preset {j+1}: {', '.join(pset) if pset else '(empty)'}")
                        with cols_p[1]:
                            if st.button(f"Apply preset {j+1}", key=f"conf_apply_preset_{lvl}_{label}_{j}"):
                                new_overrides, n_aff = resolve_keep_set_for_all(
                                    overrides_sheet, store, level=lvl, parent_label=label,
                                    childset_to_keep=pset, friendly_labels=True
                                )
                                parents_affected = parents_for_label_at_level(store, lvl, label, friendly_labels=True)
                                for p in parents_affected:
                                    _mark_session_edit(sheet, level_key_tuple(lvl, p))
                                overrides_all = _ss_get(override_root, {})
                                overrides_all[sheet] = new_overrides
                                st.session_state[override_root] = overrides_all
                                edited_keys_for_sheet = set(_ss_get("session_edited_keys", {}).get(sheet, []))
                                df_new, tstats = build_raw_plus_v630(df, new_overrides, include_scope="session", edited_keys_for_sheet=edited_keys_for_sheet)
                                wb[sheet] = df_new
                                if source == "Upload workbook":
                                    st.session_state["upload_workbook"] = wb
                                else:
                                    st.session_state["gs_workbook"] = wb
                                st.session_state["conflicts_focus_key"] = key_group
                                st.session_state["conflicts_open_keys"] = open_keys
                                st.success(f"✅ Applied preset {j+1} to {n_aff} parent(s). Added {tstats['new_rows']} row(s), filled {tstats['inplace_filled']} anchors. 👍")
                                st.rerun()  # updated from st.experimental_rerun()
                        with cols_p[2]:
                            if st.button(f"Delete preset {j+1}", key=f"conf_delete_preset_{lvl}_{label}_{j}"):
                                presets_sheet[key_group].pop(j)
                                presets_root[sheet] = presets_sheet
                                st.session_state["conflict_presets"] = presets_root
                                st.success("✅ Preset deleted.")

            st.markdown("---")

            # --- Custom set builder (union across variants + extra)
            st.markdown("**Build a custom child set (pick up to 5)**")

            # State keys for this group
            sel_key = f"conf_custom_sel_{lvl}_{label}"
            add_key = f"conf_custom_add_{lvl}_{label}"

            options = list(union_children)
            selected = _ss_get(sel_key, [])
            selected = [x for x in selected if x in options]  # prune any stale values

            # Allow adding a new child label not present in union
            add_extra = st.text_input("Add another option (optional)", key=add_key).strip()
            if add_extra and add_extra not in options:
                options = options + [add_extra]

            # Multi-select with cap (enforced after selection)
            selected = st.multiselect("Select up to 5 children", options=options, default=selected, key=sel_key)
            if len(selected) > 5:
                st.warning("⚠️ Please select at most 5 items. Extra selections will be ignored on save.")

            # Quick buttons to fill common sets
            cquick1, cquick2 = st.columns([1, 1])
            with cquick1:
                if st.button("Select first 5 of union", key=f"conf_quick_union5_{lvl}_{label}"):
                    st.session_state[sel_key] = options[:5]
                    st.rerun()  # updated from st.experimental_rerun()
            with cquick2:
                if st.button("Clear selection", key=f"conf_clear_sel_{lvl}_{label}"):
                    st.session_state[sel_key] = []
                    st.rerun()  # updated from st.experimental_rerun()

            # Apply custom set to all
            if st.button("Apply custom set to all parents with this label", key=f"conf_apply_custom_{lvl}_{label}"):
                chosen = enforce_k_five(selected)
                new_overrides, n_aff = resolve_custom_set_for_all(
                    overrides_sheet, store, level=lvl, parent_label=label,
                    custom_children=chosen, friendly_labels=True
                )

                # Mark edits for each affected parent
                parents_affected = parents_for_label_at_level(store, lvl, label, friendly_labels=True)
                for p in parents_affected:
                    _mark_session_edit(sheet, level_key_tuple(lvl, p))

                # Persist overrides
                overrides_all = _ss_get(override_root, {})
                overrides_all[sheet] = new_overrides
                st.session_state[override_root] = overrides_all

                # Auto-cascade
                edited_keys_for_sheet = set(_ss_get("session_edited_keys", {}).get(sheet, []))
                df_new, tstats = build_raw_plus_v630(df, new_overrides, include_scope="session", edited_keys_for_sheet=edited_keys_for_sheet)

                # Persist workbook
                wb[sheet] = df_new
                if source == "Upload workbook":
                    st.session_state["upload_workbook"] = wb
                else:
                    st.session_state["gs_workbook"] = wb

                # Keep this group open after re-run
                st.session_state["conflicts_focus_key"] = key_group
                st.session_state["conflicts_open_keys"] = open_keys
                st.success(f"✅ Applied custom set to {n_aff} parent(s). Added {tstats['new_rows']} row(s), filled {tstats['inplace_filled']} anchors. 👍")
                st.rerun()  # updated from st.experimental_rerun()

            # Save selection as a preset
            if st.button("Save current selection as preset", key=f"conf_save_custom_preset_{lvl}_{label}"):
                chosen = [c for c in selected][:5]
                presets_sheet.setdefault(key_group, [])
                if chosen and chosen not in presets_sheet[key_group]:
                    presets_sheet[key_group].append(chosen)
                    presets_root[sheet] = presets_sheet
                    st.session_state["conflict_presets"] = presets_root
                    st.success("✅ Preset saved. 👍")
                else:
                    st.info("ℹ️ Nothing new to save or already saved.")

    # Persist expand state list
    st.session_state["conflicts_open_keys"] = open_keys
