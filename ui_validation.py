# ui_validation.py

from typing import Dict, List, Tuple, Any
import pandas as pd
import numpy as np
import streamlit as st

from utils import (
    CANON_HEADERS, LEVEL_COLS, MAX_LEVELS,
    normalize_text, validate_headers, friendly_parent_label,
    level_key_tuple, order_decision_tree,
)

# Import validation logic functions
try:
    from logic_validation_functions import (
        compute_validation_report,
        detect_orphan_nodes,
        detect_loops,
        detect_missing_red_flags,
        detect_empty_branches,
    )
    HAVE_COMBINED = True
except Exception:
    # Fallback: try individual functions
    try:
        from logic_validation_functions import (
            detect_orphan_nodes,
            detect_loops,
            detect_missing_red_flags,
            detect_empty_branches,
        )
        HAVE_COMBINED = False
    except Exception:
        # If all else fails, set to None
        compute_validation_report = None
        detect_orphan_nodes = None
        detect_loops = None
        detect_missing_red_flags = None
        detect_empty_branches = None
        HAVE_COMBINED = False


@st.cache_data(show_spinner=False, ttl=600)
def _cached_compute_validation_report(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Cached version of compute_validation_report to prevent endless reruns.
    """
    if df is None or df.empty:
        return {
            "orphan_nodes": [],
            "loops": [],
            "missing_red_flags": [],
            "empty_branches": [],
            "summary": {
                "total_orphans": 0,
                "total_loops": 0,
                "total_missing_red_flags": 0,
                "total_empty_branches": 0,
                "total_issues": 0
            }
        }
    
    if HAVE_COMBINED and compute_validation_report:
        return compute_validation_report(df)
    else:
        # Fallback to individual functions
        orphans = detect_orphan_nodes(df) if detect_orphan_nodes else []
        loops = detect_loops(df) if detect_loops else []
        missing_rf = detect_missing_red_flags(df) if detect_missing_red_flags else []
        empty_branches = detect_empty_branches(df) if detect_empty_branches else []
        
        return {
            "orphan_nodes": orphans,
            "loops": loops,
            "missing_red_flags": missing_rf,
            "empty_branches": empty_branches,
            "summary": {
                "total_orphans": len(orphans),
                "total_loops": len(loops),
                "total_missing_red_flags": len(missing_rf),
                "total_empty_branches": len(empty_branches),
                "total_issues": len(orphans) + len(loops) + len(missing_rf) + len(empty_branches)
            }
        }





def _jump_to_symptoms(level: int, parent_tuple: Tuple[str, ...]):
    """
    Prime Symptoms tab filters so the user can quickly land on the parent they need.
    """
    # Set which level is inspected and the quick jump text (search)
    st.session_state["sym_level_sel"] = level
    pretty = " > ".join(parent_tuple) if parent_tuple else friendly_parent_label(level, tuple())
    st.session_state["sym_search_pending"] = pretty.lower()
    st.success("Primed the Symptoms tab search for this parent. Switch to the 🧬 Symptoms tab.")


def _render_orphans(orphans: List[Dict[str, Any]]):
    st.subheader("🧩 Orphan nodes")
    st.caption("A node label appears as a child but never as a parent for its next level, where a branch might reasonably continue.")

    if not orphans:
        st.success("✅ No orphans detected. 🎉")
        return

    # Build table
    rows = []
    for item in orphans:
        label = item.get("label", "")
        level = int(item.get("level", 1))
        node_id = item.get("node_id", f"Node {level}")
        appears_as_child_in = item.get("appears_as_child_in", [])
        example_txt = "; ".join(appears_as_child_in[:5]) if appears_as_child_in else "—"
        rows.append({
            "Node Label": label,
            "At Level": node_id,
            "Appears as Child In": example_txt,
            "Row Index": item.get("row_index", -1),
        })

    df_orph = pd.DataFrame(rows).sort_values(["At Level", "Node Label"])
    # Limit preview to first 100 rows for speed
    st.dataframe(df_orph.head(100), use_container_width=True, height=260)

    st.download_button(
        "Download orphans (CSV)",
        data=df_orph.to_csv(index=False).encode("utf-8"),
        file_name="validation_orphans.csv",
        mime="text/csv",
    )


def _render_loops(loops: List[Dict[str, Any]]):
    st.subheader("🔁 Loops (cycles)")

    if not loops:
        st.success("✅ No cycles detected. 🎉")
        return

    rows = []
    for item in loops:
        path = item.get("path", [])
        length = item.get("length", 0)
        start_node = item.get("start_node", "")
        cycle_type = item.get("cycle_type", "unknown")
        
        # Display as A → B → C → A
        if path and len(path) > 1:
            display = " → ".join(path)
            if path[0] != path[-1]:
                display += f" → {path[0]}"
        else:
            display = " → ".join(path) if path else "(empty)"
        
        rows.append({
            "Cycle": display,
            "Length": length,
            "Start Node": start_node,
            "Type": cycle_type
        })

    df_loops = pd.DataFrame(rows)
    # Limit preview to first 100 rows for speed
    st.dataframe(df_loops.head(100), use_container_width=True, height=220)

    st.download_button(
        "Download loops (CSV)",
        data=df_loops.to_csv(index=False).encode("utf-8"),
        file_name="validation_loops.csv",
        mime="text/csv",
    )


def _render_missing_redflag(miss_rf: List[Dict[str, Any]]):
    st.subheader("🚩 Missing Red Flag coverage")

    st.caption("Nodes that appear frequently but don't have explicit red flag indicators. Consider adding red flag coverage for these nodes.")

    if not miss_rf:
        st.success("All nodes have appropriate red flag coverage. 🎉")
        return

    rows = []
    for item in miss_rf:
        label = item.get("label", "")
        level = int(item.get("level", 1))
        node_id = item.get("node_id", f"Node {level}")
        issue_type = item.get("issue_type", "unknown")
        suggested_action = item.get("suggested_action", "")
        row_index = item.get("row_index", -1)
        
        rows.append({
            "Node Label": label,
            "At Level": node_id,
            "Issue Type": issue_type,
            "Suggested Action": suggested_action,
            "Row Index": row_index,
        })

    df_rf = pd.DataFrame(rows).sort_values(["At Level", "Node Label"])
    # Limit preview to first 100 rows for speed
    st.dataframe(df_rf.head(100), use_container_width=True, height=280)

    st.download_button(
        "Download missing red flags (CSV)",
        data=df_rf.to_csv(index=False).encode("utf-8"),
        file_name="validation_missing_redflags.csv",
        mime="text/csv",
    )

    # Per-row jump buttons (paged)
    st.caption("Quick-jump: focus a parent in the Symptoms tab for editing.")
    page_size = st.selectbox("Rows per page", [10, 25, 50, 100], index=1, key="val_rf_pagesize")
    total = len(df_rf)
    max_page = max(1, int(np.ceil(total / page_size)))
    page = st.number_input("Page", min_value=1, max_value=max_page, value=1, step=1, key="val_rf_page")
    start = (page - 1) * page_size
    end = min(start + page_size, total)
    for idx in range(start, end):
        row = df_rf.iloc[idx]
        c1, c2 = st.columns([6, 1])
        with c1:
            st.write(f"• **{row['Parent (path)']}** — children: {row['Children'] or '(none)'}")
        with c2:
            if st.button("Find in Symptoms", key=f"val_rf_jump_{idx}"):
                _jump_to_symptoms(int(row["level_int"]), tuple(row["parent_tuple"]))

    st.download_button(
        "Download missing Red Flag coverage (CSV)",
        data=df_rf.drop(columns=["parent_tuple", "level_int"]).to_csv(index=False).encode("utf-8"),
        file_name="validation_missing_redflags.csv",
        mime="text/csv",
    )


def _render_empty_branches(empty_branches: List[Dict[str, Any]]):
    st.subheader("🌿 Empty Branches")

    st.caption("Nodes that have no children but are not marked as terminal (no actions or triage). Consider adding children or marking as terminal.")

    if not empty_branches:
        st.success("✅ No empty branches detected. 🎉")
        return

    rows = []
    for item in empty_branches:
        label = item.get("label", "")
        level = int(item.get("level", 1))
        node_id = item.get("node_id", f"Node {level}")
        row_index = item.get("row_index", -1)
        
        rows.append({
            "Node Label": label,
            "At Level": node_id,
            "Row Index": row_index,
        })

    df_eb = pd.DataFrame(rows).sort_values(["At Level", "Node Label"])
    # Limit preview to first 100 rows for speed
    st.dataframe(df_eb.head(100), use_container_width=True, height=220)

    st.download_button(
        "Download empty branches (CSV)",
        data=df_eb.to_csv(index=False).encode("utf-8"),
        file_name="validation_empty_branches.csv",
        mime="text/csv",
    )


def render():
    st.header("🔎 Validation")

    # Choose data source
    sources = []
    if st.session_state.get("upload_workbook", {}):
        sources.append("Upload workbook")
    if st.session_state.get("gs_workbook", {}):
        sources.append("Google Sheets workbook")

    if not sources:
        st.info("ℹ️ Load a workbook first in the **Source** tab.")
        return

    source = st.radio("Choose data source", sources, horizontal=True, key="val_source_sel")

    if source == "Upload workbook":
        wb = st.session_state.get("upload_workbook", {})
        override_root = "branch_overrides_upload"
    else:
        wb = st.session_state.get("gs_workbook", {})
        override_root = "branch_overrides_gs"

    if not wb:
        st.warning("⚠️ No sheets found in the selected source.")
        return

    # Sheet selector
    sheet = st.selectbox("Sheet", list(wb.keys()), key="val_sheet_sel")
    df = wb.get(sheet, pd.DataFrame())
    if df.empty or not validate_headers(df):
        st.info("ℹ️ Selected sheet is empty or headers mismatch.")
        return

    # Options
    st.markdown("#### Checks to run")
    c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
    with c1:
        chk_orphans = st.checkbox("Orphan nodes", value=True, key="val_chk_orphans")
    with c2:
        chk_loops = st.checkbox("Loops", value=True, key="val_chk_loops")
    with c3:
        chk_rf = st.checkbox("Missing Red Flag coverage", value=True, key="val_chk_rf")
    with c4:
        chk_eb = st.checkbox("Empty branches", value=True, key="val_chk_eb")

    # Pull overrides + symptom quality map
    overrides_sheet = st.session_state.get(override_root, {}).get(sheet, {})
    quality_map = st.session_state.get("symptom_quality", {})  # {symptom: "Red Flag"|"Normal"}

    st.markdown("---")
    if st.button("▶ Run validation", type="primary", key="val_run_btn"):
        st.session_state["val_run_requested"] = True

    if not st.session_state.get("val_run_requested", True):
        st.info("ℹ️ Click **Run validation** to generate the report.")
        return

    # Compute report
    try:
        report = _cached_compute_validation_report(df)
        orphans = report.get("orphan_nodes", []) if chk_orphans else []
        loops = report.get("loops", []) if chk_loops else []
        missing_rf = report.get("missing_red_flags", []) if chk_rf else []
        empty_branches = report.get("empty_branches", []) if chk_eb else []
    except AssertionError as e:
        st.error(f"❌ {str(e)}")
        return
    except Exception as e:
        st.error(f"❌ Validation failed: {e}")
        return

    # Summary chips
    st.markdown("#### Summary")
    chip_cols = st.columns([1, 1, 1, 1])
    with chip_cols[0]:
        st.metric("Orphan nodes", len(orphans))
    with chip_cols[1]:
        st.metric("Loops found", len(loops))
    with chip_cols[2]:
        st.metric("Missing Red Flag (parents)", len(missing_rf))
    with chip_cols[3]:
        st.metric("Empty branches", len(empty_branches))

    st.markdown("---")

    # Render sections
    if chk_orphans:
        _render_orphans(orphans)
        st.markdown("---")
    if chk_loops:
        _render_loops(loops)
        st.markdown("---")
    if chk_rf:
        _render_missing_redflag(missing_rf)
        st.markdown("---")
    if chk_eb:
        _render_empty_branches(empty_branches)
        st.markdown("---")

    # Combined export (JSON)
    combined = {
        "sheet": sheet,
        "orphans": orphans if chk_orphans else [],
        "loops": loops if chk_loops else [],
        "missing_redflag": missing_rf if chk_rf else [],
        "empty_branches": empty_branches if chk_eb else [],
    }
    try:
        import json
        json_bytes = json.dumps(combined, indent=2).encode("utf-8")
        st.download_button(
            "Download full validation report (JSON)",
            data=json_bytes,
            file_name=f"{sheet}_validation_report.json",
            mime="application/json",
        )
    except Exception:
        pass

    st.caption("Tip: Use **Find in Symptoms** to quickly jump to a parent for editing; then re-run validation here.")
