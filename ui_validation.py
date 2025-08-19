# ui_validation.py

# TODO[Step10]: UX consistency pass:
# - Standardize header icon text, KPI row, and Save/Push controls
# - Ensure previews cap at .head(100) and metrics are cached

from typing import Dict, List, Tuple, Any
import pandas as pd
import numpy as np
import streamlit as st

from utils import (
    CANON_HEADERS, LEVEL_COLS, MAX_LEVELS,
    normalize_text, validate_headers, friendly_parent_label,
    level_key_tuple, order_decision_tree, get_current_df_and_sheet,
)
from ui_helpers import render_preview_caption, st_success, st_warning, st_error, st_info

# ========= Utility functions for dual-schema support =========

def _safe_get(d, keys, default=""):
    """Safely get value from dict using multiple possible keys."""
    for k in keys:
        if isinstance(d, dict) and k in d: 
            return d[k]
    return default


def _coerce_list_to_df(items):
    """Convert items to DataFrame, supporting both legacy and new structured outputs."""
    if isinstance(items, pd.DataFrame):
        return items.copy()
    elif isinstance(items, list) and items and isinstance(items[0], dict):
        return pd.DataFrame(items)
    else:
        return pd.DataFrame()


def _normalize_missing_rf_df(df):
    """
    Normalize missing red flags DataFrame to ensure consistent column structure.
    Supports both legacy schema (Parent (path), Children) and new schema (parent_path, children, etc.).
    """
    if df.empty:
        return df
    
    # Ensure required columns exist with fallbacks
    normalized = pd.DataFrame()
    
    # Map columns with fallbacks for dual-schema support
    normalized["Parent (path)"] = df.get("Parent (path)", df.get("parent_path", ""))
    normalized["Children"] = df.get("Children", df.get("children", ""))
    normalized["Label"] = df.get("Label", df.get("label", ""))
    normalized["Level"] = df.get("Level", df.get("level", -1))
    normalized["Issue"] = df.get("Issue", df.get("issue", "Missing red flag"))
    normalized["Severity"] = df.get("Severity", df.get("severity", "warning"))
    normalized["Row"] = df.get("Row", df.get("row", -1))
    
    # Handle children column - if it's a list, join with commas
    if "Children" in normalized.columns:
        normalized["Children"] = normalized["Children"].apply(
            lambda x: ", ".join(x) if isinstance(x, list) else str(x)
        )
    
    return normalized


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
    render_preview_caption(df_orph.head(100), df_orph, max_rows=100)

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
    render_preview_caption(df_loops.head(100), df_loops, max_rows=100)

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
        st_success("All nodes have appropriate red flag coverage. 🎉")
        return

    # Convert to DataFrame and normalize columns for dual-schema support
    df = _coerce_list_to_df(miss_rf)
    df = _normalize_missing_rf_df(df)
    
    if df.empty:
        st_success("✅ No missing red flag placements found.")
        return
    
    # Show compact list view
    st.caption(f"Showing {min(len(df), 100)} of {len(df)} entries")
    for rec in df.head(100).to_dict("records"):
        parent = _safe_get(rec, ["Parent (path)"], "-")
        children = _safe_get(rec, ["Children"], "")
        label = _safe_get(rec, ["Label"], "")
        level = _safe_get(rec, ["Level"], -1)
        issue = _safe_get(rec, ["Issue"], "Missing red flag")
        st.write(f"• **{parent}** — children: {children or '(none)'}  ·  Label: **{label}**  ·  Node: {level}  ·  ℹ️ {issue}")
    
    # Also show dataframe preview
    st.markdown("---")
    st.markdown("#### Detailed View")
    st.dataframe(df.head(100), use_container_width=True, height=280)
    render_preview_caption(df.head(100), df, max_rows=100)
    
    # Download functionality
    st.download_button(
        "Download missing red flags (CSV)",
        data=df.to_csv(index=False).encode("utf-8"),
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
    render_preview_caption(df_eb.head(100), df_eb, max_rows=100)

    st.download_button(
        "Download empty branches (CSV)",
        data=df_eb.to_csv(index=False).encode("utf-8"),
        file_name="validation_empty_branches.csv",
        mime="text/csv",
    )


def render():
    st.header("🔎 Validation")

    # Get DataFrame using shared helper
    df, sheet_name, source_code = get_current_df_and_sheet()
    if df is None or df.empty or not validate_headers(df):
        st_warning("⚠️ No data loaded. Please load a sheet in the **Workspace** tab.")
        if st.session_state.get("__debug"):
            ctx = st.session_state.get("work_context", {})
            st.caption(f"🔎 ctx={ctx} · upload={len(st.session_state.get('upload_workbook', {}))} sheets · "
                       f"gs={len(st.session_state.get('gs_workbook', {}))} sheets")
        st.stop()

    # Determine override root based on source
    override_root = "branch_overrides_upload" if source_code == "upload" else "branch_overrides_gs"

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
    overrides_sheet = st.session_state.get(override_root, {}).get(sheet_name, {})
    quality_map = st.session_state.get("symptom_quality", {})  # {symptom: "Red Flag"|"Normal"}

    st.markdown("---")
    if st.button("▶ Run validation", type="primary", key="val_run_btn"):
        st.session_state["val_run_requested"] = True

    if not st.session_state.get("val_run_requested", True):
        st_info("Click **Run validation** to generate the report.")
        return

    # Compute report
    try:
        report = _cached_compute_validation_report(df)
        orphans = report.get("orphan_nodes", []) if chk_orphans else []
        loops = report.get("loops", []) if chk_loops else []
        missing_rf = report.get("missing_red_flags", []) if chk_rf else []
        empty_branches = report.get("empty_branches", []) if chk_eb else []
    except AssertionError as e:
        st_error(f"{str(e)}")
        return
    except Exception as e:
        st_error(f"Validation failed: {e}")
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
        "sheet": sheet_name,
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
            file_name=f"{sheet_name}_validation_report.json",
            mime="application/json",
        )
    except Exception:
        pass

    st.caption("Tip: Use **Find in Symptoms** to quickly jump to a parent for editing; then re-run validation here.")
