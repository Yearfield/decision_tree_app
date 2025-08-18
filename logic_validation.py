# ui_validation.py

from typing import Dict, List, Tuple, Any
import pandas as pd
import numpy as np
import streamlit as st

from utils import (
    CANON_HEADERS, LEVEL_COLS, MAX_LEVELS,
    normalize_text, validate_headers, friendly_parent_label,
    level_key_tuple,
)

# ----------------- Import logic functions (robust to naming variants) -----------------

API_MODE = "none"

_detect_orphan_nodes = None
_detect_loops = None
_detect_missing_red_flags = None
_validate_workbook = None
_compute_validation_report = None
_detect_missing_red_flags = None
_detect_orphans_legacy = None

# Import validation logic functions
try:
    from logic_validation_functions import (
        detect_orphan_nodes as _detect_orphan_nodes,
        detect_loops as _detect_loops,
        detect_missing_red_flags as _detect_missing_red_flags,
        compute_validation_report as _compute_validation_report,
    )
    API_MODE = "new"
except Exception:
    # Fallback: try individual functions
    try:
        from logic_validation_functions import (
            detect_orphan_nodes as _detect_orphan_nodes,
            detect_loops as _detect_loops,
            detect_missing_red_flags as _detect_missing_red_flags,
        )
        API_MODE = "individual"
    except Exception:
        # If all else fails, set to None
        _detect_orphan_nodes = None
        _detect_loops = None
        _detect_missing_red_flags = None
        _compute_validation_report = None
        API_MODE = "none"


# ----------------- session helpers -----------------

def _ss_get(key, default):
    if key not in st.session_state:
        st.session_state[key] = default
    return st.session_state[key]


def _jump_to_symptoms(level: int, parent_tuple: Tuple[str, ...]):
    """
    Prime Symptoms tab filters so the user can quickly land on the parent they need.
    """
    st.session_state["sym_level_sel"] = level
    pretty = " > ".join(parent_tuple) if parent_tuple else friendly_parent_label(level, tuple())
    st.session_state["sym_search_pending"] = pretty.lower()
    st.success("Primed the Symptoms tab search for this parent. Switch to the 🧬 Symptoms tab.")


# ----------------- render helpers (normalize outputs to tables) -----------------

def _render_orphans_any(orphans: Any):
    st.subheader("🧩 Orphan nodes")
    st.caption("A child label appears but never as a parent at the next level, where a branch might reasonably continue.")

    # Accept list[str], list[dict], or None
    if not orphans:
        st.success("No orphans detected. 🎉")
        return

    rows = []
    # If new-API (list[str]) or arbitrary strings
    if isinstance(orphans, list) and (len(orphans) == 0 or isinstance(orphans[0], str)):
        for s in orphans:
            rows.append({"Issue": s})
        df_orph = pd.DataFrame(rows)
        st.dataframe(df_orph, use_container_width=True, height=260)
        st.download_button(
            "Download orphans (CSV)",
            data=df_orph.to_csv(index=False).encode("utf-8"),
            file_name="validation_orphans.csv",
            mime="text/csv",
        )
        return

    # If structured (list[dict])
    for item in orphans:
        # Try to read our richer schema; fallback to generic
        label = item.get("child") or item.get("label") or item.get("node") or ""
        level = item.get("level") or ""
        parent_tuple = tuple(item.get("parent", [])) if isinstance(item.get("parent"), (list, tuple)) else ()
        mode = item.get("mode", "")
        example = " > ".join(parent_tuple) if parent_tuple else friendly_parent_label(int(level or 1), tuple())
        rows.append({
            "Node Label": label,
            "At Level": f"Node {level}" if level else "",
            "Mode": mode,
            "Example parent (where used as child)": example,
        })

    df_orph = pd.DataFrame(rows)
    if not df_orph.empty:
        df_orph = df_orph.sort_values(["At Level", "Node Label"])
    st.dataframe(df_orph, use_container_width=True, height=260)
    st.download_button(
        "Download orphans (CSV)",
        data=df_orph.to_csv(index=False).encode("utf-8"),
        file_name="validation_orphans.csv",
        mime="text/csv",
    )


def _render_loops_any(loops: Any):
    st.subheader("🔁 Loops (cycles)")

    if not loops:
        st.success("No cycles detected. 🎉")
        return

    rows = []
    # new-API returns list[str] with paths; legacy may return list[dict]
    if isinstance(loops, list) and (len(loops) == 0 or isinstance(loops[0], str)):
        for s in loops:
            rows.append({"Cycle": s})
    else:
        for item in loops:
            cycle = item.get("cycle") or item.get("path") or []
            if isinstance(cycle, list):
                if cycle and (len(cycle) > 1 and cycle[0] != cycle[-1]):
                    display = " → ".join(cycle + [cycle[0]])
                else:
                    display = " → ".join(cycle)
            else:
                display = str(cycle)
            rows.append({"Cycle": display})

    df_loops = pd.DataFrame(rows)
    st.dataframe(df_loops, use_container_width=True, height=220)
    st.download_button(
        "Download loops (CSV)",
        data=df_loops.to_csv(index=False).encode("utf-8"),
        file_name="validation_loops.csv",
        mime="text/csv",
    )


def _render_missing_rf_any(miss_rf: Any):
    st.subheader("🚩 Missing Red Flag coverage")
    st.caption("Parents that have children but none marked as Red Flag. Fix in Dictionary (flag the child) or edit branches.")

    if not miss_rf:
        st.success("All parents with children have Red Flag coverage. 🎉")
        return

    rows = []
    # new-API returns list[str]; legacy may return structured list[dict]
    if isinstance(miss_rf, list) and (len(miss_rf) == 0 or isinstance(miss_rf[0], str)):
        for s in miss_rf:
            rows.append({"Issue": s})
        df_rf = pd.DataFrame(rows)
        st.dataframe(df_rf, use_container_width=True, height=260)
        st.download_button(
            "Download missing Red Flag coverage (CSV)",
            data=df_rf.to_csv(index=False).encode("utf-8"),
            file_name="validation_missing_redflags.csv",
            mime="text/csv",
        )
        return

    # structured (preferred)
    for item in miss_rf:
        level = int(item.get("level", 1))
        parent_tuple = tuple(item.get("parent", []))
        children = item.get("children", [])
        rf_children = item.get("rf_children", [])
        parent_pretty = " > ".join(parent_tuple) if parent_tuple else friendly_parent_label(level, tuple())
        rows.append({
            "Parent (path)": parent_pretty,
            "At Level": f"Node {level}",
            "Children": ", ".join(children) if children else "(none)",
            "Red Flag children": ", ".join(rf_children) if rf_children else "(none)",
            "parent_tuple": parent_tuple,   # keep for jump
            "level_int": level,
        })

    df_rf = pd.DataFrame(rows)
    if not df_rf.empty:
        df_rf = df_rf.sort_values(["At Level", "Parent (path)"])
        st.dataframe(df_rf[["Parent (path)", "At Level", "Children", "Red Flag children"]],
                     use_container_width=True, height=280)

        # Quick jump buttons (paged)
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
    else:
        st.info("No detailed rows to display.")


# ----------------- main render -----------------

def render():
    st.header("🔎 Validation")

    # Choose data source
    sources = []
    if _ss_get("upload_workbook", {}):
        sources.append("Upload workbook")
    if _ss_get("gs_workbook", {}):
        sources.append("Google Sheets workbook")

    if not sources:
        st.info("Load a workbook first in the **Source** tab.")
        return

    source = st.radio("Choose data source", sources, horizontal=True, key="val_source_sel")

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
    sheet = st.selectbox("Sheet", list(wb.keys()), key="val_sheet_sel")
    df = wb.get(sheet, pd.DataFrame())
    if df.empty or not validate_headers(df):
        st.info("Selected sheet is empty or headers mismatch.")
        return

    # Options
    st.markdown("#### Checks to run")
    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        chk_orphans = st.checkbox("Orphan nodes", value=True, key="val_chk_orphans")
    with c2:
        chk_loops = st.checkbox("Loops", value=True, key="val_chk_loops")
    with c3:
        chk_rf = st.checkbox("Missing Red Flag coverage", value=True, key="val_chk_rf")

    # Pull overrides + symptom quality map (used by legacy API)
    overrides_sheet = _ss_get(override_root, {}).get(sheet, {})
    quality_map = _ss_get("symptom_quality", {})  # {symptom: "Red Flag"|"Normal"}

    st.markdown("---")
    if st.button("▶ Run validation", type="primary", key="val_run_btn"):
        st.session_state["val_run_requested"] = True

    if not _ss_get("val_run_requested", True):
        st.info("Click **Run validation** to generate the report.")
        return

    # ---- Run checks according to available API ----
    try:
        if API_MODE == "new":
            # New API returns simple lists of strings
            orphans = _detect_orphan_nodes(df) if chk_orphans and _detect_orphan_nodes else []
            loops = _detect_loops(df) if chk_loops and _detect_loops else []
            miss_rf = _detect_missing_red_flags(df) if chk_rf and _detect_missing_red_flags else []

        elif API_MODE == "combined":
            # Combined report -> attempt to map to our 3 sections
            report = _compute_validation_report(df)
            # Best-effort mapping
            orphans = report.get("orphans") or report.get("orphans_loose") or []
            loops = report.get("loops") or []
            miss_rf = report.get("missing_red_flags") or report.get("missing_redflag") or []

        elif API_MODE == "legacy":
            orphans = _detect_orphans_legacy(df, overrides=overrides_sheet, strict=False) if (chk_orphans and _detect_orphans_legacy) else []
            loops = _detect_loops(df) if (chk_loops and _detect_loops) else []
            miss_rf = _detect_missing_red_flags(df) if (chk_rf and _detect_missing_red_flags) else []

        else:
            st.error("No validation functions available. Please ensure logic_validation.py is present.")
            return

    except AssertionError as e:
        st.error(str(e))
        return
    except Exception as e:
        st.error(f"Validation failed: {e}")
        return

    # ---- Summary metrics ----
    st.markdown("#### Summary")
    chip_cols = st.columns([1, 1, 1])
    with chip_cols[0]:
        st.metric("Orphan nodes", len(orphans) if isinstance(orphans, list) else 0)
    with chip_cols[1]:
        st.metric("Loops found", len(loops) if isinstance(loops, list) else 0)
    with chip_cols[2]:
        st.metric("Missing Red Flag (parents)", len(miss_rf) if isinstance(miss_rf, list) else 0)

    st.markdown("---")

    # ---- Render sections ----
    if chk_orphans:
        _render_orphans_any(orphans)
        st.markdown("---")
    if chk_loops:
        _render_loops_any(loops)
        st.markdown("---")
    if chk_rf:
        _render_missing_rf_any(miss_rf)
        st.markdown("---")

    # ---- Combined export (JSON) ----
    combined = {
        "sheet": sheet,
        "orphans": orphans if chk_orphans else [],
        "loops": loops if chk_loops else [],
        "missing_redflag": miss_rf if chk_rf else [],
        "api_mode": API_MODE,
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

    st.caption("Tip: Use **Find in Symptoms** (where available) to jump to a parent for editing; then re-run validation here.")
