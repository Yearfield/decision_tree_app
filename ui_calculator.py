from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np
import streamlit as st

# ----------------- constants & helpers -----------------

CANON_HEADERS = [
    "Vital Measurement", "Node 1", "Node 2", "Node 3", "Node 4", "Node 5",
    "Diagnostic Triage", "Actions"
]
LEVEL_COLS = ["Node 1","Node 2","Node 3","Node 4","Node 5"]
MAX_LEVELS = 5


def normalize_text(x: str) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()


def validate_headers(df: pd.DataFrame) -> bool:
    return list(df.columns[:len(CANON_HEADERS)]) == CANON_HEADERS


def friendly_parent_label(level: int, parent_tuple: Tuple[str, ...]) -> str:
    """Human-friendly parent label for a Node {level} parent path."""
    if level == 1 and not parent_tuple:
        return "Top-level (Node 1) options"
    return " > ".join(parent_tuple) if parent_tuple else "Top-level (Node 1) options"


def _rows_match_parent(df: pd.DataFrame, vm: str, parent: Tuple[str,...]) -> pd.DataFrame:
    """Filter df to rows matching VM and the full parent tuple (Node 1..len(parent))."""
    if df is None or df.empty:
        return df
    m = (df["Vital Measurement"].map(normalize_text) == normalize_text(vm))
    for i, val in enumerate(parent, 1):
        m = m & (df[f"Node {i}"].map(normalize_text) == normalize_text(val))
    return df[m].copy()


def _children_distribution(
    df: pd.DataFrame,
    vm: str,
    parent: Tuple[str, ...],
    next_level: int,
    redflag_map: Optional[Dict[str, str]] = None
) -> pd.DataFrame:
    """
    Build a distribution for Node {next_level} children under (vm, parent).
    Returns DataFrame with: Option, Count, Percent, RedFlag (bool)
    """
    if next_level < 1 or next_level > MAX_LEVELS:
        return pd.DataFrame(columns=["Option","Count","Percent","RedFlag"])
    if df is None or df.empty:
        return pd.DataFrame(columns=["Option","Count","Percent","RedFlag"])

    sub = _rows_match_parent(df, vm, parent)
    if sub.empty:
        return pd.DataFrame(columns=["Option","Count","Percent","RedFlag"])

    col = f"Node {next_level}"
    series = sub[col].map(normalize_text)
    counts = series[series != ""].value_counts(dropna=True)
    if counts.empty:
        return pd.DataFrame(columns=["Option","Count","Percent","RedFlag"])

    total = int(counts.sum())
    data = []
    for opt, cnt in counts.items():
        pct = (cnt / total) * 100.0 if total else 0.0
        rf = False
        if redflag_map:
            rf = (redflag_map.get(opt, "Normal") == "Red Flag")
        data.append({"Option": opt, "Count": int(cnt), "Percent": round(pct, 1), "RedFlag": rf})
    df_out = pd.DataFrame(data).sort_values(["Count","Option"], ascending=[False, True]).reset_index(drop=True)
    return df_out


def _available_children(
    df: pd.DataFrame,
    vm: str,
    parent: Tuple[str, ...],
    next_level: int
) -> List[str]:
    """List available child options for selection at Node {next_level} given (vm, parent)."""
    d = _children_distribution(df, vm, parent, next_level, redflag_map=None)
    return d["Option"].tolist() if not d.empty else []


def _badge_redflag_coverage(dist_df: pd.DataFrame) -> str:
    """
    Simple coverage badge: if any option is RedFlag=True → green badge; else grey.
    Returns small HTML snippet.
    """
    if dist_df.empty:
        return "<span style='color:#64748b;'>No options</span>"
    has_rf = bool(dist_df["RedFlag"].fillna(False).any())
    if has_rf:
        return "<span style='background:#10b98122; color:#065f46; padding:2px 6px; border-radius:10px; font-size:12px;'>Red Flag present ✓</span>"
    return "<span style='background:#e5e7eb; color:#374151; padding:2px 6px; border-radius:10px; font-size:12px;'>No Red Flag in this set</span>"


def _export_prob_tables(tables: List[Tuple[int, Tuple[str,...], pd.DataFrame]]) -> bytes:
    """
    Combine multiple probability tables into one CSV.
    Each tuple: (next_level, parent_tuple, dist_df)
    Output columns: Level, ParentPath, Option, Count, Percent, RedFlag
    """
    rows = []
    for lvl, parent, df_prob in tables:
        parent_label = friendly_parent_label(lvl, parent)
        for _, r in df_prob.iterrows():
            rows.append({
                "Level": lvl,
                "ParentPath": parent_label,
                "Option": r.get("Option",""),
                "Count": int(r.get("Count",0) or 0),
                "Percent": float(r.get("Percent",0.0) or 0.0),
                "RedFlag": bool(r.get("RedFlag", False))
            })
    if not rows:
        return "".encode("utf-8")
    csv = pd.DataFrame(rows).to_csv(index=False).encode("utf-8")
    return csv


# ----------------- UI -----------------

def render():
    st.header("🧮 Calculator")

    # Require a workspace context (selected in Workspace Selection)
    ctx = st.session_state.get("work_context")
    if not ctx:
        st.info("Select a sheet in **Workspace Selection** first.")
        return

    source = ctx.get("source")  # "upload" | "gs"
    sheet = ctx.get("sheet")
    if not sheet:
        st.info("Select a sheet in **Workspace Selection** first.")
        return

    if source == "upload":
        wb = st.session_state.get("upload_workbook", {})
    else:
        wb = st.session_state.get("gs_workbook", {})

    df = wb.get(sheet, pd.DataFrame())
    if df is None or df.empty or not validate_headers(df):
        st.info("Current sheet is empty or headers mismatch.")
        return

    # Load Red Flag map from Dictionary tab (if any)
    quality_map = st.session_state.get("symptom_quality", {})  # {symptom: "Red Flag"|"Normal"}

    # Vital Measurements available
    vms = sorted(set(df["Vital Measurement"].map(normalize_text).replace("", np.nan).dropna().tolist()))
    if not vms:
        st.info("No Vital Measurements found.")
        return

    st.subheader("Pick a Vital Measurement")
    vm = st.selectbox("Vital Measurement", vms, key="calc_vm_sel")
    if not vm:
        return

    st.markdown("—")

    # State keys for the path explorer (reset on VM change)
    vm_key = f"calc_vm_current::{vm}"
    vm_prev = st.session_state.get("calc_vm_prev")
    if vm_prev != vm:
        # reset choices when VM changes
        for k in ["calc_n1","calc_n2","calc_n3","calc_n4","calc_n5"]:
            st.session_state.pop(k, None)
        st.session_state["calc_vm_prev"] = vm

    # --- Node 1 distribution (top-level) ---
    st.markdown("### Node 1 distribution")
    dist_tables: List[Tuple[int, Tuple[str,...], pd.DataFrame]] = []

    dist_n1 = _children_distribution(df, vm, parent=tuple(), next_level=1, redflag_map=quality_map)
    dist_tables.append((1, tuple(), dist_n1))

    if dist_n1.empty:
        st.info("No Node 1 options found for this Vital Measurement.")
        return

    # Coverage badge
    st.markdown(_badge_redflag_coverage(dist_n1), unsafe_allow_html=True)
    st.dataframe(dist_n1, use_container_width=True)
    try:
        st.bar_chart(dist_n1.set_index("Option")["Count"])
    except Exception:
        pass

    # Node 1 select
    n1_choices = _available_children(df, vm, tuple(), 1)
    n1 = st.selectbox("Choose Node 1", ["(pick)"] + n1_choices, key="calc_n1")
    if n1 == "(pick)" or not n1:
        # export just Node 1 table if wanted
        csv1 = _export_prob_tables(dist_tables)
        st.download_button("Download probability tables (CSV)", data=csv1, file_name=f"{vm}_node1_probabilities.csv", mime="text/csv")
        st.caption("Pick Node 1 to explore deeper nodes.")
        return

    # --- Node 2 distribution ---
    parent_1 = (n1,)
    st.markdown(f"### Node 2 distribution — Parent: **{friendly_parent_label(2, parent_1)}**")
    dist_n2 = _children_distribution(df, vm, parent=parent_1, next_level=2, redflag_map=quality_map)
    dist_tables.append((2, parent_1, dist_n2))

    if dist_n2.empty:
        st.warning("No Node 2 options found under this parent.")
    else:
        st.markdown(_badge_redflag_coverage(dist_n2), unsafe_allow_html=True)
        st.dataframe(dist_n2, use_container_width=True)
        try:
            st.bar_chart(dist_n2.set_index("Option")["Count"])
        except Exception:
            pass

    n2_choices = _available_children(df, vm, parent_1, 2)
    n2 = st.selectbox("Choose Node 2", ["(pick)"] + n2_choices, key="calc_n2")
    if n2 == "(pick)" or not n2:
        csv2 = _export_prob_tables(dist_tables)
        st.download_button("Download probability tables (CSV)", data=csv2, file_name=f"{vm}_node1_node2_probabilities.csv", mime="text/csv")
        if dist_n2.empty:
            st.caption("Pick a different Node 1 to explore.")
        else:
            st.caption("Pick Node 2 to explore deeper nodes.")
        return

    # --- Node 3 distribution ---
    parent_2 = (n1, n2)
    st.markdown(f"### Node 3 distribution — Parent: **{friendly_parent_label(3, parent_2)}**")
    dist_n3 = _children_distribution(df, vm, parent=parent_2, next_level=3, redflag_map=quality_map)
    dist_tables.append((3, parent_2, dist_n3))

    if dist_n3.empty:
        st.warning("No Node 3 options found under this parent.")
    else:
        st.markdown(_badge_redflag_coverage(dist_n3), unsafe_allow_html=True)
        st.dataframe(dist_n3, use_container_width=True)
        try:
            st.bar_chart(dist_n3.set_index("Option")["Count"])
        except Exception:
            pass

    n3_choices = _available_children(df, vm, parent_2, 3)
    n3 = st.selectbox("Choose Node 3", ["(pick)"] + n3_choices, key="calc_n3")
    if n3 == "(pick)" or not n3:
        csv3 = _export_prob_tables(dist_tables)
        st.download_button("Download probability tables (CSV)", data=csv3, file_name=f"{vm}_node1_node2_node3_probabilities.csv", mime="text/csv")
        if dist_n3.empty:
            st.caption("Pick a different Node 2 to explore.")
        else:
            st.caption("Pick Node 3 to explore deeper nodes.")
        return

    # --- Node 4 distribution ---
    parent_3 = (n1, n2, n3)
    st.markdown(f"### Node 4 distribution — Parent: **{friendly_parent_label(4, parent_3)}**")
    dist_n4 = _children_distribution(df, vm, parent=parent_3, next_level=4, redflag_map=quality_map)
    dist_tables.append((4, parent_3, dist_n4))

    if dist_n4.empty:
        st.warning("No Node 4 options found under this parent.")
    else:
        st.markdown(_badge_redflag_coverage(dist_n4), unsafe_allow_html=True)
        st.dataframe(dist_n4, use_container_width=True)
        try:
            st.bar_chart(dist_n4.set_index("Option")["Count"])
        except Exception:
            pass

    n4_choices = _available_children(df, vm, parent_3, 4)
    n4 = st.selectbox("Choose Node 4", ["(pick)"] + n4_choices, key="calc_n4")
    if n4 == "(pick)" or not n4:
        csv4 = _export_prob_tables(dist_tables)
        st.download_button("Download probability tables (CSV)", data=csv4, file_name=f"{vm}_node1_node2_node3_node4_probabilities.csv", mime="text/csv")
        if dist_n4.empty:
            st.caption("Pick a different Node 3 to explore.")
        else:
            st.caption("Pick Node 4 to explore the final node.")
        return

    # --- Node 5 distribution ---
    parent_4 = (n1, n2, n3, n4)
    st.markdown(f"### Node 5 distribution — Parent: **{friendly_parent_label(5, parent_4)}**")
    dist_n5 = _children_distribution(df, vm, parent=parent_4, next_level=5, redflag_map=quality_map)
    dist_tables.append((5, parent_4, dist_n5))

    if dist_n5.empty:
        st.warning("No Node 5 options found under this parent.")
    else:
        st.markdown(_badge_redflag_coverage(dist_n5), unsafe_allow_html=True)
        st.dataframe(dist_n5, use_container_width=True)
        try:
            st.bar_chart(dist_n5.set_index("Option")["Count"])
        except Exception:
            pass

    # Export all tables we’ve shown so far in this path
    csv_all = _export_prob_tables(dist_tables)
    st.download_button("Download probability tables for this path (CSV)", data=csv_all, file_name=f"{vm}_path_probabilities.csv", mime="text/csv")

    # Reset path
    if st.button("Reset path selections"):
        for k in ["calc_n1","calc_n2","calc_n3","calc_n4","calc_n5"]:
            st.session_state.pop(k, None)
        st.rerun()  # updated from st.experimental_rerun()
