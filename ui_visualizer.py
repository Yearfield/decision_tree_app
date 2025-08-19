# ui_visualizer.py

from typing import Dict, List, Tuple, Optional, Set
import json
import pandas as pd
import streamlit as st
from pyvis.network import Network

from utils import (
    CANON_HEADERS, LEVEL_COLS, MAX_LEVELS,
    normalize_text, validate_headers,
)

# -----------------------------
# Helpers
# -----------------------------




def _get_current_df() -> Tuple[Optional[pd.DataFrame], str]:
    """
    Picks the active sheet from the 'work_context' if available,
    otherwise tries Upload -> Google Sheets as a fallback.
    Returns (df, sheet_name or hint).
    """
    ctx = st.session_state.get("work_context", {})
    src = ctx.get("source")
    sheet = ctx.get("sheet")

    if src == "upload":
        wb = st.session_state.get("upload_workbook", {})
        if sheet in wb:
            return wb[sheet], sheet
    elif src == "gs":
        wb = st.session_state.get("gs_workbook", {})
        if sheet in wb:
            return wb[sheet], sheet

    # Fallbacks
    wb_u = st.session_state.get("upload_workbook", {})
    if wb_u:
        name = next(iter(wb_u))
        return wb_u[name], name

    wb_g = st.session_state.get("gs_workbook", {})
    if wb_g:
        name = next(iter(wb_g))
        return wb_g[name], name

    return None, "(no sheet loaded)"


def _unique_vm_values(df: pd.DataFrame) -> List[str]:
    if "Vital Measurement" not in df.columns:
        return []
    vals = [normalize_text(x) for x in df["Vital Measurement"].dropna().astype(str)]
    uniq = []
    seen = set()
    for v in vals:
        if v and v not in seen:
            seen.add(v); uniq.append(v)
    return sorted(uniq)


def _build_edges(
    df: pd.DataFrame,
    limit_rows: int = 20000,
    scope_vm: Optional[str] = None,
    collapse_by_label_per_level: bool = True,
) -> Tuple[Set[str], List[Tuple[str, str]], Dict[str, Dict[str, str]]]:
    """
    Build a set of unique node ids and list of edges for the tree.
    If collapse_by_label_per_level=True, all identical labels at a given level
    are merged to one node (per level). Node id becomes f"L{level}:{label}".

    Returns:
      (nodes, edges, node_attrs)
        nodes: set of node ids (str)
        edges: list of (src_id, dst_id)
        node_attrs: node_id -> {"label": <display text>, "title": <tooltip>}
    """
    nodes: Set[str] = set()
    edges: List[Tuple[str, str]] = []
    node_attrs: Dict[str, Dict[str, str]] = {}

    if df is None or df.empty or not validate_headers(df):
        return nodes, edges, node_attrs

    # Filter by VM if selected
    df2 = df.copy()
    if scope_vm:
        df2 = df2[df2["Vital Measurement"].astype(str).map(normalize_text) == normalize_text(scope_vm)]

    # Iterate rows (respect a hard cap)
    for i, (_, row) in enumerate(df2.iterrows()):
        if i >= limit_rows:
            break

        vm = normalize_text(row.get("Vital Measurement", ""))
        path = [normalize_text(row.get(c, "")) for c in LEVEL_COLS]
        # Create edges from VM -> Node1, Node1 -> Node2, ... only when both sides exist
        prev_id = None
        if vm:
            vm_id = f"L0:{vm}" if collapse_by_label_per_level else f"L0:{vm}:{i}"
            if vm_id not in nodes:
                nodes.add(vm_id)
                node_attrs[vm_id] = {"label": vm, "title": f"Vital Measurement: {vm}"}
            prev_id = vm_id

        for li, label in enumerate(path, start=1):
            if not label:
                break
            node_id = f"L{li}:{label}" if collapse_by_label_per_level else f"L{li}:{label}:{i}"
            if node_id not in nodes:
                nodes.add(node_id)
                node_attrs[node_id] = {"label": label, "title": f"Node {li}: {label}"}

            if prev_id is not None:
                edges.append((prev_id, node_id))
            prev_id = node_id

    # De-duplicate edges
    edges = list({(a, b) for (a, b) in edges})
    return nodes, edges, node_attrs


def _apply_pyvis_options(net: Network, hierarchical: bool):
    """
    Set pyvis options with valid JSON (NOT JavaScript).
    We build JSON via json.dumps to avoid JSONDecode errors in pyvis/options.py.
    """
    if hierarchical:
        options = {
            "layout": {
                "hierarchical": {
                    "enabled": True,
                    "direction": "UD",
                    "sortMethod": "directed"
                }
            },
            "physics": {"enabled": False},
            "nodes": {"shape": "dot", "size": 12},
            "edges": {"arrows": {"to": {"enabled": True}}}
        }
    else:
        options = {
            "physics": {"enabled": True, "stabilization": {"enabled": True}},
            "nodes": {"shape": "dot", "size": 12},
            "edges": {"arrows": {"to": {"enabled": True}}}
        }
    net.set_options(json.dumps(options))


# -----------------------------
# UI
# -----------------------------

def render():
    st.header("🌳 Visualizer")

    df, sheet_name = _get_current_df()
    if df is None or df.empty or not validate_headers(df):
        st.info("No valid sheet found. Load data in **Source** and select a sheet in **Workspace**.")
        return

    st.caption(f"Showing sheet: **{sheet_name}**")

    # Controls
    c1, c2, c3, c4 = st.columns([1, 1, 1, 1])

    with c1:
        hierarchical = st.checkbox("Hierarchical layout", value=True, help="Top-to-bottom layered view.")

    with c2:
        collapse = st.checkbox(
            "Merge same labels per level",
            value=True,
            help="If ON, identical labels at the same level are merged to one node."
        )

    with c3:
        vms = _unique_vm_values(df)
        vm_scope = st.selectbox(
            "Filter by Vital Measurement",
            options=["(All)"] + vms,
            index=0,
        )
        vm_sel = None if vm_scope == "(All)" else vm_scope

    with c4:
        limit = st.number_input(
            "Row limit",
            min_value=100,
            max_value=100000,
            value=5000,
            step=500,
            help="Maximum rows to scan when building the graph."
        )

    st.markdown("---")

    # Build graph data
    nodes, edges, node_attrs = _build_edges(
        df,
        limit_rows=int(limit),
        scope_vm=vm_sel,
        collapse_by_label_per_level=collapse,
    )

    if not nodes:
        st.info("No nodes to visualize with the current filters.")
        return

    # Pyvis network
    net = Network(height="650px", width="100%", directed=True, notebook=False)
    _apply_pyvis_options(net, hierarchical=hierarchical)

    # Add nodes
    for nid in nodes:
        info = node_attrs.get(nid, {})
        label = info.get("label", nid)
        title = info.get("title", label)
        # Choose color lightly by level
        try:
            level_prefix = nid.split(":")[0]  # like "L2"
            level_num = int(level_prefix[1:])
        except Exception:
            level_num = 0
        color = [
            "#4f46e5",  # L0
            "#2563eb",  # L1
            "#059669",  # L2
            "#16a34a",  # L3
            "#d97706",  # L4
            "#dc2626",  # L5
        ][min(level_num, 5)]

        net.add_node(nid, label=label, title=title, color=color)

    # Add edges
    for (src, dst) in edges:
        net.add_edge(src, dst)

    # Render to Streamlit
    try:
        html = net.generate_html()  # returns full HTML string
        st.components.v1.html(html, height=680, scrolling=True)
    except Exception:
        # Fallback: write to a temporary file then read it back
        import tempfile, os
        with tempfile.TemporaryDirectory() as tmpd:
            out = os.path.join(tmpd, "graph.html")
            net.write_html(out)
            with open(out, "r", encoding="utf-8") as f:
                st.components.v1.html(f.read(), height=680, scrolling=True)

    # Small legend / tips
    with st.expander("Legend & Tips", expanded=False):
        st.markdown(
            """
            - **Colors by level** (VM → Node 1 → Node 2 → Node 3 → Node 4 → Node 5).
            - Turn **Merge same labels per level** OFF to see every occurrence (more crowded).
            - Use **Filter by Vital Measurement** to focus a single tree.
            - Increase **Row limit** if your sheet is large and you need deeper coverage.
            """
        )
