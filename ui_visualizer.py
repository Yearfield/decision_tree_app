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


def level_key_tuple(level: int, parent: Tuple[str, ...]) -> str:
    return f"L{level}|" + (">".join(parent) if parent else "<ROOT>")


def parent_key_from_row_strict(row: pd.Series, upto_level: int) -> Optional[Tuple[str, ...]]:
    if upto_level <= 1:
        return tuple()
    parent = []
    for c in LEVEL_COLS[:upto_level-1]:
        v = normalize_text(row[c])
        if v == "":
            return None
        parent.append(v)
    return tuple(parent)


def infer_branch_options(df: pd.DataFrame) -> Dict[str, List[str]]:
    """Build parent->children map for all levels from the dataframe (no overrides)."""
    store: Dict[str, List[str]] = {}
    for level in range(1, MAX_LEVELS+1):
        parent_to_children: Dict[Tuple[str, ...], List[str]] = {}
        for _, row in df.iterrows():
            child_col = LEVEL_COLS[level-1]
            if child_col not in df.columns:
                continue
            child = normalize_text(row[child_col])
            if child == "":
                continue
            parent = parent_key_from_row_strict(row, level)
            if parent is None:
                continue
            parent_to_children.setdefault(parent, [])
            parent_to_children[parent].append(child)
        for parent, children in parent_to_children.items():
            uniq, seen = [], set()
            for c in children:
                if c not in seen:
                    seen.add(c); uniq.append(c)
            store[level_key_tuple(level, parent)] = uniq
    return store


def infer_branch_options_with_overrides(df: pd.DataFrame, overrides: Dict[str, List[str]]) -> Dict[str, List[str]]:
    """Merge dataframe-derived store with overrides for visualization."""
    base = infer_branch_options(df)
    merged = dict(base)
    for k, v in (overrides or {}).items():
        vals = [normalize_text(x) for x in (v if isinstance(v, list) else [v])]
        merged[k] = vals
    return merged


def children_counts(df: pd.DataFrame, vm: str, parent: Tuple[str, ...], next_level: int) -> Dict[str, int]:
    """Return counts for Node {next_level} children under (vm, parent)."""
    if df is None or df.empty or not (1 <= next_level <= MAX_LEVELS):
        return {}
    m = (df["Vital Measurement"].map(normalize_text) == normalize_text(vm))
    for i, val in enumerate(parent, 1):
        m = m & (df[f"Node {i}"].map(normalize_text) == normalize_text(val))
    sub = df[m].copy()
    if sub.empty:
        return {}
    col = f"Node {next_level}"
    s = sub[col].map(normalize_text)
    vc = s[s != ""].value_counts(dropna=True)
    return {k: int(v) for k, v in vc.items()}


def friendly_parent_label(level: int, parent_tuple: Tuple[str, ...]) -> str:
    """Human-friendly name for the parent path feeding Node {level} children."""
    if level == 1 and not parent_tuple:
        return "Top-level (Node 1) options"
    return " > ".join(parent_tuple) if parent_tuple else "Top-level (Node 1) options"


# ----------------- UI: Visualizer -----------------

def render():
    st.header("🌐 Visualizer")

    # Workspace context (set in Workspace Selection)
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
        override_root = "branch_overrides_upload"
    else:
        wb = st.session_state.get("gs_workbook", {})
        override_root = "branch_overrides_gs"

    df = wb.get(sheet, pd.DataFrame())
    if df is None or df.empty or not validate_headers(df):
        st.info("Current sheet is empty or headers mismatch.")
        return

    # Optional: overrides & red flag map (from Dictionary)
    overrides_all = st.session_state.get(override_root, {})
    overrides_sheet = overrides_all.get(sheet, {})
    redflag_map = st.session_state.get("symptom_quality", {})  # {symptom: "Red Flag"|"Normal"}

    # Build store for visualization
    store = infer_branch_options_with_overrides(df, overrides_sheet)

    # Vital Measurements
    vms = sorted(set(df["Vital Measurement"].map(normalize_text).replace("", np.nan).dropna().tolist()))
    if not vms:
        st.info("No Vital Measurements found.")
        return

    c1, c2, c3, c4 = st.columns([2,1,1,1])
    with c1:
        vm = st.selectbox("Vital Measurement", vms, key="viz_vm")
    with c2:
        depth = st.selectbox("Depth", ["Node 1 only","Node 1 → Node 2"], index=1, key="viz_depth")
    with c3:
        physics = st.checkbox("Physics layout", value=True, key="viz_physics")
    with c4:
        hierarchical = st.checkbox("Hierarchical top-down", value=True, key="viz_hier")

    if not vm:
        return

    # Try to import pyvis
    try:
        from pyvis.network import Network
    except Exception:
        st.warning("PyVis is not installed. Run: `pip install pyvis`")
        return

    # Build a PyVis network
    net = Network(height="650px", width="100%", directed=True, notebook=False)
    if hierarchical:
        # Basic hierarchical options (top to bottom)
        net.set_options("""
        const options = {
          layout: { hierarchical: { enabled: true, direction: "UD", sortMethod: "hubsize", nodeSpacing: 200, levelSeparation: 200 } },
          physics: { enabled: false }
        }
        """)
    else:
        # Physics toggle
        net.set_options(f"""
        const options = {{
          physics: {{ enabled: {str(physics).lower()}, stabilization: {{ iterations: 200 }} }},
          nodes: {{ shape: "dot", size: 18 }}
        }}
        """)

    # Colors
    COLOR_VM = "#1f2937"           # charcoal
    COLOR_NODE = "#2563eb"         # blue
    COLOR_NODE_RF = "#ef4444"      # red
    COLOR_EDGE = "#94a3b8"         # slate
    COLOR_BG = "#ffffff"

    # Helper: add node with style
    def add_node(node_id: str, label: str, is_rf: bool = False, level_tag: str = ""):
        color = COLOR_NODE_RF if is_rf else COLOR_NODE
        title = label
        if level_tag:
            title = f"{level_tag} | {label}"
        net.add_node(node_id, label=label, title=title, color=color)

    def add_edge(src: str, dst: str, label: Optional[str] = None):
        if label:
            net.add_edge(src, dst, label=label, color=COLOR_EDGE, arrows="to", font={"align": "middle"})
        else:
            net.add_edge(src, dst, color=COLOR_EDGE, arrows="to")

    # Root node is the VM
    vm_node_id = f"VM::{vm}"
    net.add_node(vm_node_id, label=f"VM: {vm}", shape="box", color=COLOR_VM)

    # Node 1 children under root per VM
    # We prefer overrides if present in store; fallback to counting in df
    key_root = level_key_tuple(1, tuple())
    n1_children = [c for c in store.get(key_root, []) if normalize_text(c) != ""]
    # counts for Node 1 under VM
    counts_n1 = children_counts(df, vm=vm, parent=tuple(), next_level=1)

    # If empty, try from df directly (possible when overrides empty & no implicit rows)
    if not n1_children:
        n1_children = list(counts_n1.keys())

    # Add Node 1 nodes and edges
    for n1 in n1_children:
        is_rf = (redflag_map.get(n1, "Normal") == "Red Flag")
        cnt = counts_n1.get(n1, 0)
        label = f"{n1} ({cnt})" if cnt else n1
        n1_id = f"N1::{vm}::{n1}"
        add_node(n1_id, label, is_rf=is_rf, level_tag="Node 1")
        add_edge(vm_node_id, n1_id)

    # Depth 2: Node 2 children under each Node 1
    if depth.endswith("Node 2"):
        for n1 in n1_children:
            parent = (n1,)
            # store-based children first
            key_p = level_key_tuple(2, parent)
            n2_children = [c for c in store.get(key_p, []) if normalize_text(c) != ""]
            counts_n2 = children_counts(df, vm=vm, parent=parent, next_level=2)
            if not n2_children:
                n2_children = list(counts_n2.keys())
            for n2 in n2_children:
                is_rf2 = (redflag_map.get(n2, "Normal") == "Red Flag")
                cnt2 = counts_n2.get(n2, 0)
                label2 = f"{n2} ({cnt2})" if cnt2 else n2
                n2_id = f"N2::{vm}::{n1}::{n2}"
                add_node(n2_id, label2, is_rf=is_rf2, level_tag="Node 2")
                add_edge(f"N1::{vm}::{n1}", n2_id)

    # Render in Streamlit
    # We write to a temp html and load it back
    import os, tempfile
    with tempfile.TemporaryDirectory() as tmpd:
        out_path = os.path.join(tmpd, "tree_viz.html")
        net.write_html(out_path)
        with open(out_path, "r", encoding="utf-8") as f:
            html = f.read()
    st.components.v1.html(html, height=680, scrolling=True)

    # Download HTML artifact
    st.download_button(
        "Download this graph (HTML)",
        data=html.encode("utf-8"),
        file_name=f"viz_{vm.replace(' ','_')}_{'n1' if depth.startswith('Node 1') else 'n1_n2'}.html",
        mime="text/html"
    )

    st.caption("Tip: Toggle **Physics** for a free-form layout or use **Hierarchical** for a top-down tree.")
