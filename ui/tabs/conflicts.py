# ui/tabs/conflicts.py
"""
Conflicts Tab - Detect and resolve decision tree conflicts

DUPLICATE ROWS POLICY:
This app treats each row as a full path. Duplicate prefixes (first N nodes) are expected; 
we compute unique children per parent by set semantics. Downstream multiplication does not 
inflate child counts.

The indexer maintains counters for display purposes but uses unique labels to judge 
child-set size and conflicts. This ensures accurate conflict detection regardless of 
row duplication patterns.
"""

import streamlit as st
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional

from utils import (
    CANON_HEADERS, LEVEL_COLS, normalize_text, validate_headers
)
from utils.state import (
    get_active_workbook, get_current_sheet, get_active_df, 
    has_active_workbook, get_workbook_status, set_active_workbook, set_current_sheet, get_wb_nonce
)
from utils.helpers import normalize_text, normalize_child_set
from utils.constants import ROOT_PARENT_LABEL, MAX_CHILDREN_PER_PARENT, LEVEL_LABELS
from logic.tree import (
    infer_branch_options, set_level1_children, 
    build_parent_child_index, summarize_children_sets, find_label_set_mismatches, build_raw_plus_v630,
    majority_vote_5set
)
from logic.materialize import materialize_children_for_label_group, materialize_children_for_single_parent, materialize_children_for_label_across_tree
from ui.utils.rerun import safe_rerun
from ui.analysis import get_conflict_summary_with_root


@st.cache_data(ttl=600, show_spinner=False)
def get_conflict_summary(df: pd.DataFrame, nonce: str) -> Dict[str, Any]:
    """Get cached conflict summary for the given DataFrame and nonce."""
    # Use the new monolith-style analysis
    return get_conflict_summary_with_root(df, nonce)


def render():
    """Render the Conflicts tab for detecting and resolving decision tree conflicts."""
    try:
        st.header("⚖️ Conflicts")
        
        # Show duplicate rows policy
        with st.expander("ℹ️ Duplicate Rows Policy", expanded=False):
            st.info("""
            **This app treats each row as a full path.** 
            
            - Duplicate prefixes (first N nodes) are expected and normal
            - We compute unique children per parent by set semantics
            - Downstream multiplication does not inflate child counts
            - Counters are maintained for display but conflicts are judged by unique labels
            """)
        
        # Status badge
        has_wb, sheet_count, current_sheet = get_workbook_status()
        if has_wb and current_sheet:
            st.caption(f"Workbook: ✅ {sheet_count} sheet(s) • Active: **{current_sheet}**")
        else:
            st.caption("Workbook: ❌ not loaded")
        
        # Guard against no active workbook
        wb = get_active_workbook()
        sheet = get_current_sheet()
        if not wb or not sheet:
            st.warning("No active workbook/sheet. Load a workbook in 📂 Source or select a sheet in 🗂 Workspace.")
            return

        # Get active DataFrame
        df = get_active_df()
        if df is None:
            st.warning("No active sheet selected. Please load a workbook in the Source tab and select a sheet.")
            return
        
        if not validate_headers(df):
            st.warning("Active sheet has invalid headers. Please ensure it has the required columns.")
            return

        # Get conflict summary using cached function
        conflict_summary = get_conflict_summary(df, get_wb_nonce())
        
        # Mode toggle: Simple vs Advanced
        mode = st.segmented_control(
            "Mode", 
            options=["Simple", "Advanced"], 
            default="Simple", 
            key="__conflicts_mode"
        )
        
        if mode == "Simple":
            _render_simple_conflicts_navigator(conflict_summary, df, sheet)
        else:
            _render_advanced_conflicts(conflict_summary, df, sheet)

    except Exception as e:
        st.exception(e)


def _render_simple_conflicts_navigator(conflict_summary: Dict[str, Any], df: pd.DataFrame, sheet_name: str):
    """Render the Simple mode: Conflict Navigator with parent-first editing."""
    st.subheader("🔍 Conflict Navigator")
    
    summary = conflict_summary["summary"]
    treewide_mismatches = conflict_summary["treewide_mismatches"]
    
    # Build flat list of conflict items
    from collections import defaultdict
    conflict_items = []
    
    # 3A: parents not exact5 or >5
    for (L, parent_path), info in summary.items():
        if L == 1:
            # L1: single parent (ROOT). If count != 5 → conflict
            if info.get("count", 0) != MAX_CHILDREN_PER_PARENT:
                conflict_items.append({
                    "level": 1,
                    "parent_label": ROOT_PARENT_LABEL,
                    "parent_path": parent_path,  # "<ROOT>" path string
                    "children": info["children"],
                    "reason": f"VM (Root) needs exactly {MAX_CHILDREN_PER_PARENT}, has {info.get('count', 0)}"
                })
            continue
        if info.get("count", 0) != MAX_CHILDREN_PER_PARENT:
            parent_label = parent_path.split(">")[-1]
            conflict_items.append({
                "level": L,
                "parent_label": parent_label,
                "parent_path": parent_path,
                "children": info["children"],
                "reason": f"Parent has {info.get('count', 0)} children (needs {MAX_CHILDREN_PER_PARENT})"
            })

    # 3B: treewide label mismatches (same label across all levels, different 5-sets or >5)
    for label, block in treewide_mismatches.items():
        if (not block["all_exact5_same"]) or block["has_over5"]:
            for v in block["variants"]:
                conflict_items.append({
                    "level": v["level"],
                    "parent_label": label,
                    "parent_path": v["parent_path"],
                    "children": v["children"],
                    "reason": ("Treewide label mismatch" + (" + over5 present" if block["has_over5"] else ""))
                })

    # de-dup by (level, parent_path)
    seen = set()
    filtered = []
    for c in conflict_items:
        key = (c["level"], c["parent_path"])
        if key in seen:
            continue
        seen.add(key)
        filtered.append(c)
    conflict_items = filtered
    
    # Navigator state + controls
    st.caption(f"Conflicts found: {len(conflict_items)}")
    if "conflict_idx" not in st.session_state:
        st.session_state["conflict_idx"] = 0
    
    if conflict_items:
        idx = st.session_state["conflict_idx"]
        idx = max(0, min(idx, len(conflict_items)-1))
        st.session_state["conflict_idx"] = idx
        cur = conflict_items[idx]

        colA, colB, colC, colD = st.columns([1,1,1,3])
        if colA.button("⟵ Previous", disabled=(idx==0)):
            st.session_state["conflict_idx"] = idx-1
            safe_rerun()
        if colB.button("Resolve & Next", type="primary"):
            # will set a flag after successful apply
            pass
        if colC.button("Skip ➝", disabled=(idx==len(conflict_items)-1)):
            st.session_state["conflict_idx"] = idx+1
            safe_rerun()
        with colD:
            st.caption(f"Conflict {idx+1} / {len(conflict_items)} — {LEVEL_LABELS[cur['level']]} • Parent label: **{cur['parent_label']}**")
            st.caption(f"Parent path: `{cur['parent_path']}`")
            st.info(cur["reason"])
        
        # Parent-first editor (monolith-style)
        st.markdown("---")
        _render_parent_first_editor(cur, treewide_mismatches, df, sheet_name)
        
    else:
        st.success("No conflicts 🎉")
        st.stop()


def _render_parent_first_editor(cur: Dict[str, Any], treewide_mismatches: Dict, df: pd.DataFrame, sheet_name: str):
    """Render the parent-first editor for a single conflict."""
    st.subheader("✏️ Edit Children")
    
    # Show all combinations across the entire tree for this label
    block = treewide_mismatches.get(cur["parent_label"], {"variants": []})
    if block["variants"]:
        st.markdown(f"**All combinations across the entire tree for label '{cur['parent_label']}':**")
        
        # Build combinations table
        combinations_data = []
        for v in block["variants"]:
            combinations_data.append({
                "Level": LEVEL_LABELS[v["level"]],
                "parent_path": v["parent_path"],
                "children (unique)": ", ".join(v["children"]),
                "unique_count": len(v["children"]),
            })
        
        st.dataframe(
            combinations_data,
            use_container_width=True
        )
    
    # union from label group to offer as options
    union_opts = sorted({c for v in block["variants"] for c in v["children"]} | set(cur["children"]))
    default_set = cur["children"][:MAX_CHILDREN_PER_PARENT]

    # Build stable key seed
    seed = f"conflicts_simple_{cur['level']}_{cur['parent_label']}_{cur['parent_path']}_{get_wb_nonce()}"
    
    chosen = st.multiselect(
        f"Choose up to {MAX_CHILDREN_PER_PARENT} children", 
        options=union_opts, 
        default=default_set, 
        max_selections=MAX_CHILDREN_PER_PARENT,
        key=f"{seed}_ms_children"
    )
    
    new_child = normalize_text(st.text_input("Add new child", key=f"{seed}_ti_add"))
    if new_child and new_child not in chosen:
        chosen = normalize_child_set(chosen + [new_child])
        st.info(f"Preview after add: {', '.join(chosen)}")

    c1, c2, c3 = st.columns(3)
    
    with c1:
        if st.button("Apply to THIS parent", key=f"{seed}_btn_single"):
            _apply_to_single_parent(cur["level"], cur["parent_path"], chosen, df, sheet_name)
    
    with c2:
        if st.button(f"Apply to ALL '{cur['parent_label']}' parents across the tree", key=f"{seed}_btn_group"):
            _apply_to_label_across_tree(cur["parent_label"], chosen, df, sheet_name, treewide_mismatches)
    
    with c3:
        if st.button("Keep this parent as-is (Mark resolved)", key=f"{seed}_btn_skip"):
            st.session_state["conflict_idx"] = min(st.session_state["conflict_idx"]+1, len(st.session_state.get("conflict_items", []))-1)
            safe_rerun()


def _apply_to_single_parent(level: int, parent_path: str, children: List[str], df: pd.DataFrame, sheet_name: str):
    """Apply children to a single parent path."""
    try:
        with st.spinner(f"Applying to single parent at level {level}..."):
            wb = get_active_workbook() or {}
            if not wb or sheet_name not in wb:
                st.error("No active sheet.")
                return
            
            df0 = wb[sheet_name]
            
            # Use the single-parent materializer
            new_df = materialize_children_for_single_parent(df0, level, parent_path, children)
            
            # Update workbook
            wb[sheet_name] = new_df
            set_active_workbook(wb, default_sheet=sheet_name, source="conflicts_single_parent")
            set_current_sheet(sheet_name)
            
            st.success(f"Applied to parent: {parent_path}")
            # Move to next conflict
            if "conflict_idx" in st.session_state:
                st.session_state["conflict_idx"] = min(st.session_state["conflict_idx"]+1, 999)  # Will be bounded by actual count
            safe_rerun()
            
    except Exception as e:
        st.error(f"Error applying to single parent: {e}")
        st.exception(e)


def _apply_to_label_group(level: int, parent_label: str, children: List[str], df: pd.DataFrame, sheet_name: str):
    """Apply children to all parents with the same label at the given level."""
    try:
        with st.spinner(f"Applying to all '{parent_label}' parents at level {level}..."):
            wb = get_active_workbook() or {}
            if not wb or sheet_name not in wb:
                st.error("No active sheet.")
                return
            
            df0 = wb[sheet_name]
            
            # Use the existing label-group materializer
            new_df = materialize_children_for_label_group(df0, level, parent_label, children)
            
            # Update workbook
            wb[sheet_name] = new_df
            set_active_workbook(wb, default_sheet=sheet_name, source="conflicts_label_group")
            set_current_sheet(sheet_name)
            
            st.success(f"Applied to label-wide group: {parent_label}")
            # Move to next conflict
            if "conflict_idx" in st.session_state:
                st.session_state["conflict_idx"] = min(st.session_state["conflict_idx"]+1, 999)  # Will be bounded by actual count
            safe_rerun()
            
    except Exception as e:
        st.error(f"Error applying to label group: {e}")
        st.exception(e)


def _apply_to_label_across_tree(label: str, children: List[str], df: pd.DataFrame, sheet_name: str, treewide_mismatches: Dict):
    """Apply children to all parents with the same label across the entire tree."""
    try:
        with st.spinner(f"Applying to all '{label}' parents across the tree..."):
            wb = get_active_workbook() or {}
            if not wb or sheet_name not in wb:
                st.error("No active sheet.")
                return
            
            df0 = wb[sheet_name]
            
            # Get the summary for the materializer
            res = get_conflict_summary_with_root(df0, get_wb_nonce())
            summary = res["summary"]
            
            # Use the across-tree materializer
            new_df = materialize_children_for_label_across_tree(df0, label, children, summary)
            
            # Update workbook
            wb[sheet_name] = new_df
            set_active_workbook(wb, default_sheet=sheet_name, source="conflicts_across_tree")
            set_current_sheet(sheet_name)
            
            st.success(f"Applied to all '{label}' parents across the tree")
            # Move to next conflict
            if "conflict_idx" in st.session_state:
                st.session_state["conflict_idx"] = min(st.session_state["conflict_idx"]+1, 999)  # Will be bounded by actual count
            safe_rerun()
            
    except Exception as e:
        st.error(f"Error applying across tree: {e}")
        st.exception(e)


def _render_advanced_conflicts(conflict_summary: Dict[str, Any], df: pd.DataFrame, sheet_name: str):
    """Render the Advanced mode: existing analytics tables and tools."""
    st.subheader("🔬 Advanced Conflicts Analysis")
    
    # Display KPIs
    _render_conflict_kpis(conflict_summary)
    
    # Root children set
    st.markdown("---")
    _render_root_children(conflict_summary)
    
    # Full-path duplicates
    if conflict_summary.get("duplicates_full_path"):
        st.markdown("---")
        _render_full_path_duplicates(conflict_summary)
    
    # Level drilldown
    st.markdown("---")
    _render_level_drilldown(conflict_summary, df, sheet_name)
    
    # Override management
    st.markdown("---")
    overrides_all = st.session_state.get("branch_overrides", {})
    overrides_sheet = overrides_all.get(sheet_name, {})
    _render_override_management(df, overrides_sheet, sheet_name)
    
    # Resolve children tool
    _render_resolve_children_tool(df, sheet_name)


def _render_level1_root_editor(df: pd.DataFrame, sheet_name: str):
    """Render the Level 1 (ROOT) editor for Node-1 options."""
    st.subheader("🌱 Level 1 (ROOT) Children Editor")
    st.markdown("Manage the top-level Node-1 options that serve as the root of your decision tree.")
    
    # Get the store to see observed Node-1 options
    store = infer_branch_options(df)
    observed = store.get(f"L1|{ROOT_PARENT_LABEL}", [])
    
    if not observed:
        st.warning("No Node-1 options found. Check headers/data.")
        return
    
    st.write(f"**Observed Node-1 options ({len(observed)}):** {', '.join(observed)}")
    
    # Custom selector (limit to MAX_CHILDREN_PER_PARENT)
    chosen = st.multiselect(
        f"Custom set (choose up to {MAX_CHILDREN_PER_PARENT})", 
        options=observed, 
        default=observed[:MAX_CHILDREN_PER_PARENT], 
        max_selections=MAX_CHILDREN_PER_PARENT,
        key="__root_custom"
    )
    
    # Add new option
    new_opt = normalize_text(st.text_input("Add new Node-1 option"))
    if new_opt and new_opt not in chosen:
        chosen = normalize_child_set(chosen + [new_opt])
        st.info(f"**Preview after add:** {', '.join(chosen)}")
    
    # Action buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("✅ Apply: Keep observed (for all)", type="primary"):
            _apply_root_children(observed, sheet_name)
    with col2:
        if st.button("🔧 Apply: Use custom set (for all)", type="primary"):
            _apply_root_children(chosen, sheet_name)


def _apply_root_children(children: list[str], sheet_name: str):
    """Apply the new Level 1 children set to the entire sheet."""
    try:
        with st.spinner("Applying Level 1 changes..."):
            # Get current workbook and DataFrame
            wb = get_active_workbook()
            if not wb or sheet_name not in wb:
                st.error("No active sheet.")
                return
            
            df = wb[sheet_name]
            
            # Apply the new children set using logic.tree.set_level1_children
            new_df = set_level1_children(df, children)
            
            # Update the workbook with the new DataFrame
            wb[sheet_name] = new_df
            
            # Refresh the active workbook state
            set_active_workbook(wb, default_sheet=sheet_name, source="conflicts_level1_editor")
            
            # Bump nonce via re-setting current sheet
            set_current_sheet(sheet_name)
            
            st.success(f"✅ Updated Level-1 (ROOT) children to: {', '.join(children)}")
            st.info("The sheet has been updated. Other tabs will reflect these changes.")
            
            # Rerun to refresh the UI
            safe_rerun()
            
    except Exception as e:
        st.error(f"Error applying Level 1 changes: {str(e)}")
        st.exception(e)


def _render_conflict_kpis(conflict_summary: Dict[str, Any]):
    """Render conflict KPIs dashboard."""
    st.subheader("📊 Conflict Metrics")
    
    counts = conflict_summary["counts"]
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="Parents >5 Children",
            value=counts["parents_over5"],
            help="Number of parent nodes with more than 5 children"
        )
    
    with col2:
        st.metric(
            label="Parents Not Exactly 5",
            value=counts["parents_not_exact5"],
            help="Number of parent nodes that don't have exactly 5 children"
        )
    
    with col3:
        st.metric(
            label="Total Parents",
            value=counts["total_parents"],
            help="Total number of parent nodes in the decision tree"
        )
    
    # Show mismatches summary
    mismatches = conflict_summary["mismatches"]
    if mismatches:
        st.warning(f"⚠️ Found {len(mismatches)} parent label groups with inconsistent child sets")
        
        # Group by level for better display
        level_mismatches = {}
        for (level, parent_label), info in mismatches.items():
            if level not in level_mismatches:
                level_mismatches[level] = []
            level_mismatches[level].append((parent_label, info))
        
        for level in sorted(level_mismatches.keys()):
            level_info = level_mismatches[level]
            with st.expander(f"Level {level} Mismatches ({len(level_info)})"):
                for parent_label, info in level_info:
                    st.write(f"**{parent_label}**: {len(info['variants'])} variants")
                    if info['has_over5']:
                        st.error(f"  - Some variants have >5 children")
                    if not info['all_exact5_same']:
                        st.warning(f"  - Child sets differ between variants")
    else:
        st.success("✅ All parent labels have consistent child sets!")


def _render_root_children(conflict_summary: Dict[str, Any]):
    """Render the root children set information."""
    st.subheader("🌱 Root Children (Level 1)")
    
    root_children = conflict_summary.get("root_children", [])
    
    if root_children:
        count = len(root_children)
        if count == MAX_CHILDREN_PER_PARENT:
            st.success(f"✅ Root has exactly {MAX_CHILDREN_PER_PARENT} children: {', '.join(root_children)}")
        elif count > MAX_CHILDREN_PER_PARENT:
            st.error(f"⚠️ Root has {count} children (>5): {', '.join(root_children)}")
        else:
            st.warning(f"⚠️ Root has {count} children (<5): {', '.join(root_children)}")
    else:
        st.warning("⚠️ No root children found")


def _render_full_path_duplicates(conflict_summary: Dict[str, Any]):
    """Render full-path duplicates information."""
    st.subheader("🔄 Full-Path Duplicates")
    
    duplicates = conflict_summary.get("duplicates_full_path", [])
    
    if duplicates:
        st.warning(f"⚠️ Found {len(duplicates)} full paths that occur multiple times:")
        
        # Create a DataFrame for better display
        dup_data = []
        for path, count in duplicates:
            dup_data.append({
                "Full Path": path,
                "Occurrences": count
            })
        
        dup_df = pd.DataFrame(dup_data)
        st.dataframe(dup_df, use_container_width=True)
        
        st.info("""
        **Note:** Full-path duplicates are expected in decision trees where different 
        diagnostic paths share common prefixes but diverge at deeper levels.
        """)
    else:
        st.success("✅ No full-path duplicates found")


def _render_level_drilldown(conflict_summary: Dict[str, Any], df: pd.DataFrame, sheet_name: str):
    """Render level drilldown for conflict resolution."""
    st.subheader("🔍 Level Drilldown")
    
    # Level picker
    level = st.selectbox(
        "Select Level",
        options=list(range(1, 6)),
        format_func=lambda x: f"Level {x}",
        help="Choose which level to analyze for conflicts"
    )
    
    if level == 1:
        # Level 1 is special - always <ROOT>
        parent_label = ROOT_PARENT_LABEL
        _render_level1_resolution(conflict_summary, df, sheet_name)
    else:
        # For levels 2+, show parent label picker
        mismatches = conflict_summary["mismatches"]
        level_mismatches = [(label, info) for (l, label), info in mismatches.items() if l == level]
        
        if not level_mismatches:
            st.info(f"No conflicts found at Level {level}")
            return
        
        parent_label = st.selectbox(
            f"Parent Label at Level {level}",
            options=[label for label, _ in level_mismatches],
            help=f"Choose which parent label to resolve conflicts for at Level {level}"
        )
        
        # Find the specific mismatch info
        mismatch_info = None
        for (l, label), info in mismatches.items():
            if l == level and label == parent_label:
                mismatch_info = info
                break
        
        if mismatch_info:
            _render_level_resolution(level, parent_label, mismatch_info, df, sheet_name)


def _render_level1_resolution(conflict_summary: Dict[str, Any], df: pd.DataFrame, sheet_name: str):
    """Render Level 1 (ROOT) conflict resolution."""
    st.write(f"**Level 1: {ROOT_PARENT_LABEL}**")
    
    # Get current Node 1 options
    summary = conflict_summary["summary"]
    root_info = summary.get((1, ROOT_PARENT_LABEL))
    
    if not root_info:
        st.warning("No Level 1 data found")
        return
    
    children = root_info["children"]
    count = root_info["count"]
    
    st.write(f"**Current children ({count}):** {', '.join(children)}")
    
    if count > MAX_CHILDREN_PER_PARENT:
        st.error(f"⚠️ Too many children: {count} > {MAX_CHILDREN_PER_PARENT}")
    elif count < MAX_CHILDREN_PER_PARENT:
        st.warning(f"⚠️ Not enough children: {count} < {MAX_CHILDREN_PER_PARENT}")
    else:
        st.success(f"✅ Exactly {MAX_CHILDREN_PER_PARENT} children")
    
    # Resolution options
    st.subheader("🔧 Resolution Options")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("✅ Keep Current Set", type="primary"):
            _apply_level1_resolution(children, sheet_name)
    
    with col2:
        # Custom set editor
        custom_children = st.multiselect(
            "Custom Set (≤5)",
            options=children,
            default=children[:MAX_CHILDREN_PER_PARENT],
            max_selections=MAX_CHILDREN_PER_PARENT,
            key="__level1_custom"
        )
        
        if st.button("🔧 Apply Custom Set"):
            _apply_level1_resolution(custom_children, sheet_name)


def _render_level_resolution(level: int, parent_label: str, mismatch_info: Dict[str, Any], 
                           df: pd.DataFrame, sheet_name: str):
    """Render conflict resolution for levels 2+."""
    st.write(f"**Level {level}: {parent_label}**")
    
    variants = mismatch_info["variants"]
    has_over5 = mismatch_info["has_over5"]
    all_exact5_same = mismatch_info["all_exact5_same"]
    
    # Show current state
    if has_over5:
        st.error(f"⚠️ Some variants have >5 children")
    if not all_exact5_same:
        st.warning(f"⚠️ Child sets differ between variants ({len(variants)} variants)")
    
    # Show variants table
    st.write("**Current Variants:**")
    variant_data = []
    for variant in variants:
        parent_path = variant["parent_path"]
        children = variant["children"]
        count = len(children)
        variant_data.append({
            "Parent Path": parent_path,
            "Children": ", ".join(children),
            "Count": count,
            "Status": "⚠️ Over 5" if count > MAX_CHILDREN_PER_PARENT else "✅ OK"
        })
    
    variant_df = pd.DataFrame(variant_data)
    st.dataframe(variant_df, use_container_width=True)
    
    # Resolution options
    st.subheader("🔧 Resolution Options")
    
    # Option 1: Keep one variant as canonical
    st.write("**Option 1: Keep Variant as Canonical**")
    if variants:
        canonical_variant = st.selectbox(
            "Choose variant to keep",
            options=variants,
            format_func=lambda v: f"{v['parent_path']} → {len(v['children'])} children",
            key=f"__canonical_variant_{level}_{parent_label}"
        )
        
        if st.button(f"✅ Apply: Keep {canonical_variant['parent_path']} as Canonical"):
            children = canonical_variant['children'][:MAX_CHILDREN_PER_PARENT]  # Cap to 5
            _apply_canonical_set_for_label_group(level, parent_label, children, sheet_name)
    
    # Option 2: Custom canonical set
    st.write("**Option 2: Custom Canonical Set (≤5)**")
    
    # Get all unique children from all variants
    all_children = set()
    for variant in variants:
        all_children.update(variant["children"])
    all_children = sorted(list(all_children))
    
    # Suggest majority-vote canonical set
    variant_children = [variant["children"] for variant in variants]
    suggested_children = majority_vote_5set(variant_children)
    
    if suggested_children:
        st.info(f"💡 **Suggested canonical set (majority vote):** {', '.join(suggested_children)}")
    
    custom_children = st.multiselect(
        f"Select children (≤{MAX_CHILDREN_PER_PARENT})",
        options=all_children,
        default=suggested_children,  # Pre-fill with majority vote suggestion
        max_selections=MAX_CHILDREN_PER_PARENT,
        key=f"__custom_children_{level}_{parent_label}"
    )
    
    # Add new option
    new_child = st.text_input("Add new child option", key=f"__new_child_{level}_{parent_label}")
    if new_child and new_child not in custom_children:
        if len(custom_children) < MAX_CHILDREN_PER_PARENT:
            custom_children.append(new_child)
            st.info(f"**Preview:** {', '.join(custom_children)}")
        else:
            st.warning(f"Cannot add more than {MAX_CHILDREN_PER_PARENT} children")
    
    if st.button("🔧 Apply Custom Set"):
        _apply_canonical_set_for_label_group(level, parent_label, custom_children, sheet_name)


def _apply_level1_resolution(children: List[str], sheet_name: str):
    """Apply Level 1 resolution using existing set_level1_children function."""
    try:
        with st.spinner("Applying Level 1 resolution..."):
            # Get current workbook and DataFrame
            wb = get_active_workbook()
            if not wb or sheet_name not in wb:
                st.error("No active sheet.")
                return
            
            df = wb[sheet_name]
            
            # Apply the new children set using logic.tree.set_level1_children
            new_df = set_level1_children(df, children)
            
            # Update the workbook with the new DataFrame
            wb[sheet_name] = new_df
            
            # Refresh the active workbook state
            set_active_workbook(wb, default_sheet=sheet_name, source="conflicts_level1_resolution")
            
            # Bump nonce via re-setting current sheet
            set_current_sheet(sheet_name)
            
            st.success(f"✅ Updated Level-1 (ROOT) children to: {', '.join(children)}")
            st.info("The sheet has been updated. Other tabs will reflect these changes.")
            
            # Rerun to refresh the UI
            safe_rerun()
            
    except Exception as e:
        st.error(f"Error applying Level 1 resolution: {str(e)}")
        st.exception(e)


def _apply_level_resolution(level: int, parent_label: str, children: List[str], 
                          sheet_name: str, action_type: str):
    """Apply level resolution using the override system."""
    try:
        with st.spinner(f"Applying {action_type} resolution..."):
            # Get current overrides
            overrides_all = st.session_state.get("branch_overrides", {})
            if sheet_name not in overrides_all:
                overrides_all[sheet_name] = {}
            
            # Find all parent paths that match this parent label at this level
            wb = get_active_workbook()
            if not wb or sheet_name not in wb:
                st.error("No active sheet.")
                return
            
            df = wb[sheet_name]
            
            # Get all parent paths that end with this label at this level
            affected_paths = []
            for i, row in df.iterrows():
                if level > 1:
                    # Build parent path for this row
                    parent_path = []
                    for j in range(1, level):
                        col = f"Node {j}"
                        if col in df.columns:
                            value = normalize_text(row.get(col, ""))
                            if value:
                                parent_path.append(value)
                    
                    # Check if this path ends with our target label
                    if parent_path and parent_path[-1] == parent_label:
                        affected_paths.append(tuple(parent_path))
            
            # Apply overrides for all affected paths
            for parent_path in affected_paths:
                override_key = (level, parent_path)
                overrides_all[sheet_name][override_key] = children
            
            st.session_state["branch_overrides"] = overrides_all
            
            # Apply the overrides using logic.tree.build_raw_plus_v630
            updated_df = build_raw_plus_v630(df, overrides_all[sheet_name])
            
            # Update the active workbook
            wb[sheet_name] = updated_df
            set_active_workbook(wb, source="conflicts_level_resolution")
            
            # Clear stale caches to ensure immediate refresh
            st.cache_data.clear()
            
            # Show success message
            st.success(f"✅ {action_type} resolution applied successfully!")
            st.info(f"**Updated {len(affected_paths)} parent paths** to have children: {', '.join(children)}")
            
            # Rerun to show updated state
            safe_rerun()
            
    except Exception as e:
        st.error(f"Error applying resolution: {e}")
        st.exception(e)


def _render_override_management(df: pd.DataFrame, overrides_sheet: Dict, sheet_name: str):
    """Render the override management section."""
    st.subheader("🎛️ Override Management")
    
    if not overrides_sheet:
        st.info("No overrides defined for this sheet.")
        return
    
    st.write(f"Current overrides for '{sheet_name}':")
    
    # Display current overrides
    for override_key, override_values in overrides_sheet.items():
        if isinstance(override_key, tuple) and len(override_key) >= 2:
            level, parent_path = override_key[0], override_key[1:]
            
            with st.expander(f"Level {level}: {' > '.join(parent_path) if parent_path else 'Root'}", expanded=False):
                st.write(f"**Override values:** {override_values}")
                
                # Show edit controls
                col1, col2 = st.columns([3, 1])
                with col1:
                    new_values = st.text_input(
                        "New values (comma-separated)",
                        value=", ".join(override_values),
                        key=f"edit_override_{level}_{hash(str(parent_path))}"
                    )
                
                with col2:
                    if st.button("Update", key=f"update_override_{level}_{hash(str(parent_path))}"):
                        if new_values.strip():
                            new_list = [v.strip() for v in new_values.split(",") if v.strip()]
                            # Update the override
                            overrides_all = st.session_state.get("branch_overrides", {})
                            overrides_all[sheet_name][override_key] = new_list
                            st.session_state["branch_overrides"] = overrides_all
                            st.success("Override updated!")
                            safe_rerun()
                
                # Delete option
                if st.button("🗑️ Delete", key=f"delete_override_{level}_{hash(str(parent_path))}"):
                    overrides_all = st.session_state.get("branch_overrides", {})
                    if override_key in overrides_all[sheet_name]:
                        del overrides_all[sheet_name][override_key]
                        st.session_state["branch_overrides"] = overrides_all
                        st.success("Override deleted!")
                        safe_rerun()
    
    # Add new override
    st.markdown("---")
    st.subheader("➕ Add New Override")
    
    col1, col2 = st.columns(2)
    with col1:
        new_level = st.number_input("Level", min_value=1, max_value=5, value=1)
        new_parent = st.text_input("Parent path (comma-separated, leave empty for root)")
    
    with col2:
        new_values = st.text_input("Values (comma-separated)")
        if st.button("Add Override"):
            if new_values.strip():
                parent_path = tuple()
                if new_parent.strip():
                    parent_path = tuple(v.strip() for v in new_parent.split(",") if v.strip())
                
                override_key = (new_level, parent_path)
                override_values = [v.strip() for v in new_values.split(",") if v.strip()]
                
                # Add to overrides
                overrides_all = st.session_state.get("branch_overrides", {})
                if sheet_name not in overrides_all:
                    overrides_all[sheet_name] = {}
                overrides_all[sheet_name][override_key] = override_values
                st.session_state["branch_overrides"] = overrides_all
                
                st.success("Override added!")
                safe_rerun()


def _render_resolve_children_tool(df: pd.DataFrame, sheet_name: str):
    """Render the resolve children across sheet tool."""
    st.markdown("---")
    st.subheader("🔧 Resolve Children Across Sheet")
    st.markdown("Standardize child sets for all parents with the same label at a specific level.")
    
    # Level selection
    level = st.selectbox("Node level", [1, 2, 3, 4, 5], key="resolve_level")
    
    # Get parent labels for this level
    parent_labels = _get_parent_labels_at_level(df, level, "resolve_children_tool")
    
    if not parent_labels:
        st.info(f"No parent labels found at level {level}.")
        return
    
    # Parent label selection
    if level == 1:
        # For Level 1, show the special ROOT editor
        _render_level1_root_editor(df, sheet_name)
        return
    else:
        parent_label = st.selectbox("Parent label", parent_labels, key="resolve_parent_label")
        # Find the parent path for this label
        parent_path = _find_parent_path_for_label(df, level, parent_label, "resolve_children_tool")
    
    if not parent_label or (level > 1 and not parent_path):
        st.info("Please select a valid parent label.")
        return
    
    # Analyze current children for this parent label
    current_children_variants = _analyze_children_variants(df, level, parent_path, parent_label, "resolve_children_tool")
    
    if not current_children_variants:
        st.info(f"No children found for parent label '{parent_label}' at level {level}.")
        return
    
    st.write(f"**Current children variants for '{parent_label}' at level {level}:**")
    
    # Display variants
    for i, (children_set, count) in enumerate(current_children_variants):
        st.write(f"**Variant {i+1}** ({count} parents): {', '.join(children_set)}")
    
    # Action selection
    st.markdown("---")
    action = st.radio(
        "Choose action:",
        ["Keep Set For All", "Custom Set For All"],
        key="resolve_action"
    )
    
    if action == "Keep Set For All":
        # Keep existing variant
        if len(current_children_variants) == 1:
            st.info("Only one variant exists - no action needed.")
            return
        
        variant_choice = st.selectbox(
            "Select variant to keep:",
            [f"Variant {i+1}: {', '.join(children)} ({count} parents)" 
             for i, (children, count) in enumerate(current_children_variants)],
            key="variant_choice"
        )
        
        if st.button("Apply Keep Set For All", type="primary"):
            variant_index = int(variant_choice.split(":")[0].split()[1]) - 1
            selected_children = current_children_variants[variant_index][0]
            _apply_children_resolution(df, level, parent_path, parent_label, selected_children, sheet_name, "keep")
    
    else:  # Custom Set For All
        # Get union of all observed children
        all_children = set()
        for children_set, _ in current_children_variants:
            all_children.update(children_set)
        
        st.write(f"**Union of observed children:** {', '.join(sorted(all_children))}")
        
        # Multi-select up to 5 children
        selected_children = st.multiselect(
            f"Select up to {MAX_CHILDREN_PER_PARENT} children:",
            sorted(all_children),
            max_selections=MAX_CHILDREN_PER_PARENT,
            key="custom_children"
        )
        
        # Free-text input for new children
        new_child = st.text_input("Add new child (optional):", key="new_child_input")
        if new_child.strip():
            new_child_clean = normalize_text(new_child)
            if new_child_clean and new_child_clean not in selected_children:
                selected_children.append(new_child_clean)
        
        if len(selected_children) > 5:
            selected_children = selected_children[:5]
        
        if selected_children:
            st.write(f"**Final children set:** {', '.join(selected_children)}")
            
            if st.button("Apply Custom Set For All", type="primary"):
                _apply_children_resolution(df, level, parent_path, parent_label, selected_children, sheet_name, "custom")


@st.cache_data(ttl=600)
def _get_parent_labels_at_level(df: pd.DataFrame, level: int, nonce: str) -> List[str]:
    """Get parent labels at a specific level."""
    try:
        if level <= 1:
            return []
        
        parent_col = f"Node {level - 1}"
        if parent_col not in df.columns:
            return []
        
        values = df[parent_col].map(normalize_text).dropna()
        values = values[values != ""]
        return sorted(values.unique())
        
    except Exception:
        return []


@st.cache_data(ttl=600)
def _find_parent_path_for_label(df: pd.DataFrame, level: int, label: str, nonce: str) -> Optional[Tuple[str, ...]]:
    """Find the parent path for a specific label at a level."""
    try:
        if level <= 1:
            return tuple()
        
        parent_col = f"Node {level - 1}"
        if parent_col not in df.columns:
            return None
        
        # Find rows with this label
        mask = df[parent_col].map(normalize_text) == label
        matching_rows = df[mask]
        
        if matching_rows.empty:
            return None
        
        # Get the parent path from the first matching row
        parent_cols = [f"Node {i}" for i in range(1, level - 1)]
        if not parent_cols:
            return tuple()
        
        parent_path = []
        for col in parent_cols:
            if col in df.columns:
                value = normalize_text(matching_rows.iloc[0].get(col, ""))
                if value:
                    parent_path.append(value)
        
        return tuple(parent_path)
        
    except Exception:
        return None


@st.cache_data(ttl=600)
def _analyze_children_variants(df: pd.DataFrame, level: int, parent_path: Tuple[str, ...], parent_label: str, nonce: str) -> List[Tuple[List[str], int]]:
    """Analyze current children variants for a parent label."""
    try:
        if level > MAX_LEVELS:
            return []
        
        node_col = f"Node {level}"
        if node_col not in df.columns:
            return []
        
        # Find rows matching the parent path
        matching_rows = df.copy()
        for i, expected_value in enumerate(parent_path):
            col = f"Node {i + 1}"
            if col in df.columns:
                mask = df[col].map(normalize_text) == expected_value
                matching_rows = matching_rows[mask]
        
        if matching_rows.empty:
            return []
        
        # Get children at this level
        children_values = matching_rows[node_col].map(normalize_text).dropna()
        children_values = children_values[children_values != ""]
        
        if children_values.empty:
            return []
        
        # Group by unique children sets
        children_variants = {}
        for _, row in matching_rows.iterrows():
            child_value = normalize_text(row.get(node_col, ""))
            if child_value:
                # Create a key for this row's parent path
                row_parent_path = []
                for i in range(1, level):
                    col = f"Node {i}"
                    if col in row.index:
                        value = normalize_text(row.get(col, ""))
                        if value:
                            row_parent_path.append(value)
                
                parent_key = tuple(row_parent_path)
                if parent_key not in children_variants:
                    children_variants[parent_key] = set()
                children_variants[parent_key].add(child_value)
        
        # Count variants
        variant_counts = {}
        for children_set in children_variants.values():
            children_tuple = tuple(sorted(children_set))
            variant_counts[children_tuple] = variant_counts.get(children_tuple, 0) + 1
        
        # Return sorted by count
        result = [(list(children), count) for children, count in variant_counts.items()]
        result.sort(key=lambda x: x[1], reverse=True)
        
        return result
        
    except Exception:
        return []


def _apply_children_resolution(df: pd.DataFrame, level: int, parent_path: Tuple[str, ...], parent_label: str, 
                              new_children: List[str], sheet_name: str, action_type: str):
    """Apply children resolution across the sheet."""
    try:
        with st.spinner(f"Applying {action_type} resolution..."):
            # Create override key
            override_key = (level, parent_path)
            
            # Get current overrides
            overrides_all = st.session_state.get("branch_overrides", {})
            if sheet_name not in overrides_all:
                overrides_all[sheet_name] = {}
            
            # Set the override
            overrides_all[sheet_name][override_key] = new_children
            st.session_state["branch_overrides"] = overrides_all
            
            # Apply the override using logic.tree
            # from logic.tree import build_raw_plus_v630 # This import is now at the top
            
            # Get the active workbook
            # from utils.state import get_active_workbook, set_active_workbook # This import is now at the top
            active_wb = get_active_workbook()
            
            if active_wb and sheet_name in active_wb:
                # Apply overrides and rebuild the sheet
                updated_df = build_raw_plus_v630(df, overrides_all[sheet_name])
                
                # Update the active workbook
                active_wb[sheet_name] = updated_df
                set_active_workbook(active_wb, source="conflicts_resolution")
                
                # Clear stale caches to ensure immediate refresh
                st.cache_data.clear()
                
                # Show delta preview
                rows_affected = len(updated_df) - len(df)
                st.success(f"✅ {action_type} resolution applied successfully!")
                st.info(f"**Delta:** {rows_affected:+d} rows affected. Sheet updated with new children set: {', '.join(new_children)}")
                
                # Rerun to show updated state
                safe_rerun()
            else:
                st.error("Could not update active workbook.")
                
    except Exception as e:
        st.error(f"Error applying resolution: {e}")
        st.exception(e)


def _apply_canonical_set_for_label_group(level: int, parent_label: str, children: List[str], sheet_name: str):
    """
    Apply canonical children set to all parent paths at a given (level, parent_label),
    using monolith row-multiplication semantics.
    """
    try:
        with st.spinner(f"Applying canonical set to level {level} / label '{parent_label}'..."):
            wb = get_active_workbook()
            if not wb or sheet_name not in wb:
                st.error("No active sheet.")
                return
            
            df = wb[sheet_name]
            
            # Apply materialization using the monolith rule
            new_df = materialize_children_for_label_group(df, level, parent_label, children)
            
            # Update the workbook with the new DataFrame
            wb[sheet_name] = new_df
            
            # Refresh the active workbook state
            set_active_workbook(wb, default_sheet=sheet_name, source="conflicts_materialization")
            
            # Bump nonce via re-setting current sheet
            set_current_sheet(sheet_name)
            
            # Clear stale caches to ensure immediate refresh
            st.cache_data.clear()
            
            # Show success message
            st.success(f"✅ Applied canonical set to level {level} / label '{parent_label}'")
            st.info(f"**Updated {len(children)} children** using monolith row-multiplication rule")
            
            # Rerun to show updated state
            safe_rerun()
            
    except Exception as e:
        st.error(f"Error applying canonical set: {e}")
        st.exception(e)
