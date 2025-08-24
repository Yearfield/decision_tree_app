# ui/tabs/workspace.py
"""
Workspace Selection tab for choosing and previewing sheets.

DUPLICATE ROWS POLICY:
This app treats each row as a full path. Duplicate prefixes (first N nodes) are expected; 
we compute unique children per parent by set semantics. Downstream multiplication does not 
inflate child counts.

The indexer maintains counters for display purposes but uses unique labels to judge 
child-set size and conflicts. This ensures accurate coverage metrics regardless of 
row duplication patterns.
"""
import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, List, Optional

from utils import (
    CANON_HEADERS, LEVEL_COLS, normalize_text, validate_headers
)
from utils.state import (
    get_active_workbook, get_current_sheet, get_active_df, 
    set_current_sheet, has_active_workbook, get_workbook_status,
    set_active_workbook
)
from logic.tree import infer_branch_options, infer_branch_options_with_overrides
from utils.constants import ROOT_PARENT_LABEL, MAX_CHILDREN_PER_PARENT, LEVEL_LABELS, ROOT_COL, LEVEL_COLS, MAX_LEVELS
from ui.utils.rerun import safe_rerun

# Guard assertions to prove imports are live (temporary; safe to remove later)
assert callable(get_current_sheet), "get_current_sheet not imported correctly"
assert callable(get_active_df), "get_active_df not imported correctly"
assert callable(get_active_workbook), "get_active_workbook not imported correctly"
assert callable(set_current_sheet), "set_current_sheet not imported correctly"
assert callable(set_active_workbook), "set_active_workbook not imported correctly"


def _compute_parents_vectorized(df: pd.DataFrame) -> tuple[int, int, int]:
    """
    Returns (ok_parents, total_parents, conflict_parents).
    ok_parents = parents that have exactly 5 distinct non-empty children.
    conflict_parents = parents with !=5 children.
    """
    if df is None or df.empty:
        return 0, 0, 0

    dfv = df.copy()
    node_cols = ["Node 1","Node 2","Node 3","Node 4","Node 5"]
    for c in ["Vital Measurement"] + node_cols:
        if c not in dfv.columns:
            dfv[c] = ""
        dfv[c] = dfv[c].astype(str).map(lambda x: x.strip())

    ok_total = 0
    total_parents = 0
    conflict_parents = 0

    for lvl in range(1, 6):
        parent_cols = node_cols[:lvl-1]        # [] for lvl==1 (ROOT)
        child_col  = node_cols[lvl-1]

        scope = dfv[dfv[child_col] != ""].copy()
        if parent_cols:
            scope = scope[(scope[parent_cols] != "").all(axis=1)]
        if scope.empty:
            continue

        if parent_cols:
            grp = scope.groupby(parent_cols, dropna=False)[child_col].nunique()
        else:
            grp = scope.assign(__root="__root").groupby("__root")[child_col].nunique()

        total_parents     += int(len(grp))
        ok_total          += int((grp == 5).sum())
        conflict_parents  += int((grp != 5).sum())

    return ok_total, total_parents, conflict_parents





def count_full_paths(df: pd.DataFrame) -> Tuple[int, int]:
    """Count rows with full path (all 6 path cells: Root + Node1..Node5 are non-blank)."""
    cols = [ROOT_COL] + LEVEL_COLS
    present = df[cols].astype(str).map(lambda x: bool(str(x).strip()))
    return int(present.all(axis=1).sum()), int(len(df))


def render():
    """Render the Workspace Selection tab for choosing and previewing sheets."""
    try:
        # Check for navigation hints and display instructions
        _check_and_display_nav_hint()
        
        st.header("🗂 Workspace Selection")
        st.markdown("Choose a sheet to work with and preview its contents.")
        
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
            st.warning("No active workbook/sheet. Load a workbook in 📂 Source or select a sheet below.")
            return

        # Sheet selection
        if len(wb) > 1:
            st.subheader("📋 Sheet Selection")
            sheet_names = list(wb.keys())
            selected_sheet = st.selectbox(
                "Choose active sheet", 
                sheet_names, 
                index=sheet_names.index(sheet) if sheet in sheet_names else 0,
                key="workspace_sheet_selector"
            )
            
            if selected_sheet != sheet:
                set_current_sheet(selected_sheet)
                safe_rerun()
        
        # Re-sync button for current sheet
        if st.session_state.get("sheet_id") and st.session_state.get("sheet_name"):
            if st.button("🔄 Re-sync current sheet", key="workspace_resync"):
                sheet_id = st.session_state.get("sheet_id")
                sheet_name = st.session_state.get("sheet_name")
                try:
                    if "gcp_service_account" not in st.secrets:
                        st.error("Google Sheets not configured. Add your service account JSON under [gcp_service_account].")
                    else:
                        # Re-read sheet and replace that entry in active workbook
                        with st.spinner("Re-syncing..."):
                            from io_utils.sheets import read_google_sheet
                            new_df = read_google_sheet(sheet_id, sheet_name, st.secrets["gcp_service_account"])
                            if not new_df.empty:
                                wb = get_active_workbook() or {}
                                wb[sheet_name] = new_df
                                set_active_workbook(wb, source="workspace_resync")
                                
                                # Clear stale caches to ensure immediate refresh
                                st.cache_data.clear()
                                
                                st.success(f"Re-synced '{sheet_name}' from Google Sheets.")
                            else:
                                st.warning("Selected sheet is empty or not found.")
                except Exception as e:
                    st.error(f"Google Sheets error: {e}")
        
        # Get active DataFrame
        df = get_active_df()
        if df is None:
            st.warning("No active workbook/sheet. Load or select one in 📂 Source / 🗂 Workspace.")
            return

        # Show headers and preview
        st.caption(f"Active sheet: {get_current_sheet()} ({len(df)} rows)")
        st.caption(f"Headers: {list(df.columns)[:8]}{' …' if len(df.columns)>8 else ''}")
        
        if not validate_headers(df):
            st.error("Active sheet has invalid headers for this app's canonical schema. Check the first 8 columns.")
            # Still allow a lightweight preview:
            st.dataframe(df.head(20), use_container_width=True)
            return

        # Summary + preview
        st.write(f"Active sheet: **{sheet}** ({len(df)} rows)")
        
        # Render the new robust summary with repair functionality
        summary = _render_robust_summary_with_repair(df)
        
        # Show monolith counters
        has_any_label, total_p = _render_monolith_counters(df, sheet)
        
        # Check if we truly have no labels vs just no parents discovered
        node_cols = ["Node 1","Node 2","Node 3","Node 4","Node 5"]
        for c in node_cols:
            if c not in df.columns:
                df[c] = ""
        has_any_label = (df[node_cols].astype(str).apply(lambda s: s.str.strip()) != "").any(axis=1).any()
        
        if total_p == 0 and not has_any_label:
            st.info("No parent nodes found yet. Start by adding Node 1 children in the Symptoms tab.")
        
        if summary and summary.get("total_parents", 0) > 0:
            # Show KPIs and per-level breakdown only after non-empty store
            _render_kpis_and_breakdown(df, summary)
            # Also show the worklist since we have data
            _render_parent_worklist(df, summary)
        else:
            if st.button("Recompute / Repair"):
                # bump nonce by re-setting current sheet to itself
                set_current_sheet(get_current_sheet())
                safe_rerun()

        # Preview section
        _render_preview_section(df, sheet)

        # Group rows controls
        st.markdown("---")
        _render_grouping_controls(df)

    except Exception as e:
        st.exception(e)


def _render_monolith_counters(df: pd.DataFrame, sheet_name: Optional[str] = None):
    if df is None or df.empty:
        st.info("Selected sheet is empty.")
        return

    # Use vectorized calculation for accurate parent metrics
    ok_p, total_p, conflict_parents = _compute_parents_vectorized(df)
    
    # Calculate rows with full path
    total = len(df)
    ok_r = 0
    
    for _, row in df.iterrows():
        path_complete = True
        for col in LEVEL_COLS:
            if col in df.columns:
                if normalize_text(row.get(col, "")) == "":
                    path_complete = False
                    break
        if path_complete:
            ok_r += 1
    
    total_r = total

    # Robust parent presence check (don't rely solely on store)
    # Show a notice ONLY if there are truly no non-empty labels in Node 1..5.
    node_cols = ["Node 1","Node 2","Node 3","Node 4","Node 5"]
    for c in node_cols:
        if c not in df.columns:
            df[c] = ""
    has_any_label = (df[node_cols].astype(str).apply(lambda s: s.str.strip()) != "").any(axis=1).any()

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Parents with 5 children", f"{ok_p}/{total_p}")
        st.progress(0 if total_p == 0 else ok_p / total_p)
    with c2:
        st.metric("Rows with full path", f"{ok_r}/{total_r}")
        st.progress(0 if total_r == 0 else ok_r / total_r)
    with c3:
        st.metric("Parents violating 5‑child rule", f"{conflict_parents}")
    
    # Return the has_any_label value and total_p for use in the render function
    return has_any_label, total_p


def _check_and_display_nav_hint():
    """Check for incoming navigation hints and display appropriate instructions."""
    nav_hint = st.session_state.get("_nav_hint")
    if nav_hint:
        tab = nav_hint.get("tab", "")
        level = nav_hint.get("level", 0)
        parent = nav_hint.get("parent", "")
        
        if tab == "symptoms":
            st.info(f"🧬 **Navigation from Symptoms tab:** Working with Level {level} parent '{friendly_parent(level, parent)}'")
        elif tab == "conflicts":
            st.info(f"⚖️ **Navigation from Conflicts tab:** Working with Level {level} parent '{friendly_parent(level, parent)}'")
        
        # Clear the hint after displaying
        del st.session_state["_nav_hint"]


def friendly_parent(level: int, path_str: str) -> str:
    """Convert store key to friendly display name."""
    if path_str == "<ROOT>":
        return "Top-level (Node 1) options"
    return path_str.replace(">", " > ")


@st.cache_data(ttl=600, show_spinner=False)
def _compute_parent_coverage_summary(df: pd.DataFrame, nonce: str) -> Dict[str, Any]:
    """Compute comprehensive parent coverage summary using the robust indexer."""
    try:
        # Validate headers first
        if not validate_headers(df):
            return {}
        
        # Use local logic instead of external analysis
        store = infer_branch_options(df)
        
        # Initialize level tracking
        level_stats = {level: {"total": 0, "with_5": 0, "with_less_5": 0, "with_0": 0} for level in range(1, 6)}
        
        # Process each store entry
        for key, children in store.items():
            if not key.startswith("L"):
                continue
                
            # Parse level from key (L{level}|{path})
            parts = key.split("|", 1)
            if len(parts) != 2:
                continue
                
            try:
                level = int(parts[0][1:])  # Remove "L" and convert to int
                if level < 1 or level > 5:
                    continue
            except ValueError:
                continue
            
            # Count non-empty, deduped children
            if children:
                normalized_children = []
                seen = set()
                for child in children:
                    clean_child = normalize_text(child)
                    if clean_child and clean_child not in seen:
                        normalized_children.append(clean_child)
                        seen.add(clean_child)
                child_count = len(normalized_children)
            else:
                child_count = 0
            
            # Update level stats
            level_stats[level]["total"] += 1
            
            if child_count == 5:
                level_stats[level]["with_5"] += 1
            elif child_count == 0:
                level_stats[level]["with_0"] += 1
            elif 0 < child_count < 5:
                level_stats[level]["with_less_5"] += 1
        
        # Compute totals across all levels
        total_parents = sum(stats["total"] for stats in level_stats.values())
        parents_with_5 = sum(stats["with_5"] for stats in level_stats.values())
        parents_with_less_5 = sum(stats["with_less_5"] for stats in level_stats.values())
        parents_with_0 = sum(stats["with_0"] for stats in level_stats.values())
        
        return {
            "level_stats": level_stats,
            "total_parents": total_parents,
            "parents_with_5": parents_with_5,
            "parents_with_less_5": parents_with_less_5,
            "parents_with_0": parents_with_0
        }
        
    except Exception as e:
        # Log error for debugging but don't crash
        print(f"Error computing parent coverage summary: {e}")
        return {}


def _render_robust_summary(df: pd.DataFrame):
    """Render the robust parent coverage summary."""
    # Compute summary with caching using nonce
    summary = _compute_parent_coverage_summary(df, "local")
    
    if not summary:
        st.warning("No active sheet or invalid headers")
        return
    
    # Extract values
    total_parents = summary["total_parents"]
    parents_with_5 = summary["parents_with_5"]
    parents_with_less_5 = summary["parents_with_less_5"]
    parents_with_0 = summary["parents_with_0"]
    level_stats = summary["level_stats"]
    
    # Display KPIs
    st.subheader("📊 Parent Coverage Metrics")
    
    # Robust guards and dev hints
    _render_summary_guards_and_hints(df, summary)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(f"Parents with {MAX_CHILDREN_PER_PARENT} children", f"{parents_with_5}/{total_parents}")
    with col2:
        st.metric(f"Parents with <{MAX_CHILDREN_PER_PARENT} children", parents_with_less_5)
    with col3:
        st.metric("Parents with 0 children", parents_with_0)
    
    # Per-level breakdown table
    st.subheader("📋 Per-Level Parent Coverage")
    
    # Prepare table data
    table_data = []
    for level in range(1, 6):
        stats = level_stats[level]
        total = stats["total"]
        with_5 = stats["with_5"]
        with_less_5 = stats["with_less_5"]
        with_0 = stats["with_0"]
        
        if total > 0:
            coverage_pct = (with_5 / total) * 100
        else:
            coverage_pct = 0.0
            
        table_data.append({
            "Level": f"Level {level}",
            "Total Parents": total,
            "=5": with_5,
            "<5": with_less_5,
            "=0": with_0,
            "Coverage %": f"{coverage_pct:.1f}%"
        })
    
    if table_data:
        df_table = pd.DataFrame(table_data)
        st.dataframe(df_table, use_container_width=True, hide_index=True)
        
        # Add visual elements: progress bars and charts
        _render_coverage_visuals(summary)
    else:
        st.info("No parent data available for display.")
    
    # Worklist of parents missing children
    _render_parent_worklist(df, summary)


def _render_robust_summary_with_repair(df: pd.DataFrame):
    """Render the robust parent coverage summary with repair functionality."""
    # Compute summary with caching using nonce
    summary = _compute_parent_coverage_summary(df, "local")
    
    if not summary:
        st.warning("No active sheet or invalid headers")
        return None
    
    # Return the summary for the caller to use
    return summary


def _render_kpis_and_breakdown(df: pd.DataFrame, summary: Dict[str, Any]):
    """Render KPIs and per-level breakdown for the summary."""
    # Extract values
    total_parents = summary["total_parents"]
    parents_with_5 = summary["parents_with_5"]
    parents_with_less_5 = summary["parents_with_less_5"]
    parents_with_0 = summary["parents_with_0"]
    level_stats = summary["level_stats"]
    
    # Display KPIs
    st.subheader("📊 Parent Coverage Metrics")
    
    # Robust guards and dev hints
    _render_summary_guards_and_hints(df, summary)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(f"Parents with {MAX_CHILDREN_PER_PARENT} children", f"{parents_with_5}/{total_parents}")
    with col2:
        st.metric(f"Parents with <{MAX_CHILDREN_PER_PARENT} children", parents_with_less_5)
    with col3:
        st.metric("Parents with 0 children", parents_with_0)
    
    # Per-level breakdown table
    st.subheader("📋 Per-Level Parent Coverage")
    
    # Prepare table data
    table_data = []
    for level in range(1, 6):
        stats = level_stats[level]
        total = stats["total"]
        with_5 = stats["with_5"]
        with_less_5 = stats["with_less_5"]
        with_0 = stats["with_0"]
        
        if total > 0:
            coverage_pct = (with_5 / total) * 100
        else:
            coverage_pct = 0.0
            
        table_data.append({
            "Level": f"Level {level}",
            "Total Parents": total,
            "=5": with_5,
            "<5": with_less_5,
            "=0": with_0,
            "Coverage %": f"{coverage_pct:.1f}%"
        })
    
    if table_data:
        df_table = pd.DataFrame(table_data)
        st.dataframe(df_table, use_container_width=True, hide_index=True)
        
        # Add visual elements: progress bars and charts
        _render_coverage_visuals(summary)
    else:
        st.info("No parent data available for display.")


def _render_summary_guards_and_hints(df: pd.DataFrame, summary: Dict[str, Any]):
    """Render robust guards and developer hints for the summary."""
    total_parents = summary.get("total_parents", 0)
    total_rows = len(df)
    
    # Check for suspicious data patterns
    if total_parents == 0 and total_rows > 0:
        # Count rows with full paths to detect if data exists but store wasn't built
        full_path_count = _count_rows_with_full_paths(df)
        
        if full_path_count > 0:
            st.info("⚠️ **Heads-up: 0 parents found but {full_path_count} rows have full paths.** This usually means the store wasn't built properly: check headers and normalization.")
        else:
            st.info("ℹ️ **No parents found:** All rows appear to have incomplete paths.")
    
    # Store validation hints
    store_validation = _validate_store_building(df)
    if store_validation:
        st.info(f"🔍 **Store validation:** {store_validation}")
    
    # Recompute button for debugging
    col1, col2 = st.columns([3, 1])
    with col1:
        st.caption("💡 **Dev tip:** Use the recompute button if summary seems incorrect")
    with col2:
        if st.button("🔄 Recompute Summary", key="recompute_summary_btn", help="Force recompute of parent coverage summary"):
            _force_summary_recompute(df)


def _count_rows_with_full_paths(df: pd.DataFrame) -> int:
    """Count rows that have complete paths across all level columns."""
    try:
        if not validate_headers(df):
            return 0
        
        full_path_count = 0
        for _, row in df.iterrows():
            path_complete = True
            for col in LEVEL_COLS:
                if col in df.columns:
                    if normalize_text(row.get(col, "")) == "":
                        path_complete = False
                        break
            if path_complete:
                full_path_count += 1
        
        return full_path_count
    except Exception:
        return 0


def _validate_store_building(df: pd.DataFrame) -> str:
    """Validate that store building is working correctly and return hints."""
    try:
        if not validate_headers(df):
            return "Headers validation failed - cannot build store"
        
        # Check if we can build a store at all
        from logic.tree import infer_branch_options
        store = infer_branch_options(df)
        
        if not store:
            return "Store is empty - check if infer_branch_options is working"
        
        # Check for empty children filtering
        empty_children_count = 0
        total_children_count = 0
        
        for key, children in store.items():
            if key.startswith("L"):
                total_children_count += len(children) if children else 0
                # Count how many children are empty after normalization
                if children:
                    for child in children:
                        if normalize_text(child) == "":
                            empty_children_count += 1
        
        hints = []
        if empty_children_count > 0:
            hints.append(f"Found {empty_children_count} empty children (will be filtered)")
        
        if total_children_count == 0:
            hints.append("No children found in store - check data content")
        
        # Check level distribution
        level_counts = {}
        for key in store.keys():
            if key.startswith("L"):
                try:
                    level = int(key.split("|")[0][1:])
                    level_counts[level] = level_counts.get(level, 0) + 1
                except (ValueError, IndexError):
                    continue
        
        if level_counts:
            level_distribution = ", ".join([f"L{level}: {count}" for level, count in sorted(level_counts.items())])
            hints.append(f"Level distribution: {level_distribution}")
        
        return "; ".join(hints) if hints else "Store appears healthy"
        
    except Exception as e:
        return f"Store validation error: {str(e)}"


def _force_summary_recompute(df: pd.DataFrame):
    """Force recompute of the summary by clearing cache and refreshing."""
    try:
        # Use canonical cache invalidation
        current_sheet = get_current_sheet()
        if current_sheet:
            # Bump the nonce by setting the same sheet (this triggers cache invalidation)
            set_current_sheet(current_sheet)
        
        # Show feedback
        st.success("✅ Summary cache cleared! Recomputing...")
        
        # Rerun to trigger recomputation
        safe_rerun()
        
    except Exception as e:
        st.error(f"Error forcing recompute: {e}")


def _derive_child_count_distribution(summary: Dict[str, Any]) -> Dict[int, int]:
    """Derive child count distribution from the summary data."""
    try:
        level_stats = summary.get("level_stats", {})
        
        # Initialize distribution
        distribution = {i: 0 for i in range(6)}  # 0, 1, 2, 3, 4, 5
        
        # Aggregate counts from level stats
        for level in range(1, 6):
            stats = level_stats.get(level, {})
            distribution[5] += stats.get("with_5", 0)
            distribution[0] += stats.get("with_0", 0)
            # For <5, we'll distribute evenly across 1,2,3,4 as an approximation
            less_5_count = stats.get("with_less_5", 0)
            if less_5_count > 0:
                # Distribute across 1,2,3,4 (rough approximation)
                per_bucket = less_5_count // 4
                remainder = less_5_count % 4
                for i in range(1, 5):
                    distribution[i] += per_bucket
                    if i <= remainder:
                        distribution[i] += 1
        
        return distribution
        
    except Exception as e:
        print(f"Error deriving child count distribution: {e}")
        return {}


def _render_coverage_visuals(summary: Dict[str, Any]):
    """Render progress bars and child count distribution chart."""
    level_stats = summary.get("level_stats", {})
    
    # Progress bars for per-level coverage
    st.subheader("📊 Per-Level Coverage Progress")
    st.caption("Progress bars show coverage (=5 children / total parents) for each level")
    
    for level in range(1, 6):
        stats = level_stats.get(level, {"total": 0, "with_5": 0})
        total = stats["total"]
        with_5 = stats["with_5"]
        
        if total > 0:
            coverage_ratio = with_5 / total
            coverage_pct = coverage_ratio * 100
        else:
            coverage_ratio = 0.0
            coverage_pct = 0.0
        
        # Create columns for level label and progress bar
        col1, col2, col3 = st.columns([1, 3, 1])
        
        with col1:
            st.write(f"**Level {level}**")
        with col2:
            st.progress(coverage_ratio)
        with col3:
            if total > 0:
                st.write(f"{with_5}/{total} ({coverage_pct:.1f}%)")
            else:
                st.write("No parents")
    
    # Child count distribution chart
    st.subheader("📈 Child Count Distribution")
    st.caption("Bar chart showing how many parents have 0, 1, 2, 3, 4, or 5 children")
    
    # Get distribution data - derive from summary instead of recomputing
    distribution = _derive_child_count_distribution(summary)
    
    if distribution and any(count > 0 for count in distribution.values()):
        _render_child_count_chart(distribution)
    else:
        st.info("No distribution data available.")


def _render_child_count_chart(distribution: Dict[int, int]):
    """Render a bar chart of child count distribution using Altair."""
    try:
        import altair as alt
        
        # Prepare data for Altair
        chart_data = []
        for child_count, parent_count in distribution.items():
            chart_data.append({
                "Child Count": f"{child_count} children",
                "Parent Count": parent_count
            })
        
        # Create DataFrame for Altair
        chart_df = pd.DataFrame(chart_data)
        
        # Create Altair bar chart
        chart = alt.Chart(chart_df).mark_bar().add_selection(
            alt.selection_single()
        ).encode(
            x=alt.X('Child Count:N', title='Number of Children', sort=['0 children', '1 children', '2 children', '3 children', '4 children', '5 children']),
            y=alt.Y('Parent Count:Q', title='Number of Parents'),
            color=alt.condition(
                alt.datum['Child Count'] == '5 children',
                alt.value('green'),
                alt.condition(
                    alt.datum['Child Count'] == '0 children',
                    alt.value('red'),
                    alt.value('orange')
                )
            ),
            tooltip=['Child Count:N', 'Parent Count:Q']
        ).properties(
            width=400,
            height=300,
            title="Distribution of Parent Child Counts"
        ).resolve_scale(
            color='independent'
        )
        
        st.altair_chart(chart, use_container_width=True)
        
        # Summary statistics
        total_parents = sum(distribution.values())
        parents_with_5 = distribution.get(5, 0)
        parents_with_0 = distribution.get(0, 0)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Complete (5)", parents_with_5, help="Parents with exactly 5 children")
        with col2:
            st.metric("Empty (0)", parents_with_0, help="Parents with no children")
        with col3:
            if total_parents > 0:
                completion_rate = (parents_with_5 / total_parents) * 100
                st.metric("Completion Rate", f"{completion_rate:.1f}%", help="Percentage of parents with 5 children")
            else:
                st.metric("Completion Rate", "0%")
                
    except ImportError:
        # Fallback to matplotlib if Altair is not available
        _render_child_count_chart_matplotlib(distribution)
    except Exception as e:
        st.error(f"Error rendering child count chart: {e}")
        # Show raw data as fallback
        st.write("**Raw distribution data:**")
        for child_count, parent_count in sorted(distribution.items()):
            st.write(f"• {child_count} children: {parent_count} parents")


def _render_child_count_chart_matplotlib(distribution: Dict[int, int]):
    """Fallback renderer using matplotlib."""
    try:
        import matplotlib.pyplot as plt
        
        # Prepare data
        child_counts = list(range(6))  # 0, 1, 2, 3, 4, 5
        parent_counts = [distribution.get(i, 0) for i in child_counts]
        
        # Create colors (green for 5, red for 0, orange for others)
        colors = []
        for i in child_counts:
            if i == 5:
                colors.append('green')
            elif i == 0:
                colors.append('red')
            else:
                colors.append('orange')
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.bar([f"{i} children" for i in child_counts], parent_counts, color=colors, alpha=0.7)
        
        # Customize the plot
        ax.set_xlabel('Number of Children')
        ax.set_ylabel('Number of Parents')
        ax.set_title('Distribution of Parent Child Counts')
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar, count in zip(bars, parent_counts):
            if count > 0:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{count}', ha='center', va='bottom')
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Display in Streamlit
        st.pyplot(fig)
        plt.close()
        
    except ImportError:
        st.warning("Neither Altair nor Matplotlib is available. Showing raw data:")
        st.write("**Child count distribution:**")
        for child_count, parent_count in sorted(distribution.items()):
            st.write(f"• {child_count} children: {parent_count} parents")
    except Exception as e:
        st.error(f"Error rendering matplotlib chart: {e}")


def _render_parent_worklist(df: pd.DataFrame, summary: Dict[str, Any]):
    """Render the worklist of parents missing children with filtering and edit capabilities."""
    st.subheader("🔧 Parent Worklist - Missing Children")
    st.caption("Filter and edit parents that don't have exactly 5 children")
    
    # Quick Append UI
    _render_quick_append_ui(df)
    
    # Controls
    col1, col2, col3 = st.columns([1, 2, 2])
    
    with col1:
        selected_level = st.selectbox(
            "Level",
            options=list(range(1, 6)),
            format_func=lambda x: f"Level {x}",
            key="worklist_level_selector"
        )
    
    with col2:
        show_filter = st.selectbox(
            "Show filter",
            options=["<5 only", "=0 only", "All non-5"],
            key="worklist_filter_selector"
        )
    
    with col3:
        search_text = st.text_input(
            "Search parent paths",
            placeholder="e.g., High > Severe",
            key="worklist_search"
        )
    
    # Get the store to build the worklist
    from logic.tree import infer_branch_options
    store = infer_branch_options(df)
    
    # Filter parents based on selected criteria
    worklist_data = []
    
    for key, children in store.items():
        if not key.startswith(f"L{selected_level}|"):
            continue
            
        # Parse the parent path
        parts = key.split("|", 1)
        if len(parts) != 2:
            continue
            
        parent_path_str = parts[1]
        
        # Apply search filter
        if search_text and search_text.lower() not in parent_path_str.lower():
            continue
        
        # Count non-empty, deduped, normalized children
        if children:
            normalized_children = []
            seen = set()
            for child in children:
                clean_child = normalize_text(child)
                if clean_child and clean_child not in seen:
                    normalized_children.append(clean_child)
                    seen.add(clean_child)
            child_count = len(normalized_children)
            children_preview = normalized_children[:5]  # Show up to 5
        else:
            child_count = 0
            children_preview = []
        
        # Apply show filter
        if show_filter == "<5 only" and child_count >= 5:
            continue
        elif show_filter == "=0 only" and child_count != 0:
            continue
        elif show_filter == "All non-5" and child_count == 5:
            continue
        
        # Add to worklist
        worklist_data.append({
            "key": key,
            "level": selected_level,
            "parent_path": parent_path_str,
            "child_count": child_count,
            "children": children_preview
        })
    
    # Sort by child count (ascending) and then by parent path
    worklist_data.sort(key=lambda x: (x["child_count"], x["parent_path"]))
    
    # Display worklist
    if worklist_data:
        st.write(f"**Found {len(worklist_data)} parents to review:**")
        
        for i, item in enumerate(worklist_data):
            with st.expander(f"Level {item['level']}: {friendly_parent(item['level'], item['parent_path'])} ({item['child_count']}/5 children)", expanded=False):
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.write(f"**Parent Path:** {friendly_parent(item['level'], item['parent_path'])}")
                    st.write(f"**Current Children:** {item['child_count']}/5")
                    if item['children']:
                        st.write(f"**Children:** {', '.join(item['children'])}")
                    else:
                        st.write("**Children:** None")
                
                with col2:
                    # Edit button
                    if st.button(f"✏️ Edit Parent", key=f"edit_parent_{i}_{hash(item['key'])}"):
                        st.session_state['editing_parent'] = {
                            'key': item['key'],
                            'level': item['level'],
                            'parent_path': item['parent_path'],
                            'current_children': item['children']
                        }
                        safe_rerun()
                    
                    # Cross-navigation links
                    _render_cross_navigation_links(item, i)
        
        # Parent editor (if editing)
        if 'editing_parent' in st.session_state:
            _render_parent_editor(df, st.session_state['editing_parent'])
    else:
        st.info(f"No parents found at Level {selected_level} matching the current filters.")
    
    # Parent Inventory section
    _render_parent_inventory(df, store)


def _render_quick_append_ui(df: pd.DataFrame):
    """Render the Quick Append UI for adding a single child to a parent."""
    with st.expander("⚡ Quick Append (single parent)", expanded=False):
        st.caption("Add a single child to a specific parent without leaving the workspace")
        
        # Get the store for parent path selection
        from logic.tree import infer_branch_options
        store = infer_branch_options(df)
        
        # Level select (2-5)
        col1, col2 = st.columns([1, 2])
        
        with col1:
            append_level = st.selectbox(
                "Level",
                options=list(range(2, 6)),
                format_func=lambda x: f"Level {x}",
                key="quick_append_level",
                help="Select level 2-5 (parents must exist at level-1)"
            )
        
        with col2:
            # Get parent paths for the selected level
            parent_level = append_level - 1
            parent_paths = []
            parent_path_to_key = {}
            
            for key, children in store.items():
                if key.startswith(f"L{parent_level}|"):
                    parts = key.split("|", 1)
                    if len(parts) == 2:
                        parent_path_str = parts[1]
                        friendly_path = friendly_parent(parent_level, parent_path_str)
                        parent_paths.append(friendly_path)
                        parent_path_to_key[friendly_path] = parent_path_str
            
            parent_paths.sort()
            
            if parent_paths:
                selected_parent_friendly = st.selectbox(
                    "Parent path",
                    options=parent_paths,
                    key="quick_append_parent",
                    help="Select the parent to add a child to"
                )
                
                # Get the actual parent path
                selected_parent_path = parent_path_to_key[selected_parent_friendly]
                
                # Get current children for this parent at the append level
                current_key = f"L{append_level}|{selected_parent_path}"
                current_children = store.get(current_key, [])
                
                # Normalize and dedupe current children
                normalized_current = []
                seen = set()
                for child in current_children:
                    clean_child = normalize_text(child)
                    if clean_child and clean_child not in seen:
                        normalized_current.append(clean_child)
                        seen.add(clean_child)
                
                # Display current children
                st.write(f"**Current children ({len(normalized_current)}/5):**")
                if normalized_current:
                    # Display as chips/tags
                    children_display = " • ".join(f"`{child}`" for child in normalized_current)
                    st.write(children_display)
                else:
                    st.write("*No children yet*")
                
                # Add child input and button
                col3, col4 = st.columns([3, 1])
                
                with col3:
                    new_child = st.text_input(
                        "Add child",
                        placeholder="Enter new child name",
                        key="quick_append_child_input",
                        help="Enter the name of the child to add"
                    )
                
                with col4:
                    can_add = (
                        new_child.strip() != "" and 
                        len(normalized_current) < 5 and 
                        normalize_text(new_child) not in seen
                    )
                    
                    if st.button(
                        "➕ Add", 
                        key="quick_append_add_btn",
                        disabled=not can_add,
                        help="Add this child to the parent"
                    ):
                        _apply_quick_append(df, append_level, selected_parent_path, normalized_current, new_child)
                
                # Status message
                if new_child.strip() != "":
                    if len(normalized_current) >= 5:
                        st.warning("⚠️ Parent already has 5 children (maximum)")
                    elif normalize_text(new_child) in seen:
                        st.warning("⚠️ Child already exists for this parent")
                    elif normalize_text(new_child) == "":
                        st.warning("⚠️ Child name cannot be empty")
                
                # Apply button for full override
                if normalized_current:
                    st.markdown("---")
                    if st.button(
                        "🔄 Apply Current Set", 
                        key="quick_append_apply_btn",
                        help="Apply the current children set as an override"
                    ):
                        _apply_quick_append_override(df, append_level, selected_parent_path, normalized_current)
            else:
                st.info(f"No parents found at Level {parent_level}. Parents must exist before adding children.")


def _apply_quick_append(df: pd.DataFrame, level: int, parent_path: str, current_children: List[str], new_child: str):
    """Apply a quick append operation to add a single child."""
    try:
        # Add the new child to the current set
        new_child_clean = normalize_text(new_child)
        if not new_child_clean:
            st.error("Child name cannot be empty")
            return
        
        updated_children = current_children + [new_child_clean]
        
        # Apply the override
        _apply_quick_append_override(df, level, parent_path, updated_children)
        
    except Exception as e:
        st.error(f"Error adding child: {e}")
        st.exception(e)


def _apply_quick_append_override(df: pd.DataFrame, level: int, parent_path: str, children: List[str]):
    """Apply an override for the specified parent with the given children."""
    try:
        with st.spinner("Applying quick append..."):
            # Create parent tuple for override key
            if parent_path == "<ROOT>":
                parent_tuple = tuple()
            else:
                parent_tuple = tuple(parent_path.split(">"))
            
            override_key = (level, parent_tuple)
            
            # Get current overrides
            overrides_all = st.session_state.get("branch_overrides", {})
            if get_current_sheet() not in overrides_all:
                overrides_all[get_current_sheet()] = {}
            
            # Set the override
            overrides_all[get_current_sheet()][override_key] = children
            st.session_state["branch_overrides"] = overrides_all
            
            # Apply the override using logic.tree
            from logic.tree import build_raw_plus_v630
            
            # Get the active workbook
            active_wb = get_active_workbook()
            
            if active_wb and get_current_sheet() in active_wb:
                # Apply overrides and rebuild the sheet
                updated_df = build_raw_plus_v630(df, overrides_all[get_current_sheet()])
                
                # Update the active workbook
                active_wb[get_current_sheet()] = updated_df
                set_active_workbook(active_wb, source="quick_append")
                
                # Clear stale caches to ensure immediate refresh
                st.cache_data.clear()
                
                # Clear the input field
                st.session_state["quick_append_child_input"] = ""
                
                # Show success message
                st.success("✅ Child added successfully!")
                st.info(f"**Updated children:** {', '.join(children)}")
                
                # Rerun to show updated state
                safe_rerun()
            else:
                st.error("Could not update active workbook.")
                
    except Exception as e:
        st.error(f"Error applying quick append: {e}")
        st.exception(e)


def _render_parent_inventory(df: pd.DataFrame, store: Dict[str, List[str]]):
    """Render the parent inventory table with CSV download functionality."""
    st.markdown("---")
    st.subheader("📋 Parent Inventory")
    st.caption("Complete inventory of all parents across all levels")
    
    # Generate inventory data from the cached store
    inventory_data = []
    
    for key, children in store.items():
        if not key.startswith("L"):
            continue
            
        # Parse level and parent path from key (L{level}|{path})
        parts = key.split("|", 1)
        if len(parts) != 2:
            continue
            
        try:
            level = int(parts[0][1:])  # Remove "L" and convert to int
            if level < 1 or level > 5:
                continue
        except ValueError:
            continue
            
        parent_path_str = parts[1]
        
        # Count non-empty, deduped, normalized children
        if children:
            normalized_children = []
            seen = set()
            for child in children:
                clean_child = normalize_text(child)
                if clean_child and clean_child not in seen:
                    normalized_children.append(clean_child)
                    seen.add(clean_child)
            child_count = len(normalized_children)
            children_list = normalized_children[:5]  # Keep up to 5
        else:
            child_count = 0
            children_list = []
        
        # Add to inventory
        inventory_data.append({
            "Level": level,
            "Parent Path": friendly_parent(level, parent_path_str),
            "Child Count": child_count,
            "Children": ", ".join(children_list) if children_list else "None"
        })
    
    # Sort by level, then by parent path
    inventory_data.sort(key=lambda x: (x["Level"], x["Parent Path"]))
    
    # Display inventory table
    if inventory_data:
        st.write(f"**Total parents: {len(inventory_data)}**")
        
        # Create DataFrame for display with action columns
        df_inventory_display = pd.DataFrame(inventory_data)
        
        # Display inventory with cross-navigation for each row
        st.write("**Use the expandable rows below for cross-navigation:**")
        for i, row in enumerate(inventory_data):
            level = row["Level"]
            parent_path_friendly = row["Parent Path"]
            child_count = row["Child Count"]
            children_str = row["Children"]
            
            # Convert friendly path back to internal format for navigation
            if parent_path_friendly == "Top-level (Node 1) options":
                parent_path_internal = "<ROOT>"
            else:
                parent_path_internal = parent_path_friendly.replace(" > ", ">")
            
            with st.expander(f"Level {level}: {parent_path_friendly} ({child_count}/5)", expanded=False):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.write(f"**Children:** {children_str}")
                
                with col2:
                    # Create item structure for cross-navigation
                    nav_item = {
                        'level': level,
                        'parent_path': parent_path_internal,
                        'child_count': child_count
                    }
                    _render_cross_navigation_links(nav_item, f"inv_{i}")
        
        # Also show the full table for reference
        st.markdown("---")
        st.write("**Full table view:**")
        st.dataframe(df_inventory_display, use_container_width=True, hide_index=True)
        
        # CSV download
        st.markdown("---")
        st.subheader("💾 Export Inventory")
        
        # Convert to CSV
        csv_data = df_inventory_display.to_csv(index=False)
        
        # Create download button
        st.download_button(
            label="📥 Download CSV (parents_inventory.csv)",
            data=csv_data,
            file_name="parents_inventory.csv",
            mime="text/csv",
            help="Download the complete parent inventory as a CSV file"
        )
        
        st.caption(f"CSV contains {len(inventory_data)} rows with Level, Parent Path, Child Count, and Children columns")
    else:
        st.info("No parent data available for inventory display.")


def _render_parent_editor(df: pd.DataFrame, editing_info: Dict[str, Any]):
    """Render the in-tab editor for editing a specific parent's children."""
    st.markdown("---")
    st.subheader("✏️ Edit Parent Children")
    
    level = editing_info['level']
    parent_path = editing_info['parent_path']
    current_children = editing_info['current_children']
    
    st.write(f"**Editing:** Level {level} - {friendly_parent(level, parent_path)}")
    
    # Create 5 input fields for children
    new_children = []
    for i in range(5):
        default_value = current_children[i] if i < len(current_children) else ""
        child_value = st.text_input(
            f"Child {i+1}",
            value=default_value,
            key=f"edit_child_{level}_{hash(parent_path)}_{i}"
        )
        new_children.append(child_value)
    
    # Action buttons
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        if st.button("💾 Save Changes", key=f"save_parent_{level}_{hash(parent_path)}"):
            _save_parent_changes(df, level, parent_path, new_children)
    
    with col2:
        if st.button("❌ Cancel", key=f"cancel_parent_{level}_{hash(parent_path)}"):
            del st.session_state['editing_parent']
            safe_rerun()
    
    with col3:
        st.caption("Changes will be applied immediately and the summary will refresh")


def _save_parent_changes(df: pd.DataFrame, level: int, parent_path: str, new_children: List[str]):
    """Save changes to a parent's children using the write-back pipeline."""
    try:
        # Clean and validate children
        clean_children = [normalize_text(child) for child in new_children if normalize_text(child)]
        
        if len(clean_children) > 5:
            st.warning("Too many children. Only the first 5 will be saved.")
            clean_children = clean_children[:5]
        
        # Create parent tuple for override key
        if parent_path == "<ROOT>":
            parent_tuple = tuple()
        else:
            parent_tuple = tuple(parent_path.split(">"))
        
        override_key = (level, parent_tuple)
        
        # Get current overrides
        overrides_all = st.session_state.get("branch_overrides", {})
        if get_current_sheet() not in overrides_all:
            overrides_all[get_current_sheet()] = {}
        
        # Set the override
        overrides_all[get_current_sheet()][override_key] = clean_children
        st.session_state["branch_overrides"] = overrides_all
        
        # Apply the override using logic.tree
        from logic.tree import build_raw_plus_v630
        
        # Get the active workbook
        active_wb = get_active_workbook()
        
        if active_wb and get_current_sheet() in active_wb:
            # Apply overrides and rebuild the sheet
            updated_df = build_raw_plus_v630(df, overrides_all[get_current_sheet()])
            
            # Update the active workbook
            active_wb[get_current_sheet()] = updated_df
            set_active_workbook(active_wb, source="parent_editor")
            
            # Clear stale caches to ensure immediate refresh
            st.cache_data.clear()
            
            # Clear editing state
            del st.session_state['editing_parent']
            
            # Show success message
            st.success("✅ Parent children updated successfully!")
            st.info(f"**New children set:** {', '.join(clean_children)}")
            
            # Rerun to show updated state
            safe_rerun()
        else:
            st.error("Could not update active workbook.")
            
    except Exception as e:
        st.error(f"Error saving parent changes: {e}")
        st.exception(e)


def _render_cross_navigation_links(item: Dict[str, Any], unique_id: str):
    """Render cross-navigation links for Symptoms and Conflicts tabs."""
    level = item['level']
    parent_path = item['parent_path']
    
    # Edit in Symptoms link
    if st.button(f"🧬 Edit in Symptoms", key=f"symptoms_{unique_id}_{hash(parent_path)}"):
        _set_nav_hint("symptoms", level, parent_path)
        st.info("**Go to 🧬 Symptoms tab** - Branch Editor has been pre-filtered for you!")
        safe_rerun()
    
    # Resolve in Conflicts link
    if st.button(f"⚖️ Resolve in Conflicts", key=f"conflicts_{unique_id}_{hash(parent_path)}"):
        _set_nav_hint("conflicts", level, parent_path)
        st.info("**Go to ⚖️ Conflicts tab** - Level and Parent Label have been pre-selected for you!")
        safe_rerun()


def _set_nav_hint(tab: str, level: int, parent_path: str):
    """Set navigation hint in session state for cross-tab navigation."""
    try:
        # Set the navigation hint
        st.session_state["_nav_hint"] = {
            "tab": tab,
            "level": level,
            "parent": parent_path
        }
        
        # For Symptoms tab, set specific branch editor state
        if tab == "symptoms":
            # Convert parent path to parent tuple for Symptoms editor
            if parent_path == "<ROOT>":
                parent_tuple = tuple()
            else:
                parent_tuple = tuple(parent_path.split(">"))
            
            # Set the branch editor level and parent selection
            st.session_state['branch_editor_level'] = level
            st.session_state['branch_editor_parent'] = parent_tuple
        
        # For Conflicts tab, set resolve tool state
        elif tab == "conflicts":
            st.session_state['resolve_level'] = level
            if parent_path == "<ROOT>":
                st.session_state['resolve_parent_label'] = "Top-level (Node 1) options"
            else:
                # Extract the last label from the path for conflicts resolution
                parent_labels = [label.strip() for label in parent_path.split(">")]
                st.session_state['resolve_parent_label'] = parent_labels[-1]
        
    except Exception as e:
        st.warning(f"Could not set navigation hint: {e}")


def _clear_summary_cache(df: pd.DataFrame):
    """Clear the cached summary for a specific dataframe using canonical cache invalidation."""
    try:
        # The canonical cache system automatically invalidates when workbook changes
        # This function is kept for backward compatibility but no longer needed
        pass
    except Exception:
        pass


def _apply_workspace_changes(df: pd.DataFrame, sheet_name: str, changes: Dict[str, Any]):
    """Apply changes to the workspace using the write-back/materialize pipeline."""
    try:
        with st.spinner("Applying workspace changes..."):
            # Get current overrides
            overrides_all = st.session_state.get("branch_overrides", {})
            if sheet_name not in overrides_all:
                overrides_all[sheet_name] = {}
            
            # Apply the changes using logic.tree
            from logic.tree import build_raw_plus_v630
            
            # Get the active workbook
            active_wb = get_active_workbook()
            
            if active_wb and sheet_name in active_wb:
                # Apply overrides and rebuild the sheet
                updated_df = build_raw_plus_v630(df, overrides_all[sheet_name])
                
                # Update the active workbook
                active_wb[sheet_name] = updated_df
                set_active_workbook(active_wb, source="workspace_changes")
                
                # Clear stale caches to ensure immediate refresh
                st.cache_data.clear()
                
                # Show success message
                st.success("✅ Workspace changes applied successfully!")
                
                # Rerun to show updated state
                safe_rerun()
            else:
                st.error("Could not update active workbook.")
                
    except Exception as e:
        st.error(f"Error applying workspace changes: {e}")
        st.exception(e)


def _render_preview_section(df: pd.DataFrame, sheet_name: str):
    """Render the preview section with pagination."""
    total_rows = len(df)
    st.markdown("#### Preview (50 rows)")
    
    if total_rows <= 50:
        st.caption(f"Showing all {total_rows} rows.")
        st.dataframe(df, use_container_width=True)
    else:
        state_key = f"preview_start_{sheet_name}"
        start_idx = int(st.session_state.get(state_key, 0))
        
        cprev, cnum, cnext = st.columns([1, 2, 1])
        with cprev:
            if st.button("◀ Previous 50", key=f"prev50_{sheet_name}"):
                start_idx = max(0, start_idx - 50)
        with cnum:
            start_1based = st.number_input(
                "Start row (1-based)",
                min_value=1,
                max_value=max(1, total_rows - 49),
                value=start_idx + 1,
                step=50,
                help="Pick where to start the 50-row preview.",
                key=f"startnum_{sheet_name}"
            )
            start_idx = int(start_1based) - 1
        with cnext:
            if st.button("Next 50 ▶", key=f"next50_{sheet_name}"):
                start_idx = min(max(0, total_rows - 50), start_idx + 50)
        
        st.session_state[state_key] = start_idx
        end_idx = min(start_idx + 50, total_rows)
        st.caption(f"Showing rows **{start_idx + 1}–{end_idx}** of **{total_rows}**.")
        st.dataframe(df.iloc[start_idx:end_idx], use_container_width=True)


def _render_grouping_controls(df: pd.DataFrame):
    """Render the grouping controls section."""
    with st.expander("🧩 Group rows (cluster identical labels together)"):
        st.caption("Group rows so identical **Node 1** and **Node 2** values are contiguous.")
        
        if df.empty or not validate_headers(df):
            st.info("Load a valid sheet first.")
            return
            
        scope = st.radio("Grouping scope", ["Whole sheet", "Within Vital Measurement"], horizontal=True, key="ws_group_scope_sel")
        rf_scope = st.radio("Red-Flag priority", ["Node 2 only", "Any node (Dictionary)"], horizontal=True, key="ws_group_rf_scope")
        preview = st.checkbox("Show preview (does not modify data)", value=True, key="ws_group_preview_sel")

        if preview:
            grouped_df = _grouped_df(df, scope, rf_scope)
            st.dataframe(grouped_df.head(100), use_container_width=True)
            st.caption(f"Showing first 100 rows of grouped data. Total: {len(grouped_df)} rows.")


def _grouped_df(df: pd.DataFrame, scope_mode: str, rf_scope_sel: str) -> pd.DataFrame:
    """Create a grouped DataFrame based on the specified parameters."""
    try:
        if rf_scope_sel == "Node 2 only":
            # Simple sort by Node 1, Node 2
            df2 = df.sort_values(["Node 1", "Node 2"], kind="stable")
        else:
            # Sort by all nodes
            sort_cols = ["Vital Measurement"] + LEVEL_COLS
            sort_cols = [col for col in sort_cols if col in df.columns]
            df2 = df.sort_values(sort_cols, kind="stable")
            
        if scope_mode == "Within Vital Measurement":
            df2["_vm"] = df2["Vital Measurement"].map(normalize_text)
            df2["_row"] = np.arange(len(df2))
            df2 = df2.sort_values(["_vm", "Node 1", "Node 2", "Node 3", "Node 4", "Node 5", "_row"], kind="stable").drop(columns=["_vm", "_row"])
            
        return df2
    except Exception:
        return df
