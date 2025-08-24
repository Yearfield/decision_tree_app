"""
Decision Tree Builder - Modular Streamlit Application
Refactored from monolith to clean modular architecture
"""

import streamlit as st
import pandas as pd
import numpy as np
import time
from typing import Dict, Any, Tuple, List

from utils import APP_VERSION, CANON_HEADERS, LEVEL_COLS, normalize_text, validate_headers
from utils.state import (
    set_active_workbook, get_active_workbook, set_current_sheet, get_current_sheet,
    get_active_df, has_active_workbook, get_workbook_status, migrate_legacy_state,
    get_wb_nonce, verify_active_workbook
)
from logic.tree import get_cached_branch_options
from logic.validate import (
    get_cached_orphan_nodes, get_cached_loops, get_cached_validation_report
)
from ui.tabs import (
    source, workspace, validation, conflicts, 
    symptoms, outcomes, dictionary, calculator, visualizer, push_log
)


def main():
    """Main application entry point."""
    # Page configuration
    st.set_page_config(
        page_title=f"Decision Tree App {APP_VERSION}",
        page_icon="🌳",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state
    _initialize_session_state()
    
    # Header with version and status
    _render_header()
    
    # Main content area
    _render_main_content()


def _initialize_session_state():
    """Initialize all session state variables."""
    # Core workbook state (unified)
    if "workbook" not in st.session_state:
        st.session_state["workbook"] = {}
    
    if "current_sheet" not in st.session_state:
        st.session_state["current_sheet"] = None
    
    if "wb_nonce" not in st.session_state:
        st.session_state["wb_nonce"] = ""
    
    # Legacy state (kept for backward compatibility during migration)
    if "upload_workbook" not in st.session_state:
        st.session_state["upload_workbook"] = {}
    
    if "gs_workbook" not in st.session_state:
        st.session_state["gs_workbook"] = {}
    
    if "work_context" not in st.session_state:
        st.session_state["work_context"] = {}
    
    # Branch overrides and symptom quality
    if "branch_overrides" not in st.session_state:
        st.session_state["branch_overrides"] = {}
    
    if "symptom_quality" not in st.session_state:
        st.session_state["symptom_quality"] = {}
    
    # Dictionary and push log
    if "term_dictionary" not in st.session_state:
        st.session_state["term_dictionary"] = {}
    
    if "push_log" not in st.session_state:
        st.session_state["push_log"] = []
    
    # UI state
    if "current_tab" not in st.session_state:
        st.session_state["current_tab"] = "source"
    
    # Migrate legacy state to unified format
    migrate_legacy_state()


def _render_header():
    """Render the application header."""
    import pandas as pd
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.title(f"🌳 Decision Tree App {APP_VERSION}")
        st.markdown("Build, validate, and manage decision trees with ease")
    
    with col2:
        # Global active-sheet selector
        wb = get_active_workbook()
        sheet = get_current_sheet()
        if wb:
            options = list(wb.keys())
            if options:
                idx = options.index(sheet) if sheet in options else 0
                new_sheet = st.selectbox("Active sheet", options, index=idx, key="__active_sheet_box")
                if new_sheet != sheet:
                    set_current_sheet(new_sheet)
            st.caption(f"Workbook: ✅ {len(wb)} sheet(s) • Active: **{get_current_sheet()}**")
        else:
            st.caption("Workbook: ❌ not loaded")
        
        # Header badges showing current context stats
        df0 = get_active_df()
        badges_html = ""
        if isinstance(df0, pd.DataFrame) and not df0.empty and validate_headers(df0):
            # Inline helper functions for metrics
            def _rows_full_path_counts(df0: pd.DataFrame) -> tuple[int, int]:
                if df0 is None or df0.empty:
                    return 0, 0
                node_cols = ["Node 1","Node 2","Node 3","Node 4","Node 5"]
                for c in node_cols:
                    if c not in df0.columns:
                        df0[c] = ""
                ok = int((df0[node_cols] != "").all(axis=1).sum())
                total = int(len(df0))
                return ok, total

            def _parents_vectorized_counts(df0: pd.DataFrame) -> tuple[int, int]:
                if df0 is None or df0.empty:
                    return 0, 0
                dfv = df0.copy()
                node_cols = ["Node 1","Node 2","Node 3","Node 4","Node 5"]
                for c in ["Vital Measurement"] + node_cols:
                    if c not in dfv.columns:
                        dfv[c] = ""
                    dfv[c] = dfv[c].astype(str).str.strip()
                ok_total = 0
                total_parents = 0
                for lvl in range(1, 6):
                    parent_cols = node_cols[:lvl-1]
                    child_col  = node_cols[lvl-1]
                    scope = dfv[dfv[child_col] != ""].copy()
                    if parent_cols:
                        scope = scope[(scope[parent_cols] != "").all(axis=1)]
                    if scope.empty:
                        continue
                    grp = (scope.groupby(parent_cols, dropna=False)[child_col].nunique()
                           if parent_cols else
                           scope.assign(__root="__root").groupby("__root")[child_col].nunique())
                    total_parents += int(len(grp))
                    ok_total      += int((grp == 5).sum())
                return ok_total, total_parents
            
            ok_p, total_p = _parents_vectorized_counts(df0)
            ok_r, total_r = _rows_full_path_counts(df0)
            pct_p = 0 if total_p==0 else int(round(100*ok_p/total_p))
            pct_r = 0 if total_r==0 else int(round(100*ok_r/total_r))
            badges_html = f"""
            <style>
            .badge-wrap {{ display:flex; gap:16px; flex-wrap: wrap; }}
            .b {{ padding:8px 10px; border-radius:12px; border:1px solid #dbe0e6; background:#f8fafc; 
                 font: 12px/1.2 -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Helvetica, Arial, sans-serif; color:#0f172a; }}
            .bar {{ position:relative; height:8px; background:#eef2f6; border:1px solid #dbe0e6; border-radius:8px; overflow:hidden; width:160px; }}
            .fill1 {{ position:absolute; left:0; top:0; bottom:0; width:{pct_p}%; background:linear-gradient(90deg,#60a5fa,#2563eb); }}
            .fill2 {{ position:absolute; left:0; top:0; bottom:0; width:{pct_r}%; background:linear-gradient(90deg,#34d399,#059669); }}
            .strong {{ font-weight:600; }}
            </style>
            <div class='badge-wrap'>
              <div class='b'>
                <div>Parents 5/5 — <span class='strong'>{ok_p}/{total_p}</span></div>
                <div class='bar'><div class='fill1'></div></div>
              </div>
              <div class='b'>
                <div>Rows full path — <span class='strong'>{ok_r}/{total_r}</span></div>
                <div class='bar'><div class='fill2'></div></div>
              </div>
            </div>
            """
        if badges_html:
            st.markdown(badges_html, unsafe_allow_html=True)
    
    with col3:
        # Dev Panel instrumentation
        with st.expander("🛠 Dev Panel", expanded=False):
            try:
                import pandas as pd
                
                # Show verification report
                rep = verify_active_workbook()
                st.write("**Workbook Verification:**")
                st.json(rep)
                
                # Show DataFrame preview
                df = get_active_df()
                if isinstance(df, pd.DataFrame):
                    st.caption(f"Headers: {list(df.columns)[:8]}{' …' if len(df.columns)>8 else ''}")
                    st.dataframe(df.head(10), use_container_width=True)
                else:
                    st.info("Active df not available (not a DataFrame).")
                
                # Show basic state info
                wb = get_active_workbook()
                sheet = get_current_sheet()
                st.write("**Basic State:**")
                st.write({
                    "workbook_source": st.session_state.get("workbook_source"),
                    "sheet_id": st.session_state.get("sheet_id"),
                    "sheet_name": st.session_state.get("sheet_name"),
                    "current_sheet": sheet,
                    "wb_nonce": get_wb_nonce(),
                    "wb_keys": list(wb.keys()) if isinstance(wb, dict) else None,
                    "df_shape": (df.shape if isinstance(df, pd.DataFrame) else None),
                    "ts": time.strftime("%H:%M:%S")
                })
            except Exception as e:
                st.error(f"Dev Panel error: {e}")
                st.write("State inspection failed - app continues normally")


def _render_main_content():
    """Render the main content area with tabs."""
    # Tab registry
    TAB_REGISTRY = [
        ("📂 Source", source.render),
        ("🗂 Workspace Selection", workspace.render),
        ("🔎 Validation", validation.render),
        ("⚖️ Conflicts", conflicts.render),
        ("🧬 Symptoms", symptoms.render),
        ("📝 Outcomes", outcomes.render),
        ("📖 Dictionary", dictionary.render),
        ("🧮 Calculator", calculator.render),
        ("🌐 Visualizer", visualizer.render),
        ("📜 Push Log", push_log.render),
    ]
    
    # Create tabs
    tab_names = [t[0] for t in TAB_REGISTRY]
    tabs = st.tabs(tab_names)
    
    # Render each tab
    for i, (_, fn) in enumerate(TAB_REGISTRY):
        with tabs[i]:
            try:
                fn()
            except Exception as e:
                st.error("This tab crashed:")
                st.exception(e)


def _get_cache_key(sheet_name: str, data_shape: Tuple[int, int], data_hash: str) -> Tuple:
    """Generate a cache key for heavy computations."""
    return (sheet_name, APP_VERSION, data_shape, data_hash)


@st.cache_data(ttl=600)
def compute_header_badge(df: pd.DataFrame, nonce: str) -> Dict[str, Any]:
    """Compute header badge metrics for the active sheet.
    
    Args:
        df: DataFrame with decision tree data
        nonce: Workbook nonce for cache invalidation
        
    Returns:
        Dict with 'parent_score' and 'row_score' metrics
    """
    try:
        if df.empty or not validate_headers(df):
            return {"parent_score": "0/0", "row_score": "0/0"}
        
        # Compute parent depth score (parents with 5 children)
        parent_score = _compute_parent_depth_score(df)
        
        # Compute row completeness score (rows with full paths)
        row_score = _compute_row_completeness_score(df)
        
        return {
            "parent_score": f"{parent_score['ok']}/{parent_score['total']}",
            "row_score": f"{row_score['ok']}/{row_score['total']}"
        }
        
    except Exception:
        return {"parent_score": "0/0", "row_score": "0/0"}


@st.cache_data(ttl=600)
def get_cached_branch_options_for_ui(df: pd.DataFrame, sheet_name: str, nonce: str) -> Dict[str, Any]:
    """Cached branch options computation for UI.
    
    Args:
        df: DataFrame with decision tree data
        sheet_name: Name of the current sheet
        nonce: Workbook nonce for cache invalidation
        
    Returns:
        Dictionary mapping level keys to lists of possible values
    """
    try:
        if df.empty or not validate_headers(df):
            return {}
        
        # Create cache key based on data shape and content
        data_shape = (len(df), len(df.columns))
        data_hash = str(hash(str(df.values.tobytes())))
        cache_key = _get_cache_key(sheet_name, data_shape, data_hash)
        
        return get_cached_branch_options(df, cache_key)
        
    except Exception:
        return {}


@st.cache_data(ttl=600)
def get_cached_validation_summary_for_ui(df: pd.DataFrame, sheet_name: str, nonce: str) -> Dict[str, Any]:
    """Cached validation summary computation for UI.
    
    Args:
        df: DataFrame with decision tree data
        sheet_name: Name of the current sheet
        nonce: Workbook nonce for cache invalidation
        
    Returns:
        Dictionary containing validation summary and details
    """
    try:
        if df.empty or not validate_headers(df):
            return {
                "summary": {
                    "total_orphans": 0,
                    "total_loops": 0,
                    "total_missing_red_flags": 0,
                    "total_issues": 0
                },
                "orphans": [],
                "loops": [],
                "missing_red_flags": []
            }
        
        # Create cache key based on data shape and content
        data_shape = (len(df), len(df.columns))
        data_hash = str(hash(str(df.values.tobytes())))
        cache_key = _get_cache_key(sheet_name, data_shape, data_hash)
        
        return get_cached_validation_report(df, cache_key)
        
    except Exception:
        return {
            "summary": {
                "total_orphans": 0,
                "total_loops": 0,
                "total_missing_red_flags": 0,
                "total_issues": 0
            },
            "orphans": [],
            "loops": [],
            "missing_red_flags": []
        }


@st.cache_data(ttl=600)
def get_cached_conflict_summary_for_ui(df: pd.DataFrame, sheet_name: str, nonce: str) -> Dict[str, Any]:
    """Cached conflict summary computation for UI.
    
    Args:
        df: DataFrame with decision tree data
        sheet_name: Name of the current sheet
        nonce: Workbook nonce for cache invalidation
        
    Returns:
        Dictionary containing conflict summary and details
    """
    try:
        if df.empty or not validate_headers(df):
            return {
                "total_conflicts": 0,
                "conflicts": [],
                "conflict_types": {}
            }
        
        # Create cache key based on data shape and content
        data_shape = (len(df), len(df.columns))
        data_hash = str(hash(str(df.values.tobytes())))
        cache_key = _get_cache_key(sheet_name, data_shape, data_hash)
        
        # Compute conflicts (this would be implemented in logic.conflicts)
        conflicts = _compute_basic_conflicts(df)
        
        # Group conflicts by type
        conflict_types = {}
        for conflict in conflicts:
            conflict_type = conflict.get("type", "unknown")
            if conflict_type not in conflict_types:
                conflict_types[conflict_type] = []
            conflict_types[conflict_type].append(conflict)
        
        return {
            "total_conflicts": len(conflicts),
            "conflicts": conflicts,
            "conflict_types": conflict_types
        }
        
    except Exception:
        return {
            "total_conflicts": 0,
            "conflicts": [],
            "conflict_types": {}
        }


def _compute_basic_conflicts(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Compute basic conflicts in the decision tree."""
    conflicts = []
    
    try:
        # Check for duplicate paths
        for level in range(1, 6):
            if f"Node {level}" not in df.columns:
                continue
                
            # Build paths up to this level
            path_cols = ["Vital Measurement"] + [f"Node {i}" for i in range(1, level + 1)]
            if not all(col in df.columns for col in path_cols):
                continue
                
            # Check for duplicate paths
            paths = df[path_cols].apply(lambda r: tuple(normalize_text(v) for v in r), axis=1)
            path_counts = paths.value_counts()
            
            for path, count in path_counts.items():
                if count > 1 and all(v != "" for v in path):
                    conflicts.append({
                        "type": "duplicate_path",
                        "level": level,
                        "path": " > ".join(path),
                        "count": int(count),
                        "description": f"Path appears {count} times at level {level}"
                    })

        # Check for inconsistent children counts
        for level in range(1, 5):
            if f"Node {level}" not in df.columns or f"Node {level + 1}" not in df.columns:
                continue
                
            # Group by parent path
            parent_cols = ["Vital Measurement"] + [f"Node {i}" for i in range(1, level + 1)]
            if not all(col in df.columns for col in parent_cols):
                continue
                
            parent_paths = df[parent_cols].apply(lambda r: tuple(normalize_text(v) for v in r), axis=1)
            parent_paths = parent_paths[parent_paths.apply(lambda x: all(v != "" for v in x))]
            
            for parent_path in parent_paths.unique():
                mask = parent_paths == parent_path
                children = df.loc[mask, f"Node {level + 1}"].map(normalize_text)
                children = children[children != ""]
                unique_children = children.unique()
                
                if len(unique_children) != 5:
                    conflicts.append({
                        "type": "inconsistent_children",
                        "level": level,
                        "parent_path": " > ".join(parent_path),
                        "expected": 5,
                        "actual": len(unique_children),
                        "children": list(unique_children),
                        "description": f"Parent should have 5 children, but has {len(unique_children)}"
                    })

    except Exception as e:
        # Log error but don't fail
        print(f"Error computing conflicts: {e}")
        
    return conflicts


def _compute_parent_depth_score(df: pd.DataFrame) -> Dict[str, int]:
    """Compute parent depth score (how many parents have 5 children)."""
    try:
        total = 0
        ok = 0
        
        for level in range(1, 5):  # Check levels 1-4 for having 5 children
            if f"Node {level}" not in df.columns or f"Node {level + 1}" not in df.columns:
                continue
                
            # Group by parent path
            parent_cols = [f"Node {i}" for i in range(1, level + 1)]
            if not all(col in df.columns for col in parent_cols):
                continue
                
            # Count unique parent paths
            parent_paths = df[parent_cols].apply(lambda r: tuple(normalize_text(v) for v in r), axis=1)
            parent_paths = parent_paths[parent_paths.apply(lambda x: all(v != "" for v in x))]
            unique_parents = parent_paths.unique()
            
            for parent in unique_parents:
                total += 1
                # Count children for this parent
                mask = parent_paths == parent
                children = df.loc[mask, f"Node {level + 1}"].map(normalize_text)
                children = children[children != ""]
                if len(children.unique()) == 5:
                    ok += 1
                    
        return {"ok": ok, "total": total}
    except Exception:
        return {"ok": 0, "total": 0}


def _compute_row_completeness_score(df: pd.DataFrame) -> Dict[str, int]:
    """Compute row completeness score (how many rows have full paths)."""
    try:
        total = len(df)
        ok = 0
        
        for _, row in df.iterrows():
            path_complete = True
            for col in LEVEL_COLS:
                if col in df.columns:
                    if normalize_text(row.get(col, "")) == "":
                        path_complete = False
                        break
            if path_complete:
                ok += 1
                
        return {"ok": ok, "total": total}
    except Exception:
        return {"ok": 0, "total": 0}





if __name__ == "__main__":
    main()
