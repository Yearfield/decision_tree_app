"""
Decision Tree Builder - Modular Streamlit Application
Refactored from monolith to clean modular architecture
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, List

from utils import APP_VERSION, CANON_HEADERS, LEVEL_COLS, normalize_text, validate_headers
from logic.tree import get_cached_branch_options
from logic.validate import (
    get_cached_orphan_nodes, get_cached_loops, get_cached_validation_report
)
from ui.tabs import (
    source, workspace, validation, conflicts, triage, actions, 
    symptoms, dictionary, visualizer, push_log
)


def main():
    """Main application entry point."""
    # Page configuration
st.set_page_config(
        page_title=f"Decision Tree Builder {APP_VERSION}",
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
    # Core workbook state
    if "upload_workbook" not in st.session_state:
        st.session_state["upload_workbook"] = {}
    
    if "gs_workbook" not in st.session_state:
        st.session_state["gs_workbook"] = {}
    
    if "work_context" not in st.session_state:
        st.session_state["work_context"] = {}
    
    if "current_sheet" not in st.session_state:
        st.session_state["current_sheet"] = None
    
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


def _render_header():
    """Render the application header."""
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.title(f"🌳 Decision Tree Builder {APP_VERSION}")
        st.markdown("Build, validate, and manage decision trees with ease")
    
    with col2:
        # Show active workbook info
        if _has_active_workbook():
            ctx = st.session_state.get("work_context", {})
            source_type = ctx.get("source", "None")
            sheet_name = ctx.get("sheet", "None")
            
            if source_type and sheet_name:
                st.info(f"📁 {source_type.upper()}: {sheet_name}")
            else:
                st.info("📁 No sheet selected")
        else:
            st.warning("📁 No workbook loaded")
    
    with col3:
        # Show header badge if we have data
        if _has_active_workbook():
            ctx = st.session_state.get("work_context", {})
            if ctx.get("sheet"):
                df = _get_current_dataframe()
                if not df.empty:
                    badge = compute_header_badge(df)
                    st.metric("Data Quality", f"{badge['parent_score']}/{badge['row_score']}")


def _render_main_content():
    """Render the main content area with tabs."""
    # Tab registry
    TAB_REGISTRY = [
        ("📂 Source", source.render),
        ("🗂 Workspace Selection", workspace.render),
        ("🔎 Validation", validation.render),
        ("⚖️ Conflicts", conflicts.render),
        ("🩺 Diagnostic Triage", triage.render),
        ("⚡ Actions", actions.render),
        ("🧬 Symptoms", symptoms.render),
        ("📖 Dictionary", dictionary.render),
        ("🧮 Calculator", lambda: st.info("Moved into Validation/Visualizer for now")),
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


def _has_active_workbook() -> bool:
    """Check if there's an active workbook in session state."""
    upload_wb = st.session_state.get("upload_workbook", {})
    gs_wb = st.session_state.get("gs_workbook", {})
    return bool(upload_wb or gs_wb)


def _get_current_dataframe() -> pd.DataFrame:
    """Get the current active DataFrame."""
    try:
        ctx = st.session_state.get("work_context", {})
        src = ctx.get("source")
        sheet = ctx.get("sheet")
        
        if not src or not sheet:
            return pd.DataFrame()
        
        if src == "upload":
            wb = st.session_state.get("upload_workbook", {})
else:
            wb = st.session_state.get("gs_workbook", {})
        
        if sheet in wb:
            return wb[sheet]
        
        return pd.DataFrame()
    except Exception:
        return pd.DataFrame()


def _get_cache_key(sheet_name: str, data_shape: Tuple[int, int], data_hash: str) -> Tuple:
    """Generate a cache key for heavy computations."""
    return (sheet_name, APP_VERSION, data_shape, data_hash)


@st.cache_data(ttl=600)
def compute_header_badge(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute header badge metrics for the active sheet.
    
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
def get_cached_branch_options_for_ui(df: pd.DataFrame, sheet_name: str) -> Dict[str, Any]:
    """Cached branch options computation for UI.
    
    Args:
        df: DataFrame with decision tree data
        sheet_name: Name of the current sheet
        
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
def get_cached_validation_summary_for_ui(df: pd.DataFrame, sheet_name: str) -> Dict[str, Any]:
    """Cached validation summary computation for UI.
    
    Args:
        df: DataFrame with decision tree data
        sheet_name: Name of the current sheet
        
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
def get_cached_conflict_summary_for_ui(df: pd.DataFrame, sheet_name: str) -> Dict[str, Any]:
    """Cached conflict summary computation for UI.
    
    Args:
        df: DataFrame with decision tree data
        sheet_name: Name of the current sheet
        
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


def get_active_df():
    """Get the currently active DataFrame from session state."""
    wb_u = st.session_state.get("upload_workbook", {})
    wb_g = st.session_state.get("gs_workbook", {})
    ctx = st.session_state.get("work_context", {})
    sheet = ctx.get("sheet")
    if sheet and sheet in wb_u: 
        return wb_u[sheet]
    if sheet and sheet in wb_g: 
        return wb_g[sheet]
    return None


if __name__ == "__main__":
    main()
