# utils/state.py
"""
Unified state management for workbook and sheet selection.
All tabs should use these helpers to access the active workbook state.
"""

import streamlit as st
import pandas as pd
from typing import Dict, Optional, Any, List
from uuid import uuid4

# Canonical keys
WORKBOOK_KEY = "workbook"          # Dict[str, pd.DataFrame]
CURRENT_SHEET_KEY = "current_sheet"  # str
WB_NONCE_KEY = "wb_nonce"          # random string for cache busting

def set_active_workbook(wb: Dict[str, pd.DataFrame], default_sheet: Optional[str] = None, source: str = "") -> None:
    """Set the active workbook in session state with cache busting."""
    # Verify that all values are DataFrames
    if not isinstance(wb, dict):
        raise ValueError(f"workbook must be a dict, got {type(wb).__name__}")
    
    # Check each value is a DataFrame
    non_df_keys = [k for k, v in wb.items() if not isinstance(v, pd.DataFrame)]
    if non_df_keys:
        raise ValueError(f"workbook contains non-DataFrame entries: {non_df_keys}")
    
    st.session_state[WORKBOOK_KEY] = wb
    st.session_state[WB_NONCE_KEY] = uuid4().hex
    if default_sheet is None:
        default_sheet = (list(wb.keys())[0] if wb else None)
    st.session_state[CURRENT_SHEET_KEY] = default_sheet
    if source:
        st.session_state["workbook_source"] = source

def get_active_workbook() -> Optional[Dict[str, pd.DataFrame]]:
    """Get the active workbook from session state."""
    return st.session_state.get(WORKBOOK_KEY)

def set_current_sheet(name: Optional[str]) -> None:
    """Set the current sheet name in session state with cache busting."""
    st.session_state[CURRENT_SHEET_KEY] = name
    # Bump nonce so caches that depend on current sheet refresh
    st.session_state[WB_NONCE_KEY] = uuid4().hex

def get_current_sheet() -> Optional[str]:
    """Get the current sheet name from session state."""
    return st.session_state.get(CURRENT_SHEET_KEY)

def get_wb_nonce() -> str:
    """Get the current workbook nonce for cache keys."""
    return st.session_state.get(WB_NONCE_KEY, "")

def get_active_df() -> Optional[pd.DataFrame]:
    """Get the currently active DataFrame from the active workbook and sheet."""
    wb = get_active_workbook()
    name = get_current_sheet()
    if not wb or not name:
        return None
    return wb.get(name)

def has_active_workbook() -> bool:
    """Check if there's an active workbook in session state."""
    wb = get_active_workbook()
    return bool(wb and len(wb) > 0)

def get_workbook_status() -> tuple[bool, Optional[str], Optional[str]]:
    """Get workbook status for UI display.
    
    Returns:
        Tuple of (has_workbook, sheet_count, current_sheet)
    """
    wb = get_active_workbook()
    sheet = get_current_sheet()
    
    if not wb:
        return False, None, None
    
    sheet_count = len(wb)
    return True, str(sheet_count), sheet

def migrate_legacy_state() -> None:
    """Migrate legacy state keys to the new unified format.
    This should be called during app initialization.
    """
    # Check if we need to migrate from old state
    if WORKBOOK_KEY not in st.session_state:
        # Try to migrate from legacy keys
        upload_wb = st.session_state.get("upload_workbook", {})
        gs_wb = st.session_state.get("gs_workbook", {})
        
        if upload_wb or gs_wb:
            # Merge them into the canonical workbook
            merged_wb = {}
            merged_wb.update(upload_wb)
            merged_wb.update(gs_wb)
            
            set_active_workbook(merged_wb)
            
            # Set a default current sheet if none is set
            if CURRENT_SHEET_KEY not in st.session_state and merged_wb:
                set_current_sheet(list(merged_wb.keys())[0])

def clear_workbook_state() -> None:
    """Clear all workbook-related state."""
    for key in [WORKBOOK_KEY, CURRENT_SHEET_KEY, WB_NONCE_KEY]:
        if key in st.session_state:
            del st.session_state[key]
    
    # Also clear legacy keys
    for key in ["upload_workbook", "gs_workbook", "workbook_source"]:
        if key in st.session_state:
            del st.session_state[key]

def get_cache_key(prefix: str = "") -> str:
    """Generate a cache key that includes the workbook nonce for automatic invalidation."""
    nonce = get_wb_nonce()
    sheet = get_current_sheet() or "none"
    return f"{prefix}_{sheet}_{nonce}"


def verify_active_workbook() -> Dict[str, Any]:
    """Return a report about the workbook's structure and active df status."""
    report: Dict[str, Any] = {}
    wb = st.session_state.get("workbook")
    sheet = st.session_state.get("current_sheet")
    report["has_workbook"] = isinstance(wb, dict)
    report["current_sheet"] = sheet
    if not isinstance(wb, dict):
        report["wb_type"] = type(wb).__name__
        report["keys"] = None
        report["problems"] = ["workbook is not a dict"]
        return report
    report["wb_type"] = "dict"
    keys = list(wb.keys())
    report["keys"] = keys

    problems: List[str] = []
    per_key: Dict[str, Dict[str, Any]] = {}
    for k in keys:
        val = wb.get(k)
        info = {"type": type(val).__name__, "is_df": isinstance(val, pd.DataFrame), "shape": None, "valid_headers": None}
        if isinstance(val, pd.DataFrame):
            info["shape"] = tuple(val.shape)
            try:
                from utils.helpers import validate_headers
                info["valid_headers"] = bool(validate_headers(val))
            except Exception:
                info["valid_headers"] = None
        else:
            problems.append(f"Sheet '{k}' is not a DataFrame (type: {type(val).__name__})")
        per_key[k] = info
    report["per_key"] = per_key

    if sheet not in wb:
        problems.append(f"current_sheet '{sheet}' not in workbook keys")
    else:
        active = wb.get(sheet)
        if not isinstance(active, pd.DataFrame):
            problems.append(f"active sheet '{sheet}' value is not a DataFrame (type: {type(active).__name__})")
        elif active.empty:
            problems.append(f"active sheet '{sheet}' is an empty DataFrame")

    report["problems"] = problems
    return report


def coerce_workbook_to_dataframes(raw_wb: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
    """Ensure each entry is a pandas DataFrame; drop invalid/empty entries."""
    clean: Dict[str, pd.DataFrame] = {}
    for name, val in (raw_wb or {}).items():
        if isinstance(val, pd.DataFrame):
            if not val.empty:
                clean[name] = val
            continue
        try:
            df = pd.DataFrame(val)
            if not df.empty:
                clean[name] = df
        except Exception:
            # ignore non-coercible
            pass
    return clean


def assert_workbook_integrity() -> None:
    """Assert that the current workbook contains only DataFrames."""
    wb = st.session_state.get("workbook")
    if wb is not None:
        assert isinstance(wb, dict), "workbook is not a dict"
        assert all(isinstance(v, pd.DataFrame) for v in wb.values()), "workbook contains non-DataFrame entries"
