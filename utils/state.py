# utils/state.py
"""
Unified state management for workbook and sheet selection.
All tabs should use these helpers to access the active workbook state.
"""

import streamlit as st
import pandas as pd
from typing import Dict, Optional, Any, List
from uuid import uuid4

# Legacy constants (kept for compatibility)
WORKBOOK_KEY = "workbook"          # Dict[str, pd.DataFrame]
CURRENT_SHEET_KEY = "current_sheet"  # str
WB_NONCE_KEY = "wb_nonce"          # random string for cache busting

def get_sheet_names() -> list[str]:
    """Get list of available sheet names from active workbook."""
    wb = st.session_state.get("workbook") or {}
    if isinstance(wb, dict):
        return list(wb.keys())
    return []

def ensure_current_sheet() -> str | None:
    """
    Ensure there is a valid current sheet name in session.
    Prefers st.session_state['current_sheet'], then 'sheet_name', then first sheet in workbook.
    Returns the chosen sheet name or None.
    """
    wb = st.session_state.get("workbook") or {}
    names = list(wb.keys()) if isinstance(wb, dict) else []

    # Already set & valid?
    cur = st.session_state.get("current_sheet")
    if cur and cur in names:
        return cur

    # Fallback to 'sheet_name'
    sn = st.session_state.get("sheet_name")
    if sn and sn in names:
        st.session_state["current_sheet"] = sn
        return sn

    # Fallback to first available
    if names:
        st.session_state["current_sheet"] = names[0]
        return names[0]

    # Nothing available
    st.session_state["current_sheet"] = None
    return None

def _wb_dict():
    """Prefer non-empty 'workbook', else non-empty 'gs_workbook', else None."""
    wb = st.session_state.get("workbook")
    if isinstance(wb, dict) and wb:
        return wb
    wb = st.session_state.get("gs_workbook")
    if isinstance(wb, dict) and wb:
        return wb
    return None

def set_active_workbook(wb: dict, default_sheet: str | None = None, source: str = "unspecified"):
    """Set the active workbook in session state with smart sheet selection."""
    st.session_state["workbook"] = wb
    # If current_sheet invalid or missing, pick default or the first sheet
    curr = st.session_state.get("current_sheet")
    if not curr or curr not in wb:
        chosen = default_sheet if default_sheet and default_sheet in wb else (next(iter(wb.keys())) if wb else None)
        if chosen:
            st.session_state["current_sheet"] = chosen
            st.session_state["sheet_name"] = chosen  # keep in sync
    # bump nonce to invalidate caches if needed
    st.session_state["wb_nonce"] = (st.session_state.get("wb_nonce") or "") + "•"

def get_active_workbook():
    """Get the active workbook from session state with fallback to legacy keys."""
    return st.session_state.get("workbook") or st.session_state.get("gs_workbook")

def set_current_sheet(name: str):
    """Set current sheet and keep sheet_name in sync."""
    st.session_state["current_sheet"] = name
    st.session_state["sheet_name"] = name

def ensure_active_sheet(default: str | None = None, source: str = "ensure_active_sheet"):
    """
    Guarantee a valid current_sheet in session_state if a workbook exists.
    Preference: default -> current_sheet -> sheet_name -> first key.
    """
    wb = _wb_dict()
    if not wb:
        return None
    keys = list(wb.keys())
    if not keys:
        return None

    pick = None
    if default in keys:
        pick = default
    elif st.session_state.get("current_sheet") in keys:
        pick = st.session_state["current_sheet"]
    elif st.session_state.get("sheet_name") in keys:
        pick = st.session_state["sheet_name"]
    else:
        pick = keys[0]

    st.session_state["current_sheet"] = pick
    st.session_state["sheet_name"] = pick
    return pick


def get_current_sheet():
    """Get current sheet with auto-repair fallback."""
    cs = st.session_state.get("current_sheet")
    if cs:
        return cs
    # fall back and try to set one
    return ensure_active_sheet()

def get_active_workbook_safe():
    """
    Return (wb, status, detail) where status in:
      - 'ok'           (workbook exists and has sheets)
      - 'no_wb'        (no workbook)
      - 'empty_wb'     (workbook exists but is empty)
    """
    wb = _wb_dict()
    if not wb:
        return None, "no_wb", "No workbook in session_state"
    
    if not wb:  # This checks if the dict is empty
        return None, "empty_wb", "Workbook exists but has no sheets"
    
    return wb, "ok", f"Workbook with {len(wb)} sheet(s)"

def get_active_df_safe():
    """
    Return (df, status, detail) where status in:
      - 'ok'
      - 'no_wb'        (no workbook)
      - 'no_sheet'     (no current sheet)
      - 'sheet_missing'(current sheet not in wb)
      - 'not_df'       (object is not a DataFrame)
    """
    wb = _wb_dict()
    if not wb:
        return None, "no_wb", "No workbook in session_state"

    sheet = st.session_state.get("current_sheet") or st.session_state.get("sheet_name")
    if not sheet:
        sheet = ensure_active_sheet()
    if not sheet:
        return None, "no_sheet", "Could not determine a sheet name"

    if sheet not in wb:
        # try to heal once
        healed = ensure_active_sheet(default=st.session_state.get("sheet_name"))
        if not healed:
            return None, "sheet_missing", f"'{sheet}' not in workbook keys={list(wb.keys())[:5]}..."
        sheet = healed

    df = wb.get(sheet)
    import pandas as pd
    if not isinstance(df, pd.DataFrame):
        return None, "not_df", f"Value at wb['{sheet}'] is {type(df).__name__}"

    return df, "ok", sheet

def get_wb_nonce() -> str:
    """Get the current workbook nonce for cache keys."""
    return st.session_state.get("wb_nonce", "")

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

def get_workbook_status():
    """Get workbook status for UI display."""
    wb = get_active_workbook() or {}
    current_sheet = get_current_sheet()
    return (bool(wb), len(wb), current_sheet)

def migrate_legacy_state() -> None:
    """Migrate legacy state keys to the new unified format.
    This should be called during app initialization.
    """
    # Check if we need to migrate from old state
    if "workbook" not in st.session_state:
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
            if "current_sheet" not in st.session_state and merged_wb:
                set_current_sheet(list(merged_wb.keys())[0])

def clear_workbook_state() -> None:
    """Clear all workbook-related state."""
    for key in ["workbook", "current_sheet", "wb_nonce"]:
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
    wb = get_active_workbook()
    if wb is not None:
        assert isinstance(wb, dict), "workbook is not a dict"
        assert all(isinstance(v, pd.DataFrame) for v in wb.values()), "workbook contains non-DataFrame entries"
