# ui/utils/guards.py
"""
Guard utilities for tab rendering to prevent blank tabs and provide clear error messages.
"""

import streamlit as st
from typing import Tuple, Optional
import utils.state as USTATE
from utils import validate_headers


def ensure_active_workbook_and_sheet(tab_name: str) -> Tuple[bool, Optional["pd.DataFrame"]]:
    """
    Ensure active workbook and sheet are available for tab rendering.
    
    Args:
        tab_name: Name of the tab for logging purposes
        
    Returns:
        Tuple of (success, DataFrame) where success is True if all guards pass
    """
    st.caption(f"🚦 DISPATCH {tab_name} at runtime")
    try:
        wb = USTATE.get_active_workbook()
        sheet = USTATE.get_current_sheet()
        if not wb:
            st.warning("No active workbook. Load a workbook in 📂 Source.")
            return False, None
        if not sheet:
            st.warning("No current sheet selected. Use 🗂 Workspace Selection to choose one.")
            return False, None
        df = USTATE.get_active_df()
        if df is None:
            st.warning("Active DataFrame is unavailable.")
            return False, None
        if not validate_headers(df):
            st.warning("Active sheet has invalid headers. Please fix in 🔎 Validation.")
            return False, None
        return True, df
    except Exception as e:
        st.error(f"[{tab_name}] guard failed: {e}")
        st.exception(e)
        return False, None
