"""
Safe rerun abstraction for Streamlit compatibility across versions.

This module provides a single safe_rerun() function that works across
different Streamlit versions by trying multiple rerun methods.
"""

import streamlit as st


def safe_rerun():
    """
    Safely trigger a Streamlit rerun across different versions.
    
    Streamlit 1.27+ uses experimental_rerun; older builds might still have rerun().
    This function tries multiple approaches to ensure compatibility.
    """
    try:
        # Preferred method for Streamlit 1.27+
        st.experimental_rerun()
    except AttributeError:
        # Fallback for very old versions
        try:
            st.rerun()
        except Exception:
            # Last resort: trigger a no-op state change to force a re-run
            st.session_state["__force_rerun_nonce"] = st.session_state.get("__force_rerun_nonce", 0) + 1
