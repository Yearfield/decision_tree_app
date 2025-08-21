# ui/analysis.py
"""
Shared analysis module for cached decision tree analysis.
Used by both Conflicts and Workspace tabs.
"""

import pandas as pd
import streamlit as st
from typing import Dict, Any
from utils.state import get_wb_nonce
from logic.tree import analyze_decision_tree_with_root


@st.cache_data(ttl=600, show_spinner=False)
def get_conflict_summary_with_root(df: pd.DataFrame, nonce: str) -> Dict[str, Any]:
    """
    Get cached decision tree analysis summary for the given DataFrame and nonce.
    Uses the monolith-style analysis with root + five nodes.
    """
    return analyze_decision_tree_with_root(df)
