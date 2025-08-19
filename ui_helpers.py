# ui_helpers.py

import streamlit as st
from constants import EMOJI
from typing import Dict, Any, Callable, Optional


def render_kpis(metrics: Dict[str, Any], columns: int = 3) -> None:
    """
    Render KPIs in a consistent format with specified number of columns.
    
    Args:
        metrics: Dictionary with keys like "total", "filled", "coverage_pct"
        columns: Number of columns to use (default 3)
    """
    if not metrics:
        return
    
    col_list = st.columns(columns)
    
    # Map common metric keys to display names
    metric_mapping = {
        "total": "Total rows",
        "filled": "Filled",
        "coverage_pct": "Coverage %",
        "missing": "Missing",
        "remaining": "Remaining"
    }
    
    for i, (key, value) in enumerate(metrics.items()):
        if key in metric_mapping and i < len(col_list):
            with col_list[i]:
                if key == "coverage_pct":
                    st.metric(metric_mapping[key], f"{value}%")
                else:
                    st.metric(metric_mapping[key], value)


def render_progress_bar(pct: float, label: str = "Progress") -> None:
    """
    Render a progress bar with consistent styling.
    
    Args:
        pct: Percentage value (0.0 to 1.0)
        label: Optional label for the progress bar
    """
    if label:
        st.caption(label)
    st.progress(pct)


def render_save_controls(save_fn: Callable, download_data: bytes, filename: str, 
                        save_label: str = "💾 Save to session") -> None:
    """
    Render consistent save controls with download and save buttons.
    
    Args:
        save_fn: Function to call when save button is clicked
        download_data: Data to download (CSV bytes)
        filename: Filename for download
        save_label: Label for the save button
    """
    col1, col2 = st.columns([1, 1])
    
    with col1:
        if st.button(save_label, key=f"save_{filename}"):
            save_fn()
    
    with col2:
        st.download_button(
            "Download CSV",
            data=download_data,
            file_name=f"{filename}.csv",
            mime="text/csv"
        )


def render_preview_caption(df_preview: Any, df_full: Any, max_rows: int = 100) -> None:
    """
    Render consistent preview caption showing row limits.
    
    Args:
        df_preview: Preview DataFrame (limited rows)
        df_full: Full DataFrame
        max_rows: Maximum rows to show in preview
    """
    preview_count = len(df_preview) if hasattr(df_preview, '__len__') else 0
    full_count = len(df_full) if hasattr(df_full, '__len__') else 0
    
    if full_count <= max_rows:
        st.caption(f"Showing all {full_count} rows.")
    else:
        st.caption(f"Showing first {preview_count} of {full_count} rows")


def safe_dataframe_preview(df: Any, max_rows: int = 100, **kwargs) -> Any:
    """
    Safely preview DataFrame with consistent row limits.
    
    Args:
        df: DataFrame to preview
        max_rows: Maximum rows to show
        **kwargs: Additional arguments for st.dataframe
    
    Returns:
        Limited DataFrame for preview
    """
    if df is None or (hasattr(df, 'empty') and df.empty):
        return df
    
    if hasattr(df, '__len__') and len(df) > max_rows:
        return df.head(max_rows)
    
    return df


def standardize_message(message_type: str, text: str) -> str:
    """
    Standardize message formatting using EMOJI constants.
    
    Args:
        message_type: One of "success", "warning", "error", "info"
        text: Message text
    
    Returns:
        Formatted message with emoji
    """
    emoji = EMOJI.get(message_type, "")
    return f"{emoji} {text}" if emoji else text


def st_success(text: str) -> None:
    """Standardized success message."""
    st.success(standardize_message("success", text))


def st_warning(text: str) -> None:
    """Standardized warning message."""
    st.warning(standardize_message("warning", text))


def st_error(text: str) -> None:
    """Standardized error message."""
    st.error(standardize_message("error", text))


def st_info(text: str) -> None:
    """Standardized info message."""
    st.info(standardize_message("info", text))
