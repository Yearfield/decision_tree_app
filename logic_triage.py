# logic_triage.py

"""
Logic functions for Diagnostic Triage functionality.

This module contains pure functions for triage data processing,
validation, and metrics computation. It is intentionally Streamlit-free
so it can be imported by both UI modules and other logic modules.
"""

from typing import Dict, List, Tuple, Optional, Set, Any
import pandas as pd
import numpy as np

from utils import (
    CANON_HEADERS, LEVEL_COLS, MAX_LEVELS,
    normalize_text, validate_headers,
)


def filter_triage_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter DataFrame to show only rows relevant for triage.
    
    Args:
        df: Input DataFrame with canonical headers
        
    Returns:
        DataFrame filtered for triage view
        
    Failure modes:
        - Returns empty DataFrame for non-DataFrame inputs
        - Returns empty DataFrame for empty DataFrames
        - Returns empty DataFrame if required columns missing
        - Returns empty DataFrame if processing fails
    """
    try:
        if not isinstance(df, pd.DataFrame) or df.empty:
            return pd.DataFrame()
        
        if not validate_headers(df):
            return pd.DataFrame()
        
        # Select relevant columns
        triage_columns = ["Vital Measurement"] + LEVEL_COLS + ["Diagnostic Triage"]
        # Check if required columns exist
        missing_cols = [col for col in triage_columns if col not in df.columns]
        if missing_cols:
            return pd.DataFrame()
        
        triage_df = df[triage_columns].copy()
        
        # Filter out completely empty rows
        triage_df = triage_df.dropna(subset=["Vital Measurement"] + LEVEL_COLS, how="all")
        
        return triage_df
    except Exception:
        return pd.DataFrame()


def compute_triage_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute comprehensive triage metrics.
    
    Args:
        df: DataFrame with triage data
        
    Returns:
        Dictionary containing triage metrics
        
    Failure modes:
        - Returns empty metrics structure for non-DataFrame inputs
        - Returns empty metrics structure for empty DataFrames
        - Returns empty metrics structure if required columns missing
        - Returns empty metrics structure if processing fails
    """
    try:
        if not isinstance(df, pd.DataFrame) or df.empty:
            return {
                "total_rows": 0,
                "triaged_rows": 0,
                "coverage_pct": 0.0,
                "remaining": 0,
                "priority_breakdown": {},
                "urgency_levels": {}
            }
        
        # Check if Diagnostic Triage column exists
        if "Diagnostic Triage" not in df.columns:
            return {
                "total_rows": len(df),
                "triaged_rows": 0,
                "coverage_pct": 0.0,
                "remaining": len(df),
                "priority_breakdown": {},
                "urgency_levels": {}
            }
        
        total_rows = len(df)
        
        # Count triaged rows
        triaged_mask = df["Diagnostic Triage"].notna() & (df["Diagnostic Triage"].astype(str).str.strip() != "")
        triaged_rows = triaged_mask.sum()
        
        # Calculate coverage
        coverage_pct = (triaged_rows / total_rows * 100) if total_rows > 0 else 0
        remaining = total_rows - triaged_rows
        
        # Analyze priority breakdown
        priority_breakdown = _analyze_triage_priorities(df)
        
        # Analyze urgency levels
        urgency_levels = _analyze_urgency_levels(df)
        
        return {
            "total_rows": total_rows,
            "triaged_rows": triaged_rows,
            "coverage_pct": coverage_pct,
            "remaining": remaining,
            "priority_breakdown": priority_breakdown,
            "urgency_levels": urgency_levels
        }
    except Exception:
        return {
            "total_rows": 0,
            "triaged_rows": 0,
            "coverage_pct": 0.0,
            "remaining": 0,
            "priority_breakdown": {},
            "urgency_levels": {}
        }


def validate_triage_data(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """
    Validate triage data for completeness and consistency.
    
    Args:
        df: DataFrame with triage data
        
    Returns:
        Tuple of (is_valid, list_of_validation_messages)
    """
    errors = []
    
    if df is None or df.empty:
        errors.append("No data provided for validation")
        return False, errors
    
    if not validate_headers(df):
        errors.append("Invalid data format - missing canonical headers")
        return False, errors
    
    if "Diagnostic Triage" not in df.columns:
        errors.append("Missing 'Diagnostic Triage' column")
        return False, errors
    
    # Check for rows with vital measurements but no triage
    vital_filled = df["Vital Measurement"].notna() & (df["Vital Measurement"].astype(str).str.strip() != "")
    triage_empty = df["Diagnostic Triage"].isna() | (df["Diagnostic Triage"].astype(str).str.strip() == "")
    missing_triage = vital_filled & triage_empty
    
    if missing_triage.sum() > 0:
        errors.append(f"{missing_triage.sum()} rows have vital measurements but no triage priority")
    
    # Check for invalid triage priorities
    invalid_priorities = _check_invalid_triage_priorities(df)
    if invalid_priorities:
        errors.extend(invalid_priorities)
    
    return len(errors) == 0, errors


def _analyze_triage_priorities(df: pd.DataFrame) -> Dict[str, int]:
    """
    Analyze the distribution of triage priorities.
    """
    priorities = {}
    
    for _, row in df.iterrows():
        triage = normalize_text(row.get("Diagnostic Triage", ""))
        if triage:
            # Extract priority level (first word or common patterns)
            priority = _extract_priority_level(triage)
            priorities[priority] = priorities.get(priority, 0) + 1
    
    return dict(sorted(priorities.items(), key=lambda x: x[1], reverse=True))


def _analyze_urgency_levels(df: pd.DataFrame) -> Dict[str, int]:
    """
    Analyze urgency levels in triage data.
    """
    urgency_levels = {
        "Urgent": 0,
        "High": 0,
        "Medium": 0,
        "Low": 0,
        "Routine": 0,
        "Other": 0
    }
    
    for _, row in df.iterrows():
        triage = normalize_text(row.get("Diagnostic Triage", ""))
        if triage:
            urgency = _classify_urgency_level(triage)
            urgency_levels[urgency] = urgency_levels.get(urgency, 0) + 1
    
    return urgency_levels


def _extract_priority_level(triage_text: str) -> str:
    """
    Extract priority level from triage text.
    """
    triage_lower = triage_text.lower()
    
    if any(word in triage_lower for word in ["urgent", "emergency", "critical"]):
        return "Urgent"
    elif any(word in triage_lower for word in ["high", "priority", "immediate"]):
        return "High Priority"
    elif any(word in triage_lower for word in ["medium", "moderate"]):
        return "Medium Priority"
    elif any(word in triage_lower for word in ["low", "routine", "non-urgent"]):
        return "Low Priority"
    else:
        return "Other"


def _classify_urgency_level(triage_text: str) -> str:
    """
    Classify urgency level from triage text.
    """
    triage_lower = triage_text.lower()
    
    if any(word in triage_lower for word in ["urgent", "emergency", "critical"]):
        return "Urgent"
    elif any(word in triage_lower for word in ["high", "priority", "immediate"]):
        return "High"
    elif any(word in triage_lower for word in ["medium", "moderate"]):
        return "Medium"
    elif any(word in triage_lower for word in ["low", "routine", "non-urgent"]):
        return "Low"
    elif any(word in triage_lower for word in ["routine", "standard"]):
        return "Routine"
    else:
        return "Other"


def _check_invalid_triage_priorities(df: pd.DataFrame) -> List[str]:
    """
    Check for invalid or inconsistent triage priorities.
    """
    errors = []
    
    for idx, row in df.iterrows():
        triage = normalize_text(row.get("Diagnostic Triage", ""))
        if triage:
            # Check for very short or unclear priorities
            if len(triage) < 2:
                errors.append(f"Row {idx + 1}: Triage priority too short: '{triage}'")
            
            # Check for common typos or invalid formats
            if triage.lower() in ["n/a", "na", "none", "null", "undefined"]:
                errors.append(f"Row {idx + 1}: Invalid triage priority: '{triage}'")
    
    return errors


def suggest_triage_priority(vital_measurement: str, node_path: List[str]) -> str:
    """
    Suggest a triage priority based on vital measurement and node path.
    
    Args:
        vital_measurement: The vital measurement being assessed
        node_path: List of node values in the decision path
        
    Returns:
        Suggested triage priority string
    """
    # This is a simplified suggestion algorithm
    # In a real implementation, this could use ML models or complex rules
    
    vital_lower = vital_measurement.lower()
    path_lower = " ".join(node_path).lower()
    
    # High urgency indicators
    if any(word in vital_lower for word in ["pain", "chest", "breathing", "consciousness"]):
        if any(word in path_lower for word in ["severe", "acute", "critical"]):
            return "Urgent"
        else:
            return "High Priority"
    
    # Medium urgency indicators
    if any(word in vital_lower for word in ["fever", "blood pressure", "heart rate"]):
        if any(word in path_lower for word in ["elevated", "abnormal", "concerning"]):
            return "Medium Priority"
        else:
            return "Low Priority"
    
    # Default suggestion
    return "Routine"
