# logic_actions.py

"""
Logic functions for Actions functionality.

This module contains pure functions for actions data processing,
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


def filter_actions_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter DataFrame to show only rows relevant for actions.
    
    Args:
        df: Input DataFrame with canonical headers
        
    Returns:
        DataFrame filtered for actions view
    """
    if df is None or df.empty:
        return pd.DataFrame()
    
    if not validate_headers(df):
        return pd.DataFrame()
    
    # Select relevant columns
    actions_columns = ["Vital Measurement"] + LEVEL_COLS + ["Actions"]
    actions_df = df[actions_columns].copy()
    
    # Filter out completely empty rows
    actions_df = actions_df.dropna(subset=["Vital Measurement"] + LEVEL_COLS, how="all")
    
    return actions_df


def compute_actions_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute comprehensive actions metrics.
    
    Args:
        df: DataFrame with actions data
        
    Returns:
        Dictionary containing actions metrics
    """
    if df is None or df.empty:
        return {
            "total_rows": 0,
            "actions_rows": 0,
            "coverage_pct": 0.0,
            "remaining": 0,
            "action_types": {},
            "action_categories": {},
            "missing_actions": []
        }
    
    total_rows = len(df)
    
    # Count rows with actions
    actions_mask = df["Actions"].notna() & (df["Actions"].astype(str).str.strip() != "")
    actions_rows = actions_mask.sum()
    
    # Calculate coverage
    coverage_pct = (actions_rows / total_rows * 100) if total_rows > 0 else 0
    remaining = total_rows - actions_rows
    
    # Analyze action types
    action_types = _analyze_action_types(df)
    
    # Analyze action categories
    action_categories = _analyze_action_categories(df)
    
    # Find rows missing actions
    missing_actions = _find_missing_actions(df)
    
    return {
        "total_rows": total_rows,
        "actions_rows": actions_rows,
        "coverage_pct": coverage_pct,
        "remaining": remaining,
        "action_types": action_types,
        "action_categories": action_categories,
        "missing_actions": missing_actions
    }


def validate_actions_data(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """
    Validate actions data for completeness and consistency.
    
    Args:
        df: DataFrame with actions data
        
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
    
    if "Actions" not in df.columns:
        errors.append("Missing 'Actions' column")
        return False, errors
    
    # Check for rows with vital measurements but no actions
    vital_filled = df["Vital Measurement"].notna() & (df["Vital Measurement"].astype(str).str.strip() != "")
    actions_empty = df["Actions"].isna() | (df["Actions"].astype(str).str.strip() == "")
    missing_actions = vital_filled & actions_empty
    
    if missing_actions.sum() > 0:
        errors.append(f"{missing_actions.sum()} rows have vital measurements but no actions")
    
    # Check for invalid actions
    invalid_actions = _check_invalid_actions(df)
    if invalid_actions:
        errors.extend(invalid_actions)
    
    return len(errors) == 0, errors


def _analyze_action_types(df: pd.DataFrame) -> Dict[str, int]:
    """
    Analyze the types of actions present in the data.
    """
    action_types = {}
    
    for _, row in df.iterrows():
        actions = normalize_text(row.get("Actions", ""))
        if actions:
            # Split by common delimiters and count types
            action_list = [a.strip() for a in actions.replace(";", ",").replace("|", ",").split(",") if a.strip()]
            for action in action_list:
                # Extract action type (first word or common prefixes)
                action_type = _extract_action_type(action)
                action_types[action_type] = action_types.get(action_type, 0) + 1
    
    return dict(sorted(action_types.items(), key=lambda x: x[1], reverse=True))


def _analyze_action_categories(df: pd.DataFrame) -> Dict[str, int]:
    """
    Analyze actions by category (diagnostic, treatment, follow-up, etc.).
    """
    categories = {
        "Diagnostic": 0,
        "Treatment": 0,
        "Follow-up": 0,
        "Referral": 0,
        "Monitoring": 0,
        "Other": 0
    }
    
    for _, row in df.iterrows():
        actions = normalize_text(row.get("Actions", ""))
        if actions:
            action_list = [a.strip() for a in actions.replace(";", ",").replace("|", ",").split(",") if a.strip()]
            for action in action_list:
                category = _classify_action_category(action)
                categories[category] = categories.get(category, 0) + 1
    
    return categories


def _find_missing_actions(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Find rows that are missing actions.
    """
    missing = []
    
    for idx, row in df.iterrows():
        vital = normalize_text(row.get("Vital Measurement", ""))
        actions = normalize_text(row.get("Actions", ""))
        
        if vital and not actions:
            missing.append({
                "row_index": idx,
                "vital_measurement": vital,
                "node_path": [normalize_text(row.get(col, "")) for col in LEVEL_COLS if normalize_text(row.get(col, ""))]
            })
    
    return missing


def _extract_action_type(action_text: str) -> str:
    """
    Extract action type from action text.
    """
    action_lower = action_text.lower()
    
    # Common action patterns
    if any(word in action_lower for word in ["refer", "referral", "consult"]):
        return "Refer"
    elif any(word in action_lower for word in ["order", "test", "lab", "imaging"]):
        return "Order"
    elif any(word in action_lower for word in ["monitor", "observe", "watch"]):
        return "Monitor"
    elif any(word in action_lower for word in ["treat", "medication", "therapy"]):
        return "Treat"
    elif any(word in action_lower for word in ["follow", "follow-up", "recheck"]):
        return "Follow-up"
    elif any(word in action_lower for word in ["admit", "hospitalize"]):
        return "Admit"
    elif any(word in action_lower for word in ["discharge", "send home"]):
        return "Discharge"
    else:
        return "Other"


def _classify_action_category(action_text: str) -> str:
    """
    Classify action into a category.
    """
    action_lower = action_text.lower()
    
    if any(word in action_lower for word in ["order", "test", "lab", "imaging", "x-ray", "ct", "mri", "ultrasound"]):
        return "Diagnostic"
    elif any(word in action_lower for word in ["treat", "medication", "therapy", "prescribe", "administer"]):
        return "Treatment"
    elif any(word in action_lower for word in ["follow", "follow-up", "recheck", "review"]):
        return "Follow-up"
    elif any(word in action_lower for word in ["refer", "referral", "consult", "specialist"]):
        return "Referral"
    elif any(word in action_lower for word in ["monitor", "observe", "watch", "track"]):
        return "Monitoring"
    else:
        return "Other"


def _check_invalid_actions(df: pd.DataFrame) -> List[str]:
    """
    Check for invalid or inconsistent actions.
    """
    errors = []
    
    for idx, row in df.iterrows():
        actions = normalize_text(row.get("Actions", ""))
        if actions:
            # Check for very short or unclear actions
            if len(actions) < 2:
                errors.append(f"Row {idx + 1}: Action too short: '{actions}'")
            
            # Check for common typos or invalid formats
            if actions.lower() in ["n/a", "na", "none", "null", "undefined"]:
                errors.append(f"Row {idx + 1}: Invalid action: '{actions}'")
            
            # Check for actions that are too generic
            if actions.lower() in ["action", "do something", "treat", "test"]:
                errors.append(f"Row {idx + 1}: Action too generic: '{actions}'")
    
    return errors


def suggest_actions(vital_measurement: str, node_path: List[str]) -> List[str]:
    """
    Suggest actions based on vital measurement and node path.
    
    Args:
        vital_measurement: The vital measurement being assessed
        node_path: List of node values in the decision path
        
    Returns:
        List of suggested action strings
    """
    suggestions = []
    
    vital_lower = vital_measurement.lower()
    path_lower = " ".join(node_path).lower()
    
    # Diagnostic suggestions
    if any(word in vital_lower for word in ["pain", "chest", "abdominal"]):
        suggestions.append("Order imaging studies")
        suggestions.append("Refer to specialist")
    
    if any(word in vital_lower for word in ["fever", "infection"]):
        suggestions.append("Order blood tests")
        suggestions.append("Prescribe antibiotics")
    
    if any(word in vital_lower for word in ["blood pressure", "heart rate"]):
        suggestions.append("Monitor vital signs")
        suggestions.append("Order cardiac workup")
    
    # Treatment suggestions based on severity
    if any(word in path_lower for word in ["severe", "critical", "acute"]):
        suggestions.append("Admit to hospital")
        suggestions.append("Immediate intervention")
    
    if any(word in path_lower for word in ["mild", "stable"]):
        suggestions.append("Outpatient follow-up")
        suggestions.append("Conservative management")
    
    # Default suggestions
    if not suggestions:
        suggestions.append("Monitor patient")
        suggestions.append("Follow-up in clinic")
    
    return suggestions


def validate_action_format(action_text: str) -> Tuple[bool, str]:
    """
    Validate the format of a single action.
    
    Args:
        action_text: The action text to validate
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not action_text or not action_text.strip():
        return False, "Action cannot be empty"
    
    if len(action_text.strip()) < 3:
        return False, "Action must be at least 3 characters long"
    
    if action_text.lower() in ["n/a", "na", "none", "null", "undefined"]:
        return False, "Invalid action format"
    
    return True, ""
