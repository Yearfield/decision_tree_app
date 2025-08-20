# utils/helpers.py
from typing import Dict, List, Tuple, Optional, Set
import numpy as np
import pandas as pd
from .constants import CANON_HEADERS, LEVEL_COLS, MAX_LEVELS

def normalize_text(x) -> str:
    """Return a stripped string, converting NaN/None to ""."""
    try:
        if x is None:
            return ""
        if isinstance(x, float) and np.isnan(x):
            return ""
    except Exception:
        return ""
    return str(x).strip()

def validate_headers(df: pd.DataFrame) -> bool:
    """Verify that the first len(CANON_HEADERS) columns match the canonical schema."""
    try:
        if not isinstance(df, pd.DataFrame) or df.empty:
            return False
        if len(df.columns) < len(CANON_HEADERS):
            return False
        return list(df.columns[:len(CANON_HEADERS)]) == CANON_HEADERS
    except Exception:
        return False

def ensure_canon_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure the DataFrame has all canonical columns (creates missing as "") and in order."""
    try:
        if not isinstance(df, pd.DataFrame):
            return pd.DataFrame(columns=CANON_HEADERS)
        
        df2 = df.copy()
        for c in CANON_HEADERS:
            if c not in df2.columns:
                df2[c] = ""
        return df2[CANON_HEADERS]
    except Exception:
        return pd.DataFrame(columns=CANON_HEADERS)

def drop_fully_blank_paths(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows where Vital Measurement + Node1..Node5 are all blank."""
    try:
        if not isinstance(df, pd.DataFrame) or df.empty:
            return pd.DataFrame()
        
        node_block = ["Vital Measurement"] + LEVEL_COLS
        # Check if required columns exist
        missing_cols = [col for col in node_block if col not in df.columns]
        if missing_cols:
            return df.copy()
        
        mask_blank = df[node_block].apply(lambda r: all(normalize_text(v) == "" for v in r), axis=1)
        return df[~mask_blank].copy()
    except Exception:
        return df.copy() if isinstance(df, pd.DataFrame) else pd.DataFrame()

def level_key_tuple(level: int, parent: Tuple[str, ...]) -> str:
    """Build the canonical store key for a parent tuple at a given level."""
    return f"L{level}|" + (">".join(parent) if parent else "<ROOT>")

def parent_key_from_row_strict(row: pd.Series, upto_level: int) -> Optional[Tuple[str, ...]]:
    """For a row and a target 'upto_level', return the parent tuple of length (upto_level-1)."""
    try:
        if upto_level <= 1:
            return tuple()
        parent: List[str] = []
        for c in LEVEL_COLS[:upto_level-1]:
            v = normalize_text(row.get(c, ""))
            if v == "":
                return None
            parent.append(v)
        return tuple(parent)
    except Exception:
        return None

def enforce_k_five(opts: List[str]) -> List[str]:
    """Trim/pad a list of options to exactly 5, removing blanks and normalizing text."""
    clean = [normalize_text(o) for o in (opts or []) if normalize_text(o) != ""]
    if len(clean) > 5:
        clean = clean[:5]
    elif len(clean) < 5:
        clean = clean + [""]*(5-len(clean))
    return clean
