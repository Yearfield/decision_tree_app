"""
logic_export.py

I/O utilities for:
- Google Sheets (read, push, backup)
- Overrides JSON (export/import with merge modes)
- DataFrame exporters (CSV / Excel bytes)
- Push log helpers

This module assumes it is used inside a Streamlit app and will read
GCP service account credentials from st.secrets["gcp_service_account"].
"""

# TODO Step 9: Implement branch grouping so Node 1 children are displayed/exported consecutively.

# TODO[Step10]: Implement export audit logging (sheet name, row count, timestamp, app version).
# Consider a helper: audit_export_event(details: dict) -> None

from __future__ import annotations

import io
import json
from datetime import datetime
from typing import Dict, Optional, Tuple

import pandas as pd

# Streamlit is optional at import-time; functions that need it import lazily.
try:
    import streamlit as st  # type: ignore
except Exception:  # pragma: no cover
    st = None  # allows importing in non-streamlit contexts


# ===== Canonical schema =====
from constants import CANON_HEADERS, LEVEL_COLS


# ===== Basic helpers =====
def normalize_text(x: str) -> str:
    if x is None:
        return ""
    return str(x).strip()


def validate_headers(df: pd.DataFrame) -> bool:
    return list(df.columns[: len(CANON_HEADERS)]) == CANON_HEADERS





# ===== Google Sheets: read / write / backup =====
def read_google_sheet(spreadsheet_id: str, sheet_name: str) -> pd.DataFrame:
    """
    Read a Google Sheet tab into a DataFrame with canonical headers.
    - Fills missing canonical columns
    - Drops rows where all Node columns and VM are blank
    """
    try:
        from app.sheets import get_gspread_client_from_secrets, open_spreadsheet, read_worksheet_with_canonical_headers
        
        # Get client and open spreadsheet
        client = get_gspread_client_from_secrets(st.secrets["gcp_service_account"])
        spreadsheet = open_spreadsheet(client, spreadsheet_id)
        
        # Read with canonical headers
        df = read_worksheet_with_canonical_headers(spreadsheet, sheet_name, CANON_HEADERS)
        
        # Drop fully blank node paths (VM + Node1..5 empty)
        node_block = ["Vital Measurement"] + LEVEL_COLS
        mask_blank = df[node_block].apply(lambda r: all(normalize_text(v) == "" for v in r), axis=1)
        df = df[~mask_blank].copy()
        
        return df
        
    except Exception as e:
        if st:
            st.error(f"Error reading Google Sheet: {e}")
        return pd.DataFrame(columns=CANON_HEADERS)


def push_to_google_sheets(
    spreadsheet_id: str,
    sheet_name: str,
    df: pd.DataFrame,
    *,
    min_rows: int = 200,
    min_cols: int = 8,
) -> bool:
    """
    Overwrite a Google Sheet tab with the given DataFrame.
    - Resizes the sheet before update (prevents tail remnants)
    - Writes headers + all rows
    """
    try:
        from app.sheets import get_gspread_client_from_secrets, open_spreadsheet, write_dataframe
        
        # Get client and open spreadsheet
        client = get_gspread_client_from_secrets(st.secrets["gcp_service_account"])
        spreadsheet = open_spreadsheet(client, spreadsheet_id)
        
        # Prepare DataFrame for writing
        df = df.fillna("")
        
        # Write data using the new helper
        write_dataframe(spreadsheet, sheet_name, df, mode="overwrite")
        
        return True

    except Exception as e:
        if st:
            st.error(f"Push to Google Sheets failed: {e}")
        return False


def backup_sheet_copy(spreadsheet_id: str, source_sheet: str) -> Optional[str]:
    """
    Create a backup copy of `source_sheet` within the same spreadsheet.
    Returns the created backup sheet name, or None on failure.
    """
    try:
        from app.sheets import get_gspread_client_from_secrets, open_spreadsheet, backup_worksheet
        
        # Get client and open spreadsheet
        client = get_gspread_client_from_secrets(st.secrets["gcp_service_account"])
        spreadsheet = open_spreadsheet(client, spreadsheet_id)
        
        # Create backup using the new helper
        backup_title = backup_worksheet(spreadsheet, source_sheet)
        return backup_title

    except Exception as e:
        if st:
            st.error(f"Backup failed: {e}")
        return None


# ===== Overrides JSON: export / import =====
def export_overrides_json(overrides_sheet: Dict[str, list]) -> bytes:
    """
    Serialize a single-sheet overrides dict to pretty JSON bytes.
    """
    return json.dumps(overrides_sheet or {}, indent=2).encode("utf-8")


def import_overrides_json(
    existing_overrides_sheet: Dict[str, list],
    imported_json_bytes: bytes,
    mode: str = "replace",
) -> Dict[str, list]:
    """
    Merge/replace a sheet's overrides with imported JSON.

    mode options:
      - "replace":       imported replaces existing entirely
      - "merge_import":  merge, preferring imported on conflicts
      - "merge_existing":merge, preferring existing on conflicts
    """
    try:
        imported = json.loads(imported_json_bytes.decode("utf-8"))
    except Exception as e:
        raise ValueError(f"Invalid JSON: {e}")

    if not isinstance(imported, dict):
        raise ValueError("Invalid JSON: expected an object mapping keys -> list-of-children")

    # Normalize child lists to lists of strings (trim to 5; pad with blanks)
    def _k5(vals):
        vals = vals if isinstance(vals, list) else [vals]
        vals = [normalize_text(x) for x in vals if normalize_text(x) != ""]
        if len(vals) > 5:
            vals = vals[:5]
        elif len(vals) < 5:
            vals = vals + [""] * (5 - len(vals))
        return vals

    imported_norm = {k: _k5(v) for k, v in imported.items()}

    existing = dict(existing_overrides_sheet or {})

    if mode == "replace":
        return imported_norm

    if mode == "merge_import":
        merged = existing.copy()
        merged.update(imported_norm)  # imported wins
        return merged

    if mode == "merge_existing":
        merged = imported_norm.copy()
        merged.update(existing)  # existing wins
        return merged

    raise ValueError(f"Unknown import mode: {mode}")


# ===== DataFrame exporters =====
def export_dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def export_dataframe_to_excel_bytes(df: pd.DataFrame, sheet_name: str = "Sheet1") -> bytes:
    """
    Serialize DataFrame to an .xlsx workbook (single sheet) and return bytes.
    """
    buffer = io.BytesIO()
    # Lazy import to avoid hard dependency at import-time
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:  # type: ignore
        df.to_excel(writer, index=False, sheet_name=(sheet_name or "Sheet1")[:31])
    return buffer.getvalue()


# ===== Push log helper =====
def make_push_log_entry(
    *,
    sheet: str,
    target_tab: str,
    spreadsheet_id: str,
    rows_written: int,
    new_rows_added: int = 0,
    scope: str = "all",
    extra: Optional[Dict[str, str]] = None,
) -> Dict[str, str]:
    """
    Create a standardized push-log row (dict) you can append to a session log list.
    """
    row = {
        "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "sheet": sheet,
        "target_tab": target_tab,
        "spreadsheet_id": spreadsheet_id,
        "rows_written": str(rows_written),
        "new_rows_added": str(new_rows_added),
        "scope": scope,
    }
    if extra:
        row.update({str(k): str(v) for k, v in extra.items()})
    return row
