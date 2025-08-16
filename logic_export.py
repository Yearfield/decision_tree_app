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
CANON_HEADERS = [
    "Vital Measurement", "Node 1", "Node 2", "Node 3", "Node 4", "Node 5",
    "Diagnostic Triage", "Actions",
]
LEVEL_COLS = ["Node 1", "Node 2", "Node 3", "Node 4", "Node 5"]


# ===== Basic helpers =====
def normalize_text(x: str) -> str:
    if x is None:
        return ""
    return str(x).strip()


def validate_headers(df: pd.DataFrame) -> bool:
    return list(df.columns[: len(CANON_HEADERS)]) == CANON_HEADERS


def _require_streamlit_secrets() -> dict:
    """Fetch service account info from Streamlit secrets or raise a clear error."""
    if st is None:
        raise RuntimeError("Streamlit is required for Google Sheets functions.")
    if "gcp_service_account" not in st.secrets:
        raise RuntimeError(
            "Google Sheets not configured. Add your service account JSON under [gcp_service_account] in secrets."
        )
    return st.secrets["gcp_service_account"]


def _get_gsheet_client():
    """Authorize and return a gspread client using Streamlit secrets."""
    sa_info = _require_streamlit_secrets()

    # Lazy imports to avoid hard dependency when not needed
    import gspread  # type: ignore
    from google.oauth2.service_account import Credentials  # type: ignore

    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ]
    creds = Credentials.from_service_account_info(sa_info, scopes=scopes)
    client = gspread.authorize(creds)
    return client


# ===== Google Sheets: read / write / backup =====
def read_google_sheet(spreadsheet_id: str, sheet_name: str) -> pd.DataFrame:
    """
    Read a Google Sheet tab into a DataFrame with canonical headers.
    - Fills missing canonical columns
    - Drops rows where all Node columns and VM are blank
    """
    client = _get_gsheet_client()
    sh = client.open_by_key(spreadsheet_id)
    ws = sh.worksheet(sheet_name)
    values = ws.get_all_values()
    if not values:
        return pd.DataFrame(columns=CANON_HEADERS)

    header = [normalize_text(c) for c in values[0]]
    rows = values[1:]

    df = pd.DataFrame(rows, columns=header)
    # Ensure canonical headers exist
    for c in CANON_HEADERS:
        if c not in df.columns:
            df[c] = ""
    df = df[CANON_HEADERS]

    # Normalize text cells
    for c in CANON_HEADERS:
        df[c] = df[c].map(normalize_text)

    # Drop fully blank node paths (VM + Node1..5 empty)
    node_block = ["Vital Measurement"] + LEVEL_COLS
    mask_blank = df[node_block].apply(lambda r: all(normalize_text(v) == "" for v in r), axis=1)
    df = df[~mask_blank].copy()

    return df


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
        client = _get_gsheet_client()
        sh = client.open_by_key(spreadsheet_id)

        df = df.fillna("")
        headers = list(df.columns)
        values = [headers] + df.astype(str).values.tolist()

        n_rows = max(len(values), min_rows)
        n_cols = max(len(headers), min_cols)

        try:
            ws = sh.worksheet(sheet_name)
            ws.clear()
            ws.resize(rows=n_rows, cols=n_cols)
        except Exception:
            ws = sh.add_worksheet(title=sheet_name, rows=n_rows, cols=n_cols)

        ws.update("A1", values, value_input_option="RAW")
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
        client = _get_gsheet_client()
        sh = client.open_by_key(spreadsheet_id)

        try:
            ws = sh.worksheet(source_sheet)
        except Exception:
            return None

        values = ws.get_all_values()
        ts = datetime.now().strftime("%Y-%m-%d %H%M")
        backup_title_full = f"{source_sheet} (backup {ts})"
        backup_title = backup_title_full[:99]

        rows = max(len(values), 100)
        cols = max(len(values[0]) if values else 8, 8)

        ws_bak = sh.add_worksheet(title=backup_title, rows=rows, cols=cols)
        if values:
            ws_bak.update("A1", values, value_input_option="RAW")
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
