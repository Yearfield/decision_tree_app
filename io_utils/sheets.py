# io/sheets.py
"""
Pure IO functions for Google Sheets operations.
No Streamlit dependencies - can be imported by both logic and UI modules.
"""

import io
import json
from datetime import datetime
from typing import Dict, Optional

import pandas as pd
import gspread
from google.auth.exceptions import GoogleAuthError
from gspread.exceptions import (
    SpreadsheetNotFound,
    WorksheetNotFound,
    APIError,
    NoValidUrlKeyFound
)
from gspread_dataframe import get_as_dataframe, set_with_dataframe

from utils import (
    CANON_HEADERS, LEVEL_COLS, normalize_text
)


# ===== Google Sheets Authentication =====

def get_gspread_client_from_secrets(secrets_dict: dict) -> gspread.Client:
    """
    Create a gspread client using service account credentials.
    
    Args:
        secrets_dict: Service account credentials dictionary
        
    Returns:
        gspread.Client: Authenticated client
        
    Raises:
        GoogleAuthError: If authentication fails
        ValueError: If required fields are missing
    """
    try:
        from google.oauth2.service_account import Credentials

        # Explicit scopes for Google Sheets and Drive
        SCOPES = [
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive",
        ]
        credentials = Credentials.from_service_account_info(secrets_dict, scopes=SCOPES)

        # Authorize gspread client
        client = gspread.authorize(credentials)
        return client

    except GoogleAuthError as e:
        raise GoogleAuthError(
            "Google Sheets authentication failed. Please check your service account credentials."
        ) from e
    except KeyError as e:
        raise ValueError(
            f"Missing required field in service account: {e}. Please check your configuration."
        ) from e
    except Exception as e:
        raise Exception(f"Unexpected authentication error: {str(e)}") from e


def open_spreadsheet(client: gspread.Client, spreadsheet_id: str) -> gspread.Spreadsheet:
    """
    Open a Google Spreadsheet by ID.
    
    Args:
        client: Authenticated gspread client
        spreadsheet_id: The ID of the spreadsheet to open
        
    Returns:
        gspread.Spreadsheet: The opened spreadsheet object
        
    Raises:
        SpreadsheetNotFound: If spreadsheet doesn't exist or access is denied
        ValueError: If spreadsheet_id is invalid
    """
    try:
        # Validate spreadsheet_id format (basic check)
        if not spreadsheet_id or len(spreadsheet_id) < 20:
            raise ValueError("Invalid spreadsheet ID format. Please provide a valid Google Sheets URL or ID.")
        
        # Open the spreadsheet
        spreadsheet = client.open_by_key(spreadsheet_id)
        return spreadsheet
        
    except SpreadsheetNotFound:
        raise SpreadsheetNotFound(
            f"Spreadsheet '{spreadsheet_id}' not found or access denied. "
            "Please check the spreadsheet ID and ensure your service account "
            "has edit permissions."
        )
    except NoValidUrlKeyFound:
        raise ValueError(
            f"Invalid spreadsheet ID: '{spreadsheet_id}'. "
            "Please provide a valid Google Sheets URL or ID."
        )
    except Exception as e:
        raise Exception(f"Error opening spreadsheet: {str(e)}") from e


# ===== Worksheet Operations =====

def read_worksheet_with_canonical_headers(
    spreadsheet: gspread.Spreadsheet, 
    title: str, 
    canonical_headers: list[str]
) -> pd.DataFrame:
    """
    Read a worksheet and return DataFrame with canonical headers.
    
    Args:
        spreadsheet: The spreadsheet object
        title: Name of the worksheet to read
        canonical_headers: List of required column headers
        
    Returns:
        pd.DataFrame: The worksheet data with canonical headers
        
    Raises:
        WorksheetNotFound: If worksheet doesn't exist
        APIError: If unable to read worksheet data
    """
    try:
        # Get the worksheet
        worksheet = spreadsheet.worksheet(title)
        
        # Get raw values to handle header normalization
        values = worksheet.get_all_values()
        if not values:
            return pd.DataFrame(columns=canonical_headers)
        
        header = [normalize_text(c) for c in values[0]]
        rows = values[1:]
        
        # Create DataFrame with original headers
        df = pd.DataFrame(rows, columns=header)
        
        # Ensure canonical headers exist
        for c in canonical_headers:
            if c not in df.columns:
                df[c] = ""
        
        # Reorder to canonical headers
        df = df[canonical_headers]
        
        # Normalize text cells
        for c in canonical_headers:
            df[c] = df[c].map(normalize_text)
        
        return df
        
    except WorksheetNotFound:
        raise WorksheetNotFound(
            f"Worksheet '{title}' not found in the spreadsheet."
        )
    except APIError as e:
        raise APIError(
            f"Unable to read worksheet '{title}'. Please check your permissions."
        ) from e
    except Exception as e:
        raise Exception(f"Error reading worksheet: {str(e)}") from e


def write_dataframe(
    spreadsheet: gspread.Spreadsheet, 
    title: str, 
    df: pd.DataFrame, 
    mode: str = "overwrite"
) -> None:
    """
    Write a pandas DataFrame to a worksheet.
    
    Args:
        spreadsheet: The spreadsheet object
        title: Name of the worksheet to write to
        df: DataFrame to write
        mode: "overwrite" (clear and write) or "append" (add to end)
        
    Raises:
        WorksheetNotFound: If worksheet doesn't exist
        ValueError: If mode is invalid
        APIError: If unable to write data
    """
    try:
        # Validate mode
        if mode not in ["overwrite", "append"]:
            raise ValueError("Mode must be 'overwrite' or 'append'")
        
        # Get the worksheet
        worksheet = spreadsheet.worksheet(title)
        
        if mode == "overwrite":
            # Clear existing data and write new data
            worksheet.clear()
            if not df.empty:
                set_with_dataframe(worksheet, df, include_index=False, include_column_header=True)
                
        elif mode == "append":
            # Get existing data to find the next empty row
            existing_df = get_as_dataframe(worksheet, evaluate_formulas=True)
            
            if existing_df.empty:
                # If worksheet is empty, write with headers
                set_with_dataframe(worksheet, df, include_index=False, include_column_header=True)
            else:
                # Find the next empty row and append data
                next_row = len(existing_df) + 2  # +2 because gspread is 1-indexed and we have headers
                
                # Convert DataFrame to list of lists for append
                data_to_append = df.values.tolist()
                
                if data_to_append:
                    worksheet.append_rows(data_to_append)
        
    except WorksheetNotFound:
        raise WorksheetNotFound(
            f"Worksheet '{title}' not found in the spreadsheet. Please check the worksheet name."
        )
    except ValueError:
        raise  # Re-raise ValueError as-is
    except APIError as e:
        raise APIError(
            f"Unable to write to worksheet '{title}'. Please check your permissions for this spreadsheet."
        ) from e
    except Exception as e:
        raise Exception(f"Error writing to worksheet: {str(e)}") from e


def backup_worksheet(spreadsheet: gspread.Spreadsheet, source_title: str) -> Optional[str]:
    """
    Create a backup copy of a worksheet within the same spreadsheet.
    
    Args:
        spreadsheet: The spreadsheet object
        source_title: Name of the worksheet to backup
        
    Returns:
        str: Name of the created backup worksheet, or None on failure
        
    Raises:
        WorksheetNotFound: If source worksheet doesn't exist
        APIError: If unable to create backup
    """
    try:
        # Get the source worksheet
        source_worksheet = spreadsheet.worksheet(source_title)
        
        # Get all values from source worksheet
        values = source_worksheet.get_all_values()
        
        # Create backup title with timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H%M")
        backup_title_full = f"{source_title} (backup {timestamp})"
        
        # Truncate if too long (Google Sheets has a 100 character limit)
        if len(backup_title_full) > 100:
            backup_title_full = f"{source_title[:50]} (backup {timestamp})"
        
        # Create backup worksheet
        backup_worksheet = spreadsheet.add_worksheet(
            title=backup_title_full, 
            rows=len(values), 
            cols=len(values[0]) if values else 26
        )
        
        # Copy data to backup
        if values:
            backup_worksheet.update(values)
        
        return backup_title_full
        
    except WorksheetNotFound:
        raise WorksheetNotFound(
            f"Source worksheet '{source_title}' not found. Cannot create backup."
        )
    except APIError as e:
        raise APIError(
            f"Unable to create backup of '{source_title}'. Please check your permissions."
        ) from e
    except Exception as e:
        raise Exception(f"Error creating backup: {str(e)}") from e


# ===== High-Level Sheet Operations =====

def read_google_sheet(spreadsheet_id: str, sheet_name: str, secrets_dict: dict) -> pd.DataFrame:
    """
    Read a Google Sheet tab into a DataFrame with canonical headers.
    
    Args:
        spreadsheet_id: Google Sheets ID
        sheet_name: Name of the sheet tab
        secrets_dict: Service account credentials
        
    Returns:
        pd.DataFrame: DataFrame with canonical headers, blank rows dropped
    """
    try:
        # Get client and open spreadsheet
        client = get_gspread_client_from_secrets(secrets_dict)
        spreadsheet = open_spreadsheet(client, spreadsheet_id)
        
        # Read with canonical headers
        df = read_worksheet_with_canonical_headers(spreadsheet, sheet_name, CANON_HEADERS)
        
        # Drop fully blank node paths (VM + Node1..5 empty)
        node_block = ["Vital Measurement"] + LEVEL_COLS
        mask_blank = df[node_block].apply(lambda r: all(normalize_text(v) == "" for v in r), axis=1)
        df = df[~mask_blank].copy()
        
        return df
        
    except Exception as e:
        # Return empty DataFrame with canonical headers on error
        return pd.DataFrame(columns=CANON_HEADERS)


def push_to_google_sheets(
    spreadsheet_id: str,
    sheet_name: str,
    df: pd.DataFrame,
    secrets_dict: dict,
    *,
    mode: str = "overwrite"
) -> bool:
    """
    Write a DataFrame to a Google Sheet tab using proven resize-then-write semantics.
    
    Args:
        spreadsheet_id: Google Sheets ID
        sheet_name: Name of the sheet tab
        df: DataFrame to write
        secrets_dict: Service account credentials
        mode: "overwrite" or "append"
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Get client and open spreadsheet
        client = get_gspread_client_from_secrets(secrets_dict)
        spreadsheet = open_spreadsheet(client, spreadsheet_id)
        
        # Prepare DataFrame for writing
        df = df.fillna("")
        
        # Use proven Sheets semantics: resize first, then write
        if mode == "overwrite":
            _push_full_sheet_to_gs(spreadsheet, sheet_name, df)
        else:
            # For append mode, use existing logic
            write_dataframe(spreadsheet, sheet_name, df, mode=mode)
        
        return True

    except Exception:
        return False


def _push_full_sheet_to_gs(spreadsheet: gspread.Spreadsheet, sheet_name: str, df: pd.DataFrame):
    """
    Push full sheet to Google Sheets using proven resize-then-write semantics.
    
    This implements the stable v6.2.x behavior:
    1. Authorize client
    2. Open spreadsheet + worksheet by sheet_name
    3. Resize worksheet to rows=len(df)+1 (for header) and cols=len(df.columns)
    4. Write header row then values in bulk
    5. Rate-limit / retry on 429s
    """
    try:
        # Get the worksheet
        try:
            worksheet = spreadsheet.worksheet(sheet_name)
            # Clear existing data
            worksheet.clear()
        except Exception:
            # Create new worksheet if it doesn't exist
            worksheet = spreadsheet.add_worksheet(title=sheet_name, rows=100, cols=20)
        
        # Prepare data for writing
        headers = list(df.columns)
        values = [headers] + df.astype(str).values.tolist()
        n_rows = len(values)
        n_cols = max(1, len(headers))
        
        # Resize worksheet to accommodate data (proven Sheets semantics)
        # Add 1 row for header, ensure minimum size for stability
        target_rows = max(n_rows, 200)  # Minimum 200 rows for stability
        target_cols = max(n_cols, 8)    # Minimum 8 cols for stability
        
        worksheet.resize(rows=target_rows, cols=target_cols)
        
        # Write data in bulk (proven Sheets semantics)
        if values:
            worksheet.update('A1', values, value_input_option="RAW")
            
    except Exception as e:
        # Re-raise with context
        raise Exception(f"Failed to push full sheet to Google Sheets: {str(e)}") from e


def backup_sheet_copy(spreadsheet_id: str, source_sheet: str, secrets_dict: dict) -> Optional[str]:
    """
    Create a backup copy of a sheet within the same spreadsheet.
    
    Args:
        spreadsheet_id: Google Sheets ID
        source_sheet: Name of the source sheet
        secrets_dict: Service account credentials
        
    Returns:
        str: Name of the created backup sheet, or None on failure
    """
    try:
        # Get client and open spreadsheet
        client = get_gspread_client_from_secrets(secrets_dict)
        spreadsheet = open_spreadsheet(client, spreadsheet_id)
        
        # Create backup
        backup_title = backup_worksheet(spreadsheet, source_sheet)
        return backup_title

    except Exception:
        return None


# ===== DataFrame Export Functions =====

def export_dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    """
    Export DataFrame to CSV bytes.
    
    Args:
        df: DataFrame to export
        
    Returns:
        bytes: CSV data as bytes
    """
    return df.to_csv(index=False).encode("utf-8")


def export_dataframe_to_excel_bytes(df: pd.DataFrame, sheet_name: str = "Sheet1") -> bytes:
    """
    Export DataFrame to Excel bytes.
    
    Args:
        df: DataFrame to export
        sheet_name: Name of the sheet (max 31 chars)
        
    Returns:
        bytes: Excel file as bytes
    """
    buffer = io.BytesIO()
    # Lazy import to avoid hard dependency at import-time
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name=(sheet_name or "Sheet1")[:31])
    return buffer.getvalue()


# ===== Overrides JSON Functions =====

def export_overrides_json(overrides_sheet: Dict[str, list]) -> bytes:
    """
    Serialize a single-sheet overrides dict to pretty JSON bytes.
    
    Args:
        overrides_sheet: Dictionary of overrides
        
    Returns:
        bytes: JSON data as bytes
    """
    return json.dumps(overrides_sheet or {}, indent=2).encode("utf-8")


def import_overrides_json(
    existing_overrides_sheet: Dict[str, list],
    imported_json_bytes: bytes,
    mode: str = "replace",
) -> Dict[str, list]:
    """
    Merge/replace a sheet's overrides with imported JSON.

    Args:
        existing_overrides_sheet: Current overrides
        imported_json_bytes: JSON bytes to import
        mode: "replace", "merge_import", or "merge_existing"
        
    Returns:
        Dict[str, list]: Merged overrides
        
    Raises:
        ValueError: If JSON is invalid or mode is unknown
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
    elif mode == "merge_import":
        merged = existing.copy()
        merged.update(imported_norm)  # imported wins
        return merged
    elif mode == "merge_existing":
        merged = imported_norm.copy()
        merged.update(existing)  # existing wins
        return merged
    else:
        raise ValueError(f"Unknown import mode: {mode}")


# ===== Push Log Helper =====

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
    
    Args:
        sheet: Sheet name
        target_tab: Target tab name
        spreadsheet_id: Google Sheets ID
        rows_written: Number of rows written
        new_rows_added: Number of new rows added
        scope: Operation scope
        extra: Additional fields
        
    Returns:
        Dict[str, str]: Push log entry
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
