"""
Google Sheets helper functions for Streamlit app.

This module provides functions to interact with Google Sheets using service account
authentication from Streamlit secrets. It handles authentication, spreadsheet
operations, and data read/write with proper error handling.
"""

import gspread
import pandas as pd
from google.auth.exceptions import GoogleAuthError
from gspread.exceptions import (
    SpreadsheetNotFound, 
    WorksheetNotFound, 
    APIError,
    NoValidUrlKeyFound
)
from gspread_dataframe import get_as_dataframe, set_with_dataframe
import streamlit as st
from typing import Optional


def get_gspread_client_from_secrets(secrets_dict: dict) -> gspread.Client:
    """
    Create a gspread client using service account credentials from Streamlit secrets.
    
    Args:
        secrets_dict: Dictionary containing service account credentials
        
    Returns:
        gspread.Client: Authenticated client for Google Sheets API
        
    Raises:
        GoogleAuthError: If authentication fails
        ValueError: If required credentials are missing
    """
    try:
        # Create credentials from the service account dict (same as monolith)
        from google.oauth2.service_account import Credentials
        credentials = Credentials.from_service_account_info(secrets_dict)
        
        # Create and return the client (same as monolith)
        client = gspread.authorize(credentials)
        
        return client
        
    except GoogleAuthError as e:
        raise GoogleAuthError(
            "❌ Google Sheets authentication failed. Please check your service account credentials in "
            "st.secrets['gcp_service_account']. Make sure the service account has the necessary permissions."
        ) from e
    except KeyError as e:
        raise ValueError(
            f"❌ Missing required field in service account: {e}. Please check your "
            "st.secrets['gcp_service_account'] configuration."
        ) from e
    except Exception as e:
        raise Exception(f"❌ Unexpected authentication error: {str(e)}") from e


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
            raise ValueError("❌ Invalid spreadsheet ID format. Please provide a valid Google Sheets URL or ID.")
        
        # Open the spreadsheet
        spreadsheet = client.open_by_key(spreadsheet_id)
        return spreadsheet
        
    except SpreadsheetNotFound:
        raise SpreadsheetNotFound(
            f"❌ Spreadsheet '{spreadsheet_id}' not found or access denied. "
            "Please check the spreadsheet ID and ensure your service account has edit permissions."
        )
    except NoValidUrlKeyFound:
        raise ValueError(f"❌ Invalid spreadsheet ID: '{spreadsheet_id}'. Please provide a valid Google Sheets URL or ID.")
    except Exception as e:
        raise Exception(f"❌ Error opening spreadsheet: {str(e)}") from e


def list_worksheets(spreadsheet: gspread.Spreadsheet) -> list[str]:
    """
    Get a list of worksheet names in the spreadsheet.
    
    Args:
        spreadsheet: The spreadsheet object
        
    Returns:
        list[str]: List of worksheet names
        
    Raises:
        APIError: If unable to access worksheet list
    """
    try:
        worksheets = spreadsheet.worksheets()
        return [worksheet.title for worksheet in worksheets]
        
    except APIError as e:
        raise APIError(
            "Unable to list worksheets. Please check your permissions for this spreadsheet."
        ) from e
    except Exception as e:
        raise Exception(f"Error listing worksheets: {str(e)}") from e


def create_worksheet(spreadsheet: gspread.Spreadsheet, title: str, rows: int = 1000, cols: int = 26) -> gspread.Worksheet:
    """
    Create a new worksheet in the spreadsheet.
    
    Args:
        spreadsheet: The spreadsheet object
        title: Name of the new worksheet
        rows: Number of rows (default: 1000)
        cols: Number of columns (default: 26)
        
    Returns:
        gspread.Worksheet: The newly created worksheet
        
    Raises:
        ValueError: If title is invalid or worksheet already exists
        APIError: If unable to create worksheet
    """
    try:
        # Validate title
        if not title or not title.strip():
            raise ValueError("❌ Worksheet title cannot be empty. Please provide a valid name.")
        
        # Check if worksheet already exists
        existing_titles = list_worksheets(spreadsheet)
        if title in existing_titles:
            raise ValueError(f"❌ Worksheet '{title}' already exists. Please choose a different name.")
        
        # Create the worksheet
        worksheet = spreadsheet.add_worksheet(title=title, rows=rows, cols=cols)
        return worksheet
        
    except ValueError:
        raise  # Re-raise ValueError as-is
    except APIError as e:
        raise APIError(
            f"❌ Unable to create worksheet '{title}'. Please check your permissions for this spreadsheet."
        ) from e
    except Exception as e:
        raise Exception(f"❌ Error creating worksheet: {str(e)}") from e


def read_worksheet(spreadsheet: gspread.Spreadsheet, title: str) -> pd.DataFrame:
    """
    Read a worksheet and return its data as a pandas DataFrame.
    
    Args:
        spreadsheet: The spreadsheet object
        title: Name of the worksheet to read
        
    Returns:
        pd.DataFrame: The worksheet data as a DataFrame
        
    Raises:
        WorksheetNotFound: If worksheet doesn't exist
        APIError: If unable to read worksheet data
    """
    try:
        # Get the worksheet
        worksheet = spreadsheet.worksheet(title)
        
        # Read data using gspread_dataframe
        df = get_as_dataframe(worksheet, evaluate_formulas=True)
        
        # Handle empty worksheets
        if df.empty:
            return pd.DataFrame()
        
        # Clean up the DataFrame (remove empty rows/columns)
        df = df.dropna(how='all').dropna(axis=1, how='all')
        
        return df
        
    except WorksheetNotFound:
        raise WorksheetNotFound(
            f"❌ Worksheet '{title}' not found in the spreadsheet. Please check the worksheet name."
        )
    except APIError as e:
        raise APIError(
            f"❌ Unable to read worksheet '{title}'. Please check your permissions for this spreadsheet."
        ) from e
    except Exception as e:
        raise Exception(f"❌ Error reading worksheet: {str(e)}") from e


def read_worksheet_with_canonical_headers(spreadsheet: gspread.Spreadsheet, title: str, canonical_headers: list[str]) -> pd.DataFrame:
    """
    Read a worksheet and return DataFrame with canonical headers (for decision tree app).
    
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
        
        # Normalize headers (assuming normalize_text function is available)
        try:
            from utils import normalize_text
        except ImportError:
            # Fallback normalization if utils not available
            def normalize_text(text):
                return str(text).strip().lower() if text else ""
        
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


def write_dataframe(spreadsheet: gspread.Spreadsheet, title: str, df: pd.DataFrame, mode: str = "overwrite") -> None:
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
            raise ValueError("❌ Mode must be 'overwrite' or 'append'")
        
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
            f"❌ Worksheet '{title}' not found in the spreadsheet. Please check the worksheet name."
        )
    except ValueError:
        raise  # Re-raise ValueError as-is
    except APIError as e:
        raise APIError(
            f"❌ Unable to write to worksheet '{title}'. Please check your permissions for this spreadsheet."
        ) from e
    except Exception as e:
        raise Exception(f"❌ Error writing to worksheet: {str(e)}") from e


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
        from datetime import datetime
        
        # Get the source worksheet
        source_worksheet = spreadsheet.worksheet(source_title)
        
        # Get all values from source worksheet
        values = source_worksheet.get_all_values()
        
        # Create backup title with timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H%M")
        backup_title_full = f"{source_title} (backup {timestamp})"
        backup_title = backup_title_full[:99]  # Google Sheets worksheet name limit
        
        # Calculate dimensions
        rows = max(len(values), 100)
        cols = max(len(values[0]) if values else 8, 8)
        
        # Create backup worksheet
        backup_worksheet = spreadsheet.add_worksheet(title=backup_title, rows=rows, cols=cols)
        
        # Copy data if not empty
        if values:
            backup_worksheet.update("A1", values, value_input_option="RAW")
        
        return backup_title
        
    except WorksheetNotFound:
        raise WorksheetNotFound(
            f"Source worksheet '{source_title}' not found."
        )
    except APIError as e:
        raise APIError(
            f"Unable to create backup of worksheet '{source_title}'. Please check your permissions."
        ) from e
    except Exception as e:
        raise Exception(f"Error creating backup: {str(e)}") from e
