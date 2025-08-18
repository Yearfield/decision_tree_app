"""
Reference snippet from the monolith app:
Google Sheets authentication and client creation
"""

import gspread
import streamlit as st
from google.oauth2.service_account import Credentials

def get_gsheet_client():
    """
    Authenticate with Google Sheets using service account credentials
    stored in Streamlit secrets.
    """
    try:
        # Get the full service account info from secrets
        service_account_info = st.secrets["gcp_service_account"]

        # Create credentials object
        credentials = Credentials.from_service_account_info(service_account_info)

        # Authorize gspread client
        client = gspread.authorize(credentials)
        return client

    except Exception as e:
        st.error(f"Google Sheets authentication failed: {e}")
        return None
