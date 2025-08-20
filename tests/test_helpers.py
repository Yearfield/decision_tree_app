import pytest
import pandas as pd
import numpy as np

from utils.helpers import normalize_text, validate_headers
from utils.constants import CANON_HEADERS


class TestNormalizeText:
    """Test the normalize_text helper function."""
    
    def test_normalize_text_string(self):
        """Test normalize_text with string input."""
        assert normalize_text("  Hello World  ") == "Hello World"
        assert normalize_text("test") == "test"
        assert normalize_text("") == ""
    
    def test_normalize_text_none(self):
        """Test normalize_text with None input."""
        assert normalize_text(None) == ""
    
    def test_normalize_text_nan(self):
        """Test normalize_text with NaN input."""
        assert normalize_text(np.nan) == ""
        assert normalize_text(float('nan')) == ""
    
    def test_normalize_text_numbers(self):
        """Test normalize_text with numeric input."""
        assert normalize_text(123) == "123"
        assert normalize_text(0) == "0"
        assert normalize_text(3.14) == "3.14"


class TestValidateHeaders:
    """Test the validate_headers helper function."""
    
    def test_validate_headers_valid(self):
        """Test validate_headers with valid headers."""
        df = pd.DataFrame(columns=CANON_HEADERS)
        # Empty DataFrame with valid headers should return False
        assert validate_headers(df) is False
        
        # Add some data to make it valid
        df = pd.DataFrame([["BP", "High", "Severe", "", "", "", "Immediate", "Red flag"]], columns=CANON_HEADERS)
        assert validate_headers(df) is True
    
    def test_validate_headers_invalid(self):
        """Test validate_headers with invalid headers."""
        df = pd.DataFrame(columns=["Wrong", "Headers"])
        assert validate_headers(df) is False
    
    def test_validate_headers_empty(self):
        """Test validate_headers with empty DataFrame."""
        df = pd.DataFrame()
        assert validate_headers(df) is False
    
    def test_validate_headers_none(self):
        """Test validate_headers with None input."""
        assert validate_headers(None) is False
    
    def test_validate_headers_partial(self):
        """Test validate_headers with partial headers."""
        df = pd.DataFrame(columns=CANON_HEADERS[:4])
        assert validate_headers(df) is False
    
    def test_validate_headers_extra(self):
        """Test validate_headers with extra columns."""
        extra_headers = CANON_HEADERS + ["Extra Column"]
        df = pd.DataFrame(columns=extra_headers)
        # Empty DataFrame should return False
        assert validate_headers(df) is False
        
        # Add some data to make it valid
        df = pd.DataFrame([["BP", "High", "Severe", "", "", "", "Immediate", "Red flag", "Extra"]], columns=extra_headers)
        assert validate_headers(df) is True  # Should still be valid if canonical headers are first
