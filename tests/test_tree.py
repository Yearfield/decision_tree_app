import pytest
import pandas as pd
import os

from logic.tree import infer_branch_options, order_decision_tree, build_raw_plus_v630
from utils.constants import CANON_HEADERS


@pytest.fixture
def sample_df():
    """Load sample DataFrame from fixture."""
    fixture_path = os.path.join(os.path.dirname(__file__), "fixtures", "sample.csv")
    df = pd.read_csv(fixture_path)
    return df


class TestInferBranchOptions:
    """Test the infer_branch_options function."""
    
    def test_infer_branch_options_sample_data(self, sample_df):
        """Test infer_branch_options with sample data."""
        result = infer_branch_options(sample_df)
        
        # Should return a dictionary
        assert isinstance(result, dict)
        
        # Should have entries for each level
        assert "L1|" in result  # Root level
        assert "L2|" in result  # Level 2 general
        
        # Check specific values - the keys are based on the actual values, not the Vital Measurement
        assert "L2|High" in result  # Level 2 with parent "High"
        assert "L2|Low" in result   # Level 2 with parent "Low"
        assert "L2|Normal" in result  # Level 2 with parent "Normal"
        
        # Check that the values are lists
        assert isinstance(result["L2|High"], list)
        assert isinstance(result["L2|Low"], list)
        assert isinstance(result["L2|Normal"], list)
    
    def test_infer_branch_options_empty_df(self):
        """Test infer_branch_options with empty DataFrame."""
        df = pd.DataFrame(columns=CANON_HEADERS)
        result = infer_branch_options(df)
        assert result == {}
    
    def test_infer_branch_options_invalid_headers(self):
        """Test infer_branch_options with invalid headers."""
        df = pd.DataFrame(columns=["Wrong", "Headers"])
        result = infer_branch_options(df)
        assert result == {}


class TestOrderDecisionTree:
    """Test the order_decision_tree function."""
    
    def test_order_decision_tree_sample_data(self, sample_df):
        """Test order_decision_tree with sample data."""
        result = order_decision_tree(sample_df)
        
        # Should return a DataFrame
        assert isinstance(result, pd.DataFrame)
        
        # Should have same shape as input
        assert result.shape == sample_df.shape
        
        # Should be sorted by Vital Measurement first
        vm_values = result["Vital Measurement"].tolist()
        assert vm_values == sorted(vm_values)
    
    def test_order_decision_tree_empty_df(self):
        """Test order_decision_tree with empty DataFrame."""
        df = pd.DataFrame(columns=CANON_HEADERS)
        result = order_decision_tree(df)
        assert isinstance(result, pd.DataFrame)
        assert result.empty
    
    def test_order_decision_tree_already_sorted(self, sample_df):
        """Test order_decision_tree with already sorted data."""
        # Sort the sample data first
        sorted_df = sample_df.sort_values(["Vital Measurement", "Node 1", "Node 2"])
        result = order_decision_tree(sorted_df)
        
        # Should maintain the same order
        pd.testing.assert_frame_equal(result, sorted_df)


class TestBuildRawPlusV630:
    """Test the build_raw_plus_v630 function."""
    
    def test_build_raw_plus_v630_sample_data(self, sample_df):
        """Test build_raw_plus_v630 with sample data."""
        result = build_raw_plus_v630(sample_df)
        
        # Should return a DataFrame
        assert isinstance(result, pd.DataFrame)
        
        # Should have at least as many rows as input
        assert len(result) >= len(sample_df)
        
        # Should have all canonical columns
        for col in CANON_HEADERS:
            assert col in result.columns
    
    def test_build_raw_plus_v630_empty_df(self):
        """Test build_raw_plus_v630 with empty DataFrame."""
        df = pd.DataFrame(columns=CANON_HEADERS)
        result = build_raw_plus_v630(df)
        assert isinstance(result, pd.DataFrame)
        assert result.empty
    
    def test_build_raw_plus_v630_with_overrides(self, sample_df):
        """Test build_raw_plus_v630 with overrides."""
        overrides = {
            (2, "Blood Pressure"): ["Override1", "Override2"]
        }
        result = build_raw_plus_v630(sample_df, overrides)
        
        # Should return a DataFrame
        assert isinstance(result, pd.DataFrame)
        
        # Should have all canonical columns
        for col in CANON_HEADERS:
            assert col in result.columns
    
    def test_build_raw_plus_v630_invalid_headers(self):
        """Test build_raw_plus_v630 with invalid headers."""
        df = pd.DataFrame(columns=["Wrong", "Headers"])
        result = build_raw_plus_v630(df)
        
        # Should return a copy of the input DataFrame
        assert isinstance(result, pd.DataFrame)
        pd.testing.assert_frame_equal(result, df)
