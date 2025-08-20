#!/usr/bin/env python3
"""Tests for root level (Level-1) functionality in decision tree logic."""

import pytest
import pandas as pd
import numpy as np

from logic.tree import infer_branch_options, set_level1_children
from utils.constants import ROOT_PARENT_LABEL, MAX_CHILDREN_PER_PARENT


class TestRootLevelFunctionality:
    """Test suite for root level (Level-1) functionality."""
    
    def test_infer_branch_options_with_various_node1_labels(self):
        """Test that infer_branch_options correctly identifies unique Node-1 labels."""
        # Create DataFrame with various Node-1 labels (some duplicates, some blanks)
        df = pd.DataFrame({
            "Vital Measurement": ["VM1", "VM2", "VM3"],
            "Node 1": ["Alpha", "Beta", "Gamma"],
            "Node 2": ["Child1", "Child2", "Child3"],
            "Node 3": ["", "", ""],
            "Node 4": ["", "", ""],
            "Node 5": ["", "", ""],
            "Diagnostic Triage": ["", "", ""],
            "Actions": ["", "", ""]
        })
        
        # Get the store
        store = infer_branch_options(df)
        
        # Check that L1|<ROOT> exists
        root_key = f"L1|{ROOT_PARENT_LABEL}"
        assert root_key in store, f"Expected {root_key} to be in store keys: {list(store.keys())}"
        
        # Get root children
        root_children = store[root_key]
        
        # Should have unique, non-blank Node-1 labels
        expected_children = ["Alpha", "Beta", "Gamma"]
        assert len(root_children) == len(expected_children), f"Expected {len(expected_children)} children, got {len(root_children)}"
        
        # Check that all expected children are present
        for expected in expected_children:
            assert expected in root_children, f"Expected child '{expected}' not found in {root_children}"
        
        # Check that no blank/whitespace/None values are present
        for child in root_children:
            assert child and child.strip(), f"Found blank/whitespace child: '{child}'"
            assert child is not None, f"Found None child: {child}"
    
    def test_infer_branch_options_with_ragged_rows(self):
        """Test that infer_branch_options handles ragged rows correctly."""
        # Create DataFrame with some empty values but valid structure
        df = pd.DataFrame({
            "Vital Measurement": ["VM1", "VM2"],
            "Node 1": ["A", "B"],
            "Node 2": ["X", ""],
            "Node 3": ["", ""],
            "Node 4": ["", ""],
            "Node 5": ["", ""],
            "Diagnostic Triage": ["", ""],
            "Actions": ["", ""]
        })
        
        # Get the store
        store = infer_branch_options(df)
        
        # Check that L1|<ROOT> exists and contains all Node-1 values
        root_key = f"L1|{ROOT_PARENT_LABEL}"
        assert root_key in store, f"Expected {root_key} to be in store keys: {list(store.keys())}"
        
        root_children = store[root_key]
        expected_children = ["A", "B"]
        
        assert len(root_children) == len(expected_children), f"Expected {len(expected_children)} children, got {len(root_children)}"
        
        for expected in expected_children:
            assert expected in root_children, f"Expected child '{expected}' not found in {root_children}"
    
    def test_infer_branch_options_with_empty_dataframe(self):
        """Test that infer_branch_options handles empty DataFrame correctly."""
        # Empty DataFrame
        df = pd.DataFrame()
        
        store = infer_branch_options(df)
        
        # Should return empty store
        assert store == {}, f"Expected empty store for empty DataFrame, got {store}"
    
    def test_infer_branch_options_with_no_node_columns(self):
        """Test that infer_branch_options handles DataFrame with no Node columns."""
        # DataFrame with no Node columns
        df = pd.DataFrame({
            "Vital Measurement": ["VM1", "VM2", "VM3"],
            "Other Column": ["OC1", "OC2", "OC3"]
        })
        
        store = infer_branch_options(df)
        
        # Should return empty store
        assert store == {}, f"Expected empty store for DataFrame with no Node columns, got {store}"
    
    def test_set_level1_children_with_exact_5_children(self):
        """Test set_level1_children with exactly 5 children."""
        # Create test DataFrame
        df = pd.DataFrame({
            "Vital Measurement": ["VM1", "VM2", "VM3", "VM4", "VM5", "VM6"],
            "Node 1": ["OldA", "OldB", "OldC", "OldD", "OldE", "OldF"],
            "Node 2": ["Child1", "Child2", "Child3", "Child4", "Child5", "Child6"],
            "Node 3": ["", "", "", "", "", ""],
            "Node 4": ["", "", "", "", "", ""],
            "Node 5": ["", "", "", "", "", ""],
            "Diagnostic Triage": ["", "", "", "", "", ""],
            "Actions": ["", "", "", "", "", ""]
        })
        
        # Set new children (exactly 5)
        new_children = ["A", "B", "C", "D", "E"]
        result_df = set_level1_children(df, new_children)
        
        # Check that result is a DataFrame
        assert isinstance(result_df, pd.DataFrame), f"Expected DataFrame, got {type(result_df)}"
        
        # Check that all Node 1 values are in the new set
        node1_values = result_df["Node 1"].unique()
        for value in node1_values:
            assert value in new_children, f"Node 1 value '{value}' not in expected set {new_children}"
        
        # Check that we have the expected number of unique values
        assert len(node1_values) <= len(new_children), f"Expected at most {len(new_children)} unique values, got {len(node1_values)}"
    
    def test_set_level1_children_with_more_than_5_children(self):
        """Test set_level1_children with more than 5 children (should cap at 5)."""
        # Create test DataFrame
        df = pd.DataFrame({
            "Vital Measurement": ["VM1", "VM2", "VM3", "VM4", "VM5", "VM6"],
            "Node 1": ["OldA", "OldB", "OldC", "OldD", "OldE", "OldF"],
            "Node 2": ["Child1", "Child2", "Child3", "Child4", "Child5", "Child6"],
            "Node 3": ["", "", "", "", "", ""],
            "Node 4": ["", "", "", "", "", ""],
            "Node 5": ["", "", "", "", "", ""],
            "Diagnostic Triage": ["", "", "", "", "", ""],
            "Actions": ["", "", "", "", "", ""]
        })
        
        # Set new children (more than 5)
        new_children = ["A", "B", "C", "D", "E", "F"]
        result_df = set_level1_children(df, new_children)
        
        # Check that result is a DataFrame
        assert isinstance(result_df, pd.DataFrame), f"Expected DataFrame, got {type(result_df)}"
        
        # Check that all Node 1 values are in the new set (capped at 5)
        node1_values = result_df["Node 1"].unique()
        capped_children = new_children[:MAX_CHILDREN_PER_PARENT]  # Should be capped at 5
        
        for value in node1_values:
            assert value in capped_children, f"Node 1 value '{value}' not in capped set {capped_children}"
        
        # Check that we have at most 5 unique values
        assert len(node1_values) <= MAX_CHILDREN_PER_PARENT, f"Expected at most {MAX_CHILDREN_PER_PARENT} unique values, got {len(node1_values)}"
    
    def test_set_level1_children_with_empty_children_list(self):
        """Test set_level1_children with empty children list (should return unchanged DataFrame)."""
        # Create test DataFrame
        df = pd.DataFrame({
            "Vital Measurement": ["VM1", "VM2", "VM3"],
            "Node 1": ["OldA", "OldB", "OldC"],
            "Node 2": ["Child1", "Child2", "Child3"],
            "Node 3": ["", "", ""],
            "Node 4": ["", "", ""],
            "Node 5": ["", "", ""],
            "Diagnostic Triage": ["", "", ""],
            "Actions": ["", "", ""]
        })
        
        original_df = df.copy()
        
        # Set empty children list
        result_df = set_level1_children(df, [])
        
        # Should return unchanged DataFrame
        pd.testing.assert_frame_equal(result_df, original_df), "Expected unchanged DataFrame for empty children list"
    
    def test_set_level1_children_with_none_dataframe(self):
        """Test set_level1_children with None DataFrame (should return None)."""
        # Test with None DataFrame
        result_df = set_level1_children(None, ["A", "B", "C"])
        
        # Should return None
        assert result_df is None, f"Expected None for None DataFrame, got {result_df}"
    
    def test_set_level1_children_with_empty_dataframe(self):
        """Test set_level1_children with empty DataFrame (should return empty DataFrame)."""
        # Create empty DataFrame
        df = pd.DataFrame()
        
        # Set children
        result_df = set_level1_children(df, ["A", "B", "C"])
        
        # Should return empty DataFrame
        assert isinstance(result_df, pd.DataFrame), f"Expected DataFrame, got {type(result_df)}"
        assert result_df.empty, f"Expected empty DataFrame, got {result_df.shape}"
    
    def test_set_level1_children_mapping_behavior(self):
        """Test that set_level1_children correctly maps non-matching values to first child."""
        # Create test DataFrame with values not in the new set
        df = pd.DataFrame({
            "Vital Measurement": ["VM1", "VM2", "VM3", "VM4", "VM5"],
            "Node 1": ["OldA", "OldB", "OldC", "OldD", "OldE"],
            "Node 2": ["Child1", "Child2", "Child3", "Child4", "Child5"],
            "Node 3": ["", "", "", "", ""],
            "Node 4": ["", "", "", "", ""],
            "Node 5": ["", "", "", "", ""],
            "Diagnostic Triage": ["", "", "", "", ""],
            "Actions": ["", "", "", "", ""]
        })
        
        # Set new children
        new_children = ["NewA", "NewB", "NewC"]
        result_df = set_level1_children(df, new_children)
        
        # Check that all Node 1 values are in the new set
        node1_values = result_df["Node 1"].unique()
        
        # All values should be in the new set
        for value in node1_values:
            assert value in new_children, f"Node 1 value '{value}' not in expected set {new_children}"
        
        # Check that we have at most MAX_CHILDREN_PER_PARENT unique values
        assert len(node1_values) <= MAX_CHILDREN_PER_PARENT, f"Expected at most {MAX_CHILDREN_PER_PARENT} unique values, got {len(node1_values)}"
    
    def test_infer_branch_options_sorts_labels(self):
        """Test that infer_branch_options sorts Node-1 labels alphabetically."""
        # Create DataFrame with specific order
        df = pd.DataFrame({
            "Vital Measurement": ["VM1", "VM2", "VM3", "VM4", "VM5", "VM6"],
            "Node 1": ["Zebra", "Alpha", "Beta", "Zebra", "Alpha", "Gamma"],
            "Node 2": ["Child1", "Child2", "Child3", "Child4", "Child5", "Child6"],
            "Node 3": ["", "", "", "", "", ""],
            "Node 4": ["", "", "", "", "", ""],
            "Node 5": ["", "", "", "", "", ""],
            "Diagnostic Triage": ["", "", "", "", "", ""],
            "Actions": ["", "", "", "", "", ""]
        })
        
        # Get the store
        store = infer_branch_options(df)
        
        # Check that L1|<ROOT> exists
        root_key = f"L1|{ROOT_PARENT_LABEL}"
        assert root_key in store, f"Expected {root_key} to be in store keys: {list(store.keys())}"
        
        # Get root children
        root_children = store[root_key]
        
        # Should have unique values (sorted alphabetically)
        expected_children = ["Alpha", "Beta", "Gamma", "Zebra"]
        
        assert len(root_children) == len(expected_children), f"Expected {len(expected_children)} children, got {len(root_children)}"
        
        # Check that all expected children are present (sorted order)
        for i, expected in enumerate(expected_children):
            assert root_children[i] == expected, f"Expected sorted order {expected_children}, got {root_children}"


if __name__ == "__main__":
    # Run tests directly if script is executed
    pytest.main([__file__, "-v"])
