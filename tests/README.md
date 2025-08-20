# Test Suite

This directory contains the test suite for the Decision Tree Builder application.

## Structure

- `test_helpers.py` - Tests for utility functions (`normalize_text`, `validate_headers`)
- `test_tree.py` - Tests for tree logic functions (`infer_branch_options`, `order_decision_tree`, `build_raw_plus_v630`)
- `test_validate.py` - Tests for validation functions (`detect_orphan_nodes`, `detect_missing_red_flags`)
- `fixtures/sample.csv` - Sample data for testing

## Running Tests

```bash
# Run all tests
pytest

# Run tests quietly
pytest -q

# Run specific test file
pytest tests/test_helpers.py

# Run specific test class
pytest tests/test_helpers.py::TestNormalizeText

# Run specific test function
pytest tests/test_helpers.py::TestNormalizeText::test_normalize_text_string
```

## Test Coverage

The test suite covers:

- **Helper Functions**: Text normalization, header validation
- **Tree Logic**: Branch option inference, tree ordering, tree building
- **Validation**: Orphan node detection, missing red flag detection
- **Edge Cases**: Empty DataFrames, invalid headers, None inputs
- **Data Integrity**: Proper return types, expected data structures

## Fixtures

- `sample.csv` - Contains sample decision tree data with multiple levels and various scenarios
