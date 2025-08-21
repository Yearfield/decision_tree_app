# Robust Decision Tree Indexer

This document describes the robust indexer functions that group by parent path at level L-1 and collect unique children at level L, tolerant of duplicated rows caused by deeper nodes.

## Overview

The indexer provides four main functions that work together to analyze decision tree structures:

1. **`build_parent_child_index`** - Builds the core index mapping parent paths to child counters
2. **`summarize_children_sets`** - Converts counters to ordered unique child lists with metrics
3. **`group_by_parent_label_at_level`** - Groups parents by their last label for comparison
4. **`find_label_set_mismatches`** - Detects inconsistencies in child sets for the same parent labels

## Function Details

### `build_parent_child_index(df: pd.DataFrame) -> Dict[Tuple[int, str], Counter]`

Builds a mapping from `(level, parent_path)` to `Counter(child -> row_count)` for each level L=1..MAX_LEVELS.

- **L=1**: parent_path is `<ROOT>`, child is Node 1
- **L>=2**: parent_path is `'Node 1>...>Node L-1'` (non-empty segments only)

Each counter counts how many rows contribute that child, but the caller should treat presence as unique. This is tolerant of row multiplication at deeper levels.

**Example Output:**
```python
{
    (1, '<ROOT>'): Counter({'Hypertension': 2, 'Chest Pain': 2, 'Confusion': 1}),
    (2, 'Hypertension'): Counter({'Severe': 2}),
    (2, 'Chest Pain'): Counter({'Mild': 2}),
    (3, 'Hypertension>Severe'): Counter({'Emergency': 2}),
    # ... more levels
}
```

### `summarize_children_sets(idx: Dict[Tuple[int, str], Counter]) -> Dict[Tuple[int, str], Dict]`

Converts counters to ordered unique child lists and simple metrics.

**Returns mapping:** `(level, parent_path) -> {
    'children': List[str],
    'count': int,   # unique child count
    'over5': bool,
    'exact5': bool
}`

**Example Output:**
```python
{
    (1, '<ROOT>'): {
        'children': ['Chest Pain', 'Hypertension', 'Confusion'],
        'count': 3,
        'over5': False,
        'exact5': False
    },
    (2, 'Hypertension'): {
        'children': ['Severe'],
        'count': 1,
        'over5': False,
        'exact5': False
    }
    # ... more entries
}
```

### `group_by_parent_label_at_level(summary: Dict[Tuple[int, str], Dict]) -> Dict[Tuple[int, str], List[Tuple[str, List[str]]]]`

For a given level L>=2, groups parents by their **last label** (parent label). This lets us detect 'same label, different 5-sets'.

- **L=1**: The label is ROOT (only one group)
- **L>=2**: Groups by the last segment of the parent path

**Returns mapping:** `(L, parent_label) -> [ (parent_path, children_list) ... ]`

**Example Output:**
```python
{
    (1, '<ROOT>'): [('<ROOT>', ['Chest Pain', 'Hypertension', 'Confusion'])],
    (2, 'Hypertension'): [('Hypertension', ['Severe'])],
    (2, 'Chest Pain'): [('Chest Pain', ['Mild'])],
    (3, 'Severe'): [('Hypertension>Severe', ['Emergency'])],
    (3, 'Mild'): [('Chest Pain>Mild', ['Monitor'])]
    # ... more groups
}
```

### `find_label_set_mismatches(summary: Dict[Tuple[int, str], Dict]) -> Dict[Tuple[int, str], Dict]`

For each `(L, parent_label)`, checks if all parent paths sharing that label have the **same 5 children**. Reports groups where:
- Some have >5 children, OR
- They do not all share the exact same 5-set

**Returns:** `(L, parent_label) -> {
    'variants': List[ {'parent_path': str, 'children': List[str]} ],
    'has_over5': bool,
    'all_exact5_same': bool
}`

**Example Output:**
```python
{
    (5, 'ICU'): {
        'variants': [
            {'parent_path': 'Hypertension>Severe>Emergency>ICU', 'children': ['Ventilator', 'NewTreatment']}
        ],
        'has_over5': False,
        'all_exact5_same': False
    }
    # ... more mismatch reports
}
```

## Usage Example

```python
from logic.tree import (
    build_parent_child_index,
    summarize_children_sets,
    group_by_parent_label_at_level,
    find_label_set_mismatches
)

# Load your decision tree data
df = pd.read_csv('your_data.csv')

# Build the index
idx = build_parent_child_index(df)

# Summarize the children sets
summary = summarize_children_sets(idx)

# Group by parent labels
buckets = group_by_parent_label_at_level(summary)

# Find any mismatches
mismatches = find_label_set_mismatches(summary)

# Check for issues
for (level, parent_label), info in mismatches.items():
    if not info['all_exact5_same'] or info['has_over5']:
        print(f"Issue at level {level}, parent label '{parent_label}':")
        print(f"  Has over 5 children: {info['has_over5']}")
        print(f"  All exact 5 same: {info['all_exact5_same']}")
        print(f"  Variants: {len(info['variants'])}")
```

## Key Features

1. **Robust to Duplicates**: Handles duplicated rows at deeper levels without inflating unique child counts
2. **Parent Path Grouping**: Groups by complete parent paths to maintain tree structure integrity
3. **Label-Based Comparison**: Groups by parent labels to detect inconsistencies across different paths
4. **Comprehensive Metrics**: Provides detailed information about child counts, over-5 violations, and exact-5 matches
5. **Ordered Results**: Children are ordered by frequency (descending) then alphabetically

## Tolerance to Row Multiplication

The indexer is specifically designed to handle scenarios where deeper nodes cause row multiplication. For example:

- A single parent path might appear in multiple rows due to different child combinations at deeper levels
- The indexer counts these occurrences but treats the presence of a child as unique
- This prevents inflated child counts while maintaining accurate parent-child relationships

## Constants Used

- `LEVEL_COLS`: List of Node column names (`["Node 1", "Node 2", "Node 3", "Node 4", "Node 5"]`)
- `MAX_LEVELS`: Maximum number of levels (5)
- `MAX_CHILDREN_PER_PARENT`: Maximum children per parent (5)
- `ROOT_PARENT_LABEL`: Synthetic parent for Level 1 (`"<ROOT>"`)

## Error Handling

All functions include robust error handling:
- Empty or None DataFrames return empty results
- Missing columns are gracefully handled
- Invalid data types are converted to strings and normalized
- Blank/empty values are filtered out during processing
