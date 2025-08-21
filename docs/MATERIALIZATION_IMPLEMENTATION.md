# Materialization Implementation for Conflicts Tab

This document describes the implementation of write-back actions in the Conflicts tab that apply canonical 5-child sets using the monolith row-multiplication rule.

## Overview

The materialization functionality provides two key actions for resolving conflicts:
1. **Keep Variant as Canonical**: Uses the children set from a chosen variant row
2. **Custom Canonical 5**: Multiselect + free text input, capped to 5 children

Both actions apply the canonical set to all parent paths sharing the same (level, parent_label) using monolith row-multiplication semantics.

## Implementation Components

### 1. Materializer Utility

**File: `logic/materialize.py`**

#### Core Functions

**`_path_cols(level: int) -> List[str]`**
- Returns columns for parent prefix at given level
- For level L: `[ROOT_COL, Node1, ..., Node(L-1)]`

**`_child_col(level: int) -> str`**
- Returns the child column name for a given level
- For level L: `Node L`

**`_deeper_cols(level: int) -> List[str]`**
- Returns columns that should be cleared when materializing
- For level L: `[Node L+1, ..., Node 5, Diagnostic Triage, Actions]`

**`materialize_children_for_label_group(df, level, parent_label, new_children)`**
- **Main Function**: Applies canonical children using monolith rules
- **Row Multiplication**: First child in existing row, (k-1) new rows below
- **Prefix Replication**: Copies prefix columns up to Node (L-1)
- **Deeper Clearing**: Clears Node (L+1) through Node 5 + diagnostic fields

### 2. Monolith Row-Multiplication Rule

#### For Each Affected Parent Path at Level L:

1. **Base Row Processing**:
   - Take the first row with that parent path as the base
   - Set the first child in the existing row's Node L cell
   - Clear deeper columns (Node L+1 through Node 5, Diagnostic Triage, Actions)

2. **New Row Creation**:
   - Insert (k-1) new rows immediately below the base row
   - Each new row gets one of the remaining children
   - Replicate prefix columns (ROOT through Node L-1) in all new rows

3. **Column Management**:
   - **Prefix Columns**: Replicated from base row (ROOT through Node L-1)
   - **Target Column**: Set to respective child value (Node L)
   - **Deeper Columns**: Cleared (Node L+1 through Node 5, Diagnostic Triage, Actions)

### 3. UI Integration

**File: `ui/tabs/conflicts.py`**

#### Materialization Actions

**`_apply_canonical_set_for_label_group(level, parent_label, children, sheet_name)`**
- Applies materialization using monolith rules
- Updates workbook with new DataFrame
- Refreshes active workbook state
- Bumps nonce via `set_current_sheet()`
- Clears stale caches for immediate refresh
- Triggers `safe_rerun()` to update UI

#### UI Components

**Keep Variant as Canonical**:
- Selectbox to choose variant from existing variants
- Button to apply selected variant's children
- Children capped to MAX_CHILDREN_PER_PARENT (5)

**Custom Canonical Set (≤5)**:
- Multiselect from union of existing children
- Free-text input for new children
- Majority-vote suggestion pre-fills the selection
- Button to apply custom set

## Data Flow and Processing

### 1. Input Processing
```
Selected (level, parent_label) + children list
    ↓
materialize_children_for_label_group()
    ↓
Identify affected parent paths
    ↓
Apply monolith row-multiplication rule
```

### 2. Row Transformation
```
Original Row: [ROOT, N1, N2, N3, N4, N5, DT, A]
    ↓
Base Row: [ROOT, N1, N2, child1, "", "", "", ""]
    ↓
New Row 1: [ROOT, N1, N2, child2, "", "", "", ""]
New Row 2: [ROOT, N1, N2, child3, "", "", "", ""]
New Row 3: [ROOT, N1, N2, child4, "", "", "", ""]
New Row 4: [ROOT, N1, N2, child5, "", "", "", ""]
```

### 3. Output Generation
- **Kept Rows**: Rows not affected by materialization
- **Rebuilt Rows**: Base row + new rows for the affected parent paths
- **Concatenation**: `pd.concat([kept, rebuilt], ignore_index=True)`

## Key Features

### 1. **Row Multiplication Tolerance**
- **No Double-Counting**: Children counted uniquely per parent path
- **Deterministic Expansion**: Predictable row generation pattern
- **Stable Ordering**: Maintains row appearance order

### 2. **Prefix Preservation**
- **ROOT Column**: Vital Measurement values preserved
- **Node Columns**: Values up to Node (L-1) replicated
- **Data Integrity**: Parent path structure maintained

### 3. **Deeper Column Clearing**
- **Automatic Cleanup**: Node L+1 through Node 5 cleared
- **Diagnostic Fields**: Diagnostic Triage and Actions cleared
- **Rebuild Ready**: Prepared for downstream processing

### 4. **Performance Optimization**
- **Single Pass**: All transformations in one DataFrame iteration
- **Efficient Concatenation**: Minimal memory overhead
- **Index Preservation**: Maintains row relationships

## Testing Results

### Test 1: Level 1 (ROOT) Materialization
- **Input**: 5 rows with 3 children in Node 1
- **Output**: 15 rows with 5 children in Node 1
- **Result**: ✅ All deeper columns cleared, prefix preserved

### Test 2: Level 2 Materialization for 'Severe' Parent
- **Input**: 2 rows with 'Severe' in Node 1
- **Output**: 5 rows with 5 children in Node 2 for 'Severe' paths
- **Result**: ✅ Prefix columns replicated, deeper columns cleared

### Test 3: Level 3 Materialization for 'Emergency' Parent
- **Input**: 2 rows with 'Emergency' in Node 2
- **Output**: 5 rows with 5 children in Node 3 for 'Emergency' paths
- **Result**: ✅ All levels working correctly

## Usage Examples

### Basic Materialization
```python
from logic.materialize import materialize_children_for_label_group

# Apply 5 children to Level 2, parent label 'Severe'
new_df = materialize_children_for_label_group(
    df, 
    level=2, 
    parent_label="Severe", 
    new_children=['Emergency', 'Urgent', 'Critical', 'Fast', 'Immediate']
)
```

### UI Integration
```python
# In Conflicts tab drilldown
if st.button("✅ Apply: Keep selected variant for all"):
    kept = block["variants"][sel_idx]["children"]
    _apply_canonical_set_for_label_group(level, parent_label, kept, sheet_name)

if st.button("🔧 Apply Custom Set"):
    _apply_canonical_set_for_label_group(level, parent_label, custom_children, sheet_name)
```

## Benefits

### 1. **Accurate Conflict Resolution**
- **Group-Level Application**: Applies to all parents sharing the same label
- **Consistent Structure**: Uniform 5-child sets across all affected paths
- **Deterministic Results**: Predictable row generation and ordering

### 2. **Data Integrity**
- **Prefix Preservation**: Maintains parent path structure
- **Deep Clearing**: Prepares for downstream rebuild
- **No Data Loss**: Affected rows are properly transformed

### 3. **User Experience**
- **Immediate Feedback**: Success messages and progress indicators
- **Cache Invalidation**: Automatic UI refresh after changes
- **State Management**: Proper workbook and nonce updates

### 4. **Maintainability**
- **Clean Separation**: Materialization logic isolated in utility module
- **Reusable Functions**: Can be used by other components
- **Clear API**: Well-documented function signatures

## Integration Points

### 1. **State Management**
- Uses `get_active_workbook()`, `set_active_workbook()`
- Integrates with `set_current_sheet()` for nonce bumping
- Maintains consistency across tab refreshes

### 2. **Cache Management**
- Clears `st.cache_data` for immediate refresh
- Uses nonce-aware caching for conflict summary
- Ensures fresh data display after materialization

### 3. **UI Components**
- **Conflicts Tab**: Enhanced with materialization actions
- **Level Drilldown**: Maintains existing conflict resolution functionality
- **Success Feedback**: Clear indicators of materialization results

## Future Enhancements

### 1. **Advanced Materialization**
- **Bulk Operations**: Materialize multiple levels simultaneously
- **Conditional Logic**: Apply different rules based on data patterns
- **Validation Rules**: Business rule enforcement during materialization

### 2. **Performance Optimization**
- **Incremental Updates**: Delta materialization for changed data only
- **Parallel Processing**: Multi-threaded materialization for large datasets
- **Memory Optimization**: Streaming materialization for very large trees

### 3. **User Experience**
- **Preview Mode**: Show materialization results before applying
- **Undo/Redo**: Track materialization history
- **Batch Operations**: Materialize multiple conflict groups at once

## Conclusion

The materialization implementation successfully provides:

- **Robust Conflict Resolution**: Two clear action paths (Keep Variant / Custom Canonical 5)
- **Monolith Compliance**: Follows the established row-multiplication rule
- **Data Integrity**: Preserves prefix structure while clearing deeper nodes
- **Performance**: Efficient single-pass processing with minimal memory overhead
- **User Experience**: Immediate feedback and automatic UI refresh

This implementation enables users to resolve conflicts by applying canonical 5-child sets to entire label groups, following the monolith semantics of first child in existing row + (k-1) new rows with prefix replication and deeper column clearing.
