# Monolith-Style Tree Analysis Implementation

This document describes the implementation of monolith-style tree analysis with root + five nodes, providing comprehensive decision tree analysis capabilities.

## Overview

The monolith-style analysis provides a complete view of decision tree structures by:
- **Root Column**: Uses "Vital Measurement" as the root level
- **Downstream Levels**: Analyzes "Node 1" through "Node 5" (exactly five levels)
- **Row Multiplication Tolerance**: Handles cases where many rows share the same first N nodes and diverge deeper
- **Accurate Grouping**: Prevents double-counting children due to deeper divergence

## Implementation Components

### 1. Constants and Configuration

**File: `utils/constants.py`**
```python
ROOT_COL = "Vital Measurement"
LEVEL_COLS = ["Node 1", "Node 2", "Node 3", "Node 4", "Node 5"]
MAX_LEVELS = 5
MAX_CHILDREN_PER_PARENT = 5
ROOT_PARENT_LABEL = "<ROOT>"
```

### 2. Core Analysis Functions

**File: `logic/tree.py`**

#### `_row_nodes_with_root(row: pd.Series) -> List[str]`
- Returns a 6-length list: `[root, n1, n2, n3, n4, n5]`
- Applies normalization and safe blank handling for ragged rows
- Ensures consistent length regardless of input data structure

#### `build_parent_child_index_with_root(df: pd.DataFrame) -> Dict[Tuple[int, str], Counter]`
- **Level 1**: Parent is ROOT (`<ROOT>`), child is Node 1
- **Levels 2-5**: Parent path is `'root>Node1>...>Node(L-1)'`, child is Node L
- **Row Multiplication Tolerance**: Aggregates via Counter, derives unique children from counter keys
- **No Double-Counting**: Children are counted uniquely per parent path

#### `detect_full_path_duplicates(df: pd.DataFrame) -> List[Tuple[str, int]]`
- Identifies full paths (root + 5 nodes) that occur multiple times
- Returns `[(full_path_string, count), ...]` for paths with >1 occurrence
- Useful for understanding data duplication patterns

#### `analyze_decision_tree_with_root(df: pd.DataFrame) -> Dict[str, Any]`
- **Comprehensive Analysis**: Single function call for complete tree analysis
- **UI-Ready Output**: Structured data for immediate consumption
- **Multiple Metrics**: Index, summary, mismatches, root set, duplicates, counts

### 3. Shared Analysis Module

**File: `ui/analysis.py`**
```python
@st.cache_data(ttl=600, show_spinner=False)
def get_conflict_summary_with_root(df: pd.DataFrame, nonce: str) -> Dict[str, Any]:
    """Get cached decision tree analysis summary for the given DataFrame and nonce."""
    return analyze_decision_tree_with_root(df)
```

**Features:**
- **Cached Computation**: 600-second TTL for performance
- **Nonce Awareness**: Cache invalidates when workbook data changes
- **Shared Usage**: Used by both Conflicts and Workspace tabs

### 4. UI Integration

#### Conflicts Tab (`ui/tabs/conflicts.py`)
- **Root Children Display**: Shows Level 1 children with status indicators
- **Full-Path Duplicates**: Displays duplicate paths with occurrence counts
- **Enhanced KPIs**: Uses monolith analysis for accurate metrics
- **Level Drilldown**: Maintains existing conflict resolution functionality

#### Workspace Tab (`ui/tabs/workspace.py`)
- **Accurate Coverage Metrics**: Uses monolith analysis for precise parent counts
- **Level 1 Special Handling**: ROOT always has exactly 1 parent
- **Real Parent Counts**: Levels 2+ count distinct parent paths, not inflated by row duplication

## Data Flow and Processing

### 1. Input Processing
```
DataFrame (Vital Measurement + Node 1-5) 
    ↓
_row_nodes_with_root() → [root, n1, n2, n3, n4, n5]
    ↓
Normalization and blank handling
```

### 2. Index Building
```
For each row:
    Level 1: ROOT → Node 1
    Level 2: root>Node1 → Node 2
    Level 3: root>Node1>Node2 → Node 3
    Level 4: root>Node1>Node2>Node3 → Node 4
    Level 5: root>Node1>Node2>Node3>Node4 → Node 5
```

### 3. Aggregation Strategy
- **Counter-Based**: Each parent path maintains a Counter of children
- **Frequency Preservation**: Row counts are maintained for display purposes
- **Unique Children**: Child sets are derived from Counter keys, not total counts
- **Multiplication Tolerance**: Deeper divergence doesn't inflate parent counts

## Key Benefits

### 1. **Accuracy**
- **Real Parent Counts**: Reflects actual distinct parent paths
- **No Inflation**: Row duplication doesn't affect parent counts
- **Consistent Logic**: Same analysis approach across all tabs

### 2. **Performance**
- **Cached Computation**: Prevents redundant analysis
- **Single Pass**: All metrics computed in one DataFrame iteration
- **Efficient Aggregation**: Uses Counter for optimal memory usage

### 3. **User Experience**
- **Comprehensive View**: Complete tree analysis in one place
- **Clear Metrics**: Accurate KPIs and coverage statistics
- **Duplicate Awareness**: Shows full-path duplicates for transparency

### 4. **Maintainability**
- **Shared Logic**: Single source of truth for tree analysis
- **Modular Design**: Clear separation of concerns
- **Consistent API**: Uniform interface across all components

## Usage Examples

### Basic Analysis
```python
from logic.tree import analyze_decision_tree_with_root

# Get complete analysis
result = analyze_decision_tree_with_root(df)

# Access key metrics
root_children = result["root_children"]
counts = result["counts"]
mismatches = result["mismatches"]
duplicates = result["duplicates_full_path"]
```

### Cached Analysis (UI)
```python
from ui.analysis import get_conflict_summary_with_root
from utils.state import get_wb_nonce

# Get cached analysis
res = get_conflict_summary_with_root(df, get_wb_nonce())

# Use in UI components
st.metric("Parents > 5 children", res["counts"]["parents_over5"])
st.metric("Parents not exactly 5", res["counts"]["parents_not_exact5"])
st.metric("Total parents", res["counts"]["total_parents"])
```

### Coverage Metrics
```python
# Level 1: exactly 1 parent (ROOT)
l1_total = 1 if res["summary"].get((1, "<ROOT>")) else 0
l1_covered = 1 if len(res["root_children"]) == 5 else 0

# Levels 2+: count distinct parent paths
for (L, _pth), info in res["summary"].items():
    if L == 1:  # Skip ROOT
        continue
    level_totals[L] += 1
    if info["exact5"]:
        level_covered[L] += 1
```

## Testing and Validation

### Compilation Tests
- ✅ `logic/tree.py` - All functions compile correctly
- ✅ `ui/analysis.py` - Shared analysis module compiles correctly
- ✅ `ui/tabs/conflicts.py` - Conflicts tab compiles correctly
- ✅ `ui/tabs/workspace.py` - Workspace tab compiles correctly

### Functionality Tests
- ✅ `_row_nodes_with_root()` - Correctly processes rows with root + nodes
- ✅ `build_parent_child_index_with_root()` - Builds accurate parent-child index
- ✅ `detect_full_path_duplicates()` - Identifies duplicate paths correctly
- ✅ `analyze_decision_tree_with_root()` - Provides comprehensive analysis
- ✅ Cached wrapper - Functions correctly with nonce awareness

### Test Results
```
Sample DataFrame: 5 rows, 6 columns (Vital Measurement + Node 1-5)
Index keys: 13 total entries
Root children: ['Mild', 'Severe', 'Acute'] (3 children)
Full-path duplicates: 2 entries found
Parents >5: 0
Parents not exactly 5: 13
Total parents: 13
```

## Integration Points

### 1. **State Management**
- Uses `get_wb_nonce()` for cache invalidation
- Integrates with existing workbook/sheet state
- Maintains consistency across tab refreshes

### 2. **Existing Functions**
- Leverages `normalize_text()` and `normalize_child_set()` helpers
- Uses existing `find_label_set_mismatches()` logic
- Integrates with override/materialize pipeline

### 3. **UI Components**
- **Conflicts Tab**: Enhanced with root analysis and duplicate detection
- **Workspace Tab**: Accurate coverage metrics using monolith analysis
- **Shared Analysis**: Consistent data across all components

## Future Enhancements

### 1. **Performance Optimization**
- **Parallel Processing**: Multi-threaded analysis for large datasets
- **Incremental Updates**: Delta analysis for changed data only
- **Memory Optimization**: Streaming analysis for very large trees

### 2. **Advanced Analysis**
- **Path Analytics**: Most common paths and patterns
- **Conflict Prediction**: Machine learning for conflict detection
- **Optimization Suggestions**: Automated conflict resolution recommendations

### 3. **Export and Reporting**
- **Analysis Reports**: PDF/Excel export of tree analysis
- **Visualization**: Interactive tree diagrams with conflict highlighting
- **Audit Trails**: Track analysis history and changes over time

## Conclusion

The monolith-style tree analysis provides a robust, accurate, and performant foundation for decision tree analysis. By treating each row as a full path and using intelligent aggregation strategies, it delivers:

- **Accurate Metrics**: Real parent counts without inflation from row duplication
- **Comprehensive Analysis**: Complete tree structure analysis in a single call
- **Performance**: Cached computation with nonce-aware invalidation
- **Consistency**: Uniform analysis approach across all UI components
- **Maintainability**: Clean, modular design with clear separation of concerns

This implementation successfully addresses the requirements for root-based analysis, row multiplication tolerance, and accurate parent counting while maintaining excellent performance and user experience.
