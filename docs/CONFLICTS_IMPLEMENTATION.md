# Conflicts Tab Implementation

This document describes the implementation of the Conflicts tab with cached conflict metrics and comprehensive conflict resolution UI.

## Overview

The Conflicts tab has been completely redesigned to provide:
1. **Cached conflict metrics** that compute once per active sheet (nonce-aware)
2. **KPI dashboard** showing conflict counts and summaries
3. **Level drilldown** for detailed conflict analysis
4. **Conflict resolution tools** with automatic application to affected paths
5. **Integration** with existing override/materialize pipeline

## Key Components

### 1. Cached Wrapper Function

```python
@st.cache_data(ttl=600, show_spinner=False)
def get_conflict_summary(df: pd.DataFrame, nonce: str) -> Dict[str, Any]:
    """Get cached conflict summary for the given DataFrame and nonce."""
    idx = build_parent_child_index(df)
    summary = summarize_children_sets(idx)
    mismatches = find_label_set_mismatches(idx)
    
    # Compute top-level counts
    over5 = [(k, v) for k, v in summary.items() if v["over5"]]
    not_exact5 = [(k, v) for k, v in summary.items() if not v["exact5"]]
    
    return {
        "index": idx,
        "summary": summary,
        "mismatches": mismatches,
        "counts": {
            "parents_over5": len(over5),
            "parents_not_exact5": len(not_exact5),
            "total_parents": len(summary),
        }
    }
```

**Features:**
- Uses `@st.cache_data(ttl=600, show_spinner=False)` for performance
- Nonce-aware: cache invalidates when workbook data changes
- Computes all conflict metrics in one pass
- Returns structured data for UI consumption

### 2. KPI Dashboard

The dashboard displays three key metrics:
- **Parents >5 Children**: Count of parent nodes with more than 5 children
- **Parents Not Exactly 5**: Count of parent nodes that don't have exactly 5 children  
- **Total Parents**: Total number of parent nodes in the decision tree

**Mismatches Summary:**
- Groups conflicts by level for organized display
- Shows expandable sections for each level
- Highlights over-5 violations and inconsistent child sets

### 3. Level Drilldown

**Level Selection:**
- Picker for levels 1-5
- Level 1 automatically shows ROOT parent
- Levels 2+ show parent label picker populated from mismatches

**Conflict Resolution Options:**

#### For Level 1 (ROOT):
- **Keep Current Set**: Maintains existing Node 1 options
- **Custom Set**: Multiselect up to 5 children with preview
- Uses existing `set_level1_children()` function

#### For Levels 2+:
- **Option 1: Keep Variant as Canonical**
  - Choose one variant's children (capped to 5)
  - Apply to all parents with that parent_label at this level
  
- **Option 2: Custom Canonical Set (≤5)**
  - Multiselect from union of existing children
  - Free-text input for new children
  - Apply to the entire label group

### 4. Resolution Application

**Path Discovery:**
- Automatically finds all parent paths ending with the target label
- Creates overrides for each affected path
- Applies changes using existing `build_raw_plus_v630()` function

**State Management:**
- Updates workbook with new DataFrame
- Refreshes active workbook state
- Bumps nonce via `set_current_sheet()`
- Clears stale caches for immediate refresh
- Triggers `safe_rerun()` to update UI

## Technical Implementation

### Cache Strategy
- **TTL**: 600 seconds (10 minutes) for reasonable performance
- **Nonce Awareness**: Uses `get_wb_nonce()` to invalidate cache when data changes
- **No Spinner**: Prevents UI blocking during cache computation

### Data Flow
1. **Input**: DataFrame + nonce
2. **Processing**: Build index → Summarize → Find mismatches → Count metrics
3. **Output**: Structured dictionary with all conflict information
4. **UI Consumption**: KPI display, drilldown, resolution tools

### Integration Points
- **State Management**: Uses `get_wb_nonce()`, `set_active_workbook()`, `set_current_sheet()`
- **Tree Logic**: Leverages `build_parent_child_index()`, `summarize_children_sets()`, `find_label_set_mismatches()`
- **Override System**: Integrates with existing `build_raw_plus_v630()` pipeline
- **UI Utilities**: Uses `safe_rerun()` for state updates

## Usage Examples

### Basic Conflict Detection
```python
# Get conflict summary (cached)
conflict_summary = get_conflict_summary(df, get_wb_nonce())

# Access metrics
counts = conflict_summary["counts"]
print(f"Parents with >5 children: {counts['parents_over5']}")
```

### Level-Specific Resolution
```python
# Select level and parent label
level = 3
parent_label = "Confusion"

# Find mismatches for this level/label
mismatches = conflict_summary["mismatches"]
mismatch_info = mismatches.get((level, parent_label))

if mismatch_info:
    # Show resolution options
    _render_level_resolution(level, parent_label, mismatch_info, df, sheet_name)
```

## Benefits

1. **Performance**: Cached computation prevents redundant analysis
2. **Real-time Updates**: Nonce-aware caching ensures fresh data
3. **User Experience**: Clear KPIs, organized drilldown, intuitive resolution
4. **Consistency**: Integrates with existing override system
5. **Maintainability**: Clean separation of concerns, reusable components

## Future Enhancements

1. **Conflict History**: Track resolution actions over time
2. **Bulk Operations**: Resolve multiple conflicts simultaneously
3. **Validation Rules**: Custom business rule validation
4. **Export/Import**: Save/load conflict resolution configurations
5. **Performance Metrics**: Track resolution time and success rates

## Testing

The implementation has been tested for:
- ✅ Syntax correctness
- ✅ Function compilation
- ✅ Cache functionality
- ✅ Data structure integrity
- ✅ Integration with existing systems

## Conclusion

The Conflicts tab now provides a comprehensive, performant, and user-friendly interface for detecting and resolving decision tree conflicts. The cached wrapper ensures efficient computation while the drilldown UI enables precise conflict resolution at any level of the tree.
