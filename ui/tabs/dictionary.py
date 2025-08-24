# ui/tabs/dictionary.py
import streamlit as st
import pandas as pd
from typing import Dict, Any, List

from utils import (
    CANON_HEADERS, LEVEL_COLS, normalize_text, validate_headers
)
import utils.state as USTATE
from ui.utils.rerun import safe_rerun


def render():
    """Render the Dictionary tab for managing vocabulary and term definitions."""
    
    # Add guard and debug expander
    from ui.utils.guards import ensure_active_workbook_and_sheet
    ok, df = ensure_active_workbook_and_sheet("Dictionary")
    if not ok:
        return
    
    # Debug state expander
    import json
    with st.expander("🛠 Debug: Session State (tab)", expanded=False):
        ss = {k: type(v).__name__ for k,v in st.session_state.items()}
        st.code(json.dumps(ss, indent=2))
    
    try:
        st.header("📖 Dictionary")
        
        # Get current sheet name for display
        sheet = USTATE.get_current_sheet()
        
        # Status badge
        has_wb, sheet_count, current_sheet = USTATE.get_workbook_status()
        if has_wb and current_sheet:
            st.caption(f"Workbook: ✅ {sheet_count} sheet(s) • Active: **{current_sheet}**")
        else:
            st.caption("Workbook: ❌ not loaded")

        # Get dictionary data
        dictionary = st.session_state.get("term_dictionary", {})

        # Main sections
        _render_vocabulary_overview(df, sheet)
        
        st.markdown("---")
        
        _render_term_definitions(df, dictionary, sheet)
        
        st.markdown("---")
        
        _render_dictionary_management(dictionary, sheet)

    except Exception as e:
        st.exception(e)


def _render_vocabulary_overview(df: pd.DataFrame, sheet_name: str):
    """Render the vocabulary overview section."""
    st.subheader("📚 Vocabulary Overview")
    
    # Build vocabulary from current sheet
    vocab = _build_sheet_vocabulary(df)
    
    if not vocab:
        st.info("No vocabulary found in the current sheet.")
        return
    
    st.write(f"Found {len(vocab)} unique terms in '{sheet_name}':")
    
    # Display vocabulary statistics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        vm_terms = _get_vm_terms(df)
        st.metric("Vital Measurements", len(vm_terms))
    
    with col2:
        node_terms = _get_node_terms(df)
        st.metric("Node Terms", len(node_terms))
    
    with col3:
        total_terms = len(vocab)
        st.metric("Total Terms", total_terms)
    
    # Show vocabulary breakdown
    with st.expander("📊 Vocabulary Breakdown", expanded=False):
        st.write("**By column:**")
        for col in ["Vital Measurement"] + LEVEL_COLS:
            if col in df.columns:
                col_terms = df[col].map(normalize_text).dropna().unique()
                col_terms = [t for t in col_terms if t != ""]
                st.write(f"• {col}: {len(col_terms)} terms")
        
        st.write("**Sample terms:**")
        sample_terms = vocab[:20]  # Show first 20
        st.write(", ".join(sample_terms))
        if len(vocab) > 20:
            st.caption(f"... and {len(vocab) - 20} more terms")


def _render_term_definitions(df: pd.DataFrame, dictionary: Dict, sheet_name: str):
    """Render the term definitions section."""
    st.subheader("📝 Term Definitions")
    
    # Get vocabulary
    vocab = _build_sheet_vocabulary(df)
    
    if not vocab:
        st.info("No vocabulary to define.")
        return
    
    # Term selector
    selected_term = st.selectbox(
        "Select term to define",
        [""] + sorted(vocab),
        help="Choose a term to view or edit its definition"
    )
    
    if selected_term:
        current_definition = dictionary.get(selected_term, "")
        
        # Show term context
        st.write(f"**Context for '{selected_term}':**")
        context = _get_term_context(df, selected_term)
        if context:
            for col, count in context.items():
                st.write(f"• {col}: appears {count} times")
        else:
            st.info("Term not found in current data.")
        
        # Definition editor
        st.markdown("---")
        st.write("**Definition:**")
        new_definition = st.text_area(
            "Edit definition",
            value=current_definition,
            height=100,
            placeholder="Enter a clear definition for this term...",
            key=f"def_{selected_term}"
        )
        
        if new_definition != current_definition:
            col1, col2 = st.columns([3, 1])
            with col1:
                if st.button("💾 Save Definition", key=f"save_{selected_term}"):
                    dictionary[selected_term] = new_definition
                    st.session_state["term_dictionary"] = dictionary
                    st.success("Definition saved!")
                    safe_rerun()
            
            with col2:
                if st.button("🗑️ Clear", key=f"clear_{selected_term}"):
                    safe_rerun()
        
        # Show existing definition if any
        if current_definition:
            st.markdown("---")
            st.write("**Current definition:**")
            st.info(current_definition)


def _render_dictionary_management(dictionary: Dict, sheet_name: str):
    """Render the dictionary management section."""
    st.subheader("⚙️ Dictionary Management")
    
    # Dictionary statistics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        defined_terms = len(dictionary)
        st.metric("Defined Terms", defined_terms)
    
    with col2:
        if defined_terms > 0:
            avg_length = sum(len(defn) for defn in dictionary.values()) / defined_terms
            st.metric("Avg Definition Length", f"{avg_length:.1f} chars")
        else:
            st.metric("Avg Definition Length", "N/A")
    
    with col3:
        if defined_terms > 0:
            st.metric("Dictionary Coverage", f"{(defined_terms / 100):.1f}%")
        else:
            st.metric("Dictionary Coverage", "0%")
    
    # Bulk operations
    st.markdown("---")
    st.subheader("📦 Bulk Operations")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📥 Import Dictionary"):
            _import_dictionary(dictionary)
    
    with col2:
        if st.button("📤 Export Dictionary"):
            _export_dictionary(dictionary)
    
    with col3:
        if st.button("🔄 Reset Dictionary"):
            if st.checkbox("Confirm reset all definitions"):
                st.session_state["term_dictionary"] = {}
                st.success("Dictionary reset!")
                safe_rerun()
    
    # Dictionary search
    st.markdown("---")
    st.subheader("🔍 Search Dictionary")
    
    search_query = st.text_input("Search terms or definitions", placeholder="Type to search...")
    
    if search_query:
        search_results = _search_dictionary(dictionary, search_query)
        
        if search_results:
            st.write(f"Found {len(search_results)} matching terms:")
            for term, definition in search_results:
                with st.expander(f"**{term}**", expanded=False):
                    st.write(definition)
        else:
            st.info("No matches found.")


def _build_sheet_vocabulary(df: pd.DataFrame) -> List[str]:
    """Build vocabulary from the current sheet."""
    try:
        vocab = set()
        
        # Collect all non-empty values from node columns
        for col in LEVEL_COLS:
            if col in df.columns:
                values = df[col].map(normalize_text).dropna()
                values = values[values != ""]
                vocab.update(values)
        
        # Also collect from Vital Measurement
        if "Vital Measurement" in df.columns:
            vm_values = df["Vital Measurement"].map(normalize_text).dropna()
            vm_values = vm_values[vm_values != ""]
            vocab.update(vm_values)
        
        return sorted(list(vocab))
    except Exception:
        return []


def _get_vm_terms(df: pd.DataFrame) -> List[str]:
    """Get terms from Vital Measurement column."""
    try:
        if "Vital Measurement" in df.columns:
            values = df["Vital Measurement"].map(normalize_text).dropna()
            values = values[values != ""]
            return list(values.unique())
        return []
    except Exception:
        return []


def _get_node_terms(df: pd.DataFrame) -> List[str]:
    """Get terms from Node columns."""
    try:
        node_terms = set()
        for col in LEVEL_COLS:
            if col in df.columns:
                values = df[col].map(normalize_text).dropna()
                values = values[values != ""]
                node_terms.update(values)
        return list(node_terms)
    except Exception:
        return []


def _get_term_context(df: pd.DataFrame, term: str) -> Dict[str, int]:
    """Get context information for a specific term."""
    try:
        context = {}
        
        # Check Vital Measurement
        if "Vital Measurement" in df.columns:
            vm_count = (df["Vital Measurement"].map(normalize_text) == term).sum()
            if vm_count > 0:
                context["Vital Measurement"] = vm_count
        
        # Check Node columns
        for col in LEVEL_COLS:
            if col in df.columns:
                node_count = (df[col].map(normalize_text) == term).sum()
                if node_count > 0:
                    context[col] = node_count
        
        return context
    except Exception:
        return {}


def _import_dictionary(dictionary: Dict):
    """Import dictionary from file."""
    st.info("Dictionary import functionality would be implemented here.")
    st.write("This would allow importing definitions from CSV, JSON, or other formats.")


def _export_dictionary(dictionary: Dict):
    """Export dictionary to file."""
    try:
        if not dictionary:
            st.warning("No dictionary to export.")
            return
        
        # Create DataFrame
        dict_df = pd.DataFrame([
            {"term": term, "definition": definition}
            for term, definition in dictionary.items()
        ])
        
        # Sort by term
        dict_df = dict_df.sort_values("term")
        
        # Download button
        csv = dict_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Dictionary (CSV)",
            data=csv,
            file_name="term_dictionary.csv",
            mime="text/csv"
        )
        
    except Exception as e:
        st.error(f"Error exporting dictionary: {e}")


def _search_dictionary(dictionary: Dict, query: str) -> List[tuple]:
    """Search dictionary for terms or definitions matching query."""
    try:
        query_lower = query.lower()
        results = []
        
        for term, definition in dictionary.items():
            if (query_lower in term.lower() or 
                query_lower in definition.lower()):
                results.append((term, definition))
        
        return results
    except Exception:
        return []
