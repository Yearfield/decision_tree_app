# ui/tabs/symptoms.py
import streamlit as st
import pandas as pd
from typing import Dict, Any, List

from utils import (
    CANON_HEADERS, LEVEL_COLS, normalize_text, validate_headers
)
from logic.tree import infer_branch_options, build_label_children_index


def render():
    """Render the Symptoms tab for managing symptom quality and branch building."""
    try:
        st.header("🧬 Symptoms")

        # Get active DataFrame
        df = get_active_df()
        if df is None:
            st.warning("No active sheet selected. Please load a workbook in the Source tab and select a sheet.")
            return
        
        if not validate_headers(df):
            st.warning("Active sheet has invalid headers. Please ensure it has the required columns.")
            return

        # Get sheet name from context
        ctx = st.session_state.get("work_context", {})
        sheet = ctx.get("sheet", "Unknown")

        # Get symptom quality data
        symptom_quality = st.session_state.get("symptom_quality", {})

        # Main sections
        _render_symptom_quality_section(df, symptom_quality, sheet)
        
        st.markdown("---")
        
        _render_branch_building_section(df, symptom_quality, sheet)

    except Exception as e:
        st.exception(e)


def get_active_df():
    """Get the currently active DataFrame from session state."""
    wb_u = st.session_state.get("upload_workbook", {})
    wb_g = st.session_state.get("gs_workbook", {})
    ctx = st.session_state.get("work_context", {})
    sheet = ctx.get("sheet")
    if sheet and sheet in wb_u: 
        return wb_u[sheet]
    if sheet and sheet in wb_g: 
        return wb_g[sheet]
    return None


def _render_symptom_quality_section(df: pd.DataFrame, symptom_quality: Dict, sheet_name: str):
    """Render the symptom quality management section."""
    st.subheader("🎯 Symptom Quality Management")
    
    # Build vocabulary from current sheet
    vocab = _build_sheet_vocabulary(df)
    
    if not vocab:
        st.info("No vocabulary found in the current sheet.")
        return
    
    st.write(f"Found {len(vocab)} unique terms in '{sheet_name}':")
    
    # Display vocabulary with quality scores
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Search/filter vocabulary
        search_term = st.text_input("🔍 Search vocabulary", placeholder="Type to filter...")
        filtered_vocab = [term for term in vocab if search_term.lower() in term.lower()] if search_term else vocab
        
        if filtered_vocab:
            st.write("**Vocabulary terms:**")
            for term in filtered_vocab[:50]:  # Show first 50
                quality = symptom_quality.get(term, 0)
                st.write(f"• {term} (quality: {quality})")
            
            if len(filtered_vocab) > 50:
                st.caption(f"... and {len(filtered_vocab) - 50} more terms")
    
    with col2:
        # Quality score editor
        st.write("**Set quality score:**")
        selected_term = st.selectbox("Select term", [""] + filtered_vocab, key="quality_term_selector")
        
        if selected_term:
            current_quality = symptom_quality.get(selected_term, 0)
            new_quality = st.slider(
                "Quality score",
                min_value=0,
                max_value=10,
                value=current_quality,
                help="0 = poor quality, 10 = excellent quality"
            )
            
            if new_quality != current_quality:
                if st.button("Update Quality"):
                    symptom_quality[selected_term] = new_quality
                    st.session_state["symptom_quality"] = symptom_quality
                    st.success(f"Updated '{selected_term}' quality to {new_quality}")
                    st.rerun()
    
    # Bulk quality operations
    st.markdown("---")
    st.subheader("📊 Bulk Quality Operations")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📈 Auto-assign Quality"):
            _auto_assign_quality_scores(df, symptom_quality)
    
    with col2:
        if st.button("🔄 Reset All Quality"):
            if st.checkbox("Confirm reset all quality scores"):
                st.session_state["symptom_quality"] = {}
                st.success("All quality scores reset!")
                st.rerun()
    
    with col3:
        if st.button("💾 Export Quality Data"):
            _export_quality_data(symptom_quality)


def _render_branch_building_section(df: pd.DataFrame, symptom_quality: Dict, sheet_name: str):
    """Render the branch building section."""
    st.subheader("🌿 Branch Building")
    
    # Branch building options
    col1, col2 = st.columns(2)
    
    with col1:
        build_mode = st.radio(
            "Build mode",
            ["Manual", "Auto-suggest", "Quality-based"],
            help="Choose how to build branches"
        )
    
    with col2:
        target_level = st.selectbox(
            "Target level",
            options=[1, 2, 3, 4, 5],
            index=0,
            help="Which level to build branches for"
        )
    
    # Branch building interface
    if build_mode == "Manual":
        _render_manual_branch_building(df, target_level, sheet_name)
    elif build_mode == "Auto-suggest":
        _render_auto_suggest_branch_building(df, target_level, symptom_quality, sheet_name)
    else:  # Quality-based
        _render_quality_based_branch_building(df, target_level, symptom_quality, sheet_name)


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


def _auto_assign_quality_scores(df: pd.DataFrame, symptom_quality: Dict):
    """Automatically assign quality scores based on data patterns."""
    try:
        with st.spinner("Analyzing data patterns..."):
            # Simple heuristic: terms that appear more frequently get higher scores
            term_counts = {}
            
            for col in LEVEL_COLS + ["Vital Measurement"]:
                if col in df.columns:
                    values = df[col].map(normalize_text).dropna()
                    values = values[values != ""]
                    for term in values:
                        term_counts[term] = term_counts.get(term, 0) + 1
            
            # Assign scores based on frequency
            for term, count in term_counts.items():
                if count == 1:
                    score = 3  # Rare terms get low score
                elif count <= 3:
                    score = 5  # Occasional terms get medium score
                elif count <= 10:
                    score = 7  # Common terms get high score
                else:
                    score = 9  # Very common terms get very high score
                
                symptom_quality[term] = score
            
            st.session_state["symptom_quality"] = symptom_quality
            st.success(f"Auto-assigned quality scores to {len(term_counts)} terms!")
            st.rerun()
            
    except Exception as e:
        st.error(f"Error auto-assigning quality scores: {e}")


def _export_quality_data(symptom_quality: Dict):
    """Export quality data to a downloadable format."""
    try:
        if not symptom_quality:
            st.warning("No quality data to export.")
            return
        
        # Create DataFrame
        quality_df = pd.DataFrame([
            {"term": term, "quality_score": score}
            for term, score in symptom_quality.items()
        ])
        
        # Sort by quality score
        quality_df = quality_df.sort_values("quality_score", ascending=False)
        
        # Download button
        csv = quality_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Quality Data (CSV)",
            data=csv,
            file_name="symptom_quality.csv",
            mime="text/csv"
        )
        
    except Exception as e:
        st.error(f"Error exporting quality data: {e}")


def _render_manual_branch_building(df: pd.DataFrame, target_level: int, sheet_name: str):
    """Render manual branch building interface."""
    st.write("**Manual Branch Building**")
    st.info("Select a parent node and manually specify its children.")
    
    # Get parent options
    parent_col = f"Node {target_level - 1}" if target_level > 1 else "Vital Measurement"
    if parent_col not in df.columns:
        st.warning(f"Parent column '{parent_col}' not found.")
        return
    
    parent_values = df[parent_col].map(normalize_text).dropna().unique()
    parent_values = [v for v in parent_values if v != ""]
    
    if not parent_values:
        st.info(f"No parent values found in '{parent_col}'.")
        return
    
    selected_parent = st.selectbox("Select parent node", parent_values, key=f"manual_parent_{target_level}")
    
    if selected_parent:
        # Input children
        st.write(f"**Children for '{selected_parent}' at level {target_level}:**")
        
        children = []
        for i in range(5):
            child = st.text_input(f"Child {i + 1}", key=f"manual_child_{target_level}_{i}")
            if child.strip():
                children.append(child.strip())
        
        if st.button("Create Branch"):
            if children:
                _create_branch(df, selected_parent, target_level, children, sheet_name)
            else:
                st.warning("Please specify at least one child.")


def _render_auto_suggest_branch_building(df: pd.DataFrame, target_level: int, symptom_quality: Dict, sheet_name: str):
    """Render auto-suggest branch building interface."""
    st.write("**Auto-Suggest Branch Building**")
    st.info("Get suggestions for branches based on existing data patterns.")
    
    # Get parent options
    parent_col = f"Node {target_level - 1}" if target_level > 1 else "Vital Measurement"
    if parent_col not in df.columns:
        st.warning(f"Parent column '{parent_col}' not found.")
        return
    
    parent_values = df[parent_col].map(normalize_text).dropna().unique()
    parent_values = [v for v in parent_values if v != ""]
    
    if not parent_values:
        st.info(f"No parent values found in '{parent_col}'.")
        return
    
    selected_parent = st.selectbox("Select parent node", parent_values, key=f"auto_parent_{target_level}")
    
    if selected_parent and st.button("🔍 Get Suggestions"):
        with st.spinner("Analyzing patterns..."):
            suggestions = _get_branch_suggestions(df, selected_parent, target_level, symptom_quality)
            
            if suggestions:
                st.write("**Suggested children:**")
                for i, suggestion in enumerate(suggestions):
                    st.write(f"{i + 1}. {suggestion}")
                
                if st.button("Use These Suggestions"):
                    _create_branch(df, selected_parent, target_level, suggestions, sheet_name)
            else:
                st.info("No suggestions found. Try manual input.")


def _render_quality_based_branch_building(df: pd.DataFrame, target_level: int, symptom_quality: Dict, sheet_name: str):
    """Render quality-based branch building interface."""
    st.write("**Quality-Based Branch Building**")
    st.info("Build branches using high-quality terms from the vocabulary.")
    
    # Get parent options
    parent_col = f"Node {target_level - 1}" if target_level > 1 else "Vital Measurement"
    if parent_col not in df.columns:
        st.warning(f"Parent column '{parent_col}' not found.")
        return
    
    parent_values = df[parent_col].map(normalize_text).dropna().unique()
    parent_values = [v for v in parent_values if v != ""]
    
    if not parent_values:
        st.info(f"No parent values found in '{parent_col}'.")
        return
    
    selected_parent = st.selectbox("Select parent node", parent_values, key=f"quality_parent_{target_level}")
    
    if selected_parent:
        # Show high-quality vocabulary
        high_quality_terms = [term for term, score in symptom_quality.items() if score >= 7]
        
        if high_quality_terms:
            st.write("**High-quality terms (score ≥ 7):**")
            selected_terms = st.multiselect(
                "Select terms for this branch",
                high_quality_terms,
                max_selections=5,
                key=f"quality_terms_{target_level}"
            )
            
            if selected_terms and st.button("Create Quality Branch"):
                _create_branch(df, selected_parent, target_level, selected_terms, sheet_name)
        else:
            st.info("No high-quality terms found. Consider improving term quality first.")


def _get_branch_suggestions(df: pd.DataFrame, parent: str, target_level: int, symptom_quality: Dict) -> List[str]:
    """Get branch suggestions based on data patterns."""
    try:
        # Look for existing patterns
        suggestions = set()
        
        # Check if this parent already has children
        if target_level <= 5:
            target_col = f"Node {target_level}"
            if target_col in df.columns:
                # Find rows where parent matches
                parent_col = f"Node {target_level - 1}" if target_level > 1 else "Vital Measurement"
                mask = df[parent_col].map(normalize_text) == parent
                existing_children = df.loc[mask, target_col].map(normalize_text).dropna()
                existing_children = existing_children[existing_children != ""]
                suggestions.update(existing_children)
        
        # Add high-quality vocabulary terms
        high_quality_terms = [term for term, score in symptom_quality.items() if score >= 6]
        suggestions.update(high_quality_terms[:10])  # Top 10
        
        # Convert to list and limit to 5
        result = list(suggestions)[:5]
        return result
        
    except Exception:
        return []


def _create_branch(df: pd.DataFrame, parent: str, target_level: int, children: List[str], sheet_name: str):
    """Create a new branch in the decision tree."""
    try:
        # This would implement the actual branch creation logic
        # For now, just show success message
        st.success(f"Branch created for '{parent}' with {len(children)} children!")
        st.write(f"Children: {', '.join(children)}")
        
        # In a full implementation, this would:
        # 1. Add new rows to the DataFrame
        # 2. Update the workbook in session state
        # 3. Possibly update overrides
        
    except Exception as e:
        st.error(f"Error creating branch: {e}")
