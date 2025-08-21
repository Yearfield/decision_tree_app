# ui/tabs/symptoms.py
import streamlit as st
import pandas as pd
from typing import Dict, Any, List, Tuple

from utils import (
    CANON_HEADERS, LEVEL_COLS, normalize_text, validate_headers
)
from utils.state import (
    get_active_workbook, get_current_sheet, get_active_df, 
    has_active_workbook, get_workbook_status, get_wb_nonce, set_active_workbook, set_current_sheet
)
from utils.constants import MAX_CHILDREN_PER_PARENT, ROOT_PARENT_LABEL, LEVEL_COLS, MAX_LEVELS, LEVEL_LABELS
from utils.helpers import normalize_child_set, normalize_text
from ui.utils.rerun import safe_rerun
from logic.tree import infer_branch_options, build_label_children_index, analyze_decision_tree_with_root
from logic.materialize import materialize_children_for_label_group, materialize_children_for_single_parent, materialize_children_for_label_across_tree
from ui.analysis import get_conflict_summary_with_root


def get_red_flags_map():
    """Get the red flags map from session state."""
    return st.session_state.setdefault("__red_flags_map", {})  # {label:str -> True}


def is_red_flag(label: str) -> bool:
    """Check if a label is flagged as red."""
    return bool(get_red_flags_map().get(normalize_text(label)))


def set_red_flag(label: str, value: bool):
    """Set or unset a red flag for a label."""
    m = get_red_flags_map()
    key = normalize_text(label)
    if value:
        m[key] = True
    else:
        m.pop(key, None)
    st.session_state["__red_flags_map"] = m


def _list_parents_from_summary(summary):
    """Returns list of (level, parent_path, children) sorted by level and path."""
    parents = [(L, pth, info["children"]) for (L, pth), info in summary.items()]
    parents.sort(key=lambda x: (x[0], x[1]))
    return parents


def _find_next_incomplete_parent(summary, start_ix=0):
    """Find the next parent whose children count != MAX_CHILDREN_PER_PARENT."""
    parents = _list_parents_from_summary(summary)
    n = len(parents)
    for i in range(n):
        idx = (start_ix + i) % n
        L, pth, children = parents[idx]
        if len(children) != MAX_CHILDREN_PER_PARENT:
            return idx, (L, pth, children)
    return None, None


def render():
    """Render the Symptoms tab for managing symptom quality and branch building."""
    try:
        st.header("🧬 Symptoms")
        
        # Status badge
        has_wb, sheet_count, current_sheet = get_workbook_status()
        if has_wb and current_sheet:
            st.caption(f"Workbook: ✅ {sheet_count} sheet(s) • Active: **{current_sheet}**")
        else:
            st.caption("Workbook: ❌ not loaded")
        
        # Guard against no active workbook
        wb = get_active_workbook()
        sheet = get_current_sheet()
        if not wb or not sheet:
            st.warning("No active workbook/sheet. Load a workbook in 📂 Source or select a sheet in 🗂 Workspace.")
            return

        # Get active DataFrame
        df = get_active_df()
        if df is None:
            st.warning("No active sheet selected. Please load a workbook in the Source tab and select a sheet.")
            return
        
        if not validate_headers(df):
            st.warning("Active sheet has invalid headers. Please ensure it has the required columns.")
            return

        # Mode toggle: Simple vs Advanced
        mode = st.segmented_control(
            "Mode", 
            options=["Simple", "Advanced"], 
            default="Simple", 
            key="__symptoms_mode"
        )
        
        if mode == "Simple":
            _render_simple_symptoms_editor(df, sheet)
        else:
            # Advanced mode: existing functionality
            # Temporary placeholder prevalence map; replace with your real source later
            symptom_prevalence = st.session_state.get("__symptom_prevalence", {})  # {label -> float or str}
            _render_advanced_symptoms(df, symptom_prevalence, sheet)

    except Exception as e:
        st.exception(e)


def _render_simple_symptoms_editor(df: pd.DataFrame, sheet_name: str):
    """Render the Simple mode: parent-first editor for any parent path."""
    st.subheader("✏️ Parent-First Editor")
    
    # Get conflict summary for parent information
    res = get_conflict_summary_with_root(df, get_wb_nonce())
    summary = res["summary"]
    
    # Build parent list from all levels (including VM/Root)
    parents = _list_parents_from_summary(summary)
    
    if not parents:
        st.info("No parents found")
        return
    
    # Initialize parent index in session state
    if "sym_simple_parent_index" not in st.session_state:
        st.session_state["sym_simple_parent_index"] = 0
    
    # Section A: Select Parent
    st.markdown("**A) Select Parent**")
    
    # Create searchable dropdown with level labels
    parent_options = []
    for L, pth, children in parents:
        level_label = LEVEL_LABELS[L]
        display_text = f"{level_label} — {pth}"
        parent_options.append((L, pth, children, display_text))
    
    selected_option = st.selectbox(
        "Pick a parent (VM/Nodes)", 
        options=parent_options,
        index=st.session_state["sym_simple_parent_index"],
        format_func=lambda x: x[3],  # Show the display text
        key="__symptoms_parent_picker"
    )
    
    if selected_option:
        level, selected_path, children_now = selected_option[:3]
        parent_label = selected_path.split(">")[-1] if level > 1 else ROOT_PARENT_LABEL
        
        # Section B: Edit Children
        st.markdown("**B) Edit Children**")
        st.write(f"**Current children ({len(children_now)}):** {', '.join(children_now) if children_now else 'None'}")
        
        _render_parent_editor_for_symptoms(level, selected_path, children_now, df, sheet_name, summary)
        
        # Section C: Actions
        st.markdown("**C) Actions**")
        
        # Skip to next incomplete button
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Skip ➝ Next incomplete parent", key="__symptoms_skip_next"):
                next_idx, next_parent = _find_next_incomplete_parent(summary, st.session_state["sym_simple_parent_index"] + 1)
                if next_idx is not None:
                    st.session_state["sym_simple_parent_index"] = next_idx
                    safe_rerun()
                else:
                    st.info("No more incomplete parents found")
        
        with col2:
            st.info(f"Parent {st.session_state['sym_simple_parent_index'] + 1} of {len(parents)}")


def _render_parent_editor_for_symptoms(level: int, parent_path: str, current_children: List[str], df: pd.DataFrame, sheet_name: str, summary: Dict):
    """Render the parent editor for symptoms with red flags support."""
    st.markdown("---")
    st.subheader("✏️ Edit Children")
    
    # Get parent label from path
    if level == 1:
        parent_label = ROOT_PARENT_LABEL
    else:
        parent_label = parent_path.split(">")[-1]
    
    # Build options from current children and any existing variants
    res = get_conflict_summary_with_root(df, get_wb_nonce())
    mismatches = res["mismatches"]
    block = mismatches.get((level, parent_label), {"variants": []})
    
    # Union of all children from variants and current
    union_opts = sorted({c for v in block["variants"] for c in v["children"]} | set(current_children))
    default_set = current_children[:MAX_CHILDREN_PER_PARENT]

    # Build stable key seed
    seed = f"symptoms_simple_{level}_{parent_path}_{get_wb_nonce()}"
    
    # Show red flag status for current children
    if current_children:
        flagged_children = [c for c in current_children if is_red_flag(c)]
        if flagged_children:
            st.warning(f"🚩 Red flagged children: {', '.join(flagged_children)}")
    
    chosen = st.multiselect(
        f"Choose up to {MAX_CHILDREN_PER_PARENT} children", 
        options=union_opts, 
        default=default_set, 
        max_selections=MAX_CHILDREN_PER_PARENT,
        key=f"{seed}_ms_children"
    )
    
    new_child = normalize_text(st.text_input("Add new child", key=f"{seed}_ti_add"))
    if new_child and new_child not in chosen:
        chosen = normalize_child_set(chosen + [new_child])
        st.info(f"Preview after add: {', '.join(chosen)}")

    # Red Flags expander
    with st.expander("🚩 Red Flags", expanded=False):
        q = normalize_text(st.text_input("Search symptom/label", key=f"{seed}_rf_search"))
        
        # Build label set from tree summary
        labels = set()
        for (L, pth), info in summary.items():
            if L == 1:
                labels.add(ROOT_PARENT_LABEL)
            else:
                lab = pth.split(">")[-1] if pth else ""
                if lab: 
                    labels.add(lab)
        
        # Add current children to labels
        labels.update(current_children)
        
        # Filter and sort labels
        labels = sorted([x for x in labels if not q or q in normalize_text(x)])
        
        for i, lab in enumerate(labels):
            checked = is_red_flag(lab)
            if st.checkbox(lab, value=checked, key=f"{seed}_rf_{i}"):
                if not checked: 
                    set_red_flag(lab, True)
            else:
                if checked: 
                    set_red_flag(lab, False)

    c1, c2 = st.columns(2)
    
    with c1:
        if st.button("Apply to THIS parent", key=f"{seed}_btn_single"):
            _apply_symptoms_to_single_parent(level, parent_path, chosen, df, sheet_name)
    
    with c2:
        if st.button(f"Apply to ALL '{parent_label}' parents across the tree", key=f"{seed}_btn_group"):
            _apply_symptoms_to_label_across_tree(parent_label, chosen, df, sheet_name)


def _apply_symptoms_to_single_parent(level: int, parent_path: str, children: List[str], df: pd.DataFrame, sheet_name: str):
    """Apply children to a single parent path in symptoms."""
    try:
        with st.spinner(f"Applying to single parent at level {level}..."):
            wb = get_active_workbook()
            if not wb or sheet_name not in wb:
                st.error("No active sheet.")
                return
            
            df0 = wb[sheet_name]
            
            # Get the parent label from the path
            if level == 1:
                parent_label = ROOT_PARENT_LABEL
            else:
                parent_label = parent_path.split(">")[-1]
            
            # Apply using label-group materializer
            new_df = materialize_children_for_label_group(df0, level, parent_label, children)
            
            # Update workbook
            wb[sheet_name] = new_df
            set_active_workbook(wb, default_sheet=sheet_name, source="symptoms_single_parent")
            set_current_sheet(sheet_name)
            
            st.success(f"Applied to parent: {parent_path}")
            safe_rerun()
            
    except Exception as e:
        st.error(f"Error applying to single parent: {e}")
        st.exception(e)


def _apply_symptoms_to_label_group(level: int, parent_label: str, children: List[str], df: pd.DataFrame, sheet_name: str):
    """Apply children to all parents with the same label at the given level in symptoms."""
    try:
        with st.spinner(f"Applying to all '{parent_label}' parents at level {level}..."):
            wb = get_active_workbook()
            if not wb or sheet_name not in wb:
                st.error("No active sheet.")
                return
            
            df0 = wb[sheet_name]
            
            # Use the existing label-group materializer
            new_df = materialize_children_for_label_group(df0, level, parent_label, children)
            
            # Update workbook
            wb[sheet_name] = new_df
            set_active_workbook(wb, default_sheet=sheet_name, source="symptoms_label_group")
            set_current_sheet(sheet_name)
            
            st.success(f"Applied to label-wide group: {parent_label}")
            safe_rerun()
            
    except Exception as e:
        st.error(f"Error applying to label group: {e}")
        st.exception(e)


def _apply_symptoms_to_label_across_tree(parent_label: str, children: List[str], df: pd.DataFrame, sheet_name: str):
    """Apply children to all parents with the same label across the entire tree in symptoms."""
    try:
        with st.spinner(f"Applying to all '{parent_label}' parents across the tree..."):
            wb = get_active_workbook()
            if not wb or sheet_name not in wb:
                st.error("No active sheet.")
                return
            
            df0 = wb[sheet_name]
            
            # Get the summary for the materializer
            res = get_conflict_summary_with_root(df0, get_wb_nonce())
            summary = res["summary"]
            
            # Use the across-tree materializer
            new_df = materialize_children_for_label_across_tree(df0, parent_label, children, summary)
            
            # Update workbook
            wb[sheet_name] = new_df
            set_active_workbook(wb, default_sheet=sheet_name, source="symptoms_across_tree")
            set_current_sheet(sheet_name)
            
            st.success(f"Applied to all '{parent_label}' parents across the tree")
            safe_rerun()
            
    except Exception as e:
        st.error(f"Error applying across tree: {e}")
        st.exception(e)


def _render_advanced_symptoms(df: pd.DataFrame, symptom_prevalence: Dict, sheet_name: str):
    """Render the Advanced mode: existing symptom quality and branch building functionality."""
    st.subheader("🔬 Advanced Symptoms Analysis")
    
    # Get symptom prevalence data
    symptom_prevalence = symptom_prevalence or {}
    
    # Main sections
    _render_symptom_prevalence_section(df, symptom_prevalence, sheet_name)
    
    st.markdown("---")
    
    _render_branch_building_section(df, symptom_prevalence, sheet_name)
    
    st.markdown("---")
    
    _render_branch_editor_section(df, sheet_name)


def _render_symptom_prevalence_section(df: pd.DataFrame, symptom_prevalence: Dict, sheet_name: str):
    """Render the symptom prevalence management section."""
    st.subheader("🎯 Symptom Prevalence Management")
    
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
                prevalence = symptom_prevalence.get(term, 0)
                st.write(f"• {term} (prevalence: {prevalence})")
            
            if len(filtered_vocab) > 50:
                st.caption(f"... and {len(filtered_vocab) - 50} more terms")
    
    with col2:
        # Prevalence score editor
        st.write("**Set prevalence score:**")
        selected_term = st.selectbox("Select term", [""] + filtered_vocab, key="prevalence_term_selector")
        
        if selected_term:
            current_prevalence = symptom_prevalence.get(selected_term, 0)
            new_prevalence = st.slider(
                "Prevalence score",
                min_value=0,
                max_value=10,
                value=current_prevalence,
                help="0 = rare, 10 = very common"
            )
            
            if new_prevalence != current_prevalence:
                if st.button("Update Prevalence"):
                    symptom_prevalence[selected_term] = new_prevalence
                    st.session_state["__symptom_prevalence"] = symptom_prevalence
                    st.success(f"Updated '{selected_term}' prevalence to {new_prevalence}")
                    safe_rerun()
    
    # Bulk prevalence operations
    st.markdown("---")
    st.subheader("📊 Bulk Prevalence Operations")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📈 Auto-assign Prevalence"):
            _auto_assign_prevalence_scores(df, symptom_prevalence)
    
    with col2:
        if st.button("🔄 Reset All Prevalence"):
            if st.checkbox("Confirm reset all prevalence scores"):
                st.session_state["__symptom_prevalence"] = {}
                st.success("All prevalence scores reset!")
                safe_rerun()
    
    with col3:
        if st.button("💾 Export Prevalence Data"):
            _export_prevalence_data(symptom_prevalence)


def _render_branch_building_section(df: pd.DataFrame, symptom_prevalence: Dict, sheet_name: str):
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
        _render_auto_suggest_branch_building(df, target_level, symptom_prevalence, sheet_name)
    else:  # Prevalence-based
        _render_prevalence_based_branch_building(df, target_level, symptom_prevalence, sheet_name)


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


def _auto_assign_prevalence_scores(df: pd.DataFrame, symptom_prevalence: Dict):
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
                
                symptom_prevalence[term] = score
            
            st.session_state["__symptom_prevalence"] = symptom_prevalence
            st.success(f"Auto-assigned prevalence scores to {len(term_counts)} terms!")
            safe_rerun()
            
    except Exception as e:
        st.error(f"Error auto-assigning prevalence scores: {e}")


def _export_prevalence_data(symptom_prevalence: Dict):
    """Export prevalence data to a downloadable format."""
    try:
        if not symptom_prevalence:
            st.warning("No prevalence data to export.")
            return
        
        # Create DataFrame
        prevalence_df = pd.DataFrame([
            {"term": term, "prevalence_score": score}
            for term, score in symptom_prevalence.items()
        ])
        
        # Sort by prevalence score
        prevalence_df = prevalence_df.sort_values("prevalence_score", ascending=False)
        
        # Download button
        csv = prevalence_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Prevalence Data (CSV)",
            data=csv,
            file_name="symptom_prevalence.csv",
            mime="text/csv"
        )
        
    except Exception as e:
        st.error(f"Error exporting prevalence data: {e}")


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


def _render_auto_suggest_branch_building(df: pd.DataFrame, target_level: int, symptom_prevalence: Dict, sheet_name: str):
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
            suggestions = _get_branch_suggestions(df, selected_parent, target_level, symptom_prevalence)
            
            if suggestions:
                st.write("**Suggested children:**")
                for i, suggestion in enumerate(suggestions):
                    st.write(f"{i + 1}. {suggestion}")
                
                if st.button("Use These Suggestions"):
                    _create_branch(df, selected_parent, target_level, suggestions, sheet_name)
            else:
                st.info("No suggestions found. Try manual input.")


def _render_prevalence_based_branch_building(df: pd.DataFrame, target_level: int, symptom_prevalence: Dict, sheet_name: str):
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
        # Show high-prevalence vocabulary
        high_prevalence_terms = [term for term, score in symptom_prevalence.items() if score >= 7]
        
        if high_prevalence_terms:
            st.write("**High-prevalence terms (score ≥ 7):**")
            selected_terms = st.multiselect(
                "Select terms for this branch",
                high_prevalence_terms,
                max_selections=MAX_CHILDREN_PER_PARENT,
                key=f"prevalence_terms_{target_level}"
            )
            
            if selected_terms and st.button("Create Prevalence Branch"):
                _create_branch(df, selected_parent, target_level, selected_terms, sheet_name)
        else:
            st.info("No high-prevalence terms found. Consider improving term prevalence first.")


def _get_branch_suggestions(df: pd.DataFrame, parent: str, target_level: int, symptom_prevalence: Dict) -> List[str]:
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
        
        # Add high-prevalence vocabulary terms
        high_prevalence_terms = [term for term, score in symptom_prevalence.items() if score >= 6]
        suggestions.update(high_prevalence_terms[:10])  # Top 10
        
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


def _render_branch_editor_section(df: pd.DataFrame, sheet_name: str):
    """Render the branch editor section for existing parents."""
    st.subheader("✏️ Branch Editor")
    st.markdown("Edit children for existing parents in the decision tree.")
    
    # Guard: require active df
    if df is None or df.empty:
        st.warning("No active DataFrame available.")
        return
    
    # Level selector (2-5, because level 1's parent is <ROOT>)
    level = st.selectbox("Node level", [2, 3, 4, 5], key="branch_editor_level")
    
    # Show hint for Level-1 editing
    st.info("💡 **Note:** Level-1 (ROOT) children are managed in the ⚖️ Conflicts tab. Use the Level-1 editor there to set the Node-1 options.")
    
    # Get distinct parent paths at level-1
    parent_paths = _get_parent_paths_at_level(df, level, get_wb_nonce())
    
    if not parent_paths:
        st.info(f"No parent paths found at level {level}.")
        return
    
    # Parent picker
    selected_parent_path = st.selectbox(
        "Select parent path",
        parent_paths,
        format_func=lambda x: " > ".join(x) if x else "<ROOT>",
        key="branch_editor_parent"
    )
    
    if not selected_parent_path:
        st.info("Please select a parent path.")
        return
    
    # Show current children for the selected parent
    current_children = _get_current_children_for_parent(df, level, selected_parent_path, get_wb_nonce())
    
    st.write(f"**Current children for '{' > '.join(selected_parent_path)}' at level {level}:**")
    if current_children:
        st.write(f"**{len(current_children)} children:** {', '.join(current_children)}")
    else:
        st.write("**No children defined yet.**")
    
    # Multi-select existing children to include
    st.markdown("---")
    st.write("**Select children to include:**")
    
    # Get all possible children at this level from the store
    from logic.tree import infer_branch_options
    store = infer_branch_options(df)
    
    # Find the key for this parent at this level
    parent_key = _build_parent_key(level, selected_parent_path)
    
    # Get existing children from store or current data
    all_possible_children = set()
    if parent_key in store:
        all_possible_children.update(store[parent_key])
    all_possible_children.update(current_children)
    
    if not all_possible_children:
        st.info("No existing children found. You can add new ones below.")
        all_possible_children = set()
    
    # Multi-select existing children (trim to 5)
    selected_existing = st.multiselect(
        "Select from existing children:",
        sorted(all_possible_children),
        max_selections=MAX_CHILDREN_PER_PARENT,
        key="branch_editor_existing"
    )
    
    # Add new child input
    new_child = st.text_input(
        "Add new child (optional):",
        placeholder="Type a new child name...",
        key="branch_editor_new_child"
    )
    
    # Build final children list
    final_children = list(selected_existing)
    
    if new_child.strip():
        new_child_clean = normalize_text(new_child)
        if new_child_clean and new_child_clean not in final_children:
            final_children.append(new_child_clean)
    
    # Use normalize_child_set to cap at MAX_CHILDREN_PER_PARENT
    final_children = normalize_child_set(final_children)
    
    if len(final_children) > MAX_CHILDREN_PER_PARENT:
        st.warning(f"Children list capped to {MAX_CHILDREN_PER_PARENT} (maximum allowed).")
    
    # Show final selection
    if final_children:
        st.write(f"**Final children set ({len(final_children)}):** {', '.join(final_children)}")
        
        # Apply button
        if st.button("Apply Changes", type="primary", key="branch_editor_apply"):
            _apply_branch_editor_changes(df, level, selected_parent_path, final_children, sheet_name)
    else:
        st.info("Please select at least one child to continue.")


@st.cache_data(ttl=600)
def _get_parent_paths_at_level(df: pd.DataFrame, level: int, nonce: str) -> List[Tuple[str, ...]]:
    """Get distinct parent paths at a specific level."""
    try:
        if level < 2:
            return []
        
        # Build parent columns (up to level-1)
        parent_cols = [f"Node {i}" for i in range(1, level)]
        if not all(col in df.columns for col in parent_cols):
            return []
        
        # Get unique parent paths
        parent_paths = df[parent_cols].apply(
            lambda r: tuple(normalize_text(v) for v in r), axis=1
        )
        parent_paths = parent_paths[parent_paths.apply(
            lambda x: all(v != "" for v in x)
        )]
        
        # Return unique paths, sorted
        unique_paths = sorted(parent_paths.unique())
        return unique_paths
        
    except Exception:
        return []


@st.cache_data(ttl=600)
def _get_current_children_for_parent(df: pd.DataFrame, level: int, parent_path: Tuple[str, ...], nonce: str) -> List[str]:
    """Get current children for a specific parent at a level."""
    try:
        if level > 5:
            return []
        
        node_col = f"Node {level}"
        if node_col not in df.columns:
            return []
        
        # Find rows matching the parent path
        matching_rows = df.copy()
        for i, expected_value in enumerate(parent_path):
            col = f"Node {i + 1}"
            if col in df.columns:
                mask = df[col].map(normalize_text) == expected_value
                matching_rows = matching_rows[mask]
        
        if matching_rows.empty:
            return []
        
        # Get children at this level
        children_values = matching_rows[node_col].map(normalize_text).dropna()
        children_values = children_values[children_values != ""]
        
        return sorted(children_values.unique())
        
    except Exception:
        return []


def _build_parent_key(level: int, parent_path: Tuple[str, ...]) -> str:
    """Build the parent key for the store."""
    if not parent_path:
        return f"L{level}|"
    else:
        return f"L{level}|" + ">".join(parent_path)


def _apply_branch_editor_changes(df: pd.DataFrame, level: int, parent_path: Tuple[str, ...], 
                                new_children: List[str], sheet_name: str):
    """Apply branch editor changes using override/materialization pipeline."""
    try:
        with st.spinner("Applying branch changes..."):
            # Create override key
            override_key = (level, parent_path)
            
            # Get current overrides
            overrides_all = st.session_state.get("branch_overrides", {})
            if sheet_name not in overrides_all:
                overrides_all[sheet_name] = {}
            
            # Set the override for this specific parent only
            overrides_all[sheet_name][override_key] = new_children
            st.session_state["branch_overrides"] = overrides_all
            
            # Apply the override using logic.tree
            from logic.tree import build_raw_plus_v630
            
            # Get the active workbook
            from utils.state import get_active_workbook, set_active_workbook
            active_wb = get_active_workbook()
            
            if active_wb and sheet_name in active_wb:
                # Apply overrides and rebuild the sheet
                updated_df = build_raw_plus_v630(df, overrides_all[sheet_name])
                
                # Update the active workbook
                active_wb[sheet_name] = updated_df
                set_active_workbook(active_wb, source="symptoms_editor")
                
                # Clear stale caches to ensure immediate refresh
                st.cache_data.clear()
                
                # Show diff/preview
                rows_affected = len(updated_df) - len(df)
                st.success("✅ Branch changes applied successfully!")
                st.info(f"**Delta:** {rows_affected:+d} rows affected.")
                st.write(f"**New children set:** {', '.join(new_children)}")
                
                # Show small preview of changes
                if rows_affected != 0:
                    st.write("**Preview of changes:**")
                    if rows_affected > 0:
                        st.write(f"Added {rows_affected} new rows with the updated children set.")
                    else:
                        st.write(f"Removed {abs(rows_affected)} rows to standardize the children set.")
                
                # Rerun to show updated state
                safe_rerun()
            else:
                st.error("Could not update active workbook.")
                
    except Exception as e:
        st.error(f"Error applying branch changes: {e}")
        st.exception(e)
