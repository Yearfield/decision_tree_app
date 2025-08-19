# streamlit_app_upload.py — v6.3.0 (modular main)

from typing import Dict, List, Tuple, Optional, Set
from datetime import datetime
import io
import importlib
import traceback
import types

import numpy as np
import pandas as pd
import streamlit as st

# --- Utils & shared helpers
from utils import (
    CANON_HEADERS, LEVEL_COLS, MAX_LEVELS,
    normalize_text, validate_headers,
    infer_branch_options,  # used for header badge
)
from constants import APP_VERSION, TAB_ICONS

st.set_page_config(
    page_title=f"Decision Tree Builder — {APP_VERSION}",
    page_icon="🌳",
    layout="wide",
)

# ---------- Debug mode toggle ----------
st.sidebar.checkbox("🐞 Debug mode", value=False, key="__debug")

# ---------- Session state safety ----------
for _k in ["current_sheet", "df", "save_mode"]:
    if _k not in st.session_state:
        st.session_state[_k] = None

# ---------- Default context initialization ----------
if "work_context" not in st.session_state:
    # Best-effort default: pick first available sheet automatically
    from utils import get_current_df_and_sheet
    _df0, _name0, _src0 = get_current_df_and_sheet()
    # (get_current_df_and_sheet will set work_context if anything is available)

# ---------- Robust safe-import system ----------
def _safe_import(mod_name: str):
    try:
        mod = importlib.import_module(mod_name)
        return mod, None
    except Exception as e:
        tb = traceback.format_exc()
        return None, f"{e.__class__.__name__}: {e}\n{tb}"

# ---------- Sidebar ----------
with st.sidebar:
    if st.button("🔄 Reload app"):
        st.rerun()


# ---------- Mini metrics for header badge ----------
def parent_key_from_row_strict(row: pd.Series, upto_level: int) -> Optional[Tuple[str, ...]]:
    if upto_level <= 1:
        return tuple()
    parent = []
    for c in LEVEL_COLS[:upto_level-1]:
        v = normalize_text(row[c])
        if v == "":
            return None
        parent.append(v)
    return tuple(parent)

def compute_parent_depth_score(df: pd.DataFrame) -> Tuple[int, int]:
    store = infer_branch_options(df)
    total = 0; ok = 0
    for level in range(1, MAX_LEVELS+1):
        parents = set()
        for _, row in df.iterrows():
            p = parent_key_from_row_strict(row, level)
            if p is not None:
                parents.add(p)
        for p in parents:
            total += 1
            key = f"L{level}|" + (">".join(p) if p else "<ROOT>")
            if len([x for x in store.get(key, []) if normalize_text(x)!=""]) == 5:
                ok += 1
    return ok, total

def compute_row_path_score(df: pd.DataFrame) -> Tuple[int, int]:
    if df.empty:
        return (0,0)
    nodes = df[LEVEL_COLS].map(lambda col: col.map(normalize_text))
    full = nodes.ne("").all(axis=1)
    return int(full.sum()), int(len(df))

def progress_badge_html(df: pd.DataFrame) -> str:
    ok_p, total_p = compute_parent_depth_score(df)
    ok_r, total_r = compute_row_path_score(df)
    pct_p = 0 if total_p==0 else int(round(100*ok_p/total_p))
    pct_r = 0 if total_r==0 else int(round(100*ok_r/total_r))
    bar_css = """
    <style>
      .badge-wrap { display:inline-block; padding:8px 10px; border-radius:12px; border:1px solid #dbe0e6; background:#f8fafc; }
      .badge-row { font: 12px/1.2 -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Helvetica, Arial, sans-serif; margin:4px 0 6px 0; color:#0f172a;}
      .bar { position:relative; height:10px; background:#eef2f6; border:1px solid #dbe0e6; border-radius:8px; overflow:hidden; }
      .fill { position:absolute; left:0; top:0; bottom:0; width:VAR%; background:linear-gradient(90deg,#60a5fa,#2563eb); }
      .fill2 { position:absolute; left:0; top:0; bottom:0; width:VAR2%; background:linear-gradient(90deg,#34d399,#059669); }
      .badge-top { display:flex; justify-content:space-between; }
      .muted { color:#334155; }
      .strong { font-weight:600; }
    </style>
    """
    html = f"""
    {bar_css}
    <div class='badge-wrap'>
      <div class='badge-row'>
        <div class='badge-top'><span class='muted'>Parents 5/5</span><span class='strong'>{ok_p}/{total_p}</span></div>
        <div class='bar'><div class='fill' style='width:{pct_p}%;'></div></div>
      </div>
      <div class='badge-row'>
        <div class='badge-top'><span class='muted'>Rows full path</span><span class='strong'>{ok_r}/{total_r}</span></div>
        <div class='bar'><div class='fill2' style='width:{pct_r}%;'></div></div>
      </div>
    </div>
    """
    return html


# ---------- Import UI modules (with robust error handling) ----------
tab_source, tab_source_err = _safe_import("ui_source")
tab_workspace, tab_workspace_err = _safe_import("ui_workspace")
tab_validation, tab_validation_err = _safe_import("ui_validation")
tab_conflicts, tab_conflicts_err = _safe_import("ui_conflicts")
tab_triage, tab_triage_err = _safe_import("ui_triage")
tab_actions, tab_actions_err = _safe_import("ui_actions")
tab_symptoms, tab_symptoms_err = _safe_import("ui_symptoms")
tab_dictionary, tab_dictionary_err = _safe_import("ui_dictionary")
tab_calculator, tab_calculator_err = _safe_import("ui_calculator")
tab_visualizer, tab_visualizer_err = _safe_import("ui_visualizer")
tab_pushlog, tab_pushlog_err = _safe_import("ui_pushlog")

_IMPORT_ERRORS = {
    "Source": tab_source_err,
    "Validation": tab_validation_err,
    "Conflicts": tab_conflicts_err,
    "Diagnostic Triage": tab_triage_err,
    "Actions": tab_actions_err,
    "Symptoms": tab_symptoms_err,
    "Dictionary": tab_dictionary_err,
    "Calculator": tab_calculator_err,
    "Visualizer": tab_visualizer_err,
    "Push Log": tab_pushlog_err,
}

def _render_or_error(tab_name: str, module):
    err = _IMPORT_ERRORS.get(tab_name)
    if err:
        st.error(f"❌ {tab_name} module failed to import.")
        with st.expander("Show import error", expanded=st.session_state.get("__debug", False)):
            st.code(err, language="text")
        return False
    if module is None or not hasattr(module, "render"):
        st.error(f"❌ {tab_name} module not available.")
        return False
    return True


# ---------- Header ----------
left, right = st.columns([1, 2])
with left:
    st.title(f"🌳 Decision Tree Builder — {APP_VERSION}")
with right:
    # Try to show a quick badge based on the first available sheet (Upload or Google Sheets workbook)
    badge_html = ""
    wb_upload = st.session_state.get("upload_workbook", {})
    wb_gs     = st.session_state.get("gs_workbook", {})
    df0 = None
    if wb_upload:
        df0 = wb_upload[next(iter(wb_upload))]
    elif wb_gs:
        df0 = wb_gs[next(iter(wb_gs))]
    if df0 is not None and not df0.empty and validate_headers(df0):
        try:
            badge_html = progress_badge_html(df0)
        except Exception:
            badge_html = ""
    if badge_html:
        st.markdown(badge_html, unsafe_allow_html=True)
    if "gcp_service_account" in st.secrets:
        st.caption("Google Sheets linked ✓")
    else:
        st.caption("Google Sheets not configured (optional). Add your service account JSON under [gcp_service_account].")

st.caption("Canonical headers: Vital Measurement, Node 1, Node 2, Node 3, Node 4, Node 5, Diagnostic Triage, Actions")


# ---------- Tabs ----------
tabs = st.tabs([
    "📂 Source",
    "🗂 Workspace Selection",
    "🔎 Validation",
    "⚖️ Conflicts",
    "🩺 Diagnostic Triage",
    "⚡ Actions",
    "🧬 Symptoms",
    "📖 Dictionary",
    "🧮 Calculator",
    "🌐 Visualizer",
    "📜 Push Log",
])

with tabs[0]:
    if _render_or_error("Source", tab_source):
        tab_source.render()

with tabs[1]:
    if _render_or_error("Workspace", tab_workspace):
        tab_workspace.render()

with tabs[2]:
    if _render_or_error("Validation", tab_validation):
        tab_validation.render()

with tabs[3]:
    if _render_or_error("Conflicts", tab_conflicts):
        tab_conflicts.render()

with tabs[4]:
    # Diagnostic Triage tab
    if _render_or_error("Diagnostic Triage", tab_triage):
        tab_triage.render()

with tabs[5]:
    # Actions tab
    if _render_or_error("Actions", tab_actions):
        tab_actions.render()

with tabs[6]:
    if _render_or_error("Symptoms", tab_symptoms):
        tab_symptoms.render()

with tabs[7]:
    if _render_or_error("Dictionary", tab_dictionary):
        tab_dictionary.render()

with tabs[8]:
    if _render_or_error("Calculator", tab_calculator):
        tab_calculator.render()

with tabs[9]:
    if _render_or_error("Visualizer", tab_visualizer):
        tab_visualizer.render()

with tabs[10]:
    # Fallback Push Log if ui_pushlog module is not present yet
    if _render_or_error("Push Log", tab_pushlog):
        tab_pushlog.render()
    else:
        st.subheader("📜 Push Log")
        log = st.session_state.get("push_log", [])
        if not log:
            st.info("No pushes recorded this session.")
        else:
            df_log = pd.DataFrame(log)
            st.dataframe(df_log, use_container_width=True)
            csv = df_log.to_csv(index=False).encode("utf-8")
            st.download_button("Download push log (CSV)", data=csv, file_name="push_log.csv", mime="text/csv")

# ---------- Footer note ----------
st.markdown(
    f"<div style='text-align:right;color:#64748b;font-size:12px;'>v{APP_VERSION} • "
    f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>",
    unsafe_allow_html=True,
)

# App version footer
st.markdown(f"---\nApp Version: **{APP_VERSION}**")
