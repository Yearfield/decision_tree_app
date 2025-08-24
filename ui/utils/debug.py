# ui/utils/debug.py
from __future__ import annotations
import json, datetime as dt, traceback
import streamlit as st

def _summ(v):
    try:
        t = type(v).__name__
        if hasattr(v, "shape"):
            return f"<{t} shape={getattr(v, 'shape', None)}>"
        if isinstance(v, dict):
            return "dict"
        if isinstance(v, list):
            return f"list(len={len(v)})"
        if isinstance(v, (str, int, float, bool)) or v is None:
            s = repr(v)
            return s if len(s) <= 120 else s[:117] + "..."
        return f"<{t}>"
    except Exception:
        return "<unrepr>"

def dump_state(where: str, keys: list[str] | None = None, expanded: bool = True):
    """Side-bar dump of session state (types, shapes, or short reprs)."""
    try:
        snap = {}
        for k in sorted(st.session_state.keys()):
            if keys and k not in keys:
                continue
            snap[k] = _summ(st.session_state[k])
        with st.sidebar.expander(f"🛠 Debug: {where}", expanded=expanded):
            st.code(json.dumps(snap, indent=2), language="json")
    except Exception as e:
        with st.sidebar.expander(f"🛠 Debug: {where} (error)", expanded=True):
            st.error(f"{type(e).__name__}: {e}")
            st.code(traceback.format_exc())

def banner(msg: str):
    """Inline banner in the main area; never throws."""
    try:
        st.caption(f"🚦 {msg} • {dt.datetime.utcnow().isoformat(timespec='seconds')}Z")
    except Exception:
        pass

def render_guard(label: str, fn):
    """Run a tab render function with visible error reporting (no blank tabs)."""
    import traceback as _tb
    banner(f"DISPATCH {label}")
    try:
        return fn()
    except Exception as e:
        st.error(f"Exception in {label}.render(): {type(e).__name__}: {e}")
        st.code(_tb.format_exc())
        return None
