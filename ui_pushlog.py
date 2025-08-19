# ui_pushlog.py

from typing import List, Dict, Any
from datetime import datetime, date
import json

import pandas as pd
import streamlit as st





def _normalize_log_df(items: List[Dict[str, Any]]) -> pd.DataFrame:
    if not items:
        return pd.DataFrame(columns=[
            "ts", "sheet", "target_tab", "spreadsheet_id",
            "rows_written", "new_rows_added", "scope"
        ])
    df = pd.DataFrame(items).copy()

    # Ensure columns exist
    for c in ["ts","sheet","target_tab","spreadsheet_id","rows_written","new_rows_added","scope"]:
        if c not in df.columns:
            df[c] = None

    # Parse timestamps (best effort)
    def _parse_ts(x):
        if pd.isna(x):
            return None
        s = str(x)
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y/%m/%d %H:%M:%S"):
            try:
                return datetime.strptime(s, fmt)
            except Exception:
                continue
        # last resort: pandas parser
        try:
            return pd.to_datetime(s)
        except Exception:
            return None

    df["ts_parsed"] = df["ts"].map(_parse_ts)
    df["date"] = df["ts_parsed"].dt.date
    df["rows_written"] = pd.to_numeric(df["rows_written"], errors="coerce")
    df["new_rows_added"] = pd.to_numeric(df["new_rows_added"], errors="coerce")
    df["rows_written"].fillna(0, inplace=True)
    df["new_rows_added"].fillna(0, inplace=True)

    # Pretty columns for display
    df["When"] = df["ts_parsed"].dt.strftime("%Y-%m-%d %H:%M:%S")
    df.rename(columns={
        "sheet": "Sheet",
        "target_tab": "Target Tab",
        "spreadsheet_id": "Spreadsheet ID",
        "rows_written": "Rows Written",
        "new_rows_added": "New Rows Added",
        "scope": "Scope",
    }, inplace=True)

    # Order columns for the table
    display_cols = [
        "When", "Sheet", "Target Tab", "Spreadsheet ID",
        "Rows Written", "New Rows Added", "Scope"
    ]
    return df, display_cols


def render():
    st.header("📜 Push Log")

    log = st.session_state.get("push_log", [])

    if not log:
        st.info("No pushes recorded this session.")
        return

    df, display_cols = _normalize_log_df(log)

    # ---- Filters ----
    st.subheader("Filters")

    c1, c2, c3, c4 = st.columns([1, 1, 2, 2])

    with c1:
        sheets = sorted([s for s in df["Sheet"].dropna().unique()])
        pick_sheets = st.multiselect("Sheet", options=sheets, default=sheets, key="pl_sheets")

    with c2:
        tabs = sorted([t for t in df["Target Tab"].dropna().unique()])
        pick_tabs = st.multiselect("Target Tab", options=tabs, default=tabs, key="pl_tabs")

    with c3:
        q = st.text_input("Search (Spreadsheet ID / Scope)", key="pl_search").strip().lower()

    with c4:
        # Date range (optional)
        dates = sorted([d for d in df["date"].dropna().unique()])
        min_d = dates[0] if dates else None
        max_d = dates[-1] if dates else None
        st.caption("Date range (optional)")
        col_d1, col_d2 = st.columns(2)
        with col_d1:
            d_from = st.date_input("From", value=min_d if isinstance(min_d, date) else None, key="pl_from")
        with col_d2:
            d_to = st.date_input("To", value=max_d if isinstance(max_d, date) else None, key="pl_to")

    view = df.copy()

    if pick_sheets:
        view = view[view["Sheet"].isin(pick_sheets)]
    if pick_tabs:
        view = view[view["Target Tab"].isin(pick_tabs)]
    if q:
        mask = (
            view["Spreadsheet ID"].astype(str).str.lower().str.contains(q, na=False) |
            view["Scope"].astype(str).str.lower().str.contains(q, na=False)
        )
        view = view[mask]
    if isinstance(d_from, date):
        view = view[(view["date"].isna()) | (view["date"] >= d_from)]
    if isinstance(d_to, date):
        view = view[(view["date"].isna()) | (view["date"] <= d_to)]

    # ---- Metrics ----
    st.subheader("Summary")
    last_ts = view["ts_parsed"].max()
    total_pushes = len(view)
    total_rows = int(view["Rows Written"].sum()) if total_pushes else 0
    total_new = int(view["New Rows Added"].sum()) if total_pushes else 0

    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Total pushes (filtered)", total_pushes)
    with m2:
        st.metric("Rows written (sum)", f"{total_rows:,}")
    with m3:
        st.metric("New rows added (sum)", f"{total_new:,}")

    if pd.notna(last_ts):
        st.caption(f"Last push: **{last_ts.strftime('%Y-%m-%d %H:%M:%S')}**")

    # ---- Chart (optional): pushes per day ----
    with st.expander("📈 Activity (per day)", expanded=False):
        per_day = (
            view
            .dropna(subset=["date"])
            .groupby("date")
            .agg(pushes=("date", "count"), rows=("Rows Written", "sum"))
            .reset_index()
            .sort_values("date")
        )
        if per_day.empty:
            st.info("No dated entries to chart.")
        else:
            st.line_chart(per_day.set_index("date")[["pushes", "rows"]])

    # ---- Table ----
    st.subheader("History")
    st.dataframe(
        view[display_cols].sort_values("When", ascending=False),
        use_container_width=True,
        height=420
    )

    # ---- Export ----
    st.subheader("Export")
    csv_bytes = view[display_cols].to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download filtered log (CSV)",
        data=csv_bytes,
        file_name="push_log_filtered.csv",
        mime="text/csv",
        key="pl_dl_csv",
    )

    # JSON export (original raw entries that match current filter)
    # We map back to original dicts by row index if available; otherwise export the flattened view.
    try:
        # Rebuild raw-like dicts but filtered
        raw_like = []
        for _, r in view.iterrows():
            raw_like.append({
                "ts": r.get("When"),
                "sheet": r.get("Sheet"),
                "target_tab": r.get("Target Tab"),
                "spreadsheet_id": r.get("Spreadsheet ID"),
                "rows_written": int(r.get("Rows Written", 0)) if pd.notna(r.get("Rows Written")) else 0,
                "new_rows_added": int(r.get("New Rows Added", 0)) if pd.notna(r.get("New Rows Added")) else 0,
                "scope": r.get("Scope"),
            })
        json_bytes = json.dumps(raw_like, indent=2).encode("utf-8")
        st.download_button(
            "Download filtered log (JSON)",
            data=json_bytes,
            file_name="push_log_filtered.json",
            mime="application/json",
            key="pl_dl_json",
        )
    except Exception:
        pass

    # ---- Maintenance ----
    st.subheader("Maintenance")
    cclear1, cclear2 = st.columns([1, 3])
    with cclear1:
        confirm_clear = st.checkbox("Confirm clear log", key="pl_confirm_clear")
    with cclear2:
        if st.button("🗑️ Clear push log", disabled=not confirm_clear, key="pl_clear_btn"):
            st.session_state["push_log"] = []
            st.success("Push log cleared.")
