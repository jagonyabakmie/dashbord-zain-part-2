"""Shared page helpers: exploratory filtering and selected-school resolution."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from utils import data as D
from utils import styles as S


def selected_school_ids(ctx) -> list[int]:
    """Schools chosen in the sidebar (analytical filter on Overview, highlight elsewhere)."""
    return list(ctx.get("schools") or [])


def selected_school_id(ctx) -> int | None:
    """First chosen school, for views that highlight a single school."""
    ids = selected_school_ids(ctx)
    return ids[0] if ids else None


def highlight_school_ids(ctx) -> list[int]:
    return list(ctx.get("hl_schools") or [])


def view_summary(df: pd.DataFrame, ctx, n_label: str = "Students") -> str:
    """'Full Sample: 30 Schools • 900 Students' or 'Active View: 2 Schools • 43 Students • Female'."""
    filtered = bool(ctx.get("schools")) or ctx["gender"] != ctx["ALL"] or bool(ctx.get("students"))
    n_sc = df["school_id"].nunique()
    parts = [f"{n_sc} School{'s' if n_sc != 1 else ''}", f"{len(df):,} {n_label if len(df) != 1 else n_label[:-1]}"]
    if ctx["gender"] != ctx["ALL"]:
        parts.append(ctx["gender"])
    return ("Active View: " if filtered else "Full Sample: ") + " • ".join(parts)


def filter_students(ctx, students: bool = True) -> tuple[pd.DataFrame, str]:
    """Apply sidebar filters (schools, gender and, if students=True, student IDs) to the 900 student rows."""
    df = D.students()
    if ctx.get("schools"):
        df = df[df["school_id"].isin(ctx["schools"])]
    if ctx["gender"] != ctx["ALL"]:
        df = df[df["gender"] == ctx["gender"]]
    if students and ctx.get("students"):
        df = df[df["student_id"].isin(ctx["students"])]
    df = df.reset_index(drop=True)
    return df, view_summary(df, ctx)


def gender_filtered_students(ctx) -> pd.DataFrame:
    """Students filtered by gender only, used to build school-level (30-row) aggregates."""
    df = D.students()
    if ctx["gender"] != ctx["ALL"]:
        df = df[df["gender"] == ctx["gender"]]
    return df


def empty_note(msg: str) -> None:
    S.insight(msg, title="Not enough data", kind="warn")


def chart(fig, key: str | None = None) -> None:
    """Render a chart that stretches to the free height of its card."""
    from utils.charts import CONFIG
    fig.update_layout(height=None, autosize=True)
    try:
        st.plotly_chart(fig, width="stretch", height="stretch", theme=None, config=CONFIG, key=key)
    except TypeError:  # Streamlit without height="stretch": fall back to a fixed height
        fig.update_layout(height=260)
        st.plotly_chart(fig, use_container_width=True, theme=None, config=CONFIG, key=key)
