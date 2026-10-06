"""Page 2: Exploratory Analysis (raw relationships; school and gender filters subset student rows,
student IDs are highlighted).

School-level predictors (facility, teacher experience) are analysed on 30 school rows,
never on 900 pseudo-replicated student rows.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from utils import charts as C
from utils import data as D
from utils import interpretations as I
from utils import styles as S
from views.common import chart, empty_note, filter_students, selected_school_ids

STUDENT_PREDICTORS = {
    "SES": "ses",
    "Prior Mathematics Score": "prior.score",
    "Study Hours": "study.hours",
}
SCHOOL_PREDICTORS = {
    "Facility Score": "facility",
    "Teacher Experience": "teacher.exp",
}
PRED_HELP = {
    "SES": "SES (socioeconomic status) is an index of family economic and social background.",
    "Prior Mathematics Score": "Mathematics score from an earlier assessment.",
    "Study Hours": "Self-reported weekly study hours.",
    "Facility Score": "School facility rating, one value per school.",
    "Teacher Experience": "Average teaching experience of the school's teachers, in years.",
}
R_HELP = "Pearson r measures how closely two variables move together on a straight line, from -1 to +1."
SLOPE_HELP = "Raw slope: the change in mathematics score for each one-unit increase, before adjusting for anything else."


def render(ctx) -> None:
    df, desc = filter_students(ctx, students=False)  # Student IDs highlight here instead of filtering
    filtered = desc.startswith("Active")

    S.header_row(
        "Exploratory Analysis",
        "Explore how student- and school-level characteristics relate to mathematics achievement.",
        chips=["Raw associations", "Filtered View" if filtered else "Full Sample"],
        gold_chip=desc if filtered else None,
        deco="26i_pencil.png",
    )
    with st.container(key="ptabs"):
        tab_a, tab_b = st.tabs(["Student Relationships", "School Relationships"])
    with tab_a:
        _student_tab(ctx, df, desc)
    with tab_b:
        _school_tab(ctx)


def _student_tab(ctx, df, desc) -> None:
    hl = set(ctx.get("students") or [])
    c1, c2 = st.columns([7, 5], gap="small")
    with c1, S.card("ex_scatter", rows=1):
        pred = st.session_state.get("ex_pred", "SES")
        if pred not in STUDENT_PREDICTORS:
            pred = "SES"
        col = STUDENT_PREDICTORS[pred]
        xy = df[["student_id", "school_id", "gender", col, "math.score"]].dropna()
        n = len(xy)
        r = I.safe_corr(xy[col], xy["math.score"]) if n >= 3 else np.nan
        band = C.ols_band(xy[col], xy["math.score"]) if n >= 3 else None
        S.card_head("Student Level", f"{pred} and Mathematics Achievement",
                    "Each point is a student · least-squares line with 95% confidence band", tone="emerald",
                    pill_icon="scatter_plot", help=PRED_HELP[pred])
        a, b = st.columns([3, 4], gap="small", vertical_alignment="center")
        with a:
            st.segmented_control("Predictor", list(STUDENT_PREDICTORS), default="SES", key="ex_pred",
                                 label_visibility="collapsed")
        with b:
            S.stats_row([
                ("Pearson r =", I.num(r, 3, na="–"), "", R_HELP),
                ("Raw slope =", f"{band['slope']:+.3f}" if band else "–", "gold", SLOPE_HELP),
                ("n =", f"{n:,}", "blue", "Students in the current view"),
                ("Highlighted:", f"{len(hl & set(xy['student_id']))}", "", "Students chosen in the sidebar") if hl else None,
            ])
        if n < 3:
            empty_note("Select at least three students to draw a relationship.")
        else:
            base = xy[~xy["student_id"].isin(hl)]
            fig = go.Figure()
            fig.add_trace(go.Scattergl(
                x=base[col], y=base["math.score"], mode="markers", name="Students",
                marker=dict(size=5, color="rgba(15,138,95,0.42)", line=dict(width=0)),
                customdata=np.stack([base["student_id"], base["school_id"], base["gender"]], axis=1),
                hovertemplate="Student %{customdata[0]}<br>School %{customdata[1]} · %{customdata[2]}<br>" + pred
                              + ": %{x:.2f}<br>Mathematics score: %{y:.2f}<extra></extra>"))
            if band is not None:
                C.add_fit_band(fig, band, S.GOLD)
            sel = xy[xy["student_id"].isin(hl)]
            if not sel.empty:
                fig.add_trace(go.Scatter(
                    x=sel[col], y=sel["math.score"], mode="markers+text", name="Selected students",
                    text=sel["student_id"], textposition="top center", textfont=dict(size=10, color=S.INK),
                    marker=dict(size=11, color=S.GOLD, line=dict(color="#FFFFFF", width=1.5)),
                    customdata=np.stack([sel["student_id"], sel["school_id"], sel["gender"]], axis=1),
                    hovertemplate="Student %{customdata[0]}<br>School %{customdata[1]} · %{customdata[2]}<br>" + pred
                                  + ": %{x:.2f}<br>Mathematics score: %{y:.2f}<extra></extra>"))
            C.base_layout(fig)
            fig.update_xaxes(title_text=pred)
            fig.update_yaxes(title_text="Mathematics score")
            chart(fig, "ex_scatter_chart")
            txt = (f"{pred} shows a <b>{I.corr_strength(r)} {I.direction(r)}</b> raw association with mathematics score "
                   f"(r = {r:.2f}, n = {n}). On average, one more unit of {pred} goes with "
                   f"<b>{band['slope']:+.3f}</b> points. ")
            if not sel.empty:
                txt += f"The {len(sel)} highlighted student(s) are shown in gold against everyone in the view. "
            S.insight(txt + "This is unadjusted. The Research Questions page shows associations that hold other "
                            "predictors constant.")

    with c2, S.card("ex_corr", rows=1):
        S.card_head("Correlation", "How Are the Variables Related?", "Pearson correlations",
                    tone="slate", pill_icon="grid_on", help=R_HELP)
        lvl = st.segmented_control("Level", ["Student Level", "School Level"], default="Student Level",
                                   key="ex_corr_level", label_visibility="collapsed") or "Student Level"
        if lvl == "Student Level":
            labels = list(D.STUDENT_VARS.values())
            if not desc.startswith("Active"):
                cm = D.sheet(D.S_CORR_STUDENT).set_index("variable").loc[list(D.STUDENT_VARS), list(D.STUDENT_VARS)]
                unit = f"Unit: students · full sample (n = {len(df)})"
            elif len(df) >= 3:
                cm = df[list(D.STUDENT_VARS)].corr()
                unit = f"Unit: students · current view (n = {len(df)})"
            else:
                cm, unit = None, ""
        else:
            labels = list(D.SCHOOL_VARS.values())
            cm = D.sheet(D.S_CORR_SCHOOL).set_index("variable").loc[list(D.SCHOOL_VARS), list(D.SCHOOL_VARS)]
            unit = "Unit: 30 schools (school averages; filters do not apply)"
        if cm is None:
            empty_note("Select at least three students for a student-level correlation matrix.")
        else:
            S.stats_row([("", unit, "blue")])
            z = cm.to_numpy(dtype=float)
            fig = go.Figure(go.Heatmap(
                z=z, x=labels, y=labels, zmin=-1, zmax=1, zmid=0, xgap=2, ygap=2,
                colorscale=[[0, "#B9573E"], [0.5, "#F4F2EC"], [1, "#0B6B48"]],
                text=np.vectorize(lambda v: "" if np.isnan(v) else f"{v:.2f}")(z), texttemplate="%{text}",
                textfont=dict(size=11), colorbar=dict(thickness=8, outlinewidth=0, tickfont=dict(size=9)),
                hovertemplate="%{y} × %{x}<br>r = %{z:.3f}<extra></extra>"))
            C.base_layout(fig)
            fig.update_layout(margin=dict(l=80))
            fig.update_yaxes(autorange="reversed", showgrid=False)
            fig.update_xaxes(showgrid=False, tickangle=-30)
            chart(fig, "ex_corr_chart")
            others = [(labels[i], z[0, i]) for i in range(1, len(labels)) if not np.isnan(z[0, i])]
            best = max(others, key=lambda t: abs(t[1])) if others else None
            txt = ""
            if best:
                txt += f"{labels[0]} is most closely related to <b>{best[0]}</b> (r = {best[1]:.2f}). "
            txt += ("With only 30 schools these correlations are uncertain, and each school counts once."
                    if lvl == "School Level" else
                    "SES and prior score are almost unrelated to each other, so each carries its own information. "
                    "These raw correlations ignore that students are grouped in schools.")
            S.insight(txt)


def _school_tab(ctx) -> None:
    sel_ids = selected_school_ids(ctx)
    sc = D.schools().copy()
    c1, c2 = st.columns([6, 6], gap="small")
    with c1, S.card("ex_school_scatter", rows=1):
        pred = st.session_state.get("ex_school_pred", "Facility Score")
        if pred not in SCHOOL_PREDICTORS:
            pred = "Facility Score"
        col = SCHOOL_PREDICTORS[pred]
        r = I.safe_corr(sc[col], sc["mean_math"])
        band = C.ols_band(sc[col], sc["mean_math"])
        S.card_head("School Level", f"{pred} and Mean Mathematics Score",
                    "Each point is a school · least-squares line with 95% confidence band", tone="blue",
                    pill_icon="apartment", help=PRED_HELP[pred])
        a, b = st.columns([1, 1], gap="small", vertical_alignment="center")
        with a:
            st.segmented_control("School predictor", list(SCHOOL_PREDICTORS), default="Facility Score",
                                 key="ex_school_pred", label_visibility="collapsed")
        with b:
            S.stats_row([
                ("Pearson r =", I.num(r, 3), "", R_HELP),
                ("Raw slope =", f"{band['slope']:+.3f}" if band else "–", "gold", SLOPE_HELP),
                ("n =", "30 Schools", "blue", "Every school counts once; filters only highlight schools here."),
            ])
        fig = go.Figure()
        if band is not None:
            C.add_fit_band(fig, band, S.FOREST_2, fill="rgba(15,138,95,0.12)")
        colors = [S.GOLD if s in sel_ids else S.EMERALD for s in sc["school_id"]]
        fig.add_trace(go.Scatter(
            x=sc[col], y=sc["mean_math"], mode="markers+text",
            text=[f"S{int(s):02d}" if s in sel_ids else "" for s in sc["school_id"]], textposition="top center",
            textfont=dict(size=10, color=S.INK),
            marker=dict(size=[14 if c == S.GOLD else 10 for c in colors], color=colors, line=dict(color="#FFFFFF", width=1.2)),
            customdata=sc["school_id"], showlegend=False,
            hovertemplate="School %{customdata}<br>" + pred + ": %{x:.2f}<br>Mean mathematics score: %{y:.2f}<extra></extra>"))
        C.base_layout(fig)
        fig.update_xaxes(title_text=pred)
        fig.update_yaxes(title_text="School mean mathematics score")
        chart(fig, "ex_school_scatter_chart")
        txt = (f"Across the 30 schools, {pred} has a <b>{I.corr_strength(r)} {I.direction(r)}</b> raw association with "
               f"the school mean score (r = {r:.2f}; raw slope {band['slope']:+.3f} points per unit). ")
        txt += ("With 30 schools a single school can move this line noticeably, and school means also reflect "
                "the students each school enrols.")
        S.insight(txt)

    with c2, S.card("ex_slopes", rows=1):
        rs = D.sheet(D.S_RAW_SLOPES).sort_values("raw_hours_slope", ascending=False).reset_index(drop=True)
        rs["lo"] = rs["raw_hours_slope"] - 1.96 * rs["raw_hours_se"]
        rs["hi"] = rs["raw_hours_slope"] + 1.96 * rs["raw_hours_se"]
        rs["label"] = rs["school_id"].map(lambda s: f"S{int(s):02d}")
        S.card_head("Heterogeneity Check", "Does the Study-Hours Relationship Differ Across Schools?",
                    "Raw school-specific relationships between study hours and mathematics achievement.",
                    tone="blue", pill_icon="trending_up",
                    help="Each dot is the raw slope of mathematics score on study hours inside one school, with a ±1.96 SE interval.")
        mslope = rs["raw_hours_slope"].mean()
        S.stats_row([("n =", "30 Schools", "blue"), ("Slopes from", f"{rs['raw_hours_slope'].min():+.2f} to "
                                                    f"{rs['raw_hours_slope'].max():+.2f}", ""),
                     ("Average", f"{mslope:+.2f} points/hour", "gold")])
        colors = [S.GOLD if s in sel_ids else (S.EMERALD if v >= 0 else S.TERRA)
                  for s, v in zip(rs["school_id"], rs["raw_hours_slope"])]
        fig = go.Figure(go.Scatter(
            x=rs["label"], y=rs["raw_hours_slope"], mode="markers",
            marker=dict(size=[12 if c == S.GOLD else 8 for c in colors], color=colors, line=dict(color="#FFFFFF", width=1)),
            error_y=dict(type="data", symmetric=False, array=rs["hi"] - rs["raw_hours_slope"],
                         arrayminus=rs["raw_hours_slope"] - rs["lo"], color="rgba(62,120,178,0.35)", thickness=1.4, width=0),
            customdata=np.stack([rs["raw_hours_se"], rs["raw_hours_p"], rs["lo"], rs["hi"]], axis=1),
            hovertemplate="<b>School %{x}</b><br>Raw slope: %{y:.3f} points/hour<br>SE: %{customdata[0]:.3f}"
                          "<br>Interval: [%{customdata[2]:.3f}, %{customdata[3]:.3f}]<br>p = %{customdata[1]:.3f}<extra></extra>"))
        fig.add_hline(y=0, line=dict(color="rgba(11,61,44,0.45)", width=1.2, dash="dot"),
                      annotation_text="zero", annotation_position="bottom right", annotation_font=dict(size=10, color=S.MUTED))
        fig.add_hline(y=mslope, line=dict(color=S.GOLD, width=1.5, dash="dash"), annotation_text=f"average {mslope:.2f}",
                      annotation_position="top right", annotation_font=dict(size=10, color=S.MUTED))
        C.base_layout(fig)
        fig.update_xaxes(tickangle=-90, tickfont=dict(size=9), showgrid=False)
        fig.update_yaxes(title_text="Points per extra study hour")
        chart(fig, "ex_slopes_chart")
        npos = int((rs["raw_hours_slope"] > 0).sum())
        txt = (f"Raw slopes run from <b>{rs['raw_hours_slope'].min():.2f}</b> to <b>{rs['raw_hours_slope'].max():.2f}</b> "
               f"points per hour, and {npos} of {len(rs)} are positive. Each school has only 30 students, so part of this "
               "spread is noise. The spread is the reason the multilevel model lets the study-hours slope vary by school.")
        for sid in sel_ids[:3]:
            rr = rs[rs["school_id"] == sid]
            if not rr.empty:
                txt = f"<b>School {sid:02d}</b>: raw slope {rr['raw_hours_slope'].iloc[0]:+.3f}. " + txt
        S.insight(txt)
