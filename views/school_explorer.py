"""Page 5: School Performance Explorer. School-specific values from the locked final model (13_final_school)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils import charts as C
from utils import data as D
from utils import interpretations as I
from utils import styles as S
from views.common import chart, selected_school_id


def _coef(fx: pd.DataFrame, term: str) -> float:
    return float(fx.loc[fx["predictor"] == term, "estimate"].iloc[0])


SEL_COLORS = [S.GOLD, S.BLUE, S.TERRA]
ADJ_HELP = "Performance relative to model expectation after accounting for observed characteristics."
ASSOC_HELP = ("How many extra points one more weekly study hour is associated with in this school "
              "(the average slope plus this school's random slope).")


def _join(vals) -> str:
    return " · ".join(vals)


def _rank_move(change) -> str:
    c = int(change)
    if c > 0:
        return f"↑ {c} position{'s' if c != 1 else ''}"
    if c < 0:
        return f"↓ {-c} position{'s' if c != -1 else ''}"
    return "No change"


def render(ctx) -> None:
    fs = D.final_school().copy()
    fx = D.sheet(D.S_FINAL_FIXED)
    mid = D.final_model_id()
    ids = sorted(fs["school_id"].astype(int).tolist())
    labels = [D.school_label(s) for s in ids]

    if "perf_schools" not in st.session_state:
        st.session_state["perf_schools"] = [labels[0]]
    st.session_state["perf_schools"] = [x for x in st.session_state["perf_schools"] if x in labels][:3]

    actions = S.header_row(
        "School Performance Explorer",
        "Compare raw and adjusted school performance.",
        chips=[(f"Final model {mid}", D.model_tooltip(mid)), "Random intercept + study-hours slope"],
        deco="26b_leaf_top_right.png", actions_width=4.6,
    )
    cols = ["school_id", "mean_math", "raw_rank", "adjusted_intercept", "adjusted_intercept_low", "adjusted_intercept_high",
            "adjusted_rank", "rank_change", "hours_total_slope", "facility", "teacher.exp", "mean_ses", "mean_prior", "mean_hours"]
    with actions:
        a, b = st.columns([3.8, 1], vertical_alignment="bottom", gap="small")
        with a:
            st.multiselect("Schools to compare", labels, key="perf_schools", max_selections=3, placeholder="Choose up to 3 schools",
                           help="Choose up to three schools to compare. Gender and student filters do not apply on this page.")
        with b, st.popover("Table", icon=":material/table:", width="stretch"):
            st.caption("All schools: raw and adjusted values from the workbook (13_final_school).")
            st.dataframe(fs[cols].sort_values("adjusted_rank"), hide_index=True, width="stretch")
            st.download_button("Download school table (CSV)", fs[cols].to_csv(index=False).encode(),
                               file_name="school_effects_final.csv", mime="text/csv", icon=":material/download:")
    sel = [int(x.split()[-1]) for x in st.session_state["perf_schools"]]
    if not sel:
        sel = [ids[0]]
    rows = [fs[fs["school_id"] == s].iloc[0] for s in sel]
    color_of = {s: SEL_COLORS[i] for i, s in enumerate(sel)}

    with st.container(key="ptabs"):
        tab_a, tab_b = st.tabs(["Ranking & Adjusted Performance", "Study-Hours Heterogeneity"])
    with tab_a:
        _ranking_tab(fs, sel, rows, color_of)
    with tab_b:
        _hours_tab(fs, fx, mid, sel, rows, color_of)


def _ranking_tab(fs, sel, rows, color_of) -> None:
    n_sc = len(fs)
    S.kpi_strip(items=[
        dict(label="Raw Mean", value=_join(I.num(r["mean_math"]) for r in rows), icon="calculate", tone="forest",
             sub=_join(f"S{s:02d}" for s in sel), help="The school's raw average mathematics score."),
        dict(label="Raw Rank", value=_join(str(int(r["raw_rank"])) for r in rows), icon="format_list_numbered", tone="teal",
             sub=f"of {n_sc} schools", help="Rank by raw average score (1 = highest)."),
        dict(label="Adjusted Rank", value=_join(str(int(r["adjusted_rank"])) for r in rows), icon="swap_vert", tone="emerald",
             sub=f"of {n_sc} schools", help="Rank by adjusted school performance (1 = furthest above model expectation)."),
        dict(label="Rank Change", value=_join(_rank_move(r["rank_change"]).replace(" positions", "").replace(" position", "")
                                              for r in rows), icon="moving", tone="gold",
             sub="raw → adjusted", help="How many places the school moves from raw rank to adjusted rank."),
        dict(label="Adjusted School Performance", value=_join(I.signed(r["adjusted_intercept"], 2) for r in rows),
             icon="target", tone="gold", sub="points vs expectation", help=ADJ_HELP),
        dict(label="Facility Score", value=_join(I.num(r["facility"], 1) for r in rows), icon="domain", tone="blue",
             help="School facility rating."),
        dict(label="Teacher Experience", value=_join(I.num(r["teacher.exp"], 1) for r in rows), icon="school", tone="mint",
             sub="years", help="Average teaching experience of the school's teachers."),
    ], multi=len(sel) > 1)

    c1, c2 = st.columns([5, 7], gap="small")
    with c1, S.card("se_slope", rows=1):
        S.card_head("Ranking", "Does the Ranking Change After Adjustment?",
                    "Raw rank (left) vs adjusted rank (right) · 1 = highest", tone="gold", pill_icon="swap_vert",
                    help="Adjusted rank orders schools by performance relative to model expectation.")
        S.stats_row([(f"School {s:02d}:", _rank_move(r["rank_change"]), "gold" if i == 0 else ("blue" if i == 1 else "terra"))
                     for i, (s, r) in enumerate(zip(sel, rows))])
        fig = go.Figure()
        for _, s in fs.iterrows():
            if int(s["school_id"]) in sel:
                continue
            ch = s["rank_change"]
            col = "rgba(15,138,95,0.28)" if ch > 0 else ("rgba(185,87,62,0.28)" if ch < 0 else "rgba(107,123,115,0.25)")
            fig.add_trace(go.Scatter(x=["Raw rank", "Adjusted rank"], y=[s["raw_rank"], s["adjusted_rank"]], mode="lines+markers",
                                     line=dict(color=col, width=1.1), marker=dict(size=4, color=col),
                                     hovertemplate=f"School {int(s['school_id']):02d}<br>%{{x}}: %{{y}}<extra></extra>", showlegend=False))
        for s, r in zip(sel, rows):
            c = color_of[s]
            fig.add_trace(go.Scatter(x=["Raw rank", "Adjusted rank"], y=[r["raw_rank"], r["adjusted_rank"]], mode="lines+markers+text",
                                     line=dict(color=c, width=4), marker=dict(size=12, color=c, line=dict(color="#FFFFFF", width=2)),
                                     text=[f"S{s:02d} · {int(r['raw_rank'])}", f"{int(r['adjusted_rank'])} · S{s:02d}"],
                                     textposition=["middle left", "middle right"], textfont=dict(color=S.INK, size=11),
                                     hovertemplate=f"School {s:02d}<br>%{{x}}: %{{y}}<extra></extra>", showlegend=False))
        C.base_layout(fig)
        fig.update_yaxes(autorange="reversed", title_text="Rank", dtick=5, range=[n_sc + 1, 0])
        fig.update_xaxes(showgrid=False, range=[-0.6, 1.6])
        chart(fig, "se_slope_chart")
        moved = fs.loc[fs["rank_change"].abs().idxmax()]
        txt = " ".join(f"<b>School {s:02d}</b> moves from raw rank {int(r['raw_rank'])} to adjusted rank "
                       f"{int(r['adjusted_rank'])} ({_rank_move(r['rank_change'])})." for s, r in zip(sel, rows))
        S.insight(txt + f" Across all schools, {int((fs['rank_change'] != 0).sum())} of {n_sc} change position; the "
                  f"largest move is School {int(moved['school_id']):02d} ({int(moved['raw_rank'])} → "
                  f"{int(moved['adjusted_rank'])}). A school that rises was doing better than its student intake "
                  "would suggest.")

    with c2, S.card("se_cat", rows=1):
        d = fs.sort_values("adjusted_intercept").reset_index(drop=True)
        d["label"] = d["school_id"].map(lambda s: f"S{int(s):02d}")
        d["band"] = np.where(d["adjusted_intercept_low"] > 0, "Above expectation",
                             np.where(d["adjusted_intercept_high"] < 0, "Below expectation", "Around expectation"))
        S.card_head("Adjusted Performance", "Which Schools Perform Above Model Expectation?",
                    "Adjusted school performance with 95% intervals · 0 = model expectation", tone="emerald",
                    pill_icon="target", help=ADJ_HELP)
        counts = d["band"].value_counts()
        S.stats_row([("Above expectation:", int(counts.get("Above expectation", 0)), ""),
                     ("Around expectation:", int(counts.get("Around expectation", 0)), "blue"),
                     ("Below expectation:", int(counts.get("Below expectation", 0)), "terra")])
        band_col = {"Above expectation": S.EMERALD, "Around expectation": S.NEUTRAL, "Below expectation": S.TERRA}
        colors = [color_of.get(int(s), band_col[b]) for s, b in zip(d["school_id"], d["band"])]
        fig = go.Figure(go.Scatter(
            x=d["label"], y=d["adjusted_intercept"], mode="markers",
            marker=dict(size=[14 if int(s) in sel else 8 for s in d["school_id"]], color=colors,
                        line=dict(color="#FFFFFF", width=1.2)),
            error_y=dict(type="data", symmetric=False, array=d["adjusted_intercept_high"] - d["adjusted_intercept"],
                         arrayminus=d["adjusted_intercept"] - d["adjusted_intercept_low"], color="rgba(11,61,44,0.3)",
                         thickness=1.6, width=0),
            customdata=np.stack([d["adjusted_intercept_low"], d["adjusted_intercept_high"], d["raw_rank"], d["adjusted_rank"],
                                 d["band"]], axis=1),
            hovertemplate="<b>School %{x}</b> · %{customdata[4]}<br>Adjusted performance: %{y:+.3f} points"
                          "<br>95% interval: [%{customdata[0]:+.3f}, %{customdata[1]:+.3f}]"
                          "<br>Raw rank %{customdata[2]} → adjusted rank %{customdata[3]}<extra></extra>"))
        fig.add_hline(y=0, line=dict(color=S.FOREST_2, width=1.6, dash="dash"), annotation_text="Model expectation",
                      annotation_position="top left", annotation_font=dict(size=10, color=S.FOREST_2))
        C.base_layout(fig)
        fig.update_xaxes(showgrid=False, tickfont=dict(size=9), tickangle=-90)
        fig.update_yaxes(title_text="Points vs model expectation")
        chart(fig, "se_cat_chart")
        parts = []
        for s, r in zip(sel, rows):
            pos = ("above" if r["adjusted_intercept_low"] > 0 else
                   ("below" if r["adjusted_intercept_high"] < 0 else "around"))
            parts.append(f"<b>School {s:02d}</b> scores {r['adjusted_intercept']:+.2f} points against expectation ({pos} "
                         "expectation).")
        S.insight(f"{int(counts.get('Above expectation', 0))} schools sit clearly above model expectation and "
                  f"{int(counts.get('Below expectation', 0))} clearly below (their intervals exclude zero). " + " ".join(parts)
                  + " This compares schools with similar students; it is not a direct measure of how good a school is.")


def _hours_tab(fs, fx, mid, sel, rows, color_of) -> None:
    b_hours = _coef(fx, "hours_c")
    S.kpi_strip(items=[
        dict(label="Selected School(s)", value=_join(f"S{s:02d}" for s in sel), icon="apartment", tone="forest"),
        dict(label="Study-Hours Association", value=_join(f"{r['hours_total_slope']:+.3f}" for r in rows), icon="trending_up",
             tone="blue", sub="points/hour", help=ASSOC_HELP),
        dict(label="Average Study-Hours Association", value=f"{b_hours:+.3f}", icon="horizontal_rule", tone="emerald",
             sub="points/hour", help="The average (fixed) study-hours slope across all schools in the final model."),
        dict(label="Difference from Average", value=_join(f"{r['hours_random_effect']:+.3f}" for r in rows), icon="compare_arrows",
             tone="gold", sub="points/hour", help="This school's random slope: how far its study-hours association is from the average."),
        dict(label="Adjusted School Performance", value=_join(I.signed(r["adjusted_intercept"], 2) for r in rows),
             icon="target", tone="gold", sub="points vs expectation", help=ADJ_HELP),
        dict(label="Mean Study Hours", value=_join(I.num(r["mean_hours"], 2) for r in rows), icon="schedule", tone="teal",
             sub="per week", help="Average weekly study hours of the school's students."),
    ], multi=len(sel) > 1)

    c3, c4 = st.columns([7, 5], gap="small")
    with c3, S.card("se_hours", rows=1):
        d = fs.sort_values("hours_total_slope").reset_index(drop=True)
        d["label"] = d["school_id"].map(lambda s: f"S{int(s):02d}")
        d["lo"] = b_hours + d["hours_random_low"]
        d["hi"] = b_hours + d["hours_random_high"]
        S.card_head("Random Slope", "Is Study Time Linked to Scores Equally Everywhere?",
                    "School-specific adjusted study-hours associations with intervals · dashed line = average",
                    tone="blue", pill_icon="trending_up",
                    help="A random slope lets the study-hours association differ from school to school around the average.")
        colors = [color_of.get(int(s), S.BLUE_SOFT) for s in d["school_id"]]
        fig = go.Figure(go.Scatter(
            x=d["label"], y=d["hours_total_slope"], mode="markers",
            marker=dict(size=[14 if int(s) in sel else 8 for s in d["school_id"]], color=colors,
                        line=dict(color="#FFFFFF", width=1)),
            error_y=dict(type="data", symmetric=False, array=d["hi"] - d["hours_total_slope"],
                         arrayminus=d["hours_total_slope"] - d["lo"], color="rgba(62,120,178,0.3)", thickness=1.6, width=0),
            customdata=np.stack([d["hours_random_effect"], d["lo"], d["hi"]], axis=1),
            hovertemplate="<b>School %{x}</b><br>Study-hours association: %{y:+.3f} points/hour<br>Difference from average: "
                          "%{customdata[0]:+.3f}<br>Interval: [%{customdata[1]:.3f}, %{customdata[2]:.3f}]<extra></extra>"))
        fig.add_hline(y=b_hours, line=dict(color=S.EMERALD, width=2, dash="dash"),
                      annotation_text=f"average {b_hours:.3f} points/hour", annotation_position="top left",
                      annotation_font=dict(size=10, color=S.MUTED))
        fig.add_hline(y=0, line=dict(color="rgba(11,61,44,0.35)", width=1, dash="dot"))
        C.base_layout(fig)
        fig.update_xaxes(showgrid=False, tickfont=dict(size=9), tickangle=-90)
        fig.update_yaxes(title_text="Points per extra study hour")
        chart(fig, "se_hours_chart")
        parts = []
        for s, r in zip(sel, rows):
            rank_h = int((fs["hours_total_slope"] > r["hours_total_slope"]).sum()) + 1
            parts.append(f"<b>School {s:02d}</b>: {r['hours_total_slope']:+.3f} points per hour ({I.ordinal(rank_h)} "
                         f"steepest, {r['hours_random_effect']:+.3f} vs average).")
        S.insight(f"Across schools the study-hours association runs from {fs['hours_total_slope'].min():+.3f} to "
                  f"{fs['hours_total_slope'].max():+.3f} points per hour, around an average of {b_hours:+.3f}. "
                  + " ".join(parts) + " These are adjusted associations, not the result of assigning more study time.")

    with c4, S.card("se_curve", rows=1):
        stu = D.students()
        hours_mean = float((stu["study.hours"] - stu["hours_c"]).mean())
        p_female = float(stu["female"].mean())
        b0 = _coef(fx, "(Intercept)")
        b_fem, b_fac, b_tch = _coef(fx, "female"), _coef(fx, "facility_c"), _coef(fx, "teacher_c")
        h = np.linspace(stu["study.hours"].min(), stu["study.hours"].max(), 60)
        hc = h - hours_mean
        avg = b0 + b_fem * p_female + b_hours * hc
        S.card_head("Response Curve", "Selected School vs Average School",
                    "Predicted score across study hours · other predictors held constant", tone="gold",
                    pill_icon="show_chart",
                    help="SES and prior score are held at the sample average and gender at the sample mix; each school's "
                         "facility, teacher experience and random effects are included.")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=h, y=avg, mode="lines", name="Average school",
                                 line=dict(color=S.EMERALD, width=2.5, dash="dash"),
                                 hovertemplate="Average school<br>%{x:.1f} h/week → %{y:.2f}<extra></extra>"))
        gaps = []
        for s, r in zip(sel, rows):
            curve = (b0 + b_fem * p_female + b_fac * r["facility_c"] + b_tch * r["teacher_c"] + r["adjusted_intercept"]
                     + (b_hours + r["hours_random_effect"]) * hc)
            gaps.append((s, float(curve[np.argmin(np.abs(hc))] - avg[np.argmin(np.abs(hc))])))
            fig.add_trace(go.Scatter(x=h, y=curve, mode="lines", name=f"School {s:02d}",
                                     line=dict(color=color_of[s], width=3.2),
                                     hovertemplate=f"School {s:02d}<br>%{{x:.1f}} h/week → %{{y:.2f}}<extra></extra>"))
        C.base_layout(fig, legend=True)
        fig.update_layout(legend=dict(x=0, xanchor="left"))
        fig.update_xaxes(title_text="Weekly study hours")
        fig.update_yaxes(title_text="Predicted mathematics score")
        chart(fig, "se_curve_chart")
        S.insight(f"At the average study time ({hours_mean:.1f} hours a week) " +
                  "; ".join(f"School {s:02d} sits {g:+.2f} points from the average school" for s, g in gaps) +
                  ". The curves use only the locked final model's fixed effects and each school's stored random effects.")
