"""Page 1: Overview Profile (descriptive; school, gender and student filters truly subset student rows)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils import charts as C
from utils import data as D
from utils import interpretations as I
from utils import styles as S
from views.common import chart, empty_note, filter_students, gender_filtered_students, selected_school_ids

CHAR_VARS = {"SES": "ses", "Prior Mathematics Score": "prior.score", "Study Hours": "study.hours"}
GENDER_COLORS = {"Male": S.BLUE, "Female": S.EMERALD}


def render(ctx) -> None:
    df, desc = filter_students(ctx)
    full = D.students()
    sel_ids = selected_school_ids(ctx)
    filtered = desc.startswith("Active")

    actions = S.header_row(
        "Overview Profile",
        "Profile of mathematics achievement, students, and schools.",
        chips=["Filtered View" if filtered else "Full Sample"],
        gold_chip=desc if filtered else None,
        deco="26h_books.png", actions_width=1.6,
    )
    with actions:
        cols = ["student_id", "school_id", "gender", "ses", "prior.score", "study.hours", "facility", "teacher.exp", "math.score"]
        st.download_button("Filtered data", df[cols].to_csv(index=False).encode("utf-8"), file_name="filtered_students.csv",
                           mime="text/csv", icon=":material/download:", width="stretch",
                           help=f"Download the {len(df)} student rows in the current view as CSV")

    n = len(df)
    sd = df["math.score"].std() if n > 1 else np.nan
    sch = df.drop_duplicates("school_id")  # school-level values: one row per school, not per student
    n_f = int((df["gender"] == "Female").sum())
    S.kpi_strip([
        dict(label="Students", value=f"{n:,}", icon="groups", tone="forest"),
        dict(label="Schools", value=f"{df['school_id'].nunique():,}", icon="apartment", tone="teal"),
        dict(label="Average Mathematics Score", value=I.num(df["math.score"].mean()), icon="calculate", tone="emerald"),
        dict(label="Standard Deviation of Mathematics Score", value=I.num(sd, na="–"), icon="stacked_bar_chart", tone="gold",
             help="How far scores typically sit from the average. A larger value means scores are more spread out."),
        dict(label="Average SES", value=I.num(df["ses"].mean()), icon="savings", tone="blue",
             help="SES (socioeconomic status) is an index of family economic and social background."),
        dict(label="Average Prior Mathematics Score", value=I.num(df["prior.score"].mean()), icon="menu_book", tone="mint",
             help="Mathematics score from an earlier assessment, before the final round."),
        dict(label="Average Study Hours", value=I.num(df["study.hours"].mean()), icon="schedule", tone="blue", sub="per week"),
        dict(label="Gender Composition", value=(f"{100 * n_f / n:.0f}% Female" if n else "–"), icon="wc", tone="slate",
             sub=(f"{n_f} female · {n - n_f} male" if n else None)),
        dict(label="Average Facility Score", value=I.num(sch["facility"].mean(), 1), icon="domain", tone="teal",
             help="School facility rating. Averaged over the schools in view, one value per school."),
        dict(label="Average Teacher Experience", value=I.num(sch["teacher.exp"].mean(), 1), icon="school", tone="terra",
             sub="years", help="Average teaching experience of each school's teachers, averaged over the schools in view."),
    ], cols=5)

    with st.container(key="ptabs"):
        tab_a, tab_b = st.tabs(["Achievement Overview", "Student & School Profiles"])
    with tab_a:
        _achievement(ctx, df, desc, full, sel_ids, filtered)
    with tab_b:
        _profiles(ctx, df, desc, sel_ids)


def _achievement(ctx, df, desc, full, sel_ids, filtered) -> None:
    n = len(df)
    c1, c2, c3 = st.columns([3, 4, 5], gap="small")
    with c1, S.card("narr_overview", rows=1):
        books = S.img_uri(S.deco_b64("26h_books.png"))
        gm = full["math.score"].mean()
        if not filtered or not n:
            sel = f"The overall average score is <b>{gm:.2f}</b>."
        else:
            diff = df["math.score"].mean() - gm
            sel = (f"This view averages <b>{df['math.score'].mean():.2f}</b>, {abs(diff):.2f} points "
                   f"{'above' if diff >= 0 else 'below'} the overall {gm:.2f}.")
        st.markdown(
            f"""<span class="pill">{S.icon('diversity_3')}Context</span><img class="narr-books" src="{books}"/>
<div class="narr-title">Achievement Reflects Both Students and Schools</div>
<div class="narr-body">The {len(full)} students are grouped in {full['school_id'].nunique()} schools, and students in the
same school tend to score alike. So scores vary both <b>within</b> and <b>between</b> schools. A raw gap between two school
averages is not automatically a school effect, because schools also enrol different students. {sel}</div>""",
            unsafe_allow_html=True,
        )

    with c2, S.card("ov_dist", rows=1):
        S.card_head("Distribution", "Mathematics Score Distribution",
                    f"Histogram with density curve · n = {n:,}", tone="emerald", pill_icon="bar_chart")
        if n < 3:
            empty_note("Select at least three students to draw a distribution. A single student's score is "
                       f"<b>{I.num(df['math.score'].iloc[0]) if n else '–'}</b>.")
        else:
            x = df["math.score"].to_numpy()
            mean, med = x.mean(), np.median(x)
            nbins = int(np.clip(np.sqrt(n) * 1.3, 8, 36))
            counts, edges = np.histogram(x, bins=nbins)
            bw = edges[1] - edges[0]
            fig = go.Figure()
            fig.add_trace(go.Bar(x=(edges[:-1] + edges[1:]) / 2, y=counts, width=bw * 0.94,
                                 marker=dict(color="rgba(63,174,128,0.62)", line=dict(width=0)), name="Students",
                                 customdata=np.stack([edges[:-1], edges[1:]], axis=1),
                                 hovertemplate="Score %{customdata[0]:.1f}–%{customdata[1]:.1f}<br>%{y} students<extra></extra>"))
            grid = np.linspace(x.min() - 2, x.max() + 2, 200)
            fig.add_trace(go.Scatter(x=grid, y=C.kde_curve(x, grid) * n * bw, mode="lines", name="Density",
                                     line=dict(color=S.FOREST_2, width=2), hoverinfo="skip"))
            fig.add_vline(x=mean, line=dict(color=S.GOLD, width=2, dash="dash"))
            fig.add_vline(x=med, line=dict(color=S.BLUE, width=2, dash="dot"))
            fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines", line=dict(color=S.GOLD, dash="dash", width=2),
                                     name=f"Mean {mean:.2f}"))
            fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines", line=dict(color=S.BLUE, dash="dot", width=2),
                                     name=f"Median {med:.2f}"))
            C.base_layout(fig, legend=True)
            fig.update_layout(margin=dict(t=22))
            fig.update_xaxes(title_text="Mathematics score")
            fig.update_yaxes(title_text="Students")
            chart(fig, "ov_dist")
            skew = pd.Series(x).skew()
            shape = "roughly symmetric" if abs(skew) < 0.3 else ("skewed to the right" if skew > 0 else "skewed to the left")
            S.insight(f"Scores centre on <b>{mean:.2f}</b> (median {med:.2f}) with a standard deviation of "
                      f"<b>{x.std(ddof=1):.2f}</b> and a range of {x.min():.1f} to {x.max():.1f}. The shape is {shape} "
                      f"(skewness {skew:.2f}).")

    with c3, S.card("ov_school", rows=1):
        g_df = gender_filtered_students(ctx)
        if ctx["gender"] == ctx["ALL"]:
            sm = D.schools()[["school_id", "n_students", "mean_math", "ci_math_low", "ci_math_high"]].copy()
        else:
            sm = D.school_means_from_students(g_df)
        sm = sm.sort_values("mean_math", ascending=False).reset_index(drop=True)
        sm["label"] = sm["school_id"].map(lambda s: f"S{int(s):02d}")
        sub = "Raw comparison: school means with 95% CI, not adjusted for student background"
        if ctx["gender"] != ctx["ALL"]:
            sub += f" · {ctx['gender']} students"
        S.card_head("Raw Comparison", "Average Mathematics Score by School", sub, tone="blue", pill_icon="apartment",
                    help="Each dot is a school's raw average score. The bar is a 95% confidence interval for that average.")
        colors = [S.GOLD if s in sel_ids else S.EMERALD for s in sm["school_id"]]
        fig = go.Figure(go.Scatter(
            x=sm["label"], y=sm["mean_math"], mode="markers",
            marker=dict(size=[13 if c == S.GOLD else 8 for c in colors], color=colors, line=dict(color="#FFFFFF", width=1.2)),
            error_y=dict(type="data", symmetric=False, array=sm["ci_math_high"] - sm["mean_math"],
                         arrayminus=sm["mean_math"] - sm["ci_math_low"], color="rgba(15,138,95,0.45)", thickness=1.6, width=2),
            customdata=np.stack([sm["ci_math_low"], sm["ci_math_high"], sm["n_students"], np.arange(1, len(sm) + 1)], axis=1),
            hovertemplate="<b>School %{x}</b><br>Mean: %{y:.2f}<br>95% CI: [%{customdata[0]:.2f}, %{customdata[1]:.2f}]"
                          "<br>n = %{customdata[2]}<br>Raw rank: %{customdata[3]}<extra></extra>"))
        grand = g_df["math.score"].mean()
        fig.add_hline(y=grand, line=dict(color=S.MUTED, width=1, dash="dot"),
                      annotation_text=f"overall mean {grand:.2f}", annotation_position="top right",
                      annotation_font=dict(size=10, color=S.MUTED))
        C.base_layout(fig)
        fig.update_xaxes(tickangle=-90, tickfont=dict(size=9), showgrid=False)
        fig.update_yaxes(title_text="Mean mathematics score")
        chart(fig, "ov_school")
        top, bot = sm.iloc[0], sm.iloc[-1]
        txt = (f"School averages range from <b>{bot['mean_math']:.2f}</b> (S{int(bot['school_id']):02d}) to "
               f"<b>{top['mean_math']:.2f}</b> (S{int(top['school_id']):02d}). ")
        for sid in sel_ids[:3]:
            if sid in set(sm["school_id"]):
                r = sm[sm["school_id"] == sid].iloc[0]
                pos = int(sm.index[sm["school_id"] == sid][0]) + 1
                rel = "above" if r["ci_math_low"] > grand else "below" if r["ci_math_high"] < grand else "close to"
                txt += f"<b>School {sid:02d}</b> ranks {I.ordinal(pos)} ({r['mean_math']:.2f}), {rel} the overall mean. "
        txt += ("These are raw averages. Schools enrol different students, so the adjusted comparison on the School "
                "Performance Explorer page can rank them differently.")
        S.insight(txt)


def _profiles(ctx, df, desc, sel_ids) -> None:
    n = len(df)
    c4, c5 = st.columns([6, 6], gap="small")
    with c4, S.card("ov_chars", rows=1):
        S.card_head("Student Level", "Student Characteristics",
                    f"Distribution in the current view · n = {n:,}", tone="mint", pill_icon="person")
        a, b = st.columns([3, 2], gap="small")
        with a:
            choice = st.segmented_control("Characteristic", list(CHAR_VARS), default="SES", key="ov_char",
                                          label_visibility="collapsed") or "SES"
        with b:
            mode = st.segmented_control("Split", ["Overall", "Compare Gender"], default="Overall", key="ov_char_mode",
                                        label_visibility="collapsed") or "Overall"
        col = CHAR_VARS[choice]
        if n < 2:
            empty_note(f"One student selected: {choice} = <b>{I.num(df[col].iloc[0]) if n else '–'}</b>.")
        else:
            fig = go.Figure()
            if mode == "Compare Gender":
                groups = [(g, df[df["gender"] == g], GENDER_COLORS.get(g, S.EMERALD)) for g in ["Male", "Female"]
                          if (df["gender"] == g).any()]
            else:
                name = "All students" if ctx["gender"] == ctx["ALL"] else f"{ctx['gender']} students"
                groups = [(name, df, S.EMERALD)]
            for g, sub, colr in groups:
                fig.add_trace(go.Violin(
                    x=[g] * len(sub), y=sub[col], name=g, box_visible=True, meanline_visible=True, points=False,
                    line=dict(color=colr, width=1.4), fillcolor=colr, opacity=0.55,
                    spanmode="hard", hovertemplate=f"{choice}: %{{y:.2f}}<extra>{g}</extra>"))
            C.base_layout(fig, violingap=0.3)
            fig.update_yaxes(title_text=choice)
            fig.update_xaxes(showgrid=False)
            chart(fig, "ov_chars")
            v = df[col]
            r = I.safe_corr(df[col], df["math.score"])
            q1, q3 = v.quantile([0.25, 0.75])
            txt = (f"{choice}: average <b>{v.mean():.2f}</b>, standard deviation {v.std():.2f}, middle half of students "
                   f"between {q1:.1f} and {q3:.1f}. Its raw correlation with mathematics score is <b>r = {r:.2f}</b> "
                   f"({I.corr_strength(r)}).")
            if mode == "Compare Gender" and len(groups) == 2:
                m = {g: s_[col].mean() for g, s_, _ in groups}
                txt += f" Male students average {m['Male']:.2f} and female students {m['Female']:.2f}."
            S.insight(txt + " This is a description, not an adjusted association.")

    with c5, S.card("ov_heat", rows=1):
        prof = D.sheet(D.S_PROFILE_Z).copy().sort_values("mean_math_z", ascending=False).reset_index(drop=True)
        zcols = ["mean_math_z", "mean_ses_z", "mean_prior_z", "mean_hours_z", "facility_z", "teacher_exp_z"]
        names = ["Mean Mathematics Score", "Mean SES", "Mean Prior Mathematics Score", "Mean Study Hours", "Facility Score",
                 "Teacher Experience"]
        xlab = [f"S{int(s):02d}" for s in prof["school_id"]]
        S.card_head("School Level", "School Characteristics Heatmap",
                    "Each column is a school, sorted by mean score · colours are z-scores (0 = average school)",
                    tone="slate", pill_icon="grid_view",
                    help="A z-score shows how far a school is from the average school, in standard deviations.")
        fig = go.Figure(go.Heatmap(
            z=prof[zcols].to_numpy().T, x=xlab, y=names, zmin=-2.5, zmax=2.5, zmid=0,
            colorscale=[[0, "#B9573E"], [0.3, "#E8B9A8"], [0.5, "#F4F2EC"], [0.7, "#9FD8BD"], [1, "#0B6B48"]],
            xgap=2, ygap=2, colorbar=dict(title=dict(text="z", font=dict(size=10)), thickness=8, len=0.95, outlinewidth=0,
                                          tickfont=dict(size=9)),
            hovertemplate="School %{x}<br>%{y}: z = %{z:.2f}<extra></extra>"))
        C.base_layout(fig)
        fig.update_layout(margin=dict(l=130))
        fig.update_xaxes(tickangle=-90, tickfont=dict(size=9), showgrid=False)
        fig.update_yaxes(showgrid=False, autorange="reversed")
        for sid in sel_ids:
            if f"S{sid:02d}" in xlab:
                idx = xlab.index(f"S{sid:02d}")
                fig.add_shape(type="rect", x0=idx - 0.5, x1=idx + 0.5, y0=-0.5, y1=len(names) - 0.5,
                              line=dict(color=S.GOLD, width=3))
        chart(fig, "ov_heat")
        cs = D.sheet(D.S_CORR_SCHOOL).set_index("variable")
        top = prof.iloc[0]
        txt = (f"Mean score tracks mean SES most closely (r = {cs.loc['mean_math', 'mean_ses']:.2f}). Its link with facility "
               f"(r = {cs.loc['mean_math', 'facility']:.2f}) and teacher experience (r = {cs.loc['mean_math', 'teacher.exp']:.2f}) "
               f"is weaker. The top school (S{int(top['school_id']):02d}) has SES z = {top['mean_ses_z']:+.2f} but facility "
               f"z = {top['facility_z']:+.2f}, so a high score does not mean the highest value on every characteristic.")
        if sel_ids:
            sid = sel_ids[0]
            r = prof[prof["school_id"] == sid].iloc[0]
            hi = max(zip(names, r[zcols]), key=lambda t: t[1])
            lo = min(zip(names, r[zcols]), key=lambda t: t[1])
            txt = (f"<b>School {sid:02d}</b> (gold frame) is highest on {hi[0]} (z = {hi[1]:+.2f}) and lowest on {lo[0]} "
                   f"(z = {lo[1]:+.2f}). " + txt)
        S.insight(txt)
