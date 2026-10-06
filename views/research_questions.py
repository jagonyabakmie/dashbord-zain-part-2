"""Page 3: Research Questions. Locked full-sample results only; filters never subset anything here."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils import charts as C
from utils import data as D
from utils import interpretations as I
from utils import styles as S
from views.common import chart

L1_TERMS = ["ses_c", "prior_c", "hours_c", "female"]
L2_TERMS = ["facility_c", "teacher_c"]


def _std_ci(fx: pd.DataFrame) -> pd.DataFrame:
    """Standardized CI = raw CI scaled by the workbook's standardized_beta / estimate ratio."""
    fx = fx.copy()
    ratio = fx["standardized_beta"] / fx["estimate"]
    a, b = fx["ci_low"] * ratio, fx["ci_high"] * ratio
    fx["std_low"], fx["std_high"] = np.minimum(a, b), np.maximum(a, b)
    fx["label"] = fx["predictor"].map(D.TERM_LABELS).fillna(fx["predictor"])
    return fx


def _sig_label(p, control: bool = False) -> str:
    if I.isna(p):
        return "Not available"
    if p < 0.05:
        return "Significant (control)" if control else "Significant"
    return "Not significant"


def render(ctx) -> None:
    rq = D.sheet(D.S_RQ).set_index("rq")
    vp = D.sheet(D.S_VAR_PATH)
    fx = _std_ci(D.sheet(D.S_FINAL_FIXED))
    fx_model = fx["model_id"].iloc[0]
    inference = fx["inference"].iloc[0] if "inference" in fx else ""

    actions = S.header_row(
        "Research Questions",
        "Four research questions, four statistical answers.",
        chips=[(f"Final model {fx_model}", D.model_tooltip(fx_model)), inference, "Full sample"],
        deco="26k_sparkles.png", actions_width=2,
    )
    with actions, st.popover("Workbook tables", icon=":material/table:", width="stretch"):
        st.caption("Research-question summary (07_RQ_summary) and locked fixed effects (09_final_fixed_KR). "
                   "These results always use the full sample.")
        S.spec_box(fx_model)
        st.dataframe(D.sheet(D.S_RQ), hide_index=True, width="stretch")
        st.dataframe(D.sheet(D.S_FINAL_FIXED), hide_index=True, width="stretch")

    l1 = fx[fx["predictor"].isin(L1_TERMS)].copy()
    l1["abs"] = l1["standardized_beta"].abs()
    l1 = l1.sort_values("abs", ascending=False)
    strongest = l1.iloc[0]

    icc = rq.loc["RQ1", "metric_1"]
    within = rq.loc["RQ1", "metric_2"]
    within_red = rq.loc["RQ2", "metric_1"]
    add_between_red = rq.loc["RQ3", "metric_1"]
    between_red = rq.loc["RQ4", "metric_1"]
    remaining = rq.loc["RQ4", "metric_2"]

    S.kpi_strip([
        dict(label="ICC (Between Schools)", value=I.pct(icc), icon="apartment", tone="forest", sub="RQ1",
             help="ICC (intraclass correlation): the share of score differences that lies between schools rather than within them."),
        dict(label="Within-School Share", value=I.pct(within), icon="groups", tone="teal", sub="RQ1",
             help="The share of score differences that lies between students in the same school."),
        dict(label="Strongest Student Link", value=strongest["label"], icon="military_tech", tone="gold",
             sub=f"Standardized β = {strongest['standardized_beta']:.3f} · RQ2",
             help="Standardized β puts predictors on the same scale (standard deviations) so their strength can be compared."),
        dict(label="Within-School Explained", value=I.pct(within_red), icon="trending_down", tone="emerald", sub="RQ2",
             help="How much of the within-school variance the student characteristics account for, compared with the null model."),
        dict(label="Extra School Variance Explained", value=I.pct(add_between_red), icon="domain", tone="blue", sub="RQ3",
             help="How much more between-school variance is accounted for once school characteristics are added."),
        dict(label="School Variance Remaining", value=I.pct(remaining), icon="hourglass_bottom", tone="terra", sub="RQ4",
             help="Share of the original between-school variance that the final model still leaves unexplained."),
    ])

    # ---------------- RQ1 + RQ2 ----------------
    c1, c2 = st.columns(2, gap="small")
    with c1, S.card("rq1"):
        S.card_head("RQ 01", "Where Does the Variation Come From?",
                    "Null-model variance split: within schools vs between schools", tone="blue", pill_icon="pie_chart",
                    help="The null model has no predictors, so it shows how score variance divides between students and schools.")
        S.stats_row([("Within Schools", I.pct(within), ""), ("Between Schools", I.pct(icc), "gold"),
                      ("ICC =", f"{icc:.3f}", "blue",
                       "ICC: share of score variance between schools. 0 would mean schools do not differ at all.")])
        null = vp.iloc[0]
        tb, tw = null["between_school_variance"], null["within_student_variance"]
        fig = go.Figure()
        fig.add_trace(go.Bar(y=["Variance"], x=[within], orientation="h", name="Within schools",
                             marker=dict(color=S.EMERALD, line=dict(color="#FFFFFF", width=2)),
                             text=[f"Within schools {I.pct(within)} (σ² = {tw:.2f})"], textposition="inside",
                             insidetextanchor="middle", textfont=dict(color="#FFFFFF", size=12),
                             hovertemplate="Within schools: %{x:.1%}<extra></extra>"))
        fig.add_trace(go.Bar(y=["Variance"], x=[icc], orientation="h", name="Between schools (ICC)",
                             marker=dict(color=S.GOLD, line=dict(color="#FFFFFF", width=2)),
                             text=[f"{I.pct(icc)}"], textposition="outside", textfont=dict(color=S.INK, size=12),
                             hovertemplate=f"Between schools: %{{x:.1%}}<br>σ² = {tb:.3f}<extra></extra>"))
        C.base_layout(fig, legend=True, barmode="stack")
        fig.update_layout(margin=dict(t=24, l=10))
        fig.update_xaxes(tickformat=".0%", range=[0, 1.14], title_text="Share of total variance")
        fig.update_yaxes(showticklabels=False, showgrid=False)
        chart(fig, "rq1_chart")
        S.insight(f"About <b>{I.pct(icc)}</b> of the variation in scores lies between schools and <b>{I.pct(within)}</b> "
                  f"within schools (ICC = {icc:.3f}). Most differences are between students in the same school. Still, "
                  "students in one school are not independent, which is why a multilevel model is used.")

    with c2, S.card("rq2"):
        S.card_head("RQ 02 · Student Level", "Which Student Characteristics Matter Most?",
                    f"Standardized coefficients with 95% CI · final model ({inference})", tone="mint",
                    pill_icon="person_search",
                    help="Standardized β: the change in score (in SDs) for a one-SD change in the predictor, holding the others constant.")
        main_terms = l1[l1["predictor"] != "female"]
        sig_main = [r["label"] for _, r in main_terms.iterrows() if r["p_value"] < 0.05]
        gender = l1[l1["predictor"] == "female"]
        S.stats_row([
            ("Strongest student-level association:", strongest["label"], "gold"),
            ("Significant student predictors:", " • ".join(sig_main) if sig_main else "none", ""),
        ])
        d = l1.copy()
        d["color"] = [S.EMERALD if v >= 0 else S.TERRA for v in d["standardized_beta"]]
        d["label"] = [f"{lab} <span style='font-size:10px;color:{S.EMERALD}'>· {_sig_label(p, pr == 'female')}</span>"
                      for lab, p, pr in zip(d["label"], d["p_value"], d["predictor"])]
        d["hover"] = [f"Raw estimate {e:+.4f} (SE {se:.4f})<br>95% CI {lo:.4f} to {hi:.4f}<br>{I.pval(p)} · {_sig_label(p)}"
                      for e, se, lo, hi, p in zip(d["estimate"], d["std_error"], d["ci_low"], d["ci_high"], d["p_value"])]
        fig = C.forest(d, "label", "standardized_beta", "std_low", "std_high", color_col="color",
                       x_title="Standardized β", hover_extra="hover")
        fig.update_layout(margin=dict(l=150))
        fig.update_yaxes(range=[-0.6, len(d) - 0.4])
        chart(fig, "rq2_chart")
        g_txt = ""
        if not gender.empty and gender["p_value"].iloc[0] < 0.05:
            g_txt = (f" Gender is also significant as a control variable (β = {gender['standardized_beta'].iloc[0]:+.3f}).")
        S.insight(f"Holding the other predictors constant, <b>{strongest['label']}</b> has the strongest association with "
                  f"mathematics score (β = {strongest['standardized_beta']:.3f}). Significant student predictors: "
                  f"<b>{', '.join(sig_main)}</b>.{g_txt} Together, the student characteristics account for "
                  f"{I.pct(within_red)} of the within-school variance. These are associations, not proof of cause.")

    # ---------------- RQ3 + RQ4 ----------------
    c3, c4 = st.columns(2, gap="small")
    with c3, S.card("rq3"):
        S.card_head("RQ 03 · School Level", "Do School Characteristics Matter?",
                    "Standardized school-level coefficients with 95% CI (Kenward-Roger df)", tone="slate", pill_icon="domain",
                    help="Standardized effects allow a fair comparison between predictors measured on different scales.")
        d = fx[fx["predictor"].isin(L2_TERMS)].copy()
        d["abs"] = d["standardized_beta"].abs()
        strongest2 = d.sort_values("abs", ascending=False).iloc[0]
        sig2 = [D.TERM_LABELS[p] for p, pv in zip(d["predictor"], d["p_value"]) if pv < 0.05]
        S.stats_row([
            ("Strongest standardized school-level association:", D.TERM_LABELS[strongest2["predictor"]], "gold"),
            ("Significant school predictors:", " • ".join(sig2) if sig2 else "none", ""),
        ])
        sc = D.schools()
        sd_map = {"facility_c": sc["facility"].std(), "teacher_c": sc["teacher.exp"].std()}
        unit = {"facility_c": "per facility point", "teacher_c": "per year"}
        d["color"] = [S.BLUE] * len(d)
        d["label"] = [f"{D.TERM_LABELS[p]} <span style='font-size:10px;color:{S.EMERALD}'>· {_sig_label(pv)}</span>"
                      for p, pv in zip(d["predictor"], d["p_value"])]
        d["hover"] = [f"Raw estimate {e:+.4f} points {unit[p]} (95% CI {lo:.3f} to {hi:.3f})<br>df = {df_:.1f}, {I.pval(pv)}"
                      for e, lo, hi, p, df_, pv in zip(d["estimate"], d["ci_low"], d["ci_high"], d["predictor"], d["df"], d["p_value"])]
        fig = C.forest(d, "label", "standardized_beta", "std_low", "std_high", color_col="color",
                       x_title="Standardized β", hover_extra="hover")
        fig.update_layout(margin=dict(l=150))
        fig.update_yaxes(range=[-0.6, len(d) - 0.4])
        chart(fig, "rq3_chart")
        parts = [f"<b>{D.TERM_LABELS[r['predictor']]}</b>: β = {r['standardized_beta']:.3f}; raw {r['estimate']:+.3f} points "
                 f"{unit[r['predictor']]} ({I.pval(r['p_value'])}; 1 SD ≈ {r['estimate'] * sd_map[r['predictor']]:+.2f} points)"
                 for _, r in d.iterrows()]
        S.insight("; ".join(parts) + f". Both are significant, with <b>{D.TERM_LABELS[strongest2['predictor']]}</b> the "
                  f"stronger on the standardized scale, but both are small next to SES. Adding them accounts for a further "
                  f"{I.pct(add_between_red)} of the between-school variance. With 30 schools, read these as associations.")

    with c4, S.card("rq4"):
        S.card_head("RQ 04", "What Still Remains Unexplained?",
                    "Between-school variance from the null model to the final model", tone="terra",
                    pill_icon="waterfall_chart",
                    help="Each step adds predictors; the bar shows how much between-school variance that step accounts for.")
        S.stats_row([("", f"{I.pct(between_red)} explained", "", "Share of null-model between-school variance accounted for"),
                      ("", f"{I.pct(remaining)} remaining", "terra", "Share of between-school variance still unexplained")])
        stages = vp["model_stage"].tolist()
        bv = vp["between_school_variance"].to_numpy(dtype=float)
        step_names = {1: "+ Student<br>Characteristics", 2: "+ School<br>Characteristics", 3: "+ Random Study-<br>Hours Slope"}
        x_lab = ["Null<br>Model"] + [step_names.get(i, stages[i]) for i in range(1, len(bv))] + ["Remaining<br>Between-School"]
        measures = ["absolute"] + ["relative"] * (len(bv) - 1) + ["total"]
        fig = go.Figure(go.Waterfall(
            x=x_lab, y=[bv[0]] + list(np.diff(bv)) + [0], measure=measures,
            text=[f"{bv[0]:.3f}"] + [f"{v:+.3f}" for v in np.diff(bv)] + [f"{bv[-1]:.3f}"], textposition="outside",
            textfont=dict(size=10, color=S.INK),
            connector=dict(line=dict(color="rgba(11,61,44,0.25)", width=1, dash="dot")),
            increasing=dict(marker=dict(color=S.TERRA)), decreasing=dict(marker=dict(color=S.EMERALD)),
            totals=dict(marker=dict(color=S.GOLD)), hovertemplate="%{x}<br>%{y:.3f}<extra></extra>"))
        C.base_layout(fig)
        fig.update_yaxes(title_text="Between-school σ²", range=[0, bv[0] * 1.22])
        fig.update_xaxes(showgrid=False, tickfont=dict(size=10))
        chart(fig, "rq4_chart")
        last = vp.iloc[-1]
        slope_txt = (f" The final model also lets the study-hours slope vary by school (variance {last['random_slope_variance']:.4f})."
                     if pd.notna(last.get("random_slope_variance")) else "")
        S.insight(f"Between-school variance falls from <b>{bv[0]:.3f}</b> to <b>{bv[-1]:.3f}</b>: <b>{I.pct(between_red)}</b> "
                  f"is explained and <b>{I.pct(remaining)}</b> remains. So schools still differ in ways the measured "
                  f"characteristics do not capture.{slope_txt}")
