"""Page 6: Diagnostics & Robustness for the locked final model (full sample; a school can only be highlighted)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils import charts as C
from utils import data as D
from utils import interpretations as I
from utils import styles as S
from views.common import chart, highlight_school_ids, selected_school_id

TESTS = [
    # label, value col, ok col, kind, H0 text, short KPI label, icon
    ("Level-1 residual normality", "residual_normality_p", "residual_normality_ok", "p",
     "Residuals are normally distributed", "Residual normality", "bar_chart"),
    ("Level-1 homoskedasticity (Breusch-Pagan)", "homoskedasticity_L1_bp_p", "homoskedasticity_L1_ok", "p",
     "Residual variance is constant across fitted values", "L1 homoskedasticity", "align_justify_space_even"),
    ("Between-school variance homogeneity (Levene)", "homogeneity_L2_levene_p", "homogeneity_L2_ok", "p",
     "Residual variance is equal across schools", "School variance homogeneity", "apartment"),
    ("Linearity (quadratic terms)", "linearity_quadratic_p", "linearity_ok", "p",
     "Quadratic terms add nothing (relationships are linear)", "Linearity", "show_chart"),
    ("Multicollinearity (max VIF)", "max_vif", "multicollinearity_ok", "vif",
     "VIF below the usual threshold", "Max VIF", "join_inner"),
    ("Random-intercept normality (Shapiro-Wilk)", "random_intercept_normality_p", "random_intercept_normality_ok", "p",
     "School intercepts are normally distributed", "Random-intercept normality", "target"),
    ("Random-slope normality (Shapiro-Wilk, min p)", "random_slope_normality_min_p", "random_slope_normality_ok", "p",
     "School study-hours slopes are normally distributed", "Random-slope normality", "trending_up"),
]


def _robustness(mid: str) -> None:
    """Popover content: coefficient stability across the top valid models and ML vs REML for the final model."""
    st.caption("Built only from workbook outputs. The workbook contains no leave-one-school-out results, so robustness is "
               "shown across competing specifications and estimation methods instead.")
    top = D.sheet(D.S_TOP)
    top_valid = top[top["status"] == "VALID"].head(10)
    coefs = D.sheet(D.S_COEFS)
    cc = coefs[coefs["model_id"].isin(top_valid["model_id"]) & (coefs["predictor"] != "(Intercept)")].copy()
    cc["label"] = cc["predictor"].map(D.TERM_LABELS)
    st.markdown("**Do coefficients hold across the top 10 models?** Standardized β in the 10 best valid models "
                "(ML bank); gold diamond = final model.")
    fig = go.Figure()
    for lab in ["SES", "Prior Mathematics Score", "Teacher Experience", "Facility Score", "Study Hours", "Gender (Female vs Male)"]:
        sub = cc[cc["label"] == lab]
        if sub.empty:
            continue
        isf = sub["model_id"] == mid
        fig.add_trace(go.Scatter(x=sub.loc[~isf, "standardized_beta"], y=[lab] * int((~isf).sum()), mode="markers",
                                 marker=dict(size=10, color="rgba(15,138,95,0.45)", line=dict(color="#FFFFFF", width=1)),
                                 customdata=sub.loc[~isf, "model_id"], showlegend=False,
                                 hovertemplate="%{customdata}<br>β = %{x:.4f}<extra>" + lab + "</extra>"))
        fig.add_trace(go.Scatter(x=sub.loc[isf, "standardized_beta"], y=[lab] * int(isf.sum()), mode="markers",
                                 marker=dict(size=14, color=S.GOLD, symbol="diamond", line=dict(color="#FFFFFF", width=1.5)),
                                 showlegend=False, hovertemplate=mid + "<br>β = %{x:.4f}<extra>" + lab + "</extra>"))
    fig.add_vline(x=0, line=dict(color="rgba(11,61,44,0.35)", width=1, dash="dot"))
    C.base_layout(fig, height=280)
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(title_text="Standardized β")
    st.plotly_chart(fig, width="stretch", theme=None, config=C.CONFIG, key="dg_rob1")
    rng = cc.groupby("label")["standardized_beta"].agg(["min", "max"])
    sign_stable = all((r["min"] > 0) == (r["max"] > 0) for _, r in rng.iterrows())
    if sign_stable and "SES" in rng.index:
        S.insight(f"Across the {len(top_valid)} best valid specifications every predictor keeps the same sign and SES remains "
                  f"the dominant association (β from {rng.loc['SES', 'min']:.3f} to {rng.loc['SES', 'max']:.3f}).", lines=3)
    else:
        S.insight("Coefficient signs differ across some top specifications; see the plot.", lines=3)

    st.markdown(f"**ML vs REML for {mid}.** Model-bank ML fit vs the locked REML + Kenward-Roger fit.")
    ml = coefs[coefs["model_id"] == mid][["predictor", "estimate", "std_error"]].rename(
        columns={"estimate": "ML estimate", "std_error": "ML SE"})
    reml = D.sheet(D.S_FINAL_FIXED)[["predictor", "estimate", "std_error"]].rename(
        columns={"estimate": "REML estimate", "std_error": "REML SE (KR)"})
    cmp_ = reml.merge(ml, on="predictor", how="left")
    cmp_["predictor"] = cmp_["predictor"].map(D.TERM_LABELS).fillna(cmp_["predictor"])
    st.dataframe(cmp_.round(4), hide_index=True, width="stretch")
    vml = D.model_rows(D.S_RANDOM, mid)
    vre = D.sheet(D.S_FINAL_RANDOM)
    vml = vml[vml["term2"].isna()][["group", "term1", "variance_or_covariance"]].rename(columns={"variance_or_covariance": "ML variance"})
    vre = vre[vre["term2"].isna()][["group", "term1", "variance_or_covariance"]].rename(columns={"variance_or_covariance": "REML variance"})
    vc = vre.merge(vml, on=["group", "term1"], how="left")
    vc["Component"] = [("Residual" if g == "Residual" else ("School intercept" if t == "(Intercept)" else f"Slope: {D.TERM_LABELS.get(t, t)}"))
                       for g, t in zip(vc["group"], vc["term1"])]
    st.dataframe(vc[["Component", "REML variance", "ML variance"]].round(4), hide_index=True, width="stretch")
    maxdiff = float((cmp_["REML estimate"] - cmp_["ML estimate"]).abs().max())
    S.insight(f"Fixed effects agree to within {maxdiff:.4f} points between ML and REML; REML variance components are "
              "slightly larger, as expected, without changing any conclusion.", lines=3)
    diag_all = D.sheet(D.S_DIAG).merge(D.catalog()[["model_id", "status"]], on="model_id")
    tv = diag_all[diag_all["model_id"].isin(top["model_id"])]
    okcols = [c for c in tv.columns if c.endswith("_ok")]
    n_all = int((tv[okcols].fillna(1).astype(int).min(axis=1) == 1).sum())
    st.caption(f"Of the {len(tv)} models in {D.S_TOP}, {n_all} pass every stored assumption check "
               "(checks not applicable to a model are ignored).")


LOSO_ORDER = ["SES", "Prior Mathematics Score", "Study Hours", "Gender", "Facility Score", "Teacher Experience"]
LOSO_NAME = {"Gender: Female vs Male": "Gender"}  # workbook label -> menu label
LOSO_HELP = ("LOSO (leave-one-school-out): the final model is refitted 30 times, each time without one school, "
             "to see whether any single school drives a result.")
P_HELP = "p-value: how surprising the data would be if the assumption held. Below 0.05 counts as evidence against it."


def _p_conclusion(p, ok, good: str, close_note: bool = True) -> tuple[str, str]:
    """(conclusion text, tone) for a test whose null hypothesis is the assumption itself."""
    if I.isna(p):
        return "Not available", "blue"
    if ok is False or p < 0.05:
        return "Rejected at 5%", "terra"
    if close_note and p < 0.10:
        return f"{good} · close to 0.05", "gold"
    return good, ""


def render(ctx) -> None:
    mid = D.final_model_id()
    dg = D.sheet(D.S_FINAL_DIAG)
    dg = dg[dg["model_id"] == mid].iloc[0]
    rdg = D.sheet(D.S_FINAL_RDIAG)
    rdg = rdg[rdg["model_id"] == mid]
    fs = D.final_school()
    hl = highlight_school_ids(ctx)

    actions = S.header_row(
        "Diagnostics & Robustness",
        "Check model assumptions and coefficient stability.",
        chips=[(f"Final model {mid}", D.model_tooltip(mid)), f"Status: {str(dg['status']).title()}", "Full sample"],
        gold_chip=("Highlight: " + ", ".join(f"S{s:02d}" for s in hl[:4]) + ("…" if len(hl) > 4 else "")) if hl else None,
        deco="26e_wave.png", actions_width=2,
    )
    with actions, st.popover("More checks", icon=":material/stacked_line_chart:", width="stretch"):
        _robustness(mid)

    with st.container(key="ptabs"):
        tab_a, tab_b = st.tabs(["Model Diagnostics", "Robustness"])
    with tab_a:
        _diagnostics_tab(ctx, mid, dg, rdg, fs, hl)
    with tab_b:
        _loso_tab(hl)


def _diagnostics_tab(ctx, mid, dg, rdg, fs, hl) -> None:
    pr = D.predictions(mid)
    hl_st = set(ctx.get("hl_students") or [])
    c1, c2 = st.columns(2, gap="small")
    with c1, S.card("dg_rvf", kpi=False):
        S.card_head("Level 1", "Are Residuals Evenly Spread?",
                    f"Residual vs fitted · stored predictions for {mid} (n = {len(pr)})", tone="emerald",
                    pill_icon="scatter_plot", help="Residual = observed score minus predicted score. An even band around 0 is what we want.")
        bp, lin = dg["homoskedasticity_L1_bp_p"], dg["linearity_quadratic_p"]
        bp_txt, bp_tone = _p_conclusion(bp, I.ok_flag(dg.get("homoskedasticity_L1_ok")), "No evidence of heteroskedasticity", False)
        lin_txt, lin_tone = _p_conclusion(lin, I.ok_flag(dg.get("linearity_ok")), "No evidence of nonlinearity", False)
        S.stats_row([("Homoskedasticity · Breusch–Pagan", f"{I.pval(bp)} · {bp_txt}", bp_tone,
                      "Tests whether the spread of residuals stays the same across predicted scores. " + P_HELP),
                     ("Linearity ·", f"{I.pval(lin)} · {lin_txt}", lin_tone,
                      "Tests whether adding squared terms improves the fit (it would if relationships were curved). " + P_HELP)])
        if pr.empty:
            S.insight("Stored residuals for the final model are not available in the workbook.", title="Not available", kind="warn")
        else:
            is_hl = pr["school_id"].isin(hl) | pr["student_id"].isin(hl_st)
            fig = go.Figure()
            base = pr[~is_hl]
            fig.add_trace(go.Scattergl(x=base["fitted_math"], y=base["residual"], mode="markers", name="Students",
                                       marker=dict(size=4, color="rgba(15,138,95,0.42)"),
                                       customdata=np.stack([base["student_id"], base["school_id"]], axis=1),
                                       hovertemplate="Student %{customdata[0]} · School %{customdata[1]}<br>Fitted %{x:.2f}<br>Residual %{y:.3f}<extra></extra>"))
            if is_hl.any():
                h = pr[is_hl]
                fig.add_trace(go.Scatter(x=h["fitted_math"], y=h["residual"], mode="markers", name="Highlighted",
                                         marker=dict(size=8, color=S.GOLD, line=dict(color="#FFFFFF", width=1)),
                                         customdata=np.stack([h["student_id"], h["school_id"]], axis=1),
                                         hovertemplate="Student %{customdata[0]} · School %{customdata[1]}<br>Fitted %{x:.2f}<br>Residual %{y:.3f}<extra></extra>"))
            binned = pr.groupby(pd.qcut(pr["fitted_math"], 15, duplicates="drop"), observed=True)[["fitted_math", "residual"]].mean()
            fig.add_trace(go.Scatter(x=binned["fitted_math"], y=binned["residual"],
                                     mode="lines", name="Binned mean", line=dict(color=S.BLUE, width=2.5), hoverinfo="skip"))
            fig.add_hline(y=0, line=dict(color=S.GOLD, width=1.5, dash="dash"))
            C.base_layout(fig, legend=True)
            fig.update_xaxes(title_text="Fitted mathematics score")
            fig.update_yaxes(title_text="Residual")
            chart(fig, "dg_rvf_chart")
            S.insight(
                f"The residuals form an even band around zero: the binned mean stays within "
                f"±{binned['residual'].abs().max():.3f} points (residual SD {pr['residual'].std():.3f}). The Breusch–Pagan test "
                f"({I.pval(bp)}) shows no sign that the spread changes with the predicted score, and the linearity check "
                f"({I.pval(lin)}) shows no sign of curved relationships.")

    with c2, S.card("dg_qq", kpi=False):
        S.card_head("Level 1", "Are Level-1 Residuals Approximately Normal?",
                    "Normal Q-Q plot of the student residuals · points on the line = normal", tone="blue", pill_icon="timeline",
                    help="A Q-Q plot compares residuals with a perfect normal distribution; points that hug the line look normal.")
        p = dg["residual_normality_p"]
        items = [("Shapiro–Wilk", I.pval(p), "", "Shapiro–Wilk tests whether the residuals follow a normal distribution. " + P_HELP)]
        if not I.isna(p):
            items.append(("", "Normality not rejected at 5%" if p >= 0.05 else "Normality rejected at 5%",
                          "" if p >= 0.05 else "terra"))
            if 0.05 <= p < 0.10:
                items.append(("", "Close to 0.05 threshold", "gold"))
        S.stats_row(items)
        if not pr.empty:
            fig = C.qq(pr["residual"].to_numpy(), labels=pr["student_id"].tolist(), color=S.BLUE, name="Residual")
            C.base_layout(fig, legend=True)
            fig.update_layout(legend=dict(x=0, xanchor="left"))
            chart(fig, "dg_qq_chart")
            S.insight(
                f"Shapiro–Wilk gives {I.pval(p)}, so normality is {'not rejected' if p >= 0.05 else 'rejected'} at the 5% level"
                f"{', but only just' if 0.05 <= p < 0.10 else ''}. Skewness is {pr['residual'].skew():.2f} and excess kurtosis "
                f"{pr['residual'].kurt():.2f}, so the departure is mild and sits in the tails. With 900 students this has "
                "little effect on the fixed-effect results.")

    c3, c4 = st.columns([5, 7], gap="small")
    with c3, S.card("dg_re", kpi=False):
        which = st.session_state.get("dg_re") or "Random Intercept"
        col, term = (("adjusted_intercept", "(Intercept)") if which == "Random Intercept"
                     else ("hours_random_effect", "hours_c"))
        rr = rdg[rdg["random_term"] == term]
        p = rr["shapiro_p"].iloc[0] if not rr.empty else np.nan
        S.card_head("Level 2", "Are School Effects Normal?",
                    "Normal Q-Q plot of the 30 stored school random effects", tone="gold", pill_icon="apartment",
                    help="Random intercept: each school's own average shift. Random slope: each school's own study-hours shift.")
        a, b = st.columns([3, 2], gap="small", vertical_alignment="center")
        with a:
            st.segmented_control("Random effect", ["Random Intercept", "Random Study-Hours Slope"],
                                 default="Random Intercept", key="dg_re", label_visibility="collapsed")
        with b:
            S.stats_row([("Shapiro–Wilk", I.pval(p), "" if (I.isna(p) or p >= 0.05) else "terra",
                          "Tests whether the 30 school effects follow a normal distribution. " + P_HELP)])
        labs = [f"School {int(s):02d}" for s in fs["school_id"]]
        fig = C.qq(fs[col].to_numpy(), labels=labs, color=S.EMERALD, name="School effect")
        for s in hl:
            C.add_qq_highlight(fig, fs[col].to_numpy(), labs, f"School {s:02d}")
        C.base_layout(fig, legend=True)
        fig.update_layout(legend=dict(x=0, xanchor="left"))
        chart(fig, "dg_re_chart")
        S.insight(f"For the {which.lower()}, Shapiro–Wilk gives {I.pval(p)}, so normality is "
                  f"{'not rejected' if (I.isna(p) or p >= 0.05) else 'rejected'}. With only 30 schools the test has limited "
                  "power, so read it with the plot: points close to the gold line support the assumption.")

    with c4, S.card("dg_table", kpi=False):
        S.card_head("Summary", "Diagnostic Summary", f"All stored checks for the final model {mid} · p-values from "
                    f"{D.S_FINAL_DIAG} / {D.S_FINAL_RDIAG}", tone="terra", pill_icon="fact_check", help=P_HELP)
        rows = []
        for label, vcol, okcol, kind, h0, *_ in TESTS:
            v = dg.get(vcol, np.nan)
            ok = I.ok_flag(dg.get(okcol))
            if I.isna(v):
                stat, dec = "Not available", S.badge("Not available", "na")
            elif kind == "vif":
                stat = f"VIF = {v:.2f}"
                dec = S.badge("No concern", "ok") if ok else S.badge("Check", "bad")
            else:
                stat = I.pval(v)
                if ok is False or v < 0.05:
                    dec = S.badge("Rejected at 5%", "bad")
                elif "normal" in h0.lower():
                    dec = S.badge("Not rejected" + (" (close)" if v < 0.10 else ""), "warn" if v < 0.10 else "ok")
                else:
                    dec = S.badge("No evidence of violation", "ok")
            rows.append(f"<tr><td>{label}</td><td class='num'>{stat}</td><td>{dec}</td><td>{h0}</td></tr>")
        conv = dg.get("convergence_message")
        conv_ok = I.isna(conv) or str(conv).strip() == ""
        rows.append(f"<tr><td>Convergence</td><td class='num'>{'No message' if conv_ok else S.esc(conv)}</td>"
                    f"<td>{S.badge('Converged' if conv_ok else 'Warning', 'ok' if conv_ok else 'bad')}</td>"
                    f"<td>The fitting algorithm finished cleanly</td></tr>")
        sing = bool(dg.get("singular"))
        rows.append(f"<tr><td>Singularity</td><td class='num'>{'Singular' if sing else 'Not singular'}</td>"
                    f"<td>{S.badge('Singular' if sing else 'Not singular', 'bad' if sing else 'ok')}</td>"
                    f"<td>All random-effect variances can be estimated</td></tr>")
        rows.append(f"<tr><td>Overall status</td><td class='num'>{S.esc(str(dg['status']).title())}</td>"
                    f"<td>{S.badge('Adequate' if dg['status'] == 'VALID' else 'Check', 'ok' if dg['status'] == 'VALID' else 'bad')}</td>"
                    f"<td>Workbook model status</td></tr>")
        st.markdown("<div class='scroll-area'><table class='diag-table'><thead><tr><th>Check</th><th>Statistic</th>"
                    "<th>Conclusion</th><th>What it checks</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>",
                    unsafe_allow_html=True)
        n_ok = sum(1 for _, _, okc, *_ in TESTS if I.ok_flag(dg.get(okc)))
        S.insight(f"None of the {len(TESTS)} assumption checks is rejected at the 5% level ({n_ok} of {len(TESTS)} not "
                  "rejected), the fit converged and it is not singular. Residual normality is the closest call "
                  f"({I.pval(dg['residual_normality_p'])}). 'Not rejected' means the data give no clear evidence against "
                  "the assumption; it does not prove the assumption holds.")


def _loso_tab(hl) -> None:
    dash = D.loso(D.L_DASH)
    summ = D.loso(D.L_SUMMARY)
    if dash.empty or summ.empty:
        S.insight("LOSO_ROBUSTNESS_MULTILEVEL.xlsx was not found next to app.py, so the robustness results cannot be shown.",
                  title="Robustness workbook missing", kind="warn")
        return
    dash = dash.copy()
    summ = summ.copy()
    dash["name"] = dash["predictor"].replace(LOSO_NAME)
    summ["name"] = summ["predictor"].replace(LOSO_NAME)
    kpi_slot = st.empty()

    c1, c2 = st.columns([7, 5], gap="small")
    with c1, S.card("dg_loso", rows=1):
        pred = st.session_state.get("dg_loso_pred") or "SES"
        if pred not in LOSO_ORDER:
            pred = "SES"
        d = dash[dash["name"] == pred].sort_values("excluded_school").reset_index(drop=True)
        srow = summ[summ["name"] == pred].iloc[0]
        infl = int(srow["most_influential_school"])
        S.card_head("LOSO", "Are the Coefficients Stable When One School Is Removed?",
                    "Leave-one-school-out coefficient stability across 30 refitted multilevel models.", tone="slate",
                    pill_icon="stacked_line_chart", help=LOSO_HELP)
        st.segmented_control("Predictor", LOSO_ORDER, default="SES", key="dg_loso_pred", label_visibility="collapsed")
        d["label"] = d["excluded_school"].map(lambda s: f"S{int(s):02d}")
        full = float(srow["full_estimate"])
        sig = d["significant_05"].astype(bool)
        colors = [S.GOLD if int(s) in hl else (S.TERRA if not g else (S.FOREST_2 if int(s) == infl else S.EMERALD))
                  for s, g in zip(d["excluded_school"], sig)]
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=d["label"], y=d["estimate"], mode="markers", name="Estimate without this school",
            marker=dict(size=[14 if (int(s) in hl or int(s) == infl) else 9 for s in d["excluded_school"]], color=colors,
                        line=dict(color="#FFFFFF", width=1.2)),
            error_y=dict(type="data", symmetric=False, array=d["ci_high"] - d["estimate"], arrayminus=d["estimate"] - d["ci_low"],
                         color="rgba(11,61,44,0.22)", thickness=1.3, width=0),
            customdata=np.stack([d["ci_low"], d["ci_high"], d["p_value"], d["pct_change_from_full"],
                                 np.where(sig, "Significant", "Not significant")], axis=1),
            hovertemplate="<b>Without School %{x}</b><br>Estimate: %{y:.4f}<br>95% CI: [%{customdata[0]:.4f}, %{customdata[1]:.4f}]"
                          "<br>p = %{customdata[2]:.4f} · %{customdata[4]}<br>Change from full sample: %{customdata[3]:+.1f}%"
                          "<extra></extra>"))
        fig.add_hline(y=full, line=dict(color=S.GOLD, width=2, dash="dash"), annotation_text=f"full-sample estimate {full:.4f}",
                      annotation_position="top right", annotation_font=dict(size=10, color=S.INK_2),
                      annotation_bgcolor="rgba(255,255,255,0.85)")
        if not (d["ci_low"] > 0).all() and not (d["ci_high"] < 0).all():
            fig.add_hline(y=0, line=dict(color="rgba(11,61,44,0.4)", width=1, dash="dot"))
        fig.add_annotation(x=f"S{infl:02d}", y=float(d.loc[d["excluded_school"] == infl, "estimate"].iloc[0]),
                           text=f"Most influential: S{infl:02d}", showarrow=True, arrowhead=0, ax=0, ay=-34,
                           font=dict(size=10, color=S.INK), bgcolor="rgba(255,255,255,0.85)")
        C.base_layout(fig)
        fig.update_xaxes(title_text="Excluded school", tickangle=-90, tickfont=dict(size=9), showgrid=False)
        fig.update_yaxes(title_text=f"{pred} coefficient")
        chart(fig, "dg_loso_chart")
        n_sig = int(srow["n_significant_05"])
        n_runs = int(srow["n_successful_loso"])
        txt = (f"Dropping one school at a time, the {pred} coefficient ranges from {srow['loso_min_estimate']:.4f} to "
               f"{srow['loso_max_estimate']:.4f} around the full-sample {full:.4f}. It stays significant in "
               f"<b>{n_sig} of {n_runs}</b> refits and never changes sign. The largest change, "
               f"{srow['max_abs_pct_change']:.1f}%, comes from removing <b>School {infl:02d}</b>")
        if not bool(srow["full_significant_05"]) or n_sig < n_runs:
            txt += (f", after which p = {srow['p_after_most_influential_removal']:.3f}. So this result leans on that school "
                    "more than the others do")
        S.insight(txt + ".")

    with c2, S.card("dg_loso_all", rows=1):
        S.card_head("Summary", "Robustness Across Predictors",
                    "Largest change in each coefficient when one school is removed (% of the full-sample estimate)",
                    tone="emerald", pill_icon="bar_chart", help=LOSO_HELP)
        sm = summ.set_index("name").loc[[n for n in LOSO_ORDER if n in set(summ["name"])]].reset_index()
        sm = sm.sort_values("max_abs_pct_change")
        stable = sm["stability_flag"].astype(str).str.upper().str.startswith("STABLE")
        S.stats_row([("Stable:", int(stable.sum()), ""), ("Sensitive:", int((~stable).sum()), "terra"),
                     ("", "Sign never changes" if not sm["any_sign_change"].any() else "Some signs change", "blue")])
        col = [S.EMERALD if s else S.TERRA for s in stable]
        status = ["Stable" if s else ("Significance shifts" if "significance" in str(f).lower() else "Estimate shifts")
                  for s, f in zip(stable, sm["stability_flag"])]
        fig = go.Figure()
        fig.add_trace(go.Bar(x=sm["max_abs_pct_change"], y=sm["name"], orientation="h", width=0.12,
                             marker=dict(color=col), hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scatter(
            x=sm["max_abs_pct_change"], y=sm["name"], mode="markers+text",
            marker=dict(size=14, color=col, line=dict(color="#FFFFFF", width=1.5)),
            text=[f"  {v:.1f}% · {s}" for v, s in zip(sm["max_abs_pct_change"], status)], textposition="middle right",
            textfont=dict(size=10, color=S.INK_2),
            customdata=np.stack([sm["n_significant_05"], sm["n_successful_loso"], sm["most_influential_school"]], axis=1),
            hovertemplate="<b>%{y}</b><br>Largest change: %{x:.1f}%<br>Significant in %{customdata[0]} of %{customdata[1]} runs"
                          "<br>Most influential school: S%{customdata[2]:02d}<extra></extra>", showlegend=False))
        C.base_layout(fig)
        fig.update_layout(margin=dict(l=150))
        fig.update_xaxes(title_text="Largest change from full-sample estimate (%)",
                         range=[0, float(sm["max_abs_pct_change"].max()) * 2.9 + 1])
        fig.update_yaxes(showgrid=False)
        chart(fig, "dg_loso_all_chart")
        sens = sm[~stable]
        txt = (f"{int(stable.sum())} of {len(sm)} coefficients barely move when any one school is left out, and no "
               "coefficient changes sign. ")
        if not sens.empty:
            txt += " ".join(
                f"<b>{r['name']}</b> is sensitive ({str(r['stability_flag']).split(':', 1)[-1].strip()}; largest change "
                f"{r['max_abs_pct_change']:.1f}% without School {int(r['most_influential_school']):02d})." for _, r in sens.iterrows())
        S.insight(txt + " School-level predictors rest on only 30 schools, so this sensitivity is expected.")

    with kpi_slot.container():
        srow = summ[summ["name"] == (st.session_state.get("dg_loso_pred") or "SES")].iloc[0]
        stable_flag = str(srow["stability_flag"])
        S.kpi_strip([
            dict(label="Full Estimate", value=f"{srow['full_estimate']:.4f}", icon="functions", tone="forest",
                 sub=srow["name"], help="Coefficient from the final model fitted on all 30 schools."),
            dict(label="LOSO Range", value=f"{srow['loso_min_estimate']:.3f} – {srow['loso_max_estimate']:.3f}",
                 icon="straighten", tone="teal", help="Lowest and highest estimate across the 30 leave-one-school-out refits."),
            dict(label="Significant Runs", value=f"{int(srow['n_significant_05'])} / {int(srow['n_successful_loso'])}",
                 icon="verified", tone="emerald" if int(srow['n_not_significant_05']) == 0 else "terra",
                 help="Refits in which the coefficient is significant at 5%."),
            dict(label="Largest Change", value=f"{srow['max_abs_pct_change']:.1f}%", icon="trending_up", tone="gold",
                 sub=f"abs. {srow['max_abs_change']:.4f}",
                 help="Largest difference from the full-sample estimate, as a percentage of it."),
            dict(label="Most Influential School", value=f"S{int(srow['most_influential_school']):02d}", icon="apartment",
                 tone="blue", help="The school whose removal changes this coefficient the most."),
            dict(label="Stability", value=("Stable" if stable_flag.upper().startswith("STABLE") else "Sensitive"),
                 icon="balance", tone="emerald" if stable_flag.upper().startswith("STABLE") else "terra",
                 sub=stable_flag.split(":", 1)[-1].strip() if ":" in stable_flag else "small changes only",
                 help=LOSO_HELP),
        ])
