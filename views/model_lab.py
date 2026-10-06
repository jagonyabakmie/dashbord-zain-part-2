"""Page 4: Multilevel Model Selection. Every specification maps to a precomputed model in 14_model_catalog."""

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

L1_OPTS = {0: "Exclude", 1: "Fixed", 2: "Fixed + Random Slope"}
L2_OPTS = {0: "Exclude", 1: "Fixed"}
STATE_COLS = [c for c, *_ in D.L1_PREDICTORS] + [c for c, *_ in D.L2_PREDICTORS]

TEX = {"ses_c": r"\mathrm{SES}", "prior_c": r"\mathrm{Prior}", "hours_c": r"\mathrm{Hours}", "female": r"\mathrm{Female}",
       "facility_c": r"\mathrm{Facility}", "teacher_c": r"\mathrm{TeacherExp}"}


def _apply_states(states: dict) -> None:
    for col, val in states.items():
        st.session_state[f"lab_{col}"] = int(val)


def _states_of(model_id: str) -> dict:
    row = D.catalog().set_index("model_id").loc[model_id]
    return {c: int(row[c]) for c in STATE_COLS}


def _equations(states: dict) -> tuple[str, str, str]:
    l1 = [(t, states[c]) for c, t, *_ in D.L1_PREDICTORS if states[c] > 0]
    l2 = [(t, states[c]) for c, t, *_ in D.L2_PREDICTORS if states[c] > 0]
    # Level 1
    rhs = [r"\beta_{0j}"] + [rf"\beta_{{{k}j}}\,{TEX[t]}_{{ij}}" for k, (t, _) in enumerate(l1, start=1)]
    eq1 = r"\mathrm{Math}_{ij} = " + " + ".join(rhs) + r" + e_{ij},\quad e_{ij}\sim N(0,\sigma^2)"
    # Level 2
    g0 = [r"\gamma_{00}"] + [rf"\gamma_{{0{m}}}\,{TEX[t]}_{{j}}" for m, (t, _) in enumerate(l2, start=1)] + [r"u_{0j}"]
    lines = [r"\beta_{0j} = " + " + ".join(g0)]
    for k, (t, s) in enumerate(l1, start=1):
        lines.append(rf"\beta_{{{k}j}} = \gamma_{{{k}0}}" + (rf" + u_{{{k}j}}" if s == 2 else ""))
    eq2 = r"\begin{aligned}" + r" \\ ".join(l.replace(" = ", " &= ", 1) for l in lines) + r"\end{aligned}"
    # Combined
    fixed = [r"\gamma_{00}"] + [rf"\gamma_{{{k}0}}\,{TEX[t]}_{{ij}}" for k, (t, _) in enumerate(l1, start=1)] \
        + [rf"\gamma_{{0{m}}}\,{TEX[t]}_{{j}}" for m, (t, _) in enumerate(l2, start=1)]
    rand = [r"u_{0j}"] + [rf"u_{{{k}j}}\,{TEX[t]}_{{ij}}" for k, (t, s) in enumerate(l1, start=1) if s == 2]
    eq3 = (r"\mathrm{Math}_{ij} = \underbrace{" + " + ".join(fixed) + r"}_{\text{fixed}} + \underbrace{"
           + " + ".join(rand) + r"}_{\text{random (school)}} + e_{ij}")
    return eq1, eq2, eq3


def render(ctx) -> None:
    cat = D.catalog()
    final_id = D.final_model_id()
    best_id = D.best_model_id()
    n_valid = int((cat["status"] == "VALID").sum())

    if "lab_ses_state" not in st.session_state:
        _apply_states(_states_of(final_id))

    S.header_row(
        "Multilevel Model Selection",
        "Select predictors, compare specifications, and inspect model quality.",
        chips=[f"{len(cat)} precomputed models", f"{n_valid} valid · {len(cat) - n_valid} singular", "ML fits for AIC/AICc"],
        gold_chip=(f"Best AICc: {best_id}", D.model_tooltip(best_id)), deco="26j_decorative_circles.png",
    )

    states = {c: int(st.session_state.get(f"lab_{c}", 0) or 0) for c in STATE_COLS}
    m = D.find_model(states)
    if m is None:
        S.insight("This combination is not in the precomputed model bank.", title="Model not available", kind="warn")
        return
    mid = m["model_id"]
    with st.container(key="ptabs"):
        tab_a, tab_b = st.tabs(["Select & Compare Models", f"Inspect Selected Model ({mid})"])
    with tab_a:
        _select_compare(ctx, cat, m, states, final_id, best_id, n_valid)
    with tab_b:
        _inspect(ctx, m, states)


def _role(mid: str, final_id: str, best_id: str) -> str:
    roles = [r for r, ok in (("Final model", mid == final_id), ("Best AICc", mid == best_id)) if ok]
    return " · ".join(roles)


def _select_compare(ctx, cat, m, states, final_id, best_id, n_valid) -> None:
    mid = m["model_id"]
    is_valid = m["status"] == "VALID"
    warn = m["warnings"] if isinstance(m["warnings"], str) and m["warnings"].strip() else None
    ok_tone = "emerald" if is_valid and not warn else "terra"
    S.kpi_strip([
        dict(label="Selected Model ID", value=mid, icon="fingerprint", tone="forest",
             sub=_role(mid, final_id, best_id) or f"{int(m['npar'])} parameters", help=D.model_tooltip(mid)),
        dict(label="AICc", value=I.num(m["AICc"], 1), icon="analytics", tone="emerald", sub=f"AIC {I.num(m['AIC'], 1)}",
             help="AIC and AICc measure fit with a penalty for extra parameters; lower is better. AICc adds a "
                  "small-sample correction and is used for ranking."),
        dict(label="ΔAICc", value=I.num(m["delta_AICc_valid"], 2, na="Not ranked"), icon="leaderboard", tone="gold",
             help="Distance from the best valid model's AICc. 0 is the best model; above 10 means much less support."),
        dict(label="BIC", value=I.num(m["BIC"], 1), icon="analytics", tone="blue",
             help="Bayesian Information Criterion: like AIC but with a stronger penalty for extra parameters. Lower is better."),
        dict(label="Model Rank", value=(f"{int(m['AICc_rank_valid'])} / {n_valid}" if I.is_finite(m["AICc_rank_valid"])
                                        else "Not ranked"), icon="format_list_numbered", tone="teal",
             help="Position among the valid (non-singular) models, ranked by AICc."),
        dict(label="Status", value=m["status"].title(), icon="check_circle" if ok_tone == "emerald" else "warning",
             tone=ok_tone, sub="convergence warning" if warn else ("singular fit" if bool(m["singular"]) else "converged, not singular"),
             help="Valid means the fit converged and no random-effect variance sits at the boundary (not singular)."),
    ])

    c1, c2 = st.columns([4, 8], gap="small")
    with c1, S.card("lab_controls", rows=1):
        S.card_head("Build a Model", "Choose the Predictors", "Every choice maps to a precomputed model; nothing is re-fitted",
                    tone="emerald", pill_icon="tune")
        b1, b2, b3 = st.columns(3, gap="small")
        b1.button("Final Model", on_click=_apply_states, args=(_states_of(final_id),), width="stretch",
                  help=f"Load the locked final model. {D.model_spec(final_id)['plain']}")
        b2.button("Best AICc", on_click=_apply_states, args=(_states_of(best_id),), width="stretch",
                  help=f"Load the valid model with the lowest AICc. {D.model_spec(best_id)['plain']}")
        b3.button("Null Model", on_click=_apply_states, args=({c: 0 for c in STATE_COLS},),
                  width="stretch", help="Load the model with no predictors (school random intercept only).")
        S.stats_row([("School Random Intercept:", "always included", "gold",
                      "Every model lets each school have its own average score (random intercept).")])
        st.markdown('<div class="ctl-label">Level 1 · students</div>', unsafe_allow_html=True)
        slope_help = ("Fixed: one common slope for all schools. Fixed + Random Slope: the common slope plus a "
                      "school-specific deviation, so the relationship can differ between schools.")
        grid = st.columns(2, gap="small")
        for i, (sc, term, _, label) in enumerate(D.L1_PREDICTORS):
            with grid[i % 2]:
                st.selectbox(label, list(L1_OPTS), format_func=lambda v: L1_OPTS[v], key=f"lab_{sc}", help=slope_help)
        st.markdown('<div class="ctl-label">Level 2 · schools</div>', unsafe_allow_html=True)
        grid = st.columns(2, gap="small")
        for i, (sc, term, _, label) in enumerate(D.L2_PREDICTORS):
            with grid[i % 2]:
                st.selectbox(label, list(L2_OPTS), format_func=lambda v: L2_OPTS[v], key=f"lab_{sc}")
        st.markdown('<div class="ctl-label">Selected specification</div>', unsafe_allow_html=True)
        S.spec_box(mid)
        if not (is_valid and not warn):
            msgs = []
            if bool(m["singular"]):
                msgs.append("<b>Singular fit</b>: a random-effect variance or correlation is at the boundary, so "
                            "ΔAICc, conditional R² and ICC are not reported.")
            if warn:
                msgs.append(f"<b>Convergence warning</b>: {S.esc(warn)}")
            st.markdown(f'<div class="status-line bad">{S.icon("error")}<div>{"<br/>".join(msgs)}</div></div>',
                        unsafe_allow_html=True)

    with c2, S.card("lab_board", rows=1):
        S.card_head("Model Comparison", "Which Model Receives the Strongest Support?",
                    "Top 20 valid models by ΔAICc · gold = best AICc, dark green = your selection · hover a bar for its specification",
                    tone="gold", pill_icon="emoji_events",
                    help="ΔAICc is the distance from the best model. Below 2: similar support; above 10: much less support.")
        valid = cat[cat["status"] == "VALID"].sort_values("delta_AICc_valid")
        top = valid.head(20)
        if is_valid and mid not in set(top["model_id"]):
            top = pd.concat([top, valid[valid["model_id"] == mid]])
        top = top.reset_index(drop=True)
        colors = [S.GOLD if r == best_id else (S.FOREST_2 if r == mid else S.MINT) for r in top["model_id"]]
        ylab = [f"{r}{' ★' if r == best_id else ''}{' ◀' if r == mid else ''}" for r in top["model_id"]]
        specs = [D.model_spec(r) for r in top["model_id"]]
        fig = go.Figure(go.Bar(
            x=top["delta_AICc_valid"], y=ylab, orientation="h", marker=dict(color=colors, line=dict(color="#FFFFFF", width=1)),
            customdata=np.stack([[sp["plain"].replace("; ", ";<br>") for sp in specs], [sp["formula"] for sp in specs],
                                 top["AICc"], top["R2_marginal"], top["R2_conditional"]], axis=1),
            text=[f"{v:.1f}" for v in top["delta_AICc_valid"]], textposition="outside", textfont=dict(size=10, color=S.INK_2),
            hovertemplate="<b>%{y}</b><br>%{customdata[0]}<br>%{customdata[1]}<br>ΔAICc = %{x:.2f} · AICc = %{customdata[2]:.1f}"
                          "<br>Marginal R² = %{customdata[3]:.3f} · Conditional R² = %{customdata[4]:.3f}<extra></extra>"))
        C.base_layout(fig)
        fig.update_layout(margin=dict(l=70))
        fig.update_yaxes(autorange="reversed", showgrid=False, tickfont=dict(size=10))
        fig.update_xaxes(title_text="ΔAICc (0 = best valid model)", range=[0, top["delta_AICc_valid"].max() * 1.12 + 1])
        chart(fig, "lab_board_chart")
        sec = valid.iloc[1] if len(valid) > 1 else None
        bsp = D.model_spec(best_id)
        txt = (f"The best supported model is <b>{best_id}</b>: {S.esc(bsp['plain'])}. It has the lowest AICc of "
               f"{len(valid)} valid models")
        if sec is not None:
            txt += (f"; the runner-up {sec['model_id']} ({S.esc(D.model_spec(sec['model_id'])['plain'])}) is "
                    f"{sec['delta_AICc_valid']:.1f} units behind, so support for {best_id} is "
                    f"{'strong (ΔAICc above 10)' if sec['delta_AICc_valid'] > 10 else 'clear but not overwhelming'}")
        txt += ". "
        if mid != best_id:
            txt += (f"Your selection {mid} is {I.num(m['delta_AICc_valid'], 1)} units behind." if is_valid
                    else f"Your selection {mid} is singular and is not ranked.")
        S.insight(txt)


def _inspect(ctx, m, states) -> None:
    mid = m["model_id"]
    rnd = D.model_rows(D.S_RANDOM, mid)
    var = rnd[rnd["term2"].isna()].copy()
    v_int = var.loc[(var["group"] != "Residual") & (var["term1"] == "(Intercept)"), "variance_or_covariance"]
    v_res = var.loc[var["group"] == "Residual", "variance_or_covariance"]
    v_slope = var.loc[(var["group"] != "Residual") & (var["term1"] != "(Intercept)")]
    slope_val = (" · ".join(f"{v:.4f}" for v in v_slope["variance_or_covariance"]) if not v_slope.empty else "None")
    slope_sub = (", ".join(D.TERM_LABELS.get(t, t) for t in v_slope["term1"]) if not v_slope.empty else "no random slope")
    S.kpi_strip([
        dict(label="Marginal R²", value=I.num(m["R2_marginal"], 3), icon="donut_large", tone="mint",
             help="Share of score variance explained by the predictors (fixed effects) alone."),
        dict(label="Conditional R²", value=I.num(m["R2_conditional"], 3), icon="donut_small", tone="emerald",
             help="Share of score variance explained by the predictors plus the school random effects."),
        dict(label="ICC", value=I.num(m["ICC"], 3), icon="apartment", tone="slate",
             help="Share of the remaining (unexplained) variance that lies between schools in this model."),
        dict(label="Random Intercept Variance", value=I.num(v_int.iloc[0] if len(v_int) else np.nan, 3), icon="target",
             tone="gold", help="How much school averages still differ after the predictors (random intercept)."),
        dict(label="Random Slope Variance", value=slope_val, icon="trending_up", tone="blue", sub=slope_sub,
             help="How much a predictor's slope differs between schools (random slope)."),
        dict(label="Residual Variance", value=I.num(v_res.iloc[0] if len(v_res) else np.nan, 3), icon="scatter_plot",
             tone="teal", help="Variance left between students within the same school."),
    ])

    with S.card("lab_details", rows=1):
        sp = D.model_spec(mid)
        S.card_head("Selected Model", f"Inside Model {mid}", f"{S.esc(sp['plain'])}", tone="emerald", pill_icon="ssid_chart",
                    help=D.model_tooltip(mid))
        t1, t2, t3, t4, t5 = st.tabs(["Coefficients", "Variance", "Fit", "Equations", "Diagnostics"])
        with t1:
            coefs_all = D.model_rows(D.S_COEFS, mid)
            coefs = coefs_all[coefs_all["predictor"] != "(Intercept)"].copy()
            if coefs.empty:
                icpt = coefs_all["estimate"].iloc[0] if not coefs_all.empty else np.nan
                S.insight(f"The null model has no predictors; its intercept (the overall average) is <b>{I.num(icpt)}</b>.",
                          title="No predictors", kind="warn")
            else:
                scale = st.segmented_control("Scale", ["Standardized β", "Raw Estimate"], default="Standardized β",
                                             key="lab_scale", label_visibility="collapsed") or "Standardized β"
                coefs["label"] = coefs["predictor"].map(D.TERM_LABELS).fillna(coefs["predictor"])
                ratio = coefs["standardized_beta"] / coefs["estimate"]
                coefs["s_lo"] = np.minimum(coefs["ci_low"] * ratio, coefs["ci_high"] * ratio)
                coefs["s_hi"] = np.maximum(coefs["ci_low"] * ratio, coefs["ci_high"] * ratio)
                coefs["color"] = [S.EMERALD if s else S.NEUTRAL for s in coefs["significant_5pct"]]
                coefs["hover"] = [f"Estimate {e:+.4f} (SE {se:.4f})<br>95% CI [{lo:.4f}, {hi:.4f}]<br>{I.pval(p)} · "
                                  f"{'Significant' if sg else 'Not significant'} at 5%"
                                  for e, se, lo, hi, p, sg in zip(coefs["estimate"], coefs["std_error"], coefs["ci_low"],
                                                                  coefs["ci_high"], coefs["p_value"], coefs["significant_5pct"])]
                if scale == "Standardized β":
                    coefs = coefs.reindex(coefs["standardized_beta"].abs().sort_values(ascending=False).index)
                    fig = C.forest(coefs, "label", "standardized_beta", "s_lo", "s_hi", color_col="color",
                                   x_title="Standardized β", hover_extra="hover")
                else:
                    fig = C.forest(coefs, "label", "estimate", "ci_low", "ci_high", color_col="color",
                                   x_title="Raw estimate (points per unit)", hover_extra="hover")
                fig.update_layout(margin=dict(l=170))
                chart(fig, "lab_coef")
                nsig = int(coefs["significant_5pct"].sum())
                lead = coefs.loc[coefs["standardized_beta"].abs().idxmax()]
                S.insight(f"{nsig} of {len(coefs)} predictors are significant at 5% (green; grey means the 95% interval "
                          f"includes zero). The strongest standardized association is <b>{lead['label']}</b> "
                          f"(β = {lead['standardized_beta']:.3f}). Model-bank results are ML fits; the locked REML results "
                          "are on the Research Questions page.")
        with t2:
            var["label"] = [("Residual (within schools)" if g == "Residual" else
                             ("Random intercept (schools)" if t == "(Intercept)" else f"Random slope: {D.TERM_LABELS.get(t, t)}"))
                            for g, t in zip(var["group"], var["term1"])]
            var["color"] = [S.EMERALD if g == "Residual" else (S.GOLD if t == "(Intercept)" else S.BLUE)
                            for g, t in zip(var["group"], var["term1"])]
            fig = go.Figure(go.Bar(
                x=var["variance_or_covariance"], y=var["label"], orientation="h",
                marker=dict(color=var["color"], line=dict(color="#FFFFFF", width=1)),
                text=[f"{v:.4f}" for v in var["variance_or_covariance"]], textposition="outside", textfont=dict(size=11),
                customdata=var["stddev_or_correlation"],
                hovertemplate="%{y}<br>Variance: %{x:.4f}<br>SD: %{customdata:.4f}<extra></extra>"))
            C.base_layout(fig)
            fig.update_layout(margin=dict(l=190))
            fig.update_yaxes(autorange="reversed", showgrid=False)
            fig.update_xaxes(title_text="Variance", range=[0, var["variance_or_covariance"].max() * 1.25])
            chart(fig, "lab_var")
            cov = rnd[rnd["term2"].notna()]
            txt = (f"Random intercept variance {v_int.iloc[0]:.3f} vs residual {v_res.iloc[0]:.3f}. "
                   if len(v_int) and len(v_res) else "")
            if not cov.empty:
                txt += "; ".join(f"Correlation of the school intercept and the {D.TERM_LABELS.get(b, b)} slope: r = {r_:.3f}"
                                 for a, b, r_ in zip(cov["term1"], cov["term2"], cov["stddev_or_correlation"])) + ". "
                if (cov["stddev_or_correlation"].abs() > 0.95).any():
                    txt += "A correlation above |0.95| is close to the boundary, so read it with care."
            else:
                txt += "No random slopes are in this model."
            S.insight(txt)
        with t3:
            pr = D.predictions(mid)
            if pr.empty:
                S.insight("Predictions for this model are not available in the workbook.", title="Not available", kind="warn")
            else:
                rmse = float(np.sqrt(np.mean(pr["residual"] ** 2)))
                S.stats_row([("Marginal R² =", I.num(m["R2_marginal"], 3), ""),
                             ("Conditional R² =", I.num(m["R2_conditional"], 3), ""),
                             ("RMSE =", f"{rmse:.3f}", "gold", "Root mean squared residual, in score points (computed from the stored residuals)."),
                             ("Highlights only", "full-sample fit", "blue", "School/student selections in the sidebar only highlight points.")])
                hl_s, hl_st = set(ctx.get("hl_schools") or []), set(ctx.get("hl_students") or [])
                is_hl = pr["school_id"].isin(hl_s) | pr["student_id"].isin(hl_st)
                lo = min(pr["observed_math"].min(), pr["fitted_math"].min()) - 1
                hi = max(pr["observed_math"].max(), pr["fitted_math"].max()) + 1
                fig = go.Figure()
                base = pr[~is_hl]
                fig.add_trace(go.Scattergl(
                    x=base["fitted_math"], y=base["observed_math"], mode="markers", name="Students",
                    marker=dict(size=4, color="rgba(15,138,95,0.45)"),
                    customdata=np.stack([base["student_id"], base["school_id"], base["residual"]], axis=1),
                    hovertemplate="Student %{customdata[0]} · School %{customdata[1]}<br>Predicted: %{x:.2f}<br>Observed: %{y:.2f}"
                                  "<br>Residual: %{customdata[2]:.2f}<extra></extra>"))
                if is_hl.any():
                    h = pr[is_hl]
                    fig.add_trace(go.Scatter(
                        x=h["fitted_math"], y=h["observed_math"], mode="markers", name="Highlighted",
                        marker=dict(size=8, color=S.GOLD, line=dict(color="#FFFFFF", width=1)),
                        customdata=np.stack([h["student_id"], h["school_id"], h["residual"]], axis=1),
                        hovertemplate="Student %{customdata[0]} · School %{customdata[1]}<br>Predicted: %{x:.2f}<br>"
                                      "Observed: %{y:.2f}<br>Residual: %{customdata[2]:.2f}<extra></extra>"))
                fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", line=dict(color=S.GOLD, width=2, dash="dash"),
                                         hoverinfo="skip", name="Perfect prediction"))
                C.base_layout(fig, legend=bool(is_hl.any()))
                fig.update_xaxes(title_text="Predicted mathematics score", range=[lo, hi])
                fig.update_yaxes(title_text="Observed mathematics score", range=[lo, hi])
                chart(fig, "lab_pred")
                r = I.safe_corr(pr["fitted_math"], pr["observed_math"])
                S.insight(f"Predictions sit close to the diagonal: the typical error (RMSE) is <b>{rmse:.3f}</b> points and the "
                          f"observed–predicted correlation is {r:.3f} (n = {len(pr)}). Predictions include each school's random "
                          "effects, so this is in-sample fit.")
        with t4:
            eq1, eq2, eq3 = _equations(states)
            S.spec_box(mid)
            st.caption("Level 1 (students)")
            st.latex(eq1)
            st.caption("Level 2 (schools)")
            st.latex(eq2)
            st.caption("Combined model")
            st.latex(eq3)
            st.caption("Built from this model's actual terms. Predictors are grand-mean centred; Female is a 0/1 indicator.")
        with t5:
            dg = D.model_rows(D.S_DIAG, mid)
            rd = D.model_rows(D.S_RDIAG, mid)
            st.caption("Compact summary. The full diagnostics for the final model are on the Diagnostics & Robustness page.")
            st.markdown(f'<div class="scroll-area">{_diag_table(dg.iloc[0] if not dg.empty else None, rd, m)}</div>',
                        unsafe_allow_html=True)
            st.download_button("Coefficients CSV", D.model_rows(D.S_COEFS, mid).to_csv(index=False).encode(),
                               file_name=f"{mid}_coefficients.csv", mime="text/csv", icon=":material/download:")


DIAG_SPEC = [
    ("Level-1 residual normality (Shapiro–Wilk)", "residual_normality_p", "residual_normality_ok", "p"),
    ("Level-1 homoskedasticity (Breusch–Pagan)", "homoskedasticity_L1_bp_p", "homoskedasticity_L1_ok", "p"),
    ("Between-school variance homogeneity (Levene)", "homogeneity_L2_levene_p", "homogeneity_L2_ok", "p"),
    ("Linearity (quadratic terms)", "linearity_quadratic_p", "linearity_ok", "p"),
    ("Multicollinearity (max VIF)", "max_vif", "multicollinearity_ok", "vif"),
    ("Random-intercept normality", "random_intercept_normality_p", "random_intercept_normality_ok", "p"),
    ("Random-slope normality (min p)", "random_slope_normality_min_p", "random_slope_normality_ok", "p"),
]


def _diag_table(dg, rd: pd.DataFrame, m) -> str:
    rows = []
    for name, vcol, okcol, kind in DIAG_SPEC:
        v = dg[vcol] if dg is not None and vcol in dg else np.nan
        ok = I.ok_flag(dg[okcol]) if dg is not None and okcol in dg else None
        if I.isna(v):
            val, b = "Not available", S.badge("Not available", "na")
        elif kind == "vif":
            val = f"VIF = {v:.2f}"
            b = S.badge("No concern", "ok") if ok else S.badge("Check", "bad")
        else:
            val = I.pval(v)
            b = S.badge("Not rejected", "ok") if ok else (S.badge("Rejected at 5%", "bad") if ok is False
                                                          else S.badge("Not available", "na"))
        rows.append(f"<tr><td>{name}</td><td class='num'>{val}</td><td>{b}</td></tr>")
    for _, r in rd.iterrows():
        if r["random_term"] == "(Intercept)":
            continue
        okr = "Normal" in str(r["conclusion"]) and "Depart" not in str(r["conclusion"])
        rows.append(f"<tr><td>Slope normality: {D.TERM_LABELS.get(r['random_term'], r['random_term'])}</td>"
                    f"<td class='num'>{I.pval(r['shapiro_p'])}</td>"
                    f"<td>{S.badge('Not rejected' if okr else 'Rejected at 5%', 'ok' if okr else 'bad')}</td></tr>")
    sing = bool(m["singular"])
    rows.append(f"<tr><td>Singularity</td><td class='num'>{'Singular' if sing else 'Not singular'}</td>"
                f"<td>{S.badge('Singular' if sing else 'Not singular', 'bad' if sing else 'ok')}</td></tr>")
    w = m["warnings"] if isinstance(m["warnings"], str) and m["warnings"].strip() else None
    rows.append(f"<tr><td>Convergence</td><td>{S.esc(w) if w else 'No warning recorded'}</td>"
                f"<td>{S.badge('Warning' if w else 'Converged', 'bad' if w else 'ok')}</td></tr>")
    return ("<table class='diag-table'><thead><tr><th>Check</th><th>Statistic</th><th>Conclusion</th></tr></thead><tbody>"
            + "".join(rows) + "</tbody></table>")
