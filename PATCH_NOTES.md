# Patch Notes

This patch updates the existing dashboard in place, following `PATCH_INSTRUCTIONS.md`. The palette, background, cards, sidebar and decorative assets are unchanged. No model is fitted inside Streamlit, no workbook number is changed or hard-coded, and no new package is needed.

Run it as before:

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Files changed

| File | Change |
|---|---|
| `app.py` | New sidebar names and page subtitles. RASIO 10.0 / Team Gabutt · 014SO2026 branding above Math Achievement. Multi-select School, single-choice Gender (All / Male / Female) and dependent multi-select Student ID. Per-page filter rules: real filters, highlight only, or none. Per-page layout settings. |
| `utils/data.py` | Reads `LOSO_ROBUSTNESS_MULTILEVEL.xlsx` (cached, with a safe fallback if the file is missing). Plain-English predictor labels. Gender shown as Male / Female. New `model_spec()` and `model_tooltip()` build the plain form and the technical formula of any model from the catalog. |
| `utils/styles.py` | Branding block with a 5% maths-symbol texture. Page tabs, "💡 Interpretation" popovers, ⓘ help icons on KPIs, chips and card titles, stat callouts, model specification box, two-row KPI grid. Layout heights adjusted so every page still fits one screen. |
| `utils/charts.py` | Least-squares fit line with a 95% confidence band for the scatter plots. |
| `views/common.py` | Multi-school and multi-student filtering, highlight lists, and the "Full Sample / Active View" summary. |
| `views/overview.py` | Overview Profile. |
| `views/exploration.py` | Exploratory Analysis. |
| `views/research_questions.py` | Research Questions. |
| `views/model_lab.py` | Multilevel Model Selection. |
| `views/school_explorer.py` | School Performance Explorer. |
| `views/diagnostics.py` | Diagnostics & Robustness. |
| `README.md` | Updated for the new pages, filters and the LOSO workbook. |
| `PATCH_NOTES.md` | This file. |

Not changed: `MASTER_STREAMLIT_MULTILEVEL.xlsx`, `LOSO_ROBUSTNESS_MULTILEVEL.xlsx`, `utils/interpretations.py`, `requirements.txt`, `.streamlit/config.toml`, everything in `assets/`.

## Features added

**All pages**
- Interpretations moved into "💡 Interpretation" popovers at the top right of each card, so charts get the full card height.
- ⓘ help icons on technical KPIs and terms (ICC, R², AICc, BIC, standardized β, adjusted school performance, p-values, LOSO and so on).
- Model IDs are never shown alone. Each one has a plain form, for example `Math Score = f(SES, Study Hours; School Random Intercept)`, and a technical formula. Both are built from the catalog, so they work for all 324 models.
- Page tabs keep each page on one screen with no vertical scroll (checked at 1366×650, 1536×760 and 1920×960). Below 1100px wide the page stacks and scrolls normally, for tablets and phones.

**Overview Profile**
- 10 KPIs in two rows of five. Facility and teacher averages use one row per school.
- Achievement Overview tab: context card, score distribution, raw school means.
- Student & School Profiles tab: SES / Prior Mathematics Score / Study Hours by Overall or Compare Gender, and a six-variable school heatmap.

**Exploratory Analysis**
- Student Relationships tab: scatter with fit line, 95% band, Pearson r, raw slope and n; correlation matrix with Student / School levels (gender removed from the student matrix).
- School Relationships tab: Facility or Teacher Experience vs school mean; raw study-hours slopes by school with a heterogeneity check.

**Research Questions**
- Still one page with four panels.
- RQ1 shows 91.8% within, 8.2% between and ICC 0.082.
- RQ2 and RQ3 label each predictor as significant or not, and name the strongest association and the significant predictors. Gender is marked as a significant control variable.
- RQ4 shows 40.7% explained and 59.3% remaining.

**Multilevel Model Selection** (replaces Model Laboratory)
- Select & Compare tab: Level-1 states Exclude / Fixed / Fixed + Random Slope; Final Model, Best AICc and Null Model buttons; "School Random Intercept: always included"; specification preview; larger leaderboard; KPIs for Model ID, AICc, ΔAICc, BIC, rank and status.
- Inspect Selected Model tab: R², ICC and variance KPIs, with Coefficients (Standardized β or Raw Estimate, hover shows SE, CI, p and significance), Variance, Fit (with RMSE), Equations and Diagnostics sub-tabs.

**School Performance Explorer**
- Compare up to three schools.
- Ranking & Adjusted Performance tab: "Adjusted School Performance" KPI and tooltip, slopegraph with ↑ / ↓ rank changes, and a caterpillar chart split into above, around and below expectation.
- Study-Hours Heterogeneity tab: school slopes and "Selected School vs Average School" response curves.

**Diagnostics & Robustness**
- Model Diagnostics tab: p-values next to each check (Breusch–Pagan, linearity, Shapiro–Wilk with "Normality not rejected at 5%" and "Close to 0.05 threshold", random-effects Q-Q with its p-value). Conclusions say "No evidence of violation", "Not rejected" and similar, never "PASS".
- Robustness tab, read from `LOSO_ROBUSTNESS_MULTILEVEL.xlsx`:
  - Coefficient stability across the 30 leave-one-school-out refits, with a selector for the six predictors, the full-sample line and the most influential school.
  - KPIs: Full Estimate, LOSO Range, Significant Runs, Largest Change, Most Influential School and Stability.
  - "Robustness Across Predictors" chart of the largest % change for each predictor.

## Not implemented as written, and why

- **Gender violin chart (old Exploratory page) removed.** The spec asks for two large charts per Exploratory tab, and gender comparison is already available in Overview → Student & School Profiles → Compare Gender.
- **Exploratory KPI strip replaced.** Pearson r, slope and n now sit inside each scatter card, next to the chart they describe, which leaves room for the two large charts without scrolling.
- **`ppt_border_reference.png` used as a style reference only.** The maths-symbol texture behind the sidebar branding is drawn in CSS at 5% opacity, inspired by that image. The image itself is not displayed, because at full size it would cover the existing sidebar design.
- **Old robustness checks moved, not deleted.** The previous top-10-models and ML vs REML comparison now opens from the "More checks" button in the Diagnostics header, since the Robustness tab is now based on LOSO.
- **Student-level residual plots** still use the ML fit stored in `24_model_predictions`, because the workbook keeps student residuals only there. The Shapiro–Wilk p from those residuals (0.0546) matches the workbook value (0.0545).
- **Mobile.** Desktop is the primary target. On phones the cards stack and the page scrolls, and interpretation popovers sit below each chart instead of at the top right.
