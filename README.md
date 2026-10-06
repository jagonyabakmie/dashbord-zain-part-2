# Mathematics Achievement Analytics

A six-page Streamlit dashboard for the RASIO 10.0 Statistics Olympiad final round: 900 students nested in 30 schools, analysed with two-level (multilevel) models.

Every number comes from `MASTER_STREAMLIT_MULTILEVEL.xlsx`, plus `LOSO_ROBUSTNESS_MULTILEVEL.xlsx` for the leave-one-school-out robustness tab. No model is re-fitted inside Streamlit.

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

Keep `app.py`, `utils/`, `views/`, `assets/`, `.streamlit/`, `MASTER_STREAMLIT_MULTILEVEL.xlsx` and `LOSO_ROBUSTNESS_MULTILEVEL.xlsx` in the same folder. If the LOSO workbook is missing, the Robustness tab shows a notice and the rest of the dashboard still works.

The one-screen layout needs Streamlit 1.50 or newer (charts use `height="stretch"`). If an older version is installed, run `pip install -U streamlit`.

## Layout

Each page fits one browser screen on laptops and desktops, with no page scroll: a slim header, a KPI strip, page tabs where a page has more than one view, and a grid of equal-height cards whose charts stretch to fill them. Interpretations open from a "💡 Interpretation" button at the top right of each card. Technical KPIs and terms carry an ⓘ icon; hover it for a plain-English explanation. Secondary tables (workbook tables, the all-schools table, model-bank checks) open from a button in the header.

Below 1100px wide or 600px tall the cards stack vertically and the page scrolls normally, so the dashboard stays usable on tablets and phones. On screens up to 720px tall the sidebar notes are hidden to keep the sidebar from scrolling.

## Project structure

```
app.py                     page config, sidebar navigation, filters, routing
.streamlit/config.toml     light theme, hides default page navigation
requirements.txt
utils/
  data.py                  workbook access: sheet constants, cached per-sheet loading, model lookup
  styles.py                palette, global CSS, decorative-asset processing, HTML components (hero, KPI, cards, insights)
  charts.py                transparent Plotly theme, forest plot, Q-Q plot, density helper
  interpretations.py       number / p-value formatting and wording helpers
views/
  common.py                exploratory filtering, chart wrapper
  overview.py              1. Overview Profile
  exploration.py           2. Exploratory Analysis
  research_questions.py    3. Research Questions
  model_lab.py             4. Multilevel Model Selection
  school_explorer.py       5. School Performance Explorer
  diagnostics.py           6. Diagnostics & Robustness
assets/branding/           RASIO 10.0 wordmark and partner logo strip (sidebar)
```

## Filtering logic

| Page | Sidebar behaviour |
|---|---|
| Overview Profile | Real filters: School (multi-select), Gender (All / Male / Female) and Student ID (multi-select, limited to the chosen schools and gender) subset the 900 student rows. KPIs, charts and text update, and the header shows an "Active View" summary. Facility and teacher averages use one row per school. |
| Exploratory Analysis | Same filters. The Student Relationships tab uses the filtered students; selected Student IDs are highlighted, not removed. The School Relationships tab always shows all 30 schools and only highlights the selected ones. |
| Research Questions | No filters. Always the locked final model (M293, REML + Kenward-Roger) on the full sample. |
| Multilevel Model Selection | Highlight only (schools, students). Predictor states (Exclude / Fixed / Fixed + Random Slope) map to one of the 324 precomputed models in `14_model_catalog`. |
| School Performance Explorer | Its own selector at the top of the page: up to 3 schools. Values come from the locked final model (`13_final_school`). |
| Diagnostics & Robustness | Highlight only. Diagnostics come from the locked model; robustness comes from the LOSO workbook. |

## Workbook sheets used

`00_README`, `01_students`, `02_schools`, `03_school_profiles_z`, `04_raw_hours_slopes`, `05_corr_student`, `06_corr_school`, `07_RQ_summary`, `08_variance_path`, `09_final_fixed_KR`, `10_final_random`, `11_final_diag`, `12_final_random_diag`, `13_final_school`, `14_model_catalog`, `15_model_coefficients`, `16_model_random`, `17_model_diag`, `18_random_diag`, `20_top_models`, `24_model_predictions`.

From `LOSO_ROBUSTNESS_MULTILEVEL.xlsx`: `03_LOSO_Dashboard` (one row per excluded school and predictor) and `04_LOSO_Summary` (one row per predictor).

Not used: `19_school_effects` (per-model BLUPs; the locked model's school effects come from `13_final_school`), `21_dashboard_map`, `22_data_dictionary`, `23_relationships` (documentation sheets that guided the build).

## Notes

- **Final model.** `00_README` names `M293` as both the best valid model and the final report model; `09_final_fixed_KR` marks it `LOCKED FINAL REPORT`.
- **Speed.** Sheets load one at a time and are cached with `st.cache_data`. The four largest sheets are also mirrored to `.cache/*.parquet` on first use, keyed on the workbook's size and modification time, so later restarts are fast. Delete `.cache/` at any time; it is rebuilt automatically.
- **Plots for the final model's residuals** use `24_model_predictions` (the ML model-bank fit of the same specification), because the workbook stores student-level residuals only there. The diagnostic tests use the REML results in `11_final_diag`.
- **Standardized confidence intervals** are the workbook's raw CIs multiplied by `standardized_beta / estimate`.
- **Response curve** (School Explorer) uses only the locked fixed effects and the school's stored random effects, with SES and prior score at the sample mean and gender at the sample proportion female.
- **Decorative assets.** Leaves, books, pencil, sparkles, waves, header and background art are used for decoration. Their caption strip is cropped off and the white backdrop is made transparent at runtime. KPI, card and chart PNGs that contain baked-in numbers are never displayed; all KPIs and charts are built from data.
- **Robustness.** The Robustness tab reads the leave-one-school-out results from `LOSO_ROBUSTNESS_MULTILEVEL.xlsx`: 30 refits of the final model, each with one school removed. The earlier top-10-models and ML vs REML comparison is still available from the "More checks" button in the page header.
- **Model IDs.** A model ID is never shown alone. Its plain form (for example "Math Score = f(SES, Study Hours; School Random Intercept)") and its technical formula are built from the catalog's predictor states, so they work for any of the 324 models.
- **Gender labels.** The workbook stores gender as Laki-laki / Perempuan; the dashboard shows Male / Female.
