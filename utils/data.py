"""Workbook access layer.

Every number shown in the dashboard comes from MASTER_STREAMLIT_MULTILEVEL.xlsx, except the
leave-one-school-out robustness results, which come from LOSO_ROBUSTNESS_MULTILEVEL.xlsx.
Sheets are read one at a time (only when a page needs them) and cached with
st.cache_data, so widget changes never trigger a re-read of the workbook.
Large sheets are additionally mirrored to a parquet file in .cache/ (keyed on the
workbook's size and modification time) so a server restart stays fast.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent.parent
WORKBOOK = BASE_DIR / "MASTER_STREAMLIT_MULTILEVEL.xlsx"
LOSO_WORKBOOK = BASE_DIR / "LOSO_ROBUSTNESS_MULTILEVEL.xlsx"
CACHE_DIR = BASE_DIR / ".cache"

# Sheet names exactly as they exist in the workbook.
S_README = "00_README"
S_STUDENTS = "01_students"
S_SCHOOLS = "02_schools"
S_PROFILE_Z = "03_school_profiles_z"
S_RAW_SLOPES = "04_raw_hours_slopes"
S_CORR_STUDENT = "05_corr_student"
S_CORR_SCHOOL = "06_corr_school"
S_RQ = "07_RQ_summary"
S_VAR_PATH = "08_variance_path"
S_FINAL_FIXED = "09_final_fixed_KR"
S_FINAL_RANDOM = "10_final_random"
S_FINAL_DIAG = "11_final_diag"
S_FINAL_RDIAG = "12_final_random_diag"
S_FINAL_SCHOOL = "13_final_school"
S_CATALOG = "14_model_catalog"
S_COEFS = "15_model_coefficients"
S_RANDOM = "16_model_random"
S_DIAG = "17_model_diag"
S_RDIAG = "18_random_diag"
S_SCHOOL_EFFECTS = "19_school_effects"
S_TOP = "20_top_models"
S_MAP = "21_dashboard_map"
S_DICT = "22_data_dictionary"
S_PRED = "24_model_predictions"

# Sheets in the LOSO robustness workbook.
L_FULL = "02_Full_Model"
L_DASH = "03_LOSO_Dashboard"
L_SUMMARY = "04_LOSO_Summary"
L_FIT = "06_LOSO_Model_Fit"
L_LOG = "07_Run_Log"

# The workbook codes gender in Indonesian; the dashboard shows English labels.
GENDER_EN = {"Laki-laki": "Male", "Perempuan": "Female"}

LARGE_SHEETS = {S_PRED, S_SCHOOL_EFFECTS, S_COEFS, S_RANDOM}

# Level-1 / Level-2 predictors as coded in the workbook.
L1_PREDICTORS = [
    # (catalog state column, model term, raw column, label)
    ("ses_state", "ses_c", "ses", "SES"),
    ("prior_state", "prior_c", "prior.score", "Prior Mathematics Score"),
    ("hours_state", "hours_c", "study.hours", "Study Hours"),
    ("gender_state", "female", "female", "Gender"),
]
L2_PREDICTORS = [
    ("facility_state", "facility_c", "facility", "Facility Score"),
    ("teacher_state", "teacher_c", "teacher.exp", "Teacher Experience"),
]

TERM_LABELS = {
    "(Intercept)": "Intercept",
    "ses_c": "SES",
    "prior_c": "Prior Mathematics Score",
    "hours_c": "Study Hours",
    "female": "Gender (Female vs Male)",
    "facility_c": "Facility Score",
    "teacher_c": "Teacher Experience",
}
# Short names used inside technical formulas.
TERM_FORMULA = {
    "ses_c": "SES", "prior_c": "Prior", "hours_c": "StudyHours", "female": "Gender",
    "facility_c": "Facility", "teacher_c": "TeacherExperience",
}

# Student-level correlation matrix: gender is deliberately left out.
STUDENT_VARS = {
    "math.score": "Mathematics Score",
    "ses": "SES",
    "prior.score": "Prior Mathematics Score",
    "study.hours": "Study Hours",
}
SCHOOL_VARS = {
    "mean_math": "Mean Math",
    "mean_ses": "Mean SES",
    "mean_prior": "Mean Prior",
    "mean_hours": "Mean Hours",
    "facility": "Facility",
    "teacher.exp": "Teacher Exp",
}


def _workbook_stamp() -> str:
    stat = WORKBOOK.stat()
    return f"{stat.st_size}_{int(stat.st_mtime)}"


def _read_excel_sheet(sheet: str, header=0) -> pd.DataFrame:
    try:
        return pd.read_excel(WORKBOOK, sheet_name=sheet, header=header, engine="calamine")
    except Exception:  # calamine not installed or failed: fall back to openpyxl
        return pd.read_excel(WORKBOOK, sheet_name=sheet, header=header, engine="openpyxl")


@st.cache_data(show_spinner="Loading workbook sheet…")
def load_sheet(sheet: str, stamp: str | None = None) -> pd.DataFrame:
    """Read one sheet (parquet mirror for large sheets)."""
    stamp = stamp or _workbook_stamp()
    if sheet in LARGE_SHEETS:
        pq = CACHE_DIR / f"{sheet}_{stamp}.parquet"
        if pq.exists():
            try:
                return pd.read_parquet(pq)
            except Exception:
                pass
        df = _read_excel_sheet(sheet)
        try:
            CACHE_DIR.mkdir(exist_ok=True)
            for old in CACHE_DIR.glob(f"{sheet}_*.parquet"):
                old.unlink(missing_ok=True)
            df.to_parquet(pq, index=False)
        except Exception:
            pass
        return df
    return _read_excel_sheet(sheet)


def sheet(name: str) -> pd.DataFrame:
    return load_sheet(name, _workbook_stamp())


# ----------------------------------------------------------------------------
# Typed accessors
# ----------------------------------------------------------------------------

@st.cache_data
def readme_metrics(stamp: str) -> dict:
    raw = _read_excel_sheet(S_README, header=None)
    out = {}
    for _, row in raw.iterrows():
        k, v = row.iloc[0], row.iloc[1] if raw.shape[1] > 1 else None
        if isinstance(k, str) and pd.notna(v) and k != "Metric":
            out[k.strip()] = v
    notes = [r.iloc[0] for _, r in raw.iterrows()
             if isinstance(r.iloc[0], str) and r.iloc[0].startswith("IMPORTANT")]
    out["_note"] = notes[0] if notes else ""
    return out


def readme() -> dict:
    return readme_metrics(_workbook_stamp())


def final_model_id() -> str:
    """Locked final model: README 'Final report model bank ID', else model_id in 09_final_fixed_KR."""
    meta = readme()
    mid = meta.get("Final report model bank ID")
    if isinstance(mid, str) and mid.strip():
        return mid.strip()
    return str(sheet(S_FINAL_FIXED)["model_id"].iloc[0])


def best_model_id() -> str:
    cat = catalog()
    valid = cat[cat["status"] == "VALID"]
    return str(valid.sort_values("AICc").iloc[0]["model_id"])


def students() -> pd.DataFrame:
    df = sheet(S_STUDENTS).copy()
    df["school_label"] = df["school_id"].map(school_label)
    df["gender"] = df["gender"].map(GENDER_EN).fillna(df["gender"])
    return df


def schools() -> pd.DataFrame:
    return sheet(S_SCHOOLS)


def final_school() -> pd.DataFrame:
    return sheet(S_FINAL_SCHOOL)


def catalog() -> pd.DataFrame:
    return sheet(S_CATALOG)


def school_label(sid) -> str:
    return f"School {int(sid):02d}"


@st.cache_data
def _predictions_by_model(model_id: str, stamp: str) -> pd.DataFrame:
    pred = load_sheet(S_PRED, stamp)
    return pred[pred["model_id"] == model_id].reset_index(drop=True)


def predictions(model_id: str) -> pd.DataFrame:
    return _predictions_by_model(model_id, _workbook_stamp())


@st.cache_data
def _school_effects_by_model(model_id: str, stamp: str) -> pd.DataFrame:
    se = load_sheet(S_SCHOOL_EFFECTS, stamp)
    return se[se["model_id"] == model_id].reset_index(drop=True)


def school_effects(model_id: str) -> pd.DataFrame:
    return _school_effects_by_model(model_id, _workbook_stamp())


def model_rows(sheet_name: str, model_id: str) -> pd.DataFrame:
    df = sheet(sheet_name)
    return df[df["model_id"] == model_id].reset_index(drop=True)


def find_model(states: dict) -> pd.Series | None:
    """Map a predictor-state combination to the precomputed model in the catalog."""
    cat = catalog()
    mask = pd.Series(True, index=cat.index)
    for col, val in states.items():
        mask &= cat[col] == val
    hit = cat[mask]
    if hit.empty:
        return None
    return hit.iloc[0]


def model_spec(model_id: str) -> dict:
    """Readable specification of any model in the bank, built from its catalog state columns.

    Returns plain (functional form), formula (technical form), fixed/random term lists and
    the catalog formula. Level-1 states: 0 = excluded, 1 = fixed, 2 = fixed + random slope;
    level-2 states: 0 = excluded, 1 = fixed. Every model has a school random intercept.
    """
    cat = catalog()
    hit = cat[cat["model_id"] == model_id]
    if hit.empty:
        return {"model_id": model_id, "plain": "Specification not found in the model catalog.", "formula": "",
                "fixed": [], "random": [], "catalog_formula": ""}
    row = hit.iloc[0]
    fixed, random_ = [], []
    for state_col, term, _, _ in L1_PREDICTORS + L2_PREDICTORS:
        v = int(row[state_col])
        if v >= 1:
            fixed.append(term)
        if v == 2:
            random_.append(term)
    plain_fixed = [TERM_LABELS[t].replace("Gender (Female vs Male)", "Gender") for t in fixed]
    plain_random = ["School Random Intercept"] + [f"{TERM_LABELS[t]} Random Slope" for t in random_]
    plain = f"Math Score = f({', '.join(plain_fixed) if plain_fixed else 'intercept only'}; {', '.join(plain_random)})"
    rhs = " + ".join(TERM_FORMULA[t] for t in fixed) if fixed else "1"
    re_part = "1" + "".join(f" + {TERM_FORMULA[t]}" for t in random_)
    formula = f"MathScore ~ {rhs} + ({re_part} | School)"
    return {"model_id": model_id, "plain": plain, "formula": formula, "fixed": fixed, "random": random_,
            "catalog_formula": str(row.get("formula", ""))}


def model_tooltip(model_id: str) -> str:
    """Plain-text tooltip explaining a model ID (used wherever an ID is shown)."""
    sp = model_spec(model_id)
    return f"{model_id} is a catalog ID. Specification:\n{sp['plain']}\n\nFormula: {sp['formula']}"


# ----------------------------------------------------------------------------
# Leave-one-school-out robustness workbook
# ----------------------------------------------------------------------------

def _loso_stamp() -> str:
    if not LOSO_WORKBOOK.exists():
        return "missing"
    stat = LOSO_WORKBOOK.stat()
    return f"{stat.st_size}_{int(stat.st_mtime)}"


@st.cache_data(show_spinner="Loading robustness workbook…")
def _load_loso(sheet_name: str, stamp: str) -> pd.DataFrame:
    if stamp == "missing":
        return pd.DataFrame()
    try:
        return pd.read_excel(LOSO_WORKBOOK, sheet_name=sheet_name, engine="calamine")
    except Exception:
        return pd.read_excel(LOSO_WORKBOOK, sheet_name=sheet_name, engine="openpyxl")


def loso(sheet_name: str) -> pd.DataFrame:
    """One sheet of LOSO_ROBUSTNESS_MULTILEVEL.xlsx (empty frame if the workbook is missing)."""
    return _load_loso(sheet_name, _loso_stamp())


# ----------------------------------------------------------------------------
# Descriptive helpers for exploratory pages (computed on filtered student rows)
# ----------------------------------------------------------------------------

def school_means_from_students(df: pd.DataFrame) -> pd.DataFrame:
    """School-level aggregation (one row per school) with t-based 95% CI."""
    from scipy import stats

    g = df.groupby("school_id")
    out = g.agg(
        n_students=("math.score", "size"),
        mean_math=("math.score", "mean"),
        sd_math=("math.score", "std"),
        mean_ses=("ses", "mean"),
        mean_prior=("prior.score", "mean"),
        mean_hours=("study.hours", "mean"),
        facility=("facility", "first"),
        teacher_exp=("teacher.exp", "first"),
    ).reset_index().rename(columns={"teacher_exp": "teacher.exp"})
    se = out["sd_math"] / np.sqrt(out["n_students"])
    tcrit = stats.t.ppf(0.975, (out["n_students"] - 1).clip(lower=1))
    out["ci_math_low"] = out["mean_math"] - tcrit * se
    out["ci_math_high"] = out["mean_math"] + tcrit * se
    return out


def is_nan(v) -> bool:
    try:
        return v is None or (isinstance(v, float) and np.isnan(v)) or pd.isna(v)
    except Exception:
        return False
