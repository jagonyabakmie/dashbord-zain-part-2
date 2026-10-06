"""Formatting and wording helpers for data-driven, non-causal interpretation text."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

NA = "Not available"


def isna(v) -> bool:
    try:
        return v is None or pd.isna(v)
    except (TypeError, ValueError):
        return False


def num(v, d: int = 2, na: str = NA) -> str:
    if isna(v):
        return na
    return f"{float(v):,.{d}f}"


def signed(v, d: int = 3, na: str = NA) -> str:
    if isna(v):
        return na
    return f"{float(v):+,.{d}f}"


def pct(v, d: int = 1, na: str = NA) -> str:
    if isna(v):
        return na
    return f"{100 * float(v):.{d}f}%"


def pval(p, na: str = NA) -> str:
    if isna(p):
        return na
    p = float(p)
    if p < 0.001:
        return "p < 0.001"
    return f"p = {p:.3f}"


def pval_short(p, na: str = NA) -> str:
    if isna(p):
        return na
    p = float(p)
    return "< 0.001" if p < 0.001 else f"{p:.3f}"


def corr_strength(r: float) -> str:
    if isna(r):
        return "undefined"
    a = abs(r)
    if a < 0.10:
        return "negligible"
    if a < 0.30:
        return "weak"
    if a < 0.50:
        return "moderate"
    if a < 0.70:
        return "fairly strong"
    return "strong"


def direction(v: float, pos: str = "positive", neg: str = "negative") -> str:
    return pos if v >= 0 else neg


def safe_corr(x, y) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = ~(np.isnan(x) | np.isnan(y))
    if ok.sum() < 3 or np.std(x[ok]) == 0 or np.std(y[ok]) == 0:
        return float("nan")
    return float(np.corrcoef(x[ok], y[ok])[0, 1])


def ok_flag(v) -> bool | None:
    if isna(v):
        return None
    return bool(int(v)) if not isinstance(v, bool) else v


def ordinal(n: int) -> str:
    n = int(n)
    suf = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suf}"


def is_finite(v) -> bool:
    try:
        return math.isfinite(float(v))
    except (TypeError, ValueError):
        return False
