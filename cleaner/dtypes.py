"""Infer real data types for text columns and standardise category spellings."""
from __future__ import annotations

import re
import warnings

import numpy as np
import pandas as pd

from .utils import disp, format_dates, is_text

BOOL_MAP = {"true": True, "false": False, "yes": True, "no": False, "y": True, "n": False}
MONTH_RE = re.compile(r"jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec", re.IGNORECASE)
ISO_RE = re.compile(r"^\d{4}[-/.]\d{1,2}[-/.]\d{1,2}")
# Values like phone numbers / zip codes must stay text (leading zeros matter).
KEEP_TEXT_NAME = re.compile(
    r"phone|mobile|contact|zip|postal|pin\s?code|pincode|aadhaar|aadhar|account|ssn|isbn|barcode",
    re.IGNORECASE,
)


# ---------------------------------------------------------------- numbers
def _try_numeric(s: pd.Series) -> pd.Series:
    """Parse '₹55,000', '1,20,000', '(500)', '12%', '1.5k' ... -> float (NaN if not a number)."""
    t = s.str.replace(r"^(?i:rs\.?|inr|usd|eur)\s*", "", regex=True)
    t = t.str.replace(r"[₹$€£¥]", "", regex=True).str.replace(r"\s+", "", regex=True)
    grouped = t.str.fullmatch(r"-?\d{1,3}(,\d{2,3})+(\.\d+)?")  # 1,234 or Indian 1,23,456
    t = t.mask(grouped, t.str.replace(",", "", regex=False))
    t = t.str.replace(r"^\((.*)\)$", r"-\1", regex=True)
    t = t.str.rstrip("%")

    base = pd.to_numeric(t, errors="coerce")
    ext = t.str.extract(r"^(-?\d+(?:\.\d+)?)([kKmM])$")
    suffixed = ext[0].notna()
    if suffixed.any():
        mult = ext[1].str.lower().map({"k": 1e3, "m": 1e6})
        base = base.where(~suffixed, pd.to_numeric(ext[0], errors="coerce") * mult)
    return base.where(np.isfinite(base))


# ------------------------------------------------------------------ dates
def _parse_dates(s: pd.Series, dayfirst: bool) -> pd.Series:
    iso = s.str.match(ISO_RE)
    parts = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if iso.any():
            parts.append(pd.to_datetime(s[iso], format="mixed", dayfirst=False, errors="coerce", utc=True))
        if (~iso).any():
            parts.append(pd.to_datetime(s[~iso], format="mixed", dayfirst=dayfirst, errors="coerce", utc=True))
    out = pd.concat(parts).reindex(s.index)
    return out.dt.tz_localize(None)


def _looks_like_date(s: pd.Series, thr: float) -> pd.Series | None:
    cand = s.str.contains(r"\d") & (s.str.contains(r"[-/.]") | s.str.contains(MONTH_RE))
    return cand if cand.mean() >= thr else None


# ----------------------------------------------------------------- main
def infer_types(df: pd.DataFrame, log, opts) -> pd.DataFrame:
    df = df.copy()
    thr = opts["type_threshold"]

    for col in list(df.columns):
        if not is_text(df[col]):
            continue
        s = df[col].dropna()
        if s.empty:
            continue

        # 1) boolean (yes/no/true/false)
        low = s.str.lower()
        is_bool = low.isin(BOOL_MAP)
        if is_bool.any() and is_bool.mean() >= thr:
            for idx in s.index[~is_bool]:
                log.add("invalid_value", col, idx, old=s[idx], new=None, detail="expected yes/no/true/false")
            df[col] = low.map(BOOL_MAP).astype("boolean").reindex(df.index)
            log.add("type_converted", col, detail="text -> boolean")
            continue

        # 2) numbers
        keep_text = bool(KEEP_TEXT_NAME.search(col)) or bool(s.str.match(r"^0\d").any())
        if not keep_text:
            num = _try_numeric(s)
            ok = num.notna()
            if ok.mean() >= thr:
                vals = num[ok]
                is_int = bool((vals % 1 == 0).all() and (vals.abs() < 2**53).all())
                plain = pd.to_numeric(s, errors="coerce").notna()
                for idx in s.index[ok & ~plain]:
                    log.add("format_standardized", col, idx, old=s[idx], new=num[idx], detail="symbols/commas removed")
                for idx in s.index[~ok]:
                    log.add("invalid_value", col, idx, old=s[idx], new=None, detail="not a valid number")
                conv = num.round().astype("Int64") if is_int else num.astype("float64")
                df[col] = conv.reindex(df.index)
                log.add("type_converted", col, detail=f"text -> {'integer' if is_int else 'decimal'}")
                continue

        # 3) dates
        cand = _looks_like_date(s, thr)
        if cand is not None:
            parsed = _parse_dates(s[cand], opts["dayfirst"]).reindex(s.index)
            if parsed.notna().sum() / len(s) >= thr:
                fmt_series = format_dates(parsed)
                for idx in s.index[parsed.notna()]:
                    if s[idx] != fmt_series[idx]:
                        log.add("format_standardized", col, idx, old=s[idx], new=fmt_series[idx], detail="date standardised to ISO")
                for idx in s.index[parsed.isna()]:
                    log.add("invalid_value", col, idx, old=s[idx], new=None, detail="not a valid date")
                df[col] = parsed.reindex(df.index)
                log.add("type_converted", col, detail="text -> date")
                continue
    return df


def _canonical(variants: pd.Series) -> str:
    return max(variants.items(), key=lambda kv: (kv[1], kv[0] == kv[0].title(), kv[0]))[0]


def standardize_categories(df: pd.DataFrame, log) -> pd.DataFrame:
    """'mumbai', 'MUMBAI', 'Mumbai' -> most common spelling (low-cardinality text only)."""
    df = df.copy()
    for col in df.columns:
        s = df[col]
        if not is_text(s):
            continue
        nn = s.dropna()
        if len(nn) < 6:
            continue
        key = nn.str.lower()
        if key.nunique() > max(2, 0.5 * len(nn)):
            continue
        for _, group in nn.groupby(key):
            variants = group.value_counts()
            if len(variants) < 2:
                continue
            canon = _canonical(variants)
            for idx, v in group.items():
                if v != canon:
                    df.at[idx, col] = canon
                    log.add("category_standardized", col, idx, old=v, new=canon, detail="spelling/case variant")
    return df
