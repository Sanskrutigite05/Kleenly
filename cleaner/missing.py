"""Missing values: normalise fake nulls, drop sparse rows/columns, impute."""
from __future__ import annotations

import numpy as np
import pandas as pd
from pandas.api import types as pdt
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler

from .utils import is_id_like, is_numeric, is_text, to_float

NULL_TOKENS = {
    "", "na", "n/a", "n.a.", "#n/a", "nan", "null", "#null!", "none", "nil",
    "-", "--", "?", "missing", "undefined", "not available",
}


def normalize_nulls(df: pd.DataFrame, log, extra_tokens=()) -> pd.DataFrame:
    """Trim/collapse whitespace and turn placeholder strings into real NaN."""
    tokens = NULL_TOKENS | {str(t).strip().lower() for t in extra_tokens}
    out = df.copy()
    for col in out.columns:
        s = out[col]
        if not is_text(s):
            continue
        present = s.notna()
        cleaned = s.str.strip().str.replace(r"\s+", " ", regex=True)

        n_ws = int((present & (cleaned != s)).sum())
        log.add_many("whitespace_trimmed", n_ws, column=col)

        is_null = present & cleaned.str.lower().isin(tokens)
        fake = is_null & (cleaned != "")
        for idx in out.index[fake]:
            log.add("fake_null_normalized", col, idx, old=s[idx], new=None, detail="placeholder treated as missing")
        out[col] = cleaned.mask(is_null, np.nan)
    return out


def drop_sparse(df: pd.DataFrame, log, opts) -> pd.DataFrame:
    col_thr = opts["col_missing_threshold"]
    row_thr = opts["row_missing_threshold"]

    col_ratio = df.isna().mean()
    drop_cols = [c for c in df.columns if col_ratio[c] > col_thr]
    if len(drop_cols) == len(df.columns):
        drop_cols = []
    for c in drop_cols:
        log.add("column_dropped", c, detail=f"{col_ratio[c]:.0%} missing (limit {col_thr:.0%})")
    df = df.drop(columns=drop_cols)

    row_ratio = df.isna().mean(axis=1)
    drop_rows = df.index[row_ratio > row_thr]
    for idx in drop_rows:
        log.add("row_dropped_sparse", None, idx, detail=f"{row_ratio[idx]:.0%} of fields missing (limit {row_thr:.0%})")
    return df.drop(index=drop_rows)


def _cast(series: pd.Series, value):
    if pdt.is_integer_dtype(series):
        return int(round(float(value)))
    return float(value)


def _fill(df, col, miss, value, log, detail):
    idxs = df.index[miss]
    df.loc[miss, col] = value
    for idx in idxs:
        log.add("missing_filled", col, idx, old=None, new=value, detail=detail)


def impute(df: pd.DataFrame, log, opts) -> pd.DataFrame:
    df = df.copy()
    num_strategy = opts["numeric_impute"]
    cat_strategy = opts["categorical_impute"]
    num_cols = [c for c in df.columns if is_numeric(df[c]) and not is_id_like(df[c], c)]

    # ---- numeric columns ----
    done: set[str] = set()
    if num_strategy == "knn":
        cols = [c for c in num_cols if df[c].notna().any()]
        if len(cols) >= 2 and len(df) >= 3:
            X = np.column_stack([to_float(df[c]) for c in cols])
            if np.isnan(X).any():
                scaler = StandardScaler()
                Xi = KNNImputer(n_neighbors=min(5, len(df) - 1)).fit_transform(scaler.fit_transform(X))
                Xi = scaler.inverse_transform(Xi)
                for j, c in enumerate(cols):
                    miss = df[c].isna().to_numpy()
                    for pos in np.flatnonzero(miss):
                        idx = df.index[pos]
                        v = _cast(df[c], Xi[pos, j])
                        df.at[idx, c] = v
                        log.add("missing_filled", c, idx, old=None, new=v, detail="KNN imputation (k=5, scaled)")
                    done.add(c)
        else:
            log.note("KNN imputation needs at least 2 numeric columns; median was used instead.")
            num_strategy = "median"

    if num_strategy in ("median", "mean"):
        for c in num_cols:
            miss = df[c].isna()
            if not miss.any() or not df[c].notna().any():
                continue
            x = to_float(df[c])
            value = _cast(df[c], np.nanmedian(x) if num_strategy == "median" else np.nanmean(x))
            _fill(df, c, miss, value, log, f"{num_strategy} of the column")
            done.add(c)

    # ---- everything else ----
    for c in df.columns:
        s = df[c]
        miss = s.isna()
        n_miss = int(miss.sum())
        if n_miss == 0 or c in done:
            continue

        if c in num_cols or pdt.is_datetime64_any_dtype(s) or is_numeric(s):
            reason = "identifier column" if is_numeric(s) and c not in num_cols else None
            log.add_many("missing_left_unfilled", n_miss, column=c)
            if reason:
                log.note(f"'{c}' looks like an identifier, so its missing values were not guessed.")
            continue

        nonnull = s.dropna()
        if cat_strategy == "none" or nonnull.empty:
            log.add_many("missing_left_unfilled", n_miss, column=c)
            continue

        if pdt.is_bool_dtype(s):
            _fill(df, c, miss, bool(nonnull.mode().iloc[0]), log, "most common value")
            continue

        unique_ratio = nonnull.nunique() / len(nonnull)
        if cat_strategy == "unknown" or unique_ratio > 0.5:
            why = "chosen strategy" if cat_strategy == "unknown" else "mostly unique values, no reliable most-common value"
            _fill(df, c, miss, "Unknown", log, why)
        else:
            _fill(df, c, miss, nonnull.mode().iloc[0], log, "most common value")
    return df
