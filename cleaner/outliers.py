"""Outlier detection: IQR / z-score per column + Isolation Forest across columns."""
from __future__ import annotations

import numpy as np
import pandas as pd
from pandas.api import types as pdt
from sklearn.ensemble import IsolationForest

from .utils import is_id_like, is_numeric, to_float

MIN_VALUES = 8       # need a few values before "outlier" means anything
MIN_ROWS_IF = 30     # Isolation Forest needs a reasonable sample


def _bounds(x: np.ndarray, method: str, opts):
    v = x[~np.isnan(x)]
    if method == "iqr":
        q1, q3 = np.percentile(v, [25, 75])
        iqr = q3 - q1
        if iqr == 0:
            return None
        k = opts["iqr_k"]
        return q1 - k * iqr, q3 + k * iqr
    std = v.std()
    if std == 0:
        return None
    t = opts["zscore_threshold"]
    return v.mean() - t * std, v.mean() + t * std


def _isolation_forest(df, cols, contamination):
    X = np.column_stack([to_float(df[c]) for c in cols])
    med = np.nanmedian(X, axis=0)
    nan_r, nan_c = np.where(np.isnan(X))
    X[nan_r, nan_c] = med[nan_c]
    pred = IsolationForest(n_estimators=200, contamination=contamination, random_state=42).fit_predict(X)
    std = X.std(axis=0)
    std[std == 0] = 1.0
    top = np.abs(X - med) / std
    return pred == -1, [cols[j] for j in top.argmax(axis=1)]


def handle_outliers(df: pd.DataFrame, log, opts) -> pd.DataFrame:
    method, action = opts["outlier_method"], opts["outlier_action"]
    cols = [
        c for c in df.columns
        if is_numeric(df[c]) and not pdt.is_bool_dtype(df[c])
        and not is_id_like(df[c], c) and df[c].notna().sum() >= MIN_VALUES
    ]
    if not cols:
        return df
    df = df.copy()

    # 1) find everything first (on the untouched data)
    found: list[tuple[str, np.ndarray, float, float]] = []
    if method != "none":
        for c in cols:
            x = to_float(df[c])
            b = _bounds(x, method, opts)
            if b is None:
                continue
            lo, hi = b
            mask = (x < lo) | (x > hi)
            if mask.any():
                found.append((c, mask, lo, hi))

    if_rows, if_cols = np.zeros(len(df), dtype=bool), []
    if opts["isolation_forest"] and len(cols) >= 2 and len(df) >= MIN_ROWS_IF:
        if_rows, if_cols = _isolation_forest(df, cols, opts["if_contamination"])

    # 2) apply the chosen action
    remove: set = set()
    already = np.zeros(len(df), dtype=bool)
    for c, mask, lo, hi in found:
        already |= mask
        for pos in np.flatnonzero(mask):
            idx = df.index[pos]
            old = df.at[idx, c]
            rng = f"allowed range {lo:.6g} to {hi:.6g}"
            if action == "cap":
                new = hi if float(old) > hi else lo
                new = int(round(new)) if pdt.is_integer_dtype(df[c]) else float(new)
                df.at[idx, c] = new
                log.add("outlier_capped", c, idx, old=old, new=new, detail=rng)
            elif action == "remove":
                remove.add(idx)
                log.add("outlier_removed", c, idx, old=old, new=None, detail=rng)
            else:
                log.add("outlier_flagged", c, idx, old=old, new=None, detail=rng)
                log.flag(idx, f"outlier:{c}")

    for pos in np.flatnonzero(if_rows & ~already):
        idx = df.index[pos]
        detail = f"unusual combination of values (most unusual: {if_cols[pos]})"
        log.add("outlier_multivariate", None, idx, detail=detail)
        if action == "remove":
            remove.add(idx)
        else:
            log.flag(idx, "unusual_row")

    if remove:
        df = df.drop(index=list(remove))
    return df
