"""Small helpers shared by the cleaning modules."""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
from pandas.api import types as pdt

# Column names that look like identifiers (never impute / outlier-check these).
_ID_NAME = re.compile(
    r"(^|[_\-\s])(id|uuid|guid|index|sr|sno|srno|no|number)$|^(id|uuid)([_\-\s]|$)",
    re.IGNORECASE,
)


def is_text(s: pd.Series) -> bool:
    return pdt.is_object_dtype(s) or pdt.is_string_dtype(s)


def is_numeric(s: pd.Series) -> bool:
    return pdt.is_numeric_dtype(s) and not pdt.is_bool_dtype(s)


def to_float(s: pd.Series) -> np.ndarray:
    """Numeric series (incl. nullable Int64) -> float64 array with NaN for missing."""
    return (
        pd.to_numeric(s, errors="coerce")
        .astype("Float64")
        .to_numpy(dtype="float64", na_value=np.nan)
    )


def is_id_like(s: pd.Series, name) -> bool:
    if _ID_NAME.search(str(name).strip()):
        return True
    if pdt.is_integer_dtype(s):
        # a running counter 1, 2, 3, ... (no gaps) is an identifier, not a measurement
        v = s.dropna().astype("int64")
        if len(v) >= 10 and v.is_unique and v.is_monotonic_increasing and (v.diff().dropna() == 1).all():
            return True
    return False


def py(v):
    """numpy / pandas scalar -> plain Python value (None for missing)."""
    if v is None:
        return None
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, pd.Timestamp):
        return str(v)
    return v


def disp(v):
    """Value -> short string for the change log (None if missing)."""
    v = py(v)
    if v is None:
        return None
    if isinstance(v, float):
        if v.is_integer() and abs(v) < 1e15:
            return str(int(v))
        return format(v, ".10g")
    return str(v)


def type_name(s: pd.Series) -> str:
    if pdt.is_bool_dtype(s):
        return "boolean"
    if pdt.is_integer_dtype(s):
        return "integer"
    if pdt.is_float_dtype(s):
        return "decimal"
    if pdt.is_datetime64_any_dtype(s):
        return "date"
    return "text"


def date_format_for(s: pd.Series) -> str:
    d = s.dropna()
    if len(d) and (d != d.dt.normalize()).any():
        return "%Y-%m-%d %H:%M:%S"
    return "%Y-%m-%d"


def format_dates(s: pd.Series) -> pd.Series:
    """datetime series -> ISO strings (NaN for NaT)."""
    return s.dt.strftime(date_format_for(s))
