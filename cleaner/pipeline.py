"""The cleaning pipeline: load -> profile -> fix -> report."""
from __future__ import annotations

import io
from dataclasses import dataclass

import pandas as pd
from pandas.api import types as pdt

from . import dtypes, duplicates, missing, outliers
from .errors import CleanerError
from .loader import load_csv
from .report import CATEGORIES, CATEGORY_ORDER, ChangeLog, kind, label
from .utils import format_dates, py, type_name

DEFAULT_OPTIONS = {
    "extra_null_tokens": [],
    "col_missing_threshold": 0.6,   # drop columns missing more than this
    "row_missing_threshold": 0.5,   # drop rows missing more than this
    "numeric_impute": "median",     # median | mean | knn | none
    "categorical_impute": "mode",   # mode | unknown | none
    "type_threshold": 0.9,          # share of values that must parse to convert a column
    "dayfirst": True,               # 05/06/2023 = 5 June (Indian / UK style)
    "standardize_categories": True,
    "drop_exact_duplicates": True,
    "fuzzy_duplicates": "flag",     # off | flag | remove
    "fuzzy_threshold": 95,
    "outlier_method": "iqr",        # iqr | zscore | none
    "iqr_k": 1.5,
    "zscore_threshold": 3.0,
    "isolation_forest": True,
    "if_contamination": 0.03,
    "outlier_action": "flag",       # flag | cap | remove
    "add_flag_column": True,        # adds a `_flags` column explaining flagged rows
}

_ENUMS = {
    "numeric_impute": {"median", "mean", "knn", "none"},
    "categorical_impute": {"mode", "unknown", "none"},
    "fuzzy_duplicates": {"off", "flag", "remove"},
    "outlier_method": {"iqr", "zscore", "none"},
    "outlier_action": {"flag", "cap", "remove"},
}
_BOOLS = ["dayfirst", "standardize_categories", "drop_exact_duplicates", "isolation_forest", "add_flag_column"]
_FLOATS = {
    "col_missing_threshold": (0.0, 1.0),
    "row_missing_threshold": (0.0, 1.0),
    "type_threshold": (0.5, 1.0),
    "iqr_k": (0.5, 10.0),
    "zscore_threshold": (1.0, 10.0),
    "fuzzy_threshold": (50.0, 100.0),
    "if_contamination": (0.001, 0.5),
}


def normalize_options(options: dict | None) -> dict:
    opts = dict(DEFAULT_OPTIONS)
    for key, value in (options or {}).items():
        if key not in DEFAULT_OPTIONS or value is None:
            continue
        if key in _ENUMS:
            if value not in _ENUMS[key]:
                raise CleanerError(f"Invalid value '{value}' for '{key}'. Choose one of: {', '.join(sorted(_ENUMS[key]))}.")
            opts[key] = value
        elif key in _BOOLS:
            opts[key] = value if isinstance(value, bool) else str(value).lower() in ("1", "true", "yes", "on")
        elif key in _FLOATS:
            lo, hi = _FLOATS[key]
            try:
                f = float(value)
            except (TypeError, ValueError):
                raise CleanerError(f"'{key}' must be a number.") from None
            if not lo <= f <= hi:
                raise CleanerError(f"'{key}' must be between {lo:g} and {hi:g}.")
            opts[key] = f
        elif key == "extra_null_tokens":
            opts[key] = [str(t) for t in value][:50] if isinstance(value, (list, tuple)) else []
    return opts


@dataclass
class CleanResult:
    df: pd.DataFrame
    csv_bytes: bytes
    changes_csv_bytes: bytes
    report: dict


def clean_bytes(raw: bytes, options: dict | None = None, filename: str = "data.csv") -> CleanResult:
    opts = normalize_options(options)
    df, meta = load_csv(raw)
    meta["filename"] = filename
    return _run(df, meta, opts)


def clean_dataframe(df: pd.DataFrame, options: dict | None = None) -> CleanResult:
    """Clean an in-memory DataFrame (columns may be text or already typed)."""
    return _run(df.copy().reset_index(drop=True), {"filename": "dataframe", "notes": []}, normalize_options(options))


# ------------------------------------------------------------------ internals
def _preview(df: pd.DataFrame, n: int = 15) -> dict:
    head = df.head(n)
    return {"columns": [str(c) for c in head.columns], "rows": [[py(v) for v in row] for row in head.itertuples(index=False, name=None)]}


def _finalize(df: pd.DataFrame, log: ChangeLog, opts) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if pdt.is_datetime64_any_dtype(out[c]):
            out[c] = format_dates(out[c])
    if opts["add_flag_column"] and log.flags:
        flags = {i: "; ".join(v) for i, v in log.flags.items() if i in out.index}
        if flags:
            out["_flags"] = pd.Series(flags).reindex(out.index)
    return out.reset_index(drop=True)


def _run(df: pd.DataFrame, meta: dict, opts: dict) -> CleanResult:
    log = ChangeLog()
    rows_before, cols_before = df.shape
    preview_before = _preview(df)

    df = missing.normalize_nulls(df, log, opts["extra_null_tokens"])
    missing_before = df.isna().sum()
    cells_before = df.size

    df = missing.drop_sparse(df, log, opts)
    if df.empty or df.shape[1] == 0:
        raise CleanerError("Nothing usable is left after removing empty rows and columns.")

    df = dtypes.infer_types(df, log, opts)
    if opts["standardize_categories"]:
        df = dtypes.standardize_categories(df, log)
    df, near_pairs = duplicates.handle_duplicates(df, log, opts)
    df = outliers.handle_outliers(df, log, opts)
    if df.empty:
        raise CleanerError("Every row was removed. Try a less aggressive outlier action.")
    df = missing.impute(df, log, opts)

    types = {c: type_name(df[c]) for c in df.columns}
    out = _finalize(df, log, opts)
    data_cols = [c for c in out.columns if c != "_flags"]
    missing_after = out[data_cols].isna().sum()

    counts = log.counts
    cols_report = []
    for c in missing_before.index:
        kept = c in types
        cols_report.append(
            {
                "name": c,
                "type": types.get(c),
                "dropped": not kept,
                "missing_before": int(missing_before[c]),
                "missing_after": int(missing_after[c]) if kept else None,
                "changes": {label(k): v for k, v in log.by_column.get(c, {}).items() if k != "column_dropped"},
            }
        )

    summary = [{"category": k, "label": label(k), "kind": kind(k), "count": counts[k]} for k in CATEGORY_ORDER if counts.get(k)]
    comp_before = 1 - missing_before.sum() / cells_before if cells_before else 1
    comp_after = 1 - missing_after.sum() / out[data_cols].size if out[data_cols].size else 1

    report = {
        "file": {
            "name": meta.get("filename"),
            "encoding": meta.get("encoding"),
            "delimiter": meta.get("delimiter"),
            "malformed_lines_skipped": meta.get("malformed_lines_skipped", 0),
        },
        "totals": {
            "rows_before": rows_before,
            "rows_after": len(out),
            "cols_before": cols_before,
            "cols_after": len(data_cols),
            "flag_column_added": "_flags" in out.columns,
            "total_changes": int(sum(counts.values())),
        },
        "headline": {
            "missing_filled": counts.get("missing_filled", 0),
            "duplicates_removed": counts.get("duplicate_exact", 0) + counts.get("near_duplicate_removed", 0),
            "duplicates_flagged": counts.get("near_duplicate_flagged", 0),
            "type_fixes": counts.get("type_converted", 0) + counts.get("invalid_value", 0) + counts.get("format_standardized", 0),
            "outliers": sum(counts.get(k, 0) for k in ("outlier_flagged", "outlier_capped", "outlier_removed", "outlier_multivariate")),
        },
        "completeness": {"before": round(float(comp_before) * 100, 1), "after": round(float(comp_after) * 100, 1)},
        "summary": summary,
        "columns": cols_report,
        "near_duplicate_pairs": near_pairs[:100],
        "changes": log.sample(),
        "notes": meta.get("notes", []) + log.notes,
        "options": opts,
        "preview_before": preview_before,
        "preview_after": _preview(out),
    }
    buf = io.StringIO()
    out.to_csv(buf, index=False)
    return CleanResult(out, buf.getvalue().encode("utf-8"), log.to_csv_bytes(), report)
