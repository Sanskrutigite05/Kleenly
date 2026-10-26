"""Exact and near-duplicate detection."""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process

from .utils import is_id_like

FUZZY_MAX_ROWS = 4000  # pairwise comparison is O(n^2)
MIN_KEY_LEN = 12


def _show(v) -> str:
    if isinstance(v, pd.Timestamp):
        return v.strftime("%Y-%m-%d") if v == v.normalize() else str(v)
    return str(v)


def _row_text(row_values) -> str:
    """Readable version of a row, used in the report."""
    return " | ".join(_show(v) for v in row_values if not pd.isna(v))


def _row_key(row_values) -> str:
    """Normalised text used for the similarity comparison."""
    text = " ".join(_show(v) for v in row_values if not pd.isna(v)).lower()
    return re.sub(r"\s+", " ", re.sub(r"[^\w@.\s]", "", text)).strip()


def handle_duplicates(df: pd.DataFrame, log, opts):
    near_pairs: list[dict] = []

    # ---- exact duplicates (all columns identical) ----
    if opts["drop_exact_duplicates"] and len(df) > 1:
        keys = pd.util.hash_pandas_object(df.astype(str), index=False)
        keys.index = df.index
        dup = keys.duplicated(keep="first")
        if dup.any():
            first_of = {}
            for idx, k in zip(df.index[~dup], keys[~dup]):
                first_of[k] = idx
            for idx in df.index[dup]:
                log.add("duplicate_exact", None, idx, detail=f"identical to row {first_of[keys[idx]] + 2}")
            df = df.loc[~dup]

    # ---- near duplicates (same record, different ID / small typos) ----
    mode = opts["fuzzy_duplicates"]
    if mode != "off" and len(df) > 1:
        if len(df) > FUZZY_MAX_ROWS:
            log.note(f"Near-duplicate check skipped: files above {FUZZY_MAX_ROWS} rows are not compared pairwise.")
            return df, near_pairs
        cols = [c for c in df.columns if not is_id_like(df[c], c)]
        if len(cols) >= 2:
            rows = list(df[cols].itertuples(index=False, name=None))
            keys = [_row_key(r) for r in rows]
            scores = process.cdist(
                keys, keys, scorer=fuzz.ratio, score_cutoff=opts["fuzzy_threshold"],
                dtype=np.uint8, workers=-1,
            )
            ii, jj = np.nonzero(np.triu(scores, k=1))
            labels = df.index.to_list()
            drop = set()
            for i, j in zip(ii, jj):
                if len(keys[i]) < MIN_KEY_LEN or len(keys[j]) < MIN_KEY_LEN:
                    continue
                a, b, score = labels[i], labels[j], int(scores[i, j])
                near_pairs.append({"row_a": a + 2, "row_b": b + 2, "similarity": score, "a": _row_text(rows[i])[:140], "b": _row_text(rows[j])[:140]})
                if mode == "remove":
                    if b not in drop:
                        drop.add(b)
                        log.add("near_duplicate_removed", None, b, detail=f"{score}% similar to row {a + 2}")
                else:
                    log.add("near_duplicate_flagged", None, b, detail=f"{score}% similar to row {a + 2}")
                    log.flag(b, f"possible_duplicate_of_row_{a + 2}")
            if drop:
                df = df.drop(index=list(drop))
    return df, near_pairs
