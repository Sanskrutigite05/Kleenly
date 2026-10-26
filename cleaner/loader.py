"""Read a messy CSV: detect encoding + delimiter, keep everything as text."""
from __future__ import annotations

import io

import pandas as pd
from charset_normalizer import from_bytes

from .errors import CleanerError

_DELIMITERS = [",", ";", "\t", "|"]


def _detect_encoding(raw: bytes) -> str:
    if raw.startswith(b"\xef\xbb\xbf"):
        return "utf-8-sig"
    try:
        raw.decode("utf-8")
        return "utf-8"
    except UnicodeDecodeError:
        best = from_bytes(raw[:500_000]).best()
        return best.encoding if best else "latin-1"


def _detect_delimiter(text: str) -> str:
    first = next((ln for ln in text.splitlines() if ln.strip()), "")
    counts = {d: first.count(d) for d in _DELIMITERS}
    best = max(counts, key=counts.get)
    return best if counts[best] > 0 else ","


def _clean_headers(cols) -> tuple[list[str], list[str]]:
    seen: dict[str, int] = {}
    out, notes = [], []
    for i, c in enumerate(cols):
        name = str(c).strip()
        if not name or name.lower().startswith("unnamed:"):
            name = f"unnamed_{i + 1}"
            notes.append(f"Column {i + 1} had no header and was named '{name}'.")
        if name in seen:
            seen[name] += 1
            new = f"{name}_{seen[name]}"
            notes.append(f"Duplicate column name '{name}' renamed to '{new}'.")
            name = new
        else:
            seen[name] = 1
        out.append(name)
    return out, notes


def load_csv(raw: bytes) -> tuple[pd.DataFrame, dict]:
    if not raw or not raw.strip():
        raise CleanerError("The file is empty.")

    encoding = _detect_encoding(raw)
    text = raw.decode(encoding, errors="replace").lstrip("﻿")
    delimiter = _detect_delimiter(text)

    skipped: list[int] = []

    def _bad_line(line):  # rows with too many fields are skipped and counted
        skipped.append(len(line))
        return None

    try:
        df = pd.read_csv(
            io.StringIO(text),
            sep=delimiter,
            dtype=str,
            keep_default_na=False,
            skipinitialspace=True,
            engine="python",
            on_bad_lines=_bad_line,
            header=None,  # we read the header ourselves so duplicate names are handled by us
        )
    except (pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
        raise CleanerError(f"Could not read this file as CSV: {exc}") from exc

    if df.shape[1] == 0 or len(df) < 2:
        raise CleanerError("The file has a header but no data rows.")

    header = ["" if pd.isna(c) else c for c in df.iloc[0].tolist()]
    df = df.iloc[1:].reset_index(drop=True)
    df.columns, notes = _clean_headers(header)
    meta = {
        "encoding": encoding,
        "delimiter": {"\t": "tab"}.get(delimiter, delimiter),
        "malformed_lines_skipped": len(skipped),
        "notes": notes,
    }
    return df, meta
