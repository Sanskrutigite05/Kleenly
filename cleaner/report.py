"""Change log + labels used to build the cleaning report."""
from __future__ import annotations

import csv
import io
from collections import defaultdict

from .utils import disp

MAX_STORED = 100_000  # hard cap on stored change entries (memory safety)

# category -> (label, kind). kind: fix | flag | info
CATEGORIES = {
    "whitespace_trimmed": ("Whitespace trimmed", "fix"),
    "fake_null_normalized": ("Fake nulls (N/A, -, null...) turned into real missing values", "fix"),
    "column_dropped": ("Mostly-empty columns dropped", "fix"),
    "row_dropped_sparse": ("Mostly-empty rows dropped", "fix"),
    "type_converted": ("Columns converted to the right data type", "fix"),
    "format_standardized": ("Values reformatted (currency, commas, dates...)", "fix"),
    "invalid_value": ("Invalid values for the column type, set to missing", "fix"),
    "category_standardized": ("Inconsistent category spellings unified", "fix"),
    "duplicate_exact": ("Exact duplicate rows removed", "fix"),
    "near_duplicate_removed": ("Near-duplicate rows removed", "fix"),
    "near_duplicate_flagged": ("Possible near-duplicate rows flagged", "flag"),
    "outlier_flagged": ("Outliers flagged", "flag"),
    "outlier_capped": ("Outliers capped to the allowed range", "fix"),
    "outlier_removed": ("Outlier rows removed", "fix"),
    "outlier_multivariate": ("Unusual rows found by Isolation Forest", "flag"),
    "missing_filled": ("Missing values filled", "fix"),
    "missing_left_unfilled": ("Missing values left as they are", "info"),
}
CATEGORY_ORDER = list(CATEGORIES)


def label(category: str) -> str:
    return CATEGORIES.get(category, (category, "info"))[0]


def kind(category: str) -> str:
    return CATEGORIES.get(category, (category, "info"))[1]


class ChangeLog:
    """Records every change so nothing happens silently."""

    def __init__(self) -> None:
        self.entries: list[dict] = []
        self.counts: dict[str, int] = defaultdict(int)
        self.by_column: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.flags: dict[int, list[str]] = defaultdict(list)
        self.notes: list[str] = []

    # `index` is the 0-based data-row label; shown as the spreadsheet row (header = row 1).
    @staticmethod
    def _row(index):
        try:
            return int(index) + 2
        except (TypeError, ValueError):
            return None

    def add(self, category, column=None, index=None, old=None, new=None, detail=""):
        self.counts[category] += 1
        if column is not None:
            self.by_column[column][category] += 1
        if len(self.entries) < MAX_STORED:
            self.entries.append(
                {
                    "category": category,
                    "row": self._row(index) if index is not None else None,
                    "column": column,
                    "old": disp(old),
                    "new": disp(new),
                    "detail": detail,
                }
            )

    def add_many(self, category, n, column=None):
        """Count-only entry for bulk changes (e.g. whitespace trimming)."""
        if n <= 0:
            return
        self.counts[category] += n
        if column is not None:
            self.by_column[column][category] += n

    def flag(self, index, text):
        self.flags[index].append(text)

    def note(self, text):
        if text not in self.notes:
            self.notes.append(text)

    def sample(self, per_category=40):
        taken: dict[str, int] = defaultdict(int)
        out = []
        for cat in CATEGORY_ORDER:
            for e in self.entries:
                if e["category"] == cat and taken[cat] < per_category:
                    taken[cat] += 1
                    out.append({**e, "label": label(cat)})
        return out

    def to_csv_bytes(self) -> bytes:
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(["category", "description", "row", "column", "old_value", "new_value", "detail"])
        for e in self.entries:
            w.writerow(
                [
                    e["category"],
                    label(e["category"]),
                    e["row"] if e["row"] is not None else "",
                    e["column"] or "",
                    e["old"] if e["old"] is not None else "",
                    e["new"] if e["new"] is not None else "",
                    e["detail"],
                ]
            )
        return buf.getvalue().encode("utf-8")
