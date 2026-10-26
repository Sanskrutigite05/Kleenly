"""Command line use:  python -m cleaner messy.csv -o cleaned.csv --report report.json"""
import argparse
import json
import sys
from pathlib import Path

from . import CleanerError, clean_bytes


def main() -> int:
    p = argparse.ArgumentParser(prog="python -m cleaner", description="Clean a messy CSV file.")
    p.add_argument("input", help="path to the CSV file")
    p.add_argument("-o", "--output", help="cleaned CSV path (default: <input>_cleaned.csv)")
    p.add_argument("--report", help="write the full JSON report here")
    p.add_argument("--changes", help="write the row-level change log CSV here")
    p.add_argument("--outliers", choices=["flag", "cap", "remove"], default="flag")
    p.add_argument("--fuzzy", choices=["off", "flag", "remove"], default="flag")
    p.add_argument("--numeric-impute", choices=["median", "mean", "knn", "none"], default="median")
    args = p.parse_args()

    src = Path(args.input)
    try:
        res = clean_bytes(
            src.read_bytes(),
            {"outlier_action": args.outliers, "fuzzy_duplicates": args.fuzzy, "numeric_impute": args.numeric_impute},
            filename=src.name,
        )
    except (CleanerError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    out = Path(args.output) if args.output else src.with_name(f"{src.stem}_cleaned.csv")
    out.write_bytes(res.csv_bytes)
    if args.report:
        Path(args.report).write_text(json.dumps(res.report, indent=2, default=str), encoding="utf-8")
    if args.changes:
        Path(args.changes).write_bytes(res.changes_csv_bytes)

    t = res.report["totals"]
    print(f"Cleaned {t['rows_before']} -> {t['rows_after']} rows, {t['total_changes']} changes. Saved to {out}")
    for item in res.report["summary"]:
        print(f"  {item['count']:>5}  {item['label']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
