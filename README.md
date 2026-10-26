# Kleenly

*A CSV cleaner.*

Upload a messy CSV. Get back a cleaned file and a record of every change that was made.

It finds missing values, duplicates, wrong data types and outliers. Almost everything is plain,
explainable rules. Two small ML checks (Isolation Forest and KNN imputation) are optional extras.

## Run it

```bash
pip install -r requirements.txt
uvicorn app:app --reload
```

Open http://127.0.0.1:8000, click **Use the sample file instead**, then **Clean file**.

Without the browser:

```bash
python -m cleaner sample_data/messy_customers.csv -o cleaned.csv --changes changes.csv --report report.json
python -m pytest
```

## Tech stack

| Part | Choice | Why |
|---|---|---|
| Language | Python 3.11+ | Standard for data work |
| Data handling | pandas, NumPy | Loading, cleaning, transformations |
| ML checks | scikit-learn | `IsolationForest` for outliers, `KNNImputer` for filling gaps |
| Near-duplicates | rapidfuzz | Fast fuzzy text similarity |
| Encoding detection | charset-normalizer | Messy files often aren't UTF-8 |
| API | FastAPI + uvicorn | Small, typed, easy to test |
| UI | HTML, CSS, JavaScript (no framework) | Served by FastAPI from `static/` |
| Tests | pytest | One test per cleaning rule |

## How it works

```
upload -> read -> fix text -> drop empty -> convert types -> unify spellings
       -> duplicates -> outliers -> fill missing -> cleaned CSV + change log + report
```

Each step lives in its own file in `cleaner/` and writes to one shared change log, so nothing
happens silently.

| Step | File | What it does |
|---|---|---|
| Read | `loader.py` | Detects encoding and delimiter, skips malformed lines, reads every value as text |
| Missing values | `missing.py` | `N/A`, `-`, `null`, `?` become real blanks. Sparse columns/rows are dropped. Numbers filled with median, mean or KNN. Text filled with the most common value, or `Unknown` when values are mostly unique |
| Data types | `dtypes.py` | Text columns that are really numbers (`₹1,20,000`, `(500)`, `12%`, `1.5k`), dates (mixed formats, day-first) or yes/no answers are converted. Values that don't fit are reported. Phone numbers and zip codes stay text |
| Spelling | `dtypes.py` | `mumbai`, `MUMBAI`, `Mumbai` become one spelling (low-variety text columns only) |
| Duplicates | `duplicates.py` | Exact duplicates are removed. Near-duplicates (same person, new ID, small typo) are found with fuzzy matching and flagged or removed |
| Outliers | `outliers.py` | IQR rule or z-score per column, plus Isolation Forest for rows that are odd across several columns. Flag, cap or remove |
| Report | `report.py`, `pipeline.py` | Counts per kind of change, per-column summary, row-level log |

### Decisions worth explaining in a viva

- **Outliers are flagged, not deleted, by default.** A value of 999 in an age column is probably
  a mistake, but a salary of 9,999,999 might be real. The tool shows it and lets you decide.
  Flagged rows get a note in an extra `_flags` column.
- **Near-duplicates are flagged, not deleted, by default.** Two people can differ by one letter.
- **Identifier columns are protected.** Columns like `customer_id` are never imputed or checked
  for outliers.
- **IQR over a neural model.** It needs no training, works on small files and can be explained in
  one sentence. Isolation Forest is added only for combinations of columns, which IQR can't see.
- **Every change is logged** with its row number (matching your spreadsheet), old value, new value
  and reason. Cleaning an already-cleaned file changes nothing (there is a test for this).

## Project layout

```
app.py                 FastAPI app: /api/clean, downloads, serves static/
cleaner/               The cleaning engine (usable without the web app)
static/                index.html, style.css, app.js
sample_data/           messy_customers.csv with every kind of problem
tests/                 test_cleaner.py (rules), test_api.py (endpoints)
```

## Limits

- Upload limit is 20 MB. Results are kept in memory for the last 25 files.
- Near-duplicate matching compares rows pairwise, so it is skipped above 4,000 rows.
- Day-first dates (`05/06/2023` = 5 June) are the default. Turn it off for US-style dates.
- `1.200` is read as the decimal 1.2, not twelve hundred. European-style thousands separators are not detected.
- Row numbers in the log assume the header is row 1. Malformed lines that were skipped can shift them.
