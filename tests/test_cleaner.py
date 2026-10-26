import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cleaner import CleanerError, clean_bytes
from cleaner.loader import load_csv

SAMPLE = Path(__file__).parent.parent / "sample_data" / "messy_customers.csv"


def run(text: str, **opts):
    return clean_bytes(text.encode("utf-8"), opts)


def counts(res) -> dict:
    return {s["category"]: s["count"] for s in res.report["summary"]}


def table(header: str, rows: list[str]) -> str:
    return header + "\n" + "\n".join(rows) + "\n"


# ------------------------------------------------------------------ loading
def test_loader_detects_semicolon_and_non_utf8_encoding():
    raw = "name;city\nJosé;Pune\nAna;Nashik\n".encode("latin-1")
    df, meta = load_csv(raw)
    assert meta["delimiter"] == ";"
    assert list(df.columns) == ["name", "city"]
    assert df.loc[0, "name"] == "José"


def test_loader_renames_blank_and_duplicate_headers():
    df, meta = load_csv(b"a,a,\n1,2,3\n")
    assert list(df.columns) == ["a", "a_2", "unnamed_3"]
    assert len(meta["notes"]) == 2


@pytest.mark.parametrize("raw", [b"", b"   \n", b"a,b\n"])
def test_empty_or_headerless_files_raise(raw):
    with pytest.raises(CleanerError):
        clean_bytes(raw)


def test_invalid_options_raise():
    with pytest.raises(CleanerError):
        clean_bytes(b"a,b\n1,2\n", {"outlier_action": "explode"})
    with pytest.raises(CleanerError):
        clean_bytes(b"a,b\n1,2\n", {"iqr_k": 99})


# ------------------------------------------------------------ missing values
def test_fake_nulls_become_real_missing():
    scores = ["10", "N/A", "-", "null", "20", "?", "30", "40", "50", "60"]
    res = run(table("id,score", [f"{i},{v}" for i, v in enumerate(scores, 1)]), numeric_impute="none")
    assert res.df["score"].isna().sum() == 4
    assert counts(res)["fake_null_normalized"] == 4


def test_median_fill_for_numbers():
    res = run(table("id,score", ["1,10", "2,20", "3,30", "4,N/A", "5,40"]))
    assert res.df.loc[3, "score"] == 25  # median of 10,20,30,40
    assert counts(res)["missing_filled"] == 1


def test_knn_fill_uses_similar_rows():
    a_values = [10, 12, 15, 17, 20, 22, 25, 27, 30, 33, 35, 38]
    rows = [f"{a},{2 * a}" if a != 22 else f"{a},N/A" for a in a_values]
    res = run(table("a,b", rows), numeric_impute="knn", outlier_method="none", isolation_forest=False)
    assert 38 <= res.df.loc[5, "b"] <= 50  # true value is 44


def test_text_fill_uses_mode_or_unknown_for_unique_values():
    rows = ["1,Pune,a@x.com", "2,Pune,b@x.com", "3,Pune,c@x.com", "4,,d@x.com", "5,Nashik,"]
    res = run(table("id,city,email", rows))
    assert res.df.loc[3, "city"] == "Pune"          # most common
    assert res.df.loc[4, "email"] == "Unknown"       # unique values: no reliable mode


def test_mostly_empty_column_and_row_are_dropped():
    rows = [f"{i},{i * 3},{i * 7},{'x' if i < 3 else ''}" for i in range(1, 11)] + ["11,,,"]
    res = run(table("id,val,other,junk", rows))
    assert "junk" not in res.df.columns
    assert len(res.df) == 10  # the row with only an id is dropped
    c = counts(res)
    assert c["column_dropped"] == 1 and c["row_dropped_sparse"] == 1


# ---------------------------------------------------------------- data types
def test_currency_percent_suffix_and_brackets_become_numbers():
    vals = ["₹1,20,000", "₹55,000", "Rs. 900", "1.5k", "(500)", "42", "7"]
    res = run(table("id,price", [f"{i},\"{v}\"" for i, v in enumerate(vals, 1)]))
    assert res.df["price"].tolist() == [120000, 55000, 900, 1500, -500, 42, 7]
    assert str(res.df["price"].dtype) == "Int64"


def test_values_that_do_not_fit_the_type_are_reported():
    ages = ["21", "34", "29", "41", "abc", "37", "25", "30", "28", "33"]
    res = run(table("id,age", [f"{i},{v}" for i, v in enumerate(ages, 1)]))
    assert counts(res)["invalid_value"] == 1
    assert res.df["age"].notna().all()  # the bad one was filled with the median


def test_leading_zero_codes_stay_text():
    res = run(table("id,zip", [f"{i},0{1000 + i}" for i in range(1, 11)]))
    assert res.df["zip"].iloc[0] == "01001"


def test_yes_no_columns_become_boolean():
    res = run(table("id,active", ["1,Yes", "2,no", "3,TRUE", "4,Y", "5,false"]))
    assert res.df["active"].tolist() == [True, False, True, True, False]


def test_mixed_date_formats_are_standardised_day_first():
    dates = ["15/02/2023", "2023-03-10", "10 Mar 2023", "05/06/2023", "March 22 2023"]
    text = table("when,label", [f"{d},{c}" for d, c in zip(dates, "abcde")])
    res = run(text)
    assert res.df["when"].tolist() == ["2023-02-15", "2023-03-10", "2023-03-10", "2023-06-05", "2023-03-22"]
    res = run(text, dayfirst=False)
    assert res.df["when"].iloc[3] == "2023-05-06"


def test_category_spellings_are_unified():
    cities = ["Mumbai", "mumbai", "MUMBAI", "Pune", "pune", "Pune", "Nashik", "Nashik"]
    res = run(table("id,city", [f"{i},{c}" for i, c in enumerate(cities, 1)]))
    assert set(res.df["city"]) == {"Mumbai", "Pune", "Nashik"}
    assert counts(res)["category_standardized"] == 3


# ---------------------------------------------------------------- duplicates
def test_exact_duplicates_are_removed():
    res = run(table("name,city", ["Asha Rao,Pune", "Dev Patil,Nashik", "Asha Rao,Pune"]))
    assert len(res.df) == 2 and counts(res)["duplicate_exact"] == 1


NEAR = table("id,name,city,email", [
    "1,Priya Patil,Pune,priya.patil@example.com",
    "2,Priya Patel,Pune,priya.patil@example.com",
    "3,Rohan Deshmukh,Nashik,rohan.d@example.com",
])


def test_near_duplicates_are_flagged_by_default():
    res = run(NEAR)
    assert len(res.df) == 3
    assert len(res.report["near_duplicate_pairs"]) == 1
    assert res.df.loc[1, "_flags"] == "possible_duplicate_of_row_2"


def test_near_duplicates_can_be_removed_or_ignored():
    assert len(run(NEAR, fuzzy_duplicates="remove").df) == 2
    assert len(run(NEAR, fuzzy_duplicates="off").df) == 3


# ------------------------------------------------------------------ outliers
OUT = table("name,v", [f"a{i},{50 + i}" for i in range(20)] + ["z,999"])


def test_iqr_outlier_is_flagged_but_kept_by_default():
    res = run(OUT)
    assert len(res.df) == 21 and res.df["v"].max() == 999
    assert res.df.loc[20, "_flags"] == "outlier:v"


def test_outlier_cap_and_remove():
    assert run(OUT, outlier_action="cap").df["v"].max() < 999
    assert len(run(OUT, outlier_action="remove").df) == 20


def test_id_columns_are_never_outlier_checked():
    rows = [f"{i},{50 + i}" for i in range(1, 30)] + ["5000,70"]
    res = run(table("customer_id,score", rows), isolation_forest=False)
    assert "outlier_flagged" not in counts(res)


def test_isolation_forest_finds_unusual_combinations():
    # age and salary each look normal on their own, but not together
    rng = np.random.default_rng(0)
    ages = rng.integers(22, 60, 60)
    rows = [f"{a},{int(a * 2000 + rng.normal(0, 1500))}" for a in ages] + ["22,118000"]
    res = run(table("age,salary", rows), outlier_method="none", if_contamination=0.03)
    assert counts(res).get("outlier_multivariate", 0) >= 1
    assert res.df["_flags"].iloc[-1] == "unusual_row"


# ---------------------------------------------------------------- whole file
def test_sample_file_end_to_end():
    res = clean_bytes(SAMPLE.read_bytes(), filename="messy_customers.csv")
    df, r = res.df, res.report
    assert (r["totals"]["rows_before"], r["totals"]["rows_after"]) == (34, 32)
    assert sorted(df["city"].unique()) == ["Mumbai", "Nagpur", "Nashik", "Pune"]
    assert df["salary"].notna().all() and df["is_active"].notna().all()
    assert df["signup_date"].isna().sum() == 1            # 32/13/2023 can't be a date
    assert df.loc[df["customer_id"] == 7, "_flags"].iloc[0] == "outlier:age"
    assert r["completeness"]["after"] > r["completeness"]["before"]


def test_report_is_plain_json_and_change_log_has_header():
    res = clean_bytes(SAMPLE.read_bytes())
    json.dumps(res.report)  # no numpy types leaking
    assert res.changes_csv_bytes.decode().startswith("category,description,row,column,old_value,new_value,detail")


def test_cleaning_twice_changes_nothing_more():
    first = clean_bytes(SAMPLE.read_bytes(), {"add_flag_column": False})
    second = clean_bytes(first.csv_bytes, {"add_flag_column": False})
    pd.testing.assert_frame_equal(first.df.reset_index(drop=True), second.df.reset_index(drop=True), check_dtype=False)
