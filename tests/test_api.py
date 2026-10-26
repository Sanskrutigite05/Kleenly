from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import app as app_module
from app import app

SAMPLE = Path(__file__).parent.parent / "sample_data" / "messy_customers.csv"
client = TestClient(app)


def upload(content: bytes, options: str = "{}", name: str = "x.csv"):
    return client.post("/api/clean", files={"file": (name, content, "text/csv")}, data={"options": options})


def test_home_page_and_assets_load():
    assert client.get("/").status_code == 200
    assert client.get("/static/app.js").status_code == 200
    assert client.get("/static/style.css").status_code == 200
    assert client.get("/api/sample").status_code == 200


def test_clean_then_download_all_three_files():
    res = upload(SAMPLE.read_bytes(), '{"outlier_action": "cap"}', "messy_customers.csv")
    assert res.status_code == 200
    body = res.json()
    assert body["report"]["totals"]["rows_after"] == 32

    cleaned = client.get(f"/api/download/{body['job_id']}")
    assert cleaned.status_code == 200
    assert cleaned.text.startswith("customer_id,name,age")
    assert "messy_customers_cleaned.csv" in cleaned.headers["content-disposition"]

    assert client.get(f"/api/changes/{body['job_id']}").text.startswith("category,description")
    assert client.get(f"/api/report/{body['job_id']}").json()["totals"]["rows_before"] == 34


@pytest.mark.parametrize(
    "content, options, status",
    [
        (b"", "{}", 400),
        (b"PK\x03\x04\x00\x00binary", "{}", 400),
        (b"a,b\n1,2\n", "not json", 400),
        (b"a,b\n1,2\n", '{"outlier_action": "explode"}', 422),
        (b"a,b\n", "{}", 422),
    ],
)
def test_bad_requests_get_clear_errors(content, options, status):
    res = upload(content, options)
    assert res.status_code == status
    assert res.json()["detail"]


def test_oversized_upload_is_rejected(monkeypatch):
    monkeypatch.setattr(app_module, "MAX_BYTES", 1000)  # tiny limit keeps the test fast
    res = upload(b"a,b\n" + b"1,2\n" * 500)
    assert res.status_code == 413
    assert res.json()["detail"]


def test_unknown_job_is_404():
    assert client.get("/api/download/doesnotexist").status_code == 404
