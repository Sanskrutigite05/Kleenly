"""Web app: FastAPI serves the cleaning API and the plain HTML/CSS/JS front end.

Run:  uvicorn app:app --reload      then open http://127.0.0.1:8000
"""
from __future__ import annotations

import json
import re
from collections import OrderedDict
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from cleaner import CleanerError, clean_bytes

BASE = Path(__file__).parent
MAX_BYTES = 20 * 1024 * 1024  # 20 MB upload limit
MAX_JOBS = 25                 # results kept in memory (oldest dropped first)

app = FastAPI(title="Kleenly")
app.mount("/static", StaticFiles(directory=BASE / "static"), name="static")

JOBS: "OrderedDict[str, dict]" = OrderedDict()


def _json_default(o):
    if isinstance(o, np.generic):
        return o.item()
    return str(o)


def _safe_stem(filename: str) -> str:
    stem = Path(filename or "data").stem
    return re.sub(r"[^A-Za-z0-9._-]", "_", stem)[:60] or "data"


def _job(job_id: str) -> dict:
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, "This result is no longer available. Upload the file again.")
    return job


@app.get("/")
def index():
    return FileResponse(BASE / "static" / "index.html")


@app.get("/api/sample")
def sample():
    return FileResponse(BASE / "sample_data" / "messy_customers.csv", media_type="text/csv", filename="messy_customers.csv")


@app.post("/api/clean")
async def clean(file: UploadFile = File(...), options: str = Form("{}")):
    raw = await file.read(MAX_BYTES + 1)
    if len(raw) > MAX_BYTES:
        raise HTTPException(413, f"File is larger than {MAX_BYTES // (1024 * 1024)} MB.")
    if not raw.strip():
        raise HTTPException(400, "The file is empty.")
    if b"\x00" in raw[:4096]:
        raise HTTPException(400, "This looks like a binary file (for example .xlsx). Export it as CSV first.")
    try:
        opts = json.loads(options or "{}")
        if not isinstance(opts, dict):
            raise ValueError
    except ValueError:
        raise HTTPException(400, "Options must be a JSON object.") from None

    try:
        result = await run_in_threadpool(clean_bytes, raw, opts, file.filename or "data.csv")
    except CleanerError as exc:
        raise HTTPException(422, str(exc)) from exc

    job_id = __import__("uuid").uuid4().hex[:12]
    JOBS[job_id] = {
        "stem": _safe_stem(file.filename),
        "csv": result.csv_bytes,
        "changes": result.changes_csv_bytes,
        "report": result.report,
    }
    while len(JOBS) > MAX_JOBS:
        JOBS.popitem(last=False)

    body = json.dumps({"job_id": job_id, "report": result.report}, default=_json_default)
    return Response(body, media_type="application/json")


def _download(data: bytes, name: str, media: str) -> Response:
    return Response(data, media_type=media, headers={"Content-Disposition": f'attachment; filename="{name}"'})


@app.get("/api/download/{job_id}")
def download_cleaned(job_id: str):
    j = _job(job_id)
    return _download(j["csv"], f"{j['stem']}_cleaned.csv", "text/csv; charset=utf-8")


@app.get("/api/changes/{job_id}")
def download_changes(job_id: str):
    j = _job(job_id)
    return _download(j["changes"], f"{j['stem']}_changes.csv", "text/csv; charset=utf-8")


@app.get("/api/report/{job_id}")
def download_report(job_id: str):
    j = _job(job_id)
    data = json.dumps(j["report"], indent=2, default=_json_default).encode("utf-8")
    return _download(data, f"{j['stem']}_report.json", "application/json")
