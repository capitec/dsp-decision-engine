from __future__ import annotations

import json

import polars as pl
import pytest

from decider.data import LoadError, load
from decider.lifecycle import JobHandle, JobStatus


def _loans() -> pl.DataFrame:
    return pl.DataFrame({"client_id": ["C-1", "C-2"], "income": [1000.0, 2000.0]})


def test_load_csv_surfaces_schema_row_count_and_suggested_key(tmp_path):
    path = tmp_path / "loans.csv"
    _loans().write_csv(path)
    data = load(path)
    assert data.row_count == 2
    assert data.columns == (("client_id", "String"), ("income", "Float64"))
    assert data.suggested_key == ("client_id",)
    assert data.frame["client_id"].to_list() == ["C-1", "C-2"]
    assert data.dataset == "loans.csv"
    assert data.fingerprint


def test_load_parquet(tmp_path):
    path = tmp_path / "loans.parquet"
    _loans().write_parquet(path)
    data = load(path)
    assert data.row_count == 2
    assert data.suggested_key == ("client_id",)


def test_load_json_accepts_ndjson_and_array(tmp_path):
    ndjson = tmp_path / "loans.jsonl"
    _loans().write_ndjson(ndjson)
    assert load(ndjson).row_count == 2

    array = tmp_path / "loans.json"
    array.write_text(json.dumps([{"client_id": "C-1", "income": 1000.0}]))
    assert load(array).row_count == 1


def test_load_bytes_requires_an_explicit_format(tmp_path):
    raw = b"client_id,income\nC-1,1000.0\n"
    assert load(raw, format="csv").row_count == 1
    with pytest.raises(LoadError):
        load(raw)


def test_load_unknown_extension_requires_format(tmp_path):
    path = tmp_path / "loans.data"
    path.write_text("whatever")
    with pytest.raises(LoadError):
        load(path)


def test_load_failure_is_a_load_error(tmp_path):
    path = tmp_path / "loans.parquet"
    path.write_text("this is not parquet")
    with pytest.raises(LoadError):
        load(path)


def test_load_accepts_an_already_loaded_frame():
    data = load(_loans(), format="csv")
    assert data.row_count == 2


def test_load_reports_progress_and_result_through_a_job(tmp_path):
    path = tmp_path / "loans.csv"
    _loans().write_csv(path)
    handle = JobHandle(job_id="j1", kind="data_load")
    data = load(path, job=handle)
    snap = handle.snapshot()
    assert snap.status is JobStatus.SUCCEEDED
    assert snap.total == 3
    assert snap.done == 2
    assert snap.result.row_count == 2
    assert data.row_count == 2


def test_load_aborts_when_the_job_is_cancelled(tmp_path):
    path = tmp_path / "loans.csv"
    _loans().write_csv(path)
    handle = JobHandle(job_id="j1", kind="data_load")
    handle.start()
    handle.cancel()
    with pytest.raises(LoadError):
        load(path, job=handle)


def test_load_failure_is_recorded_on_the_job(tmp_path):
    path = tmp_path / "loans.parquet"
    path.write_text("not parquet")
    handle = JobHandle(job_id="j1", kind="data_load")
    with pytest.raises(LoadError):
        load(path, job=handle)
    assert handle.snapshot().status is JobStatus.FAILED
