from __future__ import annotations

import json

import polars as pl
import pytest

from decider.data import LoadError, read, scan


def _loans() -> pl.DataFrame:
    return pl.DataFrame({"client_id": ["C-1", "C-2"], "income": [1000.0, 2000.0]})


def test_read_csv_returns_a_frame(tmp_path):
    path = tmp_path / "loans.csv"
    _loans().write_csv(path)
    assert read(path).equals(_loans())


def test_read_parquet(tmp_path):
    path = tmp_path / "loans.parquet"
    _loans().write_parquet(path)
    assert read(path).equals(_loans())


def test_read_json_accepts_ndjson_and_array(tmp_path):
    ndjson = tmp_path / "loans.jsonl"
    _loans().write_ndjson(ndjson)
    assert read(ndjson).height == 2

    array = tmp_path / "loans.json"
    array.write_text(json.dumps([{"client_id": "C-1", "income": 1000.0}]))
    assert read(array).height == 1


def test_read_bytes_requires_an_explicit_format():
    raw = b"client_id,income\nC-1,1000.0\n"
    assert read(raw, format="csv").height == 1
    with pytest.raises(LoadError):
        read(raw)


def test_read_unknown_extension_requires_format(tmp_path):
    path = tmp_path / "loans.data"
    path.write_text("whatever")
    with pytest.raises(LoadError):
        read(path)


def test_read_failure_is_a_load_error(tmp_path):
    path = tmp_path / "loans.parquet"
    path.write_text("this is not parquet")
    with pytest.raises(LoadError):
        read(path)


def test_read_accepts_an_already_loaded_frame():
    assert read(_loans(), format="csv").height == 2


def test_scan_is_lazy(tmp_path):
    path = tmp_path / "loans.csv"
    _loans().write_csv(path)
    assert isinstance(scan(str(path)), pl.LazyFrame)
