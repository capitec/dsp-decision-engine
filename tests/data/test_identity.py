from __future__ import annotations

import polars as pl
import pytest

from decider.contract import RecordRef
from decider.data import (DuplicateRecordKeyError, MissingKeyColumnError, MissingRecordKeyError, check_key, record_ref,
                          resolve_key, suggest_key)


def test_suggest_key_prefers_id_then_suffix_then_prefix():
    assert suggest_key(["name", "id", "client_id"]) == ("id",)
    assert suggest_key(["name", "client_id", "id_number"]) == ("client_id",)
    assert suggest_key(["name", "id_number"]) == ("id_number",)
    assert suggest_key(["name", "age"]) is None


def test_resolve_key_prefers_explicit_composite_and_id_column():
    cols = ["client_id", "loan_ref", "name"]
    assert resolve_key(cols, key_columns=["client_id", "loan_ref"]) == ("client_id", "loan_ref")
    assert resolve_key(cols, id_column="loan_ref") == ("loan_ref",)
    assert resolve_key(cols) == ("client_id",)


def test_resolve_key_rejects_missing_column_and_no_suggestion():
    with pytest.raises(MissingKeyColumnError):
        resolve_key(["name", "age"], id_column="client_id")
    with pytest.raises(MissingKeyColumnError):
        resolve_key(["name", "age"])


def test_check_key_accepts_a_unique_complete_key():
    df = pl.DataFrame({"client_id": ["C-1", "C-2"], "v": [1, 2]})
    check_key(df, ("client_id",))


def test_check_key_rejects_duplicates():
    df = pl.DataFrame({"client_id": ["C-1", "C-1"], "v": [1, 2]})
    with pytest.raises(DuplicateRecordKeyError):
        check_key(df, ("client_id",))


def test_check_key_rejects_missing_values():
    df = pl.DataFrame({"client_id": ["C-1", None], "v": [1, 2]})
    with pytest.raises(MissingRecordKeyError):
        check_key(df, ("client_id",))


def test_record_ref_builds_a_durable_reference():
    df = pl.DataFrame({"client_id": ["C-1"], "income": [1000.0]})
    ref = record_ref(df, ("client_id",), dataset="loans.parquet")
    assert ref == RecordRef(dataset="loans.parquet", key={"client_id": "C-1"})
    assert ref.durable
