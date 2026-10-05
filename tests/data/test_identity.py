from __future__ import annotations

import polars as pl
import pytest

from decider.data import DuplicateRecordKeyError, MissingRecordKeyError, check_key, suggest_key


def test_suggest_key_prefers_id_then_suffix_then_prefix():
    assert suggest_key(["name", "id", "client_id"]) == ("id",)
    assert suggest_key(["name", "client_id", "id_number"]) == ("client_id",)
    assert suggest_key(["name", "id_number"]) == ("id_number",)
    assert suggest_key(["name", "age"]) is None


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

