"""Record identity: the id-column heuristic as a suggestion, and explicit/composite keys."""
from __future__ import annotations

from typing import Iterable

import polars as pl

from decider.contract import RecordRef
from decider.exceptions import DeciderError


class RecordKeyError(DeciderError, ValueError):
    """A record key can't be resolved, or doesn't identify every record."""

    _STATUS_CODE = 400


class MissingKeyColumnError(RecordKeyError):
    """A named key column isn't in the data, or no id-like column was found."""


class MissingRecordKeyError(RecordKeyError):
    """A record's key value is null, so it has no durable identity."""


class DuplicateRecordKeyError(RecordKeyError):
    """Two records share a key value, so the key doesn't identify one record."""


def suggest_key(columns: Iterable[str]) -> tuple[str, ...] | None:
    """The first id-like column: `id`, else the first `*_id`, else the first `id_*`; `None` if none.

    A load-time suggestion only; a caller that needs a durable identity should
    confirm or replace it with an explicit `id_column` or `key_columns`.
    """
    names = list(columns)
    for name in names:
        if name == "id":
            return (name,)
    for name in names:
        if name.endswith("_id"):
            return (name,)
    for name in names:
        if name.startswith("id_"):
            return (name,)
    return None


def resolve_key(columns: Iterable[str], *, id_column: str | None = None,
                key_columns: Iterable[str] | None = None) -> tuple[str, ...]:
    """The key columns: an explicit composite `key_columns`, else `id_column`, else the suggestion.

    Raises `MissingKeyColumnError` when no key can be found or a named column isn't in `columns`.
    """
    names = list(columns)
    if key_columns is not None:
        key = tuple(key_columns)
    elif id_column is not None:
        key = (id_column,)
    else:
        key = suggest_key(names)
    if key is None:
        raise MissingKeyColumnError("no id-like column found; pass id_column or key_columns")
    missing = [c for c in key if c not in names]
    if missing:
        raise MissingKeyColumnError(f"key column {missing!r} is not in the data; columns are {names!r}")
    return key


def check_key(frame: pl.DataFrame, key: tuple[str, ...]) -> None:
    """Raise if `key` doesn't uniquely and completely identify every record.

    Rejects a key column absent from the frame (`MissingKeyColumnError`), null
    key values (`MissingRecordKeyError`) and duplicate key values
    (`DuplicateRecordKeyError`); call it when an experiment or debug
    reproduction needs a durable record identity.
    """
    missing = [c for c in key if c not in frame.columns]
    if missing:
        raise MissingKeyColumnError(f"key column {missing!r} is not in the data")
    if not key:
        return
    keyframe = frame.select(key)
    nulls = sum(keyframe[c].null_count() for c in key)
    if nulls:
        raise MissingRecordKeyError(f"{nulls} record(s) have a null value in the key columns {list(key)!r}")
    if keyframe.is_duplicated().any():
        raise DuplicateRecordKeyError(f"the key columns {list(key)!r} do not uniquely identify every record")


def record_ref(frame: pl.DataFrame, key: tuple[str, ...], *, dataset: str = "") -> RecordRef:
    """The durable `RecordRef` of one selected record.

    `frame` must hold exactly one row, and its key columns must be non-null and
    unique (see `check_key`). The key values are the identity; row order never is.

    Example::

        record_ref(df.filter(pl.col("client_id") == "C-1"), ("client_id",), dataset="loans.parquet")
    """
    if frame.height != 1:
        raise RecordKeyError(f"expected exactly one selected record, got {frame.height}")
    check_key(frame, key)
    values = dict(zip(key, frame.select(key).row(0)))
    return RecordRef(dataset=dataset, key=values)
