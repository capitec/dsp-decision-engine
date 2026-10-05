"""Read Redshift-export data (JSON, CSV, Parquet) into a frame; `LoadedData` adds the schema, row count, key and fingerprint a run records."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

import polars as pl
from pydantic import BaseModel, PrivateAttr

from decider.data._formats import format_of, scan
from decider.exceptions import DeciderError
from decider.contract import InputFingerprint

_FORMATS = ("json", "csv", "parquet")


class LoadError(DeciderError):
    """A data source can't be read; the message names the source and the reason."""

    _STATUS_CODE = 400


class LoadedData(BaseModel):
    """A loaded dataset: the polars frame plus the summary a run manifest records.

    `frame` is the data; `summary` is the `InputFingerprint` (schema, row count,
    content fingerprint) a run manifest binds; `suggested_key` is the record-key
    suggestion, `None` when no column looks like an id.

    Example::

        data = dataset("loans").load()
        data.row_count, data.columns, data.suggested_key
    """

    summary: InputFingerprint
    suggested_key: tuple[str, ...] | None = None
    _frame: pl.DataFrame = PrivateAttr(default=None)

    @classmethod
    def from_frame(cls, frame: pl.DataFrame, dataset: str = "",
                   suggested_key: tuple[str, ...] | None = None, fingerprint: str | None = None) -> "LoadedData":
        columns = tuple((name, str(dtype)) for name, dtype in frame.schema.items())
        data = cls(
            summary=InputFingerprint(dataset=dataset, columns=columns, row_count=frame.height,
                                     fingerprint=fingerprint or _fingerprint(frame)),
            suggested_key=suggested_key,
        )
        data._frame = frame
        return data

    @property
    def frame(self) -> pl.DataFrame:
        return self._frame

    @property
    def columns(self) -> tuple[tuple[str, str], ...]:
        return self.summary.columns

    @property
    def row_count(self) -> int | None:
        return self.summary.row_count

    @property
    def dataset(self) -> str:
        return self.summary.dataset

    @property
    def fingerprint(self) -> str | None:
        return self.summary.fingerprint


def read(source: Any, *, format: str | None = None) -> pl.DataFrame:
    """Read `source`, a `.json`/`.jsonl`/`.csv`/`.parquet` path, bytes or a polars frame, into a frame.

    JSON is one object per line (a Redshift UNLOAD) or an array of objects.
    `format` is required only when the source has no file extension.

    Example::

        df = read("loans.parquet")
    """
    try:
        return _read(source, format)
    except Exception as e:
        raise LoadError(f"could not load {_name(source)!r}: {e}") from e


def _read(source: Any, format: str | None) -> pl.DataFrame:
    if isinstance(source, pl.DataFrame):
        return source
    fmt = format
    if fmt is None and not isinstance(source, (bytes, bytearray)):
        fmt = format_of(str(source))
    if fmt is None:
        raise LoadError("no data format given; pass format='json'|'csv'|'parquet' or use a .json/.csv/.parquet path")
    if fmt not in _FORMATS:
        raise LoadError(f"unknown data format {fmt!r}; expected one of {', '.join(_FORMATS)}")
    if isinstance(source, (bytes, bytearray)):
        return {"json": _read_json, "csv": pl.read_csv, "parquet": pl.read_parquet}[fmt](source)
    return scan(str(source), fmt).collect()


def _read_json(source: Any) -> pl.DataFrame:
    # Redshift UNLOAD exports JSON one object per line; an array of objects is also accepted.
    try:
        return pl.read_ndjson(source)
    except Exception:
        return pl.read_json(source)


def _name(source: Any) -> str:
    if isinstance(source, (str, os.PathLike)):
        return Path(source).name
    return ""


def _fingerprint(frame: pl.DataFrame) -> str | None:
    if frame.width == 0:
        return None
    digest = hashlib.sha256()
    digest.update(b"decider-frame-v1\n")
    digest.update(str(frame.schema).encode())
    digest.update(frame.hash_rows().to_numpy().tobytes())
    return digest.hexdigest()
