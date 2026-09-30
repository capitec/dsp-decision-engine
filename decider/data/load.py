"""Load Redshift-export data (JSON, CSV, Parquet) into a frame, with schema, row count, key and fingerprint."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

import polars as pl
from pydantic import BaseModel, PrivateAttr

from decider.data.identity import suggest_key
from decider.exceptions import DeciderError
from decider.lifecycle import InputFingerprint, JobHandle

FORMATS = ("json", "csv", "parquet")


class LoadError(DeciderError):
    """A data source can't be read; the message names the source and the reason."""

    _STATUS_CODE = 400


class LoadedData(BaseModel):
    """A loaded dataset: the polars frame plus the summary a run manifest records.

    `frame` is the data; `summary` is the `InputFingerprint` (schema, row count,
    content fingerprint) a run manifest binds; `suggested_key` is the load-time
    record-key suggestion, `None` when no column looks like an id.

    Example::

        data = load("loans.parquet")
        data.row_count, data.columns, data.suggested_key
    """

    summary: InputFingerprint
    suggested_key: tuple[str, ...] | None = None
    _frame: pl.DataFrame = PrivateAttr(default=None)

    @classmethod
    def from_frame(cls, frame: pl.DataFrame, dataset: str = "",
                   suggested_key: tuple[str, ...] | None = None) -> "LoadedData":
        columns = tuple((name, str(dtype)) for name, dtype in frame.schema.items())
        data = cls(
            summary=InputFingerprint(dataset=dataset, columns=columns, row_count=frame.height,
                                     fingerprint=_fingerprint(frame)),
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


def load(source: Any, *, format: str | None = None, job: JobHandle | None = None) -> LoadedData:
    """Load `source` — a `.json`/`.jsonl`/`.csv`/`.parquet` path, bytes, or a polars frame — as `LoadedData`.

    JSON is one object per line (a Redshift UNLOAD) or an array of objects.
    `format` is required only when the source has no file extension. A
    `JobHandle` passed as `job` reports progress and can cancel the load: its
    progress is coarse (reading, describing), so a cancel takes effect between
    those phases rather than inside one read.

    Example::

        data = load("loans.parquet")
    """
    dataset = _name(source)
    if job is not None:
        job.start()
        job.progress(0, total=3, message="reading")
    if job is not None and job.cancelled:
        raise LoadError(f"load of {dataset!r} was cancelled")
    try:
        frame = _read(source, format)
    except Exception as e:
        if job is not None:
            job.fail(str(e))
        raise LoadError(f"could not load {dataset!r}: {e}") from e
    if job is not None:
        job.progress(1, total=3, message="describing")
    data = LoadedData.from_frame(frame, dataset=dataset, suggested_key=suggest_key(frame.columns))
    if job is not None:
        if job.cancelled:
            raise LoadError(f"load of {dataset!r} was cancelled")
        job.progress(2, total=3, message="done")
        job.succeed(result=data.summary)
    return data


def _read(source: Any, format: str | None) -> pl.DataFrame:
    if isinstance(source, pl.DataFrame):
        return source
    fmt = format
    if fmt is None and not isinstance(source, (bytes, bytearray)):
        fmt = _format_from_name(str(source))
    if fmt is None:
        raise LoadError("no data format given; pass format='json'|'csv'|'parquet' or use a .json/.csv/.parquet path")
    if fmt not in FORMATS:
        raise LoadError(f"unknown data format {fmt!r}; expected one of {', '.join(FORMATS)}")
    return {"json": _read_json, "csv": pl.read_csv, "parquet": pl.read_parquet}[fmt](source)


def _read_json(source: Any) -> pl.DataFrame:
    # Redshift UNLOAD exports JSON one object per line; an array of objects is also accepted.
    try:
        return pl.read_ndjson(source)
    except Exception:
        return pl.read_json(source)


def _format_from_name(name: str) -> str | None:
    lower = name.lower()
    if lower.endswith((".json", ".jsonl", ".ndjson")):
        return "json"
    if lower.endswith(".csv"):
        return "csv"
    if lower.endswith(".parquet"):
        return "parquet"
    return None


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
