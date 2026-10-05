from __future__ import annotations

import typing as t
from pathlib import Path

import polars as pl
from pydantic import PrivateAttr

from decider.data import formats
from decider.data.providers.base import DatasetError, DatasetProvider

_JSON_SUFFIXES = (".jsonl", ".ndjson", ".json")


class FileProvider(DatasetProvider):
    """A dataset in one file (or a glob, or an `s3://` URL); the format comes from the suffix.

    Example::

        FileProvider(path="datasets/loans.parquet", key=("loan_id",))
    """

    type: t.Literal["file"] = "file"
    path: str
    format: str | None = None

    def scan(self) -> pl.LazyFrame:
        fmt = self.format or formats.format_of(self.path)
        if fmt is None:
            raise DatasetError(f"cannot infer the format of {self.path!r}; set `format` to json, csv or parquet")
        return formats.scan(self.path, fmt)


class FolderProvider(DatasetProvider):
    """A dataset split into part files, `{path}/{part}.{ext}`; parquet `region=za/` folders become columns.

    Example::

        FolderProvider(path="datasets/loans")
    """

    type: t.Literal["folder"] = "folder"
    path: str
    format: str | None = None
    _found: tuple[str, str] | None = PrivateAttr(default=None)

    def scan(self) -> pl.LazyFrame:
        fmt, suffix = self._inferred()
        if fmt == "parquet":
            return pl.scan_parquet(f"{self.path}/**/*.parquet", hive_partitioning=True)
        if fmt == "csv":
            return pl.scan_csv(f"{self.path}/*.csv")
        if fmt == "json":
            return pl.scan_ndjson(f"{self.path}/*{suffix}")
        raise DatasetError(f"unknown format {fmt!r} for folder {self.path!r}; expected json, csv or parquet")

    def _inferred(self) -> tuple[str, str]:
        if self._found is None:
            self._found = self._infer()
        return self._found

    def _infer(self) -> tuple[str, str]:
        if "://" in self.path:
            if self.format is None:
                raise DatasetError(f"cannot list the remote folder {self.path!r}; set `format` to json, csv or parquet")
            return self.format, ".json" if self.format == "json" else ""
        suffixes = {
            p.suffix.lower()
            for p in Path(self.path).rglob("*")
            if p.is_file() and not any(part.startswith(".") for part in p.relative_to(self.path).parts)
            and formats.format_of(p.name)
        }
        by_format: dict[str, list[str]] = {}
        for s in sorted(suffixes):
            by_format.setdefault(formats.format_of(s), []).append(s)
        fmt = self.format
        if fmt is None:
            if not by_format:
                raise DatasetError(f"no data files in folder {self.path!r}")
            if len(by_format) > 1:
                raise DatasetError(f"mixed formats in folder {self.path!r}: {', '.join(sorted(by_format))}")
            fmt = next(iter(by_format))
        found = by_format.get(fmt, [])
        if not found:
            raise DatasetError(f"no {fmt} files in folder {self.path!r}")
        return fmt, found[0]
