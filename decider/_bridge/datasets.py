"""The project's named datasets, for the editor's dataset picker."""
from __future__ import annotations

import polars as pl

from decider.data import FileProvider, FolderProvider, dataset, dataset_documents, datasets_folder
from decider.data import _formats as formats


def datasets(file=None):
    """The datasets folder and each dataset's type, key and local path; reads no data."""
    found = []
    for name, doc in dataset_documents(file).items():
        path = doc.get("path") if doc.get("type") in ("file", "folder") else None
        entry = {"name": name, "type": doc.get("type"), "key": None,
                 "path": path if isinstance(path, str) and "://" not in path else None}
        try:
            key = dataset(name, file).key
            entry["key"] = list(key) if key else None
        except Exception as e:  # noqa: BLE001  one broken entry must not hide the rest
            entry.update(path=None, error=str(e))
        found.append(entry)
    return {"root": str(datasets_folder(file)), "datasets": found}


def dataset_schema(name, file=None):
    """Column names and dtypes, plus the row count when the format holds it in metadata."""
    provider = dataset(name, file)
    count = None
    fmt = (provider._inferred()[0] if isinstance(provider, FolderProvider)
           else provider.format or formats.format_of(provider.path) if isinstance(provider, FileProvider) else None)
    if fmt == "parquet":
        count = provider.scan().select(pl.len()).collect().item()
    return {"columns": [list(c) for c in provider.schema().items()], "row_count": count}


def dataset_head(name, n=50, file=None):
    """The first `n` rows of a dataset, with its columns."""
    provider = dataset(name, file)
    frame = provider.head(n)
    return {"columns": [[c, str(t)] for c, t in frame.schema.items()], "rows": frame.to_dicts()}
