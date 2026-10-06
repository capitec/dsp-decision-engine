"""Reading data, the project's named datasets, and record identity.

`read` and `scan` load Redshift-export JSON, CSV and Parquet as a polars frame
or lazy frame. `datasets()` and `dataset(name)` give a named dataset;
its `load()` returns `LoadedData`, the frame plus the schema, row count, key
suggestion and content fingerprint a run records.

Example::

    from decider.lab.data import read
    df = read("loans.parquet")
"""
from decider.lab.data._formats import scan
from decider.lab.data._keys import DuplicateRecordKeyError, MissingKeyColumnError, MissingRecordKeyError, RecordKeyError
from decider.lab.data._load import InputFingerprint, LoadedData, LoadError, read
from decider.lab.data._providers import DatasetError, DatasetProvider, FileProvider, FolderProvider, dataset, datasets

__all__ = [
    "DatasetError", "DatasetProvider", "DuplicateRecordKeyError", "FileProvider", "FolderProvider", "InputFingerprint",
    "LoadError", "LoadedData", "MissingKeyColumnError", "MissingRecordKeyError", "RecordKeyError",
    "dataset", "datasets", "read", "scan",
]
