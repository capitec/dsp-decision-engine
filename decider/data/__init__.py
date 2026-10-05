"""Reading data, the project's named datasets, record identity and pre-flight checks.

`read` and `scan` load Redshift-export JSON, CSV and Parquet as a polars frame
or lazy frame. `datasets()` and `dataset(name)` give a named dataset;
its `load()` returns `LoadedData`, the frame plus the schema, row count, key
suggestion and content fingerprint a run records. `preflight` reports data
compatibility, param constraints, capabilities and a cost estimate before
anything runs.

Example::

    from decider.data import preflight, read
    df = read("loans.parquet")
    preflight(pipeline, df)
"""
from decider.data._formats import scan
from decider.data._keys import DuplicateRecordKeyError, MissingKeyColumnError, MissingRecordKeyError, RecordKeyError
from decider.data._load import LoadedData, LoadError, read
from decider.data._preflight import CostEstimate, PreflightReport, preflight
from decider.data._providers import DatasetError, DatasetProvider, FileProvider, FolderProvider, dataset, datasets

__all__ = [
    "CostEstimate", "DatasetError", "DatasetProvider", "DuplicateRecordKeyError", "FileProvider", "FolderProvider",
    "LoadError", "LoadedData", "MissingKeyColumnError", "MissingRecordKeyError", "PreflightReport", "RecordKeyError",
    "dataset", "datasets", "preflight", "read", "scan",
]
