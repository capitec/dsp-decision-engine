"""Data loading, record identity and pre-flight checks for a run.

`load` reads Redshift-export JSON, CSV and Parquet into a `LoadedData`: a polars
frame plus the schema, row count, key suggestion and content fingerprint a run
manifest records. Record identity reuses the contract's `RecordRef`: the id-like
heuristic is a suggestion, an explicit column or composite key makes it durable,
and `check_key` rejects duplicates and nulls when a reproduction needs one.
`preflight` reports data compatibility,
param constraints, capabilities and a cost estimate before anything runs.

Example::

    from decider.data import load, preflight
    data = load("loans.parquet")
    preflight(pipeline, data.frame)
"""
from decider.data.identity import (DuplicateRecordKeyError, MissingKeyColumnError, MissingRecordKeyError,
                                   RecordKeyError, check_key, suggest_key)
from decider.data.load import FORMATS, LoadError, LoadedData, load
from decider.data.providers import (DatasetError, DatasetProvider, FileProvider, FolderProvider, dataset, dataset_documents,
                                    datasets, datasets_folder, project_root)
from decider.data.preflight import CostEstimate, PreflightReport, preflight

__all__ = [
    "DatasetError", "DatasetProvider", "FileProvider", "FolderProvider", "dataset", "dataset_documents", "datasets", "datasets_folder",
    "project_root",
    "CostEstimate", "DuplicateRecordKeyError", "FORMATS", "LoadError", "LoadedData",
    "MissingKeyColumnError", "MissingRecordKeyError", "PreflightReport", "RecordKeyError",
    "check_key", "load", "preflight", "suggest_key",
]
