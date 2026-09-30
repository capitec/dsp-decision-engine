"""Data loading, record identity, execution scopes and pre-flight checks for a run.

`load` reads Redshift-export JSON, CSV and Parquet into a `LoadedData`: a polars
frame plus the schema, row count, key suggestion and content fingerprint a run
manifest records. Record identity reuses the contract's `RecordRef`: the id-like
heuristic is a suggestion, an explicit column or composite key makes it durable,
and `check_key` rejects duplicates and nulls when a reproduction needs one.
`check_scope` chooses selected-record vs whole-frame execution and redirects a
record-only scope over frame steps. `preflight` reports data compatibility,
param constraints, capabilities and a cost estimate before anything runs.

Example::

    from decider.data import load, resolve_key, check_scope, preflight
    data = load("loans.parquet")
    key = resolve_key(data.frame.columns, id_column="client_id")
    check_scope(describe(pipeline), ExecutionScope.SELECTED_RECORD, focus=...)
    preflight(pipeline, data.frame)
"""
from decider.data.identity import (DuplicateRecordKeyError, MissingKeyColumnError, MissingRecordKeyError,
                                   RecordKeyError, check_key, record_ref, resolve_key, suggest_key)
from decider.data.load import FORMATS, LoadError, LoadedData, load
from decider.data.preflight import CostEstimate, PreflightReport, preflight
from decider.data.scope import ExecutionScope, ScopeCheck, ScopeError, check_scope, frame_steps

__all__ = [
    "CostEstimate", "DuplicateRecordKeyError", "ExecutionScope", "FORMATS", "LoadError", "LoadedData",
    "MissingKeyColumnError", "MissingRecordKeyError", "PreflightReport", "RecordKeyError", "ScopeCheck", "ScopeError",
    "check_key", "check_scope", "frame_steps", "load", "preflight", "record_ref", "resolve_key", "suggest_key",
]
