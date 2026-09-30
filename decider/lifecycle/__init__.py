"""The execution-lifecycle contract: finite jobs and versioned run manifests.

Finite long-running work — data loading, check runs, revision comparison,
experiment execution, trace export — shares one job model (`Job`, `JobHandle`),
so Python, VS Code and MCP report and cancel it the same way. A live debug
`Session` is deliberately not a job.

A finished run is recorded in an immutable, versioned `RunManifest`: small
metadata (revisions, environment, input fingerprint, selection, overrides,
result references), never the client's bulk results or traces. The manifest is
what a result reopens or shares from, so nothing depends on an ephemeral
session id.

Example::

    from decider.lifecycle import JobHandle, RunManifest
    handle = JobHandle(job_id="j1", kind="check_run")
    handle.start()
    handle.succeed()
"""
from decider.lifecycle.job import (Job, JobHandle, JobStatus, PartialResult, TERMINAL_STATUSES)
from decider.lifecycle.manifest import (MANIFEST_VERSION, Environment, InputFingerprint, Override, ResultRef, Revision,
                                        RunManifest, Sample, Selection, SourceState, check_manifest_version,
                                        json_schema, new_manifest_id, source_is_stale)

__all__ = [
    "MANIFEST_VERSION", "Environment", "InputFingerprint", "Job", "JobHandle", "JobStatus", "Override", "PartialResult",
    "ResultRef", "Revision", "RunManifest", "Sample", "Selection", "SourceState", "TERMINAL_STATUSES",
    "check_manifest_version", "json_schema", "new_manifest_id", "source_is_stale",
]
