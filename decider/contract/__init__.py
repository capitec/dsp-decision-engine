"""The static-flow contract: versioned Python models layered over the IR and run layers.

This is what VS Code, notebooks, checks, experiments and MCP all refer to when
they talk about a flow, instead of inventing incompatible identifiers. It is
static structure only: which nodes and value slots a flow has, and how to name
them. Runtime facts — the values a run observed, the params it validated, the
events a session emitted — live on `RunReport` and `Session.events` and are
deliberately out of scope here.

The reference models (`FlowRef`, `StepRef`, `EdgeRef`, `ValueSlotRef`,
`RecordRef`) are the contract owner: `json_schema` and
`typescript` are generated from them, so every consumer shares one schema.

Finite jobs (`Job`, `JobHandle`) and the versioned run manifest (`RunManifest`)
live here too. Finite long-running work shares one job model, so Python, VS Code
and MCP report and cancel it the same way; a finished run is an immutable
`RunManifest` of small metadata (revisions, environment, input fingerprint,
selection, overrides, result references), never the client's bulk results.

Example::

    from decider.contract import JobHandle, describe
    desc = describe(pipeline)
    for edge in desc.edges:
        print(edge.from_path, edge.to_path, edge.kind)
    handle = JobHandle(job_id="j1", kind="check_run")
    handle.start()
    handle.succeed()
"""
from decider.contract._describe import Capabilities, capabilities, describe, resolve_step, source_location
from decider.contract._jobs import Job, JobHandle, JobStatus, PartialResult
from decider.contract._manifest import (Environment, InputFingerprint, Override, ResultRef, Revision, RunManifest,
                                        Sample, Selection, SourceState, manifest_schema)
from decider.contract._refs import (CONTRACT_VERSION, EdgeRef, FlowDescription, FlowRef, RecordRef, StepRef,
                                   ValueSlotRef)
from decider.contract._schema import json_schema, typescript

__all__ = [
    "CONTRACT_VERSION", "Capabilities", "EdgeRef", "Environment", "FlowDescription", "FlowRef", "InputFingerprint",
    "Job", "JobHandle", "JobStatus", "Override", "PartialResult", "RecordRef", "ResultRef", "Revision", "RunManifest",
    "Sample", "Selection", "SourceState", "StepRef", "ValueSlotRef", "capabilities", "describe", "json_schema",
    "manifest_schema", "resolve_step", "source_location", "typescript",
]
