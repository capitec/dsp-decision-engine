"""Structured decision tracing: compact in-kernel capture, joined and conserved in the driver.

Kernels emit one int64 per event into a per-invocation buffer; the driver
drains that buffer, verifies trace-point conservation, joins kernel-local step
references to durable `Origin`s, and hands decoded events to an adapter. See
`envelope` for the wire format and `sink` for the capture surface.

This trace coexists with `Session.events` (node start/finish/visited, kept for
the debugger's step model) and with `trace_output`/`path_output` (per-row
result columns, kept for population aggregation). It adds the event-level
decision evidence those two don't carry: durable step identity, branch/loop
flow context, and a numeric payload per event.
"""
from decider.engine.trace.envelope import (SCHEMA_VERSION, Kind, StepTable, TraceEvent, bitcast, decode, pack, unpack)
from decider.engine.trace.sink import Adapter, PassThroughAdapter, TraceLossError, TraceSink

__all__ = [
    "Adapter", "Kind", "PassThroughAdapter", "SCHEMA_VERSION", "StepTable", "TraceEvent", "TraceLossError",
    "TraceSink", "bitcast", "decode", "pack", "unpack",
]
