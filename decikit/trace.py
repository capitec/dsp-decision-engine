"""Structured decision tracing: capture per-event evidence from a run.

Example::

    from decikit import Engine
    from decikit.trace import Kind, TraceSink

    sink = TraceSink()
    Engine().bind(pipeline, mode="fused").run(df, trace=sink)
    steps = [e for e in sink.events() if e.kind is Kind.STEP]
"""
from decikit._engine.trace import Adapter, Kind, PassThroughAdapter, TraceEvent, TraceLossError, TraceSink

__all__ = ["Adapter", "Kind", "PassThroughAdapter", "TraceEvent", "TraceLossError", "TraceSink"]
