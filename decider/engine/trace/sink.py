"""The decision-trace capture surface: a `TraceSink` a run drains into, and the adapter seam.

The kernel writes int64 events into a buffer the runner passes it; the runner
drains that buffer here, verifies trace-point conservation, joins each
kernel-local step reference to its durable `Origin`, and hands the decoded
events to an adapter. An adapter can transform, redact, enrich, enqueue or
export events off the request path; conservation runs *before* the adapter, so
redaction or deletion downstream can never hide that evidence went missing.

Tracing is off by default: `Engine.run`/`score` capture nothing unless a
caller passes a `TraceSink`. Building one is the explicit opt-in that turns
PII-bearing capture on.
"""
from __future__ import annotations

from typing import Protocol, Sequence

import numpy as np

from decider.engine.trace.envelope import SCHEMA_VERSION, Kind, StepTable, TraceEvent, decode


class TraceLossError(RuntimeError):
    """A kernel emitted fewer trace events than its program declares: evidence was dropped."""


class Adapter(Protocol):
    """What a `TraceSink` hands decoded events to.

    Called synchronously on the request path (conservation runs there too),
    but a client may enqueue the events and deliver them on another thread or
    process. It must not raise for the run to go on; an exception is counted
    in the sink's manifest and the run proceeds.
    """

    def accept(self, events: Sequence[TraceEvent]) -> None: ...


class PassThroughAdapter:
    """Collects every event it receives, for inspection; a named pass-through.

    Example::

        adapter = PassThroughAdapter()
        exe.run(df, trace=TraceSink(adapter))
        adapter.events[0].kind
    """

    def __init__(self) -> None:
        self.events: list[TraceEvent] = []

    def accept(self, events: Sequence[TraceEvent]) -> None:
        self.events.extend(events)


class TraceSink:
    """Where a run's decision evidence goes: capture, conservation, adapter handoff.

    Args:
        adapter: an `Adapter` that also receives each event (to transform,
            redact, enrich or export); the sink always retains events for
            inspection, so leaving it unset is a pass-through.
        max_events: drop events past this many (reported as `dropped`), so a
            high-volume batch run cannot grow unbounded.
        sample_rate: keep this fraction of events (a deterministic 1-in-N
            sample), for batch runs where full capture is not wanted.
        strict: raise `TraceLossError` when a kernel emits fewer events than it
            declared (conservation); when false, count it as `dropped` instead.

    Events are ordered within one record's stream; cross-record order is the
    driver's, unspecified. `status()` reports delivery and loss, never silently.

    Example::

        sink = TraceSink()
        exe.run(df, trace=sink)
        [e.origin.path for e in sink.events() if e.kind is Kind.STEP]
    """

    def __init__(self, adapter: Adapter | None = None, *, max_events: int | None = None,
                 sample_rate: float = 1.0, strict: bool = True):
        if not 0.0 < sample_rate <= 1.0:
            raise ValueError(f"sample_rate must be in (0, 1], not {sample_rate!r}")
        self.adapter = adapter
        self.max_events = max_events
        self.sample_rate = sample_rate
        self.strict = strict
        self._events: list[TraceEvent] = []
        self._seen = 0
        self._delivered = 0
        self._dropped = 0
        self._failed = 0
        self._errors = 0

    def emit(self, kind: Kind, origin, *, step: int = 0, arm: int | None = None,
             iteration: int | None = None, record: int | None = None,
             value: int | float | None = None) -> None:
        """Record one driver-emitted event (a Python step, a frame step, an override)."""
        self._append(TraceEvent(SCHEMA_VERSION, kind, step, origin, arm, iteration, record, value))

    def drain_kernel(self, headers: np.ndarray, offsets: np.ndarray, table: StepTable,
                     written: int, expected: int | None = None) -> None:
        """Decode a kernel's raw events, verify conservation, and hand them to the adapter.

        `headers` is the kernel's int64 buffer; `offsets[i]` is the cursor at
        the end of row `i`, so each event resolves to the record it belongs to.
        Conservation runs before the adapter: if `expected` is set and `written`
        falls short, evidence was dropped and (when `strict`) this raises rather
        than let redaction downstream hide it.
        """
        if expected is not None and written != expected:
            self._dropped += max(0, expected - written)
            if self.strict:
                raise TraceLossError(
                    f"trace conservation failed: {written} of {expected} declared events were captured; "
                    "an optimisation dropped a trace point")
            return
        if written == 0:
            return
        records = np.empty(written, np.int64)
        prev = 0
        for i, hi in enumerate(offsets):
            hi = int(hi)
            records[prev:hi] = i
            prev = hi
        for event in decode(headers[:written], table, records):
            self._append(event)

    def events(self) -> tuple[TraceEvent, ...]:
        """The events collected so far, in order; the run's decision evidence."""
        return tuple(self._events)

    def status(self) -> dict[str, int]:
        """Delivery and loss counters: `{delivered, dropped, failed, errors}`."""
        return {"delivered": self._delivered, "dropped": self._dropped,
                "failed": self._failed, "errors": self._errors}

    def _append(self, event: TraceEvent) -> None:
        self._seen += 1
        if self.sample_rate < 1.0 and self._seen % max(1, round(1 / self.sample_rate)) != 0:
            self._dropped += 1
            return
        if self.max_events is not None and self._delivered >= self.max_events:
            self._dropped += 1
            return
        self._events.append(event)
        if self.adapter is None:
            self._delivered += 1
            return
        try:
            self.adapter.accept((event,))
            self._delivered += 1
        except Exception:
            self._failed += 1
            self._errors += 1
