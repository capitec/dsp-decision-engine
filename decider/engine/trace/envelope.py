"""The decision-trace envelope: one int64 per event, decoded to a structured `TraceEvent`.

A kernel captures only fixed-width integers; names, sources and other
metadata are joined here in the driver. The one-int64 envelope carries the
event kind, a step reference into the run's constant table, and the
branch-arm / loop-iteration flow context. A fixed-width numeric payload (a
matched row, a path number, a band, a rounded value) rides `TraceEvent.value`
and is attached by the driver, never written by the kernel.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Any

import numpy as np

from decider.engine.ir.origin import Origin

SCHEMA_VERSION = 1
"""The version of the trace envelope. Bump it when the bit layout or the meaning
of a kind changes in a way older readers can't decode."""

KIND_BITS = 8
STEP_BITS = 20
ARM_BITS = 16
ITER_BITS = 20

STEP_SHIFT = KIND_BITS
ARM_SHIFT = KIND_BITS + STEP_BITS
ITER_SHIFT = KIND_BITS + STEP_BITS + ARM_BITS

_KIND_MASK = (1 << KIND_BITS) - 1
_STEP_MASK = (1 << STEP_BITS) - 1
_ARM_MASK = (1 << ARM_BITS) - 1
_ITER_MASK = (1 << ITER_BITS) - 1


class Kind(IntEnum):
    """What one trace event records.

    STEP, BRANCH_ARM, LOOP_ITERATION and FRAME are structural: they say what
    ran, in what flow context. TABLE_ROW, TREE_PATH, BAND, ROUNDING, REASON
    and VALUE carry a fixed-width numeric payload in `TraceEvent.value`, and
    OVERRIDE records a value replaced mid-run by a debug session.
    """

    STEP = 0
    BRANCH_ARM = 1
    LOOP_ITERATION = 2
    FRAME = 3
    TABLE_ROW = 4
    TREE_PATH = 5
    BAND = 6
    ROUNDING = 7
    REASON = 8
    VALUE = 9
    OVERRIDE = 10


def pack(kind: Kind | int, step: int = 0, arm: int = 0, iteration: int = 0) -> int:
    """One int64 event: `kind` in the low bits, then `step`, `arm` and `iteration`.

    `step` is a 1-based index into the run's constant table (0 means none);
    `arm` is a branch arm index plus 1 (0 means not in an arm); `iteration`
    is a loop iteration from 1 (0 means not in a loop).
    """
    return int(kind) | (int(step) << STEP_SHIFT) | (int(arm) << ARM_SHIFT) | (int(iteration) << ITER_SHIFT)


def unpack(value: int) -> tuple[int, int, int, int]:
    """`(kind, step, arm, iteration)` of a packed event; `arm` is already minus 1."""
    v = int(value)
    return (v & _KIND_MASK, (v >> STEP_SHIFT) & _STEP_MASK, (v >> ARM_SHIFT) & _ARM_MASK,
            (v >> ITER_SHIFT) & _ITER_MASK)


@dataclass(frozen=True, slots=True)
class TraceEvent:
    """One decision-trace event, decoded: what ran, where, in what flow context.

    Args:
        schema_version: the envelope version this event was written against.
        kind: the event kind (see `Kind`).
        step: the 1-based index of the step's `Origin` in the run's constant table.
        origin: the durable step reference — `Origin.id` — with its capture-time
            `path` and `source`; `None` for an event with no step (a frame-scope marker).
        arm: the branch arm (from 0), or `None` outside a branch.
        iteration: the loop iteration (from 1), or `None` outside a loop.
        record: the row index within the run the event belongs to; `None` for a
            frame-scope event.
        value: a fixed-width numeric payload for the kinds that carry one.
    """

    schema_version: int
    kind: Kind
    step: int
    origin: Origin | None
    arm: int | None
    iteration: int | None
    record: int | None
    value: int | float | None = None


class StepTable:
    """The per-run constant table: step ref (1-based) to the step's durable `Origin`.

    Versioned (`schema_version`), like the envelope, because a reader joins
    events to it; an event written by one version must not resolve against a
    table written by another.
    """

    __slots__ = ("schema_version", "steps")

    def __init__(self, steps: tuple[Origin, ...]):
        self.schema_version = SCHEMA_VERSION
        self.steps = steps

    def origin(self, ref: int) -> Origin | None:
        return self.steps[ref - 1] if 0 < ref <= len(self.steps) else None


def decode(headers: np.ndarray, table: StepTable, records: np.ndarray) -> list[TraceEvent]:
    """Decode a kernel's int64 events into `TraceEvent`s.

    `headers` is the flat event buffer; `records[i]` is the row each event
    belongs to, so `records` and `headers` must be the same length. The
    shift/mask unpack is vectorised; this is the driver's join step, where
    kernel-local step indices resolve to durable `Origin`s via `table`.

    Example::

        decode(np.array([pack(Kind.STEP, 1)]), StepTable((origin,)), np.array([0]))
    """
    kinds, steps, arms, iters = _unpack(headers)
    rows = int(headers.shape[0])
    version = table.schema_version
    events = []
    append = events.append
    for i in range(rows):
        kind = Kind(kinds[i])
        step = int(steps[i])
        arm = int(arms[i]) - 1
        iteration = int(iters[i])
        append(TraceEvent(version, kind, step, table.origin(step), None if arm < 0 else arm,
                          None if iteration == 0 else iteration, int(records[i])))
    return events


def _unpack(headers: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Vectorised shift/mask over the whole buffer at once, never per-event Python.
    v = headers.astype(np.int64, copy=False)
    kinds = v & _KIND_MASK
    steps = (v >> STEP_SHIFT) & _STEP_MASK
    arms = (v >> ARM_SHIFT) & _ARM_MASK
    iters = (v >> ITER_SHIFT) & _ITER_MASK
    return kinds, steps, arms, iters


def bitcast(value: Any) -> int:
    """A float64 as its int64 bits, losslessly, so a numeric value rides an int64 payload.

    Example::

        bitcast(42.125)  # 4631125384006467584
    """
    return int(np.float64(value).view(np.int64))
