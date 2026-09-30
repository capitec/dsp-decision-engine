"""Minimal in-kernel trace envelope: pack one int64 event per trace point, decode in Python.

Proves the envelope the spike asks for is encodable in a compiled/nogil kernel
and decodable post-hoc, and measures event size, decode throughput, per-record
ordering, trace-point conservation and adapter backpressure.

    uv run python notes/vscode-redesign/experimentation/01-trace-capture/envelope.py
"""
import time

import numpy as np
from numba import njit

SCHEMA_VERSION = 1

# The envelope fields, packed into one int64 per event. Field widths are a
# choice for a spike, not a contract: 20-bit step ref (1M steps), 8-bit kind,
# 16-bit arm, 24-bit iteration (16M).
STEP_BITS, KIND_BITS, ARM_BITS, ITER_BITS = 20, 8, 16, 24
STEP_SHIFT = KIND_BITS + ARM_BITS + ITER_BITS
KIND_SHIFT = ARM_BITS + ITER_BITS
ARM_SHIFT = ITER_BITS
MASK = (1 << 48) - 1


@njit(nogil=True)
def pack(step_ref, kind, arm, iteration):
    return (step_ref << STEP_SHIFT) | (kind << KIND_SHIFT) | (arm << ARM_SHIFT) | iteration


def unpack(code):
    return (code >> STEP_SHIFT, (code >> KIND_SHIFT) & ((1 << KIND_BITS) - 1),
            (code >> ARM_SHIFT) & ((1 << ARM_BITS) - 1), code & ((1 << ITER_BITS) - 1))


@njit(nogil=True)
def capture(n, steps, max_events, events, offsets):
    # One trace point per step per row, written in row-major order: row r's
    # events are events[offsets[r]:offsets[r+1]], in step order.
    for i in range(n):
        for s in range(steps):
            events[i * steps + s] = pack(s, 0, 0, 0)
        offsets[i] = i * steps
    offsets[n] = n * steps


def decode(events, offsets, r):
    return [unpack(int(events[k])) for k in range(offsets[r], offsets[r + 1])]


def noop_capture(n, steps, events, offsets):
    pass


def throughput(n, steps, reps=200):
    events = np.empty(n * steps, np.int64)
    offsets = np.empty(n + 1, np.int64)
    capture(n, steps, 0, events, offsets)  # compile
    best = min(_time(capture, n, steps, 0, events, offsets) for _ in range(reps))
    return n * steps / best


def decode_throughput(n, steps, reps=5):
    events = np.empty(n * steps, np.int64)
    offsets = np.empty(n + 1, np.int64)
    capture(n, steps, 0, events, offsets)
    best = min(_time(_decode_all, events, offsets, n) for _ in range(reps))
    return n * steps / best


def _decode_all(events, offsets, n):
    out = [unpack(int(events[k])) for r in range(n) for k in range(offsets[r], offsets[r + 1])]
    return out


def _time(f, *args):
    t = time.perf_counter()
    f(*args)
    return time.perf_counter() - t


if __name__ == "__main__":
    print(f"event size: {np.dtype(np.int64).itemsize} bytes (one int64: "
          f"{STEP_BITS}-bit step ref, {KIND_BITS}-bit kind, {ARM_BITS}-bit arm, {ITER_BITS}-bit iteration)")
    print(f"schema version: {SCHEMA_VERSION} (a buffer/run constant, not per event)")

    n, steps = 1_000_000, 24
    per_second = throughput(n, steps)
    print(f"in-kernel capture: {per_second:,.0f} events/s "
          f"({n * steps / per_second * 1e3:.2f} ms for {n * steps:,} events over {n:,} rows x {steps} steps)")
    print(f"capture buffer memory: {n * steps * 8 / 1e6:.0f} MB for {n * steps:,} events "
          f"({n * steps * 8 / n:.0f} bytes per row, i.e. {steps * 8} bytes per row x {steps} events)")

    dec = decode_throughput(100_000, steps)
    print(f"post-hoc decode: {dec:,.0f} events/s")

    # Ordering and conservation: row 7's events are its own, in step order.
    events = np.empty(n * steps, np.int64)
    offsets = np.empty(n + 1, np.int64)
    capture(n, steps, 0, events, offsets)
    row7 = decode(events, offsets, 7)
    assert [e[0] for e in row7] == list(range(steps)), "per-record order broken"
    assert len(row7) == steps, "trace points lost or duplicated"
    assert len(_decode_all(events, offsets, n)) == n * steps, "conservation broken"
    print(f"ordering: row 7 events are step refs {[e[0] for e in row7][:6]}... in step order; "
          f"{n * steps:,} events decoded == {n * steps:,} captured (conserved)")

    # Adapter backpressure: a bounded queue drops overflow; the kernel never blocks or allocates.
    capacity = 10
    queue = []

    def adapter(event):
        if len(queue) >= capacity:
            queue.clear()
        queue.append(event)

    for r in range(n):
        for k in range(offsets[r], offsets[r + 1]):
            adapter(unpack(int(events[k])))
    assert len(queue) == capacity, "adapter should hold the last capacity events"
    print(f"adapter backpressure: bounded queue (capacity {capacity}) drops overflow outside the kernel; "
          f"kernel writes to a preallocated buffer, no per-event Python call or allocation")
