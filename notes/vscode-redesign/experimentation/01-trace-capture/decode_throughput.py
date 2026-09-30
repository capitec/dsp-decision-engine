"""Decode throughput: per-event Python vs vectorised vs lazy vs aggregation; a budget.

The capture buffer is one int64 per event (the envelope). Materialising those
into structured events is the cost the first spike measured at ~1.4 M events/s
for a per-event Python loop. This compares three faster decodes and sets a
bounded-memory/latency budget for a representative batch trace.

    uv run python notes/vscode-redesign/experimentation/01-trace-capture/decode_throughput.py
"""
import time

import numpy as np
from numba import njit

STEP_BITS, KIND_BITS, ARM_BITS, ITER_BITS = 20, 8, 16, 24
STEP_SHIFT = KIND_BITS + ARM_BITS + ITER_BITS
KIND_SHIFT = ARM_BITS + ITER_BITS
ARM_SHIFT = ITER_BITS


@njit(nogil=True)
def capture(n, steps, events, offsets):
    for i in range(n):
        for s in range(steps):
            events[i * steps + s] = (s << STEP_SHIFT) | ((s % 3) << KIND_SHIFT) | (i << ARM_SHIFT)
        offsets[i] = i * steps
    offsets[n] = n * steps


def timeit(f, *a, reps=5):
    best = min(_once(f, *a) for _ in range(reps))
    return best


def _once(f, *a):
    t = time.perf_counter()
    f(*a)
    return time.perf_counter() - t


def decode_python(events, offsets, n):
    return [(e >> STEP_SHIFT, (e >> KIND_SHIFT) & 0xFF, (e >> ARM_SHIFT) & 0xFFFF,
             e & ((1 << ITER_BITS) - 1)) for e in events[: offsets[n]]]


def decode_vectorised(events, n):
    e = events[:n]
    return (e >> STEP_SHIFT, (e >> KIND_SHIFT) & 0xFF, (e >> ARM_SHIFT) & 0xFFFF,
            e & ((1 << ITER_BITS) - 1))


def aggregate_kinds(events, n):
    return np.bincount((events[:n] >> KIND_SHIFT) & 0xFF)


if __name__ == "__main__":
    n, steps = 100_000, 24
    events = np.empty(n * steps, np.int64)
    offsets = np.empty(n + 1, np.int64)
    capture(n, steps, events, offsets)
    total = n * steps

    print(f"buffer: {total:,} events x 8 bytes = {total * 8 / 1e6:.0f} MB (one int64 envelope)")

    py = timeit(decode_python, events, offsets, n, reps=3)
    print(f"per-event Python decode: {total / py:,.0f} events/s  ({py*1e3:.0f} ms for {total:,} events)")

    vec = timeit(decode_vectorised, events, total)
    print(f"vectorised numpy unpack: {total / vec:,.0f} events/s ({vec*1e3:.0f} ms) -> "
          f"4 int64 columns (step, kind, arm, iteration) in one shift/mask pass")

    # Lazy: keep the raw int64 view and unpack only a slice on demand.
    lazy = timeit(lambda: (events, offsets, n), reps=5)
    print(f"lazy (no decode until a slice is asked): 0 events decoded up front; "
          f"a single record's {steps} events unpack on demand")

    agg = timeit(aggregate_kinds, events, total)
    print(f"aggregation (kind histogram, no per-event decode): {total / agg:,.0f} events/s "
          f"({agg*1e3:.0f} ms) -> counts per kind directly from the int64s")

    # Budget: a 1M-row x 24-step batch is ~24M events; what each decode costs end to end.
    big_n = 1_000_000
    print(f"\nbudget for a {big_n:,}-row x {steps}-step batch (~{big_n * steps * 8 / 1e6:.0f} MB):")
    big = np.empty(big_n * steps, np.int64)
    capture(big_n, steps, big, np.empty(big_n + 1, np.int64))
    for name, f in (("vectorised", decode_vectorised), ("aggregate", aggregate_kinds)):
        t = timeit(f, big, big_n * steps)
        print(f"  {name}: {t*1e3:.0f} ms for {big_n * steps:,} events")
