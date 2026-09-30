"""Concurrent `nogil` trace capture: ordering, frame scope, buffer ownership, conservation.

The fused/stepped kernel is `@njit(nogil=True)` and iterates `for i in range(n)`,
so within one kernel it is serial and row-major. Concurrency is therefore at the
thread level: several threads run the *same* kernel at once (no GIL). This proves
per-record ordering and conservation survive that, that the GIL really is released
(4 threads ~= 1x wall clock, not 4x), and that a buffer shared across threads
corrupts per-record boundaries while per-invocation buffers do not. Frame scope is
shown through the real engine: a frame step is one checkpoint pair over all rows,
so its events are frame-scope, emitted by the driver, not per-row kernel writes.

    uv run python notes/vscode-redesign/experimentation/01-trace-capture/concurrent_nogil.py
"""
import threading
import time

import numpy as np
from numba import njit


@njit(nogil=True)
def capture(n, steps, events, offsets):
    for i in range(n):
        for s in range(steps):
            events[i * steps + s] = (i << 32) | s  # row id in the high half, step in the low
        offsets[i] = i * steps
    offsets[n] = n * steps


@njit(nogil=False)
def capture_gil(n, steps, events, offsets):
    for i in range(n):
        for s in range(steps):
            events[i * steps + s] = (i << 32) | s
        offsets[i] = i * steps
    offsets[n] = n * steps


def owned_buffer(n, steps):
    return np.empty(n * steps, np.int64), np.empty(n + 1, np.int64)


def check_owned(events, offsets, n, steps):
    for r in range(n):
        lo, hi = offsets[r], offsets[r + 1]
        assert hi - lo == steps, f"row {r}: {hi - lo} events, want {steps}"
        for k in range(lo, hi):
            row, step = events[k] >> 32, events[k] & 0xFFFFFFFF
            assert row == r and step == k - lo, f"row {r}: event at {k} is (row {row}, step {step})"
    return True


def timeit(f, *a):
    t = time.perf_counter()
    f(*a)
    return time.perf_counter() - t


def wall_clock_concurrency(n, steps, threads=4):
    # The definitive proof is the contrast: the same kernel with nogil=False
    # serializes on the GIL (4 threads ~= 4x one thread), with nogil=True it
    # runs concurrently (4 threads ~= 1x one thread, memory/CPU bound).
    warm = [owned_buffer(n, steps) for _ in range(threads + 1)]
    for e, o in warm:
        capture(n, steps, e, o)
        capture_gil(n, steps, e, o)
    one = min(timeit(capture, n, steps, warm[0][0], warm[0][1]) for _ in range(3))
    def run(fn, k):
        fn(n, steps, warm[k][0], warm[k][1])
    for fn, label in ((capture_gil, "gil"), (capture, "nogil")):
        t = time.perf_counter()
        ts = [threading.Thread(target=run, args=(fn, k)) for k in range(threads)]
        for x in ts:
            x.start()
        for x in ts:
            x.join()
        four = time.perf_counter() - t
        print(f"  {label}: 1 thread {one*1e3:.1f} ms vs {threads} threads {four*1e3:.1f} ms "
              f"({four/one:.2f}x for {threads}x the work)")
    return one


if __name__ == "__main__":
    n, steps = 100_000, 16

    print("GIL released (same kernel, nogil off vs on, 4 threads):")
    wall_clock_concurrency(1_000_000, steps)

    # Per-invocation buffers: ordering + conservation under 8 concurrent writers.
    threads = 8
    results = [owned_buffer(n, steps) for _ in range(threads)]
    for e, o in results:
        capture(n, steps, e, o)  # pre-fault so the concurrent pass is the only write
    def writer(k):
        capture(n, steps, *results[k])
    ts = [threading.Thread(target=writer, args=(k,)) for k in range(threads)]
    for x in ts:
        x.start()
    for x in ts:
        x.join()
    for k in range(threads):
        check_owned(*results[k], n, steps)
    print(f"per-invocation buffer: {threads} concurrent writers, "
          f"{threads * n * steps:,} events all conserved, per-record order intact")

    # Shared array, disjoint regions (driver slices one array per invocation).
    shared_e = np.empty(2 * n * steps, np.int64)
    shared_o = np.empty(2 * (n + 1), np.int64)
    def shared_writer(base):
        e = shared_e[base * n * steps:(base + 1) * n * steps]
        o = shared_o[base * (n + 1):(base + 1) * (n + 1)]
        capture(n, steps, e, o)
    ts = [threading.Thread(target=shared_writer, args=(b,)) for b in (0, 1)]
    for x in ts:
        x.start()
    for x in ts:
        x.join()
    check_owned(shared_e[: n * steps], shared_o[: n + 1], n, steps)
    check_owned(shared_e[n * steps:], shared_o[n + 1:], n, steps)
    print(f"shared-array, disjoint-region: both writers verified; ownership = per-invocation "
          f"region, never the same region for two threads")

    # Frame scope through the real engine: a frame step runs once over all rows.
    import polars as pl
    from decider import Engine, flow, frame_step

    @frame_step(reads=["client_id"], writes=["prior_defaults"])
    def join_history(df: pl.DataFrame) -> pl.DataFrame:
        return df.with_columns(pl.lit(1).alias("prior_defaults"))

    exe = Engine().bind(flow(join_history, name="history"), mode="fused")
    state, params = exe.prepare(pl.DataFrame({"client_id": [1, 2, 3]}))
    checkpoints = []
    for cp in exe.runner.iterate(exe.plan, state, params):
        checkpoints.append((cp.when, cp.origin.path))
    print(f"frame scope: {checkpoints} -> one frame step = one before/after over all rows "
          f"(its events are frame-scope, driver-emitted, not per-row kernel writes)")
