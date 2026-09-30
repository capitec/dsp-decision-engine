"""Batch trace overhead, separate from score(): opt-in / sampling / limit policy.

The first spike already showed one flat int64 trace buffer halves *batch*
throughput (materialisation-bound) while costing <3% on single-record `score()`.
This re-measures the two paths side by side and quantifies the policy the batch
workload needs: tracing off by default, a sampling rate, and a per-run event cap
that bounds memory, with dropped/sampled events reported rather than silent.

    uv run python notes/vscode-redesign/experimentation/01-trace-capture/batch_overhead.py
"""
import gc
import time

import numpy as np
import polars as pl

from decider import Engine, flow, step

N = 300_000
rng = np.random.default_rng(0)


def add(x):
    return x + 1.0


def build_chain(with_trace):
    if with_trace:
        # One shared int64 trace column for the whole chain (the recommended single-buffer design),
        # not one column per step (which the first spike showed is 6x slower).
        steps = [step(add, output="v0").named("a0")]
        for k in range(1, 10):
            steps.append(step(add, output=f"v{k}").relabel(reads={"x": f"v{k-1}"}).named(f"a{k}"))
        steps.append(step(lambda v9: int(v9), output="trace").named("tr"))
        return flow(*steps, name="chain")
    steps = [step(add, output="v0").named("a0")]
    for k in range(1, 10):
        steps.append(step(add, output=f"v{k}").relabel(reads={"x": f"v{k-1}"}).named(f"a{k}"))
    return flow(*steps, name="chain")


def rows_per_s(exe, frame, reps=7):
    exe.run(frame)
    best = min((lambda t: (t[1] - t[0]))(_once(exe.run, frame)) for _ in range(reps))
    return N / best


def _once(f, *a):
    t = time.perf_counter()
    f(*a)
    return (t, time.perf_counter())


def latency(exe, row, calls=20000):
    for _ in range(500):
        exe.score(row)
    gc.collect()
    s = sorted((lambda t: t[1] - t[0])(_once(exe.score, row)) for _ in range(calls))
    return s[len(s) // 2] * 1e6, s[int(len(s) * 0.99)] * 1e6


def budget(rows, steps_per_row, sample_rate, event_cap):
    """Memory and whether the cap forces extra sampling, given a sampling rate and an event limit."""
    events = int(rows * steps_per_row * sample_rate)
    capped = events > event_cap
    kept = event_cap if capped else events
    return kept * 8, events - kept  # bytes, dropped events


if __name__ == "__main__":
    frame = pl.DataFrame({"x": rng.uniform(0, 1000, N)})
    row = frame.row(0, named=True)

    for label, pipeline in (("batch no trace", build_chain(False)), ("batch +1 int64 trace col", build_chain(True))):
        exe = Engine().bind(pipeline, mode="fused")
        rps = rows_per_s(exe, frame)
        p50, p99 = latency(exe, row)
        print(f"{label:<26} {rps:>14,.0f} rows/s   score p50 {p50:.1f} us")

    # Policy: opt-in (off by default), sampling rate, per-run event cap -> bounded memory + reported drops.
    steps_per_row = 10
    for rate, cap in ((1.0, None), (0.01, None), (1.0, 10_000_000)):
        rows = 1_000_000
        mem, dropped = budget(rows, steps_per_row, rate, cap if cap is not None else 10 ** 12)
        cap_txt = f"cap {cap:,}" if cap else "uncapped"
        print(f"sample rate {rate}, {cap_txt}: {rows:,} rows x {steps_per_row} steps -> "
              f"{mem / 1e6:.0f} MB kept, {dropped:,} events dropped (reported in manifest)")
