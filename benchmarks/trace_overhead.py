"""Measure the shipped decision-trace overhead: batch throughput and single-record
latency with and without a `TraceSink`, plus the event payload per record.

Run with `uv run python benchmarks/trace_overhead.py`. One number feeds
`notes/vscode-redesign/limits-and-responsibilities.md`: tracing is opt-in because
materialising the trace costs real time on the request path.
"""
import time

import polars as pl

from decider import flow, param
from decider.engine import Engine
from decider.engine.trace import TraceSink


def ratio(income: float, debt: float) -> float:
    return debt / income


def approved(ratio: float, limit: float = param(0.4)) -> bool:
    return ratio <= limit


def tier(approved: bool) -> int:
    return 1 if approved else 0


PIPELINE = flow(ratio, approved, tier, name="credit")
ROWS = 1_000_000
df = pl.DataFrame({"income": [1000.0] * ROWS, "debt": [float(i % 900) for i in range(ROWS)]})


def timed(fn):
    t = time.perf_counter()
    result = fn()
    return time.perf_counter() - t, result


def main():
    exe = Engine().bind(PIPELINE, mode="fused")
    exe.run(pl.DataFrame({"income": [1000.0], "debt": [200.0]}))  # warm the kernel

    _, _ = timed(lambda: exe.run(df))
    batch_no_trace, _ = timed(lambda: exe.run(df))
    batch_trace, _ = timed(lambda: exe.run(df, trace=TraceSink()))

    sink = TraceSink()
    score_no_trace, _ = timed(lambda: exe.score({"income": 1000.0, "debt": 200.0}))
    score_trace, _ = timed(lambda: exe.score({"income": 1000.0, "debt": 200.0}, trace=sink))

    events = sink.events()
    per_record = len(events)

    print(f"batch {ROWS:,} rows: no trace {batch_no_trace * 1000:.1f} ms  "
          f"trace {batch_trace * 1000:.1f} ms  ({ROWS / batch_trace:,.0f} rows/s)")
    print(f"score: no trace {score_no_trace * 1e6:.1f} us  trace {score_trace * 1e6:.1f} us")
    print(f"events/record: {per_record}  decoded event payload: {sum(len(e.origin.path) if e.origin else 0 for e in events) + per_record * 32:,} bytes est.")


if __name__ == "__main__":
    main()
