"""What each codec costs in-process to turn one record's five floats into values.

The transport round trips are in `benchmarks/serving_transports.py`; run both
back to back for numbers from one session.

    uv run --with pyarrow python benchmarks/arrow_codec.py [--calls 2000]
"""
import os
import struct
import sys
import time

import numpy as np
import polars as pl
from pydantic_core import from_json

sys.path.insert(0, "benchmarks")
from serving_transports import BODIES, FLAGSHIP_ROW, NAMES, _pct  # noqa: E402

REPS = 3
WARM = 500


def pa_costs(calls):
    """In-process pyarrow IPC costs: a schema parsed per call against a persistent reader."""
    import pyarrow as pa

    schema = pa.schema([(n, pa.float64()) for n in NAMES])
    batch = pa.record_batch([[float(FLAGSHIP_ROW[n])] for n in NAMES], schema=schema)
    sink = pa.BufferOutputStream()
    w = pa.ipc.new_stream(sink, schema)
    w.write_batch(batch)
    w.close()
    whole = sink.getvalue()
    # A long stream read by one reader: the schema is parsed once, then only batches.
    many = pa.BufferOutputStream()
    w2 = pa.ipc.new_stream(many, schema)
    batches = REPS * (calls + WARM) + 50
    for _ in range(batches):
        w2.write_batch(batch)
    w2.close()
    persistent = pa.ipc.open_stream(pa.BufferReader(many.getvalue()))
    rfd, wfd = os.pipe()
    wf, rf = open(wfd, "wb"), open(rfd, "rb", buffering=0)
    writer = pa.ipc.new_stream(wf, schema)
    # `open_stream` on a pipe blocks on the schema message alone: prime it with a batch.
    writer.write_batch(batch)
    wf.flush()
    reader = pa.ipc.open_stream(rf)
    reader.read_next_batch()

    def through_pipe():
        writer.write_batch(batch)
        wf.flush()
        return reader.read_next_batch()

    rows = [
        (f"pyarrow read_next_batch, schema once ({len(many.getvalue()) // batches} B/batch)",
         persistent.read_next_batch),
        (f"pyarrow open_stream + read_next_batch, schema per call ({len(whole)} B)",
         lambda: pa.ipc.open_stream(whole).read_next_batch()),
        ("pyarrow over a pipe: write_batch + flush + read_next_batch", through_pipe),
        ("pl.read_ipc_stream (polars, schema per call)", lambda: pl.read_ipc_stream(BODIES["arrow"])),
        ("-- then getting the 5 values out of a batch --", lambda: None),
        ("batch.to_pydict()", batch.to_pydict),
        ("[batch.column(i)[0].as_py() for i in range(5)]",
         lambda: [batch.column(i)[0].as_py() for i in range(5)]),
        ("np.frombuffer of each column buffer (no copy)",
         lambda: [np.frombuffer(c.buffers()[1], np.float64, 1) for c in batch.columns]),
        ("-- what a fixed layout costs instead --", lambda: None),
        ("struct.unpack_from('<5d', body)", lambda: struct.unpack_from("<5d", BODIES["values"])),
        ("np.frombuffer(body, np.float64, 5)", lambda: np.frombuffer(BODIES["values"], np.float64, 5)),
        ("pydantic_core.from_json(json body)", lambda: from_json(BODIES["json"])),
    ]
    print(f"in-process, flagship's 5 floats, {calls} calls (min of {REPS} reps, µs)")
    for label, fn in rows:
        if label.startswith("--"):
            print(f"  {label}", flush=True)
            continue
        best = None
        for _ in range(REPS):
            for _ in range(WARM):
                fn()
            samples = np.empty(calls)
            perf = time.perf_counter
            for k in range(calls):
                t = perf()
                fn()
                samples[k] = perf() - t
            p = _pct(samples)
            best = p if best is None else (min(best[0], p[0]), min(best[1], p[1]))
        print(f"  {label:<62}{best[0]:>9.2f}{best[1]:>9.2f}", flush=True)
    print()


if __name__ == "__main__":
    n = int(sys.argv[sys.argv.index("--calls") + 1]) if "--calls" in sys.argv else 2000
    pa_costs(n)
