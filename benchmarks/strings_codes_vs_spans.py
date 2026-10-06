"""`Raw[str]` dictionary codes against `Raw[bytes]` spans, at low and high cardinality.

Both compare a string column for equality inside a fused kernel. A code first looks
every value up in the process-wide code table; a span reads the Arrow bytes in place.
`first` is the first run over a frame of strings the code table has never seen.

    uv run python benchmarks/strings_codes_vs_spans.py
"""
import gc
import random
import statistics
import time

import polars as pl

from decider import Raw, flow, param, raw_str

N = 1_000_000
MODE = "fused"
PRIVATE = raw_str("private")


def code_literal(sector: Raw[str]) -> bool:
    return sector == PRIVATE


def span_literal(sector: Raw[bytes]) -> bool:
    return sector == "private"


def code_param(sector: Raw[str], want: Raw[str] = param("private")) -> bool:
    return sector == want


def span_param(sector: Raw[bytes], want: str = param("private")) -> bool:
    return sector == want


def frame(distinct: int, tag: str) -> pl.DataFrame:
    values = ["private"] + [f"{tag}-{i:07d}" for i in range(distinct - 1)]
    return pl.DataFrame({"sector": random.choices(values, k=N)})


def measure(fn, distinct: int) -> tuple[float, float, float]:
    exe = flow(fn, name="p").compile(MODE)
    exe.run(frame(4, "warm").head(10))
    df = frame(distinct, f"{fn.__name__}-{distinct}")
    start = time.perf_counter()
    exe.run(df)
    first = N / (time.perf_counter() - start)
    best = 0.0
    for _ in range(3):
        gc.collect()
        start = time.perf_counter()
        exe.run(df)
        best = max(best, N / (time.perf_counter() - start))
    record = {"sector": "private"}
    for _ in range(500):
        exe.score(record)
    lat = []
    for _ in range(5000):
        start = time.perf_counter()
        exe.score(record)
        lat.append(time.perf_counter() - start)
    return first, best, statistics.median(lat)


if __name__ == "__main__":
    random.seed(0)
    print(f"{'shape':14} {'distinct':>9} {'first M/s':>10} {'batch M/s':>10} {'score p50 us':>13}")
    for distinct in (4, 1_000, 100_000):
        for fn in (code_literal, span_literal, code_param, span_param):
            first, best, p50 = measure(fn, distinct)
            print(f"{fn.__name__:14} {distinct:>9} {first / 1e6:>10.1f} {best / 1e6:>10.1f} {p50 * 1e6:>13.1f}")
