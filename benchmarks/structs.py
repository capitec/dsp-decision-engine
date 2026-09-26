"""A `dict` input as an Arrow struct: score() latency and batch throughput.

`uv run python benchmarks/structs.py`
"""
from __future__ import annotations

import statistics
import time
import warnings
from typing import TypedDict

import numpy as np
import polars as pl

from decider import Engine, Struct, flow

N = 200_000
CALLS = 3000


class Applicant(TypedDict):
    income: float
    age: int


def flat(income: float, age: int) -> float:
    return income * 2.0 + age


def as_dict(applicant: dict) -> float:
    return applicant["income"] * 2.0 + applicant["age"]


def as_struct(applicant: Struct[Applicant]) -> float:
    return applicant["income"] * 2.0 + applicant["age"]


def frames() -> tuple[pl.DataFrame, pl.DataFrame]:
    rng = np.random.default_rng(0)
    income = rng.uniform(1000.0, 9000.0, N)
    age = rng.integers(18, 70, N)
    flat_df = pl.DataFrame({"income": income, "age": age})
    struct_df = flat_df.select(pl.struct(["income", "age"]).alias("applicant"))
    return flat_df, struct_df


def batch(exe, df, reps=3) -> float:
    exe.run(df.head(8))
    best = min(_time(lambda: exe.run(df)) for _ in range(reps))
    return N / best


def prepare_ms(exe, df, reps=3) -> float:
    # A nested column's Python dicts are built here, before any step runs, in every mode.
    exe.prepare(df.head(8))
    return min(_time(lambda: exe.prepare(df)) for _ in range(reps)) * 1e3


def _time(fn) -> float:
    t = time.perf_counter()
    fn()
    return time.perf_counter() - t


def latency(cases: list[tuple]) -> list[tuple[float, float, float]]:
    # Interleaved: a busy box drifts every variant the same way, so p50s stay comparable.
    for exe, record in cases:
        for _ in range(200):
            exe.score(record)
    times: list[list[float]] = [[] for _ in cases]
    for _ in range(CALLS):
        for (exe, record), ts in zip(cases, times):
            t = time.perf_counter()
            exe.score(record)
            ts.append((time.perf_counter() - t) * 1e6)
    out = []
    for ts in times:
        ts.sort()
        out.append((statistics.median(ts), ts[int(0.99 * len(ts))], ts[0]))
    return out


def main() -> None:
    flat_df, struct_df = frames()
    flat_record = {"income": 4100.0, "age": 33}
    struct_record = {"applicant": {"income": 4100.0, "age": 33}}
    variants = (
        ("numeric kernel, flat columns (the ceiling)", flat, flat_df, flat_record, "fused"),
        ("`dict`, Python per row (today)", as_dict, struct_df, struct_record, "fused"),
        ("`dict`, mode='interpreted'", as_dict, struct_df, struct_record, "interpreted"),
        ("`Struct[Applicant]` record, in the kernel", as_struct, struct_df, struct_record, "fused"),
        ("`Struct[Applicant]` record, mode='stepped'", as_struct, struct_df, struct_record, "stepped"),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        exes = [Engine().bind(flow(fn, name="p"), mode=mode) for _, fn, _, _, mode in variants]
        latencies = latency([(exe, record) for exe, (*_, record, _) in zip(exes, variants)])
        rows = [(label, batch(exe, df), prepare_ms(exe, df), *lat)
                for (label, _, df, _, _), exe, lat in zip(variants, exes, latencies)]
    print(f"{'variant':44} {'rows/s':>10} {'prepare ms':>11} {'p50 us':>8} {'p99 us':>8} {'min us':>8}")
    for label, rps, prep, p50, p99, lo in rows:
        print(f"{label:44} {rps / 1e6:9.2f}M {prep:11.1f} {p50:8.1f} {p99:8.1f} {lo:8.1f}")


if __name__ == "__main__":
    main()
