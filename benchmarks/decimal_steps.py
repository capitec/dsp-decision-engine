"""Warm `run(record)` and batch cost of a Decimal money flow against the same flow on plain int cents.

    uv run python benchmarks/decimal_steps.py [--reps 5]

See docs/design_decisions/experiments/money-types-and-rounding.md.
"""
import sys
import time

import numpy as np
import polars as pl

from decikit import flow, param
from decikit.ops import decimal
from decikit.ops.decimal import Decimal, Rounding

sys.path.insert(0, "benchmarks")
from serving_latency import stats  # noqa: E402


def instalment_int(principal: int, rate: float = param(0.1275), months: int = param(12)) -> int:
    return int(decimal.round(principal * rate / months, 0, Rounding.HALF_EVEN))


def fee_int(instalment_int: int, floor: int = param(550)) -> int:
    return max(instalment_int, floor) + 125


def total_int(fee_int: int, months: int = param(12)) -> int:
    return fee_int * months


def instalment_dec(principal: Decimal[18, 2], rate: float = param(0.1275), months: int = param(12)) -> Decimal[18, 2]:
    return principal * rate / months


def fee_dec(instalment_dec: Decimal[18, 2], floor: Decimal[18, 2] = param(Decimal("5.50")),
            charge: Decimal[18, 2] = param(Decimal("1.25"))) -> Decimal[18, 2]:
    return max(instalment_dec, floor) + charge


def total_dec(fee_dec: Decimal[18, 2], months: int = param(12)) -> Decimal[18, 2]:
    return fee_dec * months


def main(reps: int) -> None:
    plain = flow(instalment_int, fee_int, total_int, name="loan").compile("fused")
    dec = flow(instalment_dec, fee_dec, total_dec, name="loan").compile("fused")
    row_int, row_dec = {"principal": 100050}, {"principal": Decimal("1000.50")}
    n = 200_000
    rng = np.random.default_rng(1)
    cents = rng.integers(10_000, 10_000_000, n)
    frame_int = pl.DataFrame({"principal": cents})
    frame_dec = pl.DataFrame({"principal": pl.Series(cents).cast(pl.Decimal(18, 0)) / 100}).cast({"principal": pl.Decimal(18, 2)})
    assert plain.fallbacks() == {} and dec.fallbacks() == {}
    print("run(record):", plain.run(row_int)["total_int"], dec.run(row_dec)["total_dec"])
    assert plain.run(row_int)["total_int"] == dec.run(row_dec)["total_dec"].raw
    best = {"int cents": [1e9, 1e9, 0.0], "decimal": [1e9, 1e9, 0.0]}
    for _ in range(reps):
        for name, exe, row, frame in (("int cents", plain, row_int, frame_int), ("decimal", dec, row_dec, frame_dec)):
            p50, p99 = stats(lambda: exe.run(row))
            exe.run(frame)
            t = time.perf_counter()
            for _ in range(5):
                exe.run(frame)
            rate = 5 * n / (time.perf_counter() - t) / 1e6
            best[name] = [min(best[name][0], p50), min(best[name][1], p99), max(best[name][2], rate)]
    for name, (p50, p99, rate) in best.items():
        print(f"{name:10s} run(record) p50 {p50:6.1f} us  p99 {p99:6.1f} us   run() {rate:6.1f} M rows/s")


if __name__ == "__main__":
    main(int(sys.argv[sys.argv.index("--reps") + 1]) if "--reps" in sys.argv else 5)
