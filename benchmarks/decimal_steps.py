"""Warm `score()` cost of Decimal arithmetic in a step against the same steps on plain ints.

The Decimal steps wrap their int64 inputs at the top of the step (`W(x)`), the way a
compiled step types a Decimal at its edge, so columns and params stay plain int64.

    uv run python benchmarks/decimal_steps.py [--reps 5]
"""
import sys

import numpy as np

from decider import flow, param
from decider._engine.compile.semantics.decimals import wrapper
from decider.ops import decimal
from decider.ops.decimal import Rounding

sys.path.insert(0, "benchmarks")
from serving_latency import stats  # noqa: E402

W = wrapper(2)


def net_plain(gross: int, tax: int) -> int:
    return gross - tax


def fee_plain(net_plain: int, rate: float = param(0.0175)) -> int:
    return int(decimal.round(net_plain * rate, 0, Rounding.HALF_EVEN))


def split_plain(net_plain: int, fee_plain: int, parts: int = param(3)) -> int:
    return int(decimal.round((net_plain - fee_plain) / parts, 0, Rounding.HALF_EVEN))


def net_dec(gross: int, tax: int) -> int:
    return (W(gross) - W(tax)).raw


def fee_dec(net_dec: int, rate: float = param(0.0175)) -> int:
    return (W(net_dec) * rate).raw


def split_dec(net_dec: int, fee_dec: int, parts: int = param(3)) -> int:
    return ((W(net_dec) - W(fee_dec)) / parts).raw


ROW = {"gross": 1234567, "tax": 123456}


def main(reps: int) -> None:
    plain = flow(net_plain, fee_plain, split_plain).compile("fused")
    dec = flow(net_dec, fee_dec, split_dec).compile("fused")
    assert plain.score(ROW)["split_plain"] == dec.score(ROW)["split_dec"], (plain.score(ROW), dec.score(ROW))
    best = {"plain": [1e9, 1e9], "decimal": [1e9, 1e9]}
    for _ in range(reps):
        for name, exe, key in (("plain", plain, "split_plain"), ("decimal", dec, "split_dec")):
            p50, p99 = stats(lambda: exe.score(ROW))
            best[name] = [min(best[name][0], p50), min(best[name][1], p99)]
    for name, (p50, p99) in best.items():
        print(f"{name:8s} score() p50 {p50:6.1f} us  p99 {p99:6.1f} us")


if __name__ == "__main__":
    main(int(sys.argv[sys.argv.index("--reps") + 1]) if "--reps" in sys.argv else 5)
