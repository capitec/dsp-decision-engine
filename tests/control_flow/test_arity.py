"""Branches and loops over many inputs and params, and mixed argument types, in every mode."""
import polars as pl
import pytest

from decider import branch, loop, param, step
from decider.engine import Engine
from decider.testing import assert_equivalent


def wide_cond(
    a0: float, a1: float, a2: float, a3: float, a4: float, a5: float, a6: float, a7: float,
    a8: float, a9: float, a10: float, a11: float, a12: float, a13: float, a14: float, a15: float,
    a16: float, a17: float, a18: float, a19: float, a20: float, a21: float, a22: float, a23: float,
    a24: float, a25: float, a26: float, a27: float, a28: float, a29: float, a30: float, a31: float,
) -> bool:
    total = (
        a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8 + a9 + a10 + a11 + a12 + a13 + a14 + a15
        + a16 + a17 + a18 + a19 + a20 + a21 + a22 + a23 + a24 + a25 + a26 + a27 + a28 + a29 + a30 + a31
    )
    return total > 16.0


@step(output="total")
def wide_weighted(
    total: float,
    a0: float, a1: float, a2: float, a3: float, a4: float, a5: float, a6: float, a7: float,
    a8: float, a9: float, a10: float, a11: float, a12: float, a13: float, a14: float, a15: float,
    a16: float, a17: float, a18: float, a19: float, a20: float, a21: float, a22: float, a23: float,
    a24: float, a25: float, a26: float, a27: float, a28: float, a29: float, a30: float, a31: float,
    w0: float = param(1.0), w1: float = param(2.0), w2: float = param(3.0), w3: float = param(4.0),
    w4: float = param(5.0), w5: float = param(6.0), w6: float = param(7.0), w7: float = param(8.0),
) -> float:
    return total + (
        w0 * (a0 + a8 + a16 + a24) + w1 * (a1 + a9 + a17 + a25) + w2 * (a2 + a10 + a18 + a26)
        + w3 * (a3 + a11 + a19 + a27) + w4 * (a4 + a12 + a20 + a28) + w5 * (a5 + a13 + a21 + a29)
        + w6 * (a6 + a14 + a22 + a30) + w7 * (a7 + a15 + a23 + a31)
    )


@step(output="total")
def wide_negate(total: float, scale: float = param(-1.0)) -> float:
    return total * scale


def wide_frame(n=5):
    cols = {f"a{k}": [float((k * 7 + r * 3) % 5) for r in range(n)] for k in range(32)}
    return pl.DataFrame(cols | {"total": [float(r) for r in range(n)]})


def weighted(a):
    return sum((j + 1) * sum(a[j::8]) for j in range(8))


def test_a_branch_with_32_inputs_and_9_params():
    frame = wide_frame()
    wide = branch(wide_cond, wide_weighted, wide_negate, modifies=["total"], name="wide")
    expected = []
    for row in frame.iter_rows(named=True):
        a = [row[f"a{k}"] for k in range(32)]
        expected.append(row["total"] + weighted(a) if sum(a) > 16.0 else -row["total"])
    assert assert_equivalent(wide, frame)["total"].to_list() == pytest.approx(expected)


def below_target(acc: float, target: float = param(400.0)) -> bool:
    return acc < target


@step(output="acc")
def wide_accumulate(
    acc: float,
    a0: float, a1: float, a2: float, a3: float, a4: float, a5: float, a6: float, a7: float,
    a8: float, a9: float, a10: float, a11: float, a12: float, a13: float, a14: float, a15: float,
    a16: float, a17: float, a18: float, a19: float, a20: float, a21: float, a22: float, a23: float,
    a24: float, a25: float, a26: float, a27: float, a28: float, a29: float, a30: float, a31: float,
    w0: float = param(1.0), w1: float = param(2.0), w2: float = param(3.0), w3: float = param(4.0),
    w4: float = param(5.0), w5: float = param(6.0), w6: float = param(7.0), w7: float = param(8.0),
) -> float:
    return acc + 1.0 + (
        w0 * (a0 + a8 + a16 + a24) + w1 * (a1 + a9 + a17 + a25) + w2 * (a2 + a10 + a18 + a26)
        + w3 * (a3 + a11 + a19 + a27) + w4 * (a4 + a12 + a20 + a28) + w5 * (a5 + a13 + a21 + a29)
        + w6 * (a6 + a14 + a22 + a30) + w7 * (a7 + a15 + a23 + a31)
    )


def test_a_loop_body_with_33_inputs_and_8_params():
    frame = wide_frame().rename({"total": "acc"})
    wide = loop(below_target, wide_accumulate, carries=["acc"], max_iterations=50, name="wide")
    expected = []
    for row in frame.iter_rows(named=True):
        per_iteration = 1.0 + weighted([row[f"a{k}"] for k in range(32)])
        acc, i = row["acc"], 0
        while acc < 400.0 and i < 50:
            acc, i = acc + per_iteration, i + 1
        expected.append(acc)
    assert assert_equivalent(wide, frame)["acc"].to_list() == pytest.approx(expected)


def sector_and_income(sector: str, income: float, private: str = param("private")) -> bool:
    return sector == private and income > 0.0


@step(output="cap")
def cap_a(cap: float) -> float:
    return cap + 1.0


@step(output="cap")
def cap_b(cap: float) -> float:
    return cap + 2.0


def test_a_condition_reading_a_str_input_alongside_others_packs():
    frame = pl.DataFrame({"sector": ["private", "public", "private"], "income": [1.0, 1.0, 0.0],
                          "cap": [10.0, 10.0, 10.0]})
    mixed = branch(sector_and_income, cap_a, cap_b, modifies=["cap"], name="mixed")
    assert assert_equivalent(mixed, frame)["cap"].to_list() == [11.0, 12.0, 12.0]
    exe = Engine().bind(mixed, "fused")
    exe.run(frame)
    assert list(exe.runner.packed) == ["mixed"]
