"""`Branch`/`Loop` past any hand-written arity — the migration that removed
the last generated Python source from `decider2.graph.control_flow` must
not have replaced the old `CALL_ARITY = 6` ceiling with a new one.

Every callee below is module-level (a `Branch`/`Loop` callee is `njit(
cache=True)`-wrapped like any other step, so it needs a real module-level
identity for numba's cache) and every argument is referenced in the body
(`decider2.params._check_every_input_is_referenced`). A 32-leaf condition,
a 32-leaf + 8-param arm and a 32-leaf + 8-param loop body are well past
anything a per-count closure family would have been written for.

Two more things this file pins that the migration changed the mechanism
of, on purpose:

- **argument types are honest across the boundary.** A callee declared
  `band: int` receives a real int64 (it indexes a tuple with it, which
  numba refuses for a float64), and a `flag: bool` a real bool — the
  register array is float64, the adapter casts each argument back to the
  callee's own declared type.
- **the one shape the row-gather cannot hold fails at build time**, with
  the `compile/` change named: a construct output reading a `str` leaf
  alongside another input (`_engine.boundary_inputs`).
"""
from __future__ import annotations

import polars as pl
import pytest

from decider2 import Branch, Loop, flow, module, param, step
from decider2.testing import assert_equivalent

# ---------------------------------------------------------------------------
# 32 leaves, 8 params — Branch
# ---------------------------------------------------------------------------


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


def _wide_frame(n: int = 5) -> pl.DataFrame:
    cols = {f"a{k}": [float((k * 7 + r * 3) % 5) for r in range(n)] for k in range(32)}
    cols["total"] = [float(r) for r in range(n)]
    return pl.DataFrame(cols)


def _expected_wide(frame: pl.DataFrame) -> list[float]:
    out = []
    for row in frame.iter_rows(named=True):
        a = [row[f"a{k}"] for k in range(32)]
        if sum(a) > 16.0:
            out.append(row["total"] + sum((j + 1) * sum(a[j::8]) for j in range(8)))
        else:
            out.append(-row["total"])
    return out


def test_branch_with_32_leaves_and_8_params_has_no_arity_ceiling():
    branch = Branch(
        wide_cond, module(wide_weighted), module(wide_negate),
        modifies=["total"], name="wide_branch",
    )
    # 33 inputs on the `total` step (path + total + 32 leaves), 32 on the
    # path step, 9 params — all past the old 6-argument ceiling.
    by_name = {s.name: s for s in branch.steps}
    assert len(by_name["wide_branch_path"].inputs) == 32
    assert len(by_name["total"].inputs) == 34 and len(by_name["total"].params) == 9
    frame = _wide_frame()
    pipeline = flow(branch)
    out = pipeline.apply(frame, mode="interpreted")
    assert out["total"].to_list() == pytest.approx(_expected_wide(frame))
    assert_equivalent(pipeline, frame)


# ---------------------------------------------------------------------------
# 32 leaves, 8 params — Loop body
# ---------------------------------------------------------------------------


def wide_should_continue(acc: float, loop_idx: int, target: float = param(400.0)) -> bool:
    return acc < target and loop_idx < 1000


@step(output="acc")
def wide_accumulate(
    acc: float, loop_idx: int,
    a0: float, a1: float, a2: float, a3: float, a4: float, a5: float, a6: float, a7: float,
    a8: float, a9: float, a10: float, a11: float, a12: float, a13: float, a14: float, a15: float,
    a16: float, a17: float, a18: float, a19: float, a20: float, a21: float, a22: float, a23: float,
    a24: float, a25: float, a26: float, a27: float, a28: float, a29: float, a30: float, a31: float,
    w0: float = param(1.0), w1: float = param(2.0), w2: float = param(3.0), w3: float = param(4.0),
    w4: float = param(5.0), w5: float = param(6.0), w6: float = param(7.0), w7: float = param(8.0),
) -> float:
    return acc + 1.0 + 0.0 * loop_idx + (
        w0 * (a0 + a8 + a16 + a24) + w1 * (a1 + a9 + a17 + a25) + w2 * (a2 + a10 + a18 + a26)
        + w3 * (a3 + a11 + a19 + a27) + w4 * (a4 + a12 + a20 + a28) + w5 * (a5 + a13 + a21 + a29)
        + w6 * (a6 + a14 + a22 + a30) + w7 * (a7 + a15 + a23 + a31)
    )


def test_loop_body_with_32_leaves_and_8_params_has_no_arity_ceiling():
    loop = Loop(
        wide_should_continue, module(wide_accumulate),
        carries=["acc"], max_iterations=50, name="wide_loop",
    )
    (acc_step,) = loop.steps
    assert len(acc_step.inputs) == 33 and len(acc_step.params) == 9
    frame = _wide_frame().rename({"total": "acc"})
    pipeline = flow(loop)
    out = pipeline.apply(frame, mode="interpreted")
    expected = []
    for row in frame.iter_rows(named=True):
        a = [row[f"a{k}"] for k in range(32)]
        per_iter = 1.0 + sum((j + 1) * sum(a[j::8]) for j in range(8))
        acc, i = row["acc"], 0
        while acc < 400.0 and i < 50:
            acc += per_iter
            i += 1
        expected.append(acc)
    assert out["acc"].to_list() == pytest.approx(expected)
    assert out["acc"].to_list() != [row["acc"] for row in frame.iter_rows(named=True)]  # it iterated
    assert_equivalent(pipeline, frame)


# ---------------------------------------------------------------------------
# argument types survive the float64 register array
# ---------------------------------------------------------------------------


def band_of(band: int) -> int:
    return band


@step(output="price")
def price_by_table(price: float, band: int, flag: bool, bump: float = param(0.5)) -> float:
    # A tuple indexed by `band` needs a REAL int (numba refuses float64
    # getitem); `flag` is used as a bool.
    base = (10.0, 20.0, 30.0)[band]
    return price + base + (bump if flag else 0.0)


@step(output="price")
def price_flat(price: float, flag: bool) -> float:
    return price + (1.0 if flag else 2.0)


def test_callee_receives_its_own_declared_types():
    branch = Branch(
        band_of, [module(price_by_table), module(price_flat), module(price_flat)],
        modifies=["price"], name="typed_branch",
    )
    frame = pl.DataFrame({
        "band": [0, 1, 2, 0], "flag": [True, False, True, False], "price": [0.0, 0.0, 0.0, 100.0],
    })
    pipeline = flow(branch).emit("typed_branch_path")
    out = pipeline.apply(frame, mode="interpreted")
    # row 0: band 0 -> table arm, 10 + 0.5 (flag); row 1: band 1 -> flat, +2.0
    # (flag False); row 2: band 2 -> flat, +1.0 (flag True); row 3: band 0,
    # flag False -> table arm, 100 + 10.
    assert out["price"].to_list() == [10.5, 2.0, 1.0, 110.0]
    assert out["typed_branch_path"].to_list() == [0, 1, 2, 0]
    assert_equivalent(pipeline, frame)


# ---------------------------------------------------------------------------
# the one unrepresentable shape fails loudly at build time
# ---------------------------------------------------------------------------


def sector_and_income(sector: str, income: float, private: str = param("private")) -> bool:
    return sector == private and income > 0.0


@step(output="cap")
def cap_a(cap: float) -> float:
    return cap + 1.0


@step(output="cap")
def cap_b(cap: float) -> float:
    return cap + 2.0


def test_str_leaf_alongside_another_input_is_a_named_build_error():
    with pytest.raises(ValueError, match="compile/driver.py change"):
        Branch(sector_and_income, module(cap_a), module(cap_b), modifies=["cap"], name="str_mix")
