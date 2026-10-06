from __future__ import annotations

import polars as pl

from decider import flow, step
from decider.lab import checks


def _run(pipeline, alias, frame):
    return checks.run(pipeline, [alias], data=frame).findings


def test_nan_output_reports_nan_and_inf_at_the_producer():
    @step(output="ratio")
    def ratio(debt: float, income: float) -> float:
        return debt / income if income else float("nan")

    @step(output="inv")
    def inv(debt: float) -> float:
        return float("inf") if debt > 5 else debt

    frame = pl.DataFrame({"debt": [1.0, 9.0], "income": [0.0, 2.0]})
    found = {f.detail["column"]: f for f in _run(flow(ratio, inv, name="f"), "nan_output", frame)}
    assert found["ratio"].detail["nans"] == 1 and found["ratio"].step.path == "f/ratio"
    assert found["inv"].detail["infs"] == 1 and found["inv"].code == "DCD502"


def test_nan_output_is_quiet_on_finite_outputs_and_skipped_without_data():
    @step(output="y")
    def double(x: float) -> float:
        return x * 2

    pipeline = flow(double, name="f")
    assert _run(pipeline, "nan_output", pl.DataFrame({"x": [1.0, 2.0]})) == ()
    assert [s.check for s in checks.run(pipeline, ["nan_output"]).skipped] == ["nan_output"]


def test_int64_range_flags_a_product_that_wraps_on_the_observed_range():
    @step(output="total")
    def total(cents: int, rate: int) -> int:
        scaled = cents * rate
        return scaled + 1

    frame = pl.DataFrame({"cents": [10, 5_000_000_000_000], "rate": [3, 4_000_000]})
    f, *downstream = _run(flow(total, name="f"), "int64_range", frame)
    assert f.code == "DCD503" and f.detail["expression"] == "cents * rate" and f.line > 0
    assert f.detail["hi"] == 5_000_000_000_000 * 4_000_000
    assert [d.detail["expression"] for d in downstream] == ["scaled + 1"]


def test_int64_range_is_quiet_when_the_data_stays_small_or_the_shape_is_unknown():
    @step(output="total")
    def total(cents: int, rate: int) -> int:
        return cents * rate

    @step(output="t2")
    def t2(big: int) -> int:
        return abs(big) * 10**12

    @step(output="t3")
    def t3(big: int) -> int:
        out = big
        if big > 0:
            out = big * big
        return out

    small = pl.DataFrame({"cents": [10, 20], "rate": [3, 4], "big": [2**40, 3]})
    assert _run(flow(total, name="f"), "int64_range", small) == ()
    assert _run(flow(t2, name="f"), "int64_range", small) == ()


def test_int64_range_flags_float_accumulator_terms_beyond_2_53():
    @step(output="acc")
    def acc(amount: int) -> float:
        total = 0.0
        total += amount * 1.0
        return total

    frame = pl.DataFrame({"amount": [2**60]})
    (f,) = _run(flow(acc, name="f"), "int64_range", frame)
    assert f.detail["expression"] == "amount * 1.0"


def test_int64_range_in_branch_is_still_evaluated():
    @step(output="y")
    def y(a: int) -> int:
        if a > 0:
            return a * a
        return 0

    (f,) = _run(flow(y, name="f"), "int64_range", pl.DataFrame({"a": [2**40]}))
    assert f.detail["expression"] == "a * a"


def test_constant_output_reports_a_column_with_one_value():
    @step(output="band")
    def band(score: float) -> str:
        return "low" if score < 10 else "high"

    pipeline = flow(band, name="f")
    (f,) = _run(pipeline, "constant_output", pl.DataFrame({"score": [1.0, 2.0, 3.0]}))
    assert f.code == "DCD506" and f.detail["value"] == "low" and f.step.path == "f/band"
    assert _run(pipeline, "constant_output", pl.DataFrame({"score": [1.0, 20.0]})) == ()
    assert _run(pipeline, "constant_output", pl.DataFrame({"score": [1.0]})) == ()


def test_value_checks_are_registered_opt_in():
    rows = {c["alias"]: c for c in checks.catalogue()}
    for alias, code in [("nan_output", "DCD502"), ("int64_range", "DCD503"), ("constant_output", "DCD506")]:
        assert rows[alias]["code"] == code and rows[alias]["needs_data"]
        assert alias not in checks.standard
    assert rows["nan_output"]["executes"] and not rows["int64_range"]["executes"]
