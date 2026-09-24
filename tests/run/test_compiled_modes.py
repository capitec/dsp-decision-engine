"""Stepped and fused modes: one answer with interpreted, one kernel for run and score, no recompiles."""
import polars as pl
import pytest

from decider import branch, flow, missing_as, param, step
from decider.engine import Engine
from decider.engine.compile import Kernel
from decider.engine.ir.decls import Input, Output
from decider.engine.ir.nodes import CallNode
from decider.engine.ir.origin import Origin
from decider.testing import assert_equivalent, no_recompile

MODES = ("interpreted", "stepped", "fused")


def disposable_income(net_income: float, expenses: float) -> float:
    return net_income - expenses


def affordability_ratio(disposable_income: float, instalment: float) -> float:
    return disposable_income / instalment


def cap_by_income_band(term_cap: float, min_net_salary: float, cap: float = param(48.0, ge=6, le=60),
                       income_threshold: float = param(5000.0, ge=0)) -> float:
    return min(term_cap, cap) if min_net_salary < income_threshold else term_cap


flagship = flow(disposable_income, affordability_ratio, cap_by_income_band)
FRAME = pl.DataFrame({
    "net_income":     [9200.0, 4100.0, 15000.0, 4999.0],
    "expenses":       [3100.0, 1500.0,  6000.0, 2000.0],
    "instalment":     [1200.0,  800.0,  2500.0,  700.0],
    "term_cap":       [  60.0,   60.0,    60.0,   60.0],
    "min_net_salary": [9200.0, 4100.0, 15000.0, 4999.0],
})
ROW = {"net_income": 4100.0, "expenses": 1500.0, "instalment": 800.0, "term_cap": 60.0, "min_net_salary": 4100.0}


def test_the_three_modes_agree_exactly_on_the_flagship():
    assert_equivalent(flagship, FRAME)


def test_fused_runs_the_flagship_as_one_kernel_and_stepped_as_one_per_step():
    fused, stepped = (Engine().bind(flagship, mode=m) for m in ("fused", "stepped"))
    fused.run(FRAME)
    stepped.run(FRAME)
    assert len({id(u) for u in fused.runner.units.values()}) == 1
    assert len({id(u) for u in stepped.runner.units.values()}) == 3
    assert all(isinstance(u, Kernel) for u in fused.runner.units.values())


def test_score_and_run_use_the_same_compiled_kernel():
    exe = Engine().bind(flagship, mode="fused")
    batch = exe.run(FRAME)
    (unit,) = {id(u): u for u in exe.runner.units.values()}.values()
    kernel, signatures = unit.fn, len(unit.fn.signatures)
    record = exe.score(ROW)
    assert Engine().bind(flagship, mode="fused").run(FRAME).equals(batch)
    (again,) = {id(u): u for u in exe.runner.units.values()}.values()
    assert again.fn is kernel and len(kernel.signatures) == signatures
    assert batch["cap_by_income_band"][1] == record["cap_by_income_band"] == 48.0


@pytest.mark.parametrize("mode", ["stepped", "fused"])
def test_repeated_calls_do_not_rebuild_the_driver(mode):
    def _sub(x: float, y: float) -> float:
        return x - y

    exe = Engine().bind(flow(_sub), mode=mode)
    frame = pl.DataFrame({"x": [1.0] * 64, "y": [2.0] * 64})
    exe.run(frame)
    exe.score({"x": 1.0, "y": 2.0})
    with no_recompile():
        for _ in range(3):
            exe.run(frame)
            exe.score({"x": 1.0, "y": 2.0})


@pytest.mark.parametrize("mode", ["stepped", "fused"])
def test_retuning_never_recompiles(mode):
    exe = Engine().bind(flagship, mode=mode)
    exe.run(FRAME)
    exe.score(ROW)
    with no_recompile():
        retuned = exe.run(FRAME, params={"cap_by_income_band": {"cap": 24.0}})
        scored = exe.score(ROW, params={"cap_by_income_band": {"cap": 12.0}})
        back = exe.score(ROW)
    assert retuned["cap_by_income_band"].to_list() == [60.0, 24.0, 60.0, 24.0]
    assert (scored["cap_by_income_band"], back["cap_by_income_band"]) == (12.0, 48.0)


def sector_rate(sector: str, private: str = param("private"), rate: float = param(0.9, gt=0)) -> float:
    return rate if sector == private else 1.0


SECTORS = pl.DataFrame({"sector": ["private", "public", "private", "government"]})


def test_retuning_a_string_literal_never_recompiles():
    exe = Engine().bind(flow(sector_rate), mode="fused")
    exe.run(SECTORS)
    with no_recompile():
        for literal, expected in [("government", [1.0, 1.0, 1.0, 0.9]), ("martian", [1.0] * 4),
                                  ("public", [1.0, 0.9, 1.0, 1.0])]:
            out = exe.run(SECTORS, params={"sector_rate": {"private": literal}})
            assert out["sector_rate"].to_list() == expected
        assert exe.score({"sector": "public"}, params={"sector_rate": {"private": "public"}})["sector_rate"] == 0.9


@pytest.mark.parametrize("mode", ["stepped", "fused"])
def test_a_step_reading_two_string_inputs_is_a_clear_error_when_strict(mode):
    def same(a: str, b: str, which: str = param("a")) -> float:
        return 1.0

    exe = Engine().bind(flow(same), mode=mode)
    exe.runner.strict = True
    with pytest.raises(ValueError, match="same: reads several `str` inputs.*split the step"):
        exe.run(pl.DataFrame({"a": ["x"], "b": ["y"]}))


@pytest.mark.parametrize("mode", ["stepped", "fused"])
def test_a_string_param_without_a_string_input_is_a_clear_error_when_strict(mode):
    def speed(x: float, mode: str = param("fast")) -> float:
        return x * 2 if mode == "fast" else x

    exe = Engine().bind(flow(speed), mode=mode)
    exe.runner.strict = True
    with pytest.raises(ValueError, match="speed: `str` param 'mode'.*reads none"):
        exe.run(pl.DataFrame({"x": [1.0]}))


def half_or_zero(x: float | None) -> float:
    return 0.0 if x is None else x / 2.0


def test_an_optional_input_reaches_every_mode_as_a_real_none():
    out = assert_equivalent(flow(half_or_zero), pl.DataFrame({"x": [10.0, None]}))
    assert out["half_or_zero"].to_list() == [5.0, 0.0]


def maybe_band(score: float) -> int | None:
    return None if score < 0 else int(score // 100)


def test_a_nullable_output_keeps_its_type_and_its_nulls_in_every_mode():
    out = assert_equivalent(flow(maybe_band), pl.DataFrame({"score": [250.0, -1.0, 720.0]}))
    assert out["maybe_band"].dtype == pl.Int64
    assert out["maybe_band"].to_list() == [2, None, 7]


def test_a_null_passed_between_fused_steps_meets_each_readers_policy():
    def band_or_nine(maybe_band: int = missing_as(9)) -> int:
        return maybe_band

    def strict(maybe_band: int) -> int:
        return maybe_band

    frame = pl.DataFrame({"score": [250.0, -1.0]})
    assert assert_equivalent(flow(maybe_band, band_or_nine), frame)["band_or_nine"].to_list() == [2, 9]
    for m in MODES:
        with pytest.raises(ValueError, match="input 'maybe_band' of step 'strict'"):
            Engine().bind(flow(maybe_band, strict), mode=m).run(frame)


def is_big(x: float) -> bool:
    return x > 100


@step(output="y")
def big(x: float, k: float = param(2.0, ge=0)) -> float:
    return x * k


@step(output="y")
def small(x: float, k: float = param(1.0, ge=0)) -> float:
    return x * k


def test_score_equals_run_row_for_row_through_a_branch_and_nulls():
    def bump(y: float, extra: float | None) -> float:
        return y + (0.0 if extra is None else extra)

    pipeline = flow(branch(is_big, big, small, modifies=["y"], name="by"), bump)
    frame = pl.DataFrame({"x": [1.0, 200.0, 50.0, 300.0], "extra": [None, 1.0, 2.0, None]})
    assert_equivalent(pipeline, frame)


def test_fused_lazy_validation_covers_every_step_of_a_kernel_that_runs_and_nothing_else():
    @step(output="z")
    def scaled(y: float, f: float = param(1.0, ge=0)) -> float:
        return y * f

    pipeline = flow(branch(is_big, big, small, modifies=["y"], name="by"), scaled, name="p")
    exe = Engine(params_validation="lazy").bind(pipeline, mode="fused")
    exe.run(pl.DataFrame({"x": [1.0]}), params={"p": {"by": {"big": {"k": -1.0}}}})
    assert exe.report.validated == ["p/by/small", "p/scaled"]
    with pytest.raises(ValueError, match=r"p/by/big: param 'k'"):
        exe.run(pl.DataFrame({"x": [1.0, 500.0]}), params={"p": {"by": {"big": {"k": -1.0}}}})


N_WIDE = 400


def _wide_sum(row, params, consts):
    total = 0.0
    for k in range(len(row)):
        total += row[k] * (k + 1)
    return (total,)


def test_a_row_node_of_400_inputs_answers_in_every_mode():
    inputs = tuple(Input(f"f{i}", float) for i in range(N_WIDE))
    node = CallNode(Origin("wide", "tests:wide"), "row", _wide_sum, inputs, (Output("total", float),), ())
    frame = pl.DataFrame({f"f{i}": [1.0, float(i)] for i in range(N_WIDE)})
    expected = [sum(range(1, N_WIDE + 1)), sum(i * (i + 1) for i in range(N_WIDE))]
    for m in MODES:
        exe = Engine().bind(node, mode=m)
        assert exe.run(frame)["total"].to_list() == expected
        assert exe.score(frame.row(1, named=True))["total"] == expected[1]


def test_a_scalar_step_of_twenty_inputs_answers_in_every_mode():
    names = [f"a{i}" for i in range(20)]

    def weighted(a0: float, a1: float, a2: float, a3: float, a4: float, a5: float, a6: float, a7: float,
                 a8: float, a9: float, a10: float, a11: float, a12: float, a13: float, a14: float,
                 a15: float, a16: float, a17: float, a18: float, a19: float) -> float:
        return (a0 + 2 * a1 + 3 * a2 + 4 * a3 + 5 * a4 + 6 * a5 + 7 * a6 + 8 * a7 + 9 * a8 + 10 * a9 + 11 * a10
                + 12 * a11 + 13 * a12 + 14 * a13 + 15 * a14 + 16 * a15 + 17 * a16 + 18 * a17 + 19 * a18 + 20 * a19)

    frame = pl.DataFrame({n: [1.0, 2.0] for n in names})
    assert assert_equivalent(flow(weighted), frame)["weighted"].to_list() == [210.0, 420.0]
