"""Grouping calls into units, what a unit keeps, and the Python fallback."""
from __future__ import annotations

import numpy as np
import pytest
from numba.core.dispatcher import Dispatcher

from decider import ConfigurableStep, branch, flow, frame_step, step
from decider.engine.compile import Fallback, Kernel, compile_plan
from decider.engine.ir.decls import Input, Output, ParamDecl
from decider.engine.ir.nodes import CallNode
from decider.engine.params import param
from decider.engine.wiring import resolve


def disposable_income(net_income: float, expenses: float) -> float:
    return net_income - expenses


def affordability_ratio(disposable_income: float, instalment: float) -> float:
    return disposable_income / instalment


def cap_by_income_band(term_cap: float, min_net_salary: float, cap: float = param(48.0),
                       income_threshold: float = param(5000.0)) -> float:
    return min(term_cap, cap) if min_net_salary < income_threshold else term_cap


def uses_regex(cap_by_income_band: float) -> float:
    import re  # numba can't compile an import

    return 1.0 if re.match(r"\d+", str(cap_by_income_band)) else 0.0


def untypable(cap_by_income_band: float) -> float:
    return float(len("%.2f" % cap_by_income_band))  # %-formatting: a numba TypingError


def divides(a: float, b: float) -> float:
    return a / b


INPUTS = {
    "net_income": np.full(4, 9200.0), "expenses": np.full(4, 3100.0), "instalment": np.full(4, 1200.0),
    "term_cap": np.full(4, 60.0), "min_net_salary": np.array([1000.0, 4000.0, 6000.0, 9000.0]),
}


def _units(units):
    return list({id(u): u for u in units.values()}.values())


def _names(unit):
    return [c.node.origin.path.rsplit("/", 1)[-1] for c in unit.calls]


def test_consecutive_scalar_calls_in_one_sequence_share_a_kernel():
    plan = resolve(flow(disposable_income, affordability_ratio, cap_by_income_band))
    (unit,) = _units(compile_plan(plan))
    assert isinstance(unit, Kernel)
    assert _names(unit) == ["disposable_income", "affordability_ratio", "cap_by_income_band"]


def test_each_innermost_sequence_is_its_own_kernel():
    plan = resolve(flow(flow(disposable_income, affordability_ratio, name="afford"), cap_by_income_band, name="p"))
    assert [_names(u) for u in _units(compile_plan(plan))] == [
        ["disposable_income", "affordability_ratio"], ["cap_by_income_band"],
    ]


def test_without_fusion_every_call_is_its_own_unit():
    plan = resolve(flow(disposable_income, affordability_ratio, cap_by_income_band))
    assert [_names(u) for u in _units(compile_plan(plan, fuse=False))] == [
        ["disposable_income"], ["affordability_ratio"], ["cap_by_income_band"],
    ]


def test_a_frame_call_breaks_the_kernel_and_gets_no_unit():
    @frame_step(reads=["affordability_ratio"], writes=["flag"])
    def flag(df):
        return df

    def after(flag: float, cap_by_income_band: float) -> float:
        return flag + cap_by_income_band

    plan = resolve(flow(disposable_income, affordability_ratio, flag, cap_by_income_band, after))
    units = compile_plan(plan)
    assert [_names(u) for u in _units(units)] == [
        ["disposable_income", "affordability_ratio"], ["cap_by_income_band", "after"],
    ]
    assert plan.calls[2].id not in units


def test_branch_conditions_and_arms_are_compiled_but_not_the_branch():
    def is_low(min_net_salary: float) -> bool:
        return min_net_salary < 5000.0

    @step(output="term_cap")
    def low(term_cap: float) -> float:
        return min(term_cap, 36.0)

    @step(output="term_cap")
    def high(term_cap: float) -> float:
        return term_cap

    plan = resolve(flow(disposable_income, branch(is_low, low, high, modifies=["term_cap"], name="by")))
    assert [_names(u) for u in _units(compile_plan(plan))] == [["disposable_income"], ["is_low"], ["low"], ["high"]]


def test_a_value_read_by_another_kernel_is_kept():
    plan = resolve(flow(flow(disposable_income, name="a"), affordability_ratio, name="p"))
    first = compile_plan(plan)[0]
    assert [v.name for v, _ in first.writes] == ["disposable_income"]


def test_a_value_only_read_inside_its_kernel_is_not_kept(run):
    plan = resolve(flow(disposable_income, affordability_ratio))
    (unit,) = _units(compile_plan(plan))
    assert [v.name for v, _ in unit.writes] == ["affordability_ratio"]
    _, values = run(plan, INPUTS)
    assert plan.calls[0].writes[0].id not in values


def test_a_plan_output_is_always_kept():
    plan = resolve(flow(disposable_income, affordability_ratio).emit("disposable_income"))
    (unit,) = _units(compile_plan(plan))
    assert [v.name for v, _ in unit.writes] == ["disposable_income", "affordability_ratio"]


def test_without_fusion_every_written_value_is_kept(run):
    plan = resolve(flow(disposable_income, affordability_ratio))
    _, values = run(plan, INPUTS, fuse=False)
    assert values[plan.calls[0].writes[0].id].tolist() == [6100.0] * 4


def test_retuning_never_grows_the_kernel_signatures(run):
    plan = resolve(flow(disposable_income, affordability_ratio, cap_by_income_band))
    units = compile_plan(plan)
    (unit,) = _units(units)
    a, _ = run(plan, INPUTS, units=units)
    before = len(unit.fn.signatures)
    b, _ = run(plan, INPUTS, units=units, params={"cap_by_income_band": {"cap": 36.0, "income_threshold": 9999.0}})
    assert len(unit.fn.signatures) == before == 1
    assert a["cap_by_income_band"].tolist() == [48.0, 48.0, 60.0, 60.0]
    assert b["cap_by_income_band"].tolist() == [36.0] * 4


@pytest.mark.parametrize("bad", [uses_regex, untypable])
def test_a_step_numba_cant_compile_splits_the_kernel_around_it(bad, run):
    def after(cap_by_income_band: float) -> float:
        return cap_by_income_band + 1.0

    plan = resolve(flow(disposable_income, affordability_ratio, cap_by_income_band, bad, after))
    units = _units(compile_plan(plan))
    assert [type(u).__name__ for u in units] == ["Kernel", "Fallback", "Kernel"]
    assert _names(units[0]) == ["disposable_income", "affordability_ratio", "cap_by_income_band"]
    assert units[1].fn is bad and units[1].reason.startswith(("UnsupportedBytecodeError", "TypingError"))
    assert [v.name for v, _ in units[0].writes] == ["affordability_ratio", "cap_by_income_band"]
    out, _ = run(plan, INPUTS)
    assert out["after"].tolist() == [49.0, 49.0, 61.0, 61.0]


def test_a_runtime_error_in_a_compiled_step_propagates(run):
    plan = resolve(flow(divides))
    (unit,) = _units(compile_plan(plan))
    assert isinstance(unit, Kernel)
    with pytest.raises(ZeroDivisionError):
        run(plan, {"a": np.array([1.0]), "b": np.array([0.0])})


def test_a_runtime_error_in_a_fallback_step_propagates(run):
    def raises(a: float) -> float:
        import math

        return math.log(-a) if a > 0 else a

    plan = resolve(flow(raises))
    assert isinstance(compile_plan(plan)[0], Fallback)
    with pytest.raises(ValueError, match="positive|domain"):
        run(plan, {"a": np.array([1.0])})


def test_an_error_that_is_not_a_compile_failure_is_not_a_fallback(monkeypatch):
    def fresh(a: float) -> float:
        return a * 7.25

    def boom(self, sig):
        raise RuntimeError("not a numba compile failure")

    monkeypatch.setattr(Dispatcher, "compile", boom)
    with pytest.raises(RuntimeError, match="not a numba"):
        compile_plan(resolve(flow(fresh)))


def test_a_fallback_writes_every_output_with_its_declared_dtype(run):
    @step(outputs=("band", "flagged"))
    def banding(a: float) -> tuple[int, bool]:
        import math

        return int(math.floor(a)), a > 2

    plan = resolve(flow(banding))
    assert isinstance(compile_plan(plan)[0], Fallback)
    out, _ = run(plan, {"a": np.array([1.5, 2.5])})
    assert out["band"].dtype == np.int64 and out["band"].tolist() == [1, 2]
    assert out["flagged"].dtype == np.bool_ and out["flagged"].tolist() == [False, True]


def test_origins_map_a_unit_back_to_its_nodes():
    plan = resolve(flow(flow(disposable_income, affordability_ratio, name="afford"), name="p"))
    (unit,) = _units(compile_plan(plan))
    assert [o.path for o in unit.origins] == ["p/afford/disposable_income", "p/afford/affordability_ratio"]


def clipped(x, lo, hi):
    return min(max(x, lo), hi)


class Clip(ConfigurableStep):
    def to_ir(self, ctx):
        return CallNode(ctx.origin(self), "scalar", clipped, (Input("disposable_income", float, arg="x"),),
                        (Output(self.name, float),), (ParamDecl("upper", float, 7000.0, arg="hi"),),
                        consts=(("lo", 6500.0),))


@pytest.mark.parametrize("fuse", [True, False])
def test_a_param_reaches_its_argument_when_its_document_key_differs(run, fuse):
    plan = resolve(flow(disposable_income, Clip(name="clip")))
    a, _ = run(plan, INPUTS, fuse=fuse)
    b, _ = run(plan, INPUTS, fuse=fuse, params={"clip": {"upper": 6000.0}})
    assert (a["clip"].tolist(), b["clip"].tolist()) == ([6500.0] * 4, [6000.0] * 4)
