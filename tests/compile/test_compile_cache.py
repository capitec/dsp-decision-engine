"""Compiled code is keyed by content: names and paths never recompile, edits recompile only their kernel."""
from __future__ import annotations

import platform
import sys

import numpy as np

from decider import flow
from decider.engine.compile import compile_plan, cpu_target, fingerprint, jit
from decider.engine.ir.decls import Input, Output
from decider.engine.ir.nodes import CallNode, SequenceNode
from decider.engine.ir.origin import Origin
from decider.engine.params import param
from decider.engine.wiring import resolve

RATE = 0.1


def disposable_income(net_income: float, expenses: float) -> float:
    return net_income - expenses


def affordability_ratio(disposable_income: float, instalment: float) -> float:
    return disposable_income / instalment


def cap_v1(affordability_ratio: float, cap: float = param(48.0)) -> float:
    return min(affordability_ratio * 10.0, cap)


def cap_v1_again(affordability_ratio: float, cap: float = param(48.0)) -> float:
    return min(affordability_ratio * 10.0, cap)


def cap_v2(affordability_ratio: float, cap: float = param(48.0)) -> float:
    return min(affordability_ratio * 12.0, cap)


def interest(x: float) -> float:
    return x * RATE


INPUTS = {"net_income": np.array([9200.0, 5000.0]), "expenses": np.array([3100.0, 900.0]),
          "instalment": np.array([1200.0, 300.0])}


def _kernels(plan):
    return [u.fn for u in {id(u): u for u in compile_plan(plan).values()}.values()]


def _pipeline(cap, outer="p", inner="afford"):
    return flow(flow(disposable_income, affordability_ratio, name=inner), flow(cap, name="term"), name=outer)


def test_renaming_a_parent_step_does_not_recompile(run):
    first = resolve(_pipeline(cap_v1, outer="p", inner="afford"))
    second = resolve(_pipeline(cap_v1, outer="pipeline", inner="affordability"))
    assert _kernels(first) == _kernels(second)
    run(first, INPUTS)
    counts = [len(k.signatures) for k in _kernels(first)]
    run(second, INPUTS)
    assert [len(k.signatures) for k in _kernels(second)] == counts


def test_moving_a_step_under_another_parent_does_not_recompile():
    moved = flow(flow(flow(disposable_income, affordability_ratio, name="afford"), name="deeper"),
                 flow(cap_v1, name="term"), name="p")
    assert _kernels(resolve(_pipeline(cap_v1))) == _kernels(resolve(moved))


def test_editing_one_step_recompiles_only_its_kernel():
    afford, term = _kernels(resolve(_pipeline(cap_v1)))
    afford2, term2 = _kernels(resolve(_pipeline(cap_v2)))
    assert afford2 is afford
    assert term2 is not term


def test_a_redefined_step_with_the_same_body_reuses_its_compiled_code():
    assert _kernels(resolve(_pipeline(cap_v1_again))) == _kernels(resolve(_pipeline(cap_v1)))
    assert jit(cap_v1_again)[1] is jit(cap_v1)[1]


def _const_plan(threshold):
    def over(x: float, threshold: float) -> bool:
        return x > threshold

    node = CallNode(Origin("p/over", "tests"), "scalar", over, (Input("x", float),), (Output("over", bool),), (),
                    consts=(("threshold", threshold),))
    return resolve(SequenceNode(Origin("p", "tests"), (node,)))


def test_changing_a_literal_const_does_not_recompile(run):
    low, high = _const_plan(1.5), _const_plan(2.5)
    (kernel,) = _kernels(low)
    assert _kernels(high) == [kernel]
    x = {"x": np.array([2.0, 3.0])}
    assert run(low, x)[0]["over"].tolist() == [True, True]
    count = len(kernel.signatures)
    assert run(high, x)[0]["over"].tolist() == [False, True]
    assert len(kernel.signatures) == count


def test_a_retune_reuses_the_kernel_and_a_changed_param_type_specialises_it(run):
    plan = resolve(_pipeline(cap_v1))
    units = compile_plan(plan)
    kernel = units[plan.calls[-1].id].fn
    run(plan, INPUTS, units=units)
    count = len(kernel.signatures)
    run(plan, INPUTS, units=units, params={"p/term/cap_v1": {"cap": 20.0}})
    assert len(kernel.signatures) == count
    run(plan, INPUTS, units=units, params={"p/term/cap_v1": {"cap": 20}})
    assert len(kernel.signatures) == count + 1


def test_the_fingerprint_ignores_names_and_sees_constants():
    def a(x: float) -> float:
        return x * 49.39775

    def b(x: float) -> float:
        return x * 49.39775

    def c(x: float) -> float:
        return x * 49.29765

    assert fingerprint(a) == fingerprint(b) != fingerprint(c)


def test_the_fingerprint_sees_closure_values_and_globals(monkeypatch):
    def make(k):
        def rule(x: float) -> float:
            return x * k

        return rule

    assert fingerprint(make(2.0)) == fingerprint(make(2.0)) != fingerprint(make(3.0))
    before = fingerprint(interest)
    monkeypatch.setattr(sys.modules[__name__], "RATE", 0.2)
    assert fingerprint(interest) != before


def make(data):
    def rule(x: float) -> float:
        return x * data[0]

    return rule


def test_the_fingerprint_sees_captured_arrays_and_frames_by_value():
    import polars as pl

    a = np.array([1.0, 2.0])
    before = fingerprint(make(a))
    assert fingerprint(make(a.copy())) == before != fingerprint(make(a.astype(np.float32)))
    a[0] = 3.0                              # an in-place edit is a change
    assert fingerprint(make(a)) != before
    frame = pl.DataFrame({"id": [1, 2], "score": [700, 650]})
    assert fingerprint(make(frame)) == fingerprint(make(frame.clone()))
    assert fingerprint(make(frame)) != fingerprint(make(frame.with_columns(score=pl.Series([700, 600]))))
    assert fingerprint(make(frame)) != fingerprint(make(frame.cast({"score": pl.Float64})))


def test_equal_captured_arrays_share_compiled_code_and_a_changed_one_compiles_its_own():
    _, compiled = jit(make(np.array([2.0])))
    assert jit(make(np.array([2.0])))[1] is compiled
    _, other = jit(make(np.array([5.0])))
    assert (compiled(1.0), other(1.0)) == (2.0, 5.0)


def test_the_cpu_target_is_recorded():
    triple, cpu, features = cpu_target()
    assert triple.startswith(platform.machine()) and cpu and features
