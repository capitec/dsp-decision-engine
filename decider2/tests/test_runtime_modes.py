"""Scratch tests for decider2.runtime.modes — doc 02 §3.1 (the equivalence
ladder): interpreted, stepped and fused must all agree exactly, and each
mode must independently produce the flagship answer.
"""
from __future__ import annotations

import numpy as np

from decider2.compile.driver import build_driver
from decider2.runtime import modes
from decider2.runtime.modes import ResolvedParams
from decider2.types import Input, NullPolicy, ParamDecl, Step


def disposable_income(net_income: float, expenses: float) -> float:
    return net_income - expenses


def affordability_ratio(disposable_income: float, instalment: float) -> float:
    return disposable_income / instalment


def cap_by_income_band(term_cap: float, min_net_salary: float, cap: float, income_threshold: float) -> float:
    return min(term_cap, cap) if min_net_salary < income_threshold else term_cap


def uses_shared(x: float, shared) -> float:
    return x * shared.base_rate


def uses_params_bundle(x: float, params) -> float:
    return x + params.offset


def half_or_zero(x: "float | None") -> float:
    return 0.0 if x is None else x / 2.0


def _flagship_steps():
    return [
        Step(name="disposable_income", fn=disposable_income, params=(),
             inputs=(Input("net_income", float), Input("expenses", float))),
        Step(name="affordability_ratio", fn=affordability_ratio, params=(),
             inputs=(Input("disposable_income", float), Input("instalment", float))),
        Step(
            name="cap_by_income_band", fn=cap_by_income_band,
            inputs=(Input("term_cap", float), Input("min_net_salary", float)),
            params=(
                ParamDecl("cap", float, 48.0, None),
                ParamDecl("income_threshold", float, 5000.0, None),
            ),
        ),
    ]


FRAME = {
    "net_income":     np.array([9200.0, 4100.0, 15000.0, 4999.0]),
    "expenses":       np.array([3100.0, 1500.0,  6000.0, 2000.0]),
    "instalment":     np.array([1200.0,  800.0,  2500.0,  700.0]),
    "term_cap":       np.array([  60.0,   60.0,    60.0,   60.0]),
    "min_net_salary": np.array([9200.0, 4100.0, 15000.0, 4999.0]),
}
EXPECTED = [60.0, 48.0, 60.0, 48.0]


def _resolved():
    return ResolvedParams(
        per_step_scalar={
            ("cap_by_income_band", "cap"): 48.0,
            ("cap_by_income_band", "income_threshold"): 5000.0,
        },
        per_step_bundle={},
    )


def test_interpreted_matches_the_flagship_answer():
    steps = _flagship_steps()
    registry = modes.run_interpreted(steps, dict(FRAME), _resolved(), 4)
    assert registry["cap_by_income_band"].tolist() == EXPECTED


def test_stepped_matches_the_flagship_answer(tmp_path):
    steps = _flagship_steps()
    driver = build_driver(steps, [0, 1, 2], build_dir=tmp_path, terminal_names=frozenset({"cap_by_income_band"}))
    registry = modes.run_stepped(driver, steps, dict(FRAME), _resolved(), 4)
    assert registry["cap_by_income_band"].tolist() == EXPECTED


def test_fused_matches_the_flagship_answer(tmp_path):
    steps = _flagship_steps()
    driver = build_driver(steps, [0, 1, 2], build_dir=tmp_path, terminal_names=frozenset({"cap_by_income_band"}))
    registry = modes.run_fused(driver, dict(FRAME), _resolved(), 4)
    assert registry["cap_by_income_band"].tolist() == EXPECTED


def test_the_three_modes_agree_exactly(tmp_path):
    steps = _flagship_steps()
    driver = build_driver(steps, [0, 1, 2], build_dir=tmp_path, terminal_names=frozenset({"cap_by_income_band"}))
    interp = modes.run_interpreted(steps, dict(FRAME), _resolved(), 4)["cap_by_income_band"]
    stepped = modes.run_stepped(driver, steps, dict(FRAME), _resolved(), 4)["cap_by_income_band"]
    fused = modes.run_fused(driver, dict(FRAME), _resolved(), 4)["cap_by_income_band"]
    assert np.array_equal(interp, stepped)
    assert np.array_equal(stepped, fused)


def test_the_three_modes_agree_exactly_when_fused_into_one_kernel(tmp_path):
    """Same assertion, but with fuse() merging all three steps into one
    compiled kernel -- doc 02 §1.2: fuse() must be semantically transparent."""
    steps = _flagship_steps()
    driver = build_driver(steps, [0, 0, 0], build_dir=tmp_path, terminal_names=frozenset({"cap_by_income_band"}))
    interp = modes.run_interpreted(steps, dict(FRAME), _resolved(), 4)["cap_by_income_band"]
    fused = modes.run_fused(driver, dict(FRAME), _resolved(), 4)["cap_by_income_band"]
    assert np.array_equal(interp, fused)
    assert len(driver.segments) == 1


def test_shared_is_passed_by_reference_to_steps_that_ask_for_it(tmp_path):
    step = Step(name="uses_shared", fn=uses_shared, inputs=(Input("x", float),), params=(), reads_shared=True)
    driver = build_driver([step], [0], build_dir=tmp_path, terminal_names=frozenset({"uses_shared"}))
    shared = ResolvedParams(per_step_scalar={}, per_step_bundle={}, shared=_SharedTuple(base_rate=2.5))
    registry = modes.run_fused(driver, {"x": np.array([10.0, 20.0])}, shared, 2)
    assert registry["uses_shared"].tolist() == [25.0, 50.0]


def test_reads_params_bundle_is_passed_as_a_namedtuple(tmp_path):
    step = Step(name="uses_params_bundle", fn=uses_params_bundle, inputs=(Input("x", float),), params=(), reads_params=True)
    driver = build_driver([step], [0], build_dir=tmp_path, terminal_names=frozenset({"uses_params_bundle"}))
    from collections import namedtuple

    Bundle = namedtuple("Bundle", ["offset"])
    resolved = ResolvedParams(per_step_scalar={}, per_step_bundle={"uses_params_bundle": Bundle(offset=7.0)})
    registry = modes.run_fused(driver, {"x": np.array([1.0, 2.0])}, resolved, 2)
    assert registry["uses_params_bundle"].tolist() == [8.0, 9.0]


def _SharedTuple(**kwargs):
    from collections import namedtuple

    return namedtuple("Shared", list(kwargs))(**kwargs)


def test_optional_input_reaches_the_step_as_a_real_none(tmp_path):
    step = Step(
        name="half_or_zero", fn=half_or_zero, params=(),
        inputs=(Input("x", float, null_policy=NullPolicy.OPTIONAL),),
    )
    driver = build_driver([step], [0], build_dir=tmp_path, terminal_names=frozenset({"half_or_zero"}))
    registry = {"x": np.array([10.0, 0.0]), "__valid__x": np.array([True, False])}
    resolved = ResolvedParams(per_step_scalar={}, per_step_bundle={})
    out = modes.run_fused(driver, registry, resolved, 2)
    assert out["half_or_zero"].tolist() == [5.0, 0.0]

    interp = modes.run_interpreted([step], dict(registry), resolved, 2)
    assert interp["half_or_zero"].tolist() == out["half_or_zero"].tolist()
