"""Scratch tests for decider2.compile.driver — doc 05 §5 (variants), §6
(fallback), §7 (fusion grouping); doc 05 §9 (acceptance criteria 5, 6).

Real numba compilation happens in this file (unlike test_compile_codegen.py
and test_compile_cache.py, which are pure-Python/text). Kept to a handful of
tiny, float64-only steps so the whole file compiles in well under a second
after the first numba import.
"""
from __future__ import annotations

from decider2.compile.driver import build_driver
from decider2.types import Input, ParamDecl, Step


# --- module-level step functions (see test_compile_codegen.py for why) ----


def disposable_income(net_income: float, expenses: float) -> float:
    return net_income - expenses


def affordability_ratio(disposable_income: float, instalment: float) -> float:
    return disposable_income / instalment


def cap_by_income_band(term_cap: float, min_net_salary: float, cap: float, income_threshold: float) -> float:
    return min(term_cap, cap) if min_net_salary < income_threshold else term_cap


def uses_regex(cap_by_income_band: float) -> float:
    # numba cannot compile the `re` module at all -- this step is un-njit-able
    # by construction, standing in for doc 05 §1.5's Utf8/regex case.
    import re

    return 1.0 if re.match(r"\d+", str(cap_by_income_band)) else 0.0


def divides(a: float, b: float) -> float:
    return a / b


def _flagship_steps():
    s_di = Step(
        name="disposable_income", fn=disposable_income, params=(),
        inputs=(Input("net_income", float), Input("expenses", float)),
    )
    s_ar = Step(
        name="affordability_ratio", fn=affordability_ratio, params=(),
        inputs=(Input("disposable_income", float), Input("instalment", float)),
    )
    s_cap = Step(
        name="cap_by_income_band", fn=cap_by_income_band,
        inputs=(Input("term_cap", float), Input("min_net_salary", float)),
        params=(
            ParamDecl("cap", float, 48.0, None),
            ParamDecl("income_threshold", float, 5000.0, None),
        ),
    )
    return [s_di, s_ar, s_cap]


def test_one_kernel_per_module_by_default(tmp_path):
    """doc 05 §7: "apply() emits one kernel per module by default." Three
    unfused single-step modules (as `flow()` produces for bare functions,
    doc 03 §5.3) must compile to three separate compiled segments, not one
    merged kernel -- fusion is authored, never inferred (doc 02 §1.2)."""
    steps = _flagship_steps()
    group_ids = [0, 1, 2]  # no fuse(): each bare function is its own module
    driver = build_driver(steps, group_ids, build_dir=tmp_path, terminal_names=frozenset({"cap_by_income_band"}))
    assert len(driver.segments) == 3
    assert all(seg.kind == "compiled" for seg in driver.segments)
    assert [seg.steps[0].name for seg in driver.segments] == [
        "disposable_income", "affordability_ratio", "cap_by_income_band",
    ]


def test_fuse_merges_a_contiguous_group_into_one_segment(tmp_path):
    steps = _flagship_steps()
    group_ids = [0, 0, 1]  # first two steps fuse()'d together
    driver = build_driver(steps, group_ids, build_dir=tmp_path, terminal_names=frozenset({"cap_by_income_band"}))
    assert len(driver.segments) == 2
    assert [s.name for s in driver.segments[0].steps] == ["disposable_income", "affordability_ratio"]


def test_intermediate_crossing_a_kernel_boundary_is_a_required_output(tmp_path):
    """disposable_income and affordability_ratio are SEPARATE kernels here
    (no fuse()), so the value has to cross in numpy (doc 05 §7) -- it must
    be a required output of the producing segment even though (per
    test_runtime_invoke.py) it never reaches the final polars frame."""
    steps = _flagship_steps()
    group_ids = [0, 1, 2]
    driver = build_driver(steps, group_ids, build_dir=tmp_path, terminal_names=frozenset({"cap_by_income_band"}))
    di_segment = driver.segments[0]
    assert di_segment.required_outputs == ("disposable_income",)


def test_an_intermediate_local_to_a_fused_segment_is_not_a_required_output(tmp_path):
    """Fuse disposable_income and affordability_ratio into ONE kernel: now
    disposable_income is purely local to that segment (nothing outside it
    reads the name), so it must live in the kernel's registers and never
    become a separate output array -- this is where doc 03 §7's write-back
    saving actually comes from."""
    steps = _flagship_steps()
    group_ids = [0, 0, 1]
    driver = build_driver(steps, group_ids, build_dir=tmp_path, terminal_names=frozenset({"cap_by_income_band"}))
    fused_segment = driver.segments[0]
    # Neither name is a declared terminal and cap_by_income_band doesn't
    # read affordability_ratio -- both are untapped, so both stay in
    # registers once fused (contrast test_a_terminal_value_is_always_a_
    # required_output, where cap_by_income_band IS a declared terminal).
    assert fused_segment.required_outputs == ()


def test_a_terminal_value_is_always_a_required_output(tmp_path):
    steps = _flagship_steps()
    group_ids = [0, 1, 2]
    driver = build_driver(steps, group_ids, build_dir=tmp_path, terminal_names=frozenset({"cap_by_income_band"}))
    assert driver.segments[-1].required_outputs == ("cap_by_income_band",)


def test_retuning_never_grows_driver_signatures(tmp_path):
    """doc 05 §9 acceptance criterion 5 / 00-BUILD point 6, verbatim."""
    steps = _flagship_steps()
    group_ids = [0, 1, 2]
    driver = build_driver(steps, group_ids, build_dir=tmp_path, terminal_names=frozenset({"cap_by_income_band"}))

    n = 50
    import numpy as np
    from decider2.runtime import modes

    registry = {
        "net_income": np.full(n, 9200.0), "expenses": np.full(n, 3100.0),
        "instalment": np.full(n, 1200.0), "term_cap": np.full(n, 60.0),
        "min_net_salary": np.full(n, 9200.0),
    }
    resolved = modes.ResolvedParams(
        per_step_scalar={("cap_by_income_band", "cap"): 48.0, ("cap_by_income_band", "income_threshold"): 5000.0},
        per_step_bundle={},
    )
    modes.run_fused(driver, dict(registry), resolved, n)
    before = len(driver.signatures)

    resolved2 = modes.ResolvedParams(
        per_step_scalar={("cap_by_income_band", "cap"): 36.0, ("cap_by_income_band", "income_threshold"): 4000.0},
        per_step_bundle={},
    )
    modes.run_fused(driver, dict(registry), resolved2, n)
    after = len(driver.signatures)

    assert before == after
    assert before >= 1


def test_an_unnjitable_step_splits_the_kernel_around_it(tmp_path):
    """doc 05 §6: the blast radius is the kernel the bad node was going
    into, never the whole pipeline. `uses_regex` must not prevent
    disposable_income/affordability_ratio from compiling."""
    steps = _flagship_steps()
    steps.append(Step(name="uses_regex", fn=uses_regex, inputs=(Input("cap_by_income_band", float),), params=()))
    group_ids = [0, 1, 2, 2]  # cap_by_income_band and uses_regex share a group
    terminal_names = frozenset({"uses_regex"})
    driver = build_driver(steps, group_ids, build_dir=tmp_path, terminal_names=terminal_names)

    kinds = [seg.kind for seg in driver.segments]
    assert "fallback" in kinds
    fallback_segments = [seg for seg in driver.segments if seg.kind == "fallback"]
    assert len(fallback_segments) == 1
    assert fallback_segments[0].steps[0].name == "uses_regex"
    # neighbours stayed compiled
    compiled_names = {s.name for seg in driver.segments if seg.kind == "compiled" for s in seg.steps}
    assert "disposable_income" in compiled_names
    assert "cap_by_income_band" in compiled_names


def test_fallback_step_fn_in_step_fns_is_the_plain_python_function(tmp_path):
    steps = _flagship_steps()
    steps.append(Step(name="uses_regex", fn=uses_regex, inputs=(Input("cap_by_income_band", float),), params=()))
    group_ids = [0, 1, 2, 3]
    driver = build_driver(steps, group_ids, build_dir=tmp_path, terminal_names=frozenset({"uses_regex"}))
    assert driver.step_fns["uses_regex"] is uses_regex


def test_a_genuine_runtime_bug_is_not_mistaken_for_a_fallback(tmp_path):
    """doc 05 §6: only numba.core.errors.NumbaError triggers fallback. A
    ZeroDivisionError inside a step that otherwise compiles fine must not be
    caught at build time -- it propagates when the kernel actually runs."""
    step = Step(name="divides", fn=divides, inputs=(Input("a", float), Input("b", float)), params=())
    driver = build_driver([step], [0], build_dir=tmp_path, terminal_names=frozenset({"divides"}))
    assert driver.segments[0].kind == "compiled"  # it DID compile

    import numpy as np
    from decider2.runtime import modes

    registry = {"a": np.array([1.0]), "b": np.array([0.0])}
    resolved = modes.ResolvedParams(per_step_scalar={}, per_step_bundle={})
    try:
        modes.run_fused(driver, registry, resolved, 1)
    except ZeroDivisionError:
        pass
    else:
        raise AssertionError("expected the genuine ZeroDivisionError to propagate")
