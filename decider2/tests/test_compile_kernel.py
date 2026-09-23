"""Tests for decider2.compile.kernel — the fused per-row kernel built
without generated source (doc 05 §4.2, §4.3, §7; doc 05 §9 criteria 3, 5).

Replaces tests/test_compile_codegen.py: every property that file checked
on the *text* of a generated kernel (deterministic, no param value baked
in, one argument per role in a fixed order, a validity array per OPTIONAL
input) is checked here on the kernel's *behaviour* instead, since there is
no text left to inspect.
"""
from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from decider2 import flow, module, param
from decider2.compile.driver import ResolvedParams, build_driver, clear_driver_cache
from decider2.compile.kernel import KernelPlan, build_fused_kernel, kernel_signature
from decider2.graph.interface import topological_steps
from decider2.runtime import modes
from decider2.testing import assert_equivalent
from decider2.types import Input, NullPolicy, ParamDecl, Step


# --- module-level step functions ----------------------------------------


def disposable_income(net_income: float, expenses: float) -> float:
    return net_income - expenses


def affordability_ratio(disposable_income: float, instalment: float) -> float:
    return disposable_income / instalment


def cap_by_income_band(term_cap: float, min_net_salary: float, cap: float, income_threshold: float) -> float:
    return min(term_cap, cap) if min_net_salary < income_threshold else term_cap


def optional_step(x: "float | None") -> float:
    return 0.0 if x is None else x


def band(disposable_income: float) -> int:
    return int(disposable_income // 1000.0)


def is_prime_band(band: int) -> bool:
    return band in (2, 3, 5, 7)


def sector_code(sector: str, is_prime_band: bool) -> int:
    return sector + (1 if is_prime_band else 0)


def scaled(net_income: float, k: float) -> float:
    return net_income * k


def _step(fn, *, name=None, inputs=(), params=(), reads_params=False, reads_shared=False) -> Step:
    return Step(
        name=name or fn.__name__,
        fn=fn,
        inputs=inputs,
        params=params,
        reads_params=reads_params,
        reads_shared=reads_shared,
    )


def _flagship_steps():
    return [
        _step(disposable_income, inputs=(Input("net_income", float), Input("expenses", float))),
        _step(affordability_ratio, inputs=(Input("disposable_income", float), Input("instalment", float))),
        _step(
            cap_by_income_band,
            inputs=(Input("term_cap", float), Input("min_net_salary", float)),
            params=(ParamDecl("cap", float, 48.0, None), ParamDecl("income_threshold", float, 5000.0, None)),
        ),
    ]


def _flagship_registry(n: int) -> dict:
    return {
        "net_income": np.full(n, 9200.0), "expenses": np.full(n, 3100.0),
        "instalment": np.full(n, 1200.0), "term_cap": np.full(n, 60.0),
        "min_net_salary": np.linspace(1000.0, 9000.0, n),
    }


def _flagship_resolved(cap: float, threshold: float) -> ResolvedParams:
    return ResolvedParams(
        per_step_scalar={("cap_by_income_band", "cap"): cap, ("cap_by_income_band", "income_threshold"): threshold},
        per_step_bundle={},
    )


# --- stable topological order (doc 05 §4.2) -------------------------------
# Retained from test_compile_codegen.py: `graph/interface.py`'s
# `topological_steps` is the ordering that reaches `build_driver`.


def test_topological_steps_respects_dependencies():
    s_di, s_ar, _ = _flagship_steps()
    ordered = topological_steps((s_ar, s_di))  # deliberately out of order
    assert [s.name for s in ordered] == ["disposable_income", "affordability_ratio"]


def test_topological_steps_is_a_pure_function_of_its_input():
    s_di, s_ar, _ = _flagship_steps()
    assert [s.name for s in topological_steps((s_ar, s_di))] == [s.name for s in topological_steps((s_ar, s_di))]


def test_topological_steps_detects_a_cycle():
    a = Step(name="a", fn=disposable_income, inputs=(Input("b", float),), params=())
    b = Step(name="b", fn=affordability_ratio, inputs=(Input("a", float),), params=())
    with pytest.raises(ValueError, match="cycl"):
        topological_steps((a, b))


# --- kernel_signature: the one true order -----------------------------------


def test_kernel_signature_orders_arrays_then_params_then_outputs():
    step = _flagship_steps()[2]
    plan = KernelPlan(group_name="g", steps=(step,), external_inputs=step.inputs, required_outputs=(step.name,))
    assert [r.kind for r in kernel_signature(plan)] == ["array", "array", "param_scalar", "param_scalar", "output"]


def test_kernel_signature_adds_a_validity_array_for_optional_inputs():
    step = _step(optional_step, inputs=(Input("x", float, null_policy=NullPolicy.OPTIONAL),))
    plan = KernelPlan(group_name="g", steps=(step,), external_inputs=step.inputs, required_outputs=(step.name,))
    assert [r.kind for r in kernel_signature(plan)] == ["array", "valid", "output"]


def test_two_steps_sharing_a_param_name_get_distinct_slots():
    """Params are namespaced per module instance (doc 03 §4.1): two steps
    both declaring `cap` are two elements of `params_all`, each keyed by
    its own step."""
    s_a = _step(scaled, name="a", inputs=(Input("net_income", float),), params=(ParamDecl("k", float, 1.0, None),))
    s_b = _step(scaled, name="b", inputs=(Input("net_income", float),), params=(ParamDecl("k", float, 2.0, None),))
    plan = KernelPlan(group_name="g", steps=(s_a, s_b), external_inputs=(Input("net_income", float),), required_outputs=("a", "b"))
    keys = [(r.step_name, r.param_name) for r in kernel_signature(plan) if r.kind == "param_scalar"]
    assert keys == [("a", "k"), ("b", "k")]


# --- the kernel itself ----------------------------------------------------


def test_fused_kernel_is_bit_identical_to_stepped_on_the_flagship(tmp_path):
    steps = _flagship_steps()
    driver = build_driver(steps, [0, 0, 0], build_dir=tmp_path, terminal_names=frozenset({"cap_by_income_band"}))
    assert len(driver.segments) == 1 and driver.segments[0].kind == "compiled"
    n = 64
    resolved = _flagship_resolved(48.0, 5000.0)
    fused = modes.run_fused(driver, _flagship_registry(n), resolved, n)
    stepped = modes.run_stepped(driver, steps, _flagship_registry(n), resolved, n)
    assert np.array_equal(fused["cap_by_income_band"], stepped["cap_by_income_band"])
    # a fused-away intermediate is not materialised (doc 03 §7)
    assert "disposable_income" not in fused
    assert "disposable_income" in stepped


def test_no_param_value_is_baked_into_the_kernel(tmp_path):
    """doc 05 §4.2: retuning is a values change. Same driver, same kernel
    specialisation, different answers."""
    steps = _flagship_steps()
    driver = build_driver(steps, [0, 0, 0], build_dir=tmp_path, terminal_names=frozenset({"cap_by_income_band"}))
    n = 8
    a = modes.run_fused(driver, _flagship_registry(n), _flagship_resolved(48.0, 5000.0), n)["cap_by_income_band"]
    before = len(driver.signatures)
    b = modes.run_fused(driver, _flagship_registry(n), _flagship_resolved(36.0, 9999.0), n)["cap_by_income_band"]
    assert len(driver.signatures) == before == 1
    assert a.tolist() != b.tolist()
    assert b.tolist() == [36.0] * n


def test_an_optional_input_reaches_the_step_as_none_when_invalid(tmp_path):
    step = _step(optional_step, inputs=(Input("x", float, null_policy=NullPolicy.OPTIONAL),))
    driver = build_driver([step], [0], build_dir=tmp_path, terminal_names=frozenset({"optional_step"}))
    x = np.array([1.5, 2.5, 3.5])
    valid = np.array([True, False, True])
    registry = {"x": x, "__valid__x": valid}
    resolved = ResolvedParams(per_step_scalar={}, per_step_bundle={})
    out = modes.run_fused(driver, dict(registry), resolved, 3)["optional_step"]
    assert out.tolist() == [1.5, 0.0, 3.5]
    assert out.tolist() == modes.run_stepped(driver, [step], dict(registry), resolved, 3)["optional_step"].tolist()


def test_mixed_dtypes_cross_between_fused_steps_as_their_declared_dtype(tmp_path):
    """float -> int -> bool -> int32 through one kernel, every intermediate
    typed exactly as `stepped` mode materialises it (doc 05 §9 criterion 4)."""
    steps = [
        _step(disposable_income, inputs=(Input("net_income", float), Input("expenses", float))),
        _step(band, inputs=(Input("disposable_income", float),)),
        _step(is_prime_band, inputs=(Input("band", int),)),
        _step(sector_code, inputs=(Input("sector", str), Input("is_prime_band", bool))),
    ]
    driver = build_driver(steps, [0, 0, 0, 0], build_dir=tmp_path, terminal_names=frozenset({"sector_code", "band"}))
    n = 6
    registry = {
        "net_income": np.array([1500.0, 2500.0, 3500.0, 5500.0, 7500.0, 8500.0]),
        "expenses": np.full(n, 100.0),
        "sector": np.arange(n, dtype=np.int32) * 10,
    }
    resolved = ResolvedParams(per_step_scalar={}, per_step_bundle={})
    fused = modes.run_fused(driver, dict(registry), resolved, n)
    stepped = modes.run_stepped(driver, steps, dict(registry), resolved, n)
    for name in ("sector_code", "band"):
        assert fused[name].dtype == stepped[name].dtype
        assert np.array_equal(fused[name], stepped[name])
    assert fused["band"].dtype == np.int64
    assert fused["sector_code"].dtype == np.int64  # `-> int`; only the `str` input is int32
    assert fused["sector_code"].tolist() == [0, 11, 21, 31, 41, 50]  # bands 1,2,3,5,7,8: primes at 2,3,5,7


def test_steps_defined_inside_a_function_now_fuse_instead_of_falling_back(tmp_path):
    """The generated-source strategy imported each step by module-level
    name, so a closure-defined step could not be fused at all and became a
    per-row Python `FallbackSegment`. The kernel closes over the step's own
    dispatcher, so it fuses like any other step."""
    def local_rule(net_income: float, k: float) -> float:
        return net_income * k

    step = _step(local_rule, inputs=(Input("net_income", float),), params=(ParamDecl("k", float, 2.0, None),))
    driver = build_driver([step], [0], build_dir=tmp_path, terminal_names=frozenset({"local_rule"}))
    assert [seg.kind for seg in driver.segments] == ["compiled"]
    out = modes.run_fused(
        driver, {"net_income": np.array([1.0, 2.0])},
        ResolvedParams(per_step_scalar={("local_rule", "k"): 3.0}, per_step_bundle={}), 2,
    )
    assert out["local_rule"].tolist() == [3.0, 6.0]


def test_a_step_reading_an_unknown_name_is_refused_at_build_time():
    step = _step(affordability_ratio, inputs=(Input("disposable_income", float), Input("instalment", float)))
    # the plan claims only `instalment` comes from outside, and nothing
    # earlier in the group produces `disposable_income`
    plan = KernelPlan(group_name="g", steps=(step,), external_inputs=(Input("instalment", float),), required_outputs=(step.name,))
    with pytest.raises(ValueError, match="disposable_income"):
        build_fused_kernel(plan, [affordability_ratio], [np.dtype(np.float64)])


# --- no arity ceiling: a fuse group far wider than anything hand-written --


N_WIDE = 120


def test_a_fuse_group_of_120_steps_120_params_and_120_outputs(tmp_path):
    """One kernel, 120 steps sharing one function, each with its own
    param and its own output: `params_all` and `outs` are 120 wide. The
    generated kernel took 240+ positional arguments for this; nothing here
    is keyed on that number. Answers are checked per step, so a slot
    off-by-one anywhere shows up as a wrong value, not just a crash."""
    steps = [
        _step(scaled, name=f"s{k}", inputs=(Input("net_income", float),), params=(ParamDecl("k", float, 1.0, None),))
        for k in range(N_WIDE)
    ]
    names = frozenset(s.name for s in steps)
    driver = build_driver(steps, [0] * N_WIDE, build_dir=tmp_path, terminal_names=names)
    assert len(driver.segments) == 1
    n = 5
    registry = {"net_income": np.arange(1, n + 1, dtype=np.float64)}
    resolved = ResolvedParams(
        per_step_scalar={(f"s{k}", "k"): float(k) for k in range(N_WIDE)}, per_step_bundle={},
    )
    out = modes.run_fused(driver, dict(registry), resolved, n)
    for k in (0, 1, 17, 64, N_WIDE - 1):
        assert out[f"s{k}"].tolist() == [float(k) * (i + 1) for i in range(n)]
    # and the ladder, exactly
    stepped = modes.run_stepped(driver, steps, dict(registry), resolved, n)
    assert all(np.array_equal(out[s.name], stepped[s.name]) for s in steps)
    # a retune of every one of the 120 params is still one specialisation
    before = len(driver.signatures)
    modes.run_fused(driver, dict(registry), ResolvedParams(
        per_step_scalar={(f"s{k}", "k"): float(k) + 0.5 for k in range(N_WIDE)}, per_step_bundle={},
    ), n)
    assert len(driver.signatures) == before


def _chain(n_links: int):
    """`n_links` distinct hand-written links in one module (one kernel per module,
    doc 05 §7), through the public API, every intermediate fused away."""
    def l0(x: float, k0: float = param(1.0)) -> float:
        return x * k0 + 1.0

    def l1(l0: float, k1: float = param(1.0)) -> float:
        return l0 * k1 + 1.0

    def l2(l1: float, k2: float = param(1.0)) -> float:
        return l1 * k2 + 1.0

    def l3(l2: float, k3: float = param(1.0)) -> float:
        return l2 * k3 + 1.0

    def l4(l3: float, k4: float = param(1.0)) -> float:
        return l3 * k4 + 1.0

    def l5(l4: float, k5: float = param(1.0)) -> float:
        return l4 * k5 + 1.0

    return [l0, l1, l2, l3, l4, l5][:n_links]


def test_fuse_group_through_the_public_api_passes_the_full_ladder():
    links = _chain(6)
    pipeline = flow(module(*links, name="chain"))
    frame = pl.DataFrame({"x": [0.0, 1.0, -2.5, 1e6]})
    out = pipeline.apply(frame, mode="fused")
    expected = frame["x"].to_numpy()
    for _ in links:
        expected = expected * 1.0 + 1.0
    assert out["l5"].to_list() == expected.tolist()
    assert_equivalent(pipeline, frame)


# --- authored flags --------------------------------------------------------


def test_parallel_group_with_scalar_params_matches_the_serial_kernel(tmp_path):
    steps = _flagship_steps()
    n = 1000
    resolved = _flagship_resolved(48.0, 5000.0)
    serial = build_driver(steps, [0, 0, 0], build_dir=tmp_path, terminal_names=frozenset({"cap_by_income_band"}))
    parallel = build_driver(
        steps, [0, 0, 0], build_dir=tmp_path, terminal_names=frozenset({"cap_by_income_band"}),
        parallel_group_ids=frozenset({0}),
    )
    assert parallel is not serial
    a = modes.run_fused(serial, _flagship_registry(n), resolved, n)["cap_by_income_band"]
    b = modes.run_fused(parallel, _flagship_registry(n), resolved, n)["cap_by_income_band"]
    assert np.array_equal(a, b)


def test_parallel_group_reading_a_bundle_is_refused_with_a_reason():
    def bundled(net_income: float, params) -> float:
        return net_income * params.k

    step = _step(bundled, inputs=(Input("net_income", float),), reads_params=True)
    plan = KernelPlan(
        group_name="g", steps=(step,), external_inputs=step.inputs, required_outputs=("bundled",), parallel=True,
    )
    with pytest.raises(ValueError, match="parallel"):
        build_fused_kernel(plan, [bundled], [np.dtype(np.float64)])


def test_nogil_is_only_released_when_every_step_asked(tmp_path):
    steps = _flagship_steps()
    driver = build_driver(steps, [0, 0, 0], build_dir=tmp_path, terminal_names=frozenset({"cap_by_income_band"}))
    assert driver.segments[0].gil_report_entry()["holds_gil"] is True
    import dataclasses
    all_nogil = [dataclasses.replace(s, nogil=True) for s in steps]
    # `nogil` is authored on the function itself (`@step(nogil=True)`), so
    # the structural driver key — (name, fn) per step — does not carry it;
    # a second build of the *same* fns must not hit the first's entry here.
    clear_driver_cache()
    driver2 = build_driver(all_nogil, [0, 0, 0], build_dir=tmp_path, terminal_names=frozenset({"cap_by_income_band"}))
    assert driver2.segments[0].gil_report_entry()["holds_gil"] is False
    assert driver2.segments[0].kernel_fn.targetoptions.get("nogil") is True
    assert driver.segments[0].kernel_fn.targetoptions.get("nogil") is False
