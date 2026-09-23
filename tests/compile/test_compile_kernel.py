"""The fused kernel: arguments, dtypes between calls, width, row nodes and flags."""
from __future__ import annotations

import numpy as np
import pytest

from decider import flow, step
from decider.engine.compile import Fallback, Kernel, compile_plan, numpy_dtype
from decider.engine.ir.decls import FeatureKind as K
from decider.engine.ir.decls import Input, NullPolicy, Output, ParamDecl, feature_kind
from decider.engine.ir.nodes import CallNode, SequenceNode
from decider.engine.ir.origin import Origin
from decider.engine.params import param
from decider.engine.wiring import resolve


def disposable_income(net_income: float, expenses: float) -> float:
    return net_income - expenses


def affordability_ratio(disposable_income: float, instalment: float) -> float:
    return disposable_income / instalment


def cap_by_income_band(term_cap: float, min_net_salary: float, cap: float = param(48.0),
                       income_threshold: float = param(5000.0)) -> float:
    return min(term_cap, cap) if min_net_salary < income_threshold else term_cap


def optional_step(x: float | None) -> float:
    return 0.0 if x is None else x


def band(disposable_income: float) -> int:
    return int(disposable_income // 1000.0)


def is_prime_band(band: int) -> bool:
    return band in (2, 3, 5, 7)


def sector_code(sector: str, is_prime_band: bool) -> int:
    return sector + (1 if is_prime_band else 0)


def scaled(net_income: float, k: float = param(1.0)) -> float:
    return net_income * k


def flagship_inputs(n):
    return {
        "net_income": np.full(n, 9200.0), "expenses": np.full(n, 3100.0), "instalment": np.full(n, 1200.0),
        "term_cap": np.full(n, 60.0), "min_net_salary": np.linspace(1000.0, 9000.0, n),
    }


def _units(units):
    return list({id(u): u for u in units.values()}.values())


def _origin(path):
    return Origin(path, "tests")


@pytest.mark.parametrize("annotation,kind", [
    (float, K.F64), (int, K.I64), (bool, K.BOOL), (str, K.CODE),
    (float | None, K.F64), (int | None, K.F64), (None, K.F64), (object, K.F64),
])
def test_numpy_dtype_mirrors_feature_kind(annotation, kind):
    assert feature_kind(annotation) is kind
    assert numpy_dtype(annotation).name == {K.F64: "float64", K.I64: "int64", K.BOOL: "bool", K.CODE: "int32"}[kind]


def test_fused_output_is_bit_identical_to_one_kernel_per_call(run):
    plan = resolve(flow(disposable_income, affordability_ratio, cap_by_income_band).emit("affordability_ratio"))
    fused, values = run(plan, flagship_inputs(64))
    stepped, _ = run(plan, flagship_inputs(64), fuse=False)
    assert all(np.array_equal(fused[k], stepped[k]) for k in stepped)
    assert plan.calls[0].writes[0].id not in values


def test_an_optional_input_reaches_the_step_as_none_when_invalid(run):
    plan = resolve(flow(optional_step))
    x, valid = np.array([1.5, 2.5, 3.5]), np.array([True, False, True])
    assert run(plan, {"x": x}, valid={"x": valid})[0]["optional_step"].tolist() == [1.5, 0.0, 3.5]
    assert run(plan, {"x": x})[0]["optional_step"].tolist() == [1.5, 2.5, 3.5]


def test_values_cross_between_fused_steps_as_their_declared_dtype(run):
    plan = resolve(flow(disposable_income, band, is_prime_band, sector_code).emit("band"))
    inputs = {
        "net_income": np.array([1500.0, 2500.0, 3500.0, 5500.0, 7500.0, 8500.0]),
        "expenses": np.full(6, 100.0), "sector": np.arange(6, dtype=np.int32) * 10,
    }
    fused, _ = run(plan, inputs)
    stepped, _ = run(plan, inputs, fuse=False)
    for name in ("sector_code", "band"):
        assert fused[name].dtype == stepped[name].dtype == np.int64
        assert np.array_equal(fused[name], stepped[name])
    assert fused["sector_code"].tolist() == [0, 11, 21, 31, 41, 50]


def test_a_mixed_int_float_bool_pipeline_fused_equals_per_call_bit_for_bit(run):
    def ratio(a: float, b: float) -> float:
        return a / (abs(b) + 1.0)

    def bucket(ratio: float) -> int:
        return int(ratio * 7.3)  # truncates

    def odd(bucket: int) -> bool:
        return bucket % 2 == 1

    def truthy(ratio: float) -> bool:
        return ratio  # a float stored as bool

    @step(outputs=("lo", "hi"))
    def split(bucket: int, odd: bool, truthy: bool) -> tuple[float, int]:
        return bucket * 0.1 + odd, bucket + truthy

    def unannotated(lo, hi):
        return lo * hi  # no annotation: stored as float64

    plan = resolve(flow(ratio, bucket, odd, truthy, split, unannotated).emit("ratio", "bucket", "odd", "truthy", "lo", "hi"))
    rng = np.random.default_rng(3)
    inputs = {"a": rng.normal(0, 50, 1000), "b": rng.normal(0, 5, 1000)}
    fused, _ = run(plan, inputs)
    stepped, _ = run(plan, inputs, fuse=False)
    assert set(fused) == set(stepped) >= {"lo", "hi", "unannotated"}
    for name in stepped:
        assert fused[name].dtype == stepped[name].dtype
        assert fused[name].tobytes() == stepped[name].tobytes()
    assert fused["truthy"].dtype == np.bool_ and fused["hi"].dtype == np.int64
    assert fused["unannotated"].dtype == np.float64


def test_a_step_defined_inside_a_function_fuses(run):
    def local_rule(net_income: float, k: float = param(2.0)) -> float:
        return net_income * k

    plan = resolve(flow(local_rule))
    assert isinstance(compile_plan(plan)[0], Kernel)
    out, _ = run(plan, {"net_income": np.array([1.0, 2.0])}, params={"local_rule": {"k": 3.0}})
    assert out["local_rule"].tolist() == [3.0, 6.0]


def test_two_steps_sharing_a_param_name_get_their_own_values(run):
    plan = resolve(flow(step(scaled, name="a", output="a"), step(scaled, name="b", output="b")))
    out, _ = run(plan, {"net_income": np.array([1.0, 2.0])}, params={"a": {"k": 2.0}, "b": {"k": 5.0}})
    assert out["a"].tolist() == [2.0, 4.0] and out["b"].tolist() == [5.0, 10.0]


def test_an_argument_that_is_no_input_const_or_param_is_refused_at_build_time():
    node = CallNode(_origin("p/affordability_ratio"), "scalar", affordability_ratio,
                    (Input("instalment", float),), (Output("affordability_ratio", float),), ())
    plan = resolve(SequenceNode(_origin("p"), (node,)))
    with pytest.raises(ValueError, match="disposable_income"):
        compile_plan(plan)


def test_consts_reach_the_step_as_arguments(run):
    def over(x: float, threshold: float) -> bool:
        return x > threshold

    node = CallNode(_origin("p/over"), "scalar", over, (Input("x", float),), (Output("over", bool),), (),
                    consts=(("threshold", 1.5),))
    out, _ = run(resolve(SequenceNode(_origin("p"), (node,))), {"x": np.array([1.0, 2.0])})
    assert out["over"].tolist() == [False, True]


N_WIDE = 120


def test_a_kernel_of_120_steps_120_params_and_120_outputs(run):
    steps = [step(scaled, name=f"s{k}", output=f"s{k}") for k in range(N_WIDE)]
    plan = resolve(flow(*steps))
    units = compile_plan(plan)
    (unit,) = _units(units)
    inputs = {"net_income": np.arange(1, 6, dtype=np.float64)}
    out, _ = run(plan, inputs, units=units, params={f"s{k}": {"k": float(k)} for k in range(N_WIDE)})
    for k in (0, 1, 17, 64, N_WIDE - 1):
        assert out[f"s{k}"].tolist() == [float(k) * (i + 1) for i in range(5)]
    stepped, _ = run(plan, inputs, fuse=False, params={f"s{k}": {"k": float(k)} for k in range(N_WIDE)})
    assert all(np.array_equal(out[name], stepped[name]) for name in stepped)
    before = len(unit.fn.signatures)
    run(plan, inputs, units=units, params={f"s{k}": {"k": k + 0.5} for k in range(N_WIDE)})
    assert len(unit.fn.signatures) == before


def test_a_chain_through_the_public_api_uses_param_defaults(run):
    def l0(x: float, k0: float = param(1.0)) -> float:
        return x * k0 + 1.0

    def l1(l0: float, k1: float = param(1.0)) -> float:
        return l0 * k1 + 1.0

    def l2(l1: float, k2: float = param(1.0)) -> float:
        return l1 * k2 + 1.0

    plan = resolve(flow(l0, l1, l2, name="chain"))
    x = np.array([0.0, 1.0, -2.5, 1e6])
    out, _ = run(plan, {"x": x})
    assert out["l2"].tolist() == (x + 3.0).tolist()


def test_kernels_release_the_gil_and_keep_fastmath_off():
    plan = resolve(flow(disposable_income, affordability_ratio))
    (unit,) = _units(compile_plan(plan))
    assert unit.fn.targetoptions.get("nogil") is True
    assert not unit.fn.targetoptions.get("fastmath", False)


def score_row(row, params, consts):
    income, age = row
    return income * params.w + consts[0], age > 40


def score_row_python(row, params, consts):
    import math  # not compilable: runs in Python

    income, age = row
    return math.fsum((income * params.w, consts[0])), age > 40


@pytest.mark.parametrize("fn,kind", [(score_row, Kernel), (score_row_python, Fallback)])
def test_a_row_node_is_its_own_unit_called_with_row_params_and_consts(fn, kind, run):
    def before(income: float) -> float:
        return income * 2.0

    row = CallNode(_origin("p/score"), "row", fn, (Input("before", float), Input("age", int)),
                   (Output("score", float), Output("senior", bool)), (ParamDecl("w", float, 0.5),),
                   consts=(("offset", 10.0),))
    first = CallNode(_origin("p/before"), "scalar", before, (Input("income", float),), (Output("before", float),), ())
    plan = resolve(SequenceNode(_origin("p"), (first, row)))
    units = compile_plan(plan)
    assert units[0] is not units[1] and isinstance(units[1], kind)
    out, _ = run(plan, {"income": np.array([1.0, 3.0]), "age": np.array([30, 50])}, params={"p/score": {"w": 2.0}})
    assert out["score"].tolist() == [14.0, 22.0]
    assert out["senior"].tolist() == [False, True]


def test_a_row_node_with_an_optional_input_gets_none_where_invalid(run):
    def fill(row, params, consts):
        (x,) = row
        return (-1.0 if x is None else x,)

    node = CallNode(_origin("p/fill"), "row", fill, (Input("x", float | None, NullPolicy.OPTIONAL),),
                    (Output("filled", float),), ())
    out, _ = run(resolve(SequenceNode(_origin("p"), (node,))), {"x": np.array([1.0, 2.0])},
                 valid={"x": np.array([True, False])})
    assert out["filled"].tolist() == [1.0, -1.0]
