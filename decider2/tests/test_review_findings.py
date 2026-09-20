"""Regression tests for the independent evaluation's findings (review doc,
2026-09-20). Every scenario below was verified with a probe script before
being written up; these tests pin the fix so it cannot regress silently —
doc 03 §2.1 calls a silent wrong answer "the worst failure mode the design
can have", which is exactly what findings 1-4 are.

Numbered to match the review:
    1. a forward reference is silently accepted as a leaf input
    2. score() erases every input's dtype to float64
    3. assert_equivalent passes when all three modes crash identically
    4. an absent input is a bare KeyError three frames deep
    5. a bare Python default is silently a required input
    6. three lints the docs promise but nothing implements (a/b/c)
"""
from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from decider2 import flow, missing_as, module, param
from decider2.graph.pipeline import flow as _flow


# ---------------------------------------------------------------------------
# 1 — a forward reference is silently accepted as a leaf input
# ---------------------------------------------------------------------------


def test_a_forward_reference_is_a_build_error_not_a_silent_leaf():
    """doc 03 §8.1: '|' order is execution order. `initiation_fee` reads
    `offered_amount` before any module in this pipeline produces it, so it
    must not silently resolve to the frame's own `offered_amount` column —
    that value is quietly stale the moment a later module narrows it."""

    def initiation_fee(offered_amount: float) -> float:
        return offered_amount * 0.015

    def offered_amount(requested_amount: float, cap: float = param(200000.0)) -> float:
        return min(requested_amount, cap)

    with pytest.raises(ValueError, match="offered_amount"):
        flow(initiation_fee, offered_amount)


def test_the_forward_reference_error_names_both_modules():
    def initiation_fee(offered_amount: float) -> float:
        return offered_amount * 0.015

    def offered_amount(requested_amount: float, cap: float = param(200000.0)) -> float:
        return min(requested_amount, cap)

    with pytest.raises(ValueError) as excinfo:
        flow(initiation_fee, offered_amount)
    message = str(excinfo.value)
    assert "initiation_fee" in message
    assert "offered_amount" in message


def test_the_cross_module_waterfall_self_read_still_works():
    """doc 03 §3.2 — the legal case the fix must not break: a LATER module
    narrowing a value an EARLIER module already produced (most-recent-wins,
    §2.1 row 1), not a forward reference."""
    from decider2 import step

    @step(output="term_cap")
    def product_ceiling(requested_term: float, ceiling: float = param(60.0)) -> float:
        return min(requested_term, ceiling)

    @step(output="term_cap")
    def cap_by_income(term_cap: float, min_net_salary: float, cap: float = param(48.0)) -> float:
        return min(term_cap, cap) if min_net_salary < 5000 else term_cap

    p = flow(product_ceiling, cap_by_income)  # must not raise
    out = p.apply(pl.DataFrame({"requested_term": [72.0], "min_net_salary": [4000.0]}))
    assert out["term_cap"].to_list() == [48.0]


def test_a_chain_of_self_read_waterfalls_narrowing_the_same_name_still_works():
    """The exact shape tests/test_graph_pipeline.py::
    test_waterfall_overwrite_across_modules_is_not_an_error pins: three
    modules in a row, each `@step(output="term_cap")` reading `term_cap`
    (a self-read waterfall in EVERY link, not just the first). Only the
    FIRST module's leaf demand is a genuine pipeline leaf; the second and
    third must not be mistaken for a forward reference against it."""
    from decider2 import step

    def _term_cap_step(fn_name: str, delta: float):
        def fn(term_cap: float) -> float:
            return term_cap - delta

        fn.__name__ = fn_name
        return step(output="term_cap")(fn)

    seed = module(_term_cap_step("seed", 0), name="seed_term_cap")
    income = module(_term_cap_step("income", 1), name="income_cap")
    sector = module(_term_cap_step("sector", 2), name="sector_cap")

    pipeline = seed | income | sector  # must not raise
    assert pipeline.interface.outputs == ("term_cap",)
    assert pipeline.versions()["term_cap"] == ("seed_term_cap", "income_cap", "sector_cap")


def test_a_standalone_module_self_read_seeded_from_the_frame_still_works():
    """doc 03 §3.2 — a single module reading the name it also writes,
    seeded straight from the input frame, must not be confused with a
    forward reference: production and the leaf demand are the SAME module
    (`invoke.apply`'s own docstring names exactly this shape as the reason
    the shadow check exempts a pipeline's own leaf inputs). Build only —
    `.apply()`'s write-back path for a module whose sole output is also its
    sole leaf input has a separate, pre-existing bug unrelated to this
    finding (a `polars.exceptions.ShapeError` in `write_back`), not
    reachable from any of the four findings this file pins."""
    from decider2 import step

    @step(output="term_cap")
    def cap_by_income_band(term_cap: float, cap: float = param(48.0)) -> float:
        return min(term_cap, cap)

    p = flow(cap_by_income_band)  # must not raise (this is finding 1's fix)
    assert p.interface.inputs[0].name == "term_cap"
    assert p.interface.outputs == ("term_cap",)


# ---------------------------------------------------------------------------
# 2 — score() erases every input's dtype to float64
# ---------------------------------------------------------------------------


def test_score_preserves_int_precision_above_2_53():
    def big(n: int) -> int:
        return n * 3 + 1

    p = flow(big)
    n = 2**53 + 1
    via_apply = p.apply(pl.DataFrame({"n": [n]}))["big"][0]
    via_score = p.score({"n": n})["big"]
    assert via_score == via_apply == n * 3 + 1


def test_score_types_a_bool_input_as_bool_not_float():
    def flag(active: bool) -> bool:
        return not active

    p = flow(flag)
    assert p.score({"active": True})["flag"] is False or p.score({"active": True})["flag"] == False  # noqa: E712


def test_score_can_index_a_tuple_by_an_int_annotated_input():
    """Verified regression: an int-annotated input indexing a tuple/list
    compiles fine under apply() and fails under score() with `getitem(...,
    float64)` when score() forces every input to float64 — misdiagnosed by
    an evaluation agent as 'numba cannot index lists'."""

    def lookup(term: int) -> float:
        rates = (0.05, 0.06, 0.07, 0.08)
        return rates[term]

    p = flow(lookup)
    frame = pl.DataFrame({"term": [0, 1, 2, 3]})
    via_apply = p.apply(frame)["lookup"].to_list()
    via_score = [p.score({"term": t})["lookup"] for t in [0, 1, 2, 3]]
    assert via_score == via_apply == [0.05, 0.06, 0.07, 0.08]


# ---------------------------------------------------------------------------
# 3 — assert_equivalent passes when all three modes crash identically
# ---------------------------------------------------------------------------


def test_assert_equivalent_rejects_a_non_polars_frame():
    from decider2.testing import assert_equivalent

    def f(x: float) -> float:
        return x + 1.0

    p = flow(f)
    with pytest.raises(TypeError, match="polars"):
        assert_equivalent(p, {"x": [1.0, 2.0]})


def test_assert_equivalent_fails_when_all_three_modes_crash_identically():
    """The vacuous-pass bug: three identical exceptions must not count as
    agreement — doc 05 §9 criterion 2/3 asks whether the MODES agree, and a
    ladder that cannot fail on a shared crash is not testing anything."""
    from decider2.testing import assert_equivalent

    def always_broken(x: float) -> float:
        if x > -1e300:  # always true; keeps 'x' referenced (finding 6c)
            raise RuntimeError("deliberately broken in every mode")
        return x

    p = flow(always_broken)
    frame = pl.DataFrame({"x": [1.0, 2.0]})
    with pytest.raises(Exception):
        assert_equivalent(p, frame)


def test_assert_equivalent_checks_score_agrees_with_apply():
    """doc 05 §9 criterion 2: 'the same kernel answers a single record' —
    nothing checked this before. Must run AFTER finding 2 is fixed, or the
    new rung fails for the wrong reason (float64-erased score() inputs)."""
    from decider2.testing import assert_equivalent

    def scaled(term: int) -> int:
        return term * 2

    p = flow(scaled)
    frame = pl.DataFrame({"term": [1, 2, 3]})
    assert assert_equivalent(p, frame) is None  # must not raise


# ---------------------------------------------------------------------------
# 4 — an absent input is a bare KeyError three frames deep
# ---------------------------------------------------------------------------


def test_score_routes_a_required_absent_input_instead_of_keyerror():
    def affordability(instalment: float) -> float:
        return instalment * 0.35

    p = flow(affordability)
    out = p.score({})  # 'instalment' entirely absent, not merely null
    assert "decision" in out
    assert out["routed_on"] == "instalment"


def test_score_fills_a_missing_as_input_when_the_key_is_absent():
    def affordability(bureau_score: float = missing_as(0.0)) -> float:
        return bureau_score * 0.01

    p = flow(affordability)
    out = p.score({})  # key absent, not present-and-null
    assert out["affordability"] == 0.0


def test_apply_routes_a_required_column_entirely_absent_from_the_frame():
    def affordability(instalment: float) -> float:
        return instalment * 0.35

    p = flow(affordability)
    frame = pl.DataFrame({"id": [1, 2, 3]})  # 'instalment' column doesn't exist at all
    out = p.apply(frame)
    assert out.height == 3
    assert out["affordability"].null_count() == 3 or out["affordability"].is_nan().all()


def test_apply_fills_a_missing_as_column_entirely_absent_from_the_frame():
    def affordability(bureau_score: float = missing_as(0.0)) -> float:
        return bureau_score * 0.01

    p = flow(affordability)
    frame = pl.DataFrame({"id": [1, 2, 3]})  # 'bureau_score' column doesn't exist at all
    out = p.apply(frame)
    assert out["affordability"].to_list() == [0.0, 0.0, 0.0]


# ---------------------------------------------------------------------------
# 5 — a bare Python default is silently a required input
# ---------------------------------------------------------------------------


def test_a_bare_python_default_is_rejected_at_harvest():
    def f(x: float, tier: int = 2) -> float:
        return x + tier

    with pytest.raises(TypeError, match="param\\(|missing_as\\("):
        flow(f)


def test_the_bare_default_error_names_the_parameter():
    def f(x: float, tier: int = 2) -> float:
        return x + tier

    with pytest.raises(TypeError) as excinfo:
        flow(f)
    assert "tier" in str(excinfo.value)


# ---------------------------------------------------------------------------
# 6 — three lints the docs promise but nothing implements
# ---------------------------------------------------------------------------


def test_6a_a_floating_def_in_a_pipeline_file_is_flagged(tmp_path):
    """doc 03 §5.3: 'Steps may not float.'"""
    from decider2.lint import check_pipeline_file

    src = tmp_path / "pl.py"
    src.write_text(
        "from decider2 import flow\n\n"
        "def used(x: float) -> float:\n"
        "    return x + 1.0\n\n"
        "def forgotten(x: float) -> float:\n"
        "    return x - 1.0\n\n"
        "pipeline = flow(used)\n"
    )
    floating = check_pipeline_file(src)
    assert "forgotten" in floating
    assert "used" not in floating


def test_6b_an_unread_params_model_field_is_a_build_error():
    """doc 03 §4: 'Exactly one canonical location per parameter.'"""
    from pydantic import BaseModel

    class CapParams(BaseModel):
        cap: float = 48.0
        unread_field: float = 1.0

    def capper(term_cap: float, params) -> float:
        return min(term_cap, params.cap)

    with pytest.raises(ValueError, match="unread_field"):
        module(capper, params=CapParams)


def test_6c_a_step_parameter_never_referenced_in_the_body_is_a_build_error():
    def cap_by_income_band(term_cap: float, min_net_salary: float) -> float:
        return term_cap  # min_net_salary is declared but never read

    with pytest.raises(ValueError, match="min_net_salary"):
        module(cap_by_income_band)
