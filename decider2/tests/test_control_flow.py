"""`Branch` and `Loop` — doc 03 §8.2/§8.3.

Every construct-under-test is run through `assert_equivalent` at least
once: interpreted/stepped/fused/score is doc 02 §3.1's core correctness
test, and a Branch/Loop that only works in one mode is not built (this
agent's own brief, verbatim).

**Every condition/arm/body function below is module-level, with no leading
underscore.** Two real constraints discovered while writing this file, both
now documented in this agent's report:

- Branch/Loop generate real, content-addressed source that IMPORTS the
  condition/arm/body functions by name (`decider2.compile.cache`, the same
  mechanism kernel/tree codegen already uses) — a function defined inside
  another function (a test-local closure) has no module-level name that
  import can reach, the same pre-existing limitation
  `decider2.compile.driver` documents for a fused kernel's own steps.
- `param()` fields are namespaced by the arm/body's own module instance
  name, prefixed onto the field (doc 03 §4.1, one level up) — and a module
  instance name derived from a leading-underscore function name produces a
  leading-underscore pydantic field name, which pydantic rejects outright
  as a private-attribute name. Ordinary (non-underscore) names sidestep it;
  nothing stops a real project's OWN naming convention from hitting it, so
  it is reported as a rough edge, not fixed here.
"""
from __future__ import annotations

import polars as pl
import pytest

from decider2 import Branch, Loop, flow, module, param, step
from decider2.testing import assert_equivalent

# ---------------------------------------------------------------------------
# Branch fixtures (module-level; see module docstring)
# ---------------------------------------------------------------------------


def is_private(is_private_sector: bool) -> bool:
    """Trivial condition reading a precomputed bool — a str leaf cannot
    cross a Branch/Loop node boundary in this build (see
    decider2.graph.control_flow._engine.merge_inputs's null-policy check
    and this agent's report)."""
    return is_private_sector


@step(output="is_private_sector")
def sector_is_private(sector: str, private: str = param("private")) -> bool:
    return sector == private


@step(output="term_cap")
def cap_for_private(term_cap: float, private_cap: float = param(48.0, ge=6, le=60)) -> float:
    return min(term_cap, private_cap)


@step(output="term_cap")
def cap_for_public(term_cap: float, public_cap: float = param(60.0, ge=6, le=60)) -> float:
    return min(term_cap, public_cap)


def public_noop(is_private_sector: bool) -> bool:
    """An arm that never mentions term_cap at all — doc 03 §8.2's "an arm
    may stay silent about a name, and then that name passes through
    unchanged"."""
    return is_private_sector


@step(output="term_cap")
def cap_for_private_as_int(term_cap: float, private_cap: float = param(48.0)) -> int:
    return int(min(term_cap, private_cap))


def band_index(band: int) -> int:
    return band


@step(output="price")
def band0(price: float) -> float:
    return 1.0 + 0.0 * price


@step(output="price")
def band1(price: float) -> float:
    return 2.0 + 0.0 * price


@step(output="price")
def band2(price: float) -> float:
    return 3.0 + 0.0 * price


def term_cap_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "sector": ["private", "public", "private"],
            "term_cap": [60.0, 60.0, 30.0],
        }
    )


def test_branch_both_arms_write_the_modifies_value():
    branch = Branch(
        is_private, module(cap_for_private), module(cap_for_public),
        modifies=["term_cap"], name="term_cap_rules_both",
    )
    pipeline = flow(sector_is_private, branch)
    out = pipeline.apply(term_cap_frame(), mode="interpreted")
    assert out["term_cap"].to_list() == [48.0, 60.0, 30.0]
    assert_equivalent(pipeline, term_cap_frame())


def test_branch_one_arm_stays_silent_and_passes_through():
    branch = Branch(
        is_private, module(cap_for_private), module(public_noop),
        modifies=["term_cap"], name="term_cap_rules_silent",
    )
    pipeline = flow(sector_is_private, branch)
    out = pipeline.apply(term_cap_frame(), mode="interpreted")
    # private (index 0) is capped at 48; public (index 1) is silent about
    # term_cap, so it passes the incoming value through unchanged (60.0).
    assert out["term_cap"].to_list() == [48.0, 60.0, 30.0]
    assert_equivalent(pipeline, term_cap_frame())


def test_branch_x_path_reports_the_arm_that_fired():
    branch = Branch(
        is_private, module(cap_for_private), module(cap_for_public),
        modifies=["term_cap"], name="term_cap_rules_path_test",
    )
    pipeline = flow(sector_is_private, branch).emit("term_cap_rules_path_test_path")
    out = pipeline.apply(term_cap_frame(), mode="interpreted")
    assert out["term_cap_rules_path_test_path"].to_list() == [0, 1, 0]


def test_branch_cross_arm_type_disagreement_is_a_build_error():
    with pytest.raises(ValueError, match="disagree on the type"):
        Branch(
            is_private, module(cap_for_private_as_int), module(cap_for_public),
            modifies=["term_cap"], name="mismatched_types",
        )


def test_branch_routing_form_selects_by_index():
    branch = Branch(
        band_index, [module(band0), module(band1), module(band2)],
        modifies=["price"], name="price_by_band",
    )
    pipeline = flow(branch).emit("price_by_band_path")
    frame = pl.DataFrame({"band": [0, 1, 2, 1], "price": [0.0, 0.0, 0.0, 0.0]})
    out = pipeline.apply(frame, mode="interpreted")
    assert out["price"].to_list() == [1.0, 2.0, 3.0, 2.0]
    assert out["price_by_band_path"].to_list() == [0, 1, 2, 1]
    assert_equivalent(pipeline, frame)


def test_branch_needs_name():
    with pytest.raises(ValueError, match="needs name="):
        Branch(is_private, module(cap_for_private), module(cap_for_public), modifies=["term_cap"], name="")


def test_branch_needs_modifies():
    with pytest.raises(ValueError, match="needs modifies"):
        Branch(is_private, module(cap_for_private), module(cap_for_public), modifies=[], name="x")


# ---------------------------------------------------------------------------
# Loop fixtures
# ---------------------------------------------------------------------------


def should_continue_offer(best_offer: float, loop_idx: int, target: float = param(90.0)) -> bool:
    return best_offer < target and loop_idx < 1000


@step(output="best_offer")
def offer_step(best_offer: float, loop_idx: int, increment: float = param(5.0)) -> float:
    return best_offer + increment + 0.0 * loop_idx


def should_stop_at_3(iterations_run: int) -> bool:
    return iterations_run < 3


@step(output="iterations_run")
def tick(iterations_run: int) -> int:
    return iterations_run + 1


@step(output="best_offer")
def offer_step_missing_output(best_offer: float, loop_idx: int) -> float:
    return best_offer + 1.0 + 0.0 * loop_idx


@step(output="best_offer")
def offer_step_no_self_read(loop_idx: int) -> float:
    return 1.0 + 0.0 * loop_idx


def should_continue_score(best_score: float, loop_idx: int, target: float = param(90.0)) -> bool:
    return best_score < target and loop_idx < 1000


@step(output="best_offer")
def offer_step_two_carry(best_offer: float, loop_idx: int, increment: float = param(5.0)) -> float:
    return best_offer + increment + 0.0 * loop_idx


@step(output="best_score")
def score_step_two_carry(best_score: float) -> float:
    return best_score + 1.0


def best_offer_loop(*, max_iterations: int = 511) -> object:
    return Loop(
        should_continue_offer, module(offer_step),
        carries=["best_offer"], max_iterations=max_iterations, name="best_offer_loop",
    )


def test_loop_runs_to_its_condition():
    loop = best_offer_loop()
    pipeline = flow(loop)
    frame = pl.DataFrame({"best_offer": [0.0, 85.0, 88.0], "id": [1, 2, 3]})
    out = pipeline.apply(frame, mode="interpreted")
    # 0 -> 90 in 18 steps of 5; 85 -> 90 in 1; 88 -> 93 (overshoots once).
    assert out["best_offer"].to_list() == [90.0, 90.0, 93.0]
    assert_equivalent(pipeline, frame)


def test_loop_hits_max_iterations():
    # target defaults to 90.0 but max_iterations=3 caps it at 3*5=15 well
    # short of convergence — no exception, just the bound (doc 03 §8.3).
    loop = best_offer_loop(max_iterations=3)
    pipeline = flow(loop)
    frame = pl.DataFrame({"best_offer": [0.0], "id": [1]})
    out = pipeline.apply(frame, mode="interpreted")
    assert out["best_offer"].to_list() == [15.0]
    assert_equivalent(pipeline, frame)


def test_loop_early_exit_does_not_run_the_full_iteration_count():
    # carries the iteration count itself (a pure, self-incrementing step) so
    # the test can observe, from the RETURNED value alone, that early exit
    # genuinely stopped the loop well short of max_iterations.
    loop = Loop(
        should_stop_at_3, module(tick),
        carries=["iterations_run"], max_iterations=500, name="tick_loop",
    )
    pipeline = flow(loop)
    frame = pl.DataFrame({"iterations_run": [0], "id": [1]})
    out = pipeline.apply(frame, mode="interpreted")
    assert out["iterations_run"].to_list() == [3]
    assert out["iterations_run"][0] < 500
    assert_equivalent(pipeline, frame)


def test_loop_body_must_produce_every_carries_value():
    with pytest.raises(ValueError, match="never produces it"):
        Loop(
            should_continue_offer, module(offer_step_missing_output),
            carries=["best_offer", "best_score"], max_iterations=10, name="incomplete_loop",
        )


def test_loop_carries_must_be_a_body_leaf_too():
    with pytest.raises(ValueError, match="does not read it as an input"):
        Loop(
            should_continue_offer, module(offer_step_no_self_read),
            carries=["best_offer"], max_iterations=10, name="no_self_read_loop",
        )


def should_continue_reads_non_carry_output(instalment: float, loop_idx: int) -> bool:
    return instalment > 100.0 and loop_idx < 1000


@step(output="instalment")
def instalment_step(term_cap: float) -> float:
    return 1000.0 / term_cap


@step(output="term_cap")
def term_cap_step(term_cap: float, loop_idx: int) -> float:
    return term_cap + 1.0 + 0.0 * loop_idx


def test_loop_should_continue_cannot_read_a_non_carry_body_output():
    """should_continue is checked before the body's very first iteration —
    a body output that is not itself a carry has no value yet at that
    point (a real gap found while building this: see this agent's
    report)."""
    with pytest.raises(ValueError, match="has no value before the body's first iteration"):
        Loop(
            should_continue_reads_non_carry_output,
            module(term_cap_step, instalment_step, name="term_cap_and_instalment"),
            carries=["term_cap"], max_iterations=10, name="bad_should_continue_loop",
        )


def test_loop_max_iterations_is_required():
    with pytest.raises(TypeError):
        Loop(should_continue_offer, module(offer_step), carries=["best_offer"], name="missing_bound")  # type: ignore[call-arg]


def test_loop_max_iterations_must_be_a_positive_int():
    with pytest.raises(ValueError, match="positive int"):
        Loop(
            should_continue_offer, module(offer_step),
            carries=["best_offer"], max_iterations=0, name="zero_bound",
        )


def test_loop_two_mutually_needed_carries_is_a_clear_build_error():
    """Doc 03 §8.3's own worked example carries two names sharing one loop
    run. This agent's report explains why that cannot be built safely in
    this pass — every carry's generated function ends up needing every
    OTHER carry's pre-loop value too (should_continue runs regardless of
    target), which is an unbreakable cycle within one decider2 Module. The
    useful, testable behaviour is that this fails LOUDLY at build time,
    naming the carries, rather than compiling into a silently wrong
    answer."""
    with pytest.raises(ValueError, match=r"carries=.*has 2\+ names"):
        Loop(
            should_continue_score,
            module(offer_step_two_carry, score_step_two_carry, name="offer_and_score"),
            carries=["best_offer", "best_score"], max_iterations=511, name="two_carry_loop",
        )


# ---------------------------------------------------------------------------
# Nesting: Loop(Branch(...)) and Branch(..., Loop(...))
# ---------------------------------------------------------------------------


def high_risk(risk_flag: bool) -> bool:
    return risk_flag


def inner_should_continue(rate: float, loop_idx: int, cap: float = param(9.0)) -> bool:
    return rate < cap and loop_idx < 1000


@step(output="rate")
def bump(rate: float, loop_idx: int, step_size: float = param(3.0)) -> float:
    return rate + step_size + 0.0 * loop_idx


@step(output="rate")
def flat_rate(rate: float, flat: float = param(1.0)) -> float:
    return flat + 0.0 * rate


def outer_should_continue(rate: float, loop_idx: int, cap: float = param(50.0)) -> bool:
    return rate < cap and loop_idx < 1000


def test_nested_loop_of_branch_assert_equivalent():
    """`Loop(Branch(steps1, Loop(steps2)))`'s shape, worked: the outer Loop's
    body is a Branch, and one of that Branch's own arms is itself a Loop —
    two levels of nesting, both directions, in one pipeline."""
    inner_loop = Loop(
        inner_should_continue, module(bump),
        carries=["rate"], max_iterations=20, name="inner_loop",
    )  # a Loop used as one arm of a Branch

    risk_branch = Branch(
        high_risk, inner_loop, module(flat_rate),
        modifies=["rate"], name="risk_branch",
    )  # a Branch whose arm is a Loop

    outer_loop = Loop(
        outer_should_continue, risk_branch,
        carries=["rate"], max_iterations=20, name="outer_loop",
    )  # a Loop whose body is a Branch containing a Loop

    pipeline = flow(high_risk, outer_loop)
    frame = pl.DataFrame({"risk_flag": [True, False], "rate": [0.0, 0.0]})
    out = pipeline.apply(frame, mode="interpreted")
    # high risk: the inner loop converges to its own cap (9.0) on the
    # outer loop's first iteration and is idempotent there afterwards (a
    # fixed point, since 9.0 is no longer < should_continue's own 9.0).
    # low risk: flat_rate is idempotent from its first application (1.0
    # regardless of the incoming rate). Neither reaches the OUTER loop's
    # own cap (50.0) — both are genuinely stuck at an inner fixed point —
    # which is exactly the scenario worth pinning: the outer loop still
    # runs to its OWN max_iterations bound without error, and both nested
    # constructs still agree across all three execution modes.
    assert out["rate"].to_list() == [9.0, 1.0]
    assert_equivalent(pipeline, frame)
