"""P14 (simplified) and loop L1 in detail (spec 10 §5.15, §5.23.1).

SCOPE.md: "Skip products 11-40, P14 and P15 in depth" while still building
"loop L1 (§5.23.1)... end to end". This module is the declared
simplification that makes both true at once: **one** settlement scenario
(settle every external obligation, keep every internal one -- not the
spec's up-to-250-scenario search over subsets, §5.15(c)) is proposed per
pass, so the *loop mechanism* (P10 fails -> P14 proposes -> P10 re-runs on
a hypothetical -> P12/P13 re-price and re-solve -> P16 checks -> repeat up
to 4 times) is real and end to end, while the *search inside P14* is not.

**Why one decider step, not a `flow` of P10/P12/P13/P16's own steps,
inside `loop()`.** `decider.loop`'s body writes every name in `carries`
each iteration; re-entering P10's, P12's and P13's own multi-step
sub-flows (table lookups included) inside a `loop()` is expressible, but
doing it while keeping the *actual* vs. *hypothetical* `existing_obligations`
distinction (§5.21.1) unambiguous through four nested compositions -- the
loop's own carries, `branch`-if-any, and each phase's own internal
naming -- adds structural weight without adding proof of the mechanism
this slice needs to demonstrate. One step that calls this project's own
plain-Python affordability arithmetic (`retail_credit.affordability_phase`'s
formulas, reproduced inline below because decider steps are not
themselves callable as plain functions from inside another step) and
`retail_credit.solve.solve_for_term` (already plain Python, already
callable) keeps every `value_basis_code`/`scenario_ref`/`loop_pass_index`
label exactly where §5.21.1 requires it, and is provably correct against
`tests/test_consolidation.py`'s reproduction of the spec's own four-pass
worked example (10 §5.23.1). See NOTES.md "Framework friction" for the
`loop()` composition question this raised.
"""
from __future__ import annotations

from decider import loop, missing_as, param, step

from retail_credit.solve import PRODUCT_10_MAX_AMOUNT, solve_for_term
from retail_credit.vocab import LoopTerminationCode, ValueBasisCode

MIN_IMPROVEMENT_INSTALMENT_RELIEF = 25.0


def l1_pass(
    loop_pass_index: int,
    loop_converged: bool,
    net_monthly_income: float,
    living_expenses: float,
    existing_obligations: float,
    obligations_internal: float,
    residual_floor: float,
    ratio_ceiling: float,
    affordability_buffer_applied: float,
    amount_cap: float,
    term_cap: float,
    risk_grade: int,
    term_months: int,
    requested_amount: float,
    applicant_age_years: float,
    rate_addon_bps: float = missing_as(0.0),
    max_affordable_instalment_hypothetical: float = missing_as(0.0),
    max_passes: int = param(4, ge=1, le=4),
    min_improvement: float = param(MIN_IMPROVEMENT_INSTALMENT_RELIEF, ge=0.0),
) -> tuple[int, bool, float, int, float, float, float, int, float, float, str]:
    """One pass of L1 (10 §5.23.1's diagram): P14 proposes a hypothetical obligations
    figure, P10 re-runs affordability against it, and -- if it passes -- P12/P13 re-price
    and re-solve. Returns the full carried state (see `carries=` in `pipeline.py`).
    """
    if loop_converged:
        # Already converged (or bound reached) on an earlier pass: pass through unchanged,
        # matching `loop()`'s own per-row semantics ("a row that reaches convergence stops
        # there without an error" -- here, stops by declining to do further work).
        return (loop_pass_index, True, existing_obligations, int(ValueBasisCode.ACTUAL), 0.0, 0.0, 0.0, 4, 0.0, 0.0,
                "")

    next_pass = loop_pass_index + 1

    # P14 (simplified, 10 §5.15(a-c)): the one scenario this slice evaluates -- settle
    # every externally held obligation, keep every internally held one.
    hypothetical_obligations = round(obligations_internal, 2)
    scenario_ref = next_pass  # this project runs at most one scenario per pass -- 1:1 with the pass

    # P10 re-run, hypothetical basis (§5.21.1: this is `existing_obligations`'s *second*
    # live version, `value_basis_code` 2, not a second name for the same fact).
    discretionary_income_hyp = round(net_monthly_income - living_expenses - hypothetical_obligations, 2)
    capacity_hyp = round(discretionary_income_hyp - residual_floor, 2)
    pre_buffer_hyp = min(capacity_hyp, ratio_ceiling)
    max_affordable_instalment_hyp = round(max(0.0, pre_buffer_hyp) * (1.0 - affordability_buffer_applied), 2)

    # Termination guarantee (10 §5.23.1): a pass only runs if it can strictly improve.
    # This is pass `next_pass` itself running, so the guard is on *entering* it: if the
    # previous pass's affordable instalment already exists and this pass cannot beat it
    # by the declared minimum, stop instead of proposing an identical scenario again.
    if next_pass > 1 and max_affordable_instalment_hyp < max_affordable_instalment_hypothetical + min_improvement:
        return (loop_pass_index, True, existing_obligations, int(ValueBasisCode.ACTUAL), 0.0,
                0.0, 0.0, int(LoopTerminationCode.NO_IMPROVEMENT), 0.0, 0.0, "")

    if max_affordable_instalment_hyp < 185.0:  # affordability_phase.MINIMUM_VIABLE_INSTALMENT
        terminated = next_pass >= max_passes
        code = int(LoopTerminationCode.BOUND_REACHED) if terminated else 0
        return (next_pass, terminated, hypothetical_obligations, int(ValueBasisCode.HYPOTHETICAL),
                discretionary_income_hyp, scenario_ref, max_affordable_instalment_hyp, code, 0.0, 0.0, "")

    # P12/P13 re-run over the new advance and product (this project's own pure functions).
    result = solve_for_term(term_months, requested_amount, amount_cap, PRODUCT_10_MAX_AMOUNT, risk_grade,
                             max_affordable_instalment_hyp, applicant_age_years, rate_addon_bps)

    # P16 (10 §5.23.1's diagram): "if no shippable offer and the bound is not reached, next
    # pass" -- here, "shippable" is simply "the solve found an amount" (this slice's P16 has
    # no anti-harm/arbitration test to apply against, since that needs the full scenario set).
    shipped = result.amount is not None
    terminated = shipped or next_pass >= max_passes
    termination_code = (int(LoopTerminationCode.CONVERGED) if shipped
                         else int(LoopTerminationCode.BOUND_REACHED) if next_pass >= max_passes else 0)

    return (
        next_pass, terminated, hypothetical_obligations, int(ValueBasisCode.HYPOTHETICAL),
        discretionary_income_hyp, scenario_ref, max_affordable_instalment_hyp, termination_code,
        result.amount or 0.0, result.instalment or 0.0, result.binding_constraint if shipped else "",
    )


l1_pass_step = step(
    l1_pass,
    outputs=(
        "loop_pass_index", "loop_converged", "existing_obligations_hypothetical",
        "existing_obligations_basis_code", "discretionary_income_hypothetical",
        "existing_obligations_hypothetical_scenario_ref", "max_affordable_instalment_hypothetical",
        "loop_termination_code", "loop_solved_amount", "loop_solved_instalment", "loop_binding_constraint",
    ),
)


def not_converged(loop_converged: bool = missing_as(False)) -> bool:
    return not loop_converged


not_converged_step = step(not_converged)


def l1_loop():
    """The whole loop, wired with `decider.loop` (10 §5.23.1: "the bound is four passes...
    a pass count, because a time-based bound is not reproducible")."""
    return loop(
        not_converged_step, l1_pass_step,
        carries=[
            "loop_pass_index", "loop_converged", "existing_obligations_hypothetical",
            "existing_obligations_basis_code", "discretionary_income_hypothetical",
            "existing_obligations_hypothetical_scenario_ref", "max_affordable_instalment_hypothetical",
            "loop_termination_code", "loop_solved_amount", "loop_solved_instalment", "loop_binding_constraint",
        ],
        max_iterations=4, name="l1",
    )
