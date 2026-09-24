"""EP-1 -- new-to-bank business application (spec 11 §5.1, §5.3, products 50 and
51), built over project 05's whole assessment (unmodified: structure, events,
roll-up, entity scoring, people blend, financial ratios, combined grade,
pricing, sole-proprietor affordability -- 05's `pipeline.build()`, one call)
plus this project's own facility identity, DSCR covenant binding, and the
re-pricing vocabulary L1 will need against this decision as predecessor.

This is the pipeline `pipeline.py` serves (SCOPE.md's "one real path").
"""
from __future__ import annotations

from decider import dag

from business_credit_e2e import covenant, vocab
from business_credit_e2e.reuse import load_project05_pipeline


def new_facility_instalment(product_code: int, instalment: float, offered_amount: float) -> float:
    """Product 50's instalment comes straight from 05's pricing unit (a term
    loan, unmodified). Product 51 (revolving) has no instalment in 05's
    single-product-50-shaped pricing (05 NOTES.md: "take structuring and
    pricing only as far as a single product 50 lookup") -- a **declared gap**
    (§5.17.2), resolved here by **composing around it**: project 07's
    `notional_instalment` (the client's committed payment at the new limit, at
    the contractual minimum-payment rate) rather than forking 05's pricing to
    add a second product shape. See NOTES.md "Reuse" for why compose, not
    extend or fork, was the right call for this specific gap."""
    if product_code in vocab.REVOLVING_PRODUCTS:
        from limit_mgmt.affordability import notional_instalment  # project 07, real call
        return notional_instalment(offered_amount, product_code)
    return instalment


def assessment_kind_code() -> int:
    return vocab.EP1_NEW_TO_BANK


def origination_comparison_basis_code() -> int:
    """§5.10.4: every decision of record carries a `comparison_basis_code`
    against its predecessor. At origination there is no predecessor -- this
    project's own extension of the three-value enum (`vocab.COMPARISON_ORIGINATION`),
    not a fourth spec value, purely so the field is never silently absent."""
    return vocab.COMPARISON_ORIGINATION


def predecessor_decision_of_record_id() -> int | None:
    return None


def build():
    """`project05.build()` is composed **unmodified** -- the whole point of this
    slice (SCOPE.md: "origination (EP-1) for products 50 and 51 **over 05's
    components**"). Everything below it is this project's own: facility
    identity fields, and the DSCR covenant bound at this decision.

    `facility_id`, `knowledge_date` and `existing_accounts` are request
    inputs nothing here produces, so they are deliberately **not** named in
    `.emit()` -- naming an untouched pass-through column in an outer `dag`
    wrapping an already-`.emit()`-ed sub-dag raises a `WiringError` even
    though the column survives to the output regardless (project 02's
    NOTES.md "Framework friction" 4.4, reproduced identically here)."""
    project05 = load_project05_pipeline()
    return dag(
        project05.build(),
        new_facility_instalment,
        assessment_kind_code,
        origination_comparison_basis_code,
        predecessor_decision_of_record_id,
        covenant.covenant_definition_version,
        covenant.covenant_threshold,
        covenant.covenant_cure_days,
        covenant.covenant_instance_id,
        covenant.dscr_debt_service_step,
        covenant.dscr_total_debt_service,
        covenant.measured_dscr,
        covenant.dscr_headroom_ratio,
        covenant.dscr_breach_class_code,
        name="business_credit_e2e_origination",
    ).emit(
        "assessment_kind_code", "origination_comparison_basis_code", "predecessor_decision_of_record_id",
        "covenant_instance_id", "covenant_definition_version", "covenant_threshold", "covenant_cure_days",
        "new_facility_instalment", "dscr_debt_service", "dscr_total_debt_service", "measured_dscr",
        "dscr_headroom_ratio", "dscr_breach_class_code",
    )
