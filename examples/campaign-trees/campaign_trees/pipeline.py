"""One client's evaluation for campaign 23 (the Flex Loan pre-approved top-up): the assignment record.

`decikit build` and `decikit serve` use `build`, which receives the tree document `configs/<version>/campaign_23.json`,
the rate card document `configs/<version>/rate_card_flex_loan.json` and the cap reductions
`configs/<version>/cap_reductions.json`.
"""
from __future__ import annotations

from decikit import allow_fallback, branch, flow, missing_as, param, step

from credit_core import eligibility
from credit_core.appetite import build_appetite_table
from credit_core.consent import channel_permitted_step, consent_verdict_step

from campaign_trees import holdout, overlays, preassessment, suppressions, vocab

TREE_OUTPUTS = ("leaf", "tree_path", "leaf_outcome_code", "offer_tier_code", "tier_ceiling", "channel_1", "channel_2",
                "priority_weight", "reason_label")

# What the evaluation writes. A client kept out of it (absolute suppression, or not eligible) has these null.
EVALUATED = (
    "appetite_max_amount", "appetite_cell_id", "max_affordable_instalment", "pre_assessed_amount", "pre_assessed_term",
    "binding_constraint_code", "pre_assessment_evaluations", "pre_assessment_valid_from", "pre_assessment_valid_to",
    *TREE_OUTPUTS, *(f"unadjusted_{name}" for name in TREE_OUTPUTS),
    "advertised_tier_ceiling", "tier_ceiling_before_cap_overlay", "cap_adjustment_set_id", "cap_adjustments_applied",
    "offer_amount",
)


@allow_fallback
def overlay_stack_id(overlay_stack_id: str = param("OS-NONE")) -> str:
    """Which overlays set the tree's params; `overlays.resolve` writes it to the params document with them."""
    return overlay_stack_id


def evaluated(is_eligible: bool, suppressed_absolute: bool) -> bool:
    return is_eligible and not suppressed_absolute


def offer_amount(pre_assessed_amount: float, advertised_tier_ceiling: float) -> float:
    """A campaign may advertise less than the pre-assessed amount, never more."""
    return round(min(pre_assessed_amount, advertised_tier_ceiling), 2)


@allow_fallback
def assignment_outcome_code(
    evaluated: bool, suppression_codes: list[str], is_control: bool, is_universal_holdout: bool,
    leaf_outcome_code: int = missing_as(vocab.DO_NOT_TARGET),
) -> int:
    if not evaluated or leaf_outcome_code != vocab.TARGET:
        return vocab.DO_NOT_TARGET
    if suppression_codes:
        return vocab.SUPPRESSED
    return vocab.HOLD_OUT if is_control or is_universal_holdout else vocab.TARGET


def advertised_amount(assignment_outcome_code: int, offer_amount: float = missing_as(0.0)) -> float:
    return offer_amount if assignment_outcome_code == vocab.TARGET else 0.0


def build(campaign_23, rate_card_flex_loan, cap_reductions):
    # The tree runs twice from one document: with the params document (overlays applied) and with the published
    # thresholds, so the unadjusted answer is recorded beside the adjusted one.
    adjusted = campaign_23.named("tree_adjusted")
    unadjusted = campaign_23.named("tree_unadjusted").relabel(writes={name: f"unadjusted_{name}" for name in TREE_OUTPUTS})

    appetite = build_appetite_table().relabel(writes={
        "max_amount": "appetite_max_amount", "max_term": "appetite_max_term", "max_ratio": "appetite_max_ratio",
        "min_price": "appetite_min_price", "cell_id": "appetite_cell_id"})

    cap_reduction = cap_reductions.relabel(
        reads={"decision_date": "cycle_date"},
        writes={"adjustment_set_id": "cap_adjustment_set_id", "adjustments_applied": "cap_adjustments_applied"})

    # The tree reads pre_assessed_amount, so the amount is settled first.
    evaluation = flow(
        appetite, step(preassessment.max_affordable_instalment), preassessment.pre_assessment_step(rate_card_flex_loan),
        adjusted, unadjusted, cap_reduction, step(offer_amount),
        name="evaluation")

    return flow(
        eligibility.decline_reason_codes_step.relabel(writes={"decline_reason_codes": "eligibility_decline_reasons"}),
        eligibility.is_eligible_step.relabel(reads={"decline_reason_codes": "eligibility_decline_reasons"}),
        consent_verdict_step,
        channel_permitted_step,
        *suppressions.steps,
        step(overlay_stack_id),
        step(evaluated),
        branch(evaluated, evaluation, modifies=list(EVALUATED), name="evaluate"),
        holdout.is_control, holdout.is_universal_holdout,
        step(assignment_outcome_code), step(advertised_amount),
        name="campaign_trees",
    ).emit(
        "eligibility_decline_reasons", "is_eligible", "consent_verdict", "channel_permitted", "suppression_codes",
        "suppressed_absolute", "overlay_stack_id", "evaluated", *EVALUATED, "is_control", "is_universal_holdout",
        "assignment_outcome_code", "advertised_amount",
    )
