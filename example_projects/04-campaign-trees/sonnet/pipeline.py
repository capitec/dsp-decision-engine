"""Campaign 23 (Flex Loan pre-approved top-up) -- spec 04 §5.3.3's worked example, served
end to end: suppressions (§5.2) -> the tree, twice, adjusted and unadjusted (§5.3-§5.4) ->
the cap-reduction overlay over the tree's tier ceiling (§5.3.4) -> pre-assessment (§5.5,
reusing project 03's published solve/pricing) -> eligibility and consent (`core.eligibility`,
`core.consent`) -> appetite (`core.appetite`) -> holdout/control assignment (§5.7) -> one
assignment record (§5.8).

**One campaign, not sixty.** `generate_trees.py` proves the framework holds at the spec's
real scale (60 trees, ~9 200 nodes, SCOPE.md's "keep the tree count and size near real") and
`tests/` runs the node identity map, arbitration and the §5.9 validator against that scale;
this pipeline serves the one campaign this project builds end to end, exactly as 00 and 03's
own demo pipelines wire a representative slice rather than everything they built (see their
NOTES.md "What I built" for the same choice).

**`pre_assessed_amount` is read inside the tree** (campaign 23's nodes 7-8, §5.3.3), which
means Stage 5 (amount determination) must run *before* Stage 3 (the tree) for this campaign
even though the spec numbers them the other way -- `dag()` resolves this from each step's
own reads/writes, not from the stage numbers, so it is simply correct without this pipeline
having to say so anywhere. See NOTES.md "Spec problems" for the numbering itself.
"""
from __future__ import annotations

from datetime import date

from decider import dag, param, step
from decider.steps.trees import TreeConfig

from credit_core import eligibility
from credit_core.appetite import build_appetite_table
from credit_core.consent import channel_permitted_step, consent_verdict_step

from campaign_trees import holdout, overlays, preassessment, suppressions, tree_model
from campaign_trees.campaign23 import TREE_V1
from loan_granting.pricing import CreditLifeIndex, RateCardIndex

TREE_NAME = TREE_V1.name
STATUTORY_CEILING = 0.28  # 03's own NCA-margin-over-repo ceiling, reused as a literal here (see NOTES.md)


# --- Trivial pass-throughs: `emit()` only selects columns a step produced, never a raw
# input directly (see NOTES.md "Framework friction") -- one echo step per identifier this
# project's evidence record must carry untouched. ---

def _echo_step(name: str, py_type: type):
    out_name = f"{name}__echo"

    def fn(value):
        return value

    fn.__name__ = f"echo_{name}"
    fn.__annotations__ = {"value": py_type, "return": py_type}
    return step(fn, output=out_name).relabel(reads={"value": name}, writes={out_name: name})


# --- Stage 2: suppressions ---

def _consent_permitted(channel_permitted: bool) -> bool:
    return channel_permitted


consent_permitted_for_suppression_step = step(_consent_permitted, output="consent_permitted")


# --- Stage 3-4: the tree, twice (adjusted / unadjusted) ---

def _build_tree_pair(tree_doc: dict):
    adjusted = TreeConfig.load(tree_doc).named("tree_adjusted").relabel(writes={
        "leaf_key": "leaf", "leaf_outcome_code": "leaf_outcome_code", "offer_tier_code": "offer_tier_code",
        "tier_ceiling": "tier_ceiling", "channel_1": "channel_1", "channel_2": "channel_2",
        "priority_weight": "priority_weight", "reason_label": "reason_label",
    })
    unadjusted = TreeConfig.load(tree_doc).named("tree_unadjusted").relabel(writes={
        "leaf_key": "unadjusted_leaf", "leaf_outcome_code": "unadjusted_leaf_outcome_code",
        "offer_tier_code": "unadjusted_offer_tier_code", "tier_ceiling": "unadjusted_leaf_tier_ceiling",
        "channel_1": "unadjusted_channel_1", "channel_2": "unadjusted_channel_2",
        "priority_weight": "unadjusted_priority_weight", "reason_label": "unadjusted_reason_label",
    })
    return adjusted, unadjusted


def overlay_stack_id_param(overlay_stack_id: str = param("OS-NONE")) -> str:
    """The overlay stack in force is resolved **once per cycle**, not once per client (§5.3.4
    requirement 7: "resolved by cycle_date, never by today" -- one cycle has one
    `cycle_date`). That resolution -- `overlays.resolve_tree_params` against
    `overlays.TREE_OVERLAYS` -- therefore happens *outside* this per-record pipeline, in
    `generate_configs.py` (or a batch driver's own call), and lands here two ways: the
    node-threshold values themselves go into `tree_adjusted`'s own params (`TreeConfig`'s
    native per-call `params={...}` mechanism -- see `_build_tree_pair`), and the resulting
    id is threaded through as a plain `param()` so it rides along on the evidence record
    (§4.4 `overlay_stack_id`) without this step having to re-derive it per row. See
    NOTES.md "Framework friction" for why a per-row resolution (which is what this function
    looked like before this docstring) does not fit `TreeConfig`'s call-scoped `params=`."""
    return overlay_stack_id


overlay_stack_id_step = step(overlay_stack_id_param, output="overlay_stack_id_out").relabel(
    writes={"overlay_stack_id_out": "overlay_stack_id"})


def _advertised_amount(pre_assessed_amount: float, advertised_tier_ceiling: float) -> float:
    """§5.5 requirement 1: "A campaign may advertise less than the pre-assessed amount; it
    may never advertise more." `advertised_tier_ceiling` is already the tree's own tier
    ceiling with the cap-reduction overlay applied (§5.5 requirement 6)."""
    return round(min(pre_assessed_amount, advertised_tier_ceiling), 2)


def build(rate_card_flex_loan):
    rate_card_index = RateCardIndex.from_configurable_step(rate_card_flex_loan)
    credit_life_index = CreditLifeIndex()

    tree_doc = tree_model.to_v3_document(TREE_V1, path_output="leaf_key")
    tree_adjusted, tree_unadjusted = _build_tree_pair(tree_doc)

    appetite_table = build_appetite_table().relabel(writes={
        "max_amount": "appetite_max_amount", "max_term": "appetite_max_term",
        "max_ratio": "appetite_max_ratio", "min_price": "appetite_min_price", "cell_id": "appetite_cell_id",
    })

    return dag(
        _echo_step("client_id", str), _echo_step("campaign_id", int), _echo_step("cycle_id", int),
        _echo_step("cycle_date", date), _echo_step("assignment_id", str), _echo_step("tree_version", int),

        eligibility.decline_reason_codes_step.relabel(writes={"decline_reason_codes": "eligibility_decline_reasons"}),
        eligibility.is_eligible_step.relabel(reads={"decline_reason_codes": "eligibility_decline_reasons"}),
        consent_verdict_step, channel_permitted_step, consent_permitted_for_suppression_step,
        suppressions.suppression_evaluation_step,

        appetite_table,
        step(preassessment.estimate_max_affordable_instalment, output="max_affordable_instalment"),
        preassessment.build_preassessment_step(rate_card_index, credit_life_index, STATUTORY_CEILING),

        overlay_stack_id_step,
        tree_adjusted, tree_unadjusted,
        overlays.apply_tier_ceiling_cap_adjustment_step.relabel(reads={"decision_date": "cycle_date"}),

        step(_advertised_amount, output="advertised_amount"),
        step(preassessment.is_preassessment_expired, output="preassessment_expired"),

        holdout.is_control_step, holdout.is_universal_holdout_step,

        name="campaign_trees",
    ).emit(
        "assignment_id", "client_id", "campaign_id", "cycle_id", "cycle_date", "tree_version",
        "eligibility_decline_reasons", "is_eligible", "consent_verdict", "channel_permitted",
        "suppression_codes", "suppressed_absolute",
        "appetite_max_amount", "appetite_cell_id",
        "max_affordable_instalment", "pre_assessed_amount", "pre_assessed_term", "binding_constraint_code",
        "pre_assessment_valid_from", "pre_assessment_valid_to", "pre_assessment_evaluations",
        "preassessment_expired",
        "overlay_stack_id",
        "leaf", "leaf_outcome_code", "offer_tier_code", "tier_ceiling", "channel_1", "channel_2",
        "priority_weight", "reason_label",
        "unadjusted_leaf", "unadjusted_leaf_outcome_code", "unadjusted_offer_tier_code",
        "unadjusted_leaf_tier_ceiling",
        "advertised_tier_ceiling", "tier_ceiling_before_cap_overlay", "cap_adjustment_set_id",
        "cap_adjustments_applied",
        "advertised_amount",
        "is_control", "is_universal_holdout",
    )
