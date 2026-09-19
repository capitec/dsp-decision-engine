"""Objective, ranking, and the demonstration that the winner beat the runners-up.

Selection is a REDUCTION over the evaluated frame: 400 rows in, 3 rows out plus
a shadow plus an explanation. That is set-shaped, so it is frame tier - and it is
the one place in this whole flow where static lineage is genuinely at risk.

    "Which inputs can affect the chosen scenario's instalment?"

An argmax over a frame is opaque to lineage in exactly the way doc 02 5 warns
about: an ad-hoc `.sort().head(1)` makes column lineage unanswerable, and the
honest thing would be `@breaks_lineage`. That is unacceptable here, because the
chosen scenario's instalment is THE output of the whole flow and a lineage gap
at the last step makes every lineage query in the project return `unknown`.

So `Select` is a DECLARED REDUCTION with a known schema transform: it takes an
objective, a tie-break list, a distinctness key and a top-n, and it declares that
every output column is the same column of one input row. Lineage then composes -
"the winner's instalment can be affected by everything that can affect any
scenario's instalment, plus the objective weights, plus the tie-break keys" -
which is the correct and useful answer. See FRAMEWORK-DEMANDS D10.
"""

from decider2 import module, param, step
from decider2.search import Select, objective
from decider2.values import na

from search.measures import (
    client_outcome,
    commitment_ratio,
    expected_loss_vs_do_nothing,
    incremental_value_ratio,
    new_money_ratio,
    sustainability_12m,
    total_cost_ratio,
)


# --- the objective is a data-shaped module -----------------------------------
#
# `objective(...)` is `ruleset(...)`'s sibling: interface in code, body in a
# validated document. Its closed vocabulary is {measure_id: weight} plus an
# optional constraint, and that is all it will ever be. An objective that could
# compute would be an expression language, and doc 08 1.3 settled that.
#
# Changing a weight is an INTERIOR change: one background compile, a staged swap,
# no deployment (AC 7). Changing WHICH objective applies to which channel is the
# same change - the scope lives in the same document.

Objective = objective(
    name="search.objective",
    measures=[
        new_money_ratio,
        commitment_ratio,
        total_cost_ratio,
        incremental_value_ratio,
        client_outcome,
        sustainability_12m,
        expected_loss_vs_do_nothing,
    ],
    scope_keys=["channel_code", "assessment_mode_code", "segment_code"],
    interior="config/objective/objectives.json",
    require_weights_sum_to=1.0,
    permit_set_relative=False,
    contract="contracts/objective.json",
)


# --- the reduction ------------------------------------------------------------

SelectWinner = Select(
    by=Objective,
    among="viable",  # a rejected scenario is never a runner-up (spec 5.8.6)
    top_n=param(3, ge=1, le=10),
    distinct_on=["settlement_set_mask", "product_code"],
    tie_break=["total_cost_of_credit asc", "accounts_settled asc", "scenario_id asc"],
    indifference_band_pct=param(2.0, ge=0.0, le=10.0),
    shadow="consol:client_outcome",
    stable=True,
    writes=[
        "scenario_rank",
        "objective_score",
        "objective_id",
        "objective_weights_used",
        "objective_weights_base",
        "indifference_group",
        "shadow_scenario_id",
        "shadow_objective_score",
    ],
)


# --- why this one and not that one --------------------------------------------
#
# Spec 5.8.7: "Scenario 44 was chosen over scenario 17 because it releases R412
# more per month, at a total cost R9 100 higher, under an objective weighting
# instalment relief at 0.6" is the sentence the record must be able to produce.
#
# It is producible EXACTLY, and not approximately, because measures are absolute.
# score = SUM over i of w_i * m_i, so
#
#     delta_score = SUM over i of w_i * (m_i(winner) - m_i(other))
#
# and the component accounting for the difference is the largest |w_i * delta
# m_i|. That is arithmetic, not attribution heuristics. If any measure were
# set-relative the decomposition would not be additive and this step would have
# to become a sensitivity analysis - which is how "why did B win" becomes a data
# science request instead of a record lookup.


@step(output="superiority")
def explain_pairwise(
    winner_objective_score: float,
    other_objective_score: float,
    winner_measures: dict,
    other_measures: dict,
    objective_weights_used: dict,
) -> dict:
    """Decompose the score gap between the winner and one runner-up, per component.

    Returns, per measure: the two values, the weight, the weighted delta, and the
    rank of that weighted delta. The consultant sees the top one; the ombud file
    gets all five.
    """
    pass  # elementwise w_i * (m_i_winner - m_i_other), sorted by absolute value


@step(output="shadow_divergence")
def shadow_divergence(
    chosen_scenario_id: int,
    shadow_scenario_id: int,
    chosen_client_outcome: float,
    shadow_client_outcome: float,
) -> dict:
    """Where the objective in force and the client-outcome shadow disagree, say so.

    Spec 5.8.8. This is the step that will be read aloud at an ombud hearing, and
    it exists so the answer is "we knew, here is the number, and here is the
    approved objective that produced a different choice" rather than "we never
    looked".

    An agreeing shadow is recorded too. A record that only fires on disagreement
    cannot distinguish "agreed" from "never computed".
    """
    pass  # comparison + the OBJ-05 gap in score and in rands of instalment


@step(output="indifference_presentation")
def indifference_presentation(
    scenario_rank: int,
    objective_score: float,
    runner_up_score: float,
    indifference_band_pct: float,
) -> int:
    """Where the top two are within 2%, present both as equivalent, not ranked.

    A 0.4% difference in a modelled expected value is not a difference a
    consultant should defend to a client. The output is a presentation code, not
    a re-ranking: the ranking stays deterministic and the SCREEN changes.
    """
    pass  # RANKED | EQUIVALENT_PAIR | EQUIVALENT_TRIPLE


Selection = module(
    explain_pairwise,
    shadow_divergence,
    indifference_presentation,
    name="selection_evidence",
    taps=["objective_score", "scenario_rank"],
    contract="contracts/selection.json",
)


# --- what selection records ---------------------------------------------------
#
#   objective_id, the weight vector AND its base (pre-overlay) counterpart, the
#   overlay stack affecting it, every scenario's objective score with component
#   decomposition, the final ranking, the indifference evaluation, the shadow.
#
# `objective_weights_base` is not decoration. An overlay re-weighting the
# objective (ADJ-OBJ-011) changes which scenario wins with every rule identical,
# and the decision becomes inexplicable the moment the overlay lapses unless the
# record carries both vectors. Spec 5.8.2 says so; the mechanism that makes it
# free is params.base (FRAMEWORK-DEMANDS D3).
