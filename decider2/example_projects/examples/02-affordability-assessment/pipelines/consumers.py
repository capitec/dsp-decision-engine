"""The five consumers, on one page.

This file exists to make one thing checkable by eye: every consumer is the same
`Assessment` object, under a mode, with local params. No consumer imports a
module from `modules/`; no consumer composes anything. Five call sites, one
calculation.

That is not a convention. `Assessment` is a frozen pydantic instance with a
`contract=` file, and `modules/` is not on the consumers' import path in their
own repositories -- they depend on the published distribution, which exports
`Assessment`, the profiles and the narrowings, and nothing else. A consumer that
wanted `modules.expenses.norms.statutory_norm_cents` would have to file a
release request, which is where the conversation about why belongs.
"""

from decider2 import Runtime

from modules.verdict import (
    to_arrangement_decision,
    to_business_surety,
    to_granting_decision,
    to_limit_decision,
    to_scenario_filter,
)
from pipelines.affordability import Assessment, evidence
from policy.modes import ARRANGEMENT, LIMIT_INCREASE, NEW_APPLICATION, SCENARIO

# --------------------------------------------------------------------------
# Shape (a) -- pass or fail against a known instalment. Projects 05 and 08.
# --------------------------------------------------------------------------
NewApplication = Assessment.under(NEW_APPLICATION) | to_granting_decision

Arrangement = Assessment.under(ARRANGEMENT) | to_arrangement_decision

BusinessSurety = Assessment.under(NEW_APPLICATION).at(
    # Project 05 assesses a proprietor's personal capacity as surety for a
    # juristic applicant. The only thing that differs is where the income comes
    # from, so it is a relabel -- one entry, not a fork. Doc 03 5.2 layer 3.
    inputs={"declared_income_cents": "proprietor_declared_income_cents"},
) | to_business_surety

# --------------------------------------------------------------------------
# Shape (b) -- the largest instalment the applicant could carry. Projects 07
# and 04. No proposed instalment; the verdict is PASS or INDETERMINATE only,
# and the profile's `forbids=` makes supplying one an error rather than a
# silently ignored argument.
# --------------------------------------------------------------------------
LimitIncrease = Assessment.under(LIMIT_INCREASE) | to_limit_decision

# --------------------------------------------------------------------------
# Shape (c) -- the largest AMOUNT the applicant could borrow. This project
# cannot answer it, and the boundary is a requirement rather than a disclaimer.
#
# What this project owes project 03 is three guarantees, and each is a declared
# property on `Assessment` (pipelines/affordability.py) rather than a paragraph
# in a wiki:
#
#   monotone_in(...)                    -> the search has a valid stopping condition
#   assert_cut_equivalent(evidence)     -> the 400th call equals the first
#   assert_materialisation_neutral(...) -> the cheap shape equals the rich one
#
# There is no `max_affordable_amount` here and there is no approximate mode.
# A cheaper approximation would be used by project 06 inside its search and
# would then disagree with the authoritative assessment on the offer that
# search produced.
# --------------------------------------------------------------------------

# --------------------------------------------------------------------------
# Project 06's search. Up to 400 assessments per application, one evidence
# computation, one implementation.
# --------------------------------------------------------------------------
Scenario = Assessment.under(SCENARIO) | to_scenario_filter


def search_consolidation(runtime: Runtime, application, candidate_sets, params):
    """Project 06's inner loop, written against the published surface only."""
    held = Scenario.hold(application, upto=evidence, params=params)
    # `held` is not a cached dict. It is the prefix's declared `carries` in the
    # compiled record layout, and `resume` enters the SAME kernels at the cut.
    # `hold` and `apply` are generated from one definition; there is no second
    # entry point whose income logic could drift from the first.
    for candidate in candidate_sets:
        yield held.resume(
            accounts=candidate.remaining_accounts,
            settlement_quotes=candidate.quotes,
            proposed_instalment_cents=candidate.proposed_instalment_cents,
            materialise=["accounts_annotated"],   # project 06 orders its search on this
        )


# --------------------------------------------------------------------------
# Replay. Project 09 re-derives a 2026 assessment in 2033.
#
# The point is that this is the SAME pipeline object. The only difference is
# that the table resolver is pinned to the version ids recorded on the
# persisted assessment rather than resolving them from `decision_date`, and the
# framework asserts the two agree -- resolving the date must produce exactly the
# recorded ids. A disagreement means a version file was edited in place, which
# is a finding in itself and is otherwise undetectable.
# --------------------------------------------------------------------------
def replay(snapshot):
    return Assessment.under(snapshot.mode).replay(
        snapshot,
        resolver="pinned",           # never reaches a live system
        assert_resolver_agreement=True,
    )
