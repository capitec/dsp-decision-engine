"""The gates that decide whether this client is assessed for consolidation, and which way they are routed.

Every gate is evaluated and recorded, not only the one that fired. A client under debt review is never declined:
credit cannot be extended, but the existing agreements can be restructured, so they go to the debt counsellor path.
"""
from __future__ import annotations

from datetime import date

from decider import allow_fallback, missing_as, param, step

PROCEED, DEBT_REVIEW_PATH, DECLINE, REFER, NOT_APPLICABLE = "proceed", "debt_review_path", "decline", "refer", "not_applicable"

GATES = ("CON-ELIG-01", "CON-ELIG-02", "CON-ELIG-03", "CON-ELIG-04", "CON-ELIG-05", "CON-ELIG-06", "CON-ELIG-07")


@allow_fallback
@step(outputs=("eligibility_gate_verdicts", "eligibility_route"))
def eligibility_gates(
    decision_date: date, settleable_count: int, income_verification_tier: int, nomination_conflicts: list[int],
    client_under_debt_review: bool = missing_as(False), client_under_administration: bool = missing_as(False),
    last_consolidation_date: date | None = None, bureau_unobtainable: bool = missing_as(False),
    active_reckless_lending_allegation: bool = missing_as(False),
    serial_consolidation_months: int = param(6, ge=0, le=24), minimum_settleable_accounts: int = param(2, ge=1, le=10),
) -> tuple[dict, str]:
    """Which of the seven gates fired (by gate id), and the route that follows.

    The route takes the first of: debt review, a terminal decline (administration, no verified income, no bureau
    view), a referral (a recent consolidation, a reckless lending allegation, a nominated account that cannot be
    settled), too few settleable accounts (consolidation does not apply), or proceed.
    """
    months_since = None
    if last_consolidation_date is not None:
        months_since = (decision_date.year - last_consolidation_date.year) * 12 + decision_date.month - last_consolidation_date.month
    fired = dict(zip(GATES, (
        client_under_debt_review, client_under_administration,
        months_since is not None and months_since < serial_consolidation_months,
        settleable_count < minimum_settleable_accounts, income_verification_tier == 0, bureau_unobtainable,
        active_reckless_lending_allegation)))
    if fired["CON-ELIG-01"]:
        return fired, DEBT_REVIEW_PATH
    if fired["CON-ELIG-02"] or fired["CON-ELIG-05"] or fired["CON-ELIG-06"]:
        return fired, DECLINE
    if fired["CON-ELIG-03"] or fired["CON-ELIG-07"] or nomination_conflicts:
        return fired, REFER
    return fired, NOT_APPLICABLE if fired["CON-ELIG-04"] else PROCEED
