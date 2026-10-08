"""EP-3, the annual review: score the facility again and compare with the decision before.

A review re-runs the origination flow on refreshed figures, then answers four questions against the previous decision of
record: how the grade moved and why, whether the facility is priced right, what to do with a revolving limit, and
whether the Bank should look at an exit.
"""
from __future__ import annotations

import calendar
from datetime import date
from functools import cache

from decider import Engine, Executable

from business_credit_e2e import history, vocab

# How much of EBITDA a review will not believe, and the best grade (1 is best) it will leave a client with, by how
# complete the evidence was.
EBITDA_HAIRCUT = {vocab.REVIEW_LATE: 0.05, vocab.REVIEW_STALE: 0.15, vocab.REVIEW_TURNOVER_ONLY: 0.15}
GRADE_FLOOR = {vocab.REVIEW_STALE: 7, vocab.REVIEW_TURNOVER_ONLY: 6}
MONTHS_TO_NEXT_REVIEW = {vocab.REVIEW_COMPLETE: 12, vocab.REVIEW_LATE: 12, vocab.REVIEW_STALE: 6,
                         vocab.REVIEW_TURNOVER_ONLY: 3, vocab.REVIEW_NOT_PERFORMED: 2}
FINANCIALS = ("ebitda", "finance_charges", "current_assets", "current_liabilities", "interest_bearing_debt",
              "tangible_net_worth", "declared_annual_turnover")
ENTITY_DATA = ("bureau_score", "months_on_record", "worst_delinquency_months", "adverse_events")
LIMIT_PRODUCT = 21  # the limit programme's revolving access facility
EXIT_GRADE = 10
DECLINE = 4
REPRICE_TOLERANCE = 0.0005  # 5 basis points


def review_basis_code(statement_age_months: float, management_accounts_present: bool, turnover_months: int = 0) -> int:
    """How complete the evidence is, from the age of the latest financial statements.

    Up to 9 months is complete and up to 15 late; up to 21 is stale if management accounts are in hand. Older, or
    without them, the review falls back to the bank's own view of turnover (6 months or more) and failing that is not
    performed. Example: statements 18 months old with management accounts are stale (3).
    """
    if statement_age_months <= 9:
        return vocab.REVIEW_COMPLETE
    if statement_age_months <= 15:
        return vocab.REVIEW_LATE
    if statement_age_months <= 21 and management_accounts_present:
        return vocab.REVIEW_STALE
    return vocab.REVIEW_TURNOVER_ONLY if turnover_months >= 6 else vocab.REVIEW_NOT_PERFORMED


def next_review_date(decision_date: date, review_basis_code: int) -> date:
    """A weaker review buys less time: 12 months when complete or late, 6 when stale, 3 on turnover alone, 2 when not done."""
    month = decision_date.month - 1 + MONTHS_TO_NEXT_REVIEW[review_basis_code]
    year, month = decision_date.year + month // 12, month % 12 + 1
    return date(year, month, min(decision_date.day, calendar.monthrange(year, month)[1]))


def _with_current_entities(request: dict, previous: dict) -> list[dict]:
    """The current entities' ownership and roles, carrying the previous request's own data for each entity it knew."""
    known = {e["entity_id"]: e for e in previous["entities"]}
    return [{**e, **{f: known[e["entity_id"]][f] for f in ENTITY_DATA if f in known.get(e["entity_id"], {})}}
            for e in request["entities"]]


@cache
def _limit_executable() -> tuple[Executable, dict]:
    from limit_mgmt.matrix import build_matrix_table
    from limit_mgmt.pipeline import build

    pipeline = build(build_matrix_table())
    return Engine().bind(pipeline, mode="interpreted"), pipeline.parameters().defaults()


def limit_decision(limit_account: dict, decision_date: date, current_limit: float) -> dict:
    """The limit programme's answer for a revolving facility: the limit it would move to, and the kind of change.

    `limit_account` is the programme's own input record (conduct over the last cycles, the income on file, the
    client's debts): a business has none of its own, so the caller says whose they are.
    """
    executable, params = _limit_executable()
    record = {**limit_account, "decision_date": decision_date, "bureau_as_of_date": decision_date,
              "product_code": LIMIT_PRODUCT, "current_limit": current_limit}
    result = executable.score(record, params)
    return {"current_limit": current_limit, "proposed_limit": result["final_proposed_limit"],
            "change_type_code": result["change_type_code"], "matrix_cell_id": result["matrix_cell_id"]}


def annual_review(
    executable: Executable, params: dict, store: history.DecisionStore, request: dict, *,
    statement_age_months: float, management_accounts_present: bool = False, turnover_months: int = 0,
    limit_account: dict | None = None,
) -> dict:
    """Review a facility: score `request` (the refreshed figures, same `facility_id`) and append the decision of record.

    `executable` is the origination pipeline bound to an `Engine`, and `params` its params document, as they are now:
    a change of policy since the previous decision shows up as the `model` cause. A revolving facility needs
    `limit_account`. The record carries `review_basis_code`, `grade_migration`, `reprice`, `limit_decision`,
    `exit_triggers` and `next_review_date`.
    """
    previous = store.latest(request["facility_id"])
    if previous is None or previous["outcome_code"] == DECLINE:
        raise KeyError(f"facility {request['facility_id']} has no approved decision of record to review")
    revolving = previous["product_code"] == vocab.PRODUCT_REVOLVING
    if revolving and limit_account is None:
        raise ValueError("a revolving facility needs limit_account for its limit decision")

    basis = review_basis_code(statement_age_months, management_accounts_present, turnover_months)
    scored = {**request, "ebitda": round(request["ebitda"] * (1 - EBITDA_HAIRCUT.get(basis, 0.0)), 2)}
    result = executable.score(scored, params)
    # A weaker review restricts: the grade can be no better than the floor. With no evidence at all it stays as it was.
    grade = max(result["risk_grade"], GRADE_FLOOR.get(basis, 1))
    if basis == vocab.REVIEW_NOT_PERFORMED:
        grade = previous["risk_grade"]
    # The contract is what was agreed at origination; the rate this review would price today is `reprice.indicated_rate`.
    contract = {"offered_amount": previous["offered_amount"], "nominal_annual_rate": previous["nominal_annual_rate"]}
    current = history.decision_of_record(
        scored, {**result, "risk_grade": grade, **contract}, assessment_kind_code=vocab.EP3_ANNUAL_REVIEW,
        comparison_basis_code=vocab.COMPARISON_AS_GRADED, predecessor=previous)

    # Today's policy and decision date throughout, so each run differs from the one before it in one thing only.
    as_now = {**previous["request"], "decision_date": scored["decision_date"]}
    business = {**as_now, **{f: scored[f] for f in FINANCIALS}}
    structure = {**business, "entities": _with_current_entities(scored, previous["request"])}
    migration = history.grade_migration(
        previous, current, *(executable.score(r, params)["probability_of_default_unadjusted"]
                             for r in (as_now, business, structure)))
    current["comparison_basis_code"] = migration["comparison_basis_code"]

    shortfall = result["nominal_annual_rate"] - previous["nominal_annual_rate"]
    if abs(shortfall) < REPRICE_TOLERANCE:
        action = vocab.REPRICE_NONE
    else:
        action = vocab.REPRICE_RESET_ON_NOTICE if revolving else vocab.REPRICE_FLAG_RENEGOTIATE
    both_at_exit_grade = grade >= EXIT_GRADE and previous["risk_grade"] >= EXIT_GRADE
    return store.append({
        **current, "review_basis_code": basis, "grade_stale": basis == vocab.REVIEW_NOT_PERFORMED,
        "grade_migration": migration,
        "reprice": {"contractual_rate": previous["nominal_annual_rate"], "indicated_rate": result["nominal_annual_rate"],
                    "margin_shortfall_bps": round(shortfall * 10_000, 1),
                    "margin_shortfall_per_year": round(shortfall * previous["offered_amount"], 2), "action": action},
        "limit_decision": limit_decision(limit_account, scored["decision_date"], previous["offered_amount"]) if revolving else None,
        "exit_triggers": [vocab.EXIT_TWO_REVIEWS_AT_GRADE_10] if both_at_exit_grade else [],
        "next_review_date": next_review_date(scored["decision_date"], basis),
    })
