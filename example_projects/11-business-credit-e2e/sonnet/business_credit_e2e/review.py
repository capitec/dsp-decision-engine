"""L1 -- annual review (spec 11 §5.4), the decision-history spine (§5.10) this
slice's report (§5.17.7) is measured against.

Not decider-served (see `pipeline.py`'s docstring): a review reads the
facility's *prior* decision of record (`history.DecisionHistoryStore`) and, for
a revolving facility, calls project 07's per-account limit pipeline -- both are
orchestration above a stateless per-record `decider` step. `annual_review()` is
a plain function, tested directly (`tests/test_review.py`), the same shape
project 06's `baseline.py` uses to call into project 03's whole pipeline via
`Engine().score()` for its short-circuit.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

from decider import Engine

from business_credit_e2e import covenant, facility, history, vocab
from business_credit_e2e.reuse import load_project05_pipeline, load_project07_pipeline


def review_basis_code(statement_age_months: float, management_accounts_present: bool) -> int:
    """§5.4.3's five-basis ladder, **a mode selector**, not a branch -- exactly
    project 02 §5.8's "a mode selects evidence rules... a mode may not change
    the arithmetic" discipline, reused here for the identical reason (§5.4.3's
    own words: "the five bases must be five declared modes of one assessment").
    A working-depth version of the ladder: bases 1-3 by statement age, 4 only
    when statements are absent but turnover evidence exists (represented here
    by `management_accounts_present` standing in for that evidence), 5 when
    neither is true."""
    if statement_age_months <= 9:
        return vocab.REVIEW_COMPLETE
    if statement_age_months <= 15:
        return vocab.REVIEW_LATE
    if statement_age_months <= 21:
        return vocab.REVIEW_STALE
    if management_accounts_present:
        return vocab.REVIEW_TURNOVER_ONLY
    return vocab.REVIEW_NOT_PERFORMED


_HAIRCUT_BY_BASIS = {
    vocab.REVIEW_COMPLETE: 0.0, vocab.REVIEW_LATE: 0.05, vocab.REVIEW_STALE: 0.15,
    vocab.REVIEW_TURNOVER_ONLY: 0.15, vocab.REVIEW_NOT_PERFORMED: 0.0,
}
_GRADE_CAP_BY_BASIS = {
    vocab.REVIEW_STALE: 7, vocab.REVIEW_TURNOVER_ONLY: 6,
}


def _apply_ebitda_haircut(request: dict, review_basis: int) -> dict:
    """§5.4.3's per-basis EBITDA haircut (5%/15%), applied to a *copy* of the
    financial inputs before the re-run -- reusing project 05's own arithmetic
    unmodified, exactly as 05 §5.9 haircuts are meant to be applied (a
    parameter on the input, not a fork of the ratio calculations)."""
    haircut = _HAIRCUT_BY_BASIS.get(review_basis, 0.0)
    if haircut == 0.0:
        return request
    adjusted = dict(request)
    adjusted["ebitda"] = round(request["ebitda"] * (1 - haircut), 2)
    return adjusted


def _merge_business_fields(previous_input: dict, current_input: dict) -> dict:
    """The "business data only" counterfactual: previous decision's full input,
    with only this decision's own financial figures swapped in."""
    merged = dict(previous_input)
    for field in ("ebitda", "finance_charges", "current_assets", "current_liabilities",
                  "interest_bearing_debt", "tangible_net_worth", "declared_annual_turnover"):
        merged[field] = current_input[field]
    return merged


def _merge_structure_fields(business_only_input: dict, current_input: dict) -> dict:
    """The "business + structure" counterfactual: the above, plus this decision's
    current entity ownership/control/relationship fields -- adverse events stay
    at the previous decision's snapshot (matched positionally; both entity
    lists are assumed the same shape, which `annual_review`'s caller controls)."""
    merged = dict(business_only_input)
    prev_entities = business_only_input["entities"]
    curr_entities = current_input["entities"]
    new_entities = []
    for prev_e, curr_e in zip(prev_entities, curr_entities):
        e = dict(prev_e)
        for field in ("relationship_type_code", "direct_ownership_pct", "effective_ownership_pct",
                      "is_controlling", "is_required_surety", "is_sole_principal"):
            e[field] = curr_e[field]
        new_entities.append(e)
    merged["entities"] = new_entities
    return merged


def _reprice_decision(product_code: int, contractual_rate: float, indicated_rate: float,
                       offered_amount: float) -> dict:
    """§5.4.1 item 2: three numbers and an action, reported **whether or not it
    can be acted on** -- "the aggregate of unactionable shortfalls... is a
    portfolio fact the Credit Committee needs"."""
    shortfall_bps = round((indicated_rate - contractual_rate) * 10_000, 1)
    shortfall_rand_per_year = round((indicated_rate - contractual_rate) * offered_amount, 2)
    if abs(shortfall_bps) < 5:
        action = vocab.REPRICE_NONE
    elif product_code in vocab.REPRICEABLE_ON_NOTICE:
        action = vocab.REPRICE_RESET_ON_NOTICE
    else:
        action = vocab.REPRICE_FLAG_RENEGOTIATE
    return {
        "contractual_rate": contractual_rate, "indicated_rate": indicated_rate,
        "margin_shortfall_bps": shortfall_bps, "margin_shortfall_rand_per_year": shortfall_rand_per_year,
        "action": action,
    }


def _exit_triggers(current_grade: int, previous_grade: int) -> list[str]:
    """§5.4.1 item 4 -- a working subset of the spec's eighteen triggers
    (SCOPE.md)."""
    triggers = []
    if current_grade >= vocab.EXIT_GRADE_FLOOR and previous_grade >= vocab.EXIT_GRADE_FLOOR:
        triggers.append(vocab.EXIT_TRIGGER_GRADE_TWO_REVIEWS)
    return triggers


def _limit_decision_for_revolving(current_record: dict, facility_row: dict) -> dict | None:
    """§5.4.1 item 3: for a revolving facility, the review consumes project 07's
    machinery -- a record-level decision inside a portfolio-level budget. This
    project's slice runs 07's **per-account pipeline only** (the real decider
    steps: population -> exclusions -> scoring -> matrix -> caps ->
    affordability -> decrease), not the population-level allocation (07
    §5.8, out of scope here -- L1 for one facility is not a portfolio run).
    A declared gap (§5.17.2): 07's per-account inputs (behaviour score
    inputs, cycle balances, utilisation) are retail-account-shaped; this
    project maps what it can (current limit, product code) and leaves the
    rest at 07's own defaults -- composing around the gap rather than
    forking 07's population/scoring units for a business shape they were
    never built for."""
    if facility_row["product_code"] not in vocab.REVOLVING_PRODUCTS:
        return None
    project07 = load_project07_pipeline()
    from limit_mgmt.matrix import build_matrix_table
    exe = Engine().bind(project07.build(build_matrix_table()))
    account_record = _business_facility_as_07_account(current_record, facility_row)
    params = project07.build(build_matrix_table()).parameters().defaults()
    result = exe.score(account_record, params)
    return {
        "current_limit": facility_row["approved_limit"],
        "policy_proposed_limit": result.get("policy_proposed_limit"),
        "matrix_cell_id": result.get("matrix_cell_id"),
        "change_type_code": result.get("change_type_code"),
    }


_SEVEN_SAMPLE_ACCOUNT: dict | None = None


def _seven_sample_account_template() -> dict:
    """The field-mapping cost of reuse (§5.17's own subject), made concrete: 07's
    per-account pipeline wants ~57 retail-account fields (income sources,
    bureau accounts for two applicants, cycle balances, conduct history) --
    almost none of which a business-facility record carries. Rather than
    guess at each `missing_as` default individually (07's own NOTES.md
    documents how easy it is to get the ragged-field gotchas wrong), this
    project starts from 07's own `sample_request.json` -- a known-good,
    complete record its own tests already prove scores correctly -- and
    overrides only the handful of fields this project actually knows about a
    business facility. Composing around the gap, not modelling a business's
    behaviour score for real (that is a project 07 extension, not this
    project's to build -- see NOTES.md "Gaps in what I consumed")."""
    global _SEVEN_SAMPLE_ACCOUNT
    if _SEVEN_SAMPLE_ACCOUNT is None:
        import datetime
        import json as _json
        from limit_mgmt import __path__ as _limit_mgmt_path
        sample_path = Path(_limit_mgmt_path[0]).parent / "sample_request.json"
        record = _json.loads(sample_path.read_text())
        date_fields = ("decision_date", "bureau_as_of_date", "last_change_date", "salary_deposit_as_of_date")
        for field in date_fields:
            if record.get(field):
                record[field] = datetime.date.fromisoformat(record[field])
        for key in ("applicant1_bureau_accounts", "applicant1_internal_accounts",
                    "applicant2_bureau_accounts", "applicant2_internal_accounts"):
            for account in record.get(key, []):
                if account.get("opened_date"):
                    account["opened_date"] = datetime.date.fromisoformat(account["opened_date"])
        _SEVEN_SAMPLE_ACCOUNT = record
    return dict(_SEVEN_SAMPLE_ACCOUNT)


def _business_facility_as_07_account(current_record: dict, facility_row: dict) -> dict:
    account = _seven_sample_account_template()
    account.update({
        "decision_date": current_record["decision_date"],
        "bureau_as_of_date": current_record["decision_date"],
        "salary_deposit_as_of_date": current_record["decision_date"],
        "last_change_date": current_record["decision_date"],
        "product_code": facility_row["product_code"],
        "current_limit": facility_row["approved_limit"] or 0.0,
        "cycle_balances": [round((facility_row["approved_limit"] or 0.0) * 0.4, 2)] * 6,
        "cycle_limits": [facility_row["approved_limit"] or 0.0] * 6,
    })
    return account


def annual_review(
    *, facility_row: dict, previous_record: dict, current_input: dict,
    statement_age_months: float, management_accounts_present: bool,
    contractual_rate: float, store: history.DecisionHistoryStore,
) -> dict:
    """One L1 review: re-runs project 05's whole assessment on refreshed inputs
    (with the review basis's own haircut/cap applied), computes the four
    outputs §5.4.1 requires against `previous_record`, appends a new decision
    of record, and returns it together with the review's own extra fields
    (§5.4.4)."""
    project05 = load_project05_pipeline()
    exe = Engine().bind(project05.build())
    params05 = project05.build().parameters().defaults()

    basis = review_basis_code(statement_age_months, management_accounts_present)
    haircut_input = _apply_ebitda_haircut(current_input, basis)
    current_result = exe.score(haircut_input, params05)

    grade_cap = _GRADE_CAP_BY_BASIS.get(basis)
    current_grade = current_result["risk_grade"]
    if grade_cap is not None:
        current_grade = min(current_grade, grade_cap)

    prev_input = previous_record["input_snapshot"]
    business_only_input = _merge_business_fields(prev_input, haircut_input)
    business_only_result = exe.score(business_only_input, params05)
    business_structure_input = _merge_structure_fields(business_only_input, haircut_input)
    business_structure_result = exe.score(business_structure_input, params05)

    migration = history.grade_migration(
        previous_record, {**current_result, "risk_grade": current_grade},
        business_only_result["probability_of_default"], business_structure_result["probability_of_default"],
    )

    reprice = _reprice_decision(
        facility_row["product_code"], contractual_rate, current_result["nominal_annual_rate"],
        facility_row["approved_limit"] or facility_row["approved_amount"] or 0.0,
    )
    limit_decision = _limit_decision_for_revolving(current_result, facility_row)
    exit_triggers = _exit_triggers(current_grade, previous_record["risk_grade"])
    next_review = facility.next_review_date(current_input["decision_date"], basis)

    record = history.new_decision_of_record(
        facility_id=facility_row["facility_id"], assessment_kind_code=vocab.EP3_ANNUAL_REVIEW,
        decision_date=current_input["decision_date"], knowledge_date=current_input["decision_date"],
        predecessor_id=previous_record["decision_of_record_id"], comparison_basis_code=migration["comparison_basis_code"],
        outcome_code=current_result["outcome_code"], risk_grade=current_grade,
        master_scale_version=history.master_scale_version_from_result(current_result),
        probability_of_default=current_result["probability_of_default"],
        probability_of_default_unadjusted=current_result["probability_of_default_unadjusted"],
        extra={
            "input_snapshot": haircut_input, "review_basis_code": basis, "grade_migration": migration,
            "reprice": reprice, "limit_decision": limit_decision, "exit_triggers": exit_triggers,
            "next_review_date": next_review,
        },
    )
    store.append(record)
    return record
