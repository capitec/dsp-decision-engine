"""Stage 5.1 -- intake and eligibility gates (spec 03 §5.1).

14 gates, every one evaluated even after the first failure (the short-circuit
requirement: a client who fails four gates is entitled to know about all
four). Each gate is one of three states -- pass, fail, `not_evaluated` -- a
third state distinct from the other two, for a gate whose input value is
genuinely absent rather than tested and found acceptable.

Written from scratch rather than composed on top of `credit_core.eligibility`:
00's interface returns a flat fired-code list (a boolean OR of 9 similarly
named gates) with no per-gate identity and no `not_evaluated` state, so it
does not carry the evidence shape 03 §5.1 requires ("a per-gate verdict
vector of 14 entries"). Writing the 14 gates directly, as one function, is a
shorter and clearer path to that shape than wrapping and extending 00's
thinner one. `credit_core.reason_codes` (the registry, ranking, primary pick)
is reused unchanged -- see `pipeline.py`.
"""
from __future__ import annotations

from datetime import date

from decider import frame_step

from loan_granting import reasons

PASS, FAIL, NOT_EVALUATED = "pass", "fail", "not_evaluated"

# Gate identifiers, in the order spec 03 §5.1's table lists them (09 §5.15
# item 2: stable, declared identifiers -- never positional).
GATE_PRODUCT_CHANNEL = "G01_product_channel"
GATE_MIN_AGE = "G02_min_age"
GATE_MAX_AGE_AT_MATURITY = "G03_max_age_at_maturity"
GATE_CAPACITY = "G04_capacity"
GATE_RESIDENCY = "G05_residency"
GATE_EMPLOYMENT_TYPE = "G06_employment_type"
GATE_DEBT_REVIEW = "G07_debt_review"
GATE_ADMINISTRATION_ORDER = "G08_administration_order"
GATE_INSOLVENCY = "G09_insolvency"
GATE_DECEASED_ESTATE = "G10_deceased_estate"
GATE_EXCLUSION_LISTS = "G11_exclusion_lists"
GATE_DUPLICATE = "G12_duplicate_application"
GATE_IN_FLIGHT = "G13_in_flight_application"
GATE_INSOLVENCY_SEASONING = "G14_insolvency_seasoning"

GATE_ORDER = (
    GATE_PRODUCT_CHANNEL, GATE_MIN_AGE, GATE_MAX_AGE_AT_MATURITY, GATE_CAPACITY, GATE_RESIDENCY,
    GATE_EMPLOYMENT_TYPE, GATE_DEBT_REVIEW, GATE_ADMINISTRATION_ORDER, GATE_INSOLVENCY, GATE_DECEASED_ESTATE,
    GATE_EXCLUSION_LISTS, GATE_DUPLICATE, GATE_IN_FLIGHT, GATE_INSOLVENCY_SEASONING,
)

# residency_code classes excluded for Flex Loan (illustrative, spec front matter).
_EXCLUDED_RESIDENCY = {5, 6}
# employment_type_code: 5 social grant, 6 informal -- excluded outright.
_EXCLUDED_EMPLOYMENT = {5, 6}
PENSIONER_EMPLOYMENT_TYPE = 4
PENSIONER_AMOUNT_CAP = 80_000.0

# exclusion_list_hit_codes -> reason. 1 sanctions, 2 internal fraud, 3 staff, 4 litigation, 5 other.
_EXCLUSION_REASON = {
    1: reasons.R_EXCLUSION_SANCTIONS, 2: reasons.R_EXCLUSION_INTERNAL_FRAUD,
    3: reasons.R_EXCLUSION_STAFF, 4: reasons.R_EXCLUSION_LITIGATION, 5: reasons.R_EXCLUSION_OTHER,
}

CHANNEL_BROKER, CHANNEL_PARTNER = 6, 5
_PARTNER_MIN_AMOUNT = 10_000.0
_BROKER_CUTOFF = date(2026, 3, 1)
_INSOLVENCY_SEASONING_MONTHS = 24
_DUPLICATE_AMOUNT_TOLERANCE = 500.0


def _evaluate_gates(app: dict) -> dict:
    verdicts: dict[str, str] = {}
    tested_values: dict[str, object] = {}
    fired: list[int] = []

    def record(gate: str, value, ok: bool | None, reason: int) -> None:
        tested_values[gate] = value
        if ok is None:
            verdicts[gate] = NOT_EVALUATED
            return
        verdicts[gate] = PASS if ok else FAIL
        if not ok:
            fired.append(reason)

    channel = app.get("channel_code")
    requested_amount = app.get("requested_amount")
    decision_date = app.get("decision_date")
    ok = not (channel == CHANNEL_PARTNER and (requested_amount or 0.0) < _PARTNER_MIN_AMOUNT) and \
        not (channel == CHANNEL_BROKER and decision_date is not None and decision_date >= _BROKER_CUTOFF)
    record(GATE_PRODUCT_CHANNEL, channel, ok, reasons.R_PRODUCT_CHANNEL)

    age = app.get("applicant_age_years")
    record(GATE_MIN_AGE, age, None if age is None else age >= 18.0, reasons.R_MIN_AGE)

    term = app.get("requested_term_months")
    if age is None or term is None:
        record(GATE_MAX_AGE_AT_MATURITY, (age, term), None, reasons.R_MAX_AGE_AT_MATURITY)
    else:
        record(GATE_MAX_AGE_AT_MATURITY, age + term / 12.0, age + term / 12.0 <= 75.0, reasons.R_MAX_AGE_AT_MATURITY)

    capacity = app.get("has_contractual_capacity", True)
    record(GATE_CAPACITY, capacity, bool(capacity), reasons.R_NO_CAPACITY)

    residency = app.get("residency_code")
    record(GATE_RESIDENCY, residency, None if residency is None else residency not in _EXCLUDED_RESIDENCY,
           reasons.R_RESIDENCY)

    employment = app.get("employment_type_code")
    record(GATE_EMPLOYMENT_TYPE, employment, None if employment is None else employment not in _EXCLUDED_EMPLOYMENT,
           reasons.R_EMPLOYMENT_EXCLUDED)

    debt_review = app.get("debt_review_status_code", 0)
    record(GATE_DEBT_REVIEW, debt_review, debt_review in (0, 4), reasons.R_DEBT_REVIEW)  # 0 none, 4 cleared

    admin_order = app.get("administration_order_flag", False)
    record(GATE_ADMINISTRATION_ORDER, admin_order, not admin_order, reasons.R_ADMIN_ORDER)

    insolvency = app.get("insolvency_status_code", 0)  # 0 none, 1 sequestrated, 2 rehabilitated
    record(GATE_INSOLVENCY, insolvency, insolvency != 1, reasons.R_INSOLVENCY)

    deceased = app.get("deceased_flag", False) or app.get("estate_flag", False)
    record(GATE_DECEASED_ESTATE, deceased, not deceased, reasons.R_DECEASED_ESTATE)

    # 0 is the "no hit" sentinel: a single-record request must send a non-empty,
    # properly-typed list even when there is nothing to report (an empty list literal
    # fails arrow type inference on the single-record path -- see NOTES.md "Framework
    # friction", the same gotcha 00/02 document for ragged fields).
    hits = [h for h in (app.get("exclusion_list_hits") or []) if h]
    hit_reasons = [_EXCLUSION_REASON.get(h, reasons.R_EXCLUSION_OTHER) for h in hits]
    verdicts[GATE_EXCLUSION_LISTS] = PASS if not hits else FAIL
    tested_values[GATE_EXCLUSION_LISTS] = hits
    fired.extend(hit_reasons)

    is_duplicate = app.get("is_duplicate_application", False)
    record(GATE_DUPLICATE, is_duplicate, not is_duplicate, reasons.R_DUPLICATE_APPLICATION)

    in_flight = app.get("in_flight_application_count", 0) > 0
    record(GATE_IN_FLIGHT, in_flight, not in_flight, reasons.R_IN_FLIGHT_APPLICATION)

    # Rehabilitated insolvency needs 24 months' seasoning; not applicable unless status is 2.
    if insolvency == 2:
        seasoning_months = app.get("insolvency_rehabilitated_months") or 0
        record(GATE_INSOLVENCY_SEASONING, seasoning_months, seasoning_months >= _INSOLVENCY_SEASONING_MONTHS,
               reasons.R_INSOLVENCY)
    else:
        verdicts[GATE_INSOLVENCY_SEASONING] = PASS  # not applicable when not rehabilitated -- a true no-op, not not_evaluated
        tested_values[GATE_INSOLVENCY_SEASONING] = None

    gate_verdicts = [verdicts[g] for g in GATE_ORDER]
    is_eligible = all(v != FAIL for v in gate_verdicts)
    return {
        "eligibility_gate_ids": list(GATE_ORDER),
        "eligibility_gate_verdicts": gate_verdicts,
        "is_eligible": is_eligible,
        "eligibility_decline_reasons": sorted(set(fired)),
    }


@frame_step(
    reads=["channel_code", "requested_amount", "decision_date", "applicant_age_years", "requested_term_months",
           "has_contractual_capacity", "residency_code", "employment_type_code", "debt_review_status_code",
           "administration_order_flag", "insolvency_status_code", "deceased_flag", "estate_flag",
           "exclusion_list_hits", "is_duplicate_application", "in_flight_application_count",
           "insolvency_rehabilitated_months"],
    writes=["eligibility_gate_ids", "eligibility_gate_verdicts", "is_eligible", "eligibility_decline_reasons"],
)
def eligibility_gates(df):
    import polars as pl
    cols = ["channel_code", "requested_amount", "decision_date", "applicant_age_years", "requested_term_months",
            "has_contractual_capacity", "residency_code", "employment_type_code", "debt_review_status_code",
            "administration_order_flag", "insolvency_status_code", "deceased_flag", "estate_flag",
            "exclusion_list_hits", "is_duplicate_application", "in_flight_application_count",
            "insolvency_rehabilitated_months"]
    results = [_evaluate_gates(row) for row in df.select(cols).to_dicts()]
    return df.with_columns(pl.DataFrame(results))
