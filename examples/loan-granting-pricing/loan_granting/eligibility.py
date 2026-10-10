"""The 14 eligibility gates, each a step with its own verdict, and the step that gathers them.

Every gate runs even after one fails, so a client who fails four gates is told about all four. A gate whose input is
not there yet is `not_evaluated`, which is neither a pass nor a fail.
"""
from __future__ import annotations

import math
from datetime import date

from decikit import allow_fallback, dag, missing_as, param, step

from loan_granting import reasons

PASS, FAIL, NOT_EVALUATED = 1, 2, 3
VERDICT_NAMES = ((PASS, "pass"), (FAIL, "fail"), (NOT_EVALUATED, "not_evaluated"))

CHANNEL_PARTNER, CHANNEL_BROKER = 5, 6
BROKER_CLOSED_FROM = date(2026, 3, 1)  # ponytail: a constant; make it a param when the product moves it
REHABILITATED = 2


@allow_fallback
def gate_product_channel(
    channel_code: int, decision_date: date, requested_amount: float = missing_as(math.nan),
    partner_minimum_amount: float = param(10_000.0, ge=0.0),
) -> int:
    """Partner channel below its minimum amount, and broker channel after it closed."""
    too_small = channel_code == CHANNEL_PARTNER and requested_amount < partner_minimum_amount
    closed = channel_code == CHANNEL_BROKER and decision_date >= BROKER_CLOSED_FROM
    return FAIL if too_small or closed else PASS


def gate_min_age(applicant_age_years: float, minimum_age: float = param(18.0, ge=0.0)) -> int:
    return PASS if applicant_age_years >= minimum_age else FAIL


def gate_max_age_at_maturity(
    applicant_age_years: float, requested_term_months: float = missing_as(math.nan),
    maximum_age_at_maturity: float = param(75.0, ge=0.0),
) -> int:
    """Not evaluated when no term was asked for (a journey that asks what the client qualifies for)."""
    if math.isnan(requested_term_months):
        return NOT_EVALUATED
    return PASS if applicant_age_years + requested_term_months / 12.0 <= maximum_age_at_maturity else FAIL


def gate_capacity(has_contractual_capacity: bool) -> int:
    return PASS if has_contractual_capacity else FAIL


def gate_residency(residency_code: int, first_excluded_class: int = param(5, ge=0)) -> int:
    """Residency classes from `first_excluded_class` up are the non-resident classes the product excludes."""
    return PASS if residency_code < first_excluded_class else FAIL


def gate_employment_type(employment_type_code: float, first_excluded_type: float = param(5.0, ge=0.0)) -> int:
    """Social grant (5) and informal (6) income are excluded; a pensioner (4) is capped by the waterfall instead."""
    return PASS if employment_type_code < first_excluded_type else FAIL


def gate_debt_review(debt_review_status_code: int) -> int:
    """Only no debt review (0) or a cleared one (4) passes."""
    return PASS if debt_review_status_code == 0 or debt_review_status_code == 4 else FAIL


def gate_administration_order(administration_order_flag: bool) -> int:
    return FAIL if administration_order_flag else PASS


def gate_insolvency(insolvency_status_code: int) -> int:
    """Sequestrated (1) fails; rehabilitated (2) is judged by `gate_insolvency_seasoning`."""
    return FAIL if insolvency_status_code == 1 else PASS


def gate_deceased_estate(deceased_flag: bool, estate_flag: bool) -> int:
    return FAIL if deceased_flag or estate_flag else PASS


@allow_fallback
def gate_exclusion_lists(exclusion_list_hits: list[int] = missing_as([])) -> int:
    return FAIL if any(exclusion_list_hits) else PASS


def gate_duplicate_application(is_duplicate_application: bool) -> int:
    return FAIL if is_duplicate_application else PASS


def gate_in_flight_application(in_flight_application_count: int) -> int:
    return FAIL if in_flight_application_count > 0 else PASS


def gate_insolvency_seasoning(
    insolvency_status_code: int, insolvency_rehabilitated_months: float = missing_as(0.0),
    seasoning_months: float = param(24.0, ge=0.0),
) -> int:
    """A rehabilitated client needs `seasoning_months` since rehabilitation; for anyone else the gate does not apply."""
    if insolvency_status_code != REHABILITATED:
        return PASS
    return PASS if insolvency_rehabilitated_months >= seasoning_months else FAIL


GATE_REASONS = (
    ("gate_product_channel", reasons.R_PRODUCT_CHANNEL), ("gate_min_age", reasons.R_MIN_AGE),
    ("gate_max_age_at_maturity", reasons.R_MAX_AGE_AT_MATURITY), ("gate_capacity", reasons.R_NO_CAPACITY),
    ("gate_residency", reasons.R_RESIDENCY), ("gate_employment_type", reasons.R_EMPLOYMENT_EXCLUDED),
    ("gate_debt_review", reasons.R_DEBT_REVIEW), ("gate_administration_order", reasons.R_ADMIN_ORDER),
    ("gate_insolvency", reasons.R_INSOLVENCY), ("gate_deceased_estate", reasons.R_DECEASED_ESTATE),
    ("gate_duplicate_application", reasons.R_DUPLICATE_APPLICATION),
    ("gate_in_flight_application", reasons.R_IN_FLIGHT_APPLICATION), ("gate_insolvency_seasoning", reasons.R_INSOLVENCY),
)


@allow_fallback
@step(outputs=("eligibility_gate_verdicts", "is_eligible", "eligibility_decline_reasons"))
def eligibility_gates(
    gate_product_channel: int, gate_min_age: int, gate_max_age_at_maturity: int, gate_capacity: int,
    gate_residency: int, gate_employment_type: int, gate_debt_review: int, gate_administration_order: int,
    gate_insolvency: int, gate_deceased_estate: int, gate_exclusion_lists: int, gate_duplicate_application: int,
    gate_in_flight_application: int, gate_insolvency_seasoning: int, exclusion_list_hits: list[int] = missing_as([]),
) -> tuple[dict, bool, list[int]]:
    """The verdict of every gate by name, whether all passed, and the reason of each that failed."""
    gates = {name: verdict for name, verdict in dict(locals()).items() if name.startswith("gate_")}
    fired = [code for name, code in GATE_REASONS if gates[name] == FAIL]
    if gate_exclusion_lists == FAIL:
        by_hit = dict(reasons.EXCLUSION_REASONS)
        fired += [by_hit.get(hit, reasons.R_EXCLUSION_OTHER) for hit in exclusion_list_hits if hit]
    names = dict(VERDICT_NAMES)
    return ({name.removeprefix("gate_"): names[verdict] for name, verdict in gates.items()},
            FAIL not in gates.values(), sorted(set(fired)))


def unit():
    """All 14 gates and the step that gathers them."""
    return dag(
        gate_product_channel, gate_min_age, gate_max_age_at_maturity, gate_capacity, gate_residency,
        gate_employment_type, gate_debt_review, gate_administration_order, gate_insolvency, gate_deceased_estate,
        gate_exclusion_lists, gate_duplicate_application, gate_in_flight_application, gate_insolvency_seasoning,
        eligibility_gates, name="eligibility",
    ).emit("eligibility_gate_verdicts", "is_eligible", "eligibility_decline_reasons")
