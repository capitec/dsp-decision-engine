"""The DSCR covenant: a definition library whose versions never move under a live facility, and a test with three dates.

A covenant is bound to the definition version in force the day the facility is approved, and every later test uses
that version, not whatever is current. `decision_date` picks the version once; `evaluate` only reads it back.
"""
from __future__ import annotations

import math
from datetime import date

from decider import allow_fallback, helper, step

from credit_core.dates import EffectiveDatedSet, EffectiveVersion
from credit_core.obligations import Account, assess

from business_credit_e2e import vocab

# version -> (threshold, days to cure a breach). The 2028 version changes the denominator (lease liabilities) and
# keeps the threshold.
DSCR_VERSIONS = {
    "dscr-2024.01": (1.20, 30),
    "dscr-2026.03": (1.25, 30),
    "dscr-2028.09": (1.25, 30),
}

DSCR_HISTORY = EffectiveDatedSet("COV-DSCR", [
    EffectiveVersion("dscr-2024.01", date(2024, 1, 1), date(2026, 3, 1)),
    EffectiveVersion("dscr-2026.03", date(2026, 3, 1), date(2028, 9, 1)),
    EffectiveVersion("dscr-2028.09", date(2028, 9, 1), None),
])


def bind_covenant(facility_id: int, decision_date: date) -> dict:
    """The covenant instance of a new facility, pinned to the DSCR version in force on `decision_date`.

    Example: `bind_covenant(500001, date(2026, 9, 24))["covenant_definition_version"]` is `"dscr-2026.03"`.
    """
    version = DSCR_HISTORY.resolve(decision_date).version_id
    threshold, cure_days = DSCR_VERSIONS[version]
    return {"covenant_instance_id": f"COV-DSCR-{facility_id}", "covenant_definition_version": version,
            "threshold": threshold, "cure_days": cure_days}


@helper(signatures=[((float, float), int)])
def breach_class(dscr: float, threshold: float) -> int:
    if dscr >= threshold:
        return vocab.BREACH_NONE
    if dscr >= threshold * 0.90:
        return vocab.BREACH_TECHNICAL
    if dscr >= threshold * 0.75:
        return vocab.BREACH_MATERIAL
    return vocab.BREACH_SEVERE


@helper(signatures=[((float, float), float)])
def dscr(ebitda: float, monthly_debt_service: float) -> float:
    return round(ebitda / 12.0 / monthly_debt_service, 4) if monthly_debt_service > 0 else math.inf


def evaluate(
    instance: dict, *, ebitda: float, monthly_debt_service: float,
    test_date: date, delivery_date: date, determination_date: date,
) -> dict:
    """One test of a bound covenant. Three dates stay apart: the ratio is as at `test_date`, the certificate arrived
    on `delivery_date`, and the result was determined on `determination_date`.

    The cure clock runs from the earlier of delivery and determination. Example: a ratio of 1.0 against 1.25 is a
    material breach with `cure_days_remaining` counting down from 30.
    """
    ratio = dscr(ebitda, monthly_debt_service)
    breach = breach_class(ratio, instance["threshold"])
    cure_days_remaining = None
    if breach != vocab.BREACH_NONE:
        cure_days_remaining = instance["cure_days"] - (determination_date - min(delivery_date, determination_date)).days
    return {
        "covenant_instance_id": instance["covenant_instance_id"],
        "covenant_definition_version": instance["covenant_definition_version"],
        "test_date": test_date, "delivery_date": delivery_date, "determination_date": determination_date,
        "measured_dscr": ratio, "threshold": instance["threshold"],
        "headroom_ratio": round(ratio - instance["threshold"], 4),
        "breach_class_code": breach, "cure_days_remaining": cure_days_remaining,
    }


def certificate_not_received(instance: dict, test_date: date, determination_date: date) -> dict:
    """The fourth state, neither a pass nor a breach: the test could not run because the certificate has not arrived.

    That is a breach of the information undertaking, a separate covenant, and never a financial result.
    """
    return {
        "covenant_instance_id": instance["covenant_instance_id"],
        "covenant_definition_version": instance["covenant_definition_version"],
        "test_date": test_date, "delivery_date": None, "determination_date": determination_date,
        "measured_dscr": None, "breach_class_code": vocab.COVENANT_NOT_TESTED,
        "information_undertaking_breached": True,
    }


# The steps below bind the covenant inside the origination flow, from the same functions.

covenant_definition_version = DSCR_HISTORY.resolver_step(output="covenant_definition_version")


@allow_fallback
@step(outputs=("covenant_threshold", "covenant_cure_days"))
def covenant_terms(covenant_definition_version: str) -> tuple[float, int]:
    return DSCR_VERSIONS[covenant_definition_version]


@allow_fallback
def covenant_instance_id(facility_id: int) -> str:
    return f"COV-DSCR-{facility_id}"


@allow_fallback
def dscr_existing_debt_service(existing_accounts: list[Account]) -> float:
    """Monthly cost of the business's existing debt, treated account by account as the obligations step does."""
    return assess(existing_accounts, []).existing_obligations


def dscr_total_debt_service(dscr_existing_debt_service: float, new_facility_instalment: float) -> float:
    return round(dscr_existing_debt_service + new_facility_instalment, 2)


def measured_dscr(ebitda: float, dscr_total_debt_service: float) -> float:
    return dscr(ebitda, dscr_total_debt_service)


def dscr_headroom_ratio(measured_dscr: float, covenant_threshold: float) -> float:
    return round(measured_dscr - covenant_threshold, 4)


def dscr_breach_class_code(measured_dscr: float, covenant_threshold: float) -> int:
    return breach_class(measured_dscr, covenant_threshold)
