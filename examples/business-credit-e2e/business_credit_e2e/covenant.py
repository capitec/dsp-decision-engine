"""The DSCR covenant: a definition library whose versions never move under a live facility, and a test with three dates.

A covenant is bound to the definition version in force the day the facility is approved, and every later test uses
that version, not whatever is current. `decision_date` picks the version once; `evaluate` only reads it back.
"""
from __future__ import annotations

import math
from datetime import date
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from decider import ConfigurableStep, allow_fallback, helper

from credit_core.dates import EffectiveVersions
from credit_core.obligations import Account, assess

from business_credit_e2e import vocab

CONFIGS = Path(__file__).resolve().parents[1] / "configs" / "0.1.0"


class Terms(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    version_id: str
    threshold: float
    cure_days: int


class CovenantTerms(ConfigurableStep):
    """The threshold and cure period of each DSCR definition version, written for the version in `covenant_definition_version`.

    Example::

        terms = CovenantTerms(terms=[Terms(version_id="dscr-2026.03", threshold=1.25, cure_days=30)])
        terms.for_version("dscr-2026.03")   # (1.25, 30)
    """

    type: Literal["covenant_terms"] = "covenant_terms"
    name: str = "covenant_terms"
    terms: tuple[Terms, ...]

    def _packed(self) -> tuple[tuple[str, float, int], ...]:
        return tuple((t.version_id, t.threshold, t.cure_days) for t in self.terms)

    def for_version(self, version_id: str) -> tuple[float, int]:
        return terms_for_version(version_id, self._packed())

    def to_ir(self, ctx: Any) -> Any:
        return ctx.call(self, terms_for_version, inputs={"version_id": "covenant_definition_version"},
                        values={"terms": self._packed()}, outputs=("covenant_threshold", "covenant_cure_days"))


@allow_fallback
def terms_for_version(version_id: str, terms: tuple[tuple[str, float, int], ...]) -> tuple[float, int]:
    for known, threshold, cure_days in terms:
        if known == version_id:
            return threshold, cure_days
    raise LookupError(f"no covenant terms for DSCR version {version_id!r}")


def load_covenant_documents() -> tuple[EffectiveVersions, CovenantTerms]:
    """The shipped DSCR version history and terms (`configs/0.1.0/dscr_versions.json` and `dscr_terms.json`)."""
    # `EffectiveVersions.load` would call its own `resolve(as_of)` instead of the type lookup.
    return ConfigurableStep.load(CONFIGS / "dscr_versions.json"), ConfigurableStep.load(CONFIGS / "dscr_terms.json")


def bind_covenant(facility_id: int, decision_date: date, dscr_versions: EffectiveVersions | None = None,
                  dscr_terms: CovenantTerms | None = None) -> dict:
    """The covenant instance of a new facility, pinned to the DSCR version in force on `decision_date`.

    The version history and terms default to the shipped documents.
    Example: `bind_covenant(500001, date(2026, 9, 24))["covenant_definition_version"]` is `"dscr-2026.03"`.
    """
    shipped = load_covenant_documents()
    versions, terms = dscr_versions or shipped[0], dscr_terms or shipped[1]
    version = versions.resolve(decision_date).version_id
    threshold, cure_days = terms.for_version(version)
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


# The steps below bind the covenant inside the origination flow. The version history and the terms are documents
# (`dscr_versions.json`, `dscr_terms.json`) that `pipeline.build` receives.


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
