"""The covenant definition library (spec 11 §6.3, table 39) and one covenant --
DSCR -- built end to end: definition versions, instance binding, and a test with
the three dates of §5.5.2 kept distinct.

This is the one artefact in the estate that resolves **by the version bound at
the instance**, not by `decision_date` (§5.5.1) -- the opposite rule to every
other table this project reads. `EffectiveDatedSet` (project 00, unmodified) is
reused for exactly the one moment that rule *does* apply: picking which
template version is "current" when a **new** instance binds. After that, the
instance's own `covenant_definition_version` is looked up directly and never
re-resolved -- see `bind_covenant_instance` and its docstring.

Debt service for the DSCR denominator reuses project 00's `core.obligations`
per-account arithmetic directly (`credit_core.obligations._process`), the same
"call the underscore-prefixed unit helper, not the published frame_step"
pattern project 02's NOTES.md documents for the identical reason (a merged
account list can't flow through a second `frame_step` -- 02 NOTES.md
"Framework friction" 4.2). Project 06's obligation inventory/settleability
(§5.2-5.3) is **not** used here -- see NOTES.md "Gaps in what I consumed" for
why this project's chosen slice never reaches a call site for it.
"""
from __future__ import annotations

import itertools
from datetime import date

import polars as pl
from decider import frame_step

from credit_core.dates import EffectiveDatedSet, EffectiveVersion
from credit_core.obligations import _process as _obligations_process

from business_credit_e2e import vocab

DSCR_TEMPLATE_ID = "COV-DSCR"

# --- table 39, working depth: one template (DSCR), three dated versions -------
# (spec: "the most-revised -- the DSCR template -- has 11 [versions], of which 9
# still have live instances"; this slice carries 3, enough to prove resolution
# never moves once an instance binds -- see test_covenant.py).
_DSCR_VERSIONS = {
    "dscr-2024.01": {
        "defined_term": "Debt Service Coverage Ratio",
        "numerator": "EBITDA",
        "denominator": "interest + scheduled principal on all interest-bearing debt, including the facility",
        "accounting_basis": "IFRS, frozen at version issue",
        "test_date_rule": "financial year end",
        "threshold": 1.20,
        "delivery_days": 60,
        "grace_from": "delivery_or_knowledge",  # §5.5.2 item 2: this project's standard convention
        "cure_days": 30,
    },
    "dscr-2026.03": {
        "defined_term": "Debt Service Coverage Ratio",
        "numerator": "EBITDA",
        "denominator": "interest + scheduled principal on all interest-bearing debt, including the facility",
        "accounting_basis": "IFRS, frozen at version issue",
        "test_date_rule": "financial year end",
        "threshold": 1.25,
        "delivery_days": 60,
        "grace_from": "delivery_or_knowledge",
        "cure_days": 30,
    },
    "dscr-2028.09": {
        # Spec 11 §5.5.1's own worked example: "a new treatment of lease liabilities
        # in the denominator" -- the reason resolution must never move for a live
        # instance. Instances bound before 2028-09-01 must NOT see this version.
        "defined_term": "Debt Service Coverage Ratio",
        "numerator": "EBITDA",
        "denominator": "interest + scheduled principal on all interest-bearing debt "
                        "(including capitalised lease liabilities), including the facility",
        "accounting_basis": "IFRS 16 lease treatment, frozen at version issue",
        "test_date_rule": "financial year end",
        "threshold": 1.25,
        "delivery_days": 60,
        "grace_from": "delivery_or_knowledge",
        "cure_days": 30,
    },
}

DSCR_TEMPLATE_HISTORY = EffectiveDatedSet(DSCR_TEMPLATE_ID, [
    EffectiveVersion("dscr-2024.01", date(2024, 1, 1), date(2026, 3, 1)),
    EffectiveVersion("dscr-2026.03", date(2026, 3, 1), date(2028, 9, 1)),
    EffectiveVersion("dscr-2028.09", date(2028, 9, 1), None),
])

_instance_ids = itertools.count(1)


def bind_covenant_instance(facility_id: int, decision_date: date) -> dict:
    """A new facility gets a new covenant instance, bound to the template version
    **in force at origination** (`EffectiveDatedSet.resolve`, project 00) --
    and only at origination. The `covenant_definition_version` this returns is
    stored on the instance and is what every future test reads (`test_covenant`
    below never calls `.resolve()` again for a live instance -- spec 11 §5.5.1:
    "a test run in 2029 must use definition version 2026-03-14... not the
    current standard")."""
    version = DSCR_TEMPLATE_HISTORY.resolve(decision_date)
    definition = _DSCR_VERSIONS[version.version_id]
    return {
        "covenant_instance_id": next(_instance_ids),
        "facility_id": facility_id,
        "covenant_definition_version": version.version_id,
        "threshold": definition["threshold"],
        "delivery_days": definition["delivery_days"],
        "cure_days": definition["cure_days"],
        "bound_date": decision_date,
    }


def dscr_debt_service(existing_accounts: list[dict], new_facility_instalment: float) -> float:
    """Monthly debt service for the DSCR denominator: project 00's own per-account
    obligation arithmetic (`credit_core.obligations`, ten treatment behaviours)
    over the business's existing debt, plus the new facility's own instalment --
    the "same rand counted once" discipline spec 11 §5.17.3 requires for a sole
    proprietor applies here too: the new facility must never also appear inside
    `existing_accounts`."""
    result = _obligations_process(bureau_accounts=existing_accounts, internal_accounts=[])
    return round(result["existing_obligations"] + new_facility_instalment, 2)


def test_covenant(
    instance: dict, ebitda: float, existing_accounts: list[dict], new_facility_instalment: float,
    test_date: date, delivery_date: date, determination_date: date,
) -> dict:
    """One test of one instance. Three dates, never conflated (§5.5.2): the ratio
    is computed **as at `test_date`**, on the definition **bound to the
    instance** (never re-resolved), and the result is recorded **as at
    `determination_date`** -- the date this function actually ran, which is what
    a replay reproduces (spec 09 §5.15)."""
    definition_version = instance["covenant_definition_version"]  # pinned; never re-resolved
    threshold = instance["threshold"]

    debt_service = dscr_debt_service(existing_accounts, new_facility_instalment)
    dscr = round(ebitda / 12.0 / debt_service, 4) if debt_service > 0 else float("inf")
    headroom_ratio = round(dscr - threshold, 4)
    headroom_pct = round((dscr / threshold - 1.0) * 100.0, 2) if threshold > 0 else None

    if dscr >= threshold:
        breach_class_code = vocab.BREACH_NONE
    elif dscr >= threshold * 0.90:
        breach_class_code = vocab.BREACH_TECHNICAL
    elif dscr >= threshold * 0.75:
        breach_class_code = vocab.BREACH_MATERIAL
    else:
        breach_class_code = vocab.BREACH_SEVERE

    cure_days_remaining = None
    if breach_class_code != vocab.BREACH_NONE:
        # §5.5.2 item 2: grace runs from the earlier of delivery and the Bank's
        # knowledge -- here, `determination_date` stands in for "the Bank's
        # knowledge" (the date this test actually ran).
        grace_start = min(delivery_date, determination_date)
        cure_days_remaining = instance["cure_days"] - (determination_date - grace_start).days

    return {
        "covenant_instance_id": instance["covenant_instance_id"],
        "covenant_definition_version": definition_version,
        "test_date": test_date,
        "delivery_date": delivery_date,
        "determination_date": determination_date,
        "measured_dscr": dscr,
        "threshold": threshold,
        "headroom_ratio": headroom_ratio,
        "headroom_pct": headroom_pct,
        "breach_class_code": breach_class_code,
        "cure_days_remaining": cure_days_remaining,
    }


def certificate_not_received(instance: dict, test_date: date, determination_date: date) -> dict:
    """§5.5.2's fourth state: **not tested**, because the certificate has not
    arrived. This is a breach of the *information undertaking*, a distinct
    covenant instance from the financial one -- represented here as its own
    record rather than folded into `test_covenant`'s pass/breach result, so a
    late certificate can never be misread as a financial breach or a pass."""
    return {
        "covenant_instance_id": instance["covenant_instance_id"],
        "covenant_definition_version": instance["covenant_definition_version"],
        "test_date": test_date,
        "delivery_date": None,
        "determination_date": determination_date,
        "measured_dscr": None,
        "breach_class_code": vocab.COVENANT_NOT_TESTED,
        "information_undertaking_breached": True,
    }


# --------------------------------------------------------------------------
# Decider steps for the EP-1 origination pipeline (pipeline.py).
#
# Flat scalar/list columns, not the dict-shaped functions above -- decider's
# dag composes named columns, and a `list[dict]` (`existing_accounts`) needs a
# `frame_step`, exactly as project 00's own `core.obligations` does for the
# identical reason (list-of-struct input is fine; only a list-of-struct
# *output* crashes result materialisation -- 00 NOTES.md "Framework friction"
# 4.2). Both layers call the same pure functions above; nothing here
# duplicates the arithmetic.
# --------------------------------------------------------------------------

def covenant_definition_version(decision_date: date) -> str:
    """The template version bound **at origination** -- resolved once, by
    `EffectiveDatedSet` (project 00), and then pinned. See the module
    docstring: no later call in this project re-resolves this for a live
    instance."""
    return DSCR_TEMPLATE_HISTORY.resolve(decision_date).version_id


def covenant_threshold(covenant_definition_version: str) -> float:
    return _DSCR_VERSIONS[covenant_definition_version]["threshold"]


def covenant_cure_days(covenant_definition_version: str) -> int:
    return _DSCR_VERSIONS[covenant_definition_version]["cure_days"]


def covenant_instance_id(facility_id: int) -> int:
    return next(_instance_ids)


@frame_step(reads=["existing_accounts"], writes=["dscr_debt_service"])
def dscr_debt_service_step(df: pl.DataFrame) -> pl.DataFrame:
    """`new_facility_instalment` is added separately (`dscr_total_debt_service`
    below) once the offer's own instalment/notional payment is known -- this
    step only covers the business's *existing* debt (project 00's
    `core.obligations` arithmetic, unmodified)."""
    values = [
        round(_obligations_process(bureau_accounts=row.get("existing_accounts") or [], internal_accounts=[])
              ["existing_obligations"], 2)
        for row in df.select("existing_accounts").to_dicts()
    ]
    return df.with_columns(pl.Series("dscr_debt_service", values))


def dscr_total_debt_service(dscr_debt_service: float, new_facility_instalment: float) -> float:
    return round(dscr_debt_service + new_facility_instalment, 2)


def measured_dscr(ebitda: float, dscr_total_debt_service: float) -> float:
    return round(ebitda / 12.0 / dscr_total_debt_service, 4) if dscr_total_debt_service > 0 else float("inf")


def dscr_headroom_ratio(measured_dscr: float, covenant_threshold: float) -> float:
    return round(measured_dscr - covenant_threshold, 4)


def dscr_breach_class_code(measured_dscr: float, covenant_threshold: float) -> int:
    if measured_dscr >= covenant_threshold:
        return vocab.BREACH_NONE
    if measured_dscr >= covenant_threshold * 0.90:
        return vocab.BREACH_TECHNICAL
    if measured_dscr >= covenant_threshold * 0.75:
        return vocab.BREACH_MATERIAL
    return vocab.BREACH_SEVERE
