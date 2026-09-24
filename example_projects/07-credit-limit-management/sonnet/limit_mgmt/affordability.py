"""§5.6 -- affordability reassessment with degraded evidence.

"The affordability capability is shared with project 02 and not forked"
(07 §10 item 14). This module never re-derives discretionary income,
capacity or a verdict: it (1) classifies 07's own evidence into the tier
waterfall (§5.6, project-owned -- project 02 has no such concept), (2)
translates that evidence into the *same* income-evidence fields project
02's `core.income`-based waterfall already reads (a verified salary
deposit becomes a high-confidence bank statement, exactly the evidence
shape 02 already knows how to haircut), (3) imputes the notional
instalment (§5.6) as the `proposed_instalment` 02's shape-(a) verdict
needs, (4) layers 07's own wider buffer on top as an overlay
(`overlays.buffer_adjustment_step`), and (5) applies 07's own staleness
rule (§5.6: automatic only on tier A/B within 45 days, bureau within 35,
expenses refreshed or norm-floored) on top of 02's verdict to produce the
automatic/conditional/fail classification the spec's own table (07 §5.8)
lists this project under. See NOTES.md "Framework friction" and "Spec
problems" for why shape (a), not 02's own "capacity, shape (b)" table
entry, is what this module actually calls -- the notional instalment
gives it a real instalment to test.
"""
from __future__ import annotations

import importlib.util
import os
from datetime import date

from decider import flow, missing_as, param, step

from assessment import verdict as core_verdict

from limit_mgmt.overlays import buffer_adjustment_step
from limit_mgmt.vocab import EvidenceTier, IncreasePath


def _load_affordability_pipeline():
    """Project 02's `evidence_unit()`/`capacity_unit()` live in *its own*
    top-level `pipeline.py` (sibling to its `assessment/` package, not inside
    it) -- see 02's own `pipeline.py`. Importing it the ordinary way
    (`import pipeline`) would collide with this project's own top-level
    `pipeline.py` the moment both project directories sit on `sys.path`
    together, which SERVE.md's PYTHONPATH does deliberately (07 consumes 02
    read-only). Loaded by file path under a private module name instead, so
    the two same-named files never contend for one import slot. See
    NOTES.md "Framework friction"."""
    override = os.environ.get("AFFORDABILITY_02_PIPELINE_PATH")
    if override:
        candidate = override
    else:
        import assessment  # resolvable via PYTHONPATH; gives us 02's project root
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(assessment.__file__)))
        candidate = os.path.join(project_root, "pipeline.py")
    spec = importlib.util.spec_from_file_location("_affordability_02_pipeline", candidate)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_core_pipeline = _load_affordability_pipeline()
_core_evidence_unit = _core_pipeline.evidence_unit
_core_capacity_unit = _core_pipeline.capacity_unit

_MIN_PAYMENT_RATE = {20: 0.035, 21: 0.050}  # §6.8, keyed by product_code
_MIN_PAYMENT_FLOOR = {20: 100.0, 21: 50.0}


def notional_instalment(
    policy_proposed_limit: float, product_code: float,
) -> float:
    """§5.6: "the commitment the client would carry if the facility were drawn to the
    new limit and serviced at the contractual minimum payment"."""
    rate = _MIN_PAYMENT_RATE.get(product_code, 0.05)
    floor = _MIN_PAYMENT_FLOOR.get(product_code, 50.0)
    return round(max(policy_proposed_limit * rate, floor if policy_proposed_limit > 0 else 0.0), 2)


notional_instalment_step = step(notional_instalment)


def evidence_tier_code(
    decision_date: date,
    verified_salary_present_3m: bool = missing_as(False),
    salary_deposit_variability_pct: float = missing_as(1.0),
    declared_income_age_days: int = missing_as(9_999),
    has_bank_transactional_account: bool = missing_as(False),
) -> int:
    """§5.6 evidence tiers A-E, the waterfall in the order the spec states it."""
    if verified_salary_present_3m and salary_deposit_variability_pct <= 0.15:
        return EvidenceTier.A_VERIFIED_SALARY
    if verified_salary_present_3m:
        return EvidenceTier.B_IRREGULAR_DEPOSITS
    if declared_income_age_days <= 365:
        return EvidenceTier.C_DECLARED_REFRESHED
    if declared_income_age_days < 9_999:
        return EvidenceTier.D_DECLARED_STALE
    return EvidenceTier.E_NO_EVIDENCE


evidence_tier_code_step = step(evidence_tier_code)


def income_staleness_days(
    decision_date: date,
    salary_deposit_as_of_date: date | None = None,
    declared_income_age_days: int = missing_as(9_999),
    evidence_tier_code: int = missing_as(EvidenceTier.E_NO_EVIDENCE),
) -> int:
    """§4.3: "age of the income basis at decision_date"."""
    if evidence_tier_code in (EvidenceTier.A_VERIFIED_SALARY, EvidenceTier.B_IRREGULAR_DEPOSITS):
        if salary_deposit_as_of_date is None:
            return 9_999
        return (decision_date - salary_deposit_as_of_date).days
    return declared_income_age_days


income_staleness_days_step = step(income_staleness_days)


def _indexed_declared_income(
    declared_income_raw: float, declared_income_age_days: int,
    cpi_annual_rate: float = param(0.052, ge=0.0, le=0.30),
) -> float:
    """§5.6: "indexed forward at CPI... only to a maximum of 12 months" -- tier C only;
    tier D's stale figure is not usable automatically (07's staleness gate routes it to
    conditional/fail regardless of this arithmetic)."""
    months = min(12, max(0, declared_income_age_days // 30))
    factor = (1.0 + cpi_annual_rate) ** (months / 12.0)
    return round(declared_income_raw * factor, 2)


def build_income_evidence_fields(
    evidence_tier_code: int,
    verified_salary_amount: float = missing_as(0.0),
    declared_income_raw: float = missing_as(0.0),
    declared_income_age_days: int = missing_as(9_999),
) -> tuple[float, float, float, float]:
    """Translates 07's own tier classification into the exact income-evidence field
    shape project 02's `core.income`-based waterfall already reads -- a verified salary
    deposit is presented as a high- or low-confidence bank statement, a refreshed
    declared figure as `declared_income`; project 02 is never told "this is degraded",
    it just sees the evidence it already knows how to haircut (§5.6's own requirement:
    "the difference... must be visible in the inputs and the evidence record, not in
    the code").

    Returns (statement_income, statement_confidence, declared_income, _unused) as a
    tuple so this stays one step with a stable output order; see `outputs=` below for
    the real names.
    """
    statement_income = 0.0
    statement_confidence = 0.0
    declared_income = 0.0
    if evidence_tier_code == EvidenceTier.A_VERIFIED_SALARY:
        statement_income = verified_salary_amount
        statement_confidence = 0.95
    elif evidence_tier_code == EvidenceTier.B_IRREGULAR_DEPOSITS:
        statement_income = verified_salary_amount
        statement_confidence = 0.40
    elif evidence_tier_code == EvidenceTier.C_DECLARED_REFRESHED:
        declared_income = _indexed_declared_income(declared_income_raw, declared_income_age_days)
    elif evidence_tier_code == EvidenceTier.D_DECLARED_STALE:
        declared_income = declared_income_raw  # not usable automatically; staleness gate handles it
    # tier E: nothing established -- all three stay at 0 / NONE_ESTABLISHED downstream.
    return statement_income, statement_confidence, declared_income, 0.0


build_income_evidence_fields_step = step(
    build_income_evidence_fields,
    outputs=("_translated_statement_income", "_translated_statement_confidence",
             "_translated_declared_income", "_unused_income_field"),
)
# Deliberately *not* named `applicant1_statement_income`/`applicant1_declared_income`
# directly: `decider`'s wiring resolver flags a new, undeclared read as a likely typo of
# any already-*produced* name within `TYPO_CUTOFF` edit distance (`decider/engine/wiring/
# resolve.py::unbound`) -- and `applicant1_declared_expenses` (read a few steps later, by
# 02's own `consolidate_declared_expenses`) is close enough to a same-flow-produced
# `applicant1_declared_income` to trip it:
#
#   WiringError: input 'applicant1_declared_expenses' is not produced by any earlier
#   step and is not a declared input column. Did you mean 'applicant1_declared_income'?
#
# Producing under a private name and `.relabel()`-ing 02's *reads* onto it (below)
# avoids ever registering the colliding name in scope. See NOTES.md "Framework
# friction" -- this is a false positive: both names are genuine, distinct, intentional
# inputs, not a typo of one another.
_INCOME_EVIDENCE_RELABEL = {
    "applicant1_statement_income": "_translated_statement_income",
    "applicant1_statement_confidence": "_translated_statement_confidence",
    "applicant1_declared_income": "_translated_declared_income",
}


def increase_path_code(
    evidence_tier_code: int,
    income_staleness_days: int,
    bureau_as_of_date: date | None,
    decision_date: date,
    affordability_verdict_code: int,
    evidence_sufficiency_code: int,
    expenses_refreshed_or_normed: bool = missing_as(True),
    obligations_growth_pct: float = missing_as(0.0),
    income_staleness_max_days: int = param(45, ge=1),
    bureau_staleness_max_days: int = param(35, ge=1),
    obligations_growth_threshold: float = param(0.25, ge=0.0),
) -> int:
    """§5.6's staleness rule, verbatim: automatic only on tier A/B within 45 days, a
    bureau view within 35 days, and expenses refreshed-or-normed; anything else that
    still passes on the evidence available is conditional; anything failing on the
    evidence available is not offered at all."""
    if affordability_verdict_code == core_verdict.FAIL or evidence_sufficiency_code not in (
        core_verdict.EVIDENCE_OK,
    ):
        return IncreasePath.FAIL
    bureau_age = 9_999 if bureau_as_of_date is None else (decision_date - bureau_as_of_date).days
    tier_ok = evidence_tier_code in (EvidenceTier.A_VERIFIED_SALARY, EvidenceTier.B_IRREGULAR_DEPOSITS)
    fresh_enough = (
        tier_ok
        and income_staleness_days <= income_staleness_max_days
        and bureau_age <= bureau_staleness_max_days
        and expenses_refreshed_or_normed
        and obligations_growth_pct <= obligations_growth_threshold
    )
    if fresh_enough and affordability_verdict_code == core_verdict.PASS:
        return IncreasePath.AUTOMATIC
    return IncreasePath.CONDITIONAL


increase_path_code_step = step(increase_path_code)


def affordability_cap(policy_proposed_limit: float, max_affordable_instalment: float, product_code: float) -> float:
    """C7 (§5.5): the limit the affordable notional instalment implies -- the inverse of
    `notional_instalment`, floored so C7 never exceeds what §5.5's other caps already
    allowed (C7 only ever tightens further, never loosens the policy caps)."""
    rate = _MIN_PAYMENT_RATE.get(product_code, 0.05)
    if rate <= 0:
        return policy_proposed_limit
    implied = max_affordable_instalment / rate
    return round(min(policy_proposed_limit, implied), 2)


affordability_cap_step = step(affordability_cap, output="cap_affordability")


def affordability_assessment_unit():
    """The whole bridge: evidence -> income-evidence fields -> project 02's shared
    assessment (`assessment_mode_code=LIMIT_INCREASE`, shape (a) against the notional
    instalment) with 07's own buffer overlaid in between the two halves of 02's own
    pipeline -- a `flow`, not a `dag`, because the buffer overlay rewrites a name
    (`affordability_buffer_applied`) that 02's `evidence_unit()` already wrote, and only
    a `flow`'s waterfall semantics allow a later member to override an earlier one
    (`dag` requires a single writer per name; see NOTES.md "Framework friction")."""
    evidence_step = flow(
        evidence_tier_code_step, income_staleness_days_step, build_income_evidence_fields_step,
        name="evidence_translation",
    )
    core_evidence = _core_evidence_unit().relabel(
        reads={"risk_grade": "behaviour_grade", **_INCOME_EVIDENCE_RELABEL},
    )
    # 02's own `decline_reason_codes`/`primary_reason_code`/`reason_registry_version` are
    # the affordability assessment's own evidence-gap reasons (its own registry, e.g.
    # "bureau view stale", "applicant income unestablished") -- a different concern from
    # this project's own decline reasons (`limit_mgmt/reasons.py`: excluded, below the
    # funding line, ...), which read from a *different* registry. Composing both into one
    # `dag` needs distinct names, or the two registries' outputs collide on write
    # (`dag` requires one writer per name) -- see NOTES.md "Framework friction".
    core_capacity = _core_capacity_unit().relabel(
        reads={"proposed_instalment": "notional_instalment"},
        writes={
            "decline_reason_codes": "affordability_decline_reason_codes",
            "primary_reason_code": "affordability_primary_reason_code",
            "reason_registry_version": "affordability_reason_registry_version",
        },
    )
    return flow(
        notional_instalment_step,
        evidence_step,
        core_evidence,
        buffer_adjustment_step,
        core_capacity,
        increase_path_code_step,
        affordability_cap_step,
        name="affordability_bridge",
    )
