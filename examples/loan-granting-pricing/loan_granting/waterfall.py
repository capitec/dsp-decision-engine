"""The cap waterfall: policy rules, each owned by a different team, that narrow three ceilings.

A rule can lower the amount, the term or the worst acceptable risk grade, decline outright, or do nothing. The
rules run in register order, because a rule that takes a share of the current value gives a different answer
depending on what ran before it. One rule, the campaign uplift, may raise the amount, within an authority the Credit
Committee grants. Besides the final ceilings the step records which rule set each one, the whole chain of values,
and for every rule whether it applied, bound, coincided with an earlier rule, or did nothing.

The numbers rules compare against live in the `cap_bands` params table, so the owners of a rule change them without a
release; the register (which rules exist, in which order) is code.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

from decider import ParamTable, allow_fallback, missing_as, param, param_table, step

from credit_core.rate_card import FLEX_LOAN_MAX_AMOUNT, FLEX_LOAN_MAX_TERM

from loan_granting import reasons

AMOUNT, TERM, GRADE = "amount", "term", "grade"
WORST_GRADE = 12.0

NOT_APPLICABLE, DID_NOT_BIND, BOUND, COINCIDENT, RAISED, DECLINED = (
    "not_applicable", "evaluated_did_not_bind", "bound", "coincident", "raised", "declined")

PRODUCT, CREDIT_RISK, SYSTEMS = "Unsecured Lending Product", "Credit Risk Policy", "Credit Systems"

OPEN = 1e9  # an open band edge: a params document cannot hold infinity
UNCHANGED = -1  # a band that leaves this ceiling alone
BAND_COLUMNS = ("rule", "lo", "hi", AMOUNT, TERM, GRADE, "decline")
# (rule number, measure from, measure to, amount, term, worst grade, decline)
_BANDS = (
    (25, 6, 7, UNCHANGED, UNCHANGED, UNCHANGED, True),  # channel 6 is closed to the product
    (150, 4, 5, 80_000, UNCHANGED, UNCHANGED, False),  # pensioners
    (140, 0, 1, 60_000, UNCHANGED, UNCHANGED, False),  # a first loan for a new-to-bank client
    (175, -OPEN, 3, UNCHANGED, UNCHANGED, UNCHANGED, True),  # months employed
    (175, 3, 6, 25_000, UNCHANGED, UNCHANGED, False),
    (175, 6, 24, 70_000, UNCHANGED, UNCHANGED, False),
    (210, 3, OPEN, 40_000, UNCHANGED, 8, False),  # months of the worst arrears
    (210, 2, 3, 120_000, UNCHANGED, 9, False),
    (240, 10, OPEN, UNCHANGED, UNCHANGED, UNCHANGED, True),  # credit enquiries in the last 60 days
    (240, 6, 10, 30_000, UNCHANGED, UNCHANGED, False),
    (305, 1, 2, 50_000, UNCHANGED, UNCHANGED, False),  # employer on the exposure watchlist
    (330, 5, 6, 100_000, UNCHANGED, UNCHANGED, False),  # partner channel
    (330, 6, 7, 150_000, 60, UNCHANGED, False),  # broker channel
)


def _band_rows() -> list[dict]:
    return [dict(zip(BAND_COLUMNS, row)) for row in _BANDS]


DEFAULT_BANDS = _band_rows()
CAP_BANDS = param_table({"rule": int, "lo": float, "hi": float, AMOUNT: float, TERM: float, GRADE: float,
                         "decline": bool}, default=DEFAULT_BANDS)


@dataclass(frozen=True)
class Effect:
    """What one rule does: candidate ceilings (None leaves one alone), an outright decline, or an uplift."""

    amount: float | None = None
    term: float | None = None
    grade: float | None = None
    decline: bool = False
    uplift: bool = False


@dataclass(frozen=True)
class Context:
    bands: list[dict]
    current: dict
    uplift_max_pct: float
    uplift_authority_ceiling: float


@dataclass(frozen=True)
class Rule:
    """One rule of the register. `effect` returns None when the rule does not apply to the application."""

    rule_id: str
    owner: str
    klass: str  # product, appetite, policy, exposure, campaign or regulatory
    effect: Callable[[dict, Context], Effect | None]
    decline_reason: int | None = None


def _banded(number: int, measure: Callable[[dict], float | None], scoped: bool = False):
    """A rule whose effect is the `cap_bands` row for `number` that the measured value falls in.

    It applies whenever the value is known, or, if `scoped`, only when a band contains it.
    """

    def effect(app: dict, ctx: Context) -> Effect | None:
        value = measure(app)
        if value is None:
            return None
        for band in ctx.bands:
            if band["rule"] == number and band["lo"] <= value < band["hi"]:
                amount, term, grade = (None if band[k] == UNCHANGED else band[k] for k in (AMOUNT, TERM, GRADE))
                return Effect(amount, term, grade, decline=band["decline"])
        return None if scoped else Effect()

    return effect


def _known(key: str):
    return lambda app: None if app.get(key) is None or app[key] != app[key] else float(app[key])


def _new_to_bank(app: dict) -> float | None:
    tenure = app.get("internal_tenure_months")
    return 0.0 if tenure is None or tenure != tenure or tenure <= 0 else None


def _appetite_amount(app: dict, ctx: Context) -> Effect:
    return Effect(amount=app["appetite_max_amount"], decline=app["appetite_max_amount"] <= 0)


def _appetite_term(app: dict, ctx: Context) -> Effect:
    return Effect(term=app["appetite_max_term"])


def _exposure_headroom(app: dict, ctx: Context) -> Effect:
    return Effect(amount=max(0.0, app["group_exposure_limit"] - app["internal_exposure_total"]))


def _campaign_uplift(app: dict, ctx: Context) -> Effect | None:
    if not app.get("campaign_id") or not app.get("campaign_is_preapproved"):
        return None
    return Effect(amount=min(ctx.current[AMOUNT] * (1.0 + ctx.uplift_max_pct), ctx.uplift_authority_ceiling), uplift=True)


REGISTER = (
    Rule("CAP-0025", PRODUCT, "product", _banded(25, _known("channel_code")), reasons.R_CHANNEL_CLOSED),
    Rule("CAP-0100", CREDIT_RISK, "appetite", _appetite_amount, reasons.R_BEYOND_APPETITE),
    Rule("CAP-0150", CREDIT_RISK, "policy", _banded(150, _known("employment_type_code"), scoped=True)),
    Rule("CAP-0140", PRODUCT, "policy", _banded(140, _new_to_bank)),
    Rule("CAP-0175", CREDIT_RISK, "policy", _banded(175, _known("months_employed")),
         reasons.R_EMPLOYMENT_TENURE_TOO_SHORT),
    Rule("CAP-0210", CREDIT_RISK, "policy", _banded(210, _known("worst_arrears_months"))),
    Rule("CAP-0240", CREDIT_RISK, "policy", _banded(240, _known("bureau_enquiry_count_60d")),
         reasons.R_ENQUIRY_VELOCITY_EXCESSIVE),
    Rule("CAP-0118", CREDIT_RISK, "appetite", _appetite_term),
    Rule("CAP-0305", SYSTEMS, "exposure", _banded(305, _known("employer_on_watchlist"), scoped=True)),
    Rule("CAP-0330", PRODUCT, "policy", _banded(330, _known("channel_code"))),
    Rule("CAP-0361", SYSTEMS, "exposure", _exposure_headroom),
    Rule("CAP-0420", PRODUCT, "campaign", _campaign_uplift),
)

OUTPUTS = ("amount_cap", "term_cap", "worst_acceptable_grade", "amount_cap_binding_rule_id", "term_cap_binding_rule_id",
           "worst_acceptable_grade_binding_rule_id", "amount_cap_chain", "term_cap_chain",
           "worst_acceptable_grade_chain", "waterfall_rules", "waterfall_decline_reason_codes",
           "uplift_authority_reference", "uplift_restrained_by_rule_id")


def run_waterfall(
    app: dict, bands: list[dict] | None = None, uplift_max_pct: float = 0.25,
    uplift_authority_ceiling: float = 250_000.0, uplift_authority_reference: str = "CC-2026-14",
    register: tuple[Rule, ...] = REGISTER,
) -> dict:
    """Run the register over one application and return what `cap_waterfall` writes.

    `app` holds the inputs `cap_waterfall` reads; `bands` are `cap_bands` rows (default: the shipped policy).
    The result has the three ceilings, the rule that set each, each ceiling's chain of values, one entry per
    rule and the reason codes of any outright decline, including a `risk_grade` worse than the ceiling allows.

        out = run_waterfall(app)
        out["amount_cap"], out["amount_cap_binding_rule_id"]   # 76000.0, "CAP-0361"
    """
    current = {AMOUNT: float(FLEX_LOAN_MAX_AMOUNT), TERM: float(FLEX_LOAN_MAX_TERM), GRADE: WORST_GRADE}
    chain = {ceiling: [{"rule_id": "seed", "value": value}] for ceiling, value in current.items()}
    ctx = Context(_band_rows() if bands is None else bands, current, uplift_max_pct, uplift_authority_ceiling)
    entries, declines = [], []
    regulatory_floor, regulatory_rule = math.inf, None  # the lowest amount a regulatory rule has set
    uplift_reference = restrained_by = None
    for rule in register:
        effect = rule.effect(app, ctx)
        if effect is None:
            entries.append({"rule_id": rule.rule_id, "owner": rule.owner, "status": NOT_APPLICABLE, "changes": {}})
            continue
        changes, statuses = {}, set()
        for ceiling, candidate in ((AMOUNT, effect.amount), (TERM, effect.term), (GRADE, effect.grade)):
            before = current[ceiling]
            if candidate is None:
                continue
            if effect.uplift:
                if candidate > regulatory_floor:
                    candidate, restrained_by = regulatory_floor, regulatory_rule
                if candidate <= before:
                    statuses.add(DID_NOT_BIND)
                    continue
                statuses.add(RAISED)
                uplift_reference = uplift_authority_reference
            elif candidate < before:
                statuses.add(BOUND)
                if rule.klass == "regulatory" and candidate < regulatory_floor:
                    regulatory_floor, regulatory_rule = candidate, rule.rule_id
            else:
                statuses.add(COINCIDENT if candidate == before else DID_NOT_BIND)
                continue
            current[ceiling] = candidate
            chain[ceiling].append({"rule_id": rule.rule_id, "value": candidate})
            changes[ceiling] = [before, candidate]
        if effect.decline:
            statuses.add(DECLINED)
            declines.append(rule.decline_reason)
        status = next((s for s in (DECLINED, RAISED, BOUND, COINCIDENT) if s in statuses), DID_NOT_BIND)
        entries.append({"rule_id": rule.rule_id, "owner": rule.owner, "status": status, "changes": changes})
    if app["risk_grade"] > current[GRADE] and reasons.R_BEYOND_APPETITE not in declines:
        declines.append(reasons.R_BEYOND_APPETITE)  # the client's grade is worse than the register allows
    binder = {c: chain[c][-1]["rule_id"] if len(chain[c]) > 1 else None for c in chain}
    return {
        "amount_cap": current[AMOUNT], "term_cap": current[TERM], "worst_acceptable_grade": current[GRADE],
        "amount_cap_binding_rule_id": binder[AMOUNT], "term_cap_binding_rule_id": binder[TERM],
        "worst_acceptable_grade_binding_rule_id": binder[GRADE],
        "amount_cap_chain": chain[AMOUNT], "term_cap_chain": chain[TERM], "worst_acceptable_grade_chain": chain[GRADE],
        "waterfall_rules": entries, "waterfall_decline_reason_codes": declines,
        "uplift_authority_reference": uplift_reference, "uplift_restrained_by_rule_id": restrained_by,
    }


@allow_fallback
@step(outputs=OUTPUTS)
def cap_waterfall(
    risk_grade: int, channel_code: int, employment_type_code: float, appetite_max_amount: float, appetite_max_term: float,
    group_exposure_limit: float, internal_exposure_total: float,
    internal_tenure_months: float = missing_as(math.nan), months_employed: float = missing_as(math.nan),
    worst_arrears_months: float = missing_as(math.nan), bureau_enquiry_count_60d: float = missing_as(math.nan),
    employer_on_watchlist: float = missing_as(0.0), campaign_id: int = missing_as(0),
    campaign_is_preapproved: bool = missing_as(False),
    cap_bands: ParamTable = CAP_BANDS, uplift_max_pct: float = param(0.25, ge=0.0, le=1.0),
    uplift_authority_ceiling: float = param(250_000.0, ge=0.0),
    uplift_authority_reference: str = param("CC-2026-14"),
) -> tuple[float, float, float, str | None, str | None, str | None, list[dict], list[dict], list[dict], list[dict],
           list[int], str | None, str | None]:
    """The waterfall as a step: it reads the inputs by name and writes `OUTPUTS`."""
    app = dict(locals())
    rows = [dict(zip(cap_bands._fields, values)) for values in zip(*cap_bands)]
    out = run_waterfall(app, rows, uplift_max_pct, uplift_authority_ceiling, uplift_authority_reference)
    return tuple(out[name] for name in OUTPUTS)


def employer_on_watchlist(
    employer_id: int = missing_as(0),
    watchlist: ParamTable = param_table({"employer_id": int}, default=[{"employer_id": 90001}, {"employer_id": 90002}]),
) -> float:
    """1.0 when the employer is on the exposure watchlist (Credit Systems refreshes it monthly), else 0.0."""
    return 1.0 if employer_id in watchlist.employer_id else 0.0
