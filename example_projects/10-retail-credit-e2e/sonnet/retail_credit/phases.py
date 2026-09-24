"""The eighteen-phase registry and the ownership map (spec 10 §5.1, §5.26.1).

A **phase** is a requirements-level unit, not an implementation unit (10
§5.1): "whether a phase becomes one component or forty is exactly the
question this project asks and deliberately does not answer." This module
is the declared identity scheme for phases and the decision points inside
them -- every other module in this package references `PHASES` by
`phase_id`, never by a position in a list, so retiring or renaming a phase
is a declared, versioned event (09 §5.15 item 2) rather than an index
shift.

Only phases P01-P13, P16-P18 (entry point 1's set, minus the conditional
P14) carry real logic in this project (see each phase module's docstring
for what "real" means at this slice's depth). P02, P05, P06, P07 (partial),
P09, P10, P13 also carry real logic because entry point 1 needs them; P14
carries a declared simplification for loop L1 only (see
`retail_credit.consolidation`). P15 is a declared stub: entry point 1 never
runs it (§5.20), so there is nothing for this slice to flesh out.

Decision-point counts below are the spec's own declared totals (10 §5.1),
kept as documentation of scale -- this project does not instantiate 1 400
individual decision-point objects (that would defeat SCOPE.md's "flesh on
one path" instruction by turning the skeleton into the whole building).
Where a phase is fleshed out, its module's own registry
(`retail_credit.cap_waterfall.CAP_REGISTER`, `retail_credit.fraud.RULES`,
...) is the real, countable decision-point list for that phase.
"""
from __future__ import annotations

from dataclasses import dataclass

# owning_team_code, per spec 10 §3.1.
T1_DECISION_PLATFORM = 1
T2_CLIENT_IDENTITY_DATA = 2
T3_REGULATORY_COMPLIANCE = 3
T4_CREDIT_RISK_POLICY = 4
T5_MODEL_RISK_MODELLING = 5
T6_TREASURY_PRICING = 6
T7_UNSECURED_LENDING = 7
T8_CARDS_REVOLVING = 8
T9_SECURED_LENDING = 9
T10_FINANCIAL_CRIME = 10
T11_CAMPAIGN_ANALYTICS = 11
T12_COLLECTIONS_RESTRUCTURE = 12

TEAM_NAMES = {
    T1_DECISION_PLATFORM: "Decision Platform Engineering",
    T2_CLIENT_IDENTITY_DATA: "Client Identity & Data Platform",
    T3_REGULATORY_COMPLIANCE: "Regulatory Compliance",
    T4_CREDIT_RISK_POLICY: "Credit Risk Policy",
    T5_MODEL_RISK_MODELLING: "Model Risk & Modelling",
    T6_TREASURY_PRICING: "Treasury / Pricing",
    T7_UNSECURED_LENDING: "Unsecured Lending Product",
    T8_CARDS_REVOLVING: "Cards & Revolving Product",
    T9_SECURED_LENDING: "Secured Lending Product",
    T10_FINANCIAL_CRIME: "Financial Crime",
    T11_CAMPAIGN_ANALYTICS: "Campaign Analytics & Marketing",
    T12_COLLECTIONS_RESTRUCTURE: "Collections & Restructure",
}


@dataclass(frozen=True)
class Phase:
    phase_id: int              # P01..P18, stable, never positional (10 §4.6 `phase_id`)
    code: str                  # "P01".."P18"
    name: str
    determines: str
    decision_points: int        # the spec's declared count (10 §5.1) -- documentation, not an object count
    owning_teams: tuple[int, ...]
    entry_points: tuple[int, ...]  # which of the 8 entry points ever run this phase, at all (may be reduced)
    fleshed_out: bool           # True where this project implements real logic, not only a stub
    budget_ms: float | None     # p99 internal budget, entry point 1 (10 §5.24.1); None where n/a


PHASES: dict[int, Phase] = {
    p.phase_id: p
    for p in [
        Phase(1, "P01", "Request validation and routing",
              "entry_point_code, phase_set_id, decision_date, candidate product set",
              41, (T1_DECISION_PLATFORM,), (1, 2, 3, 4, 5, 6, 7, 8), True, 1.5),
        Phase(2, "P02", "Client and identity resolution",
              "client_id, identity confidence",
              33, (T2_CLIENT_IDENTITY_DATA,), (1, 2, 3, 5, 6), True, 2.5),
        Phase(3, "P03", "Consent and hard eligibility",
              "Whether the Bank may proceed, and on what basis",
              62, (T3_REGULATORY_COMPLIANCE, T4_CREDIT_RISK_POLICY), (1, 2, 3, 4, 5, 6, 7, 8), True, 2.5),
        Phase(4, "P04", "Data acquisition orchestration",
              "What is fetched, in what order, what arrived",
              47, (T1_DECISION_PLATFORM, T2_CLIENT_IDENTITY_DATA), (1, 2, 3, 4, 5, 6), True, 3.0),
        Phase(5, "P05", "Fraud and financial crime",
              "fraud_verdict_code and its reason set",
              205, (T10_FINANCIAL_CRIME,), (1, 2, 5, 6), True, 5.5),
        Phase(6, "P06", "Feature derivation",
              "~510 derived values, including income, expenses, obligations",
              134, (T2_CLIENT_IDENTITY_DATA, T5_MODEL_RISK_MODELLING), (1, 2, 3, 4, 5, 6), True, 13.0),
        Phase(7, "P07", "Scoring",
              "score, per-characteristic contributions",
              79, (T5_MODEL_RISK_MODELLING,), (1, 2, 3, 4, 5, 6), True, 5.0),
        Phase(8, "P08", "Calibration, grading and adjustments",
              "probability_of_default, risk_grade, the resolved overlay stack",
              46, (T5_MODEL_RISK_MODELLING, T4_CREDIT_RISK_POLICY), (1, 2, 3, 4, 5, 6), True, 3.0),
        Phase(9, "P09", "Policy gates and the cap waterfall",
              "Five ceilings, their chains, and outright declines",
              196, (T4_CREDIT_RISK_POLICY, T1_DECISION_PLATFORM, T3_REGULATORY_COMPLIANCE,
                    T7_UNSECURED_LENDING, T10_FINANCIAL_CRIME), (1, 2, 3, 4, 5, 6, 7, 8), True, 7.5),
        Phase(10, "P10", "Affordability",
              "max_affordable_instalment, affordability_verdict_code",
              88, (T4_CREDIT_RISK_POLICY, T3_REGULATORY_COMPLIANCE), (1, 2, 3, 4, 5, 6), True, 8.5),
        Phase(11, "P11", "Product routing",
              "Which products may carry this request",
              52, (T7_UNSECURED_LENDING, T8_CARDS_REVOLVING, T9_SECURED_LENDING), (1, 4, 5, 7), True, 2.0),
        Phase(12, "P12", "Pricing",
              "Rate, fees, premium, instalment, effective rate",
              83, (T6_TREASURY_PRICING, T3_REGULATORY_COMPLIANCE), (1, 2, 3, 4, 5, 6, 7, 8), True, 2.5),
        Phase(13, "P13", "The solve",
              "The largest affordable amount at each permitted term",
              31, (T1_DECISION_PLATFORM, T7_UNSECURED_LENDING), (1, 4, 5), True, 38.0),
        Phase(14, "P14", "Consolidation search",
              "Which existing debts to settle, and what replaces them",
              63, (T12_COLLECTIONS_RESTRUCTURE,), (1, 5, 4), True, None),  # simplified: loop L1 only
        Phase(15, "P15", "Limit assignment",
              "The proposed limit, and whether it is funded",
              66, (T8_CARDS_REVOLVING, T4_CREDIT_RISK_POLICY), (2, 3, 4), False, None),
        Phase(16, "P16", "Offer assembly and cross-product arbitration",
              "What the client is actually shown, and in what order",
              74, (T7_UNSECURED_LENDING, T8_CARDS_REVOLVING, T9_SECURED_LENDING, T11_CAMPAIGN_ANALYTICS),
              (1, 2, 3, 4, 5, 6), True, 6.5),
        Phase(17, "P17", "Final validation",
              "Whether the Bank is willing to be bound by it",
              61, (T1_DECISION_PLATFORM, T3_REGULATORY_COMPLIANCE), (1, 2, 3, 4, 5, 6, 7, 8), True, 8.0),
        Phase(18, "P18", "Disclosure and decision record emission",
              "The client-facing answer and the permanent record",
              39, (T3_REGULATORY_COMPLIANCE, T1_DECISION_PLATFORM), (1, 2, 3, 4, 5, 6, 7, 8), True, 2.5),
    ]
}

TOTAL_DECISION_POINTS = sum(p.decision_points for p in PHASES.values())  # 1 400, per 10 §2.1


def owners_of(phase_id: int) -> tuple[int, ...]:
    return PHASES[phase_id].owning_teams


def phases_owned_by(team_code: int) -> tuple[Phase, ...]:
    """The ownership map, read the other way (10 §5.26.1): what one team owns or co-owns."""
    return tuple(p for p in PHASES.values() if team_code in p.owning_teams)
