"""The full-width skeleton: all 23 phases (O1-O17, L1-L6) and all 9 entry points
(spec 11 §5.1, §5.2), declared at real scale -- the counterpart to project 10's
`retail_credit.phases`/`entry_points` module, which SCOPE.md's own header table
asks every "full-width skeleton plus one real path" slice in this set to build
(project 10's row; the task brief that assigned this slice states the same
requirement in those words). Declaring the skeleton is cheap and load-bearing:
"why did O12 not run for this client" must be answerable without inference, even
for the eight entry points and eighteen phases this slice does not flesh out.

**What is real vs. declared here**: `fleshed_out=True` marks exactly the phases
this project's own code (`origination.py`, `review.py`, `covenant.py`, `history.py`)
actually computes -- O1-O9, O13, O15-O17 (EP-1, via project 05's whole pipeline
plus this project's own O1/O10/O15-O17) and L1/L2 (`review.py`, `covenant.py`).
Every other phase is declared only, proven by `tests/test_phases.py` routing
correctly (the entry-point x phase matrix resolves to the right phase list), not
by running end to end -- exactly project 10's own precedent for its 2-8 entry
points and its unbuilt phases.
"""
from __future__ import annotations

from dataclasses import dataclass

from business_credit_e2e import vocab


@dataclass(frozen=True)
class Phase:
    phase_id: str          # "O1".."O17", "L1".."L6" -- stable, never positional
    name: str
    determines: str
    decision_points: int   # spec 11's own declared count (§5.2) -- documentation of scale
    owned_by: str
    fleshed_out: bool


PHASES: dict[str, Phase] = {
    p.phase_id: p for p in [
        Phase("O1", "Request and relationship resolution", "entry_point_code, candidate facility types",
              34, "written here (origination.request_resolution)", True),
        Phase("O2", "Group and entity structure resolution", "The resolved entity set as at a date",
              68, "[05] §5.1, via business_nested.pipeline.build()", True),
        Phase("O3", "KYB, screening and registration status", "Regulatory regime, screening outcome",
              76, "[05] §5.2, §5.4", True),
        Phase("O4", "Business absolute rules of disqualification", "Outright disqualification",
              88, "[05] §5.3", True),
        Phase("O5", "Per-entity rules, event classification, entity scoring", "Per-entity verdict and score",
              196, "[05] §5.4-5.7", True),
        Phase("O6", "Roll-up to the people component", "The people PD/grade blend", 62, "[05] §5.8", True),
        Phase("O7", "Financial spreading, haircuts, sector benchmarking", "Financial ratios and confidence",
              132, "[05] §5.9", True),
        Phase("O8", "Behavioural assessment", "Conduct-based score component", 38, "[05] §5.10", False),
        Phase("O9", "Combined grade and adjustments", "probability_of_default, risk_grade",
              54, "[05] §5.10, [00] §6.22", True),
        Phase("O10", "Group exposure aggregation and concentration", "Group headroom",
              58, "[00] §6.16, written here (origination.py)", True),
        Phase("O11", "Appetite and facility eligibility", "Eligible facility types, ceilings",
              96, "[05] §5.11 (single product-50-shaped lookup, reused for 51 too -- declared gap)", True),
        Phase("O12", "Security and collateral assessment", "Adjusted cover, security_type",
              88, "[05] §5.11", False),
        Phase("O13", "Pricing and structure negotiation", "The offer", 72, "[05] §5.12", True),
        Phase("O14", "Debt service coverage and personal affordability", "Serviceability",
              64, "[05] §5.12, [02] (via 05's sole_proprietor.py)", True),
        Phase("O15", "Covenant setting", "The covenant set for this facility",
              74, "written here (covenant.bind_covenant_instance)", True),
        Phase("O16", "Conditions precedent and subsequent", "What must happen before/after drawdown",
              42, "written here (origination.conditions_precedent)", True),
        Phase("O17", "Authority routing and decision record", "The approved decision of record",
              38, "written here (origination.route_authority, 2 of 7 levels)", True),
        Phase("L1", "Annual review", "Grade migration, re-pricing, limit decision, exit recommendation",
              164, "written here (review.annual_review, history.py)", True),
        Phase("L2", "Covenant monitoring", "Test result, breach class, cure, waiver",
              148, "written here (covenant.test_covenant)", True),
        Phase("L3", "Early warning and watchlist", "Watchlist grade and signal set", 142, "written here", False),
        Phase("L4", "Amendment", "Amended terms, re-tested security position", 58, "written here", False),
        Phase("L5", "Restructure and forbearance", "Concession package, forbearance classification",
              76, "[06] §5.9, extended -- not_reached this slice (SCOPE.md: skip L4-L6)", False),
        Phase("L6", "Exit and handoff", "The exit decision and handoff record", 32, "written here", False),
    ]
}

TOTAL_DECISION_POINTS = sum(p.decision_points for p in PHASES.values())  # 1 900, spec 11 §5.2


def phases_owned_here() -> tuple[Phase, ...]:
    return tuple(p for p in PHASES.values() if p.fleshed_out)


def declared_but_stubbed() -> tuple[Phase, ...]:
    return tuple(p for p in PHASES.values() if not p.fleshed_out)


# --- The 9-entry-point x 23-phase matrix (spec 11 §5.1's own table), at
# RUNS/PARTIAL/SKIPPED granularity. Only EP-1 and EP-3 are exercised end to end
# by this project (origination.py, review.py); the rest route correctly
# (`tests/test_phases.py`) but are not run. -------------------------------------
RUNS, PARTIAL, SKIPPED = "runs", "partial", "skipped"

_ENTRY_POINT_PHASES: dict[int, dict[str, str]] = {
    vocab.EP1_NEW_TO_BANK: {p: RUNS for p in
                             ("O1", "O2", "O3", "O4", "O5", "O6", "O7", "O8", "O9", "O10", "O11", "O12",
                              "O13", "O14", "O15", "O16", "O17")},
    vocab.EP2_ADDITIONAL_FACILITY: {**{p: RUNS for p in ("O1", "O4", "O6", "O7", "O8", "O9", "O10", "O11",
                                                          "O12", "O13", "O14", "O15", "O16", "O17")},
                                     "O2": PARTIAL, "O3": PARTIAL, "O5": PARTIAL},
    vocab.EP3_ANNUAL_REVIEW: {**{p: RUNS for p in ("O1", "O3", "O4", "O5", "O6", "O7", "O8", "O9", "O10",
                                                    "O11", "O12", "O13", "O14", "O15", "O16", "O17", "L1")},
                               "O2": PARTIAL},
    vocab.EP4_COVENANT_TEST: {"L2": RUNS, "O7": PARTIAL},
    vocab.EP5_EARLY_WARNING: {"L3": RUNS, "O5": PARTIAL, "O6": PARTIAL},
    vocab.EP6_AMENDMENT: {"L4": RUNS},
    vocab.EP7_RESTRUCTURE_FORBEARANCE: {**{p: RUNS for p in ("L5", "O7", "O10", "O12", "O14", "O15", "O17")},
                                          "O5": PARTIAL, "O6": PARTIAL},
    vocab.EP8_GROUP_REASSESSMENT: {"O2": RUNS, "O10": RUNS, "O11": PARTIAL, "O12": PARTIAL, "L3": PARTIAL},
    vocab.EP9_RM_PREASSESSMENT: {"O11": RUNS, "O1": PARTIAL, "O4": PARTIAL, "O7": PARTIAL, "O9": PARTIAL,
                                   "O13": PARTIAL},
}


def phases_for_entry_point(entry_point_code: int) -> dict[str, str]:
    """The declared phase set for one entry point -- §5.1's own table, transcribed.
    Raises for an unknown entry point rather than defaulting silently."""
    if entry_point_code not in _ENTRY_POINT_PHASES:
        raise ValueError(f"entry point {entry_point_code} has no declared phase row")
    return _ENTRY_POINT_PHASES[entry_point_code]
