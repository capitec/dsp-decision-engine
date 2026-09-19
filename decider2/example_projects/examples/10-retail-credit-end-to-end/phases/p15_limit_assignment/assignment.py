"""P15 limit assignment — 66 decision points. Spec 5.16. Elided from
phases/__init__.py "for length"; this package is the real file.

Four bodies of work in one file because they are genuinely one phase's
lifecycle (matrix -> caps -> decrease -> allocation), not four unrelated
concerns: the SAME matrix, the SAME caps and the SAME scorecard produce a
FUNDED answer on entry point 3 and an UNFUNDED proposal on entry points 2 and
4 (spec 5.16's closing point) — one phase, two shapes of answer, a
requirement §10 holds this project to directly (proposed limits agree across
all three; funded outcomes are permitted to differ).

Population-level allocation is suppressed on entry points 2 and 4 by
DECLARATION (entrypoints/manifest.py's `DECLARED_APPLICABILITY[0]`), not by
derivation — its inputs ARE available on both.
"""

from __future__ import annotations

from decider2 import module, param, Table

class AssignmentMatrixTable(Table):
    key: tuple[int, int, int, int]   # (risk_grade, utilisation_band, tenure_band, product_code)
    increase_pct: float
    absolute_ceiling: float
    minimum_increase: float

def matrix_proposal(risk_grade: int, utilisation_band: int, tenure_band: int,
                    product_code: int, assignment_matrix: AssignmentMatrixTable) -> float:
    """1 080 cells x 3 values = 3 240 values. Must exist in a CANDIDATE state,
    simulated but not deployed, before Credit Risk Policy trusts a re-tune."""
    pass  # assignment_matrix[(risk_grade, utilisation_band, tenure_band, product_code)]

def caps(matrix_proposal: float, net_monthly_income: float, max_affordable_instalment: float,
        product_maximum: float, group_exposure_headroom: float,
        last_increase_date: str | None, decision_date: str) -> float:
    """21 decision points: income multiple (1.2x-3.5x by grade), the
    affordability ceiling from P10, product maximum, group exposure headroom,
    a R25 000 single-cycle ceiling, a spend-cap parameter, and a cooling-off
    window (5 months since the last increase, 9 since a decline)."""
    pass  # apply all seven caps to matrix_proposal, reduce-only

def decrease_path(behavioural_score_deterioration: float, arrears_state: int,
                  adverse_bureau_movement: bool, fraud_markers: bool) -> dict | None:
    """17 decision points. Opposite direction, different governance: eleven
    risk triggers, each its own target and notice class. Reduction to zero
    needs a named authority other than fraud/deceased grounds."""
    pass  # evaluate the eleven decrease triggers; return the binding one, if any

def allocation(caps: float, entry_point_code: int,
               monthly_ceiling: float = param(2_100_000_000_00, ge=0)) -> dict:
    """10 decision points, ENTRY POINT 3 ONLY (declared, not derived — see
    module docstring). Population-level: which accounts are funded depends on
    every OTHER eligible account, not on this one. An account declined here
    has a decline reason that depends on 663 999 others (spec 5.16)."""
    pass  # rank all eligible accounts, fund from the top until monthly_ceiling is spent

def proposed_limit(caps: float, decrease_path: dict | None, allocation: dict | None,
                   entry_point_code: int) -> dict:
    """The one output shape all three entry points share: a proposed limit,
    funded (EP3) or unfunded/indicative (EP2, EP4)."""
    pass  # compose the proposed limit and, where entry_point_code == 3, the funded flag

Matrix = module(matrix_proposal, caps, name="matrix")
Allocation = module(decrease_path, allocation, proposed_limit, name="allocation")
