"""Reason registries (09 §5.15 item 11: declared codes, drawn from a registry, in
severity order, plus a designated primary).

Two distinct registries, per spec 06 §4.7: `rejection_reason_code` (why a
*scenario* was not viable -- CON-INT failures, routing failures, solve
non-convergence) and `decline_reason_codes` (why the *client* was declined or
referred -- eligibility gate failures, or "no viable scenario at all").
"""
from __future__ import annotations

from credit_core.reason_codes import ReasonCode, ReasonCodeRegistry

from consolidation import interventions

REJECTION_REGISTRY_VERSION = "consolidation-rejections-2026.09"

# rejection_reason_code -> (severity, description). CON-INT codes map 1000 + the
# intervention number; a handful of routing/solve failures sit above 1100.
_CON_INT_DESCRIPTIONS = {
    interventions.CON_INT_01: "More accounts settled than the product permits",
    interventions.CON_INT_03: "Instalment reduction below the product's minimum",
    interventions.CON_INT_04: "Total cost of credit increase exceeds the anti-harm ceiling",
    interventions.CON_INT_05: "Term extension beyond the settled accounts' longest remaining term plus the cap",
    interventions.CON_INT_06: "New rate exceeds the settled accounts' weighted average rate",
    interventions.CON_INT_08: "New money exceeds the permitted proportion or absolute cap",
    interventions.CON_INT_09: "Post-consolidation debt service ratio exceeds the ceiling",
    interventions.CON_INT_10: "Proportion of the advance paid to external creditors below the minimum",
    interventions.CON_INT_13: "Client has reached the maximum consolidations in a rolling 24 months",
    interventions.CON_INT_14: "Post-consolidation discretionary income below the minimum floor",
}

REJ_NOT_ROUTABLE = 1101
REJ_SOLVE_NOT_CONVERGED = 1102
REJ_AFFORDABILITY_FAIL = 1103
REJ_NOT_PRICED = 1104
REJ_MANDATORY_ACCOUNT_OMITTED = 1105
REJ_BUDGET_EXHAUSTED = 1106

REJECTION_REGISTRY = ReasonCodeRegistry(REJECTION_REGISTRY_VERSION, [
    *(ReasonCode(1000 + code, 10 + code, desc, True) for code, desc in _CON_INT_DESCRIPTIONS.items()),
    ReasonCode(REJ_NOT_ROUTABLE, 90, "No product in this flow can carry this settlement set", False),
    ReasonCode(REJ_SOLVE_NOT_CONVERGED, 91, "The circular advance/fee solve did not converge within its iteration bound", False),
    ReasonCode(REJ_AFFORDABILITY_FAIL, 5, "Scenario instalment exceeds the client's affordable capacity", True),
    ReasonCode(REJ_NOT_PRICED, 92, "No rate cell exists for this amount, term and grade", False),
    ReasonCode(REJ_MANDATORY_ACCOUNT_OMITTED, 3, "A client-nominated mandatory account was omitted from this scenario", True),
    ReasonCode(REJ_BUDGET_EXHAUSTED, 95, "Scenario not evaluated: the evaluation budget was exhausted first", False),
])

# decline_reason_codes -- the client-facing outcome (06 §7). Overlaps in meaning with
# the eligibility routes (eligibility.py) but is the registry-ranked, client-facing form.
DECLINE_REGISTRY_VERSION = "consolidation-decline-2026.09"

D_DEBT_REVIEW = 2001
D_ADMINISTRATION = 2002
D_SERIAL_CONSOLIDATION = 2003
D_TOO_FEW_SETTLEABLE = 2004
D_NO_INCOME = 2005
D_BUREAU_UNOBTAINABLE = 2006
D_RECKLESS_ALLEGATION = 2007
D_NO_VIABLE_SCENARIO = 2008

DECLINE_REGISTRY = ReasonCodeRegistry(DECLINE_REGISTRY_VERSION, [
    ReasonCode(D_DEBT_REVIEW, 1, "Client is under debt review", True),
    ReasonCode(D_ADMINISTRATION, 2, "Client is under administration or sequestration", True),
    ReasonCode(D_SERIAL_CONSOLIDATION, 20, "A consolidation concluded within the last 6 months", True),
    ReasonCode(D_TOO_FEW_SETTLEABLE, 30, "Fewer than 2 settleable accounts", False),
    ReasonCode(D_NO_INCOME, 5, "No verified income at the decision date", True),
    ReasonCode(D_BUREAU_UNOBTAINABLE, 6, "Bureau view could not be obtained", True),
    ReasonCode(D_RECKLESS_ALLEGATION, 15, "Active reckless lending allegation on file", True),
    ReasonCode(D_NO_VIABLE_SCENARIO, 40, "No evaluated scenario passed every intervention", True),
])
