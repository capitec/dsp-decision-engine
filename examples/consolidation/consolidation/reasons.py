"""Two taxonomies that must not be confused: why a scenario was not viable, and why a client was declined or referred."""
from credit_core.reason_codes import ReasonCode, ReasonCodeRegistry

# Scenario rejections. A policy intervention CON-INT-n is 1000 + n; the rest are 11xx.
NOT_PRICED, AFFORDABILITY_FAIL, OUTSIDE_PRODUCT_LIMITS = 1101, 1102, 1103

SCENARIO_REJECTIONS = ReasonCodeRegistry("consolidation-rejections-2026.09", [
    ReasonCode(1001, 21, "More accounts settled than the product permits", True),
    ReasonCode(1003, 23, "Instalment reduction below the product's minimum", True),
    ReasonCode(1004, 14, "Total cost of credit rises more than the anti-harm ceiling allows", True),
    ReasonCode(1005, 25, "Term extends too far beyond the settled accounts' longest remaining term", True),
    ReasonCode(1006, 26, "New rate is above the settled accounts' weighted average rate", True),
    ReasonCode(1008, 28, "New money above the permitted share of the settlement total or the absolute cap", True),
    ReasonCode(1009, 29, "Post-consolidation debt service ratio above the ceiling", True),
    ReasonCode(1010, 30, "Too little of the advance goes to external creditors", True),
    ReasonCode(1013, 33, "The client has reached the maximum consolidations in 24 months", True),
    ReasonCode(1014, 34, "Discretionary income after the new instalment is below the floor", True),
    ReasonCode(NOT_PRICED, 90, "The product's rate card has no cell for this amount, term and grade", False),
    ReasonCode(AFFORDABILITY_FAIL, 5, "The new instalment exceeds what the client can afford", True),
    ReasonCode(OUTSIDE_PRODUCT_LIMITS, 80, "The amount or limit is outside the product's limits", False),
])

DEBT_REVIEW, ADMINISTRATION, NO_INCOME, NO_BUREAU_VIEW = 2001, 2002, 2003, 2004
SERIAL_CONSOLIDATION, RECKLESS_ALLEGATION, NOMINATION_CONFLICT, NO_VIABLE_SCENARIO = 2005, 2006, 2007, 2008
NOT_AFFORDABLE_AS_REQUESTED, EVIDENCE_INSUFFICIENT = 2009, 2010

CLIENT_REASONS = ReasonCodeRegistry("consolidation-client-2026.09", [
    ReasonCode(DEBT_REVIEW, 1, "The client is under debt review", True),
    ReasonCode(ADMINISTRATION, 2, "The client is under administration or sequestration", True),
    ReasonCode(NO_INCOME, 5, "No verified income at the decision date", True),
    ReasonCode(NO_BUREAU_VIEW, 6, "No bureau view could be obtained", True),
    ReasonCode(RECKLESS_ALLEGATION, 15, "An active reckless lending allegation is on file", True),
    ReasonCode(SERIAL_CONSOLIDATION, 20, "A consolidation concluded in the last 6 months", True),
    ReasonCode(NOMINATION_CONFLICT, 25, "An account the client asked to settle cannot be settled", False),
    ReasonCode(NO_VIABLE_SCENARIO, 40, "No scenario passed every intervention", True),
    ReasonCode(EVIDENCE_INSUFFICIENT, 7, "Affordability could not be established on the evidence", True),
    ReasonCode(NOT_AFFORDABLE_AS_REQUESTED, 45, "The request is not affordable as made and consolidation does not apply", True),
])
