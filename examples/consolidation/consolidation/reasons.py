"""Two taxonomies that must not be confused: why a scenario was not viable, and why a client was declined or referred.

The codes are named here; their severity order is the `scenario_rejection_codes` and `client_reason_codes` documents in
`configs/<version>/`.
"""

# Scenario rejections. A policy intervention CON-INT-n is 1000 + n; the rest are 11xx.
NOT_PRICED, AFFORDABILITY_FAIL, OUTSIDE_PRODUCT_LIMITS = 1101, 1102, 1103

DEBT_REVIEW, ADMINISTRATION, NO_INCOME, NO_BUREAU_VIEW = 2001, 2002, 2003, 2004
SERIAL_CONSOLIDATION, RECKLESS_ALLEGATION, NOMINATION_CONFLICT, NO_VIABLE_SCENARIO = 2005, 2006, 2007, 2008
NOT_AFFORDABLE_AS_REQUESTED, EVIDENCE_INSUFFICIENT = 2009, 2010
