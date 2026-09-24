"""EP-1 (new-to-bank business application, products 50/51) is the one entry
point this project serves through `decider build`/`decider serve` -- SCOPE.md's
"one real path". L1 (annual review, `business_credit_e2e/review.py`) is not
decider-served: an annual review reads a facility's *prior* decision of record
(this project's own `history.DecisionHistoryStore`) and calls project 07's
per-account limit pipeline for revolving facilities -- both are orchestration
above a per-record `decider` step, the same shape project 07's own §5.8
population-level allocation takes (`limit_mgmt/allocation.py`'s docstring: "a
population-level decision... cannot be expressed as a per-record decider
step"). `review.annual_review` is tested directly (`tests/test_review.py`),
not through the CLI -- see NOTES.md.
"""
from __future__ import annotations

from business_credit_e2e import origination


def build():
    return origination.build()
