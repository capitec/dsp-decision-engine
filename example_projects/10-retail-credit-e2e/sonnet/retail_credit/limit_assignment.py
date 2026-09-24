"""P15 -- Limit assignment (spec 10 §5.16): declared stub only.

Entry point 1 never runs P15 (§5.20: it runs on 2, 3, 4). This module
exists so the eighteen-phase skeleton is complete (`retail_credit.phases`
lists it, `fleshed_out=False`) and so `retail_credit.entry_points`' matrix
has somewhere to point when entry points 2-4 are exercised at the routing
level (`tests/test_entry_points.py`) -- it is not called by `pipeline.py`.
Fleshing it out (the 4-dimension assignment matrix, the portfolio budget
allocation, the two-pass over-allocation loop L6) is explicitly out of
scope for this slice (SCOPE.md: "Skip products 11-40, P14 and P15 in
depth").
"""
from __future__ import annotations

PHASE_ID = 15
DECISION_POINTS = 66
