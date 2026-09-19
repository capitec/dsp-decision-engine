"""P04 data acquisition orchestration — 47 decision points. Spec 5.5. Elided
from phases/__init__.py "for length"; this package is the real file.

Decides WHICH of nine sources to call, in what order and concurrency, and
owns the ARRIVAL CONTRACT — what a late, partial or schema-invalid response
means, and which of those are degradations versus failures. A response that
arrives after its budget is never used even if the decision has not yet
completed, because a decision that sometimes uses a late response and
sometimes does not is not replayable.

Its ordering against P05 (fraud) INVERTS by entry point — O-06 in
ordering.py, the first genuine cycle. This file does not encode the inversion
itself; ordering.py's `cycle_break` does, and this file is one of the two
edges it breaks.
"""

from __future__ import annotations

from decider2 import module, param

def retrieval_plan(entry_point_code: int, product_code: int, income_evidence_present: bool,
                   resolution_path: str, tenure_months: float) -> dict:
    """Per source: issue or skip. Bureau unless a view under 40 days is held;
    statement aggregator on thin income evidence or >18% declared variance;
    identity verification on resolution path R3/R4; employer confirmation
    under 9 months' tenure or a listed employer; and so on for all nine."""
    pass  # decide, per source in section 4.4, whether this decision issues it

def wave_assignment(retrieval_plan: dict, consent_state: dict) -> dict:
    """Bureau, device, identity and internal state concurrently once consent
    is established (wave 1); statement aggregator after identity resolves,
    employer confirmation conditional on the bureau response (wave 2)."""
    pass  # partition retrieval_plan's issued sources into wave 1 / wave 2

def arrival_contract(wave_assignment: dict, response_timestamps: dict,
                     response_budgets_ms: dict) -> dict:
    """A response arriving after its own budget is DISCARDED even if the
    decision has not completed — determinism over freshness."""
    pass  # per source: on-time / late-discarded / partial / schema-invalid / absent

def what_arrived(arrival_contract: dict) -> dict:
    pass  # the accepted response set, feeding P05/P06's degradation checks

Orchestrate = module(retrieval_plan, wave_assignment, arrival_contract, what_arrived,
                     name="orchestration")
