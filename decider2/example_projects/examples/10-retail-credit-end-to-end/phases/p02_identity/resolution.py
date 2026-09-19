"""P02 client and identity resolution — 33 decision points. Spec 5.3. Not
imported by the current phases/__init__.py (P02 is one of the eight elided
"for length" there — see that file's closing comment); this package is the
real file the elision refers to.

Four resolution paths tried in order, each with its own confidence. A match
below the 0.88 floor is a FIRST-CLASS OUTCOME (refer, queue 2), not an error,
and a probabilistic match resolving to two clients is likewise first-class
(0.6% of entry point 1). The related-party set this phase builds is read five
phases later by P09's CAP-0301 (O-11, ordering.py) — a cost P02 pays for a
consumer it cannot see (spec 5.21's "producing a value is an obligation to
consumers you cannot see").

Runs in a REDUCED form on entry points 3 and 4 (identity pre-resolved in the
overnight snapshot); spec 5.3 requires that reduced form be PROVABLY
EQUIVALENT to the full form (certification/strata.py:P02_EQUIVALENCE).
"""

from __future__ import annotations

from decider2 import module, param

def r1_authenticated_session(session_token: str | None) -> tuple[int | None, float]:
    """Confidence 1.00. 61% of resolutions."""
    pass  # resolve client_id from an authenticated session, confidence 1.00

def r2_exact_identity_match(identity_number: str) -> tuple[int | None, float]:
    """Confidence 0.97. 19%."""
    pass  # exact match on the internal master

def r3_identity_verification_service(identity_number: str, biographical_fields: dict
                                     ) -> tuple[int | None, float]:
    """Confidence 0.92. 14%. Degrades: unavailable -> R4 becomes the fallback,
    confidence floor rises to 0.94 (degradation/sources.py IDENTITY_SERVICE)."""
    pass  # confirm via the external identity verification service

def r4_probabilistic_match(name: str, date_of_birth: str, contact_details: dict
                           ) -> tuple[list[int], float]:
    """Confidence 0.55-0.85. 6%. May resolve to TWO clients — a first-class
    outcome, not an error."""
    pass  # probabilistic match on name, DOB, contact; may return multiple candidates

def resolved_client(r1_authenticated_session, r2_exact_identity_match,
                    r3_identity_verification_service, r4_probabilistic_match,
                    confidence_floor: float = param(0.88, ge=0, le=1)) -> dict:
    """First path in order that clears the floor wins. Below the floor:
    outcome_code refer, queue 2, candidate set attached — never a hard
    failure."""
    pass  # try R1..R4 in order; below confidence_floor, emit the queue-2 outcome

def related_party_set(resolved_client: dict, capacity: int = param(12, ge=0, le=12)) -> list[int]:
    """Up to 12 identifiers. Nothing in THIS phase needs it — it exists for
    P09's CAP-0301, five phases downstream (O-11)."""
    pass  # the expensive, set-shaped graph query; run here, once, for a later consumer

Resolve = module(r1_authenticated_session, r2_exact_identity_match,
                 r3_identity_verification_service, r4_probabilistic_match,
                 resolved_client, related_party_set, name="resolution")
