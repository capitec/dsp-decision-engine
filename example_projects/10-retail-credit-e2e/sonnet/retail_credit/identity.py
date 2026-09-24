"""P02 -- Client and identity resolution (spec 10 §5.3).

Real logic for entry point 1 at working depth: the four resolution paths
(R1 authenticated session, R2 exact identity-number match, R3 identity
verification service, R4 probabilistic match), tried in order, each with
its own confidence, and the 0.88 confidence floor below which a match does
not become a resolved client (`identity_is_trustworthy = False`, which
routes to refer downstream rather than declining -- P02 itself never
declines).

Left out at this depth (declared, not silently dropped): the related-party
set (10 §5.3's "up to 12 identifiers", consumed five phases later by group
exposure in P09) and the identity-service degraded mode (confidence floor
rising to 0.94). This project implements exactly one declared degraded mode
end to end (bureau down, P04/P06/P07) per SCOPE.md; a second one here would
duplicate the same mechanism without teaching anything new about
composition, which is this project's actual subject.
"""
from __future__ import annotations

from decider import missing_as, param, step

R1_AUTHENTICATED_SESSION = 1
R2_IDENTITY_NUMBER_MATCH = 2
R3_IDENTITY_VERIFICATION = 3
R4_PROBABILISTIC_MATCH = 4

_CONFIDENCE = {
    R1_AUTHENTICATED_SESSION: 1.00,
    R2_IDENTITY_NUMBER_MATCH: 0.97,
    R3_IDENTITY_VERIFICATION: 0.92,
}


def resolve_identity(
    has_authenticated_session: bool = missing_as(False),
    identity_number_match: bool = missing_as(False),
    identity_verification_confirmed: bool = missing_as(False),
    probabilistic_match_confidence: float = missing_as(0.0),
    confidence_floor: float = param(0.88, ge=0.5, le=1.0),
) -> tuple[int, float, bool]:
    """(identity_resolution_path, identity_confidence, identity_is_trustworthy)."""
    if has_authenticated_session:
        path, confidence = R1_AUTHENTICATED_SESSION, _CONFIDENCE[R1_AUTHENTICATED_SESSION]
    elif identity_number_match:
        path, confidence = R2_IDENTITY_NUMBER_MATCH, _CONFIDENCE[R2_IDENTITY_NUMBER_MATCH]
    elif identity_verification_confirmed:
        path, confidence = R3_IDENTITY_VERIFICATION, _CONFIDENCE[R3_IDENTITY_VERIFICATION]
    else:
        path, confidence = R4_PROBABILISTIC_MATCH, probabilistic_match_confidence
    return path, confidence, confidence >= confidence_floor


resolve_identity_step = step(resolve_identity, outputs=("identity_resolution_path", "identity_confidence",
                                                          "identity_is_trustworthy"))
