"""Admission: reject a malformed request, resolve who the client is, and note whether the bureau is down."""
from __future__ import annotations

from decider import allow_fallback, missing_as, param, step

from retail_credit.vocab import DEGRADED_BUREAU_DOWN, DEGRADED_NONE

# rejection reasons: a rejection is not a credit decline and never appears in `decline_reason_codes`
R_AMOUNT_NOT_POSITIVE = 9001
R_TERM_NOT_POSITIVE = 9002
R_NOT_THIS_PRODUCT = 9003
R_CHANNEL_UNKNOWN = 9004
R_NOT_A_NEW_APPLICATION = 9005

NEW_APPLICATION = 1
FLEX_LOAN = 10

# identity_resolution_path
AUTHENTICATED_SESSION = 1
IDENTITY_NUMBER_MATCH = 2
IDENTITY_VERIFICATION = 3
PROBABILISTIC_MATCH = 4

_CONFIDENCE = {AUTHENTICATED_SESSION: 1.00, IDENTITY_NUMBER_MATCH: 0.97, IDENTITY_VERIFICATION: 0.92}


@allow_fallback
def validate_request(
    requested_amount: float, term_months: int, product_code: float, channel_code: int, entry_point_code: int,
) -> tuple[bool, list[int]]:
    """(p01_valid, p01_rejection_reasons): every check runs, so a rejected request gets every reason."""
    failed = [
        (requested_amount <= 0, R_AMOUNT_NOT_POSITIVE), (term_months <= 0, R_TERM_NOT_POSITIVE),
        (product_code != FLEX_LOAN, R_NOT_THIS_PRODUCT), (channel_code <= 0, R_CHANNEL_UNKNOWN),
        (entry_point_code != NEW_APPLICATION, R_NOT_A_NEW_APPLICATION),
    ]
    reasons = [code for hit, code in failed if hit]
    return not reasons, reasons


def resolve_identity(
    has_authenticated_session: bool = missing_as(False), identity_number_match: bool = missing_as(False),
    identity_verification_confirmed: bool = missing_as(False), probabilistic_match_confidence: float = missing_as(0.0),
    confidence_floor: float = param(0.88, ge=0.5, le=1.0),
) -> tuple[int, float, bool]:
    """(identity_resolution_path, identity_confidence, identity_is_trustworthy): the first path that holds wins.

    A match below `confidence_floor` is not a resolved client.
    """
    if has_authenticated_session:
        path = AUTHENTICATED_SESSION
    elif identity_number_match:
        path = IDENTITY_NUMBER_MATCH
    elif identity_verification_confirmed:
        path = IDENTITY_VERIFICATION
    else:
        return PROBABILISTIC_MATCH, probabilistic_match_confidence, probabilistic_match_confidence >= confidence_floor
    return path, _CONFIDENCE[path], True


@allow_fallback
def source_degradation(bureau_available: bool = missing_as(True)) -> tuple[int, list[int]]:
    """(degraded_mode_code, source_degradation_codes): the one degraded mode this flow models is the bureau being down."""
    return (DEGRADED_BUREAU_DOWN, [DEGRADED_BUREAU_DOWN]) if not bureau_available else (DEGRADED_NONE, [])


def bureau_down_within_envelope(
    degraded_mode_code: int, segment_code: float, requested_amount: float, term_months: int,
    internal_tenure_months: float = missing_as(0.0), worst_arrears_months: float = missing_as(0.0),
    max_amount: float = param(25_000.0), max_term: int = param(36), min_tenure_months: float = param(24.0),
) -> bool:
    """With the bureau down a request is approved only inside a reduced envelope; outside it the answer is refer.

    True when the bureau is up. Degradation narrows what can be approved; it never invents evidence.
    """
    if degraded_mode_code != DEGRADED_BUREAU_DOWN:
        return True
    return (requested_amount <= max_amount and term_months <= max_term and 3 <= segment_code <= 5
            and internal_tenure_months >= min_tenure_months and worst_arrears_months <= 0)


validate_request_step = step(validate_request, outputs=("p01_valid", "p01_rejection_reasons"))
resolve_identity_step = step(resolve_identity, outputs=("identity_resolution_path", "identity_confidence",
                                                         "identity_is_trustworthy"))
source_degradation_step = step(source_degradation, outputs=("degraded_mode_code", "source_degradation_codes"))
bureau_down_within_envelope_step = step(bureau_down_within_envelope, output="bureau_down_envelope_ok")
