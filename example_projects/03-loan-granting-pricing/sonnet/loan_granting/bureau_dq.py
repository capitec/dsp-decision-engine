"""Stage 5.3 -- bureau retrieval and data-quality verdict (spec 03 §5.3).

Written from scratch rather than layered on `credit_core.bureau`: 00's
`normalise_bureau` (per its own NOTES.md, "thin: one bureau schema") hardcodes
a 90-day staleness window and returns aggregates, not the DQ-0..DQ-3
classification or the three-way no-hit/no-accounts/enquiry-failed distinction
03 §5.3 needs, and 03's staleness tolerance (45 days) must be a per-product
`param()`, not a module constant -- the same gap 02's NOTES.md records for
the same module. This module classifies data quality on top of the fields
`sample_request.json` already supplies (bureau is retrieved upstream, exactly
like project 00/02's `bureau_response` input); it does not re-normalise raw
bureau responses.
"""
from __future__ import annotations

from datetime import date

from decider import missing_as, param, step

DQ_CLEAN = "DQ-0"
DQ_MINOR = "DQ-1"
DQ_MATERIAL = "DQ-2"
DQ_UNUSABLE = "DQ-3"

NO_HIT = "no_hit"
NO_ACCOUNTS = "no_accounts"
ENQUIRY_FAILED = "enquiry_failed"
HAS_RECORD = "has_record"


def bureau_availability_code(
    bureau_response_status: str = missing_as("ok"),
    bureau_account_count: int = missing_as(0),
) -> str:
    """The three-way distinction 03 §5.3 requires never to collapse into one another."""
    if bureau_response_status == "failed":
        return ENQUIRY_FAILED
    if bureau_response_status == "no_hit":
        return NO_HIT
    if bureau_account_count == 0:
        return NO_ACCOUNTS
    return HAS_RECORD


def data_quality_verdict(
    bureau_availability_code: str,
    bureau_identity_subject_count: int = missing_as(1),
    bureau_has_unparseable_account: bool = missing_as(False),
    bureau_account_list_truncated: bool = missing_as(False),
    bureau_has_out_of_domain_status: bool = missing_as(False),
    bureau_identity_mismatch: bool = missing_as(False),
) -> str:
    if bureau_availability_code == ENQUIRY_FAILED or bureau_identity_mismatch:
        return DQ_UNUSABLE
    if bureau_identity_subject_count > 1 or bureau_account_list_truncated or bureau_has_out_of_domain_status:
        return DQ_MATERIAL
    if bureau_has_unparseable_account:
        return DQ_MINOR
    return DQ_CLEAN


def bureau_is_stale(
    bureau_as_of_date: date | None, decision_date: date,
    staleness_tolerance_days: int = param(45, ge=0),
) -> bool:
    if bureau_as_of_date is None:
        return True
    return (decision_date - bureau_as_of_date).days > staleness_tolerance_days


def is_thin_file(bureau_account_count: int = missing_as(0), bureau_history_months: float = missing_as(0.0)) -> bool:
    return bureau_account_count < 3 or bureau_history_months < 12.0


def bureau_referral_required(data_quality_verdict: str) -> bool:
    """DQ-2/DQ-3 refer; a data-quality problem is never a decline (03 §5.3)."""
    return data_quality_verdict in (DQ_MATERIAL, DQ_UNUSABLE)


bureau_availability_code_step = step(bureau_availability_code)
data_quality_verdict_step = step(data_quality_verdict)
bureau_is_stale_step = step(bureau_is_stale)
is_thin_file_step = step(is_thin_file)
bureau_referral_required_step = step(bureau_referral_required)
