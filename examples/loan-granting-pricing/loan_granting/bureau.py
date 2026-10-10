"""The bureau enquiry: consent to make it, and whether the answer is fit to decide on.

A data-quality problem refers; it is never a decline, because a record that could not be parsed has not shown the
client to be a bad risk.
"""
from __future__ import annotations

from datetime import date

from decikit import allow_fallback, missing_as
from decikit.ops import Text

from credit_core.vocab import ConsentVerdict

HAS_RECORD, NO_ACCOUNTS, NO_HIT, ENQUIRY_FAILED = 0, 1, 2, 3  # bureau_availability_code: four different applicants
DQ_CLEAN, DQ_MINOR, DQ_MATERIAL, DQ_UNUSABLE = 0, 1, 2, 3  # data_quality_verdict


@allow_fallback
def consent_verdict(bureau_consent_given: bool, decision_date: date, bureau_consent_expires_on: date | None = None) -> int:
    """`PERMITTED` only with consent that has not expired by the decision date. Nothing is assumed: the field is required.

    The consent in `credit_core.consent` decides whether a message may go out; this one gates the enquiry itself.
    """
    valid = bureau_consent_given and (bureau_consent_expires_on is None or decision_date < bureau_consent_expires_on)
    return ConsentVerdict.PERMITTED if valid else ConsentVerdict.SUPPRESSED


def bureau_availability_code(bureau_response_status: Text, bureau_account_count: int = missing_as(0)) -> int:
    """`bureau_response_status` is "ok", "no_hit" or "failed"; a record with no accounts is not a failed enquiry."""
    if bureau_response_status == "failed":
        return ENQUIRY_FAILED
    if bureau_response_status == "no_hit":
        return NO_HIT
    return NO_ACCOUNTS if bureau_account_count == 0 else HAS_RECORD


def data_quality_verdict(
    bureau_availability_code: int, bureau_identity_subject_count: int = missing_as(1),
    bureau_has_unparseable_account: bool = missing_as(False), bureau_account_list_truncated: bool = missing_as(False),
    bureau_has_out_of_domain_status: bool = missing_as(False), bureau_identity_mismatch: bool = missing_as(False),
) -> int:
    """DQ-0 clean, DQ-1 minor defects, DQ-2 material (several subjects, truncated), DQ-3 unusable."""
    if bureau_availability_code == ENQUIRY_FAILED or bureau_identity_mismatch:
        return DQ_UNUSABLE
    if bureau_identity_subject_count > 1 or bureau_account_list_truncated or bureau_has_out_of_domain_status:
        return DQ_MATERIAL
    return DQ_MINOR if bureau_has_unparseable_account else DQ_CLEAN


def is_thin_file(
    bureau_account_count: int = missing_as(0), bureau_history_months: float = missing_as(0.0),
) -> bool:
    return bureau_account_count < 3 or bureau_history_months < 12.0


def bureau_referral_required(data_quality_verdict: int) -> bool:
    return data_quality_verdict >= DQ_MATERIAL
