"""Consent: one verdict for the message, then whether the delivery channel may carry it."""
from __future__ import annotations

from decider import missing_as, step

from credit_core.vocab import ConsentChannel, ConsentVerdict

__all__ = ["ConsentChannel", "ConsentVerdict", "consent_verdict", "channel_permitted"]


def consent_verdict(
    marketing_opt_out: bool = missing_as(False), is_regulated_notice: bool = missing_as(False),
    data_sharing_consent: bool = missing_as(True),
) -> int:
    """A regulated notice is never suppressed by a marketing preference; anything else needs consent."""
    if is_regulated_notice:
        return ConsentVerdict.REGULATED_NOTICE
    if marketing_opt_out or not data_sharing_consent:
        return ConsentVerdict.SUPPRESSED
    return ConsentVerdict.PERMITTED


def channel_permitted(
    consent_channel_code: int, consent_verdict: int,
    sms_permitted: bool = missing_as(False), push_permitted: bool = missing_as(False),
    voice_permitted: bool = missing_as(False), email_permitted: bool = missing_as(False),
) -> bool:
    """Whether the message may go out on `consent_channel_code` (a `ConsentChannel`). A regulated notice always may."""
    if consent_verdict == ConsentVerdict.REGULATED_NOTICE:
        return True
    if consent_verdict == ConsentVerdict.SUPPRESSED:
        return False
    if consent_channel_code == ConsentChannel.SMS:
        return sms_permitted
    if consent_channel_code == ConsentChannel.PUSH:
        return push_permitted
    if consent_channel_code == ConsentChannel.VOICE:
        return voice_permitted
    return consent_channel_code == ConsentChannel.EMAIL and email_permitted


consent_verdict_step = step(consent_verdict)
channel_permitted_step = step(channel_permitted)
