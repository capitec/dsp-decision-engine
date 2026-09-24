"""`core.consent` -- consent and disclosure state (spec 00 §6.21; addendum A3).

00 keyed projects 01, 03, 04, 06, 07, 08 off "the consent vocabulary from
the library" without ever publishing one (addendum A3). This module
publishes it: per-channel permissions, `consent_record_id`, and a verdict.
A regulated notice is never suppressible by a marketing preference (07
§5.9) -- `REGULATED_NOTICE` is a distinct verdict from `SUPPRESSED`, not a
flag on it.
"""
from __future__ import annotations

from decider import missing_as, step

from credit_core.vocab import ConsentChannel, ConsentVerdict


def consent_verdict(
    channel_code: int, marketing_opt_out: bool = missing_as(False), is_regulated_notice: bool = missing_as(False),
    data_sharing_consent: bool = missing_as(True),
) -> int:
    if is_regulated_notice:
        return int(ConsentVerdict.REGULATED_NOTICE)
    if marketing_opt_out or not data_sharing_consent:
        return int(ConsentVerdict.SUPPRESSED)
    return int(ConsentVerdict.PERMITTED)


def channel_permitted(channel_code: int, consent_verdict: int) -> bool:
    return consent_verdict in (int(ConsentVerdict.PERMITTED), int(ConsentVerdict.REGULATED_NOTICE))


consent_verdict_step = step(consent_verdict)
channel_permitted_step = step(channel_permitted)

__all__ = ["ConsentChannel", "ConsentVerdict", "consent_verdict", "channel_permitted"]
