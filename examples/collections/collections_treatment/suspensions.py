"""Regulatory and status suspensions: which of today's treatments are still allowed.

Every suspension is checked every time, so the record names all that applied, not the first found. Each one
blocks a declared set of treatments; what is left is `permitted_treatment_codes`, and `treatment_after_suspensions`
checks the (overlaid) recommendation against it.
"""
from __future__ import annotations

from datetime import date, timedelta

from decider import allow_fallback, missing_as, step

from credit_core.vocab import ConsentChannel

from collections_treatment import vocab as v

_ACTIONS = frozenset(range(v.AUTOMATED_SMS, v.SETTLEMENT_OFFER + 1))
_CONTACT = frozenset(range(v.AUTOMATED_SMS, v.FIELD_VISIT + 1)) | {v.SETTLEMENT_OFFER}
_VOICE_AND_FIELD = frozenset({v.AUTOMATED_VOICE_MESSAGE, v.AGENT_CALL_LOW, v.AGENT_CALL_STANDARD,
                              v.AGENT_CALL_HIGH, v.FIELD_VISIT})
_CONSENT_CHANNEL_TREATMENTS = {
    ConsentChannel.SMS: {v.AUTOMATED_SMS}, ConsentChannel.PUSH: {v.IN_APP_MESSAGE},
    ConsentChannel.VOICE: {v.AUTOMATED_VOICE_MESSAGE, v.AGENT_CALL_LOW, v.AGENT_CALL_STANDARD, v.AGENT_CALL_HIGH},
    ConsentChannel.EMAIL: {v.AUTOMATED_EMAIL},
}

# code -> (scope, source, how it lifts, treatments it blocks). Cells computed per account override source and expiry.
RULES: dict[int, tuple[str, str, str, frozenset[int]]] = {
    101: ("contact_except_statutory_notices", "debt_review_registry", "event:stage_change",
          _ACTIONS - {v.PRE_LEGAL_NOTICE}),
    102: ("all_except_arrangement_servicing", "debt_review_registry", "event:stage_change", _ACTIONS),
    103: ("all_except_rearrangement_servicing", "debt_review_registry", "event:order_set_aside_or_clearance",
          _ACTIONS),
    104: ("partial_releases_103", "debt_review_registry", "event:prescribed_process_complete", _ACTIONS),
    105: ("all_normal_collections", "administration_registry", "event:discharge", _ACTIONS),
    106: ("all", "insolvency_registry", "event:rehabilitation", _ACTIONS | {v.WRITE_OFF_RECOMMENDATION}),
    107: ("all_client_contact", "status_feed", "event:estate_finalised", _ACTIONS),
    108: ("contact_and_legal", "complaints_register", "event:complaint_resolved", _ACTIONS),
    109: ("contact_and_legal", "ombud_register", "event:case_closed", _ACTIONS),
    110: ("legal_and_agency_handover_written_contact_only", "dispute_log", "event:dispute_resolved",
          _VOICE_AND_FIELD | {v.AGENCY_HANDOVER, v.PRE_LEGAL_NOTICE, v.LEGAL_HANDOVER}),
    111: ("all_while_performing", "hardship_declarations", "event:first_missed_instalment", _ACTIONS),
    112: ("legal_handover_only", "notice_proof", "date", frozenset({v.LEGAL_HANDOVER})),
    113: ("all_non_legal", "attorney_panel_feed", "event:judgment_or_withdrawal", _ACTIONS - {v.LEGAL_HANDOVER}),
    114: ("all_collection_activity", "prescription_derivation", "permanent_unless_interrupted", _ACTIONS),
    115: ("sms_voice_and_field", "contact_hours_calendar", "event:window_boundary",
          _VOICE_AND_FIELD | {v.AUTOMATED_SMS}),
    116: ("channel", "core.consent", "event:consent_re_obtained", frozenset()),
    117: ("contact_channels", "contact_history_window", "date", frozenset(range(v.AUTOMATED_SMS, v.FIELD_VISIT + 1))),
    118: ("voice_and_field", "client_preference", "event:withdrawn_by_client", _VOICE_AND_FIELD),
    119: ("all_except_promise_reminder", "promise", "date", _ACTIONS - {v.AUTOMATED_SMS}),
    120: ("contact_channels", "core.consent", "event:contact_point_validated",
          frozenset(range(v.AUTOMATED_SMS, v.AGENT_CALL_HIGH + 1)) | {v.SETTLEMENT_OFFER}),
}


def add_business_days(start: date, days: int) -> date:
    """`start` plus `days` Monday-to-Friday days.

    Example: `add_business_days(date(2026, 8, 14), 10) == date(2026, 8, 28)`.
    """
    # ponytail: no public-holiday calendar; a holiday table would replace the weekday test
    day = start
    while days:
        day += timedelta(days=1)
        days -= day.weekday() < 5
    return day


@allow_fallback
def evaluate_suspensions(
    decision_date: date,
    debt_review_stage_code: int = missing_as(0),
    debt_review_case_number: str = missing_as(""),
    debt_review_default_date: date | None = None,
    administration_order_active: bool = missing_as(False),
    insolvency_active: bool = missing_as(False),
    deceased: bool = missing_as(False),
    complaint_open: bool = missing_as(False),
    complaint_logged_date: date | None = None,
    ombud_referral_open: bool = missing_as(False),
    dispute_raised: bool = missing_as(False),
    hardship_arrangement_performing: bool = missing_as(False),
    notice_delivered_date: date | None = None,
    litigation_in_progress: bool = missing_as(False),
    prescription_date: date | None = None,
    outside_permitted_hours: bool = missing_as(False),
    consent_withdrawn_channels: list[int] = missing_as([]),
    frequency_cap_reached: bool = missing_as(False),
    frequency_cap_window_reset_date: date | None = None,
    written_communication_only: bool = missing_as(False),
    promise_date: date | None = None,
    no_valid_contact_point: bool = missing_as(False),
) -> tuple[list[int], list[str], list[str], list[str], list[int]]:
    """Every suspension in force on `decision_date`, as parallel lists, and the treatments still permitted.

    Returns `suspension_codes`, `suspension_scopes`, `suspension_sources`, `suspension_expiries` (a date or the
    event that lifts it) and `permitted_treatment_codes`.
    """
    fired: list[tuple[int, str, str, str, frozenset[int]]] = []  # code, scope, source, expiry, blocked

    def fire(code: int, source: str | None = None, expiry: str | None = None, scope=None, blocked=None) -> None:
        default_scope, default_source, default_expiry, default_blocked = RULES[code]
        fired.append((code, scope or default_scope, source or default_source, expiry or default_expiry,
                      default_blocked if blocked is None else frozenset(blocked)))

    case = f"debt_review_registry:{debt_review_case_number}"
    if debt_review_stage_code in (1, 2, 3):
        fire(100 + debt_review_stage_code, case)
    elif debt_review_stage_code == 4:
        # The court order's protection runs until the prescribed process after the default has finished.
        lifts = None if debt_review_default_date is None else debt_review_default_date + timedelta(days=60)
        if lifts is None or decision_date < lifts:
            fire(104, case, "unknown" if lifts is None else lifts.isoformat())
    if administration_order_active:
        fire(105)
    if insolvency_active:
        fire(106)
    if deceased:
        fire(107)
    if complaint_open:
        lifts = None if complaint_logged_date is None else add_business_days(complaint_logged_date, 15)
        fire(108, expiry="unknown" if lifts is None else lifts.isoformat())
    if ombud_referral_open:
        fire(109)
    if dispute_raised:
        fire(110)
    if hardship_arrangement_performing:
        fire(111)
    if notice_delivered_date is not None:
        lifts = add_business_days(notice_delivered_date, 10)
        if decision_date < lifts:
            fire(112, f"notice_proof:{notice_delivered_date.isoformat()}", lifts.isoformat())
    if litigation_in_progress:
        fire(113)
    if prescription_date is not None and decision_date >= prescription_date:
        fire(114)
    if outside_permitted_hours:
        fire(115)
    for channel in consent_withdrawn_channels or []:
        fire(116, scope=f"channel:{channel}", blocked=_CONSENT_CHANNEL_TREATMENTS.get(channel, ()))
    if frequency_cap_reached:
        fire(117, expiry="unknown" if frequency_cap_window_reset_date is None
             else frequency_cap_window_reset_date.isoformat())
    if written_communication_only:
        fire(118)
    if promise_date is not None:
        lifts = add_business_days(promise_date, 2)
        if decision_date <= lifts:
            fire(119, f"promise:{promise_date.isoformat()}", lifts.isoformat())
    if no_valid_contact_point:
        fire(120)

    blocked = set().union(*(b for *_, b in fired))
    return (
        [f[0] for f in fired], [f[1] for f in fired], [f[2] for f in fired], [f[3] for f in fired],
        sorted(set(v.TREATMENT_NAMES) - blocked),
    )


evaluate_suspensions_step = step(
    evaluate_suspensions,
    outputs=("suspension_codes", "suspension_scopes", "suspension_sources", "suspension_expiries",
             "permitted_treatment_codes"),
)


@allow_fallback
def treatment_after_suspensions(overlaid_treatment_code: int, permitted_treatment_codes: list[int]) -> tuple[int, bool]:
    """The treatment to apply today (`treatment_code`) and whether a suspension turned it into no action.

    A blocked treatment is not downgraded to a cheaper one: the account gets no action today.
    """
    blocked = overlaid_treatment_code not in permitted_treatment_codes
    return (v.NO_ACTION if blocked else overlaid_treatment_code), blocked


treatment_after_suspensions_step = step(treatment_after_suspensions, outputs=("treatment_code", "suspension_blocked"))


@allow_fallback
def pre_prescription_flag(decision_date: date, prescription_date: date | None = None) -> bool:
    """True within 60 days before the debt prescribes."""
    return prescription_date is not None and 0 <= (prescription_date - decision_date).days <= 60
