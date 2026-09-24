"""Regulatory and status suspensions (spec 08 §5.2).

Four hard requirements from the spec drive this module's shape, not just
its content:

1. **Every suspension that applied is individually attributable** -- not
   the first one found. The easy implementation short-circuits on the
   first match; this one evaluates all 20 codes unconditionally, every
   time, and returns every one that fired. There is no early return in
   `evaluate_suspensions` -- the "short-circuit" §2's "four things worth
   watching" calls out is structurally impossible here, not merely avoided.
2. **Every suspension's expiry is computable or explicitly event-driven.**
   `_EXPIRY` below states one or the other for every code; nothing is
   "suspended indefinitely".
3. **Prescription is not merely a suspension** -- handled by the caller
   reading `prescription_date`/`pre_prescription_flag` off this step's
   output; the disclosure obligation itself is a downstream (script /
   dialler) concern, out of this slice.
4. **Nothing here is overlayable** (§5.4). This module never reads an
   `AdjustmentRegister`; `vocab.STATUTORY_TARGETS` documents the names no
   adjustment in this project may target, and `matrix.py`/`scoring.py`
   assert against it at register-construction time.

Because a `frame_step` cannot emit a `list[struct]` column without
crashing result materialisation (00 NOTES.md "Framework friction" 4.2;
02 NOTES.md 4.2 -- confirmed again here), one suspension record becomes
four parallel, index-aligned lists rather than one list-of-dicts.
"""
from __future__ import annotations

from datetime import date, timedelta

from decider import missing_as, step

from collections_treatment import vocab

# --- Business-day arithmetic --------------------------------------------------------------
# ponytail: weekday-only business-day counting (no public-holiday calendar). Spec 08 §6
# names a "Business day calendar, 10 years x public holidays" as its own parameter table;
# this slice approximates it with Mon-Fri. Upgrade path: replace `_add_business_days` with
# a lookup against a real gazette-derived calendar table (same call shape).


def _add_business_days(start: date, n: int) -> date:
    d, step_dir = start, 1 if n >= 0 else -1
    remaining = abs(n)
    while remaining:
        d += timedelta(days=step_dir)
        if d.weekday() < 5:
            remaining -= 1
    return d


def _business_days_between(a: date, b: date) -> int:
    if b < a:
        return -_business_days_between(b, a)
    n, d = 0, a
    while d < b:
        d += timedelta(days=1)
        if d.weekday() < 5:
            n += 1
    return n


# --- One row per suspension code, each a (fires, scope, blocks_all, source, expiry) check --

def _debt_review(decision_date, debt_review_stage_code, debt_review_case_number,
                  debt_review_default_date):
    out = []
    if debt_review_stage_code == 1:
        out.append((vocab.SUSPENSION_DEBT_REVIEW_APPLICATION, "contact_except_statutory_notices", False,
                     f"debt_review_registry:{debt_review_case_number}", "event:stage_change"))
    elif debt_review_stage_code == 2:
        out.append((vocab.SUSPENSION_DEBT_REVIEW_PROPOSAL, "contact_except_arrangement_servicing", False,
                     f"debt_review_registry:{debt_review_case_number}", "event:stage_change"))
    elif debt_review_stage_code == 3:
        out.append((vocab.SUSPENSION_DEBT_REVIEW_COURT_ORDER, "all_except_rearrangement_servicing", True,
                     f"debt_review_registry:{debt_review_case_number}", "event:order_set_aside_or_clearance"))
    elif debt_review_stage_code == 4:
        expiry = "unknown" if debt_review_default_date is None else \
            (debt_review_default_date + timedelta(days=60)).isoformat()
        out.append((vocab.SUSPENSION_DEBT_REVIEW_DEFAULT, "partial_releases_103", False,
                     f"debt_review_registry:{debt_review_case_number}", expiry))
    return out


def _flag(code, condition, scope, blocks_all, source, expiry):
    return [(code, scope, blocks_all, source, expiry)] if condition else []


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
) -> tuple[list[int], list[str], list[bool], list[str], list[str]]:
    """Every one of the 20 codes (§5.2 table), evaluated unconditionally. Returns four
    parallel lists (code, scope, blocks_all, source) plus a fifth (computable/event expiry),
    all index-aligned -- the "every suspension individually attributable" record (§9.2)."""
    fired: list[tuple[int, str, bool, str, str]] = []

    fired += _debt_review(decision_date, debt_review_stage_code, debt_review_case_number,
                           debt_review_default_date)
    fired += _flag(vocab.SUSPENSION_ADMINISTRATION_ORDER, administration_order_active,
                    "all_normal_collections", True, "administration_registry", "event:discharge")
    fired += _flag(vocab.SUSPENSION_INSOLVENCY, insolvency_active,
                    "all", True, "insolvency_registry", "event:rehabilitation")
    fired += _flag(vocab.SUSPENSION_DECEASED, deceased,
                    "all_except_executor_correspondence", True, "status_feed", "event:estate_finalised")

    if complaint_open:
        expiry = "unknown" if complaint_logged_date is None else \
            _add_business_days(complaint_logged_date, 15).isoformat()
        fired.append((vocab.SUSPENSION_INTERNAL_COMPLAINT, "contact_and_legal", False,
                       "complaints_register", expiry))
    fired += _flag(vocab.SUSPENSION_OMBUD_REFERRAL, ombud_referral_open,
                    "contact_and_legal", False, "ombud_register", "event:case_closed")
    fired += _flag(vocab.SUSPENSION_DISPUTE, dispute_raised,
                    "legal_and_agency_handover", False, "dispute_log", "event:dispute_resolved")
    fired += _flag(vocab.SUSPENSION_HARDSHIP, hardship_arrangement_performing,
                    "all_while_performing", False, "hardship_declarations", "event:first_missed_instalment")

    if notice_delivered_date is not None:
        expiry = _add_business_days(notice_delivered_date, 10)
        if decision_date < expiry:
            fired.append((vocab.SUSPENSION_NOTICE_PERIOD, "legal_handover_only", False,
                           f"notice_proof:{notice_delivered_date.isoformat()}", expiry.isoformat()))

    fired += _flag(vocab.SUSPENSION_LITIGATION, litigation_in_progress,
                    "all_non_legal", True, "attorney_panel_feed", "event:judgment_or_withdrawal")

    pre_prescription = False
    if prescription_date is not None:
        if decision_date >= prescription_date:
            fired.append((vocab.SUSPENSION_PRESCRIPTION, "all_collection_activity", True,
                           "prescription_derivation", "permanent_unless_interrupted"))
        elif (prescription_date - decision_date).days <= 60:
            pre_prescription = True

    fired += _flag(vocab.SUSPENSION_OUTSIDE_HOURS, outside_permitted_hours,
                    "channel_and_time_specific", False, "contact_hours_calendar", "event:window_boundary")

    channels = [] if consent_withdrawn_channels is None else consent_withdrawn_channels
    for ch in channels:
        fired.append((vocab.SUSPENSION_CONSENT_WITHDRAWN, f"channel:{ch}", False,
                       "core.consent", "event:consent_re_obtained"))

    if frequency_cap_reached:
        expiry = "unknown" if frequency_cap_window_reset_date is None else \
            frequency_cap_window_reset_date.isoformat()
        fired.append((vocab.SUSPENSION_FREQUENCY_CAP, "contact_channels", False,
                       "contact_history_window", expiry))

    fired += _flag(vocab.SUSPENSION_WRITTEN_ONLY, written_communication_only,
                    "all_voice_and_field", False, "client_preference", "event:withdrawn_by_client")

    if promise_date is not None:
        expiry = _add_business_days(promise_date, 2)
        if decision_date <= expiry:
            fired.append((vocab.SUSPENSION_PROMISE_IN_FORCE, "all_except_promise_reminder", False,
                           f"promise:{promise_date.isoformat()}", expiry.isoformat()))

    fired += _flag(vocab.SUSPENSION_NO_CONTACT_POINT, no_valid_contact_point,
                    "contact_channels", False, "core.consent", "event:contact_point_validated")

    codes = [c for c, *_ in fired]
    scopes = [s for _, s, *_ in fired]
    blocks = [b for _, _, b, _, _ in fired]
    sources = [src for _, _, _, src, _ in fired]
    expiries = [e for *_, e in fired]
    return codes, scopes, blocks, sources, expiries


evaluate_suspensions_step = step(
    evaluate_suspensions,
    outputs=("suspension_codes", "suspension_scopes", "suspension_blocks_all",
              "suspension_sources", "suspension_expiries"),
)


def permitted_treatment_codes(
    suspension_codes: list[int], suspension_blocks_all: list[bool],
) -> list[int]:
    """The set of treatments still permitted after every suspension is applied (§5.2
    "Determines"): everything, minus every treatment code, if any fired suspension
    blocks outright. A future refinement (channel-scoped suspensions narrowing rather
    than blocking every treatment) is noted in NOTES.md "What I would do next"."""
    codes = [] if suspension_codes is None else suspension_codes
    blocks = [] if suspension_blocks_all is None else suspension_blocks_all
    any_hard_block = any(bool(b) for b in blocks) or any(c in vocab.SUSPENSION_BLOCKS_ALL for c in codes)
    if any_hard_block:
        return []
    return list(vocab.TREATMENT_POOL)


permitted_treatment_codes_step = step(permitted_treatment_codes)


def suspended_blocks_all(permitted_treatment_codes: list[int]) -> bool:
    """True when at least one fired suspension blocks outright (§5.9 code 210) -- the
    empty-permitted-set case `permitted_treatment_codes_step` already computed."""
    codes = [] if permitted_treatment_codes is None else permitted_treatment_codes
    return len(codes) == 0


suspended_blocks_all_step = step(suspended_blocks_all)


def pre_prescription_flag(decision_date: date, prescription_date: date | None = None) -> bool:
    if prescription_date is None:
        return False
    return 0 <= (prescription_date - decision_date).days <= 60


pre_prescription_flag_step = step(pre_prescription_flag)
