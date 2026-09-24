"""Locally declared collections vocabulary (spec 08 §4.1).

Project 00's library does not publish collections names ("The library ...
does not publish collections names. This project declares the following
and every other spec that touches collections uses them unchanged" --
08 §4.1). Everything here is that declaration: plain constants, no logic.
"""
from __future__ import annotations

# --- Treatments (§5.4) --------------------------------------------------------------------

NO_ACTION = 0
AUTOMATED_SMS = 1
AUTOMATED_EMAIL = 2
IN_APP_MESSAGE = 3
AUTOMATED_VOICE_MESSAGE = 4
AGENT_CALL_LOW = 5
AGENT_CALL_STANDARD = 6
AGENT_CALL_HIGH = 7
FIELD_VISIT = 8
AGENCY_HANDOVER = 9
PRE_LEGAL_NOTICE = 10
LEGAL_HANDOVER = 11
SETTLEMENT_OFFER = 12
WRITE_OFF_RECOMMENDATION = 13

# treatment_code -> capacity pool name (§5.9). Settlement/write-off don't consume a
# dedicated pool of their own here (§5.7/settlement offers are out of this slice's scope).
TREATMENT_POOL = {
    NO_ACTION: None,
    AUTOMATED_SMS: "sms",
    AUTOMATED_EMAIL: "email",
    IN_APP_MESSAGE: "in_app",
    AUTOMATED_VOICE_MESSAGE: "voice_message",
    AGENT_CALL_LOW: "early_agents",
    AGENT_CALL_STANDARD: "early_agents",
    AGENT_CALL_HIGH: "late_agents",
    FIELD_VISIT: "field",
    AGENCY_HANDOVER: "agency",
    PRE_LEGAL_NOTICE: "notices",
    LEGAL_HANDOVER: "legal",
    SETTLEMENT_OFFER: None,
    WRITE_OFF_RECOMMENDATION: None,
}

# Treatment intensity is monotonic in this order within an episode (§5.5 "Escalation").
TREATMENT_RANK = {
    NO_ACTION: 0, AUTOMATED_SMS: 1, AUTOMATED_EMAIL: 1, IN_APP_MESSAGE: 1,
    AUTOMATED_VOICE_MESSAGE: 2, AGENT_CALL_LOW: 3, AGENT_CALL_STANDARD: 4,
    AGENT_CALL_HIGH: 5, FIELD_VISIT: 6, AGENCY_HANDOVER: 7, PRE_LEGAL_NOTICE: 8,
    LEGAL_HANDOVER: 9, SETTLEMENT_OFFER: 5, WRITE_OFF_RECOMMENDATION: 10,
}

# --- Non-selection reasons (§5.9) ----------------------------------------------------------

NS_MATRIX_NO_ACTION = 200
NS_OVERLAY_SUPPRESSED = 205
NS_SUSPENDED = 210
NS_COOLING_OFF = 220
NS_INTERVAL_OR_CAP = 230
NS_PROMISE_IN_FORCE = 240
NS_BELOW_CUTOFF = 250
NS_SIBLING_COVERED = 260
NS_NO_CHANNEL = 270

# --- Suspensions (§5.2) ------------------------------------------------------------------
# code -> (description, scope, hard_block). hard_block=True means "blocks outright";
# False means "downgrades" (permits a reduced set rather than nothing). Every one of the
# 20 codes is evaluated on every account, every day -- never a short-circuit (§5.2).

SUSPENSION_DEBT_REVIEW_APPLICATION = 101
SUSPENSION_DEBT_REVIEW_PROPOSAL = 102
SUSPENSION_DEBT_REVIEW_COURT_ORDER = 103
SUSPENSION_DEBT_REVIEW_DEFAULT = 104
SUSPENSION_ADMINISTRATION_ORDER = 105
SUSPENSION_INSOLVENCY = 106
SUSPENSION_DECEASED = 107
SUSPENSION_INTERNAL_COMPLAINT = 108
SUSPENSION_OMBUD_REFERRAL = 109
SUSPENSION_DISPUTE = 110
SUSPENSION_HARDSHIP = 111
SUSPENSION_NOTICE_PERIOD = 112
SUSPENSION_LITIGATION = 113
SUSPENSION_PRESCRIPTION = 114
SUSPENSION_OUTSIDE_HOURS = 115
SUSPENSION_CONSENT_WITHDRAWN = 116
SUSPENSION_FREQUENCY_CAP = 117
SUSPENSION_WRITTEN_ONLY = 118
SUSPENSION_PROMISE_IN_FORCE = 119
SUSPENSION_NO_CONTACT_POINT = 120

# Every treatment a suspension of this code blocks outright when it fires (an empty tuple
# blocks nothing outright -- it only restricts scope, handled by the caller).
SUSPENSION_BLOCKS_ALL = {
    SUSPENSION_DEBT_REVIEW_COURT_ORDER, SUSPENSION_ADMINISTRATION_ORDER, SUSPENSION_INSOLVENCY,
    SUSPENSION_DECEASED, SUSPENSION_LITIGATION, SUSPENSION_PRESCRIPTION,
}

# §5.4: "Nothing in [§5.2] is overlayable" and "an overlay may never ... relax a regulatory
# suspension, shorten or waive a prescribed notice period, raise or disapply a contact
# frequency cap or a permitted-contact-hours rule, or alter prescription handling." These
# target names may never appear as an `Adjustment.target` in any register this project
# builds -- see `matrix.assert_no_statutory_target` / `scoring.assert_no_statutory_target`.
STATUTORY_TARGETS = frozenset({
    "suspension_evaluation", "notice_period_days", "contact_frequency_cap",
    "permitted_contact_hours", "prescription_date", "prescription_handling",
})

# --- Reset events (§5.5) -------------------------------------------------------------------

RESET_QUALIFYING_PAYMENT = "qualifying_payment"
RESET_PROMISE_CAPTURED = "promise_captured"
RESET_CURE = "cure"
RESET_DISPUTE = "dispute"
RESET_ARRANGEMENT_ACTIVATED = "arrangement_activated"
