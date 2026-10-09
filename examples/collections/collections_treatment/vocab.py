"""Collections codes that every consumer shares."""
from __future__ import annotations

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

TREATMENT_NAMES = {
    NO_ACTION: "No action", AUTOMATED_SMS: "Automated SMS", AUTOMATED_EMAIL: "Automated email",
    IN_APP_MESSAGE: "In-app message", AUTOMATED_VOICE_MESSAGE: "Automated voice message",
    AGENT_CALL_LOW: "Agent call, low intensity", AGENT_CALL_STANDARD: "Agent call, standard",
    AGENT_CALL_HIGH: "Agent call, high intensity", FIELD_VISIT: "Field visit", AGENCY_HANDOVER: "Agency handover",
    PRE_LEGAL_NOTICE: "Pre-legal notice", LEGAL_HANDOVER: "Legal handover", SETTLEMENT_OFFER: "Settlement offer",
    WRITE_OFF_RECOMMENDATION: "Write-off recommendation",
}

ALL_TREATMENTS = tuple(TREATMENT_NAMES)

# The capacity pool a treatment draws on (`pool_code`). Settlement offers and write-off recommendations draw on none.
POOL_SMS, POOL_EMAIL, POOL_IN_APP, POOL_VOICE_MESSAGE, POOL_EARLY_AGENTS, POOL_LATE_AGENTS = 1, 2, 3, 4, 5, 6
POOL_FIELD, POOL_AGENCY, POOL_NOTICES, POOL_LEGAL = 7, 8, 9, 10
NO_POOL = 0

TREATMENT_POOL = {
    AUTOMATED_SMS: POOL_SMS, AUTOMATED_EMAIL: POOL_EMAIL, IN_APP_MESSAGE: POOL_IN_APP,
    AUTOMATED_VOICE_MESSAGE: POOL_VOICE_MESSAGE, AGENT_CALL_LOW: POOL_EARLY_AGENTS,
    AGENT_CALL_STANDARD: POOL_EARLY_AGENTS, AGENT_CALL_HIGH: POOL_LATE_AGENTS, FIELD_VISIT: POOL_FIELD,
    AGENCY_HANDOVER: POOL_AGENCY, PRE_LEGAL_NOTICE: POOL_NOTICES, LEGAL_HANDOVER: POOL_LEGAL,
}

# Why an account gets no treatment today (`non_selection_reason_code`). 260 and 270 are not produced.
NS_MATRIX_NO_ACTION = 200
NS_OVERLAY_SUPPRESSED = 205
NS_SUSPENDED = 210
NS_COOLING_OFF = 220
NS_INTERVAL_OR_CAP = 230
NS_PROMISE_IN_FORCE = 240
NS_BELOW_CUTOFF = 250

# The suspension codes. `suspensions.py` says what each blocks.
SUSPENSION_PROMISE_IN_FORCE = 119

# Reset events that restart or hold an escalation path.
RESET_QUALIFYING_PAYMENT = "qualifying_payment"
RESET_PROMISE_CAPTURED = "promise_captured"
RESET_CURE = "cure"
RESET_DISPUTE = "dispute"
RESET_ARRANGEMENT_ACTIVATED = "arrangement_activated"
