"""Canonical vocabulary (spec 00 §4, addendum A2-A7).

Every consuming project imports names from here rather than inventing its
own. This module holds no logic, only the shared identifiers, enums and
small contract types that more than one capability or more than one
consumer needs to agree on.
"""
from __future__ import annotations

from enum import IntEnum
from typing import NamedTuple

# --- Null kinds (spec 00 §7.4, addendum A7 item 7) --------------------------
# Four distinct null situations. A capability that cannot establish a value
# must say *which* of these applies; consumers must not collapse them.


class NullKind(IntEnum):
    NOT_COLLECTED = 0       # no attempt was made to obtain it
    COLLECTED_ZERO = 1      # obtained, and the value is genuinely zero
    INDETERMINATE = 2       # an attempt was made and failed to establish it
    NOT_APPLICABLE = 3      # the field has no meaning for this role (addendum A7)


# --- Role (addendum A7 item 7: "whose value it is") --------------------------
# A value belongs to a role, not only to the application. `applicant`,
# `co_applicant`, `surety`, `director`, or `entity_<n>` for a nested business
# entity (project 05). Capabilities that can be evaluated per-person or
# per-entity accept a `role` and must not rename the underlying field per use.

APPLICANT = "applicant"
CO_APPLICANT = "co_applicant"
SURETY = "surety"
DIRECTOR = "director"


def entity_role(n: int) -> str:
    """The role name for the nth nested entity (project 05)."""
    return f"entity_{n}"


# --- Product catalogue (spec 00 §5) ------------------------------------------

PRODUCT_FLEX_LOAN = 10
PRODUCT_FLEX_LOAN_CONSOLIDATION = 11
PRODUCT_EVERYDAY_CARD = 20
PRODUCT_ACCESS_FACILITY = 21
PRODUCT_DRIVE_FINANCE = 30
PRODUCT_HOME_LOAN_FURTHER_ADVANCE = 40
PRODUCT_BUSINESS_TERM_FACILITY = 50
PRODUCT_BUSINESS_REVOLVING_FACILITY = 51

PRODUCT_NAMES = {
    PRODUCT_FLEX_LOAN: "Flex Loan",
    PRODUCT_FLEX_LOAN_CONSOLIDATION: "Flex Loan Consolidation",
    PRODUCT_EVERYDAY_CARD: "Everyday Card",
    PRODUCT_ACCESS_FACILITY: "Access Facility",
    PRODUCT_DRIVE_FINANCE: "Drive Finance",
    PRODUCT_HOME_LOAN_FURTHER_ADVANCE: "Home Loan Further Advance",
    PRODUCT_BUSINESS_TERM_FACILITY: "Business Term Facility",
    PRODUCT_BUSINESS_REVOLVING_FACILITY: "Business Revolving Facility",
}

# --- Segment (addendum A2: 00 keys calibration/grading/appetite on "segment"
# but never publishes the name) -----------------------------------------------
# `segment_code` is a small, library-owned classification -- not a
# capability, just a name every consumer keys off unchanged.
SEGMENT_RETAIL_MASS = 1
SEGMENT_RETAIL_AFFLUENT = 2
SEGMENT_NEW_TO_BANK = 3
SEGMENT_SME_PEOPLE = 4
SEGMENT_SME_ENTITY = 5


# --- Shared identifiers reserved for consumers (addendum A4) ----------------
# 00 does not compute these, but declares the name and type so that 07 and
# 08, 06 and 11, etc. do not each invent their own and collide.
#
#   account_id            int64   one revolving account or credit agreement
#   matrix_cell_id         str    a cell identifier in any consumer-owned matrix
#   assessment_mode_code   int8   which of a capability's modes was used
#   authority_level_code   int8   the sign-off authority a decision required
#   event_id              int64   one event (payment, adverse, or other)
#   event_type_code        int16  what kind of event `event_id` is


# --- Consent (addendum A3: "the consent vocabulary come[s] from the library
# unchanged", 08 §4.1, but 00 §4 defines none) --------------------------------

class ConsentChannel(IntEnum):
    SMS = 1
    PUSH = 2
    VOICE = 3
    EMAIL = 4


class ConsentVerdict(IntEnum):
    PERMITTED = 1
    SUPPRESSED = 2
    REGULATED_NOTICE = 3  # not suppressible by a marketing preference (07 §5.9)


# --- Fraud verdict contract (addendum A5) -------------------------------------
# 00 does not run fraud logic (out of scope, §12); it publishes the stable
# shape so 03 and 10 stop inventing their own field names for a stub.

class FraudVerdictCode(IntEnum):
    APPROVE = 1
    REFER = 2
    DECLINE = 3
    UNAVAILABLE = 4  # must stay distinct from APPROVE -- an outage is not a pass


class FraudVerdict(NamedTuple):
    fraud_verdict_code: FraudVerdictCode
    fraud_reason_codes: tuple[int, ...]
    fraud_response_ms: float


# --- Overlay identifiers (addendum B1: pick one pair, int32+) ----------------
# 00 uses `adjustment_set_id` (int32) and `adjustments_applied` (list[str]) as
# the one pair every consumer keys on, wide enough for a register versioned
# weekly across the estate.
