"""The shared vocabulary: codes, enums and name reservations every consumer agrees on."""
from __future__ import annotations

from enum import IntEnum
from typing import NamedTuple


class NullKind(IntEnum):
    """Why a value is missing. Consumers must not collapse these."""

    NOT_COLLECTED = 0
    COLLECTED_ZERO = 1
    INDETERMINATE = 2
    NOT_APPLICABLE = 3  # the field has no meaning for this role


# Whose value it is. The field keeps one name; the role says whose it is.
APPLICANT = "applicant"
CO_APPLICANT = "co_applicant"
SURETY = "surety"
DIRECTOR = "director"


def entity_role(n: int) -> str:
    """The role of the nth nested business entity: `entity_role(2) == "entity_2"`."""
    return f"entity_{n}"


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

# `segment_code` keys calibration, risk grades and appetite.
SEGMENT_RETAIL_MASS = 1
SEGMENT_RETAIL_AFFLUENT = 2
SEGMENT_NEW_TO_BANK = 3
SEGMENT_SME_PEOPLE = 4
SEGMENT_SME_ENTITY = 5

# Names reserved for consumers; the library computes none of them, so projects don't each invent their own:
#   account_id int64, matrix_cell_id str, assessment_mode_code int8, authority_level_code int8,
#   event_id int64, event_type_code int16.


class ConsentChannel(IntEnum):
    SMS = 1
    PUSH = 2
    VOICE = 3
    EMAIL = 4


class ConsentVerdict(IntEnum):
    PERMITTED = 1
    SUPPRESSED = 2
    REGULATED_NOTICE = 3  # never suppressible by a marketing preference


class FraudVerdictCode(IntEnum):
    APPROVE = 1
    REFER = 2
    DECLINE = 3
    UNAVAILABLE = 4  # an outage is not a pass


class FraudVerdict(NamedTuple):
    fraud_verdict_code: FraudVerdictCode
    fraud_reason_codes: tuple[int, ...]
    fraud_response_ms: float
