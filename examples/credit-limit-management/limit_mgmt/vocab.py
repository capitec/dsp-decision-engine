"""Product codes and the integer codes the per-account pipeline and the allocation write."""
from __future__ import annotations

from enum import IntEnum

PRODUCT_EVERYDAY_CARD = 20
PRODUCT_ACCESS_FACILITY = 21


class Exclusion(IntEnum):
    X01_ARREARS_NOW = 1
    X02_ARREARS_6M = 2
    X03_DEBT_REVIEW = 3
    X05_DECEASED = 5
    X06_FRAUD = 6
    X09_COOLING_OFF = 9
    X12_NO_CONSENT = 12
    X14_TOO_YOUNG = 14
    X16_TREATMENT_SUSPENSION = 16


class Cap(IntEnum):
    """`binding_cap_code`; 0 means no cap bound. C4 (group exposure) is not modelled."""

    C1_PRODUCT_MAX = 1
    C2_INCOME_MULTIPLE = 2
    C3_TOTAL_EXPOSURE = 3
    C5_OBSERVED_SPEND = 5
    C6_MATRIX_MAX_INCREASE = 6
    C7_AFFORDABILITY = 7


NO_CAP_BINDING = 0


class EvidenceTier(IntEnum):
    A_VERIFIED_SALARY = 1
    B_IRREGULAR_DEPOSITS = 2
    C_DECLARED_REFRESHED = 3
    D_DECLARED_STALE = 4
    E_NO_EVIDENCE = 5


class IncreasePath(IntEnum):
    AUTOMATIC = 1
    CONDITIONAL = 2
    FAIL = 3


class ChangeType(IntEnum):
    INCREASE_AUTOMATIC = 1
    INCREASE_CONDITIONAL = 2
    DECREASE = 3
    NO_CHANGE = 4


class NoticeClass(IntEnum):
    IMMEDIATE = 1
    PRESCRIBED_NOTICE = 2
    CONSENT_REQUIRED = 3
    NOT_APPLICABLE = 4


class AllocationOutcome(IntEnum):
    FUNDED = 1
    BELOW_LINE = 2
    FAIRNESS_CAPPED = 3
    TAIL_SKIPPED = 4
    SUPPRESSED = 5
    NOT_RANKED = 6
