"""Names declared locally by this project (spec 07 §4.3) plus the small
codes the flow itself needs. `decision_date`, `product_code`, `client_id`,
`gross_monthly_income`, `net_monthly_income`, `living_expenses`,
`existing_obligations`, `discretionary_income`, `max_affordable_instalment`,
`affordability_verdict_code`, `total_exposure`, `probability_of_default`,
`decline_reason_codes`, `primary_reason_code`, etc. are the library's
(project 00's) and are used unchanged -- not redefined here.
"""
from __future__ import annotations

from enum import IntEnum

PRODUCT_EVERYDAY_CARD = 20
PRODUCT_ACCESS_FACILITY = 21
PRODUCTS = (PRODUCT_EVERYDAY_CARD, PRODUCT_ACCESS_FACILITY)

# --- Hard exclusions (§5.2). Working-depth subset per SCOPE.md: 8 of 16 codes,
# every one still attributed (all that applied, not merely the first). X16 is
# fed by project 08's treatment-state feed (DEPS.md "soft": stub it) --
# 07 declares the field it expects (`treatment_suspension_active`) and reads
# it, so a real 08 feed slots in unchanged (see DEPS.md in this directory).


class Exclusion(IntEnum):
    X01_ARREARS_NOW = 1
    X02_ARREARS_6M = 2
    X03_DEBT_REVIEW = 3
    X05_DECEASED = 5
    X06_FRAUD = 6
    X09_COOLING_OFF = 9
    X12_NO_CONSENT = 12
    X14_TOO_YOUNG = 14
    X16_TREATMENT_SUSPENSION = 16  # project 08's feed (stubbed)


# --- Caps (§5.5), lowest wins, tie broken by this table's own order. -------
class Cap(IntEnum):
    C1_PRODUCT_MAX = 1
    C2_INCOME_MULTIPLE = 2
    C3_TOTAL_EXPOSURE = 3
    C5_OBSERVED_SPEND = 5
    C6_MATRIX_MAX_INCREASE = 6
    C7_AFFORDABILITY = 7


# --- Affordability evidence tiers (§5.6). -----------------------------------
class EvidenceTier(IntEnum):
    A_VERIFIED_SALARY = 1
    B_IRREGULAR_DEPOSITS = 2
    C_DECLARED_REFRESHED = 3
    D_DECLARED_STALE = 4
    E_NO_EVIDENCE = 5


# --- The automatic / conditional / fail verdict (§5.6). ---------------------
class IncreasePath(IntEnum):
    AUTOMATIC = 1
    CONDITIONAL = 2
    FAIL = 3


# --- allocation_outcome_code (§4.3, §5.8). ----------------------------------
class AllocationOutcome(IntEnum):
    FUNDED = 1
    BELOW_LINE = 2
    FAIRNESS_CAPPED = 3
    TAIL_SKIPPED = 4
    OVERLAY_SUPPRESSED = 5
    NOT_RANKED = 6


# --- change_type_code / notice_class_code (§5.9, §6.7). ---------------------
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


# --- binding_cap_code when nothing binds (proposed_limit == 0 pre-cap). ----
NO_CAP_BINDING = 0
