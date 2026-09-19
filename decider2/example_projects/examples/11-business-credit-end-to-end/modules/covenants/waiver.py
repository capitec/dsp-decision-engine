"""Waivers. Spec 5.5.5, six requirements, each load-bearing.

A waiver is an adjustment. Not metaphorically -- it has every property doc 00
6.22 requires of one: it is an overlay rather than an edit, the unwaived result
survives beside it, it carries identity, approval and rationale, its scope is
declared, and its EXPIRY IS MANDATORY.

Spec 5.5.5 requirement 2 says so directly: "Exactly as an adjustment does
(00 6.22 property 5). The failure mode is identical and just as common: a waiver
granted for one bad quarter in 2027, still silently in force in 2031, with
nobody able to say what testing the covenant again would show."

So this module consumes `core.adjustments` rather than reimplementing expiry,
scope and stacking -- which is a reuse win nobody would have found by looking at
the component's name. It is recorded in consumed/__init__.py's count.

What is NOT an adjustment and is added here: a waiver names a COVENANT INSTANCE
at a DEFINITION VERSION for a TEST DATE OR PERIOD. "Covenants waived" is not a
waiver. `core.adjustments` scopes by population; this scopes by instance.
"""

from decider2 import module, step, param, adjustment_shaped
from consumed.core_library import adjustments
from time.dating import CONTRACT

Waiver = adjustment_shaped(
    "waiver",
    base=adjustments,
    scope_by="covenant_instance_id",          # not by population
    requires=[
        "covenant_instance_id",
        "covenant_definition_version",        # req 1: names WHAT it waives
        "test_date_or_period",
        "expiry",                             # req 2: mandatory. No expiry, no record.
        "authority_level_code",               # req 4: never below level 3
        "conditions",                         # req 5: themselves tracked
    ],
    recorded_against="covenant",              # req 3: NOT against the facility
    on_expiry="retest",                       # req 6: the system does not wait to be asked
)


def waiver_authority(breach_class: int, risk_grade: int, exposure_cents: int,
                     facility_type: int) -> int:
    """432-cell matrix, never below level 3. Spec 5.5.5 requirement 4.

    "No waiver is automatic, at any amount, at any grade. This is the one place
     in the flow where the automated mandate is explicitly unavailable."

    Expressed as a FLOOR on the matrix rather than as a cell value, so that a
    future edit to the 432 cells cannot lower it by accident:
        level = max(3, matrix[breach_class, grade, facility_type])
    The floor is a parameter owned by Credit Governance with `ge=3`, which means
    a policy owner physically cannot type 2.
    """
    pass  # max(floor, matrix lookup)


def repeat_waiver_limits(waiver_history: list,
                         consecutive_on_one: int = param(2, ge=1, le=3),
                         on_facility_in_24m: int = param(3, ge=1, le=6)) -> int:
    """The mechanism that stops a slow restructure happening one waiver at a
    time, unreported. Spec 5.5.5's last three rows, and the only part of this
    file that is genuinely new logic rather than consumed machinery.

      two consecutive on one covenant -> no third. Reset via L4 or go to L5. W2.
      three on any covenants of one facility in 24 months -> MANDATORY L5
        forbearance assessment, because a pattern of concessions granted to a
        business in difficulty IS forbearance regardless of what each was called.
        Authority +1.
      any waiver where the same covenant was cured in the preceding four test
        periods -> committee referral; the structure is presumed unsustainable.

    Spec 10 acceptance 10: "a third waiver on one covenant is structurally
    impossible without an amendment or a restructure." Structurally, not by
    validation -- the waiver-granting path has no arm that reaches a third, and
    the arm it does reach routes to L4/L5. `branch_path` records which.
    """
    pass  # count; route to the reset / forbearance arms


def aggregate_pattern_test(concession_history: list, window_months: int = 24) -> bool:
    """Whether a SEQUENCE of individually-innocuous concessions constituted
    forbearance in aggregate. Spec 9.6's last row.

    Three waivers, a covenant reset and a term extension across two years. Each
    was within authority. Together they are a restructure of a business in
    difficulty that was never reported as one, and the regulator's sample test
    is aimed precisely at it. Run at every waiver, every reset and every
    amendment -- not at the point somebody asks.
    """
    pass  # walk the concession history; apply the aggregate limbs of 5.8.5


Waivers = module(waiver_authority, repeat_waiver_limits, aggregate_pattern_test,
                 name="waivers", owner="credit_governance",
                 co_owners=["business_credit_risk_policy", "provisioning"])
