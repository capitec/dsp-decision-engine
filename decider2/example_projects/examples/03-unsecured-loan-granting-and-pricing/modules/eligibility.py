"""Stage 5.1 -- intake and the 14 eligibility gates.

The shape problem.  Doc 03 offers `|` (sequence) and `Branch` (short-circuit,
only the taken arm executes).  Neither expresses what §5.1 requires:

    evaluate all 14 gates, none of them short-circuiting each other,
    produce a 14-wide vector of TRIVALENT verdicts, and only then let the
    FLOW short-circuit.

A `Branch` chain loses the verdicts of every gate after the first failure --
which is precisely the complaint §5.1 records: "a decline letter naming one
reason when four applied has been the subject of a complaint".  A `|` chain of
14 `is_eligible -> is_eligible` modules keeps evaluating but gives you one
boolean and a version chain, not 14 verdicts with the value each tested.

So: `GateSet`.  It is tabular (N gates x fixed attributes), which by doc 08
§3.4's own test -- "can one compiled loop evaluate every instance of this kind,
with the instance supplied as arrays?" -- makes it a **generic kernel**, so a
gate is added, retired or re-worded as an interior document with no recompile
and no release.  See FRAMEWORK-DEMANDS #3.
"""

from __future__ import annotations

from decider2 import GateSet, gate
from decider2.missing import Maybe
from decider2.money import Money
from pipelines.reasons import ABSENT_NOT_YET_RETRIEVED

# ---------------------------------------------------------------------------
# A gate is a pure predicate plus the value it tested.  Returning the tested
# value is not a convenience -- §5.1 requires "the input value each tested" in
# the record, and a gate that returns only a bool cannot supply it.
# ---------------------------------------------------------------------------


@gate(id=1110, tests="applicant_age_years")
def minimum_age(applicant_age_years: float) -> bool:
    """The applicant is at least 18.0 years old at `decision_date`."""
    pass  # applicant_age_years >= 18.0


@gate(id=1111, tests="age_at_maturity")
def maximum_age_at_maturity(applicant_age_years: float, requested_term_months: Maybe[int]) -> bool:
    """Age at `decision_date` plus the term in years does not exceed 75.0."""
    pass  # requested term may be absent -- then the gate tests the longest permitted term


@gate(id=1130, tests="employment_type_code")
def employment_type_permitted(employment_type_code: int) -> bool:
    """Social grant and informal employment are not written on Flex Loan."""
    pass  # employment_type_code not in (5, 6)


@gate(id=1131, tests="requested_amount", applies_when="employment_type_code == 4")
def pensioner_amount_ceiling(requested_amount: Maybe[Money], employment_type_code: int) -> bool:
    """A pensioner is permitted to R80 000 only."""
    pass  # NOT a cap -- a gate, because §5.1 lists it as a gate with its own reason code


@gate(id=1142, tests="insolvency_status_code")
def insolvency(insolvency_status_code: int, months_since_rehabilitation: Maybe[float]) -> bool:
    """Sequestrated declines; rehabilitated is permitted after 24 months' seasoning."""
    pass


@gate(id=1170, tests="duplicate_application_count", needs="in_flight_applications")
def duplicate_detection(
    duplicate_application_count: int,
) -> bool:
    """No identical application -- same client, amount within R500, same day -- concluded
    in the last 24 hours."""
    pass  # duplicate_application_count == 0


# ... eight further gates: product availability by channel (1102), capacity to
# contract (1112), residency (1120), debt review (1140), administration order
# (1141), deceased/estate (1150), exclusion lists (1160-1164), in-flight
# detection (1171).  Each is six lines in this file and one row in the
# generated review artefact.

# ---------------------------------------------------------------------------
# The set.
# ---------------------------------------------------------------------------

Gates = GateSet(
    name="eligibility",
    gates=[
        minimum_age,
        maximum_age_at_maturity,
        employment_type_permitted,
        pensioner_amount_ceiling,
        insolvency,
        duplicate_detection,
        # ... 8 more
    ],
    capacity=16,  # register sized for growth; required, same reason max_iterations is
    # ------------------------------------------------------------------
    # The three-state rule, declared rather than coded.
    #
    # A gate whose declared `tests`/`needs` inputs are all present is
    # EVALUATED (-> PASS or FAIL).  A gate whose inputs are not available at
    # this point in the flow is NOT_EVALUATED, carrying the absence reason of
    # the input that was missing.  The framework derives this from the gate's
    # signature; the author never writes the third branch, which is what stops
    # it collapsing into pass or fail the first time somebody is in a hurry.
    # ------------------------------------------------------------------
    unevaluable=ABSENT_NOT_YET_RETRIEVED,
    verdicts="gate_verdicts",          # int8[16]: PASS / FAIL / NOT_EVALUATED
    tested_values="gate_tested_values",  # f8[16], the value each gate looked at
    writes=["is_eligible", "decline_reason_codes", "primary_reason_code"],
    rank_by=table("reason_code_registry").severity,  # core.reason_codes owns the order
    interior="config/flex_loan/interiors/eligibility_gates.json",
)
