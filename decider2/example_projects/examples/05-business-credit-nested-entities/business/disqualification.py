"""Stage 5.3 -- the twenty business-level absolute rules of disqualification.

The requirement that shapes this file is s5.3's: *every* rule is evaluated even
after the first fails. "A business that fails four of these must be told all
four, because fixing one and reapplying to be declined on the next is the
failure mode this requirement exists to prevent."

That is `collect="all"` on a `verdict(...)`. It is not a `Branch` and it is not
a waterfall, because both of those short-circuit. A waterfall of twenty
`outcome -> outcome` modules would be the doc 03 s3.2 idiom and would produce
exactly one reason code.

`verdict(...)` is a rule-set kind this project needs and doc 03 does not have --
see FRAMEWORK-DEMANDS D11. Its interior is a document (doc 08 s3), so Policy can
add B-AROD-21 without an engineer *provided its predicate already exists as a
registered step*. Adding a genuinely new test is a code change, which is doc 08
s3.2 working as intended and is stated plainly in the README.
"""

from decider2 import module, step, param, verdict, fires, table, witness
from grains import Application, CLEAR, REFER, DECLINE, DISPOSITION_ORDER


# --------------------------------------------------------------------------
# The predicates. One step per rule, named for the rule. A policy analyst
# reading the generated artefact sees `B-AROD-05`, its description, the params
# it reads with their bounds, and nothing else -- doc 04 s6's requirement, and
# the reason the rule id is in the function name rather than in a comment.
# --------------------------------------------------------------------------

def b_arod_01_fires(registration_status_code: int) -> bool:
    """Deregistered or in final liquidation."""
    pass


def b_arod_02_fires(
    registration_status_code: int,
    annual_returns_outstanding_months: float,
    outstanding_returns_limit: float = param(24.0, ge=0.0, le=120.0),
) -> bool:
    """In deregistration, or annual returns outstanding beyond the limit."""
    pass


def b_arod_05_fires(
    months_trading: float,
    minimum_months_trading: float = param(12.0, ge=0.0, le=60.0,
                                          description="s5.3 B-AROD-05 minimum trading history"),
) -> bool:
    """Fewer than the minimum months trading at decision_date."""
    pass


def b_arod_06_fires(
    months_trading: float,
    minimum_months_trading: float = param(12.0, ge=0.0, le=60.0),
    conditional_band_months: float = param(24.0, ge=0.0, le=60.0),
) -> bool:
    """Trading 12..23 months -- permitted only at grade <= 6 with 100% surety cover."""
    pass


def b_arod_07_fires(sector_exclusion_flag: bool) -> bool:
    """Sector carries the exclusion flag (31 of 420 codes)."""
    pass


def b_arod_11_fires(
    trailing_turnover: float,
    has_vat_registration: bool,
    vat_registration_threshold: float = param(1_000_000.0, ge=0.0),
) -> bool:
    """Turnover at or above the VAT threshold with no VAT registration."""
    pass


def b_arod_18_fires(
    trailing_turnover: float,
    product_minimum_turnover: float,      # from the product table, not a param:
                                          # R1 000 000 (product 50) / R500 000 (51)
) -> bool:
    """Trailing 12-month turnover below the product minimum."""
    pass


# ... fourteen more, one per rule, in this file. Elided for the sketch.


# --------------------------------------------------------------------------
# The verdict. `collect="all"` is the whole point; `resolve` picks the outcome
# that governs the flow; `fired` is the complete set and is what s5.3's "Emits"
# clause requires.
#
# Note that the reason codes live on the rule declarations and are checked
# against the registry version in force at decision_date (FV-07) at *build*
# time, not at run time -- a rule referencing reason 5199 in a registry that
# stops at 5121 fails the build rather than an application.
# --------------------------------------------------------------------------
BusinessDisqualification = verdict(
    name="business_disqualification",
    of=Application,
    writes="business_arod_verdict",
    resolve=DISPOSITION_ORDER,           # decline > refer > clear
    collect="all",
    rules=[
        fires(b_arod_01_fires, gives=DECLINE, reason=5101, curable=False),
        fires(b_arod_02_fires, gives=DECLINE, reason=5102, curable=True),
        fires(b_arod_05_fires, gives=DECLINE, reason=5105, curable=False),
        fires(b_arod_06_fires, gives=REFER,   reason=5106,
              unless="grade_6_with_full_surety_cover"),
        fires(b_arod_07_fires, gives=DECLINE, reason=5107, curable=False),
        fires(b_arod_11_fires, gives=DECLINE, reason=5111, curable=True),
        fires(b_arod_18_fires, gives=DECLINE, reason=5119, curable=False),
        # ... thirteen more
    ],
    interior="config/business_facility/rules/business_disqualification.json",
    contract="contracts/business_disqualification.json",
)


# --------------------------------------------------------------------------
# s5.3's parenthesis is a real requirement and is easy to lose: "the rule family
# is referred to as fourteen *classes* of disqualification in policy documents,
# and the mapping between policy classes and executable rules is itself
# something the implementation must keep visible. Policy writes classes; the
# flow evaluates rules."
#
# So the mapping is a table, and the generated review artefact groups by class.
# It is not a comment and it is not a naming convention.
# --------------------------------------------------------------------------
POLICY_CLASS_MAP = table(
    "business_disqualification_policy_class",
    key="rule_id",
    columns=("policy_class_code", "policy_class_name", "policy_document_ref"),
    source="tables/business_policy_classes.csv",
    effective_dated=True,
)
