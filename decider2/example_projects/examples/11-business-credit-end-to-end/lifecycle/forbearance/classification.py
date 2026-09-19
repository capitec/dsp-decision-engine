"""L5 -- forbearance classification, including the NEGATIVE case. Spec 5.8.5, 9.6.

    "The hardest of the five, because it tests a NEGATIVE."

Gap `forbearance_classification`, resolution COMPOSE. Project 06 supplies the
search, the NPV cost, the authority ladder, the stressed affordability test and
the before-and-after comparison, all consumed unchanged. What it does not supply
is a classification of record, because the test is regulatory-reporting shaped
rather than restructure shaped, and its three owners (Provisioning owns the
rules, Policy owns the thresholds, Recoveries grants the concession) do not
overlap with project 06's owner.

---------------------------------------------------------------------------
The two failure modes this file forecloses
---------------------------------------------------------------------------
  1. A classification derived LATER by Finance from a payment pattern. That is
     not a decision and cannot be evidenced.
  2. An empty record on the negative cases, which is indistinguishable from a
     classification that was never made.

The second is why `not_forbearance` is a step that PRODUCES EVIDENCE, and why
its output is required rather than optional. A restructure classified not
forbearance with an empty evidence field does not validate.
"""

from decider2 import module, step, param, required_output
from consumed.p06_restructure import ConcessionSearch, StressedAffordability


@step(description="Limb 1 -- financial difficulty, with the indicators named")
def financial_difficulty(arrears_days: int, covenant_breaches: list,
                         watchlist_grade: int, grade_movement: int,
                         declared_hardship: bool, own_forecast: dict) -> dict:
    """Spec 5.8.5 limb 1. Evidenced by: arrears, a covenant breach, watchlist
    grade, a grade deterioration, a declared hardship, or the business's own
    forecast -- WITH THEIR VALUES AND SOURCES AS AT THE DECISION DATE (spec 9.6).

    Returns the indicator set, not a boolean. A boolean is what gets recorded
    when a regulator asks for evidence and finds `forbearance_limb1 = true`.
    """
    pass  # evaluate each indicator; return {indicator: (value, source, as_at)}


@step(description="Limb 2 -- a concession the Bank would not otherwise have granted")
def concession(terms_before: dict, terms_after: dict, counterfactual: dict) -> dict:
    """Limb 2. A modification of terms the client could not comply with, which
    the Bank would not have granted otherwise; or a refinancing of a troubled
    facility.

    `counterfactual` is the hard part and it is why this is composed rather than
    inferred: "would the Bank have granted this otherwise" is answered by
    running the SAME structuring search with the difficulty indicators removed
    and comparing. That is a second call to project 06's bounded search per
    restructure -- 120/day, 20s p95 budget, so it doubles the most expensive
    entry point's work. Stated rather than hidden; it is the price of an
    evidenced negative.
    """
    pass  # diff the terms; run the counterfactual search; record both


@required_output
def classification(financial_difficulty: dict, concession: dict) -> int:
    """BOTH LIMBS, or it is not forbearance. Produced at decision time.

    Spec 5.8.6: it persists through the probation clocks, is visible to the next
    review, the next amendment and the next watchlist pass, and it RAISES THE
    AUTHORITY for any subsequent concession. So it is a state flag written here
    and read at O1 (modules/request/relationship.py), not a report field.
    """
    pass  # both limbs met -> FORBORNE; else NOT_FORBORNE


@required_output
def not_forbearance_evidence(financial_difficulty: dict, concession: dict,
                             classification: int) -> dict:
    """Spec 5.8.5's missed part, and spec 9.6's fourth row.

        "A restructure that is NOT forbearance must carry the evidence for the
         NEGATIVE classification. 'We assessed it and it was not forbearance' is
         exactly what a regulator tests, and an empty record is
         indistinguishable from a classification that was never made."

    `@required_output` means the framework refuses to emit a decision of record
    for a restructure without it. Not a validator on a nullable field -- a
    missing required output is a build-time interface error for any pipeline
    that reaches L5 without producing it, which is the only strength that
    survives fourteen teams and forty builds.
    """
    pass  # record both limbs as assessed-and-not-met, with their values


def staging_consequence(classification: int, performing: bool,
                        npv_loss_pct: float,
                        diminished_materiality_pct: float = param(1.0, ge=0, le=5,
                            description="NPV loss above which the obligation is diminished")) -> int:
    """Stage 1/2/3, produced BY THE FLOW at the point of decision and handed to
    Provisioning rather than derived by them. Spec 5.8.5.

      concession, performing, no loss        -> forborne; Stage 2 minimum, whole probation
      concession, already non-performing     -> non-performing forborne; Stage 3
      NPV loss above 1%                      -> default event; Stage 3; reported as such
      second concession during probation     -> re-trigger; restart the clocks
      any exposure >30 days past due in the
        performing-forborne probation        -> same

    The 1% is a STATUTORY parameter (spec 6.4): Compliance transcribing a
    published instrument, and Business Credit Risk Policy may not change it.
    That ownership class is enforced by the params document's CODEOWNERS split,
    not by the framework -- doc 04 2.1 is honest that the framework makes the
    line visible and policy attaches to it, and this is that working.
    """
    pass  # the four-row table


def probation_clocks(state: dict) -> dict:
    """Non-performing forborne -> performing forborne: 1 year.
       Performing forborne -> no longer forborne: 2 years.

    Change scenario 11: the performing-forborne probation moves from two years
    to three. Every live clock is affected; CLOCKS ALREADY EXPIRED MUST NOT
    RESTART; and the reporting for prior periods must remain as reported.

    That is the contractual-vs-policy distinction again, in a third shape: the
    RULE is policy (resolves by decision_date), the CLOCK is an instance fact
    with a start date, and an expired clock is a closed fact that a rule change
    cannot reopen. Expressed as: a clock is bound at grant, the duration is
    resolved at grant and STORED ON THE CLOCK, and the policy change applies to
    clocks started after its effective date. One line to say; three defects if
    the duration is looked up at evaluation time instead.
    """
    pass  # advance clocks; never restart an expired one


Forbearance = module(financial_difficulty, concession, classification,
                     not_forbearance_evidence, staging_consequence, probation_clocks,
                     name="forbearance",
                     owner="provisioning",
                     co_owners=["business_credit_risk_policy", "business_recoveries"],
                     consumes=[ConcessionSearch, StressedAffordability],
                     taps=["classification", "staging_code", "branch_path"])
