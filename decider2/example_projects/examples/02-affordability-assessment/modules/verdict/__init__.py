"""Stage 7 -- the verdict, and the three shapes of answer.

Spec question 12: how is `indeterminate` kept distinct from `fail` through five
consumers, four modes and every intermediate, when the cheapest implementation
of every one of them is a boolean?

The answer is that the cheap implementation is not available. `Verdict` is a
declared DOMAIN, not an int8 with a comment. Three things follow and all three
are checked rather than documented:

  1. A `Verdict` may not be used in a boolean context. `if verdict:` is a lint
     error and a codegen error; `if verdict == PASS:` is fine. The failure mode
     it closes is `if not verdict.passed` silently declining every
     indeterminate assessment, which is the most consequential error available
     in this project.

  2. A `Verdict` may not be compared with `!=` to a single member. `verdict !=
     PASS` is the boolean in disguise -- it folds three outcomes into one -- so
     the domain declares no `__ne__` and the error names the three members the
     author has just lumped together.

  3. Narrowing a `Verdict` to anything smaller must go through `narrow()`,
     which has a REQUIRED `treat_indeterminate_as=` argument with no default.
     Every one of the five consumers has had to answer the question in writing,
     in its own file, and the answers differ.

None of this is free. It costs a domain type, a lint, and five call sites that
cannot be written by autocomplete. It is cheaper than the alternative, which is
a 2029 finding that one consumer declined 40 000 people the Bank could not
assess. FRAMEWORK-DEMANDS #9.
"""

from decider2 import module, narrow, policy, rung, step
from decider2.domains import Domain, Switch


class Verdict(Domain, dtype="int8", boolean_context=False, closed=True):
    PASS = 1
    MARGINAL = 2
    FAIL = 3
    INDETERMINATE = 4


class ToleranceBasis(Switch):
    APPETITE = 1           # new application, limit increase, scenario
    SUSTAINABILITY = 2     # arrangement: a different tolerance and a larger floor


class Sufficiency(Domain, dtype="int8", closed=True):
    """Why an assessment is indeterminate. Zero when it is not."""
    SUFFICIENT = 0
    INCOME_BELOW_MINIMUM_TIER = 1
    BUREAU_VIEW_STALE = 2
    UNTREATABLE_ACCOUNT_TYPE = 3
    STATEMENT_CONFIDENCE_BELOW_THRESHOLD = 4
    JOINT_APPLICANT_A_UNESTABLISHED = 5
    JOINT_APPLICANT_B_UNESTABLISHED = 6
    CASH_INCOME_NO_BANKING_FOOTPRINT = 7
    NON_DECLARATION_OF_EXPENSES = 8


@step(
    output="evidence_sufficiency_code",
    description=(
        "The single reason this assessment could not conclude. Where several "
        "apply, the most remediable is reported, because the client-facing "
        "communication is 'send us X' and there can only be one X."
    ),
)
def evidence_sufficiency_code(
    income_sufficiency: Sufficiency,
    framing_sufficiency: Sufficiency,
    bureau_is_stale: bool,
    refer_account_count: int,
    non_declaration_flag: bool,
) -> Sufficiency:
    pass  # a declared severity order over Sufficiency; never a bool, never an OR


@rung(
    section="verdict",
    order=200,
    says=(
        "Verdict: {affordability_verdict_code:verdict}. The proposed instalment "
        "of {proposed_instalment_cents:money} against a maximum affordable "
        "instalment of {max_affordable_instalment_cents:money} leaves "
        "{discretionary_income_after_cents:money} of discretionary income."
        "{indeterminate_note}"
    ),
)
@step(
    output="affordability_verdict_code",
    description=(
        "Pass at or below the maximum; marginal within a tolerance band above "
        "it; fail above the band; indeterminate where the evidence does not "
        "support any of the three. Indeterminate is not a failure of "
        "affordability."
    ),
)
def affordability_verdict_code(
    proposed_instalment_cents: int | None,
    max_affordable_instalment_cents: int,
    evidence_sufficiency_code: Sufficiency,
    tolerance_pct: float = policy(0.05, ge=0.0, le=0.25),
    tolerance_basis: ToleranceBasis = policy(ToleranceBasis.APPETITE),
) -> Verdict:
    pass  # INDETERMINATE dominates. With no proposed instalment the result is PASS
          # or INDETERMINATE only -- shape (b) -- and MARGINAL is unreachable, which
          # is asserted rather than assumed.


@step(
    output="discretionary_income_after_cents",
    description="Discretionary income once the proposed instalment is committed. The caller's margin.",
)
def discretionary_income_after_cents(
    discretionary_income_cents: int,
    proposed_instalment_cents: int | None,
) -> int | None:
    pass  # None where no instalment was proposed; NOT zero, and not the unreduced figure


Verdicts = module(
    evidence_sufficiency_code,
    affordability_verdict_code,
    discretionary_income_after_cents,
    name="verdict",
    contract="contracts/verdict.json",
    taps=["affordability_verdict_code", "evidence_sufficiency_code", "branch_path"],
)


# --------------------------------------------------------------------------
# The five consumers' narrowings. These live here, beside the domain, rather
# than in each consumer, so that the set of answers is READABLE IN ONE PLACE --
# which is what makes "five consumers handle indeterminate differently" an
# inspectable fact rather than five buried conventions.
#
# Each is a declared artefact with an audit identity. Compliance reviews this
# list; it is four lines and it is the whole of spec question 12's surface.
# --------------------------------------------------------------------------

to_granting_decision = narrow(                       # project 03
    "affordability_verdict_code",
    to="outcome_code",
    mapping={Verdict.PASS: "APPROVE", Verdict.MARGINAL: "REFER", Verdict.FAIL: "DECLINE"},
    treat_indeterminate_as="REFER",
    reason_from="evidence_sufficiency_code",
)

to_limit_decision = narrow(                          # project 07
    "affordability_verdict_code",
    to="limit_action_code",
    mapping={Verdict.PASS: "INCREASE", Verdict.MARGINAL: "HOLD", Verdict.FAIL: "HOLD"},
    treat_indeterminate_as="CONDITIONAL",             # the increase, subject to confirmation
    reason_from="evidence_sufficiency_code",
)

to_scenario_filter = narrow(                         # project 06
    "affordability_verdict_code",
    to="scenario_admissible",
    mapping={Verdict.PASS: True, Verdict.MARGINAL: False, Verdict.FAIL: False},
    treat_indeterminate_as="ABORT_SEARCH",            # not False: an indeterminate scenario
    reason_from="evidence_sufficiency_code",          # means the whole search is unsound
)

to_arrangement_decision = narrow(                    # project 08
    "affordability_verdict_code",
    to="arrangement_sustainable",
    mapping={Verdict.PASS: True, Verdict.MARGINAL: True, Verdict.FAIL: False},
    treat_indeterminate_as="MANUAL_REVIEW",
    reason_from="evidence_sufficiency_code",
)

to_business_surety = narrow(                         # project 05
    "affordability_verdict_code",
    to="surety_capacity_established",
    mapping={Verdict.PASS: True, Verdict.MARGINAL: True, Verdict.FAIL: False},
    treat_indeterminate_as=False,                     # the only consumer for which the
    reason_from="evidence_sufficiency_code",          # conservative answer IS False, and it
)                                                     # had to write that down to get it
