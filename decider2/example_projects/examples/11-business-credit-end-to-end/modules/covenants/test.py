"""L2 -- the covenant test. 2.4 M a year. Spec 5.5.

This module is tagged `contractual=True`, which has one mechanical consequence
(time/dating.py rule R2): **it may not read any `Dated[T]`.** Every artefact it
reads resolves by instance binding. If somebody adds a `dated_table` read to
this module -- a new sector benchmark, a current DSCR threshold, anything -- the
build fails naming the step and the table.

That is the entire defence against the failure spec 5.5.1 describes:

    "an effective-dated resolution by decision_date picks the 2029 standard,
     computes a plausible number, and breaches a covenant the client did not
     breach. It will not error. It will not look wrong. It will be wrong."

A tag on a module is a weak thing to hang that on, so it is doubled: the module
declares `contractual=True` AND the artefacts it reads are `Bound[T]`, which
have no date accessor to call. Either alone would be a convention. Together the
wrong call does not typecheck and the wrong import does not build.
"""

from decider2 import module, step, param
from consumed.p05_origination import CovenantScopedSpread
from modules.covenants.definition import CovenantDefinitionVersion
from modules.covenants.schedule import threshold_on_step_schedule, three_dates
from time.dating import covenant_definitions

NOT_TESTED = 4     # neither pass nor breach. See `result` below.


@step(description="Spread only the lines this covenant's definition names")
def covenant_inputs(covenant_definition_version: str, statements: list) -> dict:
    """Fork pressure #3, resolved by scoping rather than by copying.

    O7 spreads 48 lines and computes 11 measures. A DSCR covenant needs two
    lines and one ratio, and it needs them mapped by the mapping-table version
    inside the definition's closure -- not by today's, which has been superseded
    eleven times since 2026.

    `CovenantScopedSpread` is the SAME implementation as O7's, invoked for a
    declared subset with a pinned mapping version. There is no second mapping
    table, and CI fails if one appears under tables/.
    """
    pass  # CovenantScopedSpread(lines=bound.measure_lines, mapping=bound.closure.mapping)


def measured_value(covenant_inputs: dict) -> float:
    """The ratio, on the bound definition's numerator and denominator.

    Spec 5.3 O7 item 3, and it is the sentence people refuse to believe: the
    review's DSCR and the covenant's DSCR are TWO DIFFERENT NUMBERS for the same
    business in the same year, both correct. The review pack shows both without
    implying one is wrong (spec 5.3 O14). So this value is named
    `covenant_measured_value`, never `dscr`, and the vocabulary refuses to map
    it onto the review's name.
    """
    pass  # evaluate the bound measure over the scoped spread


def headroom(measured_value: float, threshold: float) -> tuple:
    """Headroom in BOTH ratio and percentage terms. Spec 7.1 EP-4.

    Recorded on a pass as well as a breach: headroom below 10% at two
    consecutive tests is a weight-7 early-warning contributor (spec 5.6.1), so a
    pass is an input to L3 and not a null event.
    """
    pass  # (measured - threshold, (measured - threshold) / threshold)


def result(measured_value: float, threshold: float, certificate_received: bool) -> int:
    """Pass / breach / NOT TESTED. Three outcomes, not two. Spec 5.5.2.

    The fourth state is the one designs lose: **not tested**, because the
    certificate has not arrived. That is a breach of the INFORMATION
    UNDERTAKING -- a different covenant instance, with its own definition, its
    own cure and its own waiver -- and an unknown result on this one.

    One late certificate therefore produces one certain breach and one unknown
    result. A design that collapses them reports a financial covenant breach the
    client has not committed, or a pass the Bank has not established. Both have
    happened. So `result` returns NOT_TESTED and the information undertaking's
    own test row is emitted by the schedule independently -- they are two rows,
    never one row with two meanings.
    """
    pass  # if not certificate_received: NOT_TESTED; else compare


def breach_class(measured_value: float, threshold: float, template_id: int) -> int:
    """0 none / 1 technical / 2 material / 3 severe. 592-cell matrix.

    Spec 5.5.3's governance shape in miniature: the classification consumes the
    measured value and the threshold, so it is computable -- but the BAND EDGES
    are policy (Business Credit Risk Policy) and the mapping from template to
    band structure is Legal's. Two owners, one matrix.

    The matrix is therefore declared with two owner columns and CODEOWNERS
    splits on them. policy/ownership.toml carries the split; a PR touching a
    Legal column without a Legal approver does not merge. That is spec 5.15's
    problem solved by repository policy attaching to a line the framework made
    visible (doc 04 2.1), which is the correct division of labour.
    """
    pass  # band lookup; two-owner matrix


def cure_available(breach_class: int, covenant_family: int) -> bool:
    """Spec 5.5.4. Equity cure where the definition grants it -- and the
    definition states whether the injection reduces borrowings (the Bank's
    standard) or is applied to cashflow (older instances). It is NEVER applied
    to EBITDA, because that masks a profitability problem rather than solving a
    liquidity one and inflates the next three rolling tests as well.

    Which convention applies is bound, not policy. A 2026 instance on the
    cashflow convention stays on it forever, and change scenario 10 is exactly
    that instance being cured for the first time in 2031.
    """
    pass  # from the bound definition's cure_rights


def cured_is_not_a_pass(result: int, cure_applied: bool) -> int:
    """A cured breach is recorded as a BREACH THAT WAS CURED. Spec 5.5.4.

    "A design that writes 'pass' after a cure has destroyed the fact that
     matters." The distinction survives into the review pack, the watchlist
     signal set and the staging assessment, so it is a distinct result code and
     not a flag on a pass -- a flag on a pass is dropped by the first report
     that groups by result.
    """
    pass  # result stays BREACH; cure_state records the cure and its period


CovenantTest = module(
    covenant_inputs, measured_value, headroom, result, breach_class,
    cure_available, cured_is_not_a_pass,
    name="covenant_test",
    contractual=True,          # <- time/dating.py R2: no Dated[T] read permitted
    taps=["covenant_measured_value", "threshold", "breach_class", "branch_path"],
    owner="covenant_operations",
    co_owners=["legal", "business_credit_risk_policy"],
)
