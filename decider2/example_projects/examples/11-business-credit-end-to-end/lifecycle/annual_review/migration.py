"""Grade migration. The missing 10%, and the document's spine. Spec 5.4.1, 5.10.4.

    Project 05 produces a business grade. The annual review needs a grade
    MIGRATION, which nobody built.

Gap `grade_migration`, resolution COMPOSE (consumed/GAPS.toml). This file is the
compose, and it is deliberately the clearest example in the project of what
consuming 90% looks like in a codebase:

  - the grade itself is `consumed.p05_origination.CombinedGrade`, untouched;
  - what is added is entirely about the RELATIONSHIP between two grades;
  - and the reason 05 cannot supply it is structural, not political: the
    predecessor is not an input to project 05's assessment at all.

---------------------------------------------------------------------------
Marang Engineering, the worked case (spec 5.10.4)
---------------------------------------------------------------------------
  Mar 2026   graded 6, master scale v2, scorecard BUS-COMM-01
  Nov 2027   BUS-COMM-01 recalibrated. Same characteristics, new coefficients.
  Jun 2028   master scale v3: the 6/7 boundary moves 0.35pp of PD
  Feb 2030   BUS-COMM-02 replaces BUS-COMM-01: 34 characteristics, four new,
             TWO OF THE OLD ONES NO LONGER COLLECTED
  Sep 2030   master scale v4: a 15-grade scale collapsed back to 12
  2031       graded 7

  6 -> 7 says it deteriorated one notch.
  Restated onto v4, the 2026 PD of 2.1% falls in v4's grade 7: it did not move.

  Both are defensible. They contradict each other. The system's job is to know
  which it is looking at and to be able to answer NOT COMPARABLE.
"""

from decider2 import module, step, param
from consumed.p05_origination import CombinedGrade
from time.comparability import Compared, rebase, delta, comparison_basis, cause_decomposition
from time.master_scale import master_scale_registry, master_scale_current, map_exists


def three_grades(origination_decision, last_review_decision, now) -> tuple:
    """Spec 5.4.1 output 1: THREE grades, not one -- at origination, at the last
    review, and now -- each with its `master_scale_version`, its component
    decomposition and its overlay decomposition.

    Returned as three `Compared` values, so a report cannot place them in one
    column without going through `rebase`. The 2026 record says grade 6 on v2
    and, where a map exists, grade 7 on v4 by map v2->v4. IT NEVER SAYS GRADE 7.
    Spec 5.10.4 requirement 4: the original grade is never overwritten.
    """
    pass  # (Compared, Compared, Compared) with their scale versions


def restated_onto_current(prior: Compared[int], onto: str) -> Compared[int]:
    """Where a restatement would require inputs that do not exist, the basis is
    NOT COMPARABLE and the system produces that word rather than a number.

    The 2030 BUS-COMM-02 replacement is not invertible -- two characteristics
    are no longer collected, so the 2026 assessment cannot be re-scored under
    it. `map_exists` is per-CELL, not per-map (time/master_scale.py), because
    the v3 -> v4 collapse maps 11 of 15 grades cleanly and leaves 4 with no
    pre-image. Asking at map granularity restates four grades wrongly.
    """
    pass  # rebase(prior, onto); NotComparable(5915) on a per-cell miss


def movement(now: Compared[int], prior: Compared[int]) -> Compared[int]:
    """`delta` refuses across bases. There is no other subtraction defined."""
    pass  # delta(now, prior)


def six_causes(now, prior,
               residual_tolerance_bp: float = param(0.5, ge=0, le=5)) -> dict:
    """The six-way cause decomposition, and the requirement that IT SUMS.

        own_data | structure | entity_data | model | overlay | scheme

    Method, because "apportion the difference" is not a specification: re-run
    the PRIOR decision's inputs through the NOW artefacts ONE AXIS AT A TIME,
    in a declared order, and attribute each step's movement to that axis. Six
    re-runs of one facility. At 15 000 facilities a month in an 8-hour window
    that is 90 000 assessments, which is why the review batch's budget is sized
    for 6x the obvious number and why this is spec 8.3's unnamed tension.

    Each axis needs something the estate already has, or a declared gap:
      own_data     the spread. Have it.
      structure    `structure_delta` from O2. Gap, COMPOSE. Have it.
      entity_data  per-entity verdicts differenced. Have it.
      model        scorecard version pair. Have it.
      overlay      the DIFFERENCE between two adjustment stacks. Gap
                   `overlay_stack_delta`, EXTEND, owner Credit Systems.
                   WITHOUT IT THIS FUNCTION CANNOT SUM, and an unapportionable
                   residual is a reported defect (spec 10 acceptance 2). So an
                   acceptance criterion of this project depends on another
                   team's roadmap item. That is what reuse costs and it is
                   stated rather than worked around.
      scheme       `rebase`. Have it.

    A residual above tolerance is REPORTED WITH THE FACILITY NAMED, never
    absorbed into `own_data`. Absorbing it is the failure that makes the whole
    decomposition worthless, because `own_data` is the bucket a reader trusts.
    """
    pass  # cause_decomposition(now, prior, residual_tolerance_bp)


def migration_report_guard(rows: list) -> None:
    """Spec 10 acceptance 3: a migration report MIXING BASES IS REJECTED, not
    footnoted. Demonstrated by a test that attempts one.

    The guard is not really here -- it is in the type. A report builder assembling
    a column of `Compared` values with differing bases cannot produce a number,
    because `delta` will not compile for it. This function exists to turn the
    type error into a domain error with reason 5916 at the report boundary,
    where a human reads it.
    """
    pass  # assert single basis across rows; raise with 5916 naming the offenders


def portfolio_migration_across_a_recalibration(cohort: list, window) -> dict:
    """Spec 5.10.4 requirement 6, and change scenario 1.

    "A recalibration must not be able to look like a portfolio improvement."
    62 000 reviews already completed under the old version, 118 000 to come.
    Portfolio statistics crossing a recalibration date are either computed on a
    restated basis or EXCLUDED, and which was done is stated on the report.

    The requirement that bites: THE RESTATEMENT MAP MUST EXIST BEFORE THE FIRST
    REVIEW UNDER THE NEW VERSION RUNS. That is a release gate on Credit Risk
    Modelling owned by this project's build -- a `dated_table` with
    `approval_required=True` and no entry for the new pair fails the build of
    the release that introduces the new scorecard. It is the only place this
    project blocks another team's release, and it is deliberate.
    """
    pass  # partition by basis; restate or exclude; state which on the report


GradeMigration = module(
    three_grades, restated_onto_current, movement, six_causes,
    migration_report_guard, portfolio_migration_across_a_recalibration,
    name="grade_migration",
    owner="credit_risk_modelling",
    co_owners=["portfolio_management", "business_credit_risk_policy"],
    consumes=[CombinedGrade],          # rendered at the use site with its pin
    taps=["comparison_basis_code", "six_causes", "residual_bp"],
)
