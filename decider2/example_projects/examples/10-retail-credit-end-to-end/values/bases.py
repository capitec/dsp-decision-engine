"""The three orthogonal axes a value can be versioned along, and the rule that
makes reading one of them without saying which IMPOSSIBLE rather than
discouraged.

Spec 5.21.1 requirement 2, verbatim:

    "A phase that reads 'the' obligations figure without saying which must be
     IMPOSSIBLE, not merely discouraged."

Doc 03 3.3 versions values at scope boundaries and qualifies a tap by producing
module -- `term_cap@sector_cap`.  That is ONE axis: the waterfall chain.  This
project needs three more, live simultaneously:

    BASIS       actual | hypothetical(scenario_ref) | stressed | ep_substituted
    ADJUSTMENT  adjusted | unadjusted
    PASS        loop_pass_index 0..4

`existing_obligations` is one actual plus up to 250 hypotheticals inside one
execution of entry point 5 (spec 5.21.1).  `risk_grade` is four live versions,
because obligation-derived characteristics mean a settlement changes the grade.
`max_affordable_instalment` is a pair per scenario: R5 664.56 adjusted-actual,
R5 940.88 unadjusted-actual, and 250 more pairs.

A naming convention -- `existing_obligations_hypothetical_s37_unadjusted` --
handles this for about six weeks.  Spec 5.21.1's closing line is the whole
reason this file exists:

    "A naming scheme that handles two of the three axes and not the third will
     be discovered in year two, by an adjudicator."

---------------------------------------------------------------------------
THE MECHANISM: basis is AMBIENT inside a declared region, EXPLICIT at its edge
---------------------------------------------------------------------------
Writing `= hypothetical()` on all 83 of P12's decision points would be worse
than the disease -- and the day one of them is missed, an assessment prices a
scenario against the client's actual debt and nobody sees it.  So a phase
declares ONE of three postures, once, in its envelope:

    basis="actual"      the phase refuses to run under a hypothetical ambient
                        basis.  Reachability from a hypothetical region is a
                        BUILD error.  P09's exposure rules: the Bank's real
                        exposure does not fall because a scenario was evaluated.

    basis="inherited"   every multi-basis read resolves to the invocation's
                        ambient basis.  P10, P12, P13: one arithmetic, four
                        evidence modes, 251 obligation versions, zero `if`s.

    basis="explicit"    every multi-basis read must name its own.  P06 (it is
                        the producer), P17 (it must test the basis the offer
                        was PRICED on, not "the" one) and P18 (the disclosure
                        uses the actual figure while the assessment used the
                        hypothetical, on the same page of the same document).

The ambient basis is set by the invoker, never by the invoked:

    P14.evaluate(scenario, basis=Hypothetical(scenario_ref=k))

and the runtime stamps `value_basis_code` / `scenario_ref` / `loop_pass_index`
onto every emitted value automatically, because it knows the ambient basis.
Nobody types a version, which is doc 03 3.3's rule, extended to three axes.

FRAMEWORK-DEMANDS #9, #10, #11.
"""

from __future__ import annotations

from decider2.basis import Axis, AxisValue, ambient, declare_axis

# ---------------------------------------------------------------------------
# Axis 1 -- BASIS.  `value_basis_code` in the record (spec 4.6).
# ---------------------------------------------------------------------------

BASIS = declare_axis(
    "basis",
    record_field="value_basis_code",
    values={
        1: AxisValue("actual", default=True, conditional=False),
        2: AxisValue("hypothetical", keyed_by="scenario_ref", conditional=True),
        3: AxisValue("stressed", keyed_by="stress_ref", conditional=True),
        4: AxisValue("entry_point_substituted", keyed_by="entry_point_code",
                     conditional=False),
    },
    # Spec 5.21.1 requirement 3: the record must state that a hypothetical basis
    # is CONDITIONAL -- it becomes actual only when the settlements execute.
    # The conditionality is a property of the axis value, not a field somebody
    # remembers to set, so it cannot be lost in an output shape that forgot it.
    carries_conditionality=True,
)

ACTUAL = BASIS.actual
HYPOTHETICAL = BASIS.hypothetical
EP_SUBSTITUTED = BASIS.entry_point_substituted


# ---------------------------------------------------------------------------
# Axis 2 -- ADJUSTMENT.  Spec 5.9(b) and 6.4: every adjusted value keeps its
# unadjusted counterpart, because "what would we have done without the overlay?"
# is asked at every Credit Committee.
#
# This axis is NOT optional and NOT a debug mode.  `overlay()` (doc 03, and the
# project 03 sketch) writes the adjusted value; this axis is what keeps the
# input visible beside it without a second pipeline.  Acceptance criterion 20:
# "any decision can be run with the overlay stack disabled, through the SAME
# implementation".  Disabling the stack is selecting a coordinate on this axis,
# not a build flag.
# ---------------------------------------------------------------------------

ADJUSTMENT = declare_axis(
    "adjustment",
    record_field="is_adjusted",
    values={1: AxisValue("adjusted", default=True),
            0: AxisValue("unadjusted")},
    # An overlay that could not be resolved is NOT "unadjusted".  Spec 5.25
    # code 41: not knowing whether an overlay applies is different from knowing
    # that none does, and the silent version produces normal-looking answers
    # that are systematically wrong in one direction.
    forbidden_fallback="unadjusted_on_register_failure",
)


# ---------------------------------------------------------------------------
# Axis 3 -- PASS.  Spec 5.23: `loop_pass_index` 0..4.
#
# Unlike the other two this axis is created by a combinator (Loop) rather than
# declared per value, and it applies to EVERY value the loop body writes.  It
# is here so that the record's three axes have one definition site and so that
# `where()` can enumerate them together.
# ---------------------------------------------------------------------------

PASS = declare_axis(
    "pass",
    record_field="loop_pass_index",
    values={i: AxisValue(f"pass_{i}") for i in range(5)},
    created_by="Loop",
    default_when_no_loop=0,
)


# ---------------------------------------------------------------------------
# The reader-side vocabulary.  These are `param()`-slot markers, exactly like
# doc 03 1's `missing_as(0.0)` -- they sit in the signature where a reviewer
# looks for an interface, not in a body.
#
#     def group_exposure_headroom(existing_obligations: Money = actual(), ...)
#     def scenario_capacity(existing_obligations: Money = hypothetical(), ...)
#     def assertion_07(instalment: Money, ceiling: Money = basis_of("instalment"))
#
# `basis_of(x)` is the one that matters for P17.  Spec 5.18: "the re-derivation
# must test against the obligations basis the offer was PRICED on ... an
# assertion that tests the wrong basis passes when it should fail, which is the
# worst possible behaviour for a safety net."  So assertion 7 does not name a
# basis; it names the value whose basis it must share, and the runtime resolves
# it from the shipped offer's provenance.
# ---------------------------------------------------------------------------

def actual() -> Axis: ...          # noqa: D103  -- re-exported from decider2.basis
def hypothetical() -> Axis: ...    # noqa: D103
def unadjusted() -> Axis: ...      # noqa: D103
def basis_of(value_name: str) -> Axis: ...  # noqa: D103


# ---------------------------------------------------------------------------
# The build-time rule.  This is the line that makes spec 5.21.1 requirement 2
# true by construction.
# ---------------------------------------------------------------------------

BARE_READ_OF_MULTI_BASIS_VALUE = "build_error"
#   A step in a `basis="explicit"` region whose parameter name matches a value
#   registered with more than one axis coordinate, and which carries no axis
#   marker, does not compile.  The error names the value, the axes it has, and
#   the three markers available:
#
#       step 'group_exposure_headroom' reads 'existing_obligations', which is
#       declared with axes (BASIS, ADJUSTMENT) and has up to 251 live versions
#       in one execution of entry point 5.  Name the basis:
#           existing_obligations: Money = actual()
#           existing_obligations: Money = hypothetical()
#           existing_obligations: Money = basis_of("instalment")
#       or declare the phase basis="inherited" if it is basis-agnostic.
#
#   There is no default.  A default here is the defect.
