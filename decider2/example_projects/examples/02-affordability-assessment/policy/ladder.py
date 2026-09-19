"""The evidence ladder's table of contents.

Spec 9.1 lists eight things the adjudicator's artefact must show, in the
regulation's own order. This file is that list, as data, and it is the artefact
Regulatory Compliance signs off. They do not read code; they read this file's
`heading` and `must_answer` strings and the rendered sample beside it.

The relationship to the calculation is enforced in both directions:

  * every `@rung` in modules/ must name a section declared here, or the build
    fails naming the orphan;
  * every section declared here must be reachable by at least one rung, or the
    build fails naming the empty section -- which is how a stage that stops
    emitting evidence is caught rather than quietly producing a shorter
    document.

And the rung templates themselves are checked against the trace schema: a rung
referencing `{paye_bracket_index}` when no step produces it is a build error.
That is the property doc 04 6 does not have. Doc 04 generates its reviewable
artefact from the MODULE DATA -- the rules -- which is a rulebook. This is a
RECEIPT, generated from what ran on one applicant, and the two are different
documents with different failure modes. See FRAMEWORK-DEMANDS #10.
"""

from decider2 import ladder, section

EVIDENCE_LADDER = ladder(
    "affordability_evidence_ladder",
    audience="ombud_adjudicator",
    sections=[
        section(
            "framing", order=1,
            heading="Who was assessed",
            must_answer="Who was assessed, as what, and whose income and debts were in scope.",
        ),
        section(
            "income", order=2,
            heading="Income and how it was evidenced",
            must_answer=(
                "Each income source, its evidence tier, the document or observation "
                "that established it, the haircut and why, the months averaged and "
                "the months excluded with reasons, and the figures before and after."
            ),
        ),
        section(
            "deductions", order=3,
            heading="Statutory deductions",
            must_answer=(
                "Tax, insurance and retirement deductions with the table version "
                "and the rebate class, and the age at decision_date that selected it."
            ),
        ),
        section(
            "court_orders", order=4,
            heading="Court-ordered deductions",
            must_answer="Court-ordered deductions and their instruments.",
        ),
        section(
            "expenses", order=5,
            heading="Living expenses",
            must_answer=(
                "All four expense bases -- declared, statement-derived, statutory "
                "norm, internal norm -- which bound, the norm table version, and "
                "the band and dependant cell."
            ),
        ),
        section(
            "obligations", order=6,
            heading="Existing debt obligations",
            must_answer=(
                "Every account considered, its treatment, the figure used, and "
                "every account excluded with its reason."
            ),
            # The only section with a per-element appendix. `annexes` is what
            # lets the 0..80-row annotation be part of the signed document
            # without putting eighty rows in the middle of the narrative.
            annexes=["accounts_annotated"],
        ),
        section(
            "ladder", order=7,
            heading="Discretionary income",
            must_answer="The ladder to discretionary income, in the prescribed order.",
        ),
        section(
            "capacity", order=8,
            heading="The Bank's affordability buffer",
            must_answer=(
                "The buffer, the residual floor, which bound, and the "
                "unadjusted maximum instalment."
            ),
        ),
        section(
            "overlays", order=9,
            heading="Policy overlays applied by the Bank",
            must_answer=(
                "Every overlay applied, with the unadjusted figure beside the "
                "adjusted one, its approval reference, its scope and its expiry. "
                "Whether the Bank declined someone the statutory calculation "
                "would have approved, and on what authority."
            ),
            # This section renders even when it is empty, with the sentence
            # "No policy overlays applied to this assessment." An absent section
            # and a section reporting nothing are different answers to spec 9.2,
            # and only one of them is an answer.
            render_when_empty="No policy overlays applied to this assessment.",
        ),
        section(
            "verdict", order=10,
            heading="The conclusion",
            must_answer="The verdict, and where indeterminate, what was missing.",
        ),
    ],
    # Spec 9.5: a rendered evidence ladder is among the most sensitive records
    # the Bank holds about a person. The classification travels with the
    # artefact rather than being a property of wherever it is written.
    sensitivity="applicant_financial",
    masking_profile="non_production_masked",
)
