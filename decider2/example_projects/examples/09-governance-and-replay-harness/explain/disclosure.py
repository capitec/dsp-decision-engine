"""Field-level classification, by inheritance rather than by declaration.

THE DEVIATION, AND IT IS THE ONE I AM MOST CONFIDENT ABOUT
---------------------------------------------------------
Spec §5.13.6: "Every recorded field carries a classification, applied at
emission by the flow ... so that masking, redaction and export controls are
automatic rather than a matter of somebody remembering."

The obvious framework answer is a new keyword on `@step`:

    @step(output="discretionary_income", classify=PII.FINANCIAL)     # NO

That breaks doc 03 §1.1's law directly. Doc 03 makes the decorator OPTIONAL and
the bare function the cheap path, precisely because doc 01 §5.3's evidence says
the expensive path does not get used (546 inline literals, zero uses of the
config mechanism). Put classification on the decorator and you get 546
unclassified values.

So: CLASSIFICATION IS INHERITED, computed over static lineage.

A derived value's classification is the join (strictest) of its inputs'
classifications. The leaves are classified once, in the input inventory, which
spec §4.6 requires anyway and which manifest/model.py generates from
`pipeline.schema()`. Nobody classifies a step. Ever. The default is safe.

Declaration is needed only to DOWNGRADE, and a downgrade is a reviewable act
with a greppable marker, exactly like doc 04 §3's `@breaks_lineage`:

    @declassify("bureau_score_band", to=Disclose.CLIENT_FACING,
                because="a band is not a score; a band cannot be reverse-engineered "
                        "to a cut-off. Approved COMP-2026-88, review 2027-10-01")
    def bureau_score_band(bureau_score: float) -> int: ...

`grep -rn "@declassify"` over eight codebases yields the complete list of
places where applicant data was deliberately made less protected, with the
reason and the approval on the same lines. That is a governance feature, not a
warning - the same argument doc 04 §3 makes for `@breaks_lineage`, applied to
the thing that actually gets people fired.

This also resolves spec §5.2's explanation-versus-gaming tension without an
editorial habit. A threshold is a PARAM; params carry the disclosure of their
ownership class (harness/tolerances.py); library-policy params are INTERNAL.
So the consultant rendering cannot state a cut-off, because the projection
drops it - not because the template author remembered not to include it.
"""

from __future__ import annotations

from enum import IntEnum


class Disclose(IntEnum):
    """A lattice, ordered. The join of two values is the max."""
    PUBLIC = 0            # product names, term ranges, the fact that a scorecard exists
    CLIENT_FACING = 1     # reason wording, the offer, the applicant's own declared figures
    CONSULTANT = 2        # + queue, channel, what the client can do about it
    ANALYST = 3           # + every value, every version, every threshold, every score
    RESTRICTED = 4        # + raw bureau payload, other clients' data in a cohort, model internals


class PII(IntEnum):
    NONE = 0
    IDENTIFYING = 1       # client_id, id number, contact
    FINANCIAL = 2         # income, expenses, balances, arrears, score
    SPECIAL = 3           # anything touching a prohibited ground (§5.12)


def classification_of(value: str, manifest: "GovernanceManifest") -> tuple[PII, Disclose]:
    """Join over the static lineage cone of `value`. No execution, no data."""
    pass  # manifest.cone_inputs(value) -> max PII, max Disclose, unless a @declassify intervenes


def declassify(output: str, *, to: Disclose, because: str, approval: str | None = None):
    """The only way a value's disclosure becomes weaker than its inputs'."""
    pass  # records the downgrade on the step; the reviewable artefact prints `because` verbatim


def project(record: "DecisionRecord", *, audience: Disclose) -> "ProjectedRecord":
    """Drop every field above `audience`. A field with no classification is a
    BUILD error in the flow, so it can never be silently included here."""
    pass


def mask(record: "DecisionRecord", *, preserve: tuple[str, ...]) -> "DecisionRecord":
    """Spec §13 Q16: the masked derivative must preserve boundary values, null
    patterns, correlations and collection cardinalities, or the golden set stops
    testing anything.

    `preserve` is not a guess: it is the set of values that any `holds`
    assertion or any rule condition in the manifest actually tests, plus every
    band edge from every table the flow reads. Masking is then order-preserving
    within each band and rank-preserving across correlated fields, and the null
    situation (Received.null_situation) is carried through untouched because
    it is a type rather than a value.

    What this does NOT solve, and I will not pretend it does: a masked record
    that preserves band edges, correlations and null patterns over a 45-
    characteristic scorecard is re-identifiable by anyone with the bureau file.
    FRAMEWORK-DEMANDS X4.
    """
    pass
