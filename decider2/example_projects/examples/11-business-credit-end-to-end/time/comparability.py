"""`not comparable` as a value, not as a missing value. Spec 11 5.10.4 / 13-Q6.

    "A design that always produces a number will always produce a wrong one."

Doc 03 1's null policy has three tiers and doc 00 7.4 has three null situations.
None of them is this. A grade restated onto a scale for which no map exists is
not missing, not zero and not "could not be established" -- it is a **refusal**,
with a reason code, that a downstream arithmetic step must be structurally
unable to ignore.

So `Compared[T]` is a value kind with a basis tag travelling beside it:

    Compared[int8]  ==  (value: int8, basis: int8, basis_ref: int32)
                        basis 0 as_graded | 1 restated | 2 not_comparable

Two rules make it load-bearing rather than decorative:

  C1  `delta()` is the only subtraction defined over `Compared`, and it is a
      **build error** to call it on two operands whose basis cannot be shown
      equal. There is no path from two `Compared` values to a number that does
      not go through a basis check. Spec 10 acceptance 3 ("a migration report
      mixing bases is rejected, not footnoted") is then true by construction:
      the report builder cannot construct the offending column.

  C2  Rendering a basis-2 `Compared` emits the WORD, sourced from the reason
      registry (code 5915), never a blank cell and never a dash. A blank cell is
      indistinguishable from a pipeline that did not run.

At the compiled tier this is two int columns and one comparison; it costs a
branch. The expensive part is that every grade-bearing output in the project is
`Compared`, which is roughly 40 columns. That is the price of spec 5.10 and it
is stated in FRAMEWORK-DEMANDS D14 rather than hidden.
"""

from decider2 import compared_type, param
from time.master_scale import master_scale_registry, restatement_maps

Compared = compared_type(
    "compared",
    bases={0: "as_graded", 1: "restated", 2: "not_comparable"},
    carries="basis_ref",     # the restatement map id, or the reason code for a refusal
    arithmetic="basis_checked",
)


def rebase(grade: int, from_scale: str, onto_scale: str) -> Compared[int]:
    """Restate a grade onto another master scale version, or refuse.

    Spec 5.10.4 requirement 3: restatement is a **published artefact**, not a
    calculation somebody does. `restatement_maps` is a `dated_table` of 6
    pairwise 12x12 maps owned by Credit Risk Modelling. Where the pair has no
    map -- the 2030 BUS-COMM-02 replacement is not invertible, two of its
    characteristics are no longer collected -- this returns basis 2 carrying
    reason 5915, and requirement 4 means the original grade is untouched
    alongside it.
    """
    pass  # lookup (from_scale, onto_scale) in restatement_maps; refuse on miss


def delta(now: Compared[int], prior: Compared[int]) -> Compared[int]:
    """The only defined difference. Refuses rather than subtracts across bases."""
    pass  # if either basis == 2 or bases disagree -> NotComparable; else now - prior


def comparison_basis(now: Compared[int], prior: Compared[int]) -> int:
    """`comparison_basis_code` for the decision of record. Spec 4.8, 5.4.2.

    Emitted on EVERY decision of record, including the ones where nothing moved,
    because spec 10 acceptance 1 permits zero exceptions across the book.
    """
    pass  # 0 as-graded, 1 restated, 2 not comparable


def cause_decomposition(
    now, prior,
    residual_tolerance_bp: float = param(0.5, ge=0, le=5,
        description="Unapportioned movement above this is reported as a defect"),
) -> dict:
    """The six-way decomposition of spec 5.4.1, and the requirement that it sums.

        own_data | structure | entity_data | model | overlay | scheme

    Two of the six come free from machinery that already exists and were the
    reason for choosing it:

      - `overlay` is the DIFFERENCE between two `core.adjustments` stacks a year
        apart. Doc 00 6.22 produces the stack in force at one date; this needs
        two dates and a decomposition. Declared as a gap -- consumed/GAPS.toml,
        gap `overlay_stack_delta`, resolution EXTEND.
      - `scheme` is `rebase()` above: the movement attributable purely to the
        master scale, which is the movement that is NOT a credit event.

    A residual above tolerance is reported as a defect with the facility named
    (spec 10 acceptance 2), not absorbed into `own_data` -- absorbing it is the
    failure that makes the whole decomposition worthless, because `own_data` is
    the bucket a reader trusts.
    """
    pass  # re-run prior's inputs through now's artefacts one axis at a time
