"""A covenant definition version: the estate's only contractual artefact.

Spec 6.3, H2, 13-Q3. Table 39: 148 templates, ~1 900 live versions, 960 000 live
instances, and a version retired only when the last instance bound to it closes
-- up to twenty years for product 53.

The whole of this file's job is to make one sentence structurally true:

    A version is immutable once an instance binds to it.

Doc 08's three change classes (values / interiors / skeleton) do not have a row
for this. A covenant definition version is not a value -- editing it changes the
terms of live contracts. It is not an interior -- an interior change recompiles
and the new body applies to everything. It is not skeleton. It is a **fourth
class**: an APPEND-ONLY artefact whose members are addressed by binding, never
superseded, and never resolved by date.

FRAMEWORK-DEMANDS D2 asks for that fourth class by name.
"""

from decider2 import append_only_artefact, param
from time.dating import covenant_definitions, CONTRACT

# --------------------------------------------------------------------------
# The 22 attributes of spec 5.5.1, as a declared schema. They are listed rather
# than summarised because the spec's point is that a definition carries the
# SPREADING RULES too -- the 48 standard lines it references, the accounting
# basis, the annualisation convention -- and a design that stores only "DSCR >=
# 1.30" has already lost the case.
# --------------------------------------------------------------------------
CovenantDefinitionVersion = append_only_artefact(
    "covenant_definition_version",
    resolution=CONTRACT,
    fields=[
        "defined_term", "measure_numerator_lines", "measure_denominator_lines",
        "accounting_basis", "frozen_gaap", "reference_period_convention",
        "test_date_rule", "delivery_obligation", "delivery_deadline_days",
        "certification_form", "certified_by", "grace_runs_from",
        "threshold", "step_schedule", "cure_rights", "cure_limits",
        "tested_entity", "permitted_exclusions", "permitted_addbacks",
        "headroom_basis", "standard_effective_from", "standard_effective_to",
    ],
    # The last two are the version's dates as a STANDARD. They are not the dates
    # it is in force as an INSTANCE, and conflating them is H2 in miniature:
    # a standard withdrawn in 2028 is still the operative term of a 2026
    # contract. The type carries both and no accessor returns "the current one".
    standard_dates_are_not_instance_dates=True,
    immutable_once_bound=True,
    # Spec 6.3 requirement 2: the definition has a DEPENDENCY CLOSURE, and
    # retiring anything in it retires the definition. Declared on the artefact
    # so retirement is refused at build, not discovered at replay.
    closure=covenant_definitions.closure,
    retention="until the last bound instance closes, plus 7 years",
)


def bind(template_id: int, version_id: str, documented_on) -> str:
    """Create the binding. The only writer of `covenant_definition_version`.

    Called exactly once per covenant instance, at O15, at documentation. There
    is no rebind. AM-06 (covenant reset) does not rebind an instance -- it
    creates a NEW definition version dated at the amendment and a new instance,
    leaving every prior test bound to the prior version (spec 5.7 requirement 2).
    """
    pass  # append the binding; refuse if the instance already has one


def render_as_agreed(version_id: str) -> str:
    """The definition in the words it was agreed in. Spec 9.5 item 1.

    Not a paraphrase and not a regeneration from current wording. The rendered
    text is stored with the version, because the renderer itself is code that
    changes, and a 2031 render of a 2026 definition through a 2031 renderer is
    a different document from the one the client signed.
    """
    pass  # return the stored rendering, with its own render-time hash


def counterfactual_on_current_standard(version_id: str, inputs) -> dict:
    """What the result would have been on the Bank's CURRENT standard wording.

    Spec 9.5 item 7. The client will ask, and answering with silence is worse
    than answering with a caveat. The output is labelled `not_the_contractual_test`
    and carries reason 5931, so a downstream report cannot present it as the
    test. This is the ONE place in the project a covenant definition is resolved
    by date -- and it is resolved into a clearly-labelled second answer, never
    into the first.
    """
    pass  # resolve the current standard by decision_date; recompute; label it


def diff_versions(a: str, b: str) -> list:
    """Spec 6.3 requirement 3. "What changed between DSCR v4 and DSCR v7, and
    how many live instances are on each" is asked every time Legal standardises
    wording (change scenario 8). The answer includes the instance counts,
    because the answer without them is a legal opinion and the answer with them
    is an 18-month amendment programme with 40 000 client consents.
    """
    pass  # field-level diff plus live instance counts per version
