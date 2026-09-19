"""Stage 1 -- applicant and household framing.

Who is assessed, as what, and whose money and whose debts are in scope.

One placement decision in this module is forced by the evidence cut in
pipelines/affordability.py and is worth stating: framing runs BEFORE the cut,
so it may not read `accounts`. That rules out the obvious implementation of
"an account on which either applicant is a principal debtor counts once",
which would dedup by walking the two account lists here. Instead framing emits
a *principal-debtor identity set* -- derived from the applicants, not from the
accounts -- and the dedup itself happens after the cut, in
modules/obligations/assemble.py, where the accounts live.

That is not tidiness. If framing read `accounts`, a held prefix would be stale
for every one of project 06's 400 scenarios, and the framework's static-lineage
check on the cut is what says so, at build time, without running anything:

    cut 'evidence' is not sound: module 'framing' (upstream) reads 'accounts',
    which 'resumes'. A held prefix would not be recomputed when 'accounts'
    changes. Move 'framing.principal_debtor_set' downstream of the cut, or
    remove 'accounts' from resumes.
"""

from decider2 import behaviour_table, local, module, policy, rung, step

# --------------------------------------------------------------------------
# Shared-vs-personal expense classification. 14 cells. The cell selects a
# COMBINING BEHAVIOUR, not a value -- `max` for shared categories,
# `sum` for personal ones -- which is the same structural shape as the 45-row
# obligation treatment matrix and the 36-cell haircut matrix. Three uses of
# `behaviour_table` in this project, at 14, 36 and 45 rows.
#
# It is a parameter because, as the spec puts it, Credit Risk Policy will argue
# about food. It is effective-dated because spec change scenario 9 is a court
# finding that one category was consolidated when it should have been summed,
# requiring two years of re-derivation under the corrected rule *while the
# original assessments remain reproducible as they were*. That is only possible
# if the classification is a dated artefact and not a literal.
# --------------------------------------------------------------------------
EXPENSE_CLASSIFICATION = behaviour_table(
    "expense_category_classification",
    key=("expense_category_code",),
    behaviours={
        "SHARED": lambda a, b: max(a, b),      # accommodation, water and electricity, food, insurance
        "PERSONAL": lambda a, b: a + b,        # transport, medical, communication, education, maintenance
    },
    coefficients=(),
    owner=policy,
    versions="tables/policy/expense_categories/",
    unknown_key="REFER",                        # a new category is not silently personal
)


@step(description="A joint application assesses two applicants as one household.")
def household_size(is_joint_application: bool) -> int:
    pass  # 2 when joint, else 1


@rung(
    section="framing",
    order=5,
    says=(
        "Assessed as {household_size:household} with "
        "{dependants_count} dependant{dependants_count:plural}"
        "{dependants_discrepancy_note}."
    ),
)
@step(
    output="dependants_count",
    description="Household dependants. Where two applicants declare different counts, the higher is used.",
)
def household_dependants(
    applicant_a_dependants: int,
    applicant_b_dependants: int | None,
    applicant_b_is_dependant_of_a: bool,
) -> int:
    pass  # max of the two declarations, less any applicant counted as a dependant of the other


@step(description="Whether the two applicants' dependant declarations disagree. Recorded, not resolved silently.")
def dependants_discrepancy(
    applicant_a_dependants: int,
    applicant_b_dependants: int | None,
) -> bool:
    pass  # b is not None and a != b


@step(
    description=(
        "A dependant who is also an applicant is not a dependant. Checked by "
        "identity, never by name -- two people can share a name and one person "
        "can be recorded under two."
    ),
)
def applicant_b_is_dependant_of_a(
    applicant_a_id: int,
    applicant_b_id: int | None,
    applicant_a_dependant_ids: "IdSet",
) -> bool:
    pass  # applicant_b_id in applicant_a_dependant_ids


@step(
    description=(
        "The set of identities whose accounts are in scope. Derived from the "
        "applicants, not from the account list, so that framing stays upstream "
        "of the evidence cut."
    ),
)
def principal_debtor_set(
    applicant_a_id: int,
    applicant_b_id: int | None,
) -> "IdSet":
    pass  # {a} or {a, b}


@step(
    output="evidence_sufficiency_code",
    description=(
        "Where one applicant's income cannot be established to the product's "
        "minimum tier, the household is NOT assessed on the other's income "
        "alone. The verdict is indeterminate, naming the applicant."
    ),
)
def joint_evidence_sufficiency(
    applicant_a_tier_shortfall: bool,
    applicant_b_tier_shortfall: bool | None,
    is_joint_application: bool,
) -> int:
    pass  # JOINT_APPLICANT_A_UNESTABLISHED / JOINT_APPLICANT_B_UNESTABLISHED / 0


# Silently dropping an applicant is the worst available bug here, because it
# produces a plausible number. There is no arm of this step that returns 0 when
# a shortfall exists, and `evidence_sufficiency_code` is not a bool, so there is
# no arithmetic by which `indeterminate` becomes `pass` further down. See
# modules/verdict/ and FRAMEWORK-DEMANDS #9.

Framing = module(
    household_size,
    applicant_b_is_dependant_of_a,
    household_dependants,
    dependants_discrepancy,
    principal_debtor_set,
    joint_evidence_sufficiency,
    name="framing",
    contract="contracts/framing.json",
    taps=["dependants_count", "dependants_discrepancy", "branch_path"],
)
