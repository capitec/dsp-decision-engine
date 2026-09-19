"""`core.affordability` in distressed mode — one capability, two questions.

Acceptance criterion 7: "The same affordability capability serves granting and
the distressed arrangement test, with the mode as a parameter, no forked
implementation, and the mode named in the evidence."

THE PROBLEM WITH "THE MODE IS A PARAMETER". Three things differ between granting
and distressed (spec §5.6) and only one of them is a number:

  * thresholds       — R350 residual, 5% of net, 85% DTI against granting's 65%.
                       Numbers. A params bundle handles these.
  * expense treatment — in granting the statutory minimum-expense norm is a
                       DISQUALIFIER; here it is a plausibility floor for
                       RECORDING purposes only. Same inputs, different
                       consequence. That is a branch, not a number.
  * income evidence  — different haircut tiers, and "income could not be
                       established" must not collapse into "income is zero".
                       Different bins, different null policy.

So a single scalar `mode` param would be a lie: two of the three differences are
structural. What actually works is that the STRUCTURE contains both arms and the
PARAMETER selects, so both arms are in `lineage()` and `render()` whichever one
runs, and neither project can acquire an arm the other cannot see.

THE MECHANISM: a Profile. A `Profile` is a named, versioned, effective-dated,
REGISTERED bundle of parameter values for a module, owned by the module's owner
rather than by its consumer. Collections does not write
`{"min_residual": 35000, "max_dti": 0.85}` — it writes
`{"profile": "core.affordability:distressed_collections@v4"}` and the values
come from Credit Risk Policy's document.

Why this is not just "a params file with a name" (FRAMEWORK-DEMANDS #17): doc 08
§6.2's params documents are anonymous dicts with an opaque `origin` token the
framework never parses. A profile needs an IDENTITY that resolves through the
same union as a module id — so it is checked, versioned, effective-dated, and
appears in the audit record as a reference rather than as 40 loose values that
happen to have come from somewhere.
"""

from decider2 import Branch, module, param, profile, step
from decider2.types import Date, cents, f8, i1, i2

from credit_core.affordability import (          # the SHARED capability, unforked
    Affordability, ExpenseNorms, IncomeDetermination,
)

DISTRESSED = profile(
    id="core.affordability:distressed_collections",
    version=4,
    owner="credit_risk_policy",
    approver="credit_committee",
    effective_from="2026-04-01",
    description="Affordability assessed for a collections arrangement: the "
                "question is whether the arrangement will hold, not whether new "
                "debt can be added to a stable position.",
    values={
        "mode_code": 2,                       # 1 granting, 2 distressed
        "min_residual_cents": 35_000,         # R350
        "min_residual_pct_of_net": 0.05,
        "max_instalment_to_discretionary": 0.85,   # 0.65 in granting
        "expense_norm_role": "recording_floor",    # vs "disqualifier"
        "income_haircut_tier_set": "distressed_v2",
        "unestablished_income_treatment": "indeterminate",   # NOT zero
    },
)


@step(output="expense_norm_role_applied")
def expense_norm_role(params) -> i1:
    """Named in the evidence. `affordability_mode_code` and
    `expense_norm_role_applied` are both required outputs, because criterion 7
    says the mode must be NAMED in the evidence and not implied by the numbers."""
    pass


def norm_is_disqualifier(params) -> bool:
    pass  # params.expense_norm_role == "disqualifier"


ExpenseNormTreatment = Branch(
    norm_is_disqualifier,
    ExpenseNorms.as_disqualifier,      # granting arm
    ExpenseNorms.as_recording_floor,   # distressed arm
    modifies=["living_expenses", "expense_basis_code", "affordability_verdict_code"],
    # Both arms are in the graph and in lineage whichever runs. A reviewer asking
    # "what does collections do with the expense norm" reads the branch, not a
    # config file in another repository.
)


# ---------------------------------------------------------------------------
# The arrangement grid. Minimums as a percentage of the contractual instalment
# AND an absolute floor, by product family. Credit Risk Policy, quarterly.
# ---------------------------------------------------------------------------

from ..matrix.grid import Grid   # noqa: E402

ARRANGEMENT_MINIMUMS = Grid(
    name="arrangement_minimums",
    key=("product_family_code", "arrangement_type_code"),
    values={"min_pct_of_instalment": f8, "absolute_floor_cents": cents,
            "max_duration_months": i1, "max_concurrent": i1},
    source="minimums.v12.csv",
    owner="credit_risk_policy",
    overlayable=False,
)


@step(output="minimum_instalment_cents")
def minimum_instalment(
    contractual_instalment: cents | None,
    revolving_minimum_payment: cents | None,
    product_family_code: i1,
    arrangement_type_code: i1,
) -> cents:
    """41 000 revolving accounts have a null contractual instalment. The grid
    keys revolving off `revolving_minimum_payment` instead — declared in the
    signature as two Optionals rather than as a null check in a body, so a
    reviewer sees that the case exists (doc 03 §1 tier 3)."""
    pass


@step(output="required_authority_level")
def required_authority(
    proposed_instalment_cents: cents,
    minimum_instalment_cents: cents,
    proposed_duration_months: i1,
    max_duration_months: i1,
    arrangements_24m: i2,
    consecutive_failed_arrangements: i2,
    days_since_last_failed_arrangement: i2 | None,
) -> i1:
    """L1..L5 from the authority matrix. Returns the level REQUIRED; the level
    OBTAINED is an input to the acceptance decision and the two are recorded
    separately, because §9.3's audit finding is precisely "granted at L1 where
    the grid required L3"."""
    pass


@step(output="arrangement_lock_code")
def arrangement_limits(
    arrangements_24m: i2,
    consecutive_failed_arrangements: i2,
    max_arrangements_24m: i2 = param(3, ge=1, le=10, owner="credit_risk_policy"),
    max_consecutive_failed: i2 = param(2, ge=1, le=6, owner="credit_risk_policy"),
) -> i1:
    """Change scenario 7 — "a new arrangement type, income-linked variable
    instalment, that is not a fixed amount and so does not fit the
    minimum-percentage grid" — is where this design strains. The grid's value
    schema has no column for a formula, and adding one would be an expression
    language in config, which doc 08 §3.2 forbids. The answer is a registered
    step (`collections:income_linked_minimum`) referenced by id from a seventh
    arrangement type, and a nullable `min_pct_of_instalment`. It works, and it
    is not free: see FRAMEWORK-DEMANDS #19.
    """
    pass


DistressedAffordability = (
    IncomeDetermination.with_profile(DISTRESSED)
    | ExpenseNormTreatment
    | Affordability.with_profile(DISTRESSED)
    | module(expense_norm_role, minimum_instalment, required_authority,
             arrangement_limits, name="arrangement_assessment",
             taps=["affordability_mode_code", "expense_norm_role_applied",
                   "minimum_instalment_cents", "required_authority_level"])
)
