"""`Grid` — a declared, effective-dated, sparse, diffable N-dimensional table.

DEVIATION FROM DOC 03, AND THE LARGEST ONE IN THIS SKETCH.

Doc 03 §4's "Tables (keyed lookups) — provisional" is four lines, sketches
`tables.term.max_loan[term]`, and is described in the document itself as the
"lowest-confidence part". It is a 1-D dense array with a present mask. Nothing in
it survives contact with a 5 376-cell artefact that is 41% sparse, 12% empty,
authored in a spreadsheet by analysts, held in three simultaneous versions for
three cohorts, diffed monthly with volume impact, overlaid by a separately
approved stack, and attributed cell-by-cell to an outcome eighteen months later.

So `Grid` is new. Six properties, each forced by a specific line of the spec.

1. SPARSE AUTHORING, TOTAL RESOLUTION. A row may write `*` in any key column.
   Resolution is most-specific-wins under a DECLARED priority order — never file
   order, because a CSV whose row order is load-bearing cannot be sorted, diffed
   or merged. Two rows of equal specificity that overlap are a BUILD ERROR naming
   both, so "which row won" is never a question. The 5 376 cells are authored in
   ~180 rows and every one of the 5 376 resolves.

2. COVERAGE IS PROVEN, NOT TESTED. `Grid.validate()` expands the full key space
   and asserts total coverage before anything runs. Spec §5.4's "a new matrix
   version must be checkable without waiting for traffic" is a statement about
   the VALIDATOR, and this is it.

3. UNUSED IS DECLARED, NOT INFERRED. Each row carries `coverage` in
   {traffic, sparse, none}. The monthly exercise report compares the declaration
   against reality and reports both directions: a cell declared `none` that took
   traffic, and a cell declared `traffic` that has seen none in six months.
   Acceptance criterion 11. Without this, "wrong" and "unused" look identical.

4. VERSIONS COEXIST. `Grid` resolves by (decision_date, cohort_code). Three
   versions in force for three cohorts is three effective-dated registrations,
   not three deployments.

5. IT IS A GENERIC KERNEL, NOT CODEGEN. Doc 08 §3.4's test — "can one compiled
   loop evaluate every instance of this kind with the instance supplied as
   arrays?" — passes: a grid lookup is `values[linear_index(keys)]`. So a matrix
   release is a VALUE change: no compile, no staged swap, swap the arrays. This
   is what makes a mid-month cell patch a five-minute operation rather than a
   release.

6. THE DIFF IS THE REVIEWABLE ARTEFACT. `Grid.diff(v213, v214)` -> changed cells,
   prior and new values, accounts in each changed cell last month, and — through
   `impact()` — expected demand delta per capacity pool. Acceptance criterion 1,
   and the only artefact in this project a Collections Strategy analyst actually
   signs.
"""

from decider2 import Grid, MostSpecific, param
from decider2.types import Date, i1, i2

from .dimensions import MATRIX_KEY

TREATMENT_MATRIX = Grid(
    name="treatment_matrix",
    key=MATRIX_KEY,
    values={
        "treatment_code": i1,          # 0..13
        "treatment_intensity": i1,     # 1..5
        "permitted_retries": i1,       # 0..4
        "cooling_off_days": i1,        # 0..21
    },
    resolution=MostSpecific(priority=[
        "arrears_bucket_code",         # most specific dimension first
        "product_family_code",
        "collections_band_code",
        "balance_band_code",
        "contact_band_code",
    ]),
    source="treatment_matrix.v214.csv",
    version=214,
    effective_from="2026-09-01",
    cohort_scoped=True,
    owner="collections_strategy",
    cadence="monthly, plus mid-month cell patches",
    validate=[
        "total_coverage",              # every one of 5 376 cells resolves
        "no_ambiguous_overlap",        # no two equal-specificity rows collide
        "values_in_declared_range",
        "treatment_permitted_for_product_family",   # no field visits on revolving
        "intensity_monotone_in_bucket",             # bucket 6 is never lighter than bucket 3
        "cooling_off_ge_channel_min_interval",      # cross-artefact: path/intervals.yaml
    ],
    overlayable=("treatment_intensity", "permitted_retries", "cooling_off_days"),
    # treatment_code is DELIBERATELY ABSENT. Spec §5.4: an overlay "may not
    # introduce a treatment into a cell that the matrix does not permit".
    # Suppression and substitution are separate overlay kinds with their own
    # declared fallback, not a free write to this field. See overlays/surfaces.py.
)


# Other grids in the project. Same mechanism, smaller.

NOTICE_PERIODS = Grid(
    name="notice_periods",
    key=("product_family_code", "notice_type_code"),
    values={"business_days": i1, "delivery_method_code": i1, "proof_required": bool,
            "wording_script_family": str},
    source="notice_periods.v9.csv",
    owner="regulatory_compliance",
    overlayable=False,          # statutory. See overlays/guard.py.
)

CONTACT_HOURS = Grid(
    name="contact_hours",
    key=("day_of_week", "channel_group_code"),
    values={"earliest_minute": i2, "latest_minute": i2},
    source="contact_hours.v4.csv",
    owner="regulatory_compliance",
    overlayable=False,
)

FREQUENCY_CAPS = Grid(
    name="frequency_caps",
    key=("channel_group_code", "window_code", "grain_code"),
    values={"cap": i2},
    source="frequency_caps.v3.csv",
    owner="regulatory_compliance",
    overlayable=False,
)

DISCOUNT_AUTHORITY = Grid(
    name="discount_authority",
    key=("arrears_bucket_code", "recovery_band_code", "authority_level"),
    values={"max_discount_pct": i1, "requires_justification": bool,
            "instalment_variant_penalty_pp": i1},
    source="../settlement/discount_grid.csv",
    owner="credit_risk_policy",
    overlayable=False,          # authority is not a commercial dial
)


def lookup_treatment(
    matrix_cell_id: i2, cohort_code: i2, decision_date: Date, grid=TREATMENT_MATRIX,
) -> tuple:
    """One array index. The whole hot path of the matrix stage."""
    pass  # returns (treatment_code, treatment_intensity, permitted_retries, cooling_off_days)
