"""Stage 4 -- the statutory and internal expense norm tables.

Both tables are STORED FULLY EXPANDED: 12 bands x 6 dependant counts x 2
components is 144 statutory values, and x 8 products is 1 152 internal ones.
Not a base table times a factor vector. A gazette amendment can change one cell
without changing its neighbours, and a stored formula cannot represent that
(spec 5.4). The dependant multipliers in the spec's prose are how the table was
GENERATED, once; they are not how it is read.

That distinction is the whole reason `dated_table` takes a `key` tuple rather
than a formula. It costs 1 296 rows of CSV and it buys the ability to accept a
gazette that moves one cell.
"""

from decider2 import dated_table, policy, rung, statutory, step

STATUTORY_NORMS = dated_table(
    "statutory_expense_norms",
    key=("band_index", "dependants_bucket"),
    columns={
        "band_floor_cents": "int64",
        "band_ceiling_cents": "int64",       # -1 for the unbounded top band
        "fixed_component_cents": "int64",
        "marginal_rate": "float64",
    },
    owner=statutory,                          # Compliance, on gazette
    versions="tables/statutory/expense_norms/",
    # The edges of this table seed the monotonicity corpus
    # (pipelines/affordability.py). Declaring them as a key is what makes that
    # derivable rather than hand-listed.
    declares_edges=("band_floor_cents", "band_ceiling_cents"),
)

INTERNAL_NORMS = dated_table(
    "internal_expense_norms",
    key=("band_index", "dependants_bucket", "product_code"),
    columns={
        "fixed_component_cents": "int64",
        "marginal_rate": "float64",
        "addendum_cents": "int64",            # vehicle running cost (30), rates and taxes (40)
    },
    owner=policy,                             # Credit Risk Policy, quarterly
    versions="tables/policy/internal_norms/",
    declares_edges=("band_index",),
)


@step(
    description=(
        "Bands include their upper bound; the next band's floor is one cent "
        "higher, so income of exactly R25 000.00 is band 6. Joint applications "
        "select on combined household gross."
    ),
)
def norm_band_index(
    gross_monthly_income_cents: int,
    norms = STATUTORY_NORMS.asof,
) -> int:
    pass  # largest band whose band_floor_cents <= gross; the identity is what an adjudicator reads


@step(description="Dependant counts of five or more share a bucket.")
def dependants_bucket(dependants_count: int) -> int:
    pass  # min(dependants_count, 5)


@rung(
    section="expenses",
    order=74,
    says=(
        "The statutory minimum expense norm is {statutory_norm_cents:money}: "
        "band {norm_band_index} of table version "
        "{statutory_expense_norms@version} at {dependants_count} "
        "dependant{dependants_count:plural}, being a fixed component of "
        "{statutory_fixed_cents:money} plus {statutory_marginal_rate:pct} of "
        "the {statutory_excess_cents:money} by which gross income exceeds the "
        "band floor of {statutory_band_floor_cents:money}."
    ),
)
@step(description="Basis C: the statutory minimum expense norm, fixed component plus marginal rate above the band floor.")
def statutory_norm_cents(
    gross_monthly_income_cents: int,
    norm_band_index: int,
    dependants_bucket: int,
    norms = STATUTORY_NORMS.asof,
) -> int:
    pass  # fixed + (gross - band_floor) * marginal_rate; band 1 is 100% of income and is not special-cased


@step(
    description=(
        "Band 12 is unbounded and its marginal rate continues indefinitely, "
        "which at three million a month gives an absurd norm. The Bank applies "
        "an absolute ceiling above which the norm stops growing."
    ),
)
def statutory_norm_ceiling_applied(
    statutory_norm_cents: int,
    absolute_ceiling_cents: int = policy(2_000_000, ge=0),
) -> bool:
    pass  # recorded when it binds, because it is the Bank departing upward from the table
          # in the applicant's favour, which an adjudicator is entitled to see


@rung(
    section="expenses",
    order=76,
    says=(
        "The Bank's internal norm for product {product_code:product} is "
        "{internal_norm_cents:money}, from table version "
        "{internal_expense_norms@version}."
    ),
)
@step(description="Basis D: the Bank's internal norm, stricter than statutory, per product.")
def internal_norm_cents(
    gross_monthly_income_cents: int,
    norm_band_index: int,
    dependants_bucket: int,
    product_code: int,
    norms = INTERNAL_NORMS.asof,
) -> int:
    pass  # fixed + (gross - floor) * marginal + addendum_cents * financed_asset_count


# Statutory norms do not apply to a juristic person. On products 50 and 51 the
# proprietor's PERSONAL norm applies where a personal surety is assessed --
# which is a product_code row in the internal table, not a branch in this file.
