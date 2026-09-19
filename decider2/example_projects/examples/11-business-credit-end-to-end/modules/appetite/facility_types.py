"""O11 -- appetite and facility eligibility. The only phase all nine entry
points run, and what EP-9 exists to answer. Spec 5.3 O11.

Written locally, because the nine facility types are this project's. Project
05's appetite grid is 12 grades x 5 sector classes x 4 security types = 240
cells for two products; nine products make it 2 160, and the types are NOT
interchangeable -- a grade 8 client may have an overdraft and may not have a
bridging facility at any amount.

---------------------------------------------------------------------------
The lifecycle's contribution, and it is a large one
---------------------------------------------------------------------------
    Appetite is a function of FACILITY STATE, not only of client risk.

A grade 5 client on watchlist W3 has less appetite than a grade 7 client at W0,
and the reason is not in any scorecard. Two of the eight constraints below come
from state, and they are the two that most often surprise a relationship manager.
"""

from decider2 import module, param, ruleset, dated_table
from time.dating import POLICY

# 9 types x 16 attributes. Facility type is a FIRST-CLASS KEY, not a label: it
# keys the rate table, the covenant default set, the collateral expectations,
# the authority matrix, the review scope and the amendment re-open matrix.
#
# Adding a tenth is change scenario 2 and "the single most informative test of
# the structure". Here it is one row in tables/facility_types.toml plus one
# column in five matrices -- all data -- plus a candidate space, which is also
# data (consumed/p05_origination.py PricingSearch). No Python changes. That is
# the claim; scenario 2 is the test of it.
facility_types = dated_table("facility_type_register", resolution=POLICY,
                             owner="product", cadence="annual")

appetite_maximum = dated_table("appetite_maximum_facility", resolution=POLICY,
                               owner="credit_committee", cells=2_160)

eligibility_matrix = dated_table("facility_eligibility", resolution=POLICY,
                                 owner="credit_committee", cells=12 * 9,
                                 carries_reasons=True)


Constraints = ruleset(
    name="appetite_constraints",
    reads=["business_risk_grade", "sector_appetite_class", "security_type",
           "facility_type", "group_headroom_cents", "sector_portfolio_state",
           "single_name_state", "watchlist_grade", "forbearance_flag",
           "asset_life_months", "product_max_term_months", "sector_max_term_months"],
    writes=["appetite_maximum_cents", "eligible_facility_types",
            "maximum_term_months", "binding_appetite_constraint_code"],
    params="AppetiteParams",
)
"""A `ruleset` interior (doc 08 3), not Python, for one reason: Credit Committee
changes these quarterly and the changes are shaped like rows.

    W2: no increase | W3: no new facility | W4+: no new exposure of any kind
    live forbearance: no new facility without L5-level authority

Those four are the lifecycle's contribution and they are exactly the kind of
rule a UI should be able to add (doc 08 3.3). The `reads=` list is an UPPER
BOUND on the interior's inputs (doc 08 3.2's consequence note), which is what
keeps `lineage("appetite_maximum_cents")` a guarantee under a UI edit.

Note what `reads=` has to include that project 05's version does not:
`watchlist_grade` and `forbearance_flag`. Declaring them is the mechanical
statement that appetite depends on lifecycle state -- and a reviewer reading the
interface sees it without reading a rule.
"""


def maximum_term(product_max: int, sector_max: int, asset_life_months: int,
                 asset_life_share: float = param(0.80, ge=0.5, le=1.0)) -> int:
    """The lesser of the product maximum, the sector maximum, and -- for product
    52 only -- 80% of assessed asset life. Three-way min with a product-specific
    third arm, which is why it is a step and not a table cell.
    """
    pass  # min(product_max, sector_max, floor(asset_life * share)) for type 52


Appetite = module(maximum_term, name="appetite", owner="credit_committee",
                  co_owners=["portfolio_management"],
                  taps=["binding_appetite_constraint_code", "branch_path"])
