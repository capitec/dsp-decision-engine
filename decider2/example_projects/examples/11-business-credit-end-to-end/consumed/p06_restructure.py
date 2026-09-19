"""Project 06's concession search and authority model, consumed for L5.

Spec 5.8: the machinery is 06's and is consumed rather than restated -- the
option search, the bounded scenario generation, the NPV cost of concession, the
authority ladder driven by that cost, the stressed affordability test, the
before-and-after comparison, and the rule that a second concession within 12
months raises the required authority by one level.

Also consumed: project 07's portfolio-budget limit decisioning, for L1's limit
decision on the ~96 000 revolving facilities. One line, at the bottom, because
that is genuinely all it costs -- which is worth showing beside the four
components that cost a great deal more.
"""

from decider2 import consume, gap, EXTEND, COMPOSE
from consumed.manifest import MANIFEST

p06 = consume("credit.restructure", manifest=MANIFEST, major=2)
p07 = consume("credit.limit_management", manifest=MANIFEST, major=1)

# --------------------------------------------------------------------------
# The search, consumed. The catalogue, extended. The classification, new.
#
# Spec 5.17.2 is precise about which is which: "the catalogue is a table; the
# classification is new logic". Eight concessions become 21 across nine facility
# types -- three of them with no retail counterpart (hardcore overdraft to
# amortising term; borrowing-base advance-rate reduction for a longer runway;
# release of a contingent facility against cash cover).
#
# Extending a TABLE is cheap and is a gap resolved by EXTEND on our own side:
# the catalogue is a declared dimension of the component, so 21 rows is a table
# edit reviewed by Policy and Recoveries jointly. Extending the SEARCH would not
# be, which is why the concession set is data and the search is not.
# --------------------------------------------------------------------------
ConcessionSearch = p06.concession_search.with_catalogue(
    "tables/concessions/catalogue_21.csv", owner="policy+recoveries",
)
ObligationInventory = p06.obligation_inventory.over("facilities")   # not retail accounts
StressedAffordability = p06.stressed_affordability

FORBEARANCE_CLASSIFICATION = gap(
    "forbearance_classification",
    component="credit.restructure@06",
    needs=(
        "a forbearance classification of record, produced AT DECISION TIME, "
        "with two evidenced limbs -- including for the NEGATIVE case"
    ),
    resolution=COMPOSE,
    owner="this project (lifecycle/forbearance/classification.py)",
    review="2027-06-30",
    because=(
        "the test is regulatory-reporting shaped, not restructure shaped, and "
        "Provisioning owns the rules while Recoveries grants the concession "
        "(spec 5.15.1 collision 6). Composing locally puts the classification "
        "where its three owners can see it. The cost, stated: this is a "
        "COMPOSE that a second consumer (project 08) will eventually want, at "
        "which point it should be promoted. Review date exists for that reason."
    ),
)

GROUP_RESTRUCTURE = gap(
    "group_scoped_search",
    component="credit.restructure@06",
    needs="one search producing per-client outcomes across four businesses",
    resolution=EXTEND,
    owner="credit_committee + recoveries",
    review="2028-01-31",
    because=(
        "spec 5.8.4: one decision, four clients, one authority determined by "
        "the aggregate, and one set of records each client can be shown without "
        "seeing the others. The disclosure boundary is the hard half and it is "
        "general -- project 06's own change scenarios want it."
    ),
)

# Project 07, consumed whole. A record-level decision inside a portfolio budget,
# simulated before deployment. What is new is only that the record is a facility
# with covenants and shared collateral, so a limit change re-opens the
# allocation across every facility sharing that collateral -- which is declared
# on the allocation, not here. modules/security/allocation.py.
LimitDecision = p07.limit_decision.over("revolving_facilities")
