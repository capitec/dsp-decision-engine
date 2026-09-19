"""Capacity pools, declared. Twelve pools, nine of them constrained.

A pool is not a number. It carries its supply, its unit cost, its pacing basis
and — critically — its GRAIN, because the field-visit pool is routable to at most
three regions and the agency pool is contracted monthly and consumed daily.
"""

from decider2 import Pool, param
from decider2.types import cents, f8, i2, i4

EARLY_AGENTS = Pool(
    id="early_agents", supply_source="capacity_feed", unit="attempts",
    unit_cost=1280, treatments=(5, 6), pacing="daily", revisable_intraday=True)

LATE_AGENTS = Pool(
    id="late_agents", supply_source="capacity_feed", unit="attempts",
    unit_cost=2140, treatments=(6, 7), pacing="daily", revisable_intraday=True)

LEGAL_AGENTS = Pool(
    id="legal_agents", supply_source="capacity_feed", unit="attempts",
    unit_cost=2140, treatments=(10, 11), pacing="daily")

BUSINESS_AGENTS = Pool(
    id="business_agents", supply_source="capacity_feed", unit="attempts",
    unit_cost=2140, treatments=(5, 6, 7), restrict_to={"product_family_code": [4]},
    pacing="daily")

SMS = Pool(id="sms", supply_source="capacity_feed", unit="messages",
           unit_cost=9, treatments=(1,), pacing="daily")
EMAIL = Pool(id="email", supply_source="capacity_feed", unit="messages",
             unit_cost=0.4, treatments=(2,), pacing="daily")
IN_APP = Pool(id="in_app", supply_source="capacity_feed", unit="messages",
              unit_cost=0, treatments=(3,), unconstrained=True)
IVM = Pool(id="ivm", supply_source="capacity_feed", unit="messages",
           unit_cost=28, treatments=(4,), pacing="daily")

FIELD = Pool(
    id="field", supply_source="capacity_feed", unit="visits", unit_cost=42000,
    treatments=(8,), pacing="daily",
    sub_grain="region_code", max_sub_grains=3,
    # 180 visits routable to at most 3 regions. A pool with a sub-grain is a
    # nested allocation and the framework must say so rather than letting the
    # author discover it by writing a second allocator.
)

NOTICES = Pool(id="notices", supply_source="capacity_feed", unit="notices",
               unit_cost=3850, treatments=(10,), pacing="daily")

LEGAL_HANDOVER = Pool(
    id="legal_handover", supply_source="contract", unit="matters",
    unit_cost=125000, treatments=(11,),
    pacing="monthly_paced",
    monthly_quota=param(2500, ge=0, le=20000, owner="recoveries_and_legal"),
    # "The daily allocation must pace against the MONTH-TO-DATE position, not
    # simply take the best available candidates on the first of the month and
    # starve the rest" (spec §5.9). So pacing is a declared pool property with a
    # month-to-date input, not a rule somebody remembers to write.
)

AGENCY = Pool(
    id="agency", supply_source="contract", unit="placements",
    unit_cost="commission", treatments=(9,),
    pacing="monthly_paced_with_tolerance",
    monthly_quota=param(40000, ge=0, le=200000, owner="agency_management"),
    daily_tolerance_pct=param(15, ge=0, le=50, owner="agency_management"),
    split=param({"agency_a": 45, "agency_b": 35, "agency_c": 20},
                owner="agency_management"),
    # Change scenario 9 — "two agencies become five, with per-agency bucket
    # appetite and performance-based volume reallocation monthly" — is a change
    # to `split` (a value) plus a `restrict_to` per sub-grain. It is not a code
    # change, which is the test of whether the pool abstraction is right.
    sub_grain="agency_code",
)

POOLS = (EARLY_AGENTS, LATE_AGENTS, LEGAL_AGENTS, BUSINESS_AGENTS, SMS, EMAIL,
         IN_APP, IVM, FIELD, NOTICES, LEGAL_HANDOVER, AGENCY)


# Change scenario 12: "Finance requires write-off recommendations to respect a
# monthly quantum cap, making a per-account recommendation subject to a SECOND
# population-level constraint alongside capacity."
#
# A quantum cap is a pool whose supply is measured in rands of BALANCE rather
# than in units of work. Because a Pool declares its `unit`, this is a new pool
# and nothing else — no second allocator, no special case in the ranking.
WRITE_OFF_QUANTUM = Pool(
    id="write_off_quantum", supply_source="finance_feed", unit="cents_of_balance",
    treatments=(13,), pacing="monthly_paced",
    monthly_quota=param(1_800_000_000, owner="finance"),
    consumes="outstanding_balance",   # <-- each allocation consumes a per-record amount
)
