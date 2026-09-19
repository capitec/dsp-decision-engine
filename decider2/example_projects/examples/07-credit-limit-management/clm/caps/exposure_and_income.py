"""C2 income multiple, C3 total unsecured exposure, C4 group exposure, C7
affordability. Each is a panel member writing one value named `cap_value_c`.

Members are scopes, so all four may use that name. That is the property that
makes a cap cheap to add -- a new cap is a new module with the same one-value
interface, not a new name to thread through the chain.
"""

from decider2 import module, param, table
from decider2.credit import core, overlay_point

IncomeMultipleCaps = table(
    "clm.income_multiple_caps",
    keys=("product_code", "segment_code"),
    values={"multiple": float},
    domain={"product_code": [20, 21], "segment_code": range(1, 7)},
    dense=True, effective_dated=True,
    owner="Credit Committee",          # a DIFFERENT owner from the matrix (s3)
)

ApplyCapOverlays = overlay_point(
    "cap_overlays",
    register="overlay_set",
    adjusts={"income_multiple": "multiply",
             "total_unsecured_ceiling_c": "cap",
             "net_income_exposure_multiple": "multiply"},
    scope_keys=["product_code", "segment_code", "employment_type_code"],
    order="declared",
)


def income_multiple(product_code: int, segment_code: int, tables) -> float:
    """k, 2.5 - 5.0 by product and employment segment (s6.4). Segment 6 is
    'income tier C or D' -- degraded evidence tightens the multiple."""
    return tables.income_multiple_caps.multiple[
        tables.income_multiple_caps.cell(product_code, segment_code)]


def income_cap_value_c(gross_monthly_income_c: int, income_multiple: float) -> int:
    """C2 - limit at most k x gross monthly income."""
    pass


def unsecured_cap_value_c(
    net_monthly_income_c: int,
    total_unsecured_limit_c: int,
    current_limit_c: int,
    absolute_ceiling_c: int = param(45_000_000, ge=0,
                                    description="R450 000 in cents"),
    net_income_exposure_multiple: float = param(8.0, ge=1.0, le=20.0),
) -> int:
    """C3 - the client's aggregate unsecured limit across the Bank may not
    exceed the lower of R450 000 and 8 x net monthly income. Headroom is that
    ceiling less the client's other unsecured limits."""
    pass


def group_cap_value_c(client_id: int, related_party_exposure_c: int,
                      current_limit_c: int, tables) -> int:
    """C4 - from core.exposure, over the client and related parties. The graph
    walk that finds related parties is frame tier (sources/book.py); the cap
    check is record tier and reads the aggregate as a column."""
    return core.exposure.headroom(related_party_exposure_c, current_limit_c, tables)


def affordability_cap_value_c(max_affordable_instalment_c: int, product_code: int,
                              tables) -> int:
    """C7 - invert the notional instalment. The limit whose minimum payment
    equals `max_affordable_instalment_c` at this product's payment rate (s6.8)."""
    pass


IncomeMultipleCap = module(income_multiple, income_cap_value_c, name="income_multiple",
                           evidence=["income_multiple"]) | ApplyCapOverlays
TotalUnsecuredCap = module(unsecured_cap_value_c, name="total_unsecured")
GroupExposureCap = module(group_cap_value_c, name="group_exposure")
AffordabilityCap = module(affordability_cap_value_c, name="affordability")
