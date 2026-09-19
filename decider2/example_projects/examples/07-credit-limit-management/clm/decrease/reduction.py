"""Per-trigger reduction rules, the floor, the notice class and the closure
authority (s5.7 rules 1-4).

Note the reducer flips: the cap panel takes a MINIMUM over ceilings, the floor
panel takes a MAXIMUM over floors. Same combinator, different `reduce`. That is
the third use of `panel` and it is where the shape earns its generality.
"""

from decider2 import module, panel, param
from clm.vocabulary import ENVELOPES


def target_limit_from_rule_c(reduction_rule_code: int, reduction_parameter: float,
                             current_limit_c: int, current_balance_c: int) -> int:
    """Each trigger carries its own reduction rule, not merely a flag. D05
    reduces to the greater of the current balance and 90% of the current limit;
    D07 reduces to the current balance; D12 reduces to R2 000 or proposes
    closure."""
    pass


def money_owed_floor_c(statement_balance_c: int, unsettled_authorisations_c: int,
                       accrued_unbilled_interest_c: int) -> int:
    """Rule 1 - the new limit may not go below the money already owed, INCLUDING
    unsettled authorisations and accrued unbilled interest. Reducing below a
    transaction the Bank has already authorised puts the client over limit
    through the Bank's own action and generates a fee the Bank must refund."""
    return statement_balance_c + unsettled_authorisations_c + accrued_unbilled_interest_c


def regulatory_floor_c(product_code: int, tables) -> int:
    """The product's contractual minimum limit."""
    pass


def closure_floor_c(head_of_credit_risk_authority_id: int, marker_fraud: bool,
                    marker_deceased: bool, tables) -> int:
    """Rule 3 - a reduction to zero is a facility closure and requires the Head
    of Credit Risk's authority, recorded against the specific account, except on
    fraud or deceased grounds. Absent that authority this floor is the product
    minimum, so a closure is never an automatic consequence of a trigger."""
    pass


DecreaseFloors = panel(
    "decrease_floors",
    members=[module(money_owed_floor_c, name="money_owed"),
             module(regulatory_floor_c, name="product_minimum"),
             module(closure_floor_c, name="closure_authority")],
    reduce="max",                     # the HIGHEST floor binds
    writes={"value": "decrease_floor_c", "binding": "binding_floor_code"},
    evidence=["*"],
)


def decrease_target_limit_c(target_limit_from_rule_c: int, decrease_floor_c: int) -> int:
    """The applied target after the binding floor."""
    return max(target_limit_from_rule_c, decrease_floor_c)


def is_closure(decrease_target_limit_c: int) -> bool:
    """A reduction to zero. Gated by `ensures` on the apply module below."""
    return decrease_target_limit_c == 0


Reduction = (
    module(target_limit_from_rule_c, name="reduction_rule")
    | DecreaseFloors
    | module(decrease_target_limit_c, is_closure, name="decrease_target",
             evidence=["decrease_target_limit_c", "binding_floor_code"])
)

# s9.2's "structurally impossible" requirement, in the graph rather than in a
# test. A violated `ensures` fails the record loudly; it is not a flag someone
# may filter on later.
ApplyDecrease = module(
    ...,
    name="apply_decrease",
    ensures=[
        "notice_despatched_day <= notice_effective_day",
        "decrease_target_limit_c >= money_owed_floor_c",
        "is_closure -> head_of_credit_risk_authority_id != 0",
    ],
    evidence=["notice_class_code", "jurisdiction_code", "notice_effective_day",
              "notice_despatched_day", "head_of_credit_risk_authority_id"],
)
