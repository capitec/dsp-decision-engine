"""Ranking (s5.8). Still per-account: every account can compute its own ranking
value without seeing any other account. This is the last per-record stage.

Two non-obvious requirements shape it.

1. THE OBJECTIVE IS ALCO'S CHOICE, MONTHLY. Three objectives must be supported
   and the choice arrives in the cycle instruction. That is a param selecting a
   `Branch` arm: all three arms are in the graph, so `lineage("ranking_value")`
   correctly reports that all three can affect it, and swapping objectives
   (change scenario 8) needs no deploy. Doc 04 s2 is explicit that a param may
   change which arm a record takes; what it may not change is which arms exist.

2. THE SORT KEY MUST BE EXACT. s5.8 constraint 7 and s8's determinism row
   require the funded set to be bit-for-bit reproducible. A float64 ranking
   value is a fine number and a terrible sort key: two accounts differing in the
   last ULP sort differently under a different partitioning, which moves the
   funding line, which changes 380 000 answers. So the sort key is a quantised
   int64 with a declared scale, and ties fall to `account_id` ascending.
"""

from decider2 import Branch, module, param, quantised, table
from clm.vocabulary import RANK_KEY

RankingCoefficients = table(
    "clm.ranking_coefficients",
    keys=("product_code", "utilisation_band"),
    values={"credit_conversion_factor": float, "net_interest_margin": float,
            "interchange_yield": float, "fee_yield": float, "loss_given_default": float},
    domain={"product_code": [20, 21], "utilisation_band": range(1, 9)},
    dense=True, effective_dated=True,
    owner="ALCO + Model Risk",       # monthly cadence, different owner from the matrix
)


def expected_incremental_balance_c(additional_limit_c: int, product_code: int,
                                   utilisation_band: int, tables) -> int:
    """Additional limit x the credit conversion factor (0.42 card, 0.55 facility,
    varying by utilisation band). Every component is a parameter, not a constant."""
    pass


def expected_incremental_revenue_c(expected_incremental_balance_c: int,
                                   product_code: int, utilisation_band: int,
                                   tables) -> int:
    """Incremental balance x (net interest margin + interchange + fee yield)."""
    pass


def expected_incremental_loss_c(expected_incremental_balance_c: int,
                                probability_of_default: float, product_code: int,
                                tables) -> int:
    """Incremental exposure at default x PD x loss given default."""
    pass


def expected_rwa_c(expected_incremental_balance_c: int, probability_of_default: float,
                   tables) -> int:
    """Risk-weighted asset consumption, for the second envelope."""
    pass


def risk_adjusted_return(expected_incremental_revenue_c: int,
                         expected_incremental_loss_c: int,
                         additional_limit_c: int) -> float:
    """Objective 1, the default: return per rand of budget."""
    return (expected_incremental_revenue_c - expected_incremental_loss_c) / additional_limit_c


def expected_value(expected_incremental_revenue_c: int,
                   expected_incremental_loss_c: int) -> float:
    """Objective 2: absolute, when ALCO wants balance growth over efficiency."""
    return float(expected_incremental_revenue_c - expected_incremental_loss_c)


def policy_priority(behaviour_grade: int, months_on_book: int,
                    relationship_depth: int, shared) -> float:
    """Objective 3: a weighted score over grade, tenure and relationship depth,
    used when retention or a segment strategy dominates."""
    pass


def objective_arm(shared) -> int:
    """Which objective ALCO selected in this cycle's instruction."""
    return shared.ranking_objective_code - 1


Objective = Branch(
    objective_arm,
    [module(risk_adjusted_return, name="risk_adjusted_return", output="ranking_value"),
     module(expected_value, name="expected_value", output="ranking_value"),
     module(policy_priority, name="policy_priority", output="ranking_value")],
    modifies=["ranking_value"],
    name="ranking_objective",
    evidence=["ranking_value", "branch_path"],
)


def incumbency_bonus(prior_funded: bool, prior_ranking_value: float,
                     ranking_value: float,
                     band: float = param(0.05, ge=0, le=0.25,
                                         description="hysteresis band, s5.8 constraint 6"),
                     bonus: float = param(0.03, ge=0, le=0.25)) -> float:
    """Anti-oscillation, as a PER-RECORD adjustment to the ranking value rather
    than as a post-hoc correction to the funded set.

    s5.8 constraint 6 is a statement about the RESULT: an account funded in the
    prior cycle whose ranking value has moved by less than 5% must not be
    displaced by a never-funded account within 5% of it. The mechanism is this
    bonus; the requirement is asserted as a property in
    tests/test_allocation_properties.py. Keeping the mechanism per-record is what
    keeps the allocation re-derivable -- a post-hoc swap would be an amount no
    cell of the matrix produced and no audit could reproduce.
    """
    pass


def rank_key(ranking_value: float, account_id: int) -> int:
    """The exact sort key. Quantised to 1e-6 with declared half-up rounding,
    ties to account_id ascending. See vocabulary.RANK_KEY."""
    return RANK_KEY.of(ranking_value, account_id)


Ranking = (
    module(expected_incremental_balance_c, expected_incremental_revenue_c,
           expected_incremental_loss_c, expected_rwa_c, name="ranking_inputs")
    | Objective
    | module(incumbency_bonus, name="hysteresis")
    | module(rank_key, name="rank_key",
             evidence=["ranking_value", "rank_key"])
)
