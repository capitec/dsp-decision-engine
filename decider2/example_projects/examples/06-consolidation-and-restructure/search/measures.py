"""The five objective measures, as registered steps.

Spec question 7: "How is an objective function configuration?"

The answer that does NOT work: put the objective in a `ruleset` interior as a
weighted-sum expression. That is an expression language in config, which doc 08
3.2 removes for good reasons and which would have to come back for exactly one
consumer.

The answer that does: a measure is a REGISTERED STEP with a declared signature;
the objective document is a WEIGHT VECTOR over registered measure ids; and the
weight vector is an interior of a data-shaped module. Adding a sixth measure is
a code change (it is arithmetic, so it is code). Re-weighting, re-scoping and
switching between them is a config change with no release (AC 7).

THE SIGNATURE IS WHERE THE GOVERNANCE LIVES, and this is the invention I am
proudest of in this sketch.

Spec 5.8.3 requires every component measure to be ABSOLUTE, not set-relative,
because a min-max scaled measure makes the winner depend on which losers
happened to be evaluated - which means the budget tier changes the answer and a
replay with a larger budget produces a different winner from the same inputs.

That requirement is normally enforced by review, and review does not catch it,
because "scale between the best and worst evaluated" is exactly what a data
scientist writes when asked to make five measures comparable.

Here it is enforced STRUCTURALLY: an `@measure` step may declare only scalar
inputs from the scenario row and the baseline row. It cannot declare a
set-shaped input, because the decorator's registration checks the signature
against the frame schema and REFUSES an aggregate. A measure that genuinely
needs the evaluated set must declare `set_relative=True`, which is permitted and
which forces the evaluated set into the objective's recorded inputs - so the
property survives as an explicit, visible, audited exception rather than as a
silent one.

That single constraint is also what makes tiered evaluation safe (search/
budget.py). If measures were set-relative, evaluating tier 1 and tier 2 together
would give a different answer from evaluating tier 1 alone, and the degradation
ladder would change the winner rather than merely shortening the list.
"""

from decider2 import param, step
from decider2.search import measure


@measure(id="consol:new_money_ratio", absolute=True)
def new_money_ratio(new_money_released: float, new_money_requested: float) -> float:
    """OBJ-01. New money released as a ratio of what the client asked for.

    Ratio to the REQUEST, not to the best scenario's new money. A client who
    asked for R60 000 and is offered R48 000 scores 0.8 whether the evaluated
    set contained a R60 000 option or not.
    """
    pass  # new_money_released / max(new_money_requested, 1.0), clipped to [0, 1]


@measure(id="consol:commitment_ratio", absolute=True)
def commitment_ratio(
    committed_monthly: float,
    existing_obligations_after: float,
    baseline_total_commitment: float,
) -> float:
    """OBJ-02. Reduction in the client's total monthly commitment, against baseline.

    Expressed as 1 - (new / baseline) so every measure is MAXIMISED. Mixing
    minimised and maximised measures in one weight vector is how a sign error
    ships, and the weight vector is edited by people who do not read code.
    """
    pass  # 1.0 - (committed_monthly + existing_obligations_after) / baseline


@measure(id="consol:total_cost_ratio", absolute=True)
def total_cost_ratio(
    total_cost_of_credit: float,
    retained_remaining_cost: float,
    baseline_total_remaining_cost: float,
) -> float:
    """OBJ-03. Reduction in total cost of credit to debt-free, against baseline.

    Goes NEGATIVE when the scenario costs more than doing nothing, which is the
    common case for a term extension and must not be clipped at zero. A measure
    that cannot express harm cannot be weighted against one that expresses help.
    """
    pass  # 1.0 - (total_cost_of_credit + retained_remaining_cost) / baseline


@measure(id="consol:incremental_value_ratio", absolute=True)
def incremental_value_ratio(
    bank_expected_value: float,
    baseline_exposure: float,
) -> float:
    """OBJ-04. The Bank's incremental expected value per rand of baseline exposure.

    Reads search/evaluate.py's bank_expected_value, which is already net of the
    margin forgone on the Bank's own settled accounts. The subtraction is done
    there rather than here so that a re-weighting cannot accidentally remove it.
    """
    pass  # bank_expected_value / max(baseline_exposure, 1.0)


@measure(id="consol:client_outcome", absolute=True)
def client_outcome(
    instalment_relief: float,
    baseline_total_commitment: float,
    total_cost_delta: float,
    baseline_total_remaining_cost: float,
    accounts_exited: int,
    baseline_account_count: int,
    weighted_rate_reduction_bps: float,
    providers_exited: int,
    baseline_provider_count: int,
    w_relief: float = param(0.30, ge=0.0, le=1.0),
    w_cost: float = param(0.30, ge=0.0, le=1.0),
    w_exit: float = param(0.15, ge=0.0, le=1.0),
    w_rate: float = param(0.15, ge=0.0, le=1.0),
    w_providers: float = param(0.10, ge=0.0, le=1.0),
) -> float:
    """OBJ-05. The blended client-impact measure, and the SHADOW objective.

    Whatever objective is in force, the best scenario under this one is also
    identified and recorded (spec 5.8.8). When the objective in force was OBJ-01
    and the client complains eighteen months later, the question asked will be
    "what was the best outcome available to this client", and the only defensible
    position is to have known the answer at the time.

    Note that this measure has its own five weights and they are params of THIS
    step, namespaced to it. They are a different artefact from the objective
    blend weights, owned by a different committee, changed on a different
    cadence. Flattening the two into one weight vector - which is the obvious
    simplification - would let a Credit Committee re-weight the objective and
    silently change what "client outcome" means, which is the one measure that
    must not move when the objective moves.
    """
    pass  # weighted sum of five ratios, each absolute against baseline


@measure(id="consol:sustainability_12m", absolute=True)
def sustainability_12m(
    committed_monthly: float,
    discretionary_income_stressed: float,
    months_in_arrears: int,
    prior_arrangement_failures: int,
    concession_code: int,
) -> float:
    """Restructure only. Probability the client remains current for 12 months.

    The restructure objective is sustainability, not new money, and it is tested
    against the STRESSED discretionary income - income down 10%, expenses up 8%,
    retained variable-rate obligations repriced 200bp higher. A restructure that
    only works if nothing else goes wrong is not a restructure.
    """
    pass  # calibrated curve over stressed headroom ratio and arrears history


@measure(id="consol:expected_loss_vs_do_nothing", absolute=True)
def expected_loss_vs_do_nothing(
    expected_loss_under_option: float,
    expected_loss_do_nothing: float,
) -> float:
    """Restructure only, and used as a CONSTRAINT rather than a weighted term.

    Spec 5.9: the Bank's expected loss may not exceed the expected loss under the
    do-nothing path. A constraint invalidates; a weight trades off. Trading this
    one off is how a restructure book is destroyed, so the objective document
    declares it under `constraint`, not under `weights`, and the objective kind
    refuses a negative weight on a measure declared as a constraint elsewhere.
    """
    pass  # expected_loss_under_option - expected_loss_do_nothing
