"""§5.8 -- portfolio budget allocation. **This is the project.**

"This is a population-level decision and it cannot be made per account"
(§5.8). Every stage before this one (`exclusions`, `scoring`, `matrix`,
`caps`, `affordability`, `decrease`) is a `decider` pipeline scored one
account at a time -- by `decider serve` in real time, or by `.run(df)`
over the whole book in batch. This module is deliberately **not** a
`decider` step: it is a plain function over the *whole population's*
per-account results at once, because the thing it computes (a rank, a
funding line, a fairness floor) has no meaning for one row in isolation
(README §4 Q5: "population-level allocation under a constraint is not a
scorecard, a table lookup or a tree").

**Simulation = production, one implementation (§10 item 3; the hard part
named first in §2).** `run_allocation()` is the *only* allocation
function in this project. `simulation.py` calls this exact function, with
a different DataFrame (a candidate snapshot) or different budget
parameters (a candidate matrix's output, a candidate dial) -- never a
second allocation algorithm. `pipeline.batch_score()` produces the
DataFrame this function consumes for both production and simulation.

Built with `polars` directly (not a `frame_step`): the per-account pipeline
above already produced a `pl.DataFrame` via `.run()`, and this stage reads
and writes that whole frame at once -- there is no per-row decider step to
wrap it in, and wrapping a whole-population computation in `frame_step`
would only hide what is actually a plain, testable, population-level
function behind a decider abstraction that adds nothing here.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import polars as pl

from limit_mgmt.vocab import AllocationOutcome, ChangeType, IncreasePath

_CCF = {20: 0.42, 21: 0.55}          # credit conversion factor, by product (§5.8)
_LGD = {20: 0.74, 21: 0.79}          # loss given default, by product (§5.8)
_MARGIN_YIELD = 0.18                 # NIM + interchange + fee yield, combined (§5.8, illustrative)
_RWA_WEIGHT = 0.75                   # risk weight applied to incremental exposure at default
_MIN_MEANINGFUL_INCREASE = {20: 1_000.0, 21: 500.0}  # §5.9


@dataclass(frozen=True)
class BudgetInstruction:
    """§4.1 "ALCO budget instruction": budget, envelopes, ranking selection,
    over-allocation factor, and the cycle dial if one is set."""
    applied_limit_budget: float = 2_400_000_000.0
    over_allocation_factor: float = 1.38
    rwa_envelope: float = 1_150_000_000.0
    expected_loss_envelope: float = 48_000_000.0
    objective: str = "risk_adjusted_return"    # or "expected_value" or "policy_priority"
    fairness_reserve_pct: float = 0.15
    fairness_floor_ratio: float = 0.40
    max_tail_skips: int = 2_000


def _ranking_value(df: pl.DataFrame, objective: str) -> pl.Series:
    additional_limit = (df["final_proposed_limit"] - df["current_limit"]).clip(lower_bound=0.0)
    ccf = df["product_code"].replace_strict(_CCF, default=0.5)
    lgd = df["product_code"].replace_strict(_LGD, default=0.75)
    incremental_balance = additional_limit * ccf
    revenue = incremental_balance * _MARGIN_YIELD
    loss = incremental_balance * df["probability_of_default"] * lgd
    if objective == "expected_value":
        return (revenue - loss).alias("ranking_value")
    if objective == "policy_priority":
        # §5.8: "a weighted score over grade, tenure and relationship depth" -- working
        # depth: grade and tenure only (no relationship-depth input in this slice).
        grade_component = (13 - df["behaviour_grade"]).cast(pl.Float64)
        mob_component = df["mob_band"].cast(pl.Float64)
        return (grade_component * 10.0 + mob_component * 2.0).alias("ranking_value")
    # default: risk-adjusted return per rand of budget (§5.8).
    safe_limit = additional_limit.clip(lower_bound=1.0)
    return ((revenue - loss) / safe_limit).alias("ranking_value")


def _client_suppression(df: pl.DataFrame) -> pl.DataFrame:
    """§5.7 "the simultaneous case": an Immediate-class decrease on any account of a
    client suppresses every increase for that client this cycle. Resolved per client,
    across accounts -- the reason this join happens here rather than in `decrease.py`,
    which only ever sees one account."""
    immediate_clients = (
        df.filter(pl.col("change_type_code") == ChangeType.DECREASE)
        .filter(pl.col("decrease_notice_class_code") == 1)  # NoticeClass.IMMEDIATE
        .select("client_id").unique()
    )
    return df.with_columns(
        pl.col("client_id").is_in(immediate_clients["client_id"].implode()).alias("suppressed_by_client_decrease")
    )


def _eligible_mask(df: pl.DataFrame) -> pl.Series:
    additional = df["final_proposed_limit"] - df["current_limit"]
    min_increase = df["product_code"].replace_strict(_MIN_MEANINGFUL_INCREASE, default=1_000.0)
    return (
        (~df["is_excluded"])
        & (df["increase_path_code"] == IncreasePath.AUTOMATIC)
        & (additional >= min_increase)
        & (~df["suppressed_by_client_decrease"])
    )


def run_allocation(df: pl.DataFrame, budget: BudgetInstruction = BudgetInstruction()) -> pl.DataFrame:
    """§5.8, end to end. Input: the per-account pipeline's batch output for the whole
    population (`final_proposed_limit`, `current_limit`, `product_code`,
    `behaviour_grade`, `mob_band`, `probability_of_default`, `is_excluded`,
    `increase_path_code`, `change_type_code`, `decrease_notice_class_code`,
    `account_id`, `client_id`). Returns `(df, summary)`: the same frame plus
    `allocation_rank`, `ranking_value`, `allocation_outcome_code`, `funded_amount`, and
    a `summary` dict with the cycle-level figures §9's reconciliation needs (funded
    count, envelope consumed, which envelope bound, the funding line)."""
    df = _client_suppression(df)
    df = df.with_columns(_ranking_value(df, budget.objective))
    eligible = _eligible_mask(df)
    df = df.with_columns(eligible.alias("_eligible"))

    pool = (
        df.filter(pl.col("_eligible"))
        .sort(["ranking_value", "account_id"], descending=[True, False])
    )
    additional_limit = (pool["final_proposed_limit"] - pool["current_limit"]).to_numpy()
    ccf = pool["product_code"].replace_strict(_CCF, default=0.5).to_numpy()
    lgd = pool["product_code"].replace_strict(_LGD, default=0.75).to_numpy()
    pd_ = pool["probability_of_default"].to_numpy()
    incremental_exposure = additional_limit * ccf
    incremental_rwa = incremental_exposure * _RWA_WEIGHT
    incremental_el = incremental_exposure * pd_ * lgd

    offer_envelope = budget.applied_limit_budget * budget.over_allocation_factor
    main_envelope = offer_envelope * (1.0 - budget.fairness_reserve_pct)
    reserve_envelope = offer_envelope * budget.fairness_reserve_pct
    main_rwa_envelope = budget.rwa_envelope * (1.0 - budget.fairness_reserve_pct)
    main_el_envelope = budget.expected_loss_envelope * (1.0 - budget.fairness_reserve_pct)

    n = len(pool)
    outcome = [AllocationOutcome.BELOW_LINE] * n
    rank = list(range(1, n + 1))
    funded_amount = [0.0] * n

    spent_limit = spent_rwa = spent_el = 0.0
    tail_skips = 0
    which_bound = None
    for i in range(n):
        limit_i, rwa_i, el_i = additional_limit[i], incremental_rwa[i], incremental_el[i]
        fits = (
            spent_limit + limit_i <= main_envelope
            and spent_rwa + rwa_i <= main_rwa_envelope
            and spent_el + el_i <= main_el_envelope
        )
        if fits:
            outcome[i] = AllocationOutcome.FUNDED
            funded_amount[i] = float(limit_i)
            spent_limit += limit_i
            spent_rwa += rwa_i
            spent_el += el_i
        elif tail_skips < budget.max_tail_skips:
            outcome[i] = AllocationOutcome.TAIL_SKIPPED
            tail_skips += 1
        else:
            outcome[i] = AllocationOutcome.BELOW_LINE
            if which_bound is None:
                which_bound = _binding_envelope(
                    spent_limit, main_envelope, spent_rwa, main_rwa_envelope, spent_el, main_el_envelope,
                )

    pool = pool.with_columns(
        pl.Series("allocation_rank", rank), pl.Series("allocation_outcome_code", [int(o) for o in outcome]),
        pl.Series("funded_amount", funded_amount),
    )

    pool = _fairness_pass(pool, budget, reserve_envelope)

    ranked_ids = pool["account_id"]
    df = df.join(
        pool.select("account_id", "allocation_rank", "allocation_outcome_code", "funded_amount"),
        on="account_id", how="left",
    )
    df = df.with_columns(
        pl.col("allocation_rank").fill_null(0),
        pl.col("allocation_outcome_code").fill_null(int(AllocationOutcome.NOT_RANKED)),
        pl.col("funded_amount").fill_null(0.0),
    )
    df = df.with_columns(
        pl.when(pl.col("suppressed_by_client_decrease") & ~pl.col("_eligible"))
        .then(int(AllocationOutcome.OVERLAY_SUPPRESSED))
        .otherwise(pl.col("allocation_outcome_code"))
        .alias("allocation_outcome_code")
    ).drop("_eligible")

    total_ranked = n
    funded_count = int((pool["allocation_outcome_code"] == int(AllocationOutcome.FUNDED)).sum())
    line_value = float(pool.filter(pl.col("allocation_outcome_code") == int(AllocationOutcome.FUNDED))
                        ["ranking_value"].min()) if funded_count else 0.0
    summary = {
        "total_ranked": total_ranked, "funded_count": funded_count, "funding_line_ranking_value": line_value,
        "envelope_consumed": spent_limit, "rwa_consumed": spent_rwa, "expected_loss_consumed": spent_el,
        "offer_envelope": offer_envelope, "which_envelope_bound": which_bound or "limit",
        "tail_skips": tail_skips,
    }
    return df, summary


def _binding_envelope(spent_limit, main_envelope, spent_rwa, main_rwa_envelope, spent_el, main_el_envelope) -> str:
    remaining = {
        "limit": main_envelope - spent_limit, "rwa": main_rwa_envelope - spent_rwa,
        "expected_loss": main_el_envelope - spent_el,
    }
    return min(remaining, key=remaining.get)


def _fairness_pass(pool: pl.DataFrame, budget: BudgetInstruction, reserve_envelope: float) -> pl.DataFrame:
    """§5.8 rule 5: 144 segments (product x grade x mob_band), a 15% reserve, a segment
    below 40% of the population-wide funded proportion is topped up from the reserve in
    rank order; a segment the reserve cannot bring above the floor is marked
    `FAIRNESS_CAPPED` throughout, so it is visible as one starved segment rather than
    many individually unlucky accounts (§5.8 rule 5)."""
    pool = pool.with_columns(
        (pl.col("product_code").cast(pl.Utf8) + "_" + pl.col("behaviour_grade").cast(pl.Utf8)
         + "_" + pl.col("mob_band").cast(pl.Utf8)).alias("_segment")
    )
    pop_rate = float((pool["allocation_outcome_code"] == int(AllocationOutcome.FUNDED)).mean())
    if pop_rate <= 0:
        return pool.drop("_segment")
    floor_rate = pop_rate * budget.fairness_floor_ratio

    seg_stats = (
        pool.group_by("_segment")
        .agg(
            n=pl.len(),
            funded=(pl.col("allocation_outcome_code") == int(AllocationOutcome.FUNDED)).sum(),
        )
        .with_columns((pl.col("funded") / pl.col("n")).alias("rate"))
    )
    starved = set(seg_stats.filter(pl.col("rate") < floor_rate)["_segment"].to_list())
    if not starved:
        return pool.drop("_segment")

    remaining_reserve = reserve_envelope
    outcome = pool["allocation_outcome_code"].to_list()
    additional = (pool["final_proposed_limit"] - pool["current_limit"]).to_list()
    funded_amount = pool["funded_amount"].to_list()
    segments = pool["_segment"].to_list()
    for i in range(len(outcome)):
        if segments[i] not in starved or outcome[i] == int(AllocationOutcome.FUNDED):
            continue
        needed = additional[i]
        if outcome[i] == int(AllocationOutcome.TAIL_SKIPPED) or remaining_reserve < needed:
            outcome[i] = int(AllocationOutcome.FAIRNESS_CAPPED)
            continue
        outcome[i] = int(AllocationOutcome.FUNDED)
        funded_amount[i] = needed
        remaining_reserve -= needed

    pool = pool.with_columns(
        pl.Series("allocation_outcome_code", outcome), pl.Series("funded_amount", funded_amount),
    ).drop("_segment")
    return pool


def cycle_summary_from(result) -> dict:
    """`run_allocation` returns `(df, summary)`; this helper is only so tests and the
    batch driver can name the second element instead of unpacking a bare tuple."""
    _, summary = result
    return summary
