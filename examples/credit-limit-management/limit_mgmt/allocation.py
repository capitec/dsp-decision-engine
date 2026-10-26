"""Which of the proposed increases the bank can afford to make this month.

The per-account pipeline proposes an increase for every account that qualifies; the bank's monthly limit on new
lending cannot fund them all, and who is funded depends on where an account sits among all the others. So this is a
plain function over the whole book's results, not a step: a rank and a funding line mean nothing for one row.

Accounts that qualify are ranked, then walked down in rank order and funded while the limit, risk-weighted-asset
and expected-loss envelopes hold. An increase is funded in full or not at all. One that does not fit is skipped
(tail-skipped) and the walk goes on, up to `max_tail_skips`, after which the cycle stops. A slice of the offer
envelope is held back to lift segments the ranking starves.
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass

import polars as pl

from limit_mgmt import reasons
from limit_mgmt.vocab import AllocationOutcome as Outcome, ChangeType, IncreasePath, NoticeClass

# By product (card, facility). Illustrative: Treasury would supply these with the instruction.
_CCF = {20: 0.42, 21: 0.55}    # share of the extra limit expected to be drawn
_LGD = {20: 0.74, 21: 0.79}    # loss given default
_MARGIN_YIELD = 0.18           # interest margin, interchange and fees on the drawn balance
_RWA_WEIGHT = 0.75
_MIN_MEANINGFUL_INCREASE = {20: 1_000.0, 21: 500.0}  # below this no offer is worth its notice and consent handling


@dataclass(frozen=True)
class BudgetInstruction:
    """The Asset and Liability Committee's instruction for the cycle.

    Offers go out at `over_allocation_factor` times the applied budget, because only some clients accept.
    `objective` is `"risk_adjusted_return"` (return per rand of limit), `"expected_value"` or `"policy_priority"`.
    A `fairness_reserve_pct` slice of the offer envelope tops up segments (product x grade x tenure band) funded at
    under `fairness_floor_ratio` of the book-wide rate.

    Example: `allocate(scored, BudgetInstruction(applied_limit_budget=1_000_000.0))`
    """

    applied_limit_budget: float = 2_400_000_000.0
    over_allocation_factor: float = 1.38
    rwa_envelope: float = 1_150_000_000.0
    expected_loss_envelope: float = 48_000_000.0
    objective: str = "risk_adjusted_return"
    fairness_reserve_pct: float = 0.15
    fairness_floor_ratio: float = 0.40
    max_tail_skips: int = 2_000


def _by_product(df: pl.DataFrame, table: dict[int, float]) -> pl.Series:
    return df["product_code"].replace_strict(table, default=None, return_dtype=pl.Float64)


def _ranking_value(pool: pl.DataFrame, objective: str) -> pl.Series:
    increase = pool["_increase"]
    drawn = increase * _by_product(pool, _CCF)
    gain = drawn * _MARGIN_YIELD - drawn * pool["probability_of_default"] * _by_product(pool, _LGD)
    if objective == "expected_value":
        return gain
    if objective == "policy_priority":
        return (13 - pool["behaviour_grade"]) * 10.0 + pool["mob_band"] * 2.0  # better grade, longer tenure
    if objective == "risk_adjusted_return":
        return gain / increase
    raise ValueError(f"unknown objective {objective!r}")


def _walk(inc, rwa, loss, envelopes, max_tail_skips):
    """Fund down the ranking. Returns each account's outcome, what was spent, and the first envelope that refused one."""
    outcome, spent, skips, bound = [Outcome.BELOW_LINE] * len(inc), [0.0, 0.0, 0.0], 0, None
    for i, cost in enumerate(zip(inc, rwa, loss)):
        full = [name for name, s, c, cap in zip(("limit", "rwa", "expected_loss"), spent, cost, envelopes) if s + c > cap]
        if not full:
            outcome[i] = Outcome.FUNDED
            spent = [s + c for s, c in zip(spent, cost)]
            continue
        bound = bound or full[0]
        if skips == max_tail_skips:
            break
        outcome[i], skips = Outcome.TAIL_SKIPPED, skips + 1
    return outcome, spent, bound, skips


def _top_up(outcome, inc, segment, reserve, floor_ratio):
    """Fund the best-ranked unfunded accounts of each starved segment from the reserve until it reaches the floor.

    A segment the reserve cannot lift is marked fairness-capped, so it reads as one starved segment and not as many
    unlucky accounts. Returns the reserve spent.
    """
    floor_rate = floor_ratio * sum(o == Outcome.FUNDED for o in outcome) / len(outcome)
    members = _groups(segment)
    rate = lambda ids: sum(outcome[i] == Outcome.FUNDED for i in ids) / len(ids)  # noqa: E731
    left = reserve
    for key in sorted((k for k, ids in members.items() if rate(ids) < floor_rate), key=lambda k: (rate(members[k]), k)):
        ids = members[key]
        need = math.ceil(floor_rate * len(ids) - 1e-9) - sum(outcome[i] == Outcome.FUNDED for i in ids)
        for i in ids:
            if need > 0 and outcome[i] != Outcome.FUNDED and inc[i] <= left:
                outcome[i], left, need = Outcome.FUNDED, left - inc[i], need - 1
        if need > 0:
            for i in ids:
                if outcome[i] != Outcome.FUNDED:
                    outcome[i] = Outcome.FAIRNESS_CAPPED
    return reserve - left


def _flag(scored: pl.DataFrame) -> pl.DataFrame:
    """Marks the accounts whose increase qualifies, and those among them that a decrease on the client suppresses."""
    increase = pl.col("final_proposed_limit") - pl.col("current_limit")
    min_increase = pl.col("product_code").replace_strict(_MIN_MEANINGFUL_INCREASE, default=1_000.0, return_dtype=pl.Float64)
    decreased = scored.filter((pl.col("change_type_code") == ChangeType.DECREASE)
                              & (pl.col("decrease_notice_class_code") == NoticeClass.IMMEDIATE))["client_id"]
    qualifies = (~pl.col("is_excluded") & (pl.col("increase_path_code") == IncreasePath.AUTOMATIC)
                 & (increase >= min_increase) & (pl.col("change_type_code") != ChangeType.DECREASE))
    return scored.with_columns(
        qualifies.alias("_qualifies"), pl.col("client_id").is_in(decreased.implode()).alias("_client_has_decrease"),
        increase.alias("_increase"), min_increase.alias("_min_increase"))


def _fund(pool: pl.DataFrame, budget: BudgetInstruction) -> tuple[list, dict]:
    """The outcome of each ranked account, and the cycle's figures."""
    inc = pool["_increase"].to_list()
    drawn = [x * c for x, c in zip(inc, _by_product(pool, _CCF).to_list())]
    rwa = [x * _RWA_WEIGHT for x in drawn]
    loss = [x * pd * lgd for x, pd, lgd in zip(drawn, pool["probability_of_default"].to_list(), _by_product(pool, _LGD).to_list())]
    offer, main = budget.applied_limit_budget * budget.over_allocation_factor, 1.0 - budget.fairness_reserve_pct
    envelopes = (offer * main, budget.rwa_envelope * main, budget.expected_loss_envelope * main)
    outcome, spent, bound, skips = _walk(inc, rwa, loss, envelopes, budget.max_tail_skips)
    segment = list(zip(pool["product_code"].to_list(), pool["behaviour_grade"].to_list(), pool["mob_band"].to_list()))
    # ponytail: the reserve is limit-only and is spent after the ranking walk (so we know who is starved), not before
    reserve = offer * budget.fairness_reserve_pct
    reserve_spent = _top_up(outcome, inc, segment, reserve, budget.fairness_floor_ratio) if outcome and reserve else 0.0
    line = [v for v, o in zip(pool["ranking_value"].to_list(), outcome) if o == Outcome.FUNDED]
    starved = {key for key, ids in _groups(segment).items() if any(outcome[i] == Outcome.FAIRNESS_CAPPED for i in ids)}
    return outcome, {
        "total_ranked": len(outcome), "funded_count": len(line), "funding_line_ranking_value": min(line, default=None),
        "envelope_consumed": spent[0] + reserve_spent, "reserve_consumed": reserve_spent,
        "rwa_consumed": spent[1], "expected_loss_consumed": spent[2], "offer_envelope": offer,
        "which_envelope_bound": bound, "tail_skips": skips, "starved_segments": sorted(starved)}


def allocate(scored: pl.DataFrame, budget: BudgetInstruction = BudgetInstruction()) -> tuple[pl.DataFrame, dict]:
    """Decide which proposed increases are funded. Returns the book with the decision on every account, and the cycle.

    `scored` is the per-account pipeline's output for the whole book (it needs `client_id` and `account_id`). The result
    adds `allocation_rank` (0 if not ranked), `ranking_value`, `allocation_outcome_code` (`AllocationOutcome`),
    `funded_amount`, `granted_limit` (the limit the account ends the cycle with, before any client confirmation) and
    `allocation_reason_code` (why not, if not funded).

    The cycle is a dict: `total_ranked`, `funded_count`, `funding_line_ranking_value`, `offer_envelope`, the envelopes
    consumed (`envelope_consumed`, `rwa_consumed`, `expected_loss_consumed`, and `reserve_consumed`, the part of the
    limit envelope the fairness top-up spent), `which_envelope_bound` (None if none ran out), `tail_skips` and
    `starved_segments`.

    An immediate-class decrease on any account of a client suppresses every increase for that client.

    Example::

        book, cycle = allocate(scored, BudgetInstruction(applied_limit_budget=5_000_000.0))
        cycle["funded_count"], book.filter(pl.col("funded_amount") > 0)["account_id"]
    """
    df = _flag(scored)
    pool = df.filter(pl.col("_qualifies") & ~pl.col("_client_has_decrease"))
    pool = pool.with_columns(_ranking_value(pool, budget.objective).alias("ranking_value")).sort(
        ["ranking_value", "account_id"], descending=[True, False])  # a tie goes to the lower account id
    outcome, cycle = _fund(pool, budget)
    pool = pool.with_columns(
        pl.Series("allocation_rank", range(1, len(outcome) + 1), dtype=pl.Int64),
        pl.Series("allocation_outcome_code", [int(o) for o in outcome], dtype=pl.Int64),
        pl.Series("funded_amount", [x if o == Outcome.FUNDED else 0.0 for x, o in zip(pool["_increase"], outcome)], dtype=pl.Float64))
    unranked = (pl.when(pl.col("_qualifies") & pl.col("_client_has_decrease")).then(int(Outcome.SUPPRESSED))
                .otherwise(int(Outcome.NOT_RANKED)))
    granted = (pl.when(pl.col("change_type_code") == ChangeType.DECREASE).then(pl.col("final_proposed_limit"))
               .otherwise(pl.col("current_limit") + pl.col("funded_amount")))
    book = (df.join(pool.select("account_id", "allocation_rank", "ranking_value", "allocation_outcome_code", "funded_amount"),
                    on="account_id", how="left")
            .with_columns(pl.col("allocation_rank").fill_null(0), pl.col("funded_amount").fill_null(0.0),
                          pl.col("allocation_outcome_code").fill_null(unranked))
            .with_columns(granted.alias("granted_limit"), _reason())
            .drop("_qualifies", "_client_has_decrease", "_increase", "_min_increase"))
    return book, cycle


def _groups(segment) -> dict:
    members = defaultdict(list)
    for i, key in enumerate(segment):
        members[key].append(i)
    return members


def _reason() -> pl.Expr:
    """Why an account was not funded, as one of `reasons.R_*`; null when it was."""
    outcome, increase, minimum = pl.col("allocation_outcome_code"), pl.col("_increase"), pl.col("_min_increase")
    overlay_cost_it = pl.col("policy_proposed_limit_unadjusted") - pl.col("current_limit") >= minimum
    return (
        pl.when(outcome == Outcome.FUNDED).then(None)
        .when(outcome == Outcome.BELOW_LINE).then(reasons.R_BELOW_FUNDING_LINE)
        .when(outcome == Outcome.TAIL_SKIPPED).then(reasons.R_TAIL_SKIPPED)
        .when(outcome == Outcome.FAIRNESS_CAPPED).then(reasons.R_FAIRNESS_CAPPED)
        .when(outcome == Outcome.SUPPRESSED).then(reasons.R_SUPPRESSED_BY_CLIENT_DECREASE)
        .when(pl.col("is_excluded")).then(reasons.R_EXCLUDED)
        .when(pl.col("change_type_code") == ChangeType.DECREASE).then(None)
        .when(pl.col("increase_path_code") == IncreasePath.FAIL).then(
            pl.when(pl.col("evidence_sufficiency_code") != 0).then(reasons.R_AFFORDABILITY_UNESTABLISHED)
            .otherwise(reasons.R_AFFORDABILITY_FAIL))
        .when(pl.col("increase_path_code") == IncreasePath.CONDITIONAL).then(None)  # awaits the client's confirmation
        .when((increase < minimum) & overlay_cost_it).then(reasons.R_REDUCED_BELOW_MINIMUM_BY_OVERLAY)
        .when(increase > 0).then(reasons.R_BELOW_MINIMUM_MEANINGFUL_INCREASE)
        .otherwise(reasons.R_NO_INCREASE_AFTER_CAPS)
    ).cast(pl.Int64).alias("allocation_reason_code")
