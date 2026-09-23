"""The IR.md worked example on the real decider API, plus a loop and a stand-in tree."""
import polars as pl

from decider import ConfigurableStep, branch, dag, flow, frame_step, loop, missing_as, param, step
from decider.engine.ir.decls import Input, Output
from decider.engine.ir.nodes import CallNode

BUREAU = pl.DataFrame({"client_id": [1, 2], "bureau_score": [640.0, 710.0]})


@frame_step(reads=["client_id"], writes=["bureau_score"])
def join_bureau(df: pl.DataFrame) -> pl.DataFrame:
    return df.join(BUREAU, on="client_id", how="left")


def disposable_income(net_income: float, expenses: float) -> float:
    return net_income - expenses


def ratio(disposable_income: float, instalment: float) -> float:
    return disposable_income / instalment


def affordable(ratio: float, min_ratio: float = param(0.3)) -> bool:
    return ratio >= min_ratio


affordability = dag(disposable_income, ratio, affordable, name="affordability")


def term_cap(requested_term: float, ceiling: float = param(60.0)) -> float:
    return min(requested_term, ceiling)


@step(output="term_cap")
def cap_by_income(term_cap: float, min_net_salary: float = missing_as(0.0),
                  cap: float = param(48.0)) -> float:
    return min(term_cap, cap) if min_net_salary < 5000 else term_cap


@step(output="term_cap")
def cap_private(term_cap: float, cap: float = param(54.0)) -> float:
    return min(term_cap, cap)


@step(output="term_cap")
def cap_public(term_cap: float, cap: float = param(60.0)) -> float:
    return min(term_cap, cap)


def is_private(sector_code: int) -> bool:
    return sector_code == 1


@step(outputs=("band", "band_score"))
def banding(ratio: float) -> tuple[int, float]:
    return (1, 10.0) if ratio > 2 else (0, 0.0)


term = flow(
    term_cap,
    cap_by_income,
    branch(is_private, cap_private, cap_public, modifies=["term_cap"], name="by_sector"),
    name="term",
)


def offer(requested_amount: float) -> float:
    return requested_amount


def too_big(offer: float, disposable_income: float) -> bool:
    return offer / 24 > disposable_income * 0.5


@step(output="offer")
def shrink(offer: float) -> float:
    return offer * 0.8


sizing = flow(offer, loop(too_big, shrink, carries=["offer"], max_iterations=20, name="shrink_offer"), name="sizing")


class RiskTree(ConfigurableStep):
    """Stand-in for a tree: a row node whose Python reference reports each tree node it reaches."""

    def to_ir(self, ctx):
        return CallNode(ctx.origin(self), "row", _risk_kernel,
                        (Input("bureau_score", float), Input("ratio", float)), (Output("risk_band", int),), (),
                        reference=_risk_reference)


def _risk_kernel(row, params, consts):
    raise NotImplementedError("compiled trees land with the tree task")


def _risk_reference(row, params, consts, visit):
    score, r = row
    visit("root")
    if score < 680:
        visit("low_score")
        return (2,) if r < 3 else (1,)
    visit("good_score")
    return (0,)


risk_tree = RiskTree(name="risk_tree")

pipeline = join_bureau | affordability | banding | term | sizing | risk_tree

SAMPLE = [
    {"client_id": 1, "net_income": 9000.0, "expenses": 4000.0, "instalment": 1500.0, "requested_term": 72.0,
     "min_net_salary": 4000.0, "sector_code": 1, "requested_amount": 150000.0},
    {"client_id": 2, "net_income": 20000.0, "expenses": 5000.0, "instalment": 2000.0, "requested_term": 36.0,
     "min_net_salary": None, "sector_code": 2, "requested_amount": 90000.0},
]
