"""The IR.md worked example, minus the frame step and the tree, on the stub API."""
from decider_stub import branch, dag, flow, missing_as, param, step


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

pipeline = affordability | banding | term

SAMPLE = [
    {"net_income": 9000.0, "expenses": 4000.0, "instalment": 1500.0, "requested_term": 72.0,
     "min_net_salary": 4000.0, "sector_code": 1},
    {"net_income": 20000.0, "expenses": 5000.0, "instalment": 2000.0, "requested_term": 36.0,
     "min_net_salary": None, "sector_code": 2},
]
