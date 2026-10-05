"""Step functions referenced by name from the flowfile test fixtures."""
from decider import missing_as, param, step


def eligible(score: float) -> bool:
    return score > 50.0


@step(output="limit")
def full_limit(limit: float, bonus: float = param(1000.0)) -> float:
    return limit + bonus


@step(output="limit")
def no_limit(limit: float) -> float:
    return 0.0


def over_cap(limit: float, cap: float = param(4000.0)) -> bool:
    return limit > cap


@step(output="limit")
def trim(limit: float, cap: float = param(4000.0)) -> float:
    return cap


@step(output="balance")
def pay(balance: float, instalment: float = param(30.0)) -> float:
    return balance - instalment


@step(output="b")
def make_b(a: float) -> float:
    return a + 1.0


@step(output="c")
def make_c(b: float) -> float:
    return b * 2.0


def heavy(weight: float = missing_as(0.0), heavy_kg: float = param(20.0)) -> bool:
    return weight > heavy_kg


@step(output="count")
def three() -> int:
    return 3


@step(output="score")
def closeness(index: int) -> float:
    return -abs(index - 2.0)
