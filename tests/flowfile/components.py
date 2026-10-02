"""Step functions referenced by name from the flowfile test fixtures."""
from decider import param, step


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
