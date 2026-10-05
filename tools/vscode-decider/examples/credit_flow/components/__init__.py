"""The Python steps `flow.yaml` names. Set a breakpoint in any of them, then Debug flow."""
from decider import param, step


def eligible(score: float, min_score: float = param(50.0)) -> bool:
    return score > min_score


@step(output="limit")
def full_limit(limit: float, bonus: float = param(1000.0)) -> float:
    return limit + bonus


@step(output="limit")
def no_limit(limit: float) -> float:
    return 0.0


@step(output="balance")
def pay(balance: float, instalment: float = param(30.0)) -> float:
    return balance - instalment
