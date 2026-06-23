import polars as pl
from pydantic import BaseModel
from decider.modules.functional import generate_from_functions
from decider.modules import register_graph_module


class CreditScorerConfig(BaseModel):
    dti_weight: float = 200.0
    utilization_weight: float = 100.0
    score_base: float = 800.0


def dti_ratio(debt: pl.Expr, income: pl.Expr) -> pl.Expr:
    return debt / income


def utilization_rate(credit_used: pl.Expr, credit_limit: pl.Expr) -> pl.Expr:
    return credit_used / credit_limit


def credit_score_estimate(
    dti_ratio: pl.Expr,
    utilization_rate: pl.Expr,
    config: CreditScorerConfig,
) -> pl.Expr:
    return (
        pl.lit(config.score_base)
        - dti_ratio * config.dti_weight
        - utilization_rate * config.utilization_weight
    )


CreditScorer = generate_from_functions(
    "credit_scorer",
    dti_ratio,
    utilization_rate,
    credit_score_estimate,
)

register_graph_module(CreditScorer)
