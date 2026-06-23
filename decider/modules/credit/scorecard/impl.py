import typing as t
from dataclasses import dataclass, field
import polars as pl
import math


@dataclass
class DefaultBin:
    value: float
    name: t.Optional[str] = None


@dataclass
class BoundBin(DefaultBin):
    lower_bound: t.Optional[float] = None
    upper_bound: t.Optional[float] = None

    def with_bounds(self, lower_bound: t.Optional[float] = None, upper_bound: t.Optional[float] = None) -> "BoundBin":
        return BoundBin(
            value=self.value,
            name=self.name,
            lower_bound=lower_bound or self.lower_bound,
            upper_bound=upper_bound or self.upper_bound,
        )

@dataclass
class ValuesBin(DefaultBin):
    items: t.List[int|float|str] = field(default_factory=list)


def adjust_score(score: pl.Expr, offset: float = 0.0, scale: float = 1.0) -> pl.Expr:
    """Apply a linear transformation to the score using the provided offset and scale.

    Args:
        score (pl.Expr): The input score expression.
        offset (float): The value to add to the score.
        scale (float): The value to multiply the score by.
    
    Returns:
        pl.Expr: The transformed score expression.
    """
    # Note we do the checks to avoid bringing in unnecessary compute into the pl.Expr.
    if scale != 1.0:
        score = score * scale
    if offset != 0.0:
        score = score + offset
    return score

def default_output_expr(
    input: pl.Expr,
    current_bin: t.Union[BoundBin, ValuesBin, DefaultBin],
    input_name: t.Optional[str] = None,
) -> pl.Expr:
    common_args = [
        pl.lit(current_bin.value).alias("value"),
        input.alias("input"),
        pl.lit(current_bin.name).alias("bin"),
    ]
    if input_name is not None:
        common_args.append(pl.lit(input_name).alias("input_name"))
    if isinstance(current_bin, BoundBin):
        return pl.struct(
            *common_args,
            pl.lit("bound").alias("type"),
            pl.lit(current_bin.upper_bound).alias("upper_bound"),
            pl.lit(current_bin.lower_bound).alias("lower_bound"),
        )
    elif isinstance(current_bin, ValuesBin):
        return pl.struct(
            *common_args,
            pl.lit("values").alias("type"),
            pl.lit(current_bin.items).alias("values"),
        )
    elif isinstance(current_bin, DefaultBin):
        return pl.struct(
            *common_args,
            pl.lit("default").alias("type"),
        )
    else:
        raise ValueError(f"Unsupported bin type: {type(current_bin)}")


def default_get_value_from_struct(struct: pl.Expr) -> pl.Expr:
    return struct.struct.field("value")


def _get_bound_bin(bound_bins: t.List[BoundBin], i: int) -> BoundBin:
    if 0 <= i < len(bound_bins): return bound_bins[i]
    return BoundBin(value=float('nan'), lower_bound=None, upper_bound=None)


def score_variable(
    input: pl.Expr,
    bound_bins: t.List[BoundBin],
    value_bins: t.List[ValuesBin],
    default_bin: DefaultBin,
    input_name: t.Optional[str] = None,
    output_expr_fn: "type[default_output_expr] | None" = None,
) -> pl.Expr:
    # A little hack because when starting the chain it will look like pl.when
    # and then the next loop will be chain.when
    # so to avoid a check with if expression_chain is none we can just start expr_chain = pl
    # that way the first loop will be pl.when
    if output_expr_fn is None:
        output_expr_fn = default_output_expr
    expr_chain = pl

    # We apply value bins first as they are higher priority than bound bins.
    for b in value_bins:
        condition = input.is_in(pl.lit(list(b.items)))
        bin_expr = output_expr_fn(input, b, input_name)
        expr_chain = expr_chain.when(condition).then(bin_expr)

    for i, b in enumerate(bound_bins):
        lower_bound = b.lower_bound or _get_bound_bin(bound_bins, i-1).upper_bound
        upper_bound = b.upper_bound or _get_bound_bin(bound_bins, i+1).lower_bound

        condition = pl.lit(True)
        # This is the first condition so no need to do condition & ... (Optimize out the literal)
        if lower_bound is not None: condition = (input > lower_bound)
        if upper_bound is not None: condition = condition & (input <= upper_bound)

        bin_expr = output_expr_fn(
            input, 
            b.with_bounds(lower_bound=lower_bound, upper_bound=upper_bound),
            input_name
        )
        expr_chain = expr_chain.when(condition).then(bin_expr)

    default_expr = output_expr_fn(input, default_bin, input_name)

    if expr_chain is pl:
        # We know we have no conditions above so we can just return the default value
        return default_expr
    else:
        return expr_chain.otherwise(default_expr)


def calculate_score(**kwargs: pl.Expr) -> pl.Expr:
    """Calculate the total score from a set of scored variable values
    
    Args:
    **kwargs: A variable number of keyword arguments where each key is the name of a scored variable and each value is a Polars expression representing the score for that variable. For example, you might have:
        score_variable1=pl.col("score_variable1"),
        score_variable2=pl.col("score_variable2"),
        score_variable3=pl.col("score_variable3"),
    Returns:
        pl.Expr: the calculated score.
    """
    return pl.sum_horizontal(list(kwargs.values())) 


def log_odds_from_score(
    score: pl.Expr, 
    anchor_score: float, # Or known as base_points
    target_odds: float, # Or known as base_odds
    points_to_double_the_odds: float # Or known as pdo
) -> pl.Expr:
    """
    Calculate log odds from a credit score.
    
    This function converts a credit score to log odds using a linear transformation.
    The transformation is defined as:
    
        log_odds = (score - anchor_score) / factor + log(target_odds)
    
    where:
        factor = points_to_double_the_odds / log(2)
    
    Args:
        score (pl.Expr): The credit score expression.
        anchor_score (float): The reference score (also known as base_points).
        target_odds (float): The odds at the anchor score (also known as base_odds).
        points_to_double_the_odds (float): Points needed to double the odds (also known as pdo).
    
    Returns:
        pl.Expr: A Polars expression representing the log odds.
    """
    # I was considering writing like pl.lit(2).log() but it seems that its better practice to just use the literal from python side (source chatgpt)
    factor = points_to_double_the_odds / math.log(2)
    return (score - anchor_score) / factor + math.log(target_odds)


def probability_of_default_from_log_odds(
    credit_log_odds: pl.Expr,
    log_odds_safety_cap: float = 40.0,
    financial_rounding: bool = True,
) -> pl.Expr:
    """
    Calculate probability of default (PD) from log odds using numerically stable transformation.
    
    Most people know probability of default is:
        1 - (exp((score - offset) / factor) / (1 + exp((score - offset) / factor)))
    
    However, we can show that:
        log_odds = (score - offset) / factor
        x = exp(log_odds)
        
    And the mathematical transformation:
        1 - (x / (1 + x))
        = (1 + x)/(1 + x) - x/(1 + x)
        = (1 + x - x)/(1 + x)
        = 1/(1 + x)
    
    Hence we use the form 1/(1 + exp(log_odds)) as it's more numerically stable 
    and more efficient to compute.
    
    Args:
        credit_log_odds (pl.Expr): The log odds expression.
        log_odds_safety_cap (float, optional): Maximum log odds value to prevent 
            numerical overflow. Defaults to 40.0.
            pl.LazyFrame().with_columns(pl.lit(89).cast(pl.Float32).exp()).collect() -> inf
            So limiting to 40 ensures the calculation will be stable and not lead to problems later.
            (Note a lower value of closer to 8 may be needed for float16)
        financial_rounding (bool, optional): Whether to apply financial rounding 
            (7 decimal places, Float32 cast). Defaults to True.
    
    Returns:
        pl.Expr: A Polars expression representing the probability of default.
    """
    exp_log_odds = pl.min_horizontal(pl.lit(log_odds_safety_cap), credit_log_odds).exp()
    pd = 1 / (1 + exp_log_odds)
    if financial_rounding:
        pd = pd.round(7).cast(pl.Float32)
    return pd


def calculate_probability_of_default(
    score: pl.Expr,
    anchor_score: float = 660,
    target_odds: float = 15,
    points_to_double_the_odds: float = 20,
    log_odds_safety_cap: float = 40.0,
    financial_rounding: bool = True,
) -> pl.Expr:
    """
    Calculate probability of default from credit score in a two-step process.
    
    This function combines the log odds calculation and probability transformation:
    1. Convert score to log odds using linear transformation
    2. Convert log odds to probability using numerically stable logistic function
    
    Args:
        score (pl.Expr): The credit score expression.
        anchor_score (float): The reference score (also known as base_points).
        target_odds (float): The odds at the anchor score (also known as base_odds).
        points_to_double_the_odds (float): Points needed to double the odds (also known as pdo).
        log_odds_safety_cap (float, optional): Maximum log odds value to prevent 
            numerical overflow. Defaults to 40.0.
        financial_rounding (bool, optional): Whether to apply financial rounding 
            (7 decimal places, Float32 cast). Defaults to True.
    
    Returns:
        pl.Expr: A Polars expression representing the probability of default.
    """
    log_odds = log_odds_from_score(score, anchor_score, target_odds, points_to_double_the_odds)
    return probability_of_default_from_log_odds(log_odds, log_odds_safety_cap, financial_rounding)



def calculate_credit_score(
    probability_of_default: pl.Expr,
    points_to_double_the_odds: float = 20,
    anchor_score: float = 660,
    target_odds: float = 15,
    safety_factor: float = 1e-10,
    return_integer: bool = True,
) -> pl.Expr:
    """
    Calculate credit score from probability of default using inverse logistic transformation.
    
    This function is the inverse of calculate_probability_of_default, transforming a 
    probability of default (PD) back into a credit score using the inverse logistic formula.
    
    The transformation performs the inverse of the PD calculation:
    1. Convert PD to odds: odds = (1 - pd) / pd
    2. Take log odds: log_odds = log(odds)
    3. Convert to score: score = offset + factor * log_odds
    
    Where:
        factor = points_to_double_the_odds / log(2)
        offset = anchor_score - log(target_odds) * factor
    
    Args:
        probability_of_default (pl.Expr): A Polars expression representing the 
            probability of default (between 0 and 1).
        points_to_double_the_odds (float, optional): Number of score points required 
            to double the odds of default (also known as pdo). Defaults to 20.
        anchor_score (float, optional): The reference score (also known as base_points).
            Defaults to 660.
        target_odds (float, optional): The odds at the anchor score (also known as 
            base_odds). Defaults to 15.
        safety_factor (float, optional): Small value to clip PD away from 0 and 1 
            for numerical stability. Defaults to 1e-10.
        return_integer (bool, optional): Whether to round the score to the nearest 
            integer and cast to Int32. Defaults to True.
    
    Returns:
        pl.Expr: A Polars expression representing the calculated credit score,
        optionally rounded to the nearest integer and cast to Int32.
    """
    # Calculation Parameters
    factor: pl.Expr = pl.lit(factor_value := points_to_double_the_odds / math.log(2))
    offset: pl.Expr = pl.lit(anchor_score - (math.log(target_odds) * factor_value))
    # Ensure PD is within (safety_factor, 1-safety_factor) for numerical stability
    pd = probability_of_default.clip(safety_factor, 1 - safety_factor)
    # Note original implementation wrapped this with if probability_of_default.is_null() 
    # However if you push null through the calculation it returns null anyway so i feel its best to leave it out to reduce complexity.
    score = offset + factor * ((1- pd) / pd).log()
    if return_integer:
        score = score.round().cast(pl.Int32)
    return score
