import pandas as pd
import typing
from lib.annotators import dangerously_skip_type_validation
from lib.creators import parametrized_input_output


@dangerously_skip_type_validation
@parametrized_input_output()
def expression(expr: str, **kwargs: dict) -> pd.Series:
    """Evaluate an expression"""
    return eval(expr, {"pd": pd}, kwargs)


def if_statement_vec(condition: pd.Series, true_value: pd.Series, false_value: pd.Series) -> pd.Series:
    """Select value based on condition"""
    return true_value.where(condition, false_value)


T = typing.TypeVar('T')
def if_statement(condition: bool, true_value: T, false_value: T) -> T:
    """Select value based on condition"""
    if condition: return true_value
    return false_value

@dangerously_skip_type_validation
@parametrized_input_output()
def collect_to_dict(**kwargs: dict) -> dict:
    return kwargs