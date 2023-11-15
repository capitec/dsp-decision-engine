import pandas as pd
import typing
from lib.annotators import dangerously_skip_type_validation, require_inject_function_map
from lib.creators import parametrized_input_output


@dangerously_skip_type_validation
@require_inject_function_map
@parametrized_input_output()
def expression(expr: str, _function_map: typing.Optional[typing.Dict[str, typing.Callable]]=None, **kwargs: dict) -> pd.Series:
    """Evaluate an expression"""
    if _function_map is None:
        _function_map = {}
    return eval(expr, {"pd": pd, **_function_map}, kwargs)


def combine_not_null(left: pd.Series, right: pd.Series) -> pd.Series:
    """overwrite null values in the left frame with values from the right frame"""
    mask = left.isnull()
    out = left.copy() # Think you need to copy for hamilton
    out[mask] = right[mask]
    return out

def combine_not_null_df(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    """overwrite null values in the left frame with values from the right frame"""
    mask = left.isnull().all(axis=1, bool_only=True)
    out = left.copy() # Think you need to copy for hamilton
    out[mask] = right[mask]
    return out


def round_value(x: pd.Series, decimals: int = 4) -> pd.Series:
    return x.round(decimals)

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