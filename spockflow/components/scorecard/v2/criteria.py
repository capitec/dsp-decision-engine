import typing

import pandas as pd
from pydantic import Field

from spockflow._util import get_name
from .criteria_numerical import ScoreCriteriaNumerical
from .criteria_categorical import ScoreCriteriaCategorical


ScoreCriteria = typing.Annotated[
    typing.Union[ScoreCriteriaCategorical, ScoreCriteriaNumerical],
    Field(discriminator="type"),
]

ScoreCriteriaTypes = typing.Optional[
    typing.Union[typing.Literal["numerical"], typing.Literal["categorical"]]
]


def build_score_criteria(
    variable: pd.Series,
    criteria_type: ScoreCriteriaTypes = None,
    variable_name: typing.Optional[str] = None,
    **kwargs,
) -> ScoreCriteria:
    if isinstance(variable, str):
        variable_name = variable
        variable_value = None
        assert (
            criteria_type is not None
        ), "Criteria type must be specified when score criteria is not created with a pd.Series variable type"
    else:
        variable_name = get_name(variable, variable_name)
        variable_value = variable
    if criteria_type is None:
        criteria_type = (
            "numerical" if pd.api.types.is_numeric_dtype(variable) else "categorical"
        )
    if criteria_type == "numerical":
        criteria = ScoreCriteriaNumerical(variable=variable_name, **kwargs)
    elif criteria_type == "categorical":
        criteria = ScoreCriteriaCategorical(variable=variable_name, **kwargs)
    else:
        raise ValueError(
            f'Invalid Criteria type {criteria_type} expecting either "numerical" or "categorical"'
        )
    if variable_value is not None:
        criteria._variable_value = variable
    return criteria
