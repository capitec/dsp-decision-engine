import typing

import pandas as pd

from pydantic import BaseModel, field_validator, BeforeValidator


class ScoreCardAdjustmentNoneOperator(BaseModel):
    operation: typing.Union[None, typing.Literal["none"], typing.Literal[""]]
    variable: str

    def apply(self, df: pd.DataFrame) -> pd.Series:
        return df[self.variable]


class ScoreCardAdjustmentSumOperator(BaseModel):
    operation: typing.Literal["sum"]
    variable: str
    value: float

    def apply(self, df: pd.DataFrame) -> pd.Series:
        return df[self.variable] + self.value


class ScoreCardAdjustmentConstOperator(BaseModel):
    operation: typing.Literal["constant"]
    variable: str
    value: float

    def apply(self, df: pd.DataFrame) -> pd.Series:
        return pd.Series([self.value] * len(df))


class ScoreCardAdjustmentFallbackOperator(BaseModel):
    operation: str
    variable: str
    value: float

    @field_validator("operation")
    def InvalidOperatorValidator(operation):
        raise ValueError(
            f"Operation {operation} not supported. Please log a query with AI Platform"
        )

    def apply(self, df: pd.DataFrame) -> pd.Series:
        raise NotImplementedError()


def ensure_lowercase_operation(v: dict):
    operation = v.get("operation")
    if isinstance(operation, str):
        v["operation"] = operation.lower()
    return v


ScoreCardAdjustmentOperator = typing.Annotated[
    typing.Union[
        ScoreCardAdjustmentNoneOperator,
        ScoreCardAdjustmentSumOperator,
        ScoreCardAdjustmentConstOperator,
        ScoreCardAdjustmentFallbackOperator,
    ],
    BeforeValidator(ensure_lowercase_operation),
]


class ScoreCardAdjustmentModel(BaseModel):
    score_prefix: str
    variable_score_adjustments: typing.List[ScoreCardAdjustmentOperator]
