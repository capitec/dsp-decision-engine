import typing
from enum import Enum
from typing import List

import pandas as pd
import numpy as np
from pydantic import Field, field_serializer

from .common import (
    RangeScoreCriteriaBuilderMixin,
    ScoreCriteriaBuilderMixin,
    DefaultScorePattern,
)


class Bounds(Enum):
    LOWER = 0
    UPPER = 1


class NumericalDiscreteScorePattern(typing.NamedTuple):
    values: typing.List[typing.Optional[float]]
    group_id: int
    score: float
    description: typing.Optional[str]


class NumericDiscreteScoreCriteriaBuilderMixin(ScoreCriteriaBuilderMixin):
    discrete_scores: typing.List[NumericalDiscreteScorePattern] = Field(
        default_factory=list
    )

    @field_serializer("discrete_scores")
    def serialize_dt(
        self, discrete_scores: typing.List[NumericalDiscreteScorePattern], _info
    ):
        sv = []
        for dc in discrete_scores:
            sv.append(
                {
                    "values": dc.values,
                    "group_id": dc.group_id,
                    "score": dc.score,
                    "description": dc.description,
                }
            )
        return sv

    def add_discrete_score(
        self,
        matches: typing.List[typing.Optional[str]],
        value: float,
        description: str,
        override_idx: int = None,
    ):
        idx = self._get_score_idx(override_idx)
        self.discrete_scores.append(
            NumericalDiscreteScorePattern(
                matches, int(idx), float(value), str(description)
            )
        )
        return self

    def set_other_score(self, value: float, description: str, override_idx: int = None):
        idx = self._get_score_idx(override_idx)
        self.other_score = DefaultScorePattern(int(idx), float(value), str(description))
        return self


class ScoreCriteriaNumerical(
    RangeScoreCriteriaBuilderMixin, NumericDiscreteScoreCriteriaBuilderMixin
):
    type: typing.Literal["numerical"] = Field(default="numerical")
    included_bounds: typing.Tuple[Bounds, ...] = Field(
        default_factory=lambda: (Bounds.LOWER,), max_length=2
    )

    def add_discrete_score(
        self,
        matches: List[float | None],
        value: float,
        description: str,
        override_idx: int = None,
    ):
        matches = [float("nan") if m is None else float(m) for m in matches]
        return super().add_discrete_score(matches, value, description, override_idx)

    # TODO refactor this as part of the compile process
    def _get_check_array(self, bin_prefix: str, score_prefix: str, desc_prefix: str):
        other_record = {
            f"{bin_prefix}{self.variable}": -1,
            f"{score_prefix}{self.variable}": -1,
            f"{desc_prefix}{self.variable}": None,
        }
        if self.other_score is not None:
            other_record = {
                f"{bin_prefix}{self.variable}": self.other_score.group_id,
                f"{score_prefix}{self.variable}": self.other_score.score,
                f"{desc_prefix}{self.variable}": self.other_score.description,
            }

        result_array = [other_record]
        nan_idx = 0
        other_idx = 0

        equals_array = []
        equals_lookup_array = []
        # Put it in reverse so that last items have more priority over earlier items
        for ds in reversed(self.discrete_scores):
            res_idx = len(result_array)
            result_array.append(
                {
                    f"{bin_prefix}{self.variable}": ds.group_id,
                    f"{score_prefix}{self.variable}": ds.score,
                    f"{desc_prefix}{self.variable}": ds.description,
                }
            )
            for v in ds.values:
                if pd.isna(v):  # TODO see if this works with None
                    # Keep first res containing nan
                    if nan_idx == 0:
                        nan_idx = res_idx
                    continue
                else:
                    equals_array.append(v)
                    equals_lookup_array.append(res_idx)

        lower_bounds_array = []
        upper_bounds_array = []
        in_bounds_lookup_array = []
        for rs in reversed(self.range_scores):
            res_idx = len(result_array)
            result_array.append(
                {
                    f"{bin_prefix}{self.variable}": rs.group_id,
                    f"{score_prefix}{self.variable}": rs.score,
                    f"{desc_prefix}{self.variable}": rs.description,
                }
            )
            lower_bounds_array.append(rs.range.start)
            upper_bounds_array.append(rs.range.end)
            in_bounds_lookup_array.append(res_idx)

        res_df = pd.DataFrame(result_array)
        # Note: if multiple true values, items higher on the list will be taken preferentially
        lookup_array = np.array(
            equals_lookup_array  # For equals
            + in_bounds_lookup_array  # For in bounds
            + [nan_idx, other_idx],  # For nan and other
            dtype=np.uint64,
        )
        return (
            res_df,
            lookup_array,
            np.array(equals_array)[:, None],
            np.array(lower_bounds_array)[:, None],
            np.array(upper_bounds_array)[:, None],
        )

    def _execute(
        self,
        value: pd.Series,
        res_df: pd.DataFrame,
        lookup_array: np.ndarray,
        equals_array: np.ndarray,
        lower_bounds_array: np.ndarray,
        upper_bounds_array: np.ndarray,
    ):
        padded_val = value.values[None]
        equals_mask = padded_val == equals_array
        # Implementation here with search sorted could technically be
        # about twice as fast on larger inputs but the absolute difference in
        # microseconds (293us vs 367us for 10k records with 35 ranges)
        # doesn't seem worth the tradeoff in simplicity and flexibility
        if Bounds.LOWER in self.included_bounds:
            in_bounds_mask = padded_val >= lower_bounds_array
        else:
            in_bounds_mask = padded_val > lower_bounds_array
        if Bounds.UPPER in self.included_bounds:
            in_bounds_mask &= padded_val <= upper_bounds_array
        else:
            in_bounds_mask &= padded_val < upper_bounds_array
        is_nan_mask = value.isna().values[None]
        default_mask = np.ones((1, len(value)), dtype=np.bool_)
        idx = np.argmax(
            np.concatenate(
                [equals_mask, in_bounds_mask, is_nan_mask, default_mask], axis=0
            ),
            axis=0,
        )
        idx = lookup_array[idx]
        return res_df.iloc[idx]
