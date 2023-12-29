import typing
import pandas as pd
import numpy as np
from pydantic import BaseModel, Field, PrivateAttr


class DefaultScorePattern(typing.NamedTuple):
    group_id: int
    score: float
    description: typing.Optional[str]


class MatchRange(typing.NamedTuple):
    start: float
    end: float


class RangeScorePattern(typing.NamedTuple):
    range: MatchRange
    group_id: int
    score: float
    description: typing.Optional[str]


T = typing.TypeVar("T")
class DiscreteScorePattern(typing.NamedTuple, typing.Generic[T]):
    values: typing.List[T]
    group_id: int
    score: float
    description: typing.Optional[str]

class _Unset:
    pass

# Mixins can make things confusing but they are convenient in this case.
class ScoreCriteriaBuilderMixin(BaseModel):
    variable: str
    other_score: typing.Optional[DefaultScorePattern] = None

    _score_idx = PrivateAttr(default=-1)
    _variable_value: typing.Optional[pd.Series] = PrivateAttr(_Unset)
    _cache_check_array: typing.Tuple[
        typing.Tuple[str,str,str],
        typing.Tuple[typing.Union[pd.Series, np.ndarray], ...]
    ] = PrivateAttr(None)

    def _get_score_idx(self, override_idx: int = None):
        if override_idx is not None: return override_idx
        self._score_idx += 1
        return self._score_idx
    
    def set_run_value(self, value: pd.Series):
        """Sets the value for the scoreCriteria so it isn't needed during runtime"""
        self._variable_value = value


    def _get_check_array(self, bin_prefix: str, score_prefix: str, desc_prefix: str):
        raise NotImplemented()
    
    def _execute(self, *args):
        raise NotImplemented()

    def execute(self, bin_prefix: str, score_prefix: str, desc_prefix: str, value: pd.Series = _Unset):
        if value is _Unset:
            if self._variable_value is _Unset: 
                raise ValueError(f"Value for {self.variable} must either be set during initialization or during execution")
            value = self._variable_value
        # TODO maybe good to add some locks in here. Due to the GIL nothing should happen though
        if self._cache_check_array is None or self._cache_check_array[0] != (bin_prefix, score_prefix, desc_prefix):
            self._cache_check_array = (
                (bin_prefix, score_prefix, desc_prefix),
                *self._get_check_array(bin_prefix, score_prefix, desc_prefix)
            )
        res = self._execute(value, *self._cache_check_array[1:])
        res.index = value.index
        return res


class DiscreteScoreCriteriaBuilderMixin(ScoreCriteriaBuilderMixin, typing.Generic[T]):
    discrete_scores: typing.List[DiscreteScorePattern[T]] = Field(default_factory=list)
    
    def add_discrete_score(
        self, matches: typing.List[T], value: float, description: str, override_idx: int = None
    ):
        idx = self._get_score_idx(override_idx)
        self.discrete_scores.append(DiscreteScorePattern(matches, int(idx), float(value), str(description)))
        return self
    
    def set_other_score(self, value: float, description: str, override_idx: int = None):
        idx = self._get_score_idx(override_idx)
        self.other_score = DefaultScorePattern(int(idx), float(value), str(description))
        return self


class RangeScoreCriteriaBuilderMixin(ScoreCriteriaBuilderMixin):
    range_scores: typing.List[RangeScorePattern] = Field(default_factory=list)

    def add_range_score(
        self,
        lower_thresh: float,
        upper_thresh: float,
        value: float,
        description: str,
        override_idx: int = None,
    ):
        idx = self._get_score_idx(override_idx)
        self.range_scores.append(RangeScorePattern(MatchRange(float(lower_thresh), float(upper_thresh)), int(idx), float(value), str(description)))
        return self

