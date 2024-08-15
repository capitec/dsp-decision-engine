import os
import typing

import pandas as pd
from pydantic import BaseModel, Field, field_validator
from .criteria import ScoreCriteria
from ..probability import log_odds_from_score, probability_of_default_from_log_odds
from .adjustments import ScoreCardAdjustmentModel
from spockflow.nodes import creates_node, VariableNode

LATEST_VERSION = "2.2.0"


class ScoreScallingParameters(BaseModel):
    base_points: float
    base_odds: float
    pdo: float


class ScoreCardModel(VariableNode):
    bin_prefix: str
    score_prefix: str
    description_prefix: str
    variable_params: typing.List[ScoreCriteria] = Field(default_factory=list)
    version: str = Field(pattern=r"^2.[0-9]+.[0-9]$", default=LATEST_VERSION)
    score_scaling_params: typing.Optional[ScoreScallingParameters] = None

    @field_validator("score_scaling_params", mode="before")
    def allow_empty_dict(score_scaling_params: typing.Optional[dict]):
        if isinstance(score_scaling_params, dict) and len(score_scaling_params) == 0:
            return None
        return score_scaling_params

    @field_validator("variable_params")
    def validate_no_repeated_variables(variable_params: "typing.List[ScoreCriteria]"):
        seen = set()
        for p in variable_params:
            if p.variable in seen:
                raise ValueError(
                    f"Duplicate variable: {p.variable} detected in variable parameters. This can cause unexpected behaviour."
                )
            seen.add(p.variable)
        return variable_params

    def add_criteria(self, criteria: ScoreCriteria):
        # TODO automatically extract variables
        self.variable_params.append(criteria)
        return self

    def set_score_scaling_params(self, **params: dict) -> "ScoreCriteria":
        self.score_scaling_params = params
        return self

    def score_vars(
        self,
        input_data: pd.DataFrame,
        filter_var_type: typing.Optional[typing.Set[str]] = None,
        ignore_idx=True,
    ) -> pd.DataFrame:
        vars_to_evaluate = self.variable_params
        if filter_var_type is not None:
            vars_to_evaluate = filter(
                lambda x: x.type in filter_var_type, vars_to_evaluate
            )

        all_res = []
        for var in vars_to_evaluate:
            if var.variable in input_data:
                res_df = var.execute(
                    self.bin_prefix,
                    self.score_prefix,
                    self.description_prefix,
                    input_data[var.variable],
                )
            else:
                res_df = var.execute(
                    self.bin_prefix, self.score_prefix, self.description_prefix
                )
            all_res.append(res_df)
        if len(all_res) == 0:
            return pd.DataFrame()
        if ignore_idx:
            res_idx = None
            res_idx = all_res[0].index
            for v in all_res:
                v.reset_index(drop=True, inplace=True)
            res = pd.concat(all_res, axis=1)
            res.index = res_idx
        else:
            res = pd.concat(all_res, axis=1)  # Will try match all with first index
        return res

    def score_numerical_vars(self, input_data: pd.DataFrame) -> pd.DataFrame:
        """Bins numeric variables"""

        return self.score_vars(input_data, ("numerical",))

    def score_categorical_vars(self, input_data: pd.DataFrame) -> pd.DataFrame:
        """Bins categorical variables."""

        return self.score_vars(input_data, ("categorical",))

    def sum_var_scores(self, input_data: pd.DataFrame) -> pd.DataFrame:
        score_col_prefix = self.score_prefix
        sum_col_name = score_col_prefix + "SUM"
        sum_columns = [c for c in input_data.columns if c.startswith(score_col_prefix)]
        input_data[sum_col_name] = input_data[sum_columns].sum(axis=1)
        return input_data

    def score_df(self, data: pd.DataFrame) -> pd.DataFrame:
        """Perform variable binning, variable scoring and total score calculation."""
        res_data = self.score_vars(data)
        self.sum_var_scores(res_data)
        return res_data

    def _get_inputs(self, fn):
        return {p.variable: pd.Series for p in self.variable_params}

    @creates_node(kwarg_input_generator="_get_inputs", is_namespaced=False)
    def score(
        self, **kwargs: typing.Dict[str, pd.Series]
    ) -> pd.DataFrame:  # pragma: no cover
        """A convenience wrapper to score that allows calling with dict of series"""
        return self.score_df(kwargs)

    @creates_node()
    def score_pd(self, score: pd.DataFrame, score_column: str = None) -> pd.DataFrame:
        if self.score_scaling_params is None:
            raise ValueError(
                "Scaling parameters must be provided to calculate probablility scores."
            )
        # Define log odds formula
        score_col = score_column or self.score_prefix + "SUM"
        pd_col = score_col + "_PD"
        log_odds_col = pd_col + "_LOGODDS"

        # get params
        base_points = self.score_scaling_params.base_points
        base_odds = self.score_scaling_params.base_odds
        pdo = self.score_scaling_params.pdo

        # calculate pds
        score[log_odds_col] = log_odds_from_score(
            score[score_col], base_points, base_odds, pdo
        )
        score[pd_col] = probability_of_default_from_log_odds(score[log_odds_col])
        return score

    # @creates_node() # TODO
    def adjust_scores(
        self,
        score: pd.DataFrame,
        adjustment_model: typing.Union[dict, ScoreCardAdjustmentModel, None] = None,
        include_non_adjusted_scores: bool = True,
        include_pd: bool = True,
    ) -> pd.DataFrame:
        """Adjust scores based on an adjustment model."""

        if isinstance(adjustment_model, dict):  # TODO handle None value
            adjustment_model = ScoreCardAdjustmentModel(**adjustment_model)

        score_column = adjustment_model.score_prefix + "SUM"

        # validate adjustment columns
        scores_requiring_adjustment = {
            params.variable for params in adjustment_model.variable_score_adjustments
        }
        cols_not_in_df = scores_requiring_adjustment - set(score.columns)

        if len(cols_not_in_df) != 0:
            raise ValueError(
                f"The following columns do not exist in DataFrame: {cols_not_in_df}. Have you run Score.run() yet to create scores?"
            )

        # get list of all var score in model
        unadjusted_score_cols = [
            self.score_prefix + params.variable
            for params in self.variable_params
            if params.variable
        ]

        # apply adjustments

        summed_adjusted_cols = sum(
            (
                adjustment.apply(score)
                for adjustment in adjustment_model.variable_score_adjustments
            )
        )  # Returns 0 if no items
        if include_non_adjusted_scores:
            summed_adjusted_cols = sum(
                (
                    score[col_name]
                    for col_name in unadjusted_score_cols
                    if col_name not in scores_requiring_adjustment
                ),
                summed_adjusted_cols,
            )

        # sum  var scores for final adjusted score
        # self.logger.debug(f"Adjusted Columns to sum {adjustments}")
        score[score_column] = summed_adjusted_cols

        if include_pd:  # TODO split this out into its own node
            score = self.score_pd(score, score_column)

        return score

    def get_view_model(self):  # pragma: no cover
        from .ui import ScoreCardViewModel

        return ScoreCardViewModel.from_pydantic_model(self)

    # TODO migrate this to the config loader
    # def save(self, file: str):
    #     import json
    #     ext = os.path.splitext(file)[1]
    #     # Small dirty workaround for now to convert tuple and enum types before the yaml dump
    #     config = json.loads(self.model_dump_json())
    #     if ext in ['.yaml', '.yml']:
    #         try:
    #             from yaml import dump
    #         except ImportError as e:
    #             raise ImportError("Could not import yaml please install Spockflow with yaml functionality 'pip install spockflow[yaml]'") from e
    #         try:
    #             from yaml import CDumper as Dumper
    #         except ImportError: # pragma: no cover
    #             from yaml import Dumper
    #         with open(file, 'w') as fp:
    #             dump(config, fp, Dumper=Dumper)
    #     elif ext == '.json':
    #         import json
    #         with open(file, 'w') as fp:
    #             config = json.dump(config, fp)
