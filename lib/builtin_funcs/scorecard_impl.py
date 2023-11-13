""" Helper functions to execute scorecards on Spark """

# Behaviour
# None's in the discrete will be handled as isNull check in Spark
# If there is a range score, with a discrete score in the middle. The the range is evaluated first, thereafter discrete scores are allocated
# Range only supports lower <= x < upper. Use discrete values with range values for special cases.

import typing
import re
import json
from functools import reduce
import numpy as np

from typing_extensions import Annotated
from pydantic import BaseModel, Field, field_validator, ValidationError
from pydantic.functional_validators import BeforeValidator
import pandas as pd

if typing.TYPE_CHECKING:
    import logging
    from pydantic_core import ErrorDetails


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


class DiscreteScorePattern(typing.NamedTuple):
    values: typing.List[typing.Any]
    group_id: int
    score: float
    description: typing.Optional[str]


# Unfortunately not straightforward to implement with generics
# https://stackoverflow.com/questions/50530959/generic-namedtuple-in-python-3-6
class NumericDiscreteScorePattern(typing.NamedTuple):
    values: typing.List[typing.Optional[float]]
    group_id: int
    score: float
    description: typing.Optional[str]


class ConditionedScore(typing.NamedTuple):
    condition: pd.Series
    group_id: int
    score: float
    description: typing.Optional[str]


class ScoreCriteriaV2Categorical(BaseModel):
    variable: str
    type: typing.Literal["categorical"]
    discrete_scores: typing.List[DiscreteScorePattern] = Field(default_factory=list)
    other_score: typing.Optional[DefaultScorePattern] = None

    def to_conditioned_scores(
        self, input_col: pd.Series, logger: "logging.Logger"
    ) -> typing.Iterator[ConditionedScore]:
        logger.info("Set discrete values bin, score and description")
        for discrete_info in self.discrete_scores:
            discrete_list, bin_val, score_val, description_val = discrete_info
            # TODO: validation error? (Empty list causes match on everything)
            if len(discrete_list) <= 0:
                continue

            # TODO: Ask Dan if avoiding rlike is needed with the optimizer
            # TODO: Figure out how to NOT do rlike 3 times
            # Bin
            # ?: avoids creating a capture group (we want groups but no captures)
            regex_values = [f"(?:^{dv}$)" for dv in discrete_list if dv is not None]
            condition = pd.Series([False]*len(input_col))
            if len(regex_values) > 0:
                condition |= input_col.str.contains("|".join(regex_values), regex=True)
            if len(regex_values) < len(discrete_list):  # There are null values TODO check this works with NaN and Null
                condition |= input_col.isnull()
            yield ConditionedScore(
                condition, bin_val, score_val, description_val
            )


class ScoreCriteriaV2Numerical(BaseModel):
    variable: str
    type: typing.Literal["numerical"]
    range_scores: typing.List[RangeScorePattern] = Field(default_factory=list)
    discrete_scores: typing.List[NumericDiscreteScorePattern] = Field(
        default_factory=list
    )
    other_score: typing.Optional[DefaultScorePattern] = None

    def to_conditioned_scores(
        self, input_col: pd.Series, logger: "logging.Logger"
    ) -> typing.Iterator[ConditionedScore]:
        logger.info("Set range values bin, score and description")
        for range_info in self.range_scores:
            range_val, bin_val, score_val, description_val = range_info
            lower_val, upper_val = range_val
            condition = (input_col >= lower_val) & (input_col < upper_val)
            yield ConditionedScore(condition, bin_val, score_val, description_val)

        logger.info("Set discrete values bin, score and description")
        for discrete_info in self.discrete_scores:
            discrete_vals, bin_val, score_val, description_val = discrete_info
            condition = pd.Series([False]*len(input_col))
            for dv in discrete_vals:
                if dv is None:
                    condition |= input_col.isnull()
                else:
                    condition |= input_col == dv

            # TODO: Maybe catch empty conditions as a value error?
            # if len(conditions) <= 0:
            #     continue
            combined_condition = reduce(lambda c_agg, c_new: c_agg | c_new, condition)
            yield ConditionedScore(
                combined_condition, bin_val, score_val, description_val
            )


ScoreCriteriaV2 = Annotated[
    typing.Union[ScoreCriteriaV2Categorical, ScoreCriteriaV2Numerical],
    Field(discriminator="type"),
]


class ScoreScallingParameters(BaseModel):
    base_points: float
    base_odds: float
    pdo: float


class ScoreCardModelV2(BaseModel):
    bin_prefix: str
    score_prefix: str
    description_prefix: str
    variable_params: typing.List[ScoreCriteriaV2] = Field(default_factory=list)
    version: str = Field(pattern=r"^2.[0-9]+.[0-9]$")
    score_scaling_params: typing.Optional[ScoreScallingParameters] = None

    @field_validator("score_scaling_params", mode="before")
    def allow_empty_dict(score_scaling_params: typing.Optional[dict]):
        if isinstance(score_scaling_params, dict) and len(score_scaling_params) == 0:
            return None
        return score_scaling_params

    @field_validator("variable_params")
    def validate_no_repeated_variables(variable_params: "typing.List[ScoreCriteriaV2]"):
        seen = set()
        for p in variable_params:
            if p.variable in seen:
                raise ValueError(
                    f"Duplicate variable: {p.variable} detected in variable parameters. This can cause unexpected behaviour."
                )
            seen.add(p.variable)
        return variable_params


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
        return pd.Series([self.value]*len(df))


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


ScoreCardAdjustmentOperator = Annotated[
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


class ScoreCardValidationError(ValueError):
    def __init__(self, errs: "typing.List[ErrorDetails]", source_data: dict):
        self.variable_errors = []
        self.other_errors = []
        source_data_vp = source_data.get("variable_params", [])
        for err in errs:
            err_loc = err.get("loc", tuple())
            # Ensuring length is 2 for params and index
            if len(err_loc) >= 2 and err_loc[0] == "variable_params":
                variable_name = "unknown"
                if err_loc[1] >= 0 and err_loc[1] < len(source_data_vp):
                    variable_params = source_data_vp[err_loc[1]]
                    if not isinstance(variable_params, dict):
                        self.other_errors.append(err)
                    else:
                        variable_name = source_data_vp[err_loc[1]].get(
                            "variable", variable_name
                        )
                self.variable_errors.append({"variable_name": variable_name, **err})
            else:
                self.other_errors.append(err)
        super().__init__(
            self._get_error_message(self.variable_errors, self.other_errors)
        )

    @staticmethod
    def _get_error_message(variable_errors, other_errors):
        lines = [f"Found {len(variable_errors)+len(other_errors)} Validation Errors"]
        if len(variable_errors) > 0:
            lines.append("Variable Errors:")
            for err in variable_errors:
                err_loc = err.get("loc", tuple())
                if len(err_loc) < 2:
                    continue  # This shouldn't happen just as an incase
                location_path = [
                    f"  {err.get('variable_name', 'unknown')} [{err_loc[1]}]"
                ]
                location_path.extend((str(er_loc_it) for er_loc_it in err_loc[2:]))
                lines.append(" -> ".join(location_path))
                if "input" in err:
                    try:
                        input_json = json.dumps(err.get("input"))
                    except TypeError:
                        input_json = err.get("input")
                    lines.append("    Value: " + input_json)
                lines.append("    " + err.get("msg"))
        if len(other_errors) > 0:
            lines.append("Additional Errors:")
            for err in other_errors:
                err_loc = err.get("loc", tuple())
                lines.append(
                    "  " + (" -> ".join((str(er_loc_it) for er_loc_it in err_loc)))
                )
                if "input" in err:
                    try:
                        input_json = json.dumps(err.get("input"))
                    except TypeError:
                        input_json = err.get("input")
                    lines.append("    Value: " + input_json)
                lines.append("    " + err.get("msg"))
                lines.append("    " + err.get("msg"))
        return "\n".join(lines)


class Score:
    def __init__(self, model: dict, logger: "logging.Logger"):
        model_version: str = model.get("version", None)
        if model_version.startswith("2."):
            parse_error = None
            try:
                self.model = ScoreCardModelV2(**model)
            except ValidationError as err:
                parse_error = err
            if parse_error is not None:  # Avoid long stack trace
                raise ScoreCardValidationError(parse_error.errors(), model)
        else:
            raise ValueError("Model version {model_version} is not supported")
        self.logger = logger
        # Todo: validate scorecard def.
        # Todo: extract column names of importance e.g., vars, bins, score, final score
        #  this is helpful to users for manipulating dataframes

    # @classmethod
    # def from_model_in_s3(cls, file_s3_uri, logger):
    #     """Factory method that loads model from s3."""

    #     # TODO: Add error handling on this function
    #     re_source_s3_uri = re.match(r"^s3://([^/]+)/(.*?)$", file_s3_uri)
    #     source_s3_bucket, source_s3_key = re_source_s3_uri.groups()

    #     s3 = boto3.resource("s3")
    #     logger.info(f"Loading scorecard model from s3: {file_s3_uri}")
    #     content_object = s3.Object(source_s3_bucket, source_s3_key)
    #     file_content = content_object.get()["Body"].read().decode("utf-8")

    #     return cls(model=json.loads(file_content), logger=logger)

    def score(self, input_data, include_pd=True):
        """Perform variable binning, variable scoring and total score calculation."""
        self.logger.info("Scoring numerical vars...")
        input_data = self.score_numerical_vars(input_data)
        self.logger.info("Scoring categorical vars...")
        input_data = self.score_categorical_vars(input_data)
        self.logger.info("Summing scores..")
        input_data = self.sum_var_scores(input_data)
        if include_pd:
            self.logger.info("Calculating LogOdds and PD...")
            input_data = self.pd_from_scores(input_data)
        return input_data

    @staticmethod
    def _replace_col_values_on_condition(
        data: pd.DataFrame,
        condition: pd.Series,
        column_value_pairs: typing.Iterable[typing.Tuple[str, typing.Any]],
    ) -> pd.DataFrame:
        """Replace the values in the source columns with associated value if the row matches a condition"""
        res_data = data # TODO confirm if copy should be made
        for src_col, value in column_value_pairs:
            res_data.loc[condition, src_col] = value
        return res_data

    def score_vars(
        self,
        input_data: pd.DataFrame,
        filter_var_type: typing.Optional[typing.Set[str]] = None,
    ) -> pd.DataFrame:
        self.logger.info(
            "Entered numerical scoring function `bin_score_numerical_vars`"
        )
        bin_prefix = self.model.bin_prefix
        score_prefix = self.model.score_prefix
        description_prefix = self.model.description_prefix

        for var in self.model.variable_params:
            col_name = var.variable
            col_name_binned = bin_prefix + col_name
            col_name_scored = score_prefix + col_name
            col_name_descr = description_prefix + col_name

            if filter_var_type is None or var.type in filter_var_type:
                # Initialize new columns
                # Case: No other values
                # TODO: Remove repetition of code if possible
                self.logger.info("Set default/other/null bin, score and description")
                other_bin, other_score, other_descr = -1, -1, None
                if var.other_score is not None:
                    other_bin, other_score, other_descr = var.other_score

                self.logger.info("Initializing score columns")
                input_data[col_name_binned] = other_bin
                input_data[col_name_scored] = other_score
                input_data[col_name_descr] = other_descr

                for (
                    condition,
                    bin_val,
                    score_val,
                    description_val,
                ) in var.to_conditioned_scores(input_data[col_name], self.logger):
                    input_data = self._replace_col_values_on_condition(
                        input_data,
                        condition,
                        (
                            (col_name_binned, bin_val),
                            (col_name_scored, score_val),
                            (col_name_descr, description_val),
                        ),
                    )

        return input_data

    def score_numerical_vars(self, input_data: pd.DataFrame) -> pd.DataFrame:
        """Bins numeric variables"""

        self.logger.info(
            "Entered numerical scoring function `bin_score_numerical_vars`"
        )
        return self.score_vars(input_data, ("numerical",))

    def score_categorical_vars(self, input_data: pd.DataFrame) -> pd.DataFrame:
        """Bins categorical variables."""

        self.logger.info(
            "Entered categorical scoring function `score_categorical_vars`"
        )
        return self.score_vars(input_data, ("categorical",))

    def sum_var_scores(self, input_data: pd.DataFrame) -> pd.DataFrame:
        score_col_prefix = self.model.score_prefix
        sum_col_name = score_col_prefix + "SUM"
        sum_columns = [c for c in input_data.columns if c.startswith(score_col_prefix)]
        input_data[sum_col_name] = input_data[sum_columns].sum(axis=1)
        return input_data

    def pd_from_scores(self, input_data: pd.DataFrame, score_column=None) -> pd.DataFrame:
        if self.model.score_scaling_params is None:
            raise ValueError(
                "Scaling parameters must be provided to calculate probablility scores."
            )
        # Define log odds formula
        score_col = score_column or self.model.score_prefix + "SUM"
        pd_col = score_col + "_PD"
        log_odds_col = pd_col + "_LOGODDS"

        # get params
        base_points = self.model.score_scaling_params.base_points
        base_odds = self.model.score_scaling_params.base_odds
        pdo = self.model.score_scaling_params.pdo
        self.logger.info(
            f"Using scaling params for PD calc: base_points {base_points}, base_odds {base_odds}, pdo {pdo}"
        )

        # calculate pds
        input_data[log_odds_col] = log_odds_from_score(
            input_data[score_col], 
            base_points, 
            base_odds, 
            pdo
        )
        input_data[pd_col] = probability_of_default_from_log_odds(
            input_data[log_odds_col]
        )
        return input_data

    def adjust_scores(
        self,
        input_data: pd.DataFrame,
        adjustment_model: typing.Union[dict, ScoreCardAdjustmentModel],
        include_non_adjusted_scores=True,
        include_pd=True,
    ) -> pd.DataFrame:
        """Adjust scores based on an adjustment model."""

        if isinstance(adjustment_model, dict):
            adjustment_model = ScoreCardAdjustmentModel(**adjustment_model)

        score_column = adjustment_model.score_prefix + "SUM"

        # validate adjustment columns
        scores_requiring_adjustment = {
            params.variable for params in adjustment_model.variable_score_adjustments
        }
        cols_not_in_df = scores_requiring_adjustment - set(input_data.columns)

        if len(cols_not_in_df) != 0:
            raise ValueError(
                f"The following columns do not exist in DataFrame: {cols_not_in_df}. Have you run Score.run() yet to create scores?"
            )

        # get list of all var score in model
        unadjusted_score_cols = [
            self.model.score_prefix + params.variable
            for params in self.model.variable_params
            if params.variable
        ]

        # apply adjustments
        
        summed_adjusted_cols = sum((
            adjustment.apply(input_data)
            for adjustment in adjustment_model.variable_score_adjustments
        )) # Returns 0 if no items
        if include_non_adjusted_scores:
            self.logger.debug(
                f"Summing non-adjusted scores: {scores_requiring_adjustment}"
            )
            summed_adjusted_cols = sum((
                input_data[col_name]
                for col_name in unadjusted_score_cols
                if col_name not in scores_requiring_adjustment
            ), summed_adjusted_cols)

        # sum  var scores for final adjusted score
        # self.logger.debug(f"Adjusted Columns to sum {adjustments}")
        input_data[score_column] = summed_adjusted_cols

        if include_pd:
            self.logger.debug("Calculating adjusted score LogOdds abd PD...")
            input_data = self.pd_from_scores(input_data, score_column)

        return input_data


# PD calcs
def log_odds_from_score(score_col: pd.Series, base_points: float, base_odds: float, pdo: float) -> pd.Series:
    """Calculate log odds from a score."""
    # Cannot do this here unless df passed in. Makes more sense to just always require the series
    # if isinstance(score_col, str):
    #     score_col = f.col(score_col)

    factor = pdo / np.log(2)
    return (score_col - base_points) / factor + np.log(base_odds)


def probability_of_default_from_log_odds(log_odds_col: pd.Series) -> pd.Series:
    """Calculate PD from log odds."""
    exp_log_odds = np.exp(log_odds_col)
    return 1 / (1 + exp_log_odds)
