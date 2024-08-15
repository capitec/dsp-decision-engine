import typing
from pydantic import RootModel, Field
from .v2.scorecard import LATEST_VERSION, ScoreCardModel as ScoreCardModelV2
from .v2.criteria import (
    build_score_criteria,
    ScoreCriteriaCategorical,
    ScoreCriteriaNumerical,
)
from .v2.criteria_numerical import Bounds
from .probability import log_odds_from_score, probability_of_default_from_log_odds


# This is for when there is more than one version
# class ScoreCard(RootModel):
#     root: typing.Annotated[
#         typing.Union[ScoreCardModelV2],
#         Field(default=LATEST_VERSION, discriminator="version")
#     ]
ScoreCard = ScoreCardModelV2
ScoreCriteria = build_score_criteria
NumericalBounds = Bounds
__all__ = [
    "ScoreCard",
    "ScoreCriteria",
    "ScoreCriteriaCategorical",
    "ScoreCriteriaNumerical",
    "NumericalBounds",
    "log_odds_from_score",
    "probability_of_default_from_log_odds",
]
