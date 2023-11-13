import typing
import pandas as pd
from .scorecard_impl import Score
from lib.annotators import dangerously_skip_type_validation
from lib.creators import parametrized_input_output

@dangerously_skip_type_validation
@parametrized_input_output()
def run_scorecard(scorecard: Score, include_pd:bool = True, **kwargs: typing.Dict[str, pd.Series]) -> pd.DataFrame:
    return scorecard.score(pd.DataFrame(kwargs), include_pd)